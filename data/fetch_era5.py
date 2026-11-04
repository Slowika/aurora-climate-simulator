"""CLI producer: fetch ERA5 data from the Google Cloud Platform analysis-ready dataset.

Writes month-sharded data to an output folder.

    python fetch_era5.py --start-date 2020-01-01 --end-date 2020-03-01

Re-running with overlapping or extended date ranges only writes the months not already
present (a month interrupted part-way is fetched again).

When run as an Azure ML command job with jobs/fetch_era5_job.yml, the output is
auto-registered as an Azure ML data asset under a given name and version.
"""

import argparse
import calendar
import json
import logging
import math
import re
from pathlib import Path

import gcsfs
import numpy as np
import pandas as pd
import xarray as xr

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
logger = logging.getLogger(__name__)

ZARR_URL = "gs://gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3"

SURF_VAR_MAP = {
    "2t": "2m_temperature",
    "10u": "10m_u_component_of_wind",
    "10v": "10m_v_component_of_wind",
    "msl": "mean_sea_level_pressure",
}
ATMOS_VAR_MAP = {
    "t": "temperature",
    "u": "u_component_of_wind",
    "v": "v_component_of_wind",
    "q": "specific_humidity",
    "z": "geopotential",
}
STATIC_VAR_MAP = {
    "lsm": "land_sea_mask",
    "slt": "soil_type",
    "z": "geopotential_at_surface",
}
ATMOS_LEVELS = [1000, 925, 850, 700, 600, 500, 400, 300, 250, 200, 150, 100, 50]


def month_range(start: pd.Timestamp, end: pd.Timestamp) -> list[tuple[int, int]]:
    """Generate (year, month) pairs spanning [start, end], inclusive."""
    months = []
    year, month = start.year, start.month
    while (year, month) <= (end.year, end.month):
        months.append((year, month))
        month += 1
        if month > 12:  # noqa: PLR2004
            month = 1
            year += 1
    return months


def fetch_month(ds: xr.Dataset, year: int, month: int, store: Path) -> None:
    """Fetch and write one calendar month at Aurora's native 6-hourly cadence.

    Every chunk in the source archive bundles all 37 pressure levels for a single hourly
    timestep, so selecting `ATMOS_LEVELS` doesn't reduce what has to be read off GCS -
    only subsampling in time does. Aurora itself only consumes 6-hourly steps, so this
    loses nothing.

    A month of 5 atmospheric variables at 13 levels is ~30GB in float32 - too large to
    materialize at once on modest compute. `ds` must be dask-backed (chunked one native
    timestep at a time, via `open_zarr(..., chunks={"time": 1})`) so `to_zarr` streams
    the write one chunk at a time instead, keeping peak memory flat regardless of range.
    """
    _, last_day = calendar.monthrange(year, month)
    times = pd.date_range(
        pd.Timestamp(year=year, month=month, day=1),
        pd.Timestamp(year=year, month=month, day=last_day, hour=23),
        freq="6h",
    )
    variables = list(SURF_VAR_MAP.values()) + list(ATMOS_VAR_MAP.values())
    subset = ds[variables].sel(time=times, level=ATMOS_LEVELS)
    subset.to_zarr(store, mode="w")  # type: ignore[invalid-argument-type]


def shard_is_complete(store: Path) -> bool:
    """Whether every chunk of every variable exists in a local Zarr shard.

    Existence of the shard says nothing: an interrupted write leaves the metadata in
    place, and chunks that were never written read back as NaN. Such a shard must be
    fetched again.
    """
    for name in list(SURF_VAR_MAP.values()) + list(ATMOS_VAR_MAP.values()):
        variable_dir = store / name
        try:
            meta = json.loads((variable_dir / ".zarray").read_text())
            entries = tuple(variable_dir.iterdir())
        except FileNotFoundError:
            return False

        expected = math.prod(
            -(-size // chunk)
            for size, chunk in zip(meta["shape"], meta["chunks"], strict=True)
        )
        written = sum(
            entry.is_file()
            and bool(re.fullmatch(r"\d+(\.\d+)*", entry.name))
            for entry in entries
        )
        if written != expected:
            return False
    return True


def fetch_static(ds: xr.Dataset, when: pd.Timestamp, store: Path) -> None:
    """Fetch and write static variables for a given timestamp."""
    variables = list(STATIC_VAR_MAP.values())
    # The archive starts in 1900 with empty slices, not at the start of valid ERA5 data.
    static = ds[variables].sel(time=when, drop=True).compute()
    for name in variables:
        if not np.isfinite(static[name].values).all():
            msg = f"Non-finite static variable, skipped write: var={name} time={when}"
            raise ValueError(msg)
    static.to_zarr(store, mode="w")  # type: ignore[invalid-argument-type]


def main() -> None:
    """Fetch ERA5 data for a given date range, storing Zarr format monthly shards."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--start-date",
        type=pd.Timestamp,
        required=True,
        help="Start date for fetching ERA5 data in YYYY-MM-DD format.",
    )
    parser.add_argument(
        "--end-date",
        type=pd.Timestamp,
        required=True,
        help="End date for fetching ERA5 data in YYYY-MM-DD format.",
    )
    parser.add_argument(
        "--sharded",
        type=Path,
        required=True,
        help="Directory to which fetched monthly shards will be written.",
    )
    args = parser.parse_args()

    months = month_range(args.start_date, args.end_date)
    missing = [
        (year, month)
        for year, month in months
        if not shard_is_complete(args.sharded / f"{year}-{month:02d}.zarr")
    ]
    static_path = args.sharded / "static.zarr"
    need_static = not static_path.exists()

    if missing or need_static:
        logger.info("Opening source store: url=%s", ZARR_URL)
        gcs = gcsfs.GCSFileSystem(token="anon")  # noqa: S106
        ds = xr.open_zarr(gcs.get_mapper(ZARR_URL), chunks={"time": 1})

        if need_static:
            logger.info("Writing static variables: path=%s", static_path)
            fetch_static(ds, args.start_date, static_path)

        for year, month in missing:
            path = args.sharded / f"{year}-{month:02d}.zarr"
            logger.info("Writing shard: year=%s month=%s path=%s", year, month, path)
            fetch_month(ds, year, month, path)
    else:
        logger.info("All requested months (and statics) already present.")

    logger.info("Done.")


if __name__ == "__main__":
    main()
