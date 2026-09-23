"""CLI producer: fetch ERA5 weather data from ARCO-ERA5 and register it as a monthly-sharded
Azure ML data asset, written directly to the workspace's blob datastore, e.g.:

    python -m data.fetch_era5 --start-date 2020-01-01 --end-date 2020-03-01 \\
        --prefix era5 --asset-name era5-weather \\
        --subscription-id ... --resource-group ... --workspace-name ...

Re-running with an overlapping or extended date range only writes the months not already present
under `--prefix`, then registers a new version of the asset over the (now larger) folder.
"""

import argparse
import calendar
import logging
import sys

import gcsfs
import pandas as pd
import xarray as xr
from azure.ai.ml.constants import AssetTypes

from data.azure_ml import get_blob_filesystem, get_ml_client, next_version, register_data_asset

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

DATA_TAG = {"data": "GCP ERA5 ARCO 6H"}


def month_range(start: pd.Timestamp, end: pd.Timestamp) -> list[tuple[int, int]]:
    """(year, month) pairs spanning [start, end], inclusive."""
    months = []
    year, month = start.year, start.month
    while (year, month) <= (end.year, end.month):
        months.append((year, month))
        month += 1
        if month > 12:
            month = 1
            year += 1
    return months


def fetch_month(ds: xr.Dataset, year: int, month: int, store) -> None:
    """Fetch one calendar month at Aurora's native 6-hourly cadence.

    Every chunk in the source archive bundles all 37 pressure levels for a single hourly
    timestep, so selecting `ATMOS_LEVELS` doesn't reduce what has to be read off GCS - only
    subsampling in time does. Aurora itself only consumes 6-hourly steps, so this loses nothing.

    A month of 5 atmospheric variables at 13 levels is ~30GB in float32 - too large to
    materialize at once on modest compute. `ds` must be dask-backed (chunked one native
    timestep at a time, via `open_zarr(..., chunks={"time": 1})`) so `to_zarr` streams the
    write one chunk at a time instead, keeping peak memory flat regardless of range.
    """
    _, last_day = calendar.monthrange(year, month)
    times = pd.date_range(
        pd.Timestamp(year=year, month=month, day=1),
        pd.Timestamp(year=year, month=month, day=last_day, hour=23),
        freq="6h",
    )
    variables = list(SURF_VAR_MAP.values()) + list(ATMOS_VAR_MAP.values())
    subset = ds[variables].sel(time=times, level=ATMOS_LEVELS)
    subset.to_zarr(store, mode="w")


def fetch_static(ds: xr.Dataset, store) -> None:
    variables = list(STATIC_VAR_MAP.values())
    ds[variables].isel(time=0, drop=True).compute().to_zarr(store, mode="w")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start-date", required=True)
    parser.add_argument("--end-date", required=True)
    parser.add_argument("--prefix", required=True, help="blob path prefix under the datastore")
    parser.add_argument("--asset-name", required=True)
    parser.add_argument(
        "--datastore-name",
        default="workspaceblobstore",
        help="Name of the workspace blob datastore to write into.",
    )
    parser.add_argument("--subscription-id", required=True)
    parser.add_argument("--resource-group", required=True)
    parser.add_argument("--workspace-name", required=True)
    parser.add_argument(
        "--managed-identity-client-id",
        default=None,
        help="Client ID of a user-assigned identity, if the compute doesn't have a "
        "system-assigned one.",
    )
    args = parser.parse_args()

    try:
        start = pd.Timestamp(args.start_date)
        end = pd.Timestamp(args.end_date)
        prefix = args.prefix.strip("/")

        ml_client = get_ml_client(
            args.subscription_id,
            args.resource_group,
            args.workspace_name,
            args.managed_identity_client_id or None,
        )
        fs, container = get_blob_filesystem(
            ml_client, args.datastore_name, args.managed_identity_client_id or None
        )

        months = month_range(start, end)
        missing = [
            (year, month)
            for year, month in months
            if not fs.exists(f"{container}/{prefix}/{year}-{month:02d}.zarr")
        ]
        static_path = f"{container}/{prefix}/static.zarr"
        need_static = not fs.exists(static_path)

        if missing or need_static:
            logger.info(f"Opening {ZARR_URL}...")
            gcs = gcsfs.GCSFileSystem(token="anon")
            ds = xr.open_zarr(gcs.get_mapper(ZARR_URL), chunks={"time": 1})

            if need_static:
                logger.info(f"Writing static variables -> {static_path}")
                fetch_static(ds, fs.get_mapper(static_path))

            for year, month in missing:
                path = f"{container}/{prefix}/{year}-{month:02d}.zarr"
                logger.info(f"  {year}-{month:02d} -> {path}")
                fetch_month(ds, year, month, fs.get_mapper(path))
        else:
            logger.info("All requested months (and statics) already present.")

        asset_path = f"azureml://datastores/{args.datastore_name}/paths/{prefix}"
        logger.info(f"Registering '{args.asset_name}' from {asset_path}...")
        register_data_asset(
            ml_client,
            name=args.asset_name,
            path=asset_path,
            asset_type=AssetTypes.URI_FOLDER,
            description=f"ERA5 weather data, monthly shards, {args.start_date} to {args.end_date}",
            version=next_version(ml_client, args.asset_name),
            tags=DATA_TAG,
        )
        logger.info("Done.")
    except Exception as e:
        print(e)
        sys.exit(1)


if __name__ == "__main__":
    main()
