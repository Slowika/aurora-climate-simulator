"""Pipeline component: build Aurora's initial condition (history window) for a rollout.

Slices the ERA5 archive at `start_time` and packages the trailing 2-step history window
AuroraCS needs (surf + atmos variables, statics) into a single `aurora.Batch`, saved as
`<initial_condition>/batch.pt`.
"""

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd
import torch
import xarray as xr

from aurora import Batch, Metadata

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
logger = logging.getLogger(__name__)

# Must match the variables `data/fetch_era5.py` writes into each monthly shard / static.zarr.
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

NUM_HISTORY = 2  # What AuroraCS's default `max_history_size` requires.
PATCH_SIZE = 4  # AuroraCS's default `patch_size`; the batch must be pre-cropped to it so its
# shape matches what the model will hand back to `advance()` after the first rollout step.


def open_step(era5_data: Path, cache: dict, when: pd.Timestamp) -> xr.Dataset:
    """Open (and cache) the monthly shard containing `when`, sliced to that single timestep."""
    key = (when.year, when.month)
    if key not in cache:
        cache[key] = xr.open_zarr(str(era5_data / f"{when.year}-{when.month:02d}.zarr"), chunks=None)
    return cache[key].sel(time=when)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--era5-data", required=True)
    parser.add_argument("--start-time", required=True)
    parser.add_argument("--initial-condition", required=True)
    args = parser.parse_args()

    try:
        era5_data = Path(args.era5_data)
        start_time = pd.Timestamp(args.start_time)
        history_times = [start_time - pd.Timedelta(hours=6), start_time]
        logger.info(f"Building {NUM_HISTORY}-step history window ending {start_time} from {era5_data}...")

        cache: dict = {}
        steps = [open_step(era5_data, cache, when) for when in history_times]
        window = xr.concat(steps, dim="time")
        static = xr.open_zarr(str(era5_data / "static.zarr"), chunks=None)

        # Aurora requires strictly decreasing latitude; flip if the source has it ascending.
        # Latitude is always the second-to-last axis, for surf, atmos, and static arrays alike.
        # Flipping is done on plain numpy (post `.values`) since zarr can't lazily slice with a
        # negative step.
        lat = window.latitude.values
        need_flip = lat[0] < lat[-1]

        def to_tensor(arr) -> torch.Tensor:
            values = arr.values
            if need_flip:
                values = values[..., ::-1, :]
            return torch.from_numpy(values.copy())

        def stacked(var_map: dict) -> dict:
            return {short: to_tensor(window[long])[None] for short, long in var_map.items()}

        batch = Batch(
            surf_vars=stacked(SURF_VAR_MAP),
            static_vars={short: to_tensor(static[long]) for short, long in STATIC_VAR_MAP.items()},
            atmos_vars=stacked(ATMOS_VAR_MAP),
            metadata=Metadata(
                lat=torch.from_numpy(lat[::-1].copy() if need_flip else lat.copy()),
                lon=torch.from_numpy(window.longitude.values.copy()),
                time=(start_time.to_pydatetime(),),
                atmos_levels=tuple(int(level) for level in window.level.values),
            ),
        ).crop(PATCH_SIZE)

        out_dir = Path(args.initial_condition)
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / "batch.pt"
        torch.save(batch, out_path)
        logger.info(f"Wrote initial condition -> {out_path}")
    except Exception as e:
        print(e)
        sys.exit(1)


if __name__ == "__main__":
    main()
