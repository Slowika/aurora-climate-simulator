"""Loading, discovery and validation of the ERA5 and CO2 data for `train_aurora_cs`."""

import json
import math
import re
from pathlib import Path

import pandas as pd
import torch
import xarray as xr

from aurora import Batch, Metadata

from training_constants import ATMOS_VAR_MAP, PATCH_SIZE, SIX_HOURS, STATIC_VAR_MAP, SURF_VAR_MAP


def open_month(era5_data: Path, cache: dict, year: int, month: int) -> xr.Dataset:
    """Open (and cache) one monthly shard."""
    key = (year, month)
    if key not in cache:
        cache[key] = xr.open_zarr(str(era5_data / f"{year}-{month:02d}.zarr"), chunks=None)
    return cache[key]


def open_step(era5_data: Path, cache: dict, when: pd.Timestamp) -> xr.Dataset:
    """The cached monthly shard containing `when`, sliced to that single timestep."""
    return open_month(era5_data, cache, when.year, when.month).sel(time=when)


def discover_times(era5_data: Path, cache: dict) -> list[pd.Timestamp]:
    """All timestamps present across `era5_data`'s monthly shards, sorted ascending."""
    shard_names = sorted(
        p.name for p in era5_data.iterdir() if p.suffix == ".zarr" and p.stem != "static"
    )
    times: list[pd.Timestamp] = []
    for name in shard_names:
        year, month = int(name[:4]), int(name[5:7])
        times.extend(pd.Timestamp(t) for t in open_month(era5_data, cache, year, month).time.values)
    return sorted(times)


def co2_ppm_for(co2: pd.DataFrame, when: pd.Timestamp) -> float:
    """Look up the monthly-mean CO2 ppm for `when`, falling back to the nearest month."""
    exact = co2[(co2["time"].dt.year == when.year) & (co2["time"].dt.month == when.month)]
    if len(exact):
        return float(exact["co2_ppm"].iloc[0])
    idx = (co2["time"] - when).abs().idxmin()
    return float(co2.loc[idx, "co2_ppm"])


def build_batch(era5_data: Path, cache: dict, static: xr.Dataset, when: pd.Timestamp) -> Batch:
    """Aurora-ready 2-step history window ending at `when` (cropped to `PATCH_SIZE`)."""
    history_times = [when - SIX_HOURS, when]
    steps = [open_step(era5_data, cache, t) for t in history_times]
    window = xr.concat(steps, dim="time")

    # Aurora requires strictly decreasing latitude; flip if the source has it ascending.
    lat = window.latitude.values
    need_flip = lat[0] < lat[-1]

    def to_tensor(arr) -> torch.Tensor:
        values = arr.values
        if need_flip:
            values = values[..., ::-1, :]
        return torch.from_numpy(values.copy())

    def stacked(var_map: dict) -> dict:
        return {short: to_tensor(window[long])[None] for short, long in var_map.items()}

    return Batch(
        surf_vars=stacked(SURF_VAR_MAP),
        static_vars={short: to_tensor(static[long]) for short, long in STATIC_VAR_MAP.items()},
        atmos_vars=stacked(ATMOS_VAR_MAP),
        metadata=Metadata(
            lat=torch.from_numpy(lat[::-1].copy() if need_flip else lat.copy()),
            lon=torch.from_numpy(window.longitude.values.copy()),
            time=(when.to_pydatetime(),),
            atmos_levels=tuple(int(level) for level in window.level.values),
        ),
    ).crop(PATCH_SIZE)


def load_target(era5_data: Path, cache: dict, when: pd.Timestamp, crop_h: int) -> dict:
    """Load the ground-truth snapshot at `when`, cropped to match the model's output grid."""
    step = open_step(era5_data, cache, when)
    lat = step.latitude.values
    need_flip = lat[0] < lat[-1]

    def to_tensor(long_name: str) -> torch.Tensor:
        values = step[long_name].values
        if need_flip:
            values = values[..., ::-1, :]
        return torch.from_numpy(values[..., :crop_h, :].copy())[None, None]

    return {
        "surf_vars": {short: to_tensor(long) for short, long in SURF_VAR_MAP.items()},
        "atmos_vars": {short: to_tensor(long) for short, long in ATMOS_VAR_MAP.items()},
    }


def discover_examples(era5_data: Path, cache: dict) -> list[pd.Timestamp]:
    """The middle timestamp of every adjacent triple of 6-hourly steps in `era5_data`: the
    history window ends there and the next step is the target. Triples spanning a gap in
    coverage (e.g. a non-contiguous fetch) are skipped."""
    times = discover_times(era5_data, cache)
    return [
        cur_t
        for prev_t, cur_t, next_t in zip(times, times[1:], times[2:])
        if cur_t - prev_t == SIX_HOURS and next_t - cur_t == SIX_HOURS
    ]


def find_incomplete_shards(era5_data: Path, examples: list[pd.Timestamp]) -> dict[str, list[str]]:
    """Monthly shards used by `examples` whose weather variables have unwritten chunks.

    Unwritten zarr chunks read back as NaN, and an interrupted fetch leaves the shard's metadata
    in place, so a damaged shard looks fine until the first example that touches it.
    """
    months = sorted({(t.year, t.month) for cur in examples for t in (cur - SIX_HOURS, cur, cur + SIX_HOURS)})
    problems: dict[str, list[str]] = {}
    for year, month in months:
        shard = era5_data / f"{year}-{month:02d}.zarr"
        for name in [*SURF_VAR_MAP.values(), *ATMOS_VAR_MAP.values()]:
            try:
                meta = json.loads((shard / name / ".zarray").read_text())
                expected = math.prod(-(-size // chunk) for size, chunk in zip(meta["shape"], meta["chunks"]))
                written = sum(
                    bool(re.fullmatch(r"\d+(\.\d+)*", p.name)) for p in (shard / name).iterdir()
                )
            except FileNotFoundError:
                problems.setdefault(shard.name, []).append(f"{name}: missing")
                continue
            if written != expected:
                problems.setdefault(shard.name, []).append(f"{name}: {written}/{expected} chunks")
    return problems


def load_example(
    era5_data: Path,
    cache: dict,
    static: xr.Dataset,
    co2: pd.DataFrame,
    cur_t: pd.Timestamp,
    device: torch.device,
) -> tuple[Batch, float, dict]:
    """Load one training example on demand (so memory stays flat in the dataset size): the
    2-step history window ending at `cur_t` on `device`, the CO2 forcing value for the next
    step, and the ground-truth next step (left on the CPU)."""
    next_t = cur_t + SIX_HOURS
    batch = build_batch(era5_data, cache, static, cur_t).to(device)
    for group, variables in (("surf_vars", batch.surf_vars), ("atmos_vars", batch.atmos_vars)):
        for name, value in variables.items():
            require_finite(value, f"input.{group}.{name}, history ending {cur_t}")
    target = load_target(era5_data, cache, next_t, batch.metadata.lat.shape[0])
    forcing = co2_ppm_for(co2, next_t)
    if not math.isfinite(forcing):
        raise FloatingPointError(f"Non-finite CO2 forcing at {next_t}.")
    return batch, forcing, target


def require_finite(value: torch.Tensor, context: str) -> None:
    if not torch.isfinite(value).all().item():
        raise FloatingPointError(f"Non-finite values in {context}.")
