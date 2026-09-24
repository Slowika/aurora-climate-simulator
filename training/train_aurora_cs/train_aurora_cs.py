"""Pipeline component: train AuroraCS's conditioning network (forcing encoder + per-block
adapters) and register the result as a model asset.

Builds the "AuroraSmallPretrained" AuroraCS, freezes the wrapped Aurora, and trains on one-step
prediction: for every adjacent triple of 6-hourly steps found in `era5_data`, a 2-step history
window (the first two steps) is fed through a single forward pass and compared against the
ground-truth third step, using `co2_data` for the matching CO2 forcing value. This is teacher-
forced, single-step training - no autoregressive rollout.
"""

import argparse
import logging
import random
import sys
from pathlib import Path

import mlflow
import pandas as pd
import torch
import torch.nn.functional as F
import xarray as xr

from aurora import Batch, Metadata
from aurora_cs import AuroraCS

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

CHECKPOINT = "aurora-0.25-small-pretrained.ckpt"
PATCH_SIZE = 4  # AuroraCS's default `patch_size`; history windows must be pre-cropped to it.
SIX_HOURS = pd.Timedelta(hours=6)


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


def build_examples(
    era5_data: Path, co2: pd.DataFrame, device: torch.device
) -> list[tuple[Batch, float, dict]]:
    """One training example per adjacent triple of 6-hourly steps in `era5_data`: a 2-step
    history window ending at the middle step, paired with the ground-truth next step and its
    CO2 forcing value. Triples spanning a gap in coverage (e.g. a non-contiguous fetch) are
    skipped."""
    cache: dict = {}
    static = xr.open_zarr(str(era5_data / "static.zarr"), chunks=None)
    times = discover_times(era5_data, cache)

    examples = []
    for prev_t, cur_t, next_t in zip(times, times[1:], times[2:]):
        if cur_t - prev_t != SIX_HOURS or next_t - cur_t != SIX_HOURS:
            continue
        batch = build_batch(era5_data, cache, static, cur_t).to(device)
        crop_h = batch.metadata.lat.shape[0]
        target = load_target(era5_data, cache, next_t, crop_h)
        examples.append((batch, co2_ppm_for(co2, next_t), target))
    return examples


def step_loss(pred: Batch, target: dict) -> torch.Tensor:
    """Mean MSE across all surf/atmos variables between a one-step prediction and a
    ground-truth snapshot (as produced by `load_target`)."""
    losses = [
        F.mse_loss(pred.surf_vars[k], v.to(pred.surf_vars[k]))
        for k, v in target["surf_vars"].items()
    ] + [
        F.mse_loss(pred.atmos_vars[k], v.to(pred.atmos_vars[k]))
        for k, v in target["atmos_vars"].items()
    ]
    return torch.stack(losses).mean()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--era5-data", required=True)
    parser.add_argument("--co2-data", required=True)
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-3)
    args = parser.parse_args()

    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {device}")

        era5_data = Path(args.era5_data)
        co2 = pd.read_csv(args.co2_data, parse_dates=["time"])

        logger.info(f"Building one-step training examples from {era5_data}...")
        examples = build_examples(era5_data, co2, device)
        if not examples:
            raise RuntimeError(
                "No adjacent-triple training examples found - era5_data needs at least 3 "
                "consecutive 6-hourly steps."
            )
        logger.info(f"Built {len(examples)} training example(s).")

        logger.info("Building AuroraCS (AuroraSmallPretrained config) and loading its checkpoint...")
        model = AuroraCS(
            encoder_depths=(2, 6, 2),
            encoder_num_heads=(4, 8, 16),
            decoder_depths=(2, 6, 2),
            decoder_num_heads=(16, 8, 4),
            embed_dim=256,
            num_heads=8,
            use_lora=False,
        ).to(device)
        model.aurora.load_checkpoint(name=CHECKPOINT)

        trainable_params = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.Adam(trainable_params, lr=args.lr)

        logger.info(f"Training for {args.epochs} epoch(s) over {len(examples)} example(s)...")
        rng = random.Random(0)
        global_step = 0
        for epoch in range(args.epochs):
            rng.shuffle(examples)
            epoch_loss = 0.0
            for batch, forcing, target in examples:
                optimizer.zero_grad()
                pred = model(batch, forcing=forcing)
                loss = step_loss(pred, target)
                loss.backward()
                optimizer.step()
                mlflow.log_metric("loss", loss.item(), step=global_step)
                epoch_loss += loss.item()
                global_step += 1
            mean_loss = epoch_loss / len(examples)
            mlflow.log_metric("epoch_loss", mean_loss, step=epoch)
            logger.info(f"  epoch {epoch}: mean loss = {mean_loss:.6f}")

        logger.info(f"Registering trained model as '{args.model_name}'...")
        mlflow.pytorch.log_model(
            model,
            artifact_path="model",
            registered_model_name=args.model_name,
            serialization_format="pickle",
        )
        logger.info("Done.")
    except Exception as e:
        print(e)
        sys.exit(1)


if __name__ == "__main__":
    main()
