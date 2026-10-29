"""Pipeline component: train AuroraCS's conditioning network (forcing encoder + per-block
adapters) and register the result as a model asset.

Builds the "AuroraSmallPretrained" AuroraCS, freezes the wrapped Aurora, and trains on one-step
prediction: for every adjacent triple of 6-hourly steps found in `era5_data`, a 2-step history
window (the first two steps) is fed through a single forward pass and compared against the
ground-truth third step, using `co2_data` for the matching CO2 forcing value. This is teacher-
forced, single-step training - no autoregressive rollout.
"""

import argparse
import json
import logging
import math
import random
import re
import tempfile
from datetime import date
from pathlib import Path

import mlflow
import pandas as pd
import torch
import torch.nn.functional as F
import xarray as xr

from aurora import Batch, Metadata
from aurora_cs import AuroraCS, configure_mlflow_auth, save_checkpoint

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
# "AuroraSmallPretrained" architecture; stored with the registered model to rebuild it.
MODEL_CONFIG = dict(
    encoder_depths=(2, 6, 2),
    encoder_num_heads=(4, 8, 16),
    decoder_depths=(2, 6, 2),
    decoder_num_heads=(16, 8, 4),
    embed_dim=256,
    num_heads=8,
    use_lora=False,
    # CO2 in ppm is ~350-430 for the historical period; raw values make the first Adam step
    # blow the loss up by orders of magnitude. Stored in the checkpoint, so rollout matches.
    forcing_mean=350.0,
    forcing_scale=100.0,
)
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


def discover_examples(
    era5_data: Path,
    cache: dict,
    start_date: date | None = None,
    end_date: date | None = None,
) -> list[pd.Timestamp]:
    """The middle timestamp of every adjacent triple of 6-hourly steps in `era5_data`: the
    history window ends there and the next step is the target. Triples spanning a gap in
    coverage (e.g. a non-contiguous fetch) are skipped. All three timestamps must fall
    within the optional, inclusive date bounds."""
    times = discover_times(era5_data, cache)
    if start_date is not None:
        times = [t for t in times if t.date() >= start_date]
    if end_date is not None:
        times = [t for t in times if t.date() <= end_date]
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


def step_loss(pred: Batch, target: dict) -> torch.Tensor:
    """Mean MSE across all surf/atmos variables between a one-step prediction and a
    ground-truth snapshot (as produced by `load_target`)."""
    losses = []
    for group, variables in (("surf_vars", pred.surf_vars), ("atmos_vars", pred.atmos_vars)):
        for name, value in target[group].items():
            context = f"{group}.{name} at {pred.metadata.time[0]}"
            expected = value.to(variables[name])
            require_finite(expected, f"target.{context}")
            require_finite(variables[name], f"prediction.{context}")
            loss = F.mse_loss(variables[name], expected)
            require_finite(loss, f"loss.{context}")
            losses.append(loss)
    loss = torch.stack(losses).mean()
    require_finite(loss, f"total loss at {pred.metadata.time[0]}")
    return loss


def register_model(model: AuroraCS, config: dict, model_name: str) -> None:
    """Log `model` (weights + `config`, see `save_checkpoint`) to the active run and register it."""
    with tempfile.TemporaryDirectory() as tmp:
        save_checkpoint(model, config, tmp)
        mlflow.log_artifacts(tmp, artifact_path="model")
    mlflow.register_model(f"{mlflow.active_run().info.artifact_uri}/model", model_name)


def parse_date(value: str) -> date | None:
    """Accept an ISO date or the command job template's explicit unbounded default."""
    return None if value == "unbounded" else date.fromisoformat(value)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--era5-data", required=True)
    parser.add_argument("--co2-data", required=True)
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument(
        "--start-date", type=parse_date, help="First training date, inclusive (YYYY-MM-DD)."
    )
    parser.add_argument(
        "--end-date", type=parse_date, help="Last training date, inclusive (YYYY-MM-DD)."
    )
    args = parser.parse_args()
    if args.start_date is not None and args.end_date is not None and args.start_date > args.end_date:
        parser.error("--start-date must be on or before --end-date")

    configure_mlflow_auth()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    era5_data = Path(args.era5_data)
    co2 = pd.read_csv(args.co2_data, parse_dates=["time"])

    cache: dict = {}
    static = xr.open_zarr(str(era5_data / "static.zarr"), chunks=None)
    for name in STATIC_VAR_MAP.values():
        require_finite(torch.as_tensor(static[name].values), f"ERA5 static.{name}")
    examples = discover_examples(era5_data, cache, args.start_date, args.end_date)
    if not examples:
        raise RuntimeError(
            "No adjacent-triple training examples found in the selected date range - "
            "era5_data needs at least 3 consecutive 6-hourly steps within that range."
        )
    logger.info(f"Found {len(examples)} training example(s) in {era5_data}.")
    logger.info(
        f"Training history/target coverage: {examples[0] - SIX_HOURS} to "
        f"{examples[-1] + SIX_HOURS}."
    )
    problems = find_incomplete_shards(era5_data, examples)
    if problems:
        details = "; ".join(f"{shard}: {', '.join(issues)}" for shard, issues in problems.items())
        raise RuntimeError(
            f"{len(problems)} ERA5 shard(s) in the training window are incomplete: {details}. "
            "Fetch those months again with fetch_era5, or narrow --start-date/--end-date."
        )

    logger.info("Building AuroraCS (AuroraSmallPretrained config) and loading its checkpoint...")
    model = AuroraCS(**MODEL_CONFIG).to(device)
    model.aurora.load_checkpoint(name=CHECKPOINT)

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(trainable_params, lr=args.lr)

    logger.info(f"Training for {args.epochs} epoch(s) over {len(examples)} example(s)...")
    rng = random.Random(0)
    global_step = 0
    for epoch in range(args.epochs):
        rng.shuffle(examples)
        epoch_loss = 0.0
        for cur_t in examples:
            batch, forcing, target = load_example(era5_data, cache, static, co2, cur_t, device)
            optimizer.zero_grad()
            pred = model(batch, forcing=forcing)
            loss = step_loss(pred, target)
            loss.backward()
            for name, param in model.named_parameters():
                if param.grad is not None:
                    require_finite(
                        param.grad,
                        f"gradient.{name}, epoch={epoch}, step={global_step}, "
                        f"target time={cur_t + SIX_HOURS}",
                    )
            optimizer.step()
            mlflow.log_metric("loss", loss.item(), step=global_step)
            epoch_loss += loss.item()
            global_step += 1
        mean_loss = epoch_loss / len(examples)
        mlflow.log_metric("epoch_loss", mean_loss, step=epoch)
        logger.info(f"  epoch {epoch}: mean loss = {mean_loss:.6f}")

    logger.info(f"Registering trained model as '{args.model_name}'...")
    register_model(model, MODEL_CONFIG, args.model_name)
    logger.info("Done.")


if __name__ == "__main__":
    main()
