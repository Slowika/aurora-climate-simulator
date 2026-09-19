"""Pipeline component: train AuroraCS's conditioning network (forcing encoder + per-block
adapters) and register the result as a model asset.

Builds the "AuroraSmallPretrained" AuroraCS, freezes the wrapped Aurora, and repeatedly takes
gradient steps against the rollout loss for the window described by `initial_condition` and
`forcing_trajectory` (as produced by `prepare_initial_condition`/`prepare_forcing_trajectory`),
using `era5_data` for the ground-truth future snapshots.
"""

import argparse
import logging
import sys
from dataclasses import replace
from pathlib import Path

import mlflow
import pandas as pd
import torch
import torch.nn.functional as F
import xarray as xr

from aurora_cs import AuroraCS

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
logger = logging.getLogger(__name__)

# Must match the variables `data/fetch_era5.py` writes into each monthly shard.
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

CHECKPOINT = "aurora-0.25-small-pretrained.ckpt"


def open_step(era5_data: Path, cache: dict, when: pd.Timestamp) -> xr.Dataset:
    """Open (and cache) the monthly shard containing `when`, sliced to that single timestep."""
    key = (when.year, when.month)
    if key not in cache:
        cache[key] = xr.open_zarr(str(era5_data / f"{when.year}-{when.month:02d}.zarr"), chunks=None)
    return cache[key].sel(time=when)


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


def advance(batch, pred):
    """Slide the history window: drop the oldest step, append the new prediction.

    Local reimplementation of `aurora.rollout._advance_batch`'s logic (kept local rather
    than importing a private, unexported symbol).
    """
    new_surf = {
        k: torch.cat([batch.surf_vars[k][:, 1:], v], dim=1)
        for k, v in pred.surf_vars.items()
        if k in batch.surf_vars
    }
    new_atmos = {
        k: torch.cat([batch.atmos_vars[k][:, 1:], v], dim=1)
        for k, v in pred.atmos_vars.items()
        if k in batch.atmos_vars
    }
    return replace(pred, surf_vars=new_surf, atmos_vars=new_atmos)


def rollout_loss(pred, target: dict) -> torch.Tensor:
    """Mean MSE across all surf/atmos variables between a prediction and a ground-truth
    snapshot (as produced by `load_target`)."""
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
    parser.add_argument("--initial-condition", required=True)
    parser.add_argument("--forcing-trajectory", required=True)
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-3)
    args = parser.parse_args()

    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {device}")

        # Expected contract with `prepare_initial_condition`: a single `torch.save`d
        # `aurora.Batch` at `<initial_condition>/batch.pt`.
        init_batch = torch.load(
            Path(args.initial_condition) / "batch.pt", map_location=device, weights_only=False
        )
        # Expected contract with `prepare_forcing_trajectory`: a single `torch.save`d 1-D tensor
        # of per-step forcing values.
        forcing = torch.load(args.forcing_trajectory, map_location=device, weights_only=True)
        num_steps = len(forcing)
        start_time = pd.Timestamp(init_batch.metadata.time[0])
        crop_h = init_batch.metadata.lat.shape[0]

        logger.info(f"Loading {num_steps} ground-truth future snapshots from {args.era5_data}...")
        era5_data = Path(args.era5_data)
        cache: dict = {}
        step_times = [start_time + (i + 1) * pd.Timedelta(hours=6) for i in range(num_steps)]
        targets = [load_target(era5_data, cache, when, crop_h) for when in step_times]

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

        def rollout_once() -> torch.Tensor:
            batch = init_batch
            total = torch.zeros((), device=device)
            for step in range(num_steps):
                pred = model(batch, forcing=forcing[step].item())
                total = total + rollout_loss(pred, targets[step])
                batch = advance(batch, pred)
            return total

        trainable_params = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.Adam(trainable_params, lr=args.lr)

        logger.info(f"Training for {args.iters} iterations over a {num_steps}-step rollout...")
        for it in range(args.iters):
            optimizer.zero_grad()
            loss = rollout_once()
            loss.backward()
            optimizer.step()
            mlflow.log_metric("loss", loss.item(), step=it)
            logger.info(f"  iter {it}: loss = {loss.item():.6f}")

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
