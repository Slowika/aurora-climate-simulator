"""Pipeline component: run the AuroraCS autoregressive rollout.

Loads a registered AuroraCS model and an initial condition, then steps the model forward
`num_steps` times conditioned on a CO2 forcing trajectory, logging rollout progress to MLflow.
No prediction artifacts are written out yet.
"""

import argparse
import logging
import sys
from dataclasses import replace
from pathlib import Path

import mlflow
import torch
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
logger = logging.getLogger(__name__)


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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--initial-condition", required=True)
    parser.add_argument("--forcing-trajectory", required=True)
    parser.add_argument("--model-id", required=True, help="Registered model, e.g. 'aurora-cs-adapter/3'.")
    parser.add_argument("--num-steps", required=True, type=int)
    args = parser.parse_args()

    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {device}")

        logger.info(f"Loading registered model '{args.model_id}'...")
        model = mlflow.pytorch.load_model(f"models:/{args.model_id}").to(device).eval()

        # Expected contract with `prepare_initial_condition`: a single `torch.save`d
        # `aurora.Batch` at `<initial_condition>/batch.pt`.
        batch = torch.load(
            Path(args.initial_condition) / "batch.pt", map_location=device, weights_only=False
        )
        # Expected contract with `prepare_forcing_trajectory`: a single `torch.save`d 1-D tensor
        # of per-step forcing values, at `args.forcing_trajectory`.
        forcing = torch.load(args.forcing_trajectory, map_location=device, weights_only=True)

        logger.info(f"Rolling out {args.num_steps} steps...")
        pbar = tqdm(range(args.num_steps), desc="Rolling out")
        with torch.no_grad():
            for step in pbar:
                pred = model(batch, forcing=forcing[step].item())
                mean_abs_surf = torch.stack([v.abs().mean() for v in pred.surf_vars.values()]).mean()
                date = pred.metadata.time[0]
                rollout_step = pred.metadata.rollout_step
                pbar.set_postfix(date=str(date), rollout_step=rollout_step)
                mlflow.log_metric("forcing_co2_ppm", forcing[step].item(), step=step)
                mlflow.log_metric("mean_abs_surf", mean_abs_surf.item(), step=step)
                mlflow.log_metric("rollout_step", rollout_step, step=step)
                batch = advance(batch, pred)

        logger.info("Rollout complete.")
    except Exception as e:
        print(e)
        sys.exit(1)


if __name__ == "__main__":
    main()
