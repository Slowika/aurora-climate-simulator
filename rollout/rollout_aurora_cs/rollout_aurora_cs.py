"""Pipeline component: run the AuroraCS autoregressive rollout.

Loads a registered AuroraCS model and an initial condition, then steps the model forward
`num_steps` times conditioned on a CO2 forcing trajectory, logging rollout progress to MLflow
and saving each step's prediction as a `torch.save`d `aurora.Batch` at
`<predictions>/step_<NNNN>.pt`.
"""

import argparse
import logging
from dataclasses import replace
from pathlib import Path

import mlflow
import torch
from tqdm import tqdm

from aurora_cs import load_checkpoint

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
    parser.add_argument("--predictions", required=True)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    logger.info(f"Loading registered model '{args.model_id}'...")
    model_dir = mlflow.artifacts.download_artifacts(artifact_uri=f"models:/{args.model_id}")
    model = load_checkpoint(model_dir).to(device).eval()

    # Expected contract with `prepare_initial_condition`: a single `torch.save`d
    # `aurora.Batch` at `<initial_condition>/batch.pt`.
    batch = torch.load(
        Path(args.initial_condition) / "batch.pt", map_location=device, weights_only=False
    )
    # Expected contract with `prepare_forcing_trajectory`: a single `torch.save`d 1-D tensor
    # of per-step forcing values, at `args.forcing_trajectory`.
    forcing = torch.load(args.forcing_trajectory, map_location=device, weights_only=True)
    if forcing.shape[0] < args.num_steps:
        raise ValueError(
            f"Forcing trajectory has {forcing.shape[0]} value(s) but num_steps is {args.num_steps}; "
            "it needs at least one value per rollout step."
        )

    predictions_dir = Path(args.predictions)
    predictions_dir.mkdir(parents=True, exist_ok=True)

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
            torch.save(pred.to("cpu"), predictions_dir / f"step_{step:04d}.pt")
            batch = advance(batch, pred)

    logger.info("Rollout complete.")


if __name__ == "__main__":
    main()
