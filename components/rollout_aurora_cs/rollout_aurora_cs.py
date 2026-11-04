"""Pipeline component: run the AuroraCS autoregressive rollout.

Loads a registered AuroraCS model and an initial condition, then steps the model forward
`num_steps` times conditioned on a CO2 forcing trajectory, logging rollout progress to MLflow.

Every step's prediction is written as NetCDF to `<prefix>/<YYYYmmddTHHMMSS>.nc` on the workspace
blob datastore (named after the time the prediction refers to), and the folder is registered as a
data asset once the rollout is done.
"""

import argparse
import logging
import tempfile
from dataclasses import replace
from pathlib import Path

import mlflow
import torch
from azure.ai.ml.constants import AssetTypes
from tqdm import tqdm

from aurora_cs import configure_mlflow_auth, load_checkpoint
from data.azure_ml import get_blob_filesystem, get_ml_client, next_version, register_data_asset

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
    parser.add_argument("--prefix", required=True, help="blob path prefix under the datastore")
    parser.add_argument("--asset-name", required=True)
    parser.add_argument("--datastore-name", default="workspaceblobstore")
    parser.add_argument("--subscription-id", required=True)
    parser.add_argument("--resource-group", required=True)
    parser.add_argument("--workspace-name", required=True)
    parser.add_argument(
        "--managed-identity-client-id",
        default=None,
        help="Client ID of the compute's user-assigned identity; omit to use its "
        "system-assigned identity.",
    )
    args = parser.parse_args()

    configure_mlflow_auth()
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

    prefix = args.prefix.strip("/")
    identity = args.managed_identity_client_id or None
    ml_client = get_ml_client(args.subscription_id, args.resource_group, args.workspace_name, identity)
    fs, container = get_blob_filesystem(ml_client, args.datastore_name, identity)

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
            with tempfile.TemporaryDirectory() as tmp:
                local = Path(tmp) / "step.nc"
                pred.to("cpu").to_netcdf(local)
                fs.put(str(local), f"{container}/{prefix}/{date:%Y%m%dT%H%M%S}.nc")
            batch = advance(batch, pred)

    asset_path = f"azureml://datastores/{args.datastore_name}/paths/{prefix}"
    logger.info(f"Registering '{args.asset_name}' from {asset_path}...")
    register_data_asset(
        ml_client,
        name=args.asset_name,
        path=asset_path,
        asset_type=AssetTypes.URI_FOLDER,
        description=f"AuroraCS rollout predictions, {args.num_steps} 6-hourly steps (NetCDF)",
        version=next_version(ml_client, args.asset_name),
    )
    logger.info("Rollout complete.")


if __name__ == "__main__":
    main()
