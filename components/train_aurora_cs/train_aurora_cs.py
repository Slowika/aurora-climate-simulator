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
import tempfile
from datetime import date
from pathlib import Path

import mlflow
import pandas as pd
import torch
import xarray as xr

from aurora_cs import AuroraCS, configure_mlflow_auth, save_checkpoint
from aurora_cs.training_checkpoint import (
    check_writable,
    load_latest_training_state,
    open_checkpoint_store,
    restore_training_state,
    save_training_state,
)

from training_constants import CHECKPOINT, MODEL_CONFIG, SIX_HOURS, STATIC_VAR_MAP
from training_data import discover_examples, find_incomplete_shards, load_example, require_finite
from training_losses import step_loss

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
logger = logging.getLogger(__name__)


def register_model(model: AuroraCS, config: dict, model_name: str) -> None:
    """Log `model` (weights + `config`, see `save_checkpoint`) to the active run and register it."""
    with tempfile.TemporaryDirectory() as tmp:
        save_checkpoint(model, config, tmp)
        mlflow.log_artifacts(tmp, artifact_path="model")
    mlflow.register_model(f"{mlflow.active_run().info.artifact_uri}/model", model_name)


def parse_date(value: str) -> date | None:
    """Accept an ISO date or the command job template's explicit unbounded default."""
    return None if value == "unbounded" else date.fromisoformat(value)


def train_loop(
    model: AuroraCS,
    optimizer: torch.optim.Optimizer,
    examples: list[pd.Timestamp],
    get_example,
    epochs: int,
    *,
    checkpoint_dir: str | None = None,
    checkpoint_every: int = 2000,
    fingerprint: dict | None = None,
    checkpoint_fs=None,
) -> None:
    """Train for `epochs`, resuming from the newest checkpoint in `checkpoint_dir` if present.

    Epoch `e` visits `examples` in an order seeded by `e`, so a resumed run sees exactly the
    examples an uninterrupted one would. `get_example(cur_t)` returns `(batch, forcing, target)`.
    """
    fingerprint = fingerprint or {}
    epoch_loss, position, global_step, start_epoch = 0.0, 0, 0, 0
    if checkpoint_dir:
        device = next(model.parameters()).device
        state = load_latest_training_state(checkpoint_dir, device, checkpoint_fs)
        if state:
            restore_training_state(state, model, optimizer, fingerprint)
            start_epoch, position = state["epoch"], state["position"]
            global_step, epoch_loss = state["global_step"], state["epoch_loss"]
            logger.info(f"Resumed at epoch {start_epoch}, example {position}, step {global_step}.")
    last_saved = global_step if checkpoint_dir else None

    def save(epoch: int, position: int) -> None:
        nonlocal last_saved
        save_training_state(
            checkpoint_dir, model, optimizer, epoch=epoch, position=position,
            global_step=global_step, epoch_loss=epoch_loss, fingerprint=fingerprint,
            fs=checkpoint_fs,
        )
        last_saved = global_step

    for epoch in range(start_epoch, epochs):
        order = list(examples)
        random.Random(epoch).shuffle(order)
        for i in range(position, len(order)):
            cur_t = order[i]
            batch, forcing, target = get_example(cur_t)
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
            if checkpoint_dir and global_step % checkpoint_every == 0:
                save(epoch, i + 1)
        if checkpoint_dir and last_saved != global_step:
            save(epoch, len(order))
        mean_loss = epoch_loss / len(order)
        mlflow.log_metric("epoch_loss", mean_loss, step=epoch)
        logger.info(f"  epoch {epoch}: mean loss = {mean_loss:.6f}")
        epoch_loss, position = 0.0, 0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--era5-data", required=True)
    parser.add_argument("--co2-data", required=True)
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument(
        "--checkpoint-dir",
        help="Local folder for resumable checkpoints; training resumes from the newest one found.",
    )
    parser.add_argument(
        "--checkpoint-prefix",
        help="Blob path prefix for resumable checkpoints, written as the compute's identity. "
        "Needs --checkpoint-account and --checkpoint-container.",
    )
    parser.add_argument("--checkpoint-account", help="Storage account holding the checkpoints.")
    parser.add_argument("--checkpoint-container", help="Blob container holding the checkpoints.")
    parser.add_argument("--checkpoint-every", type=int, default=2000, help="Steps between checkpoints.")
    parser.add_argument(
        "--start-date", type=parse_date, help="First training date, inclusive (YYYY-MM-DD)."
    )
    parser.add_argument(
        "--end-date", type=parse_date, help="Last training date, inclusive (YYYY-MM-DD)."
    )
    args = parser.parse_args()
    if args.start_date is not None and args.end_date is not None and args.start_date > args.end_date:
        parser.error("--start-date must be on or before --end-date")
    try:
        checkpoint_fs, checkpoint_path = open_checkpoint_store(
            args.checkpoint_dir,
            prefix=args.checkpoint_prefix,
            account=args.checkpoint_account,
            container=args.checkpoint_container,
        )
    except ValueError as e:
        parser.error(str(e))

    configure_mlflow_auth()
    if checkpoint_fs is not None:
        # Before any data loading or training, so a permission or network problem shows up now.
        check_writable(checkpoint_fs, checkpoint_path)
        logger.info(f"Checkpoints: {checkpoint_path} (writable)")
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
    # `epochs` is deliberately excluded so a finished run's checkpoint can be extended.
    fingerprint = {
        "lr": args.lr,
        "num_examples": len(examples),
        "first_example": str(examples[0]),
        "last_example": str(examples[-1]),
        "model_config": MODEL_CONFIG,
    }
    train_loop(
        model,
        optimizer,
        examples,
        lambda cur_t: load_example(era5_data, cache, static, co2, cur_t, device),
        args.epochs,
        checkpoint_dir=checkpoint_path,
        checkpoint_fs=checkpoint_fs,
        checkpoint_every=args.checkpoint_every,
        fingerprint=fingerprint,
    )

    logger.info(f"Registering trained model as '{args.model_name}'...")
    register_model(model, MODEL_CONFIG, args.model_name)
    logger.info("Done.")


if __name__ == "__main__":
    main()
