"""Training losses for `train_aurora_cs`."""

import torch
import torch.nn.functional as F

from aurora import Batch

from training_data import require_finite


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
