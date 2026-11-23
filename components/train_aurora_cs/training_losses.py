"""Training losses for `train_aurora_cs`."""

import torch
import torch.nn.functional as F

from aurora import Batch
from aurora.normalisation import normalise_atmos_var, normalise_surf_var

from training_data import require_finite

LOSS_KINDS = ("raw", "normalised")
# Relative weight of the surface and atmospheric groups in the normalised loss.
LOSS_GROUP_WEIGHTS = {"surf_vars": 1.0, "atmos_vars": 1.0}


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


def normalised_step_loss(pred: Batch, target: dict) -> torch.Tensor:
    """Loss that treats every variable and pressure level equally, unlike the raw-unit MSE
    (which is dominated by geopotential and pressure, and ignores humidity).

    Errors are divided by Aurora's per-variable (and per-level) scales, squared, weighted by
    cos(latitude) normalised to mean 1, and averaged. Surface and atmospheric variables are
    averaged within their own group, then combined with `LOSS_GROUP_WEIGHTS`, so the 4 surface
    variables are not swamped by the 65 variable-level terms.
    """
    lat = pred.metadata.lat
    lat_weights = torch.cos(torch.deg2rad(lat))
    lat_weights = (lat_weights / lat_weights.mean()).unsqueeze(-1)  # (H, 1)
    levels = pred.metadata.atmos_levels
    when = pred.metadata.time[0]

    group_losses = {}
    for group, predicted in (("surf_vars", pred.surf_vars), ("atmos_vars", pred.atmos_vars)):
        terms = []
        for name, truth in target[group].items():
            context = f"{group}.{name} at {when}"
            expected = truth.to(predicted[name])
            require_finite(expected, f"target.{context}")
            require_finite(predicted[name], f"prediction.{context}")
            if group == "surf_vars":
                diff = normalise_surf_var(predicted[name], name) - normalise_surf_var(expected, name)
            else:
                diff = normalise_atmos_var(predicted[name], name, levels) - normalise_atmos_var(
                    expected, name, levels
                )
            term = (diff**2 * lat_weights.to(diff)).mean()
            require_finite(term, f"loss.{context}")
            terms.append(term)
        group_losses[group] = torch.stack(terms).mean()

    weights = LOSS_GROUP_WEIGHTS
    loss = sum(weights[g] * group_losses[g] for g in group_losses) / sum(weights[g] for g in group_losses)
    require_finite(loss, f"total loss at {when}")
    return loss


def compute_loss(pred: Batch, target: dict, kind: str = "raw") -> torch.Tensor:
    if kind == "raw":
        return step_loss(pred, target)
    if kind == "normalised":
        return normalised_step_loss(pred, target)
    raise ValueError(f"Unknown loss '{kind}'; expected one of {LOSS_KINDS}.")
