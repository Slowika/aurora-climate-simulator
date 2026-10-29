from datetime import datetime

import pytest
import torch
from aurora import Batch, Metadata
from aurora.normalisation import scales

from aurora_cs import training_checkpoint as ckpt
from components.train_aurora_cs import train_aurora_cs as training

SURF = ("2t", "10u", "10v", "msl")
ATMOS = ("z", "u", "v", "t", "q")
LEVELS = (1000, 500)
H, W = 3, 4


def _batch(surf, atmos):
    return Batch(
        surf_vars=surf,
        static_vars={},
        atmos_vars=atmos,
        metadata=Metadata(
            lat=torch.tensor([60.0, 0.0, -60.0]),
            lon=torch.linspace(0, 270, W),
            time=(datetime(2000, 1, 1),),
            atmos_levels=LEVELS,
        ),
    )


def _truth():
    surf = {n: torch.randn(1, 1, H, W) for n in SURF}
    atmos = {n: torch.randn(1, 1, len(LEVELS), H, W) for n in ATMOS}
    return {"surf_vars": surf, "atmos_vars": atmos}


def _level_scales(name):
    return torch.tensor([scales[f"{name}_{level}"] for level in LEVELS]).view(1, 1, -1, 1, 1)


def pred_with_error(truth, *, std_errors=(), rows=None):
    """Prediction that is off by one Aurora standard deviation in the given variables, in the
    given latitude rows (all rows if `rows` is None)."""
    row_mask = torch.zeros(H, 1)
    row_mask[list(range(H)) if rows is None else rows] = 1.0
    surf = {n: v + (scales[n] * row_mask if n in std_errors else 0) for n, v in truth["surf_vars"].items()}
    atmos = {
        n: v + (_level_scales(n) * row_mask if n in std_errors else 0)
        for n, v in truth["atmos_vars"].items()
    }
    return _batch(surf, atmos)


def test_unit_error_everywhere_gives_loss_of_one():
    truth = _truth()
    pred = pred_with_error(truth, std_errors=SURF + ATMOS)
    torch.testing.assert_close(training.normalised_step_loss(pred, truth), torch.tensor(1.0))


def test_perfect_prediction_gives_zero_loss():
    truth = _truth()
    assert training.normalised_step_loss(pred_with_error(truth), truth).item() == pytest.approx(0, abs=1e-6)


@pytest.mark.parametrize("variable", ["z", "u", "q", "t"])
def test_every_atmospheric_variable_counts_equally(variable):
    """An error of one standard deviation matters the same whatever the variable's units."""
    truth = _truth()
    loss = training.normalised_step_loss(pred_with_error(truth, std_errors=(variable,)), truth)
    torch.testing.assert_close(loss, torch.tensor(1.0 / len(ATMOS) / 2))


@pytest.mark.parametrize("variable", SURF)
def test_every_surface_variable_counts_equally(variable):
    truth = _truth()
    loss = training.normalised_step_loss(pred_with_error(truth, std_errors=(variable,)), truth)
    torch.testing.assert_close(loss, torch.tensor(1.0 / len(SURF) / 2))


def test_raw_loss_is_dominated_by_large_valued_variables():
    """Documents the problem the normalised loss fixes."""
    truth = _truth()
    humidity = training.step_loss(pred_with_error(truth, std_errors=("q",)), truth)
    geopotential = training.step_loss(pred_with_error(truth, std_errors=("z",)), truth)
    assert geopotential / humidity > 1e9


def test_latitude_weighting_favours_the_equator():
    truth = _truth()
    polar = training.normalised_step_loss(pred_with_error(truth, std_errors=("2t",), rows=[0]), truth)
    equator = training.normalised_step_loss(pred_with_error(truth, std_errors=("2t",), rows=[1]), truth)
    torch.testing.assert_close(equator / polar, torch.tensor(2.0))


def test_non_finite_values_are_rejected():
    truth = _truth()
    pred = pred_with_error(truth)
    pred.atmos_vars["q"][0, 0, 0, 0, 0] = float("nan")
    with pytest.raises(FloatingPointError, match="prediction.atmos_vars.q"):
        training.normalised_step_loss(pred, truth)


def test_loss_selector():
    truth = _truth()
    pred = pred_with_error(truth, std_errors=("t",))
    torch.testing.assert_close(training.compute_loss(pred, truth, "raw"), training.step_loss(pred, truth))
    torch.testing.assert_close(
        training.compute_loss(pred, truth, "normalised"), training.normalised_step_loss(pred, truth)
    )
    with pytest.raises(ValueError, match="Unknown loss"):
        training.compute_loss(pred, truth, "mae")


def test_checkpoints_from_the_other_loss_are_refused(tmp_path):
    model = torch.nn.Linear(2, 2)
    optimizer = torch.optim.Adam(model.parameters())
    ckpt.save_training_state(
        tmp_path, model, optimizer, epoch=0, position=1, global_step=1, epoch_loss=0.0,
        fingerprint={"lr": 1e-4, "loss": "normalised"},
    )
    state = ckpt.load_latest_training_state(tmp_path, torch.device("cpu"))
    with pytest.raises(RuntimeError, match="loss"):
        ckpt.restore_training_state(state, model, optimizer, {"lr": 1e-4})
