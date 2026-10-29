import sys
from datetime import datetime
from unittest.mock import Mock

import pandas as pd
import pytest
import torch
import xarray as xr
from aurora import Batch, Metadata

from components.train_aurora_cs import train_aurora_cs as training


@pytest.fixture
def prediction():
    return Batch(
        surf_vars={"2t": torch.ones(1, 1, 2, 2, requires_grad=True)},
        static_vars={},
        atmos_vars={"t": torch.ones(1, 1, 1, 2, 2, requires_grad=True)},
        metadata=Metadata(
            lat=torch.tensor([90.0, -90.0]),
            lon=torch.tensor([0.0, 180.0]),
            time=(datetime(1990, 1, 1, 12),),
            atmos_levels=(1000,),
        ),
    )


@pytest.fixture
def target(prediction):
    return {
        "surf_vars": {name: torch.zeros_like(v) for name, v in prediction.surf_vars.items()},
        "atmos_vars": {name: torch.zeros_like(v) for name, v in prediction.atmos_vars.items()},
    }


def test_finite_loss_and_gradients_unchanged(prediction, target):
    loss = training.step_loss(prediction, target)
    torch.testing.assert_close(loss, torch.tensor(1.0))
    loss.backward()
    for value in (*prediction.surf_vars.values(), *prediction.atmos_vars.values()):
        torch.testing.assert_close(value.grad, torch.full_like(value, 0.25))


@pytest.mark.parametrize("group,name", [("surf_vars", "2t"), ("atmos_vars", "t")])
@pytest.mark.parametrize("source", ["target", "prediction"])
@pytest.mark.parametrize("bad_value", [float("nan"), float("inf"), -float("inf")])
def test_nonfinite_prediction_or_target_fails(prediction, target, group, name, source, bad_value):
    values = target[group] if source == "target" else getattr(prediction, group)
    with torch.no_grad():
        values[name].fill_(bad_value)
    with pytest.raises(FloatingPointError, match=rf"{source}.{group}.{name} at 1990-01-01"):
        training.step_loss(prediction, target)


def test_loss_overflow_is_rejected(prediction, target):
    with torch.no_grad():
        prediction.surf_vars["2t"].fill_(1e30)
    with pytest.raises(FloatingPointError, match="loss.surf_vars.2t"):
        training.step_loss(prediction, target)


@pytest.mark.parametrize("bad_source", ["input", "forcing"])
def test_load_example_rejects_nonfinite_data(monkeypatch, prediction, target, tmp_path, bad_source):
    if bad_source == "input":
        with torch.no_grad():
            prediction.atmos_vars["t"].fill_(float("nan"))
    monkeypatch.setattr(training, "build_batch", lambda *args: prediction)
    monkeypatch.setattr(training, "load_target", lambda *args: target)
    monkeypatch.setattr(
        training, "co2_ppm_for", lambda *args: float("nan") if bad_source == "forcing" else 350.0
    )
    with pytest.raises(FloatingPointError, match="input.atmos_vars.t|CO2 forcing"):
        training.load_example(
            tmp_path, {}, xr.Dataset(), pd.DataFrame(), pd.Timestamp("1990-01-01 06:00"),
            torch.device("cpu"),
        )


@pytest.mark.parametrize("bad_stage", ["static", "gradient"])
def test_main_fails_before_unsafe_work(monkeypatch, prediction, target, bad_stage):
    monkeypatch.setattr(
        sys, "argv",
        ["train_aurora_cs.py", "--era5-data", "era5", "--co2-data", "co2.csv", "--model-name", "test"],
    )
    monkeypatch.setattr(training.torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(training.pd, "read_csv", lambda *args, **kwargs: pd.DataFrame())
    static = xr.Dataset({
        name: float("nan") if bad_stage == "static" else 0.0
        for name in training.STATIC_VAR_MAP.values()
    })
    monkeypatch.setattr(training.xr, "open_zarr", lambda *args, **kwargs: static)
    monkeypatch.setattr(
        training, "discover_examples", lambda *args: [pd.Timestamp("1990-01-01 06:00")]
    )
    monkeypatch.setattr(training, "find_incomplete_shards", lambda *args: {})
    monkeypatch.setattr(training, "load_example", lambda *args: (prediction, 350.0, target))
    model = torch.nn.Module()
    model.register_parameter("weight", torch.nn.Parameter(torch.tensor(1.0)))
    model.aurora = Mock()
    model_factory = Mock(return_value=model)
    monkeypatch.setattr(training, "AuroraCS", model_factory)
    monkeypatch.setattr(model, "forward", lambda *args, **kwargs: prediction)
    monkeypatch.setattr(training, "step_loss", lambda *args: model.weight * 1.0)
    model.weight.register_hook(lambda grad: torch.full_like(grad, float("nan")))
    optimizer_step = Mock()
    monkeypatch.setattr(training.torch.optim.Adam, "step", optimizer_step)
    log_metric = Mock()
    monkeypatch.setattr(training.mlflow, "log_metric", log_metric)

    with pytest.raises(FloatingPointError, match="ERA5 static|gradient.weight"):
        training.main()

    optimizer_step.assert_not_called()
    log_metric.assert_not_called()
    if bad_stage == "static":
        model_factory.assert_not_called()
    else:
        assert model.weight.item() == 1.0
