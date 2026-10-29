import sys
from datetime import date, datetime
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


def test_training_accepts_complete_shards(shard):
    path, times = shard
    assert training.find_incomplete_shards(path.parent, [times[1], times[2]]) == {}


def test_training_reports_every_damaged_variable(shard, remove_chunks):
    path, times = shard
    remove_chunks(path, "2m_temperature")
    remove_chunks(path, "temperature", 1)
    problems = training.find_incomplete_shards(path.parent, [times[1]])
    assert problems == {"1995-08.zarr": ["2m_temperature: 0/4 chunks", "temperature: 3/4 chunks"]}


def test_training_only_checks_shards_it_will_read(shard, write_shard, remove_chunks):
    path, times = shard
    remove_chunks(path, "2m_temperature")  # August is damaged, but no example touches it
    write_shard(path.parent / "1995-09.zarr")
    assert training.find_incomplete_shards(path.parent, [pd.Timestamp("1995-09-01 06:00")]) == {}


def test_training_flags_a_shard_with_no_metadata(tmp_path):
    problems = training.find_incomplete_shards(tmp_path, [pd.Timestamp("1995-08-01 06:00")])
    assert set(problems) == {"1995-08.zarr"}
    assert all(issue.endswith("missing") for issue in problems["1995-08.zarr"])


@pytest.fixture
def era5_times(tmp_path):
    times = pd.date_range("1989-12-31", "2016-01-01 18:00", freq="6h")
    cache = {}
    for month in times.to_period("M").unique():
        (tmp_path / f"{month}.zarr").mkdir()
        cache[(month.year, month.month)] = xr.Dataset(
            coords={"time": times[times.to_period("M") == month]}
        )
    return tmp_path, cache, times


@pytest.mark.parametrize(
    "start_date,end_date",
    [
        (None, None),
        (date(1990, 1, 1), None),
        (None, date(2015, 12, 31)),
        (date(1990, 1, 1), date(2015, 12, 31)),
        (date(2000, 2, 29), date(2000, 2, 29)),
        (date(2020, 1, 1), date(2020, 12, 31)),
    ],
)
def test_discover_examples_respects_inclusive_dates(era5_times, start_date, end_date):
    path, cache, times = era5_times
    examples = training.discover_examples(path, cache, start_date, end_date)
    selected = times
    if start_date is not None:
        selected = selected[selected >= pd.Timestamp(start_date)]
    if end_date is not None:
        selected = selected[selected < pd.Timestamp(end_date) + pd.Timedelta(days=1)]
    assert examples == list(selected[1:-1])
    if start_date == date(1990, 1, 1) and end_date == date(2015, 12, 31):
        assert examples[0] == pd.Timestamp("1990-01-01 06:00")
        assert examples[-1] == pd.Timestamp("2015-12-31 12:00")
        assert len(examples) == 37982


def test_date_filter_preserves_gap_detection(tmp_path, monkeypatch):
    times = pd.date_range("1995-01-01", periods=8, freq="6h").delete(3)
    monkeypatch.setattr(training, "discover_times", lambda *_: list(times))
    assert training.discover_examples(
        tmp_path, {}, date(1995, 1, 1), date(1995, 1, 2)
    ) == [times[1], times[4], times[5]]


@pytest.mark.parametrize("value", ["unbounded", "1990-01-01"])
def test_parse_date(value):
    expected = None if value == "unbounded" else date(1990, 1, 1)
    assert training.parse_date(value) == expected


@pytest.mark.parametrize(
    "date_args,error",
    [
        (["--start-date", "1990-02-30"], "invalid parse_date value"),
        (["--end-date", "2015-12-31T18:00:00"], "invalid parse_date value"),
        (
            ["--start-date", "2016-01-01", "--end-date", "1990-01-01"],
            "--start-date must be on or before --end-date",
        ),
    ],
)
def test_invalid_date_arguments_fail_before_training(monkeypatch, capsys, date_args, error):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "train_aurora_cs.py",
            "--era5-data", "era5",
            "--co2-data", "co2.csv",
            "--model-name", "test-model",
            *date_args,
        ],
    )
    with pytest.raises(SystemExit) as exc:
        training.main()
    assert exc.value.code == 2
    assert error in capsys.readouterr().err


def test_main_passes_bounds_and_rejects_empty_selection(monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "train_aurora_cs.py",
            "--era5-data", "era5",
            "--co2-data", "co2.csv",
            "--model-name", "test-model",
            "--start-date", "1990-01-01",
            "--end-date", "2015-12-31",
        ],
    )
    monkeypatch.setattr(training.pd, "read_csv", lambda *_, **__: pd.DataFrame())
    monkeypatch.setattr(
        training.xr,
        "open_zarr",
        lambda *_, **__: xr.Dataset({name: 0.0 for name in training.STATIC_VAR_MAP.values()}),
    )

    def discover_examples(path, cache, start_date, end_date):
        assert start_date == date(1990, 1, 1)
        assert end_date == date(2015, 12, 31)
        return []

    monkeypatch.setattr(training, "discover_examples", discover_examples)
    with pytest.raises(RuntimeError, match="No adjacent-triple training examples"):
        training.main()
