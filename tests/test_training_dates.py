import sys
from datetime import date

import pandas as pd
import pytest
import xarray as xr

from components.train_aurora_cs import train_aurora_cs as training


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
