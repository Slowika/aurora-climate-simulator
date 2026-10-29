import shutil
from pathlib import Path

import fsspec
import numpy as np
import pandas as pd
import pytest
import xarray as xr

from components.train_aurora_cs import train_aurora_cs as training


@pytest.fixture
def fetcher(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "data"))
    import fetch_era5

    return fetch_era5


def write_shard(fetcher, path, steps=4):
    """A monthly shard laid out like `fetch_month` writes it (one chunk per timestep)."""
    times = pd.date_range("1995-08-01", periods=steps, freq="6h")
    surf = {n: (("time", "latitude", "longitude"), np.ones((steps, 2, 3), "f4")) for n in fetcher.SURF_VAR_MAP.values()}
    atmos = {
        n: (("time", "level", "latitude", "longitude"), np.ones((steps, 2, 2, 3), "f4"))
        for n in fetcher.ATMOS_VAR_MAP.values()
    }
    ds = xr.Dataset({**surf, **atmos}, coords={"time": times, "level": [1000, 500]})
    ds.chunk({"time": 1}).to_zarr(path, mode="w")
    return times


def remove_chunks(path, variable, count=None):
    chunks = sorted(p for p in (Path(path) / variable).iterdir() if p.name[0].isdigit())
    for p in chunks[:count]:
        p.unlink()


@pytest.fixture
def shard(fetcher, tmp_path):
    path = tmp_path / "1995-08.zarr"
    times = write_shard(fetcher, path)
    return path, times


def test_complete_shard_is_complete(fetcher, shard):
    path, _ = shard
    assert fetcher.shard_is_complete(fsspec.filesystem("file"), str(path))


@pytest.mark.parametrize("variable,count", [("2m_temperature", None), ("temperature", 1), ("geopotential", 2)])
def test_shard_with_unwritten_chunks_is_incomplete(fetcher, shard, variable, count):
    """The failure seen in practice: metadata present, some or all chunks never written."""
    path, _ = shard
    remove_chunks(path, variable, count)
    assert (path / ".zmetadata").exists()
    assert not fetcher.shard_is_complete(fsspec.filesystem("file"), str(path))


def test_shard_missing_a_variable_is_incomplete(fetcher, shard):
    path, _ = shard
    shutil.rmtree(path / "specific_humidity")
    assert not fetcher.shard_is_complete(fsspec.filesystem("file"), str(path))


def test_shard_without_the_weather_data_is_incomplete(fetcher, tmp_path):
    assert not fetcher.shard_is_complete(fsspec.filesystem("file"), str(tmp_path / "absent.zarr"))


def test_training_accepts_complete_shards(shard):
    path, times = shard
    assert training.find_incomplete_shards(path.parent, [times[1], times[2]]) == {}


def test_training_reports_every_damaged_variable(shard):
    path, times = shard
    remove_chunks(path, "2m_temperature")
    remove_chunks(path, "temperature", 1)
    problems = training.find_incomplete_shards(path.parent, [times[1]])
    assert problems == {"1995-08.zarr": ["2m_temperature: 0/4 chunks", "temperature: 3/4 chunks"]}


def test_training_only_checks_shards_it_will_read(fetcher, shard):
    path, times = shard
    remove_chunks(path, "2m_temperature")  # August is damaged, but no example touches it
    write_shard(fetcher, path.parent / "1995-09.zarr")
    assert training.find_incomplete_shards(path.parent, [pd.Timestamp("1995-09-01 06:00")]) == {}


def test_training_flags_a_shard_with_no_metadata(tmp_path):
    problems = training.find_incomplete_shards(tmp_path, [pd.Timestamp("1995-08-01 06:00")])
    assert set(problems) == {"1995-08.zarr"}
    assert all(issue.endswith("missing") for issue in problems["1995-08.zarr"])
