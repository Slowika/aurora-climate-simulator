import shutil
from pathlib import Path

import fsspec
import numpy as np
import pandas as pd
import pytest
import xarray as xr


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
