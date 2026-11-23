from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from data import fetch_era5


@pytest.fixture
def write_shard():
    def write(path, steps=4):
        """A monthly shard laid out like `fetch_month` writes it (one chunk per timestep)."""
        times = pd.date_range("1995-08-01", periods=steps, freq="6h")
        surf = {
            n: (("time", "latitude", "longitude"), np.ones((steps, 2, 3), "f4"))
            for n in fetch_era5.SURF_VAR_MAP.values()
        }
        atmos = {
            n: (("time", "level", "latitude", "longitude"), np.ones((steps, 2, 2, 3), "f4"))
            for n in fetch_era5.ATMOS_VAR_MAP.values()
        }
        ds = xr.Dataset({**surf, **atmos}, coords={"time": times, "level": [1000, 500]})
        ds.chunk({"time": 1}).to_zarr(path, mode="w")
        return times

    return write


@pytest.fixture
def remove_chunks():
    def remove(path, variable, count=None):
        chunks = sorted(p for p in (Path(path) / variable).iterdir() if p.name[0].isdigit())
        for p in chunks[:count]:
            p.unlink()

    return remove


@pytest.fixture
def shard(tmp_path, write_shard):
    path = tmp_path / "1995-08.zarr"
    times = write_shard(path)
    return path, times
