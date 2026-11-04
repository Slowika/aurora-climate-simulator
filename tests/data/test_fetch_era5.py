import shutil

import fsspec
import numpy as np
import pandas as pd
import pytest
import xarray as xr

from data import fetch_era5


@pytest.fixture
def source():
    return xr.Dataset(
        {
            name: (("time", "latitude", "longitude"), [[[np.nan, np.nan]], [[0.0, 1.0]]])
            for name in fetch_era5.STATIC_VAR_MAP.values()
        },
        coords={
            "time": pd.to_datetime(["1900-01-01", "1990-01-01"]),
            "latitude": [0.0],
            "longitude": [0.0, 180.0],
        },
    )


def test_static_fields_use_requested_date(source, tmp_path):
    path = tmp_path / "static.zarr"
    when = pd.Timestamp("1990-01-01")
    fetch_era5.fetch_static(source, path, when)
    with xr.open_zarr(path) as actual:
        xr.testing.assert_equal(actual, source.sel(time=when, drop=True))
        assert all(np.isfinite(actual[name].values).all() for name in actual.data_vars)


@pytest.mark.parametrize("bad_value", [np.nan, np.inf, -np.inf])
def test_invalid_static_fields_are_not_written(source, tmp_path, bad_value):
    source["soil_type"].values[1, 0, 0] = bad_value
    path = tmp_path / "static.zarr"
    with pytest.raises(ValueError, match="soil_type at 1990-01-01"):
        fetch_era5.fetch_static(source, path, pd.Timestamp("1990-01-01"))
    assert not path.exists()


def test_complete_shard_is_complete(shard):
    path, _ = shard
    assert fetch_era5.shard_is_complete(fsspec.filesystem("file"), str(path))


@pytest.mark.parametrize("variable,count", [("2m_temperature", None), ("temperature", 1), ("geopotential", 2)])
def test_shard_with_unwritten_chunks_is_incomplete(shard, remove_chunks, variable, count):
    """The failure seen in practice: metadata present, some or all chunks never written."""
    path, _ = shard
    remove_chunks(path, variable, count)
    assert (path / ".zmetadata").exists()
    assert not fetch_era5.shard_is_complete(fsspec.filesystem("file"), str(path))


def test_shard_missing_a_variable_is_incomplete(shard):
    path, _ = shard
    shutil.rmtree(path / "specific_humidity")
    assert not fetch_era5.shard_is_complete(fsspec.filesystem("file"), str(path))


def test_shard_without_the_weather_data_is_incomplete(tmp_path):
    assert not fetch_era5.shard_is_complete(fsspec.filesystem("file"), str(tmp_path / "absent.zarr"))
