"""Tests for `data.fetch_era5`."""

import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from data import fetch_era5


@pytest.fixture
def source() -> xr.Dataset:
    """Create a minimal static ERA5 dataset for testing."""
    return xr.Dataset(
        {
            name: (
                ("time", "latitude", "longitude"),
                [[[np.nan, np.nan]],
                [[0.0, 1.0]]],
            )
            for name in fetch_era5.STATIC_VAR_MAP.values()
        },
        coords={
            "time": pd.to_datetime(["1900-01-01", "1990-01-01"]),
            "latitude": [0.0],
            "longitude": [0.0, 180.0],
        },
    )


def test_static_fields_use_requested_date(source: xr.Dataset, tmp_path: Path) -> None:
    """Test that static fields are correctly fetched for the requested date."""
    path = tmp_path / "static.zarr"
    when = pd.Timestamp("1990-01-01")
    fetch_era5.fetch_static(source, when, path)
    with xr.open_zarr(path) as actual:
        xr.testing.assert_equal(actual, source.sel(time=when, drop=True))
        assert all(np.isfinite(actual[name].values).all() for name in actual.data_vars)


@pytest.mark.parametrize("bad_value", [np.nan, np.inf, -np.inf])
def test_invalid_static_fields_are_not_written(
    source: xr.Dataset,
    tmp_path: Path,
    bad_value: float,
) -> None:
    """Test that invalid (e.g., non-finite) static fields raise and are not written."""
    source["soil_type"].to_numpy()[1, 0, 0] = bad_value
    path = tmp_path / "static.zarr"
    with pytest.raises(ValueError, match="Non-finite static variable"):
        fetch_era5.fetch_static(source, pd.Timestamp("1990-01-01"), path)
    assert not path.exists()



def write_shard(path: Path, steps: int = 4) -> pd.DatetimeIndex:
    """Write a shard laid out as `fetch_month` writes it (one chunk per timestep)."""
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
    ds.chunk({"time": 1}).to_zarr(path, mode="w")  # type: ignore[invalid-argument-type]
    return times


def remove_chunks(path: Path, variable: str, count: int | None = None) -> None:
    """Remove a specified number of chunks for a given variable in a shard."""
    chunks = sorted(p for p in (Path(path) / variable).iterdir() if p.name[0].isdigit())
    for p in chunks[:count]:
        p.unlink()


@pytest.fixture
def shard(tmp_path: Path) -> tuple[Path, pd.DatetimeIndex]:
    """Create a complete shard for testing purposes."""
    path = tmp_path / "1995-08.zarr"
    times = write_shard(path)
    return path, times


def test_complete_shard_is_complete(shard: tuple[Path, pd.DatetimeIndex]) -> None:
    """Test that a complete shard is recognised as such."""
    path, _ = shard
    assert fetch_era5.shard_is_complete(path)


@pytest.mark.parametrize(
    ("variable", "count"),
    [("2m_temperature", None), ("temperature", 1), ("geopotential", 2)],
)
def test_shard_with_unwritten_chunks_is_incomplete(
    shard: tuple[Path, pd.DatetimeIndex],
    variable: str,
    count: int | None,
) -> None:
    """Test that a shard with some chunks missing is recognised as incomplete."""
    path, _ = shard
    remove_chunks(path, variable, count)
    assert (path / ".zmetadata").exists()
    assert not fetch_era5.shard_is_complete(path)


def test_shard_missing_a_variable_is_incomplete(
    shard: tuple[Path, pd.DatetimeIndex],
) -> None:
    """Test that a shard missing a variable is recognised as incomplete."""
    path, _ = shard
    shutil.rmtree(path / "specific_humidity")
    assert not fetch_era5.shard_is_complete(path)


def test_shard_without_the_weather_data_is_incomplete(tmp_path: Path) -> None:
    """Test that a shard without any weather data is recognised as incomplete."""
    assert not fetch_era5.shard_is_complete(tmp_path / "absent.zarr")
