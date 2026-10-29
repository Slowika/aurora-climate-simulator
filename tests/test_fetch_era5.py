from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xarray as xr


@pytest.fixture
def fetcher(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "data"))
    import fetch_era5

    return fetch_era5


@pytest.fixture
def source(fetcher):
    return xr.Dataset(
        {
            name: (("time", "latitude", "longitude"), [[[np.nan, np.nan]], [[0.0, 1.0]]])
            for name in fetcher.STATIC_VAR_MAP.values()
        },
        coords={
            "time": pd.to_datetime(["1900-01-01", "1990-01-01"]),
            "latitude": [0.0],
            "longitude": [0.0, 180.0],
        },
    )


def test_static_fields_use_requested_date(fetcher, source, tmp_path):
    path = tmp_path / "static.zarr"
    when = pd.Timestamp("1990-01-01")
    fetcher.fetch_static(source, path, when)
    with xr.open_zarr(path) as actual:
        xr.testing.assert_equal(actual, source.sel(time=when, drop=True))
        assert all(np.isfinite(actual[name].values).all() for name in actual.data_vars)


@pytest.mark.parametrize("bad_value", [np.nan, np.inf, -np.inf])
def test_invalid_static_fields_are_not_written(fetcher, source, tmp_path, bad_value):
    source["soil_type"].values[1, 0, 0] = bad_value
    path = tmp_path / "static.zarr"
    with pytest.raises(ValueError, match="soil_type at 1990-01-01"):
        fetcher.fetch_static(source, path, pd.Timestamp("1990-01-01"))
    assert not path.exists()
