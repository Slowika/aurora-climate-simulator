"""Tests for `data.fetch_co2_forcing`."""

import json
from pathlib import Path
from unittest.mock import Mock

import pandas as pd
import pytest

from data import fetch_co2_forcing


@pytest.fixture
def fixture_dir() -> Path:
    """Return the path to the fixtures (test data) directory."""
    return Path(__file__).parent / "fixtures"


@pytest.fixture
def mock_esgf_response(fixture_dir: Path, monkeypatch: pytest.MonkeyPatch) -> Mock:
    """Return a mocked response using real ESGF response JSON data."""
    response = Mock()
    response.json.return_value = json.loads(
        (fixture_dir / "response.json").read_text(encoding="utf-8"),
    )
    response.raise_for_status.return_value = None
    monkeypatch.setattr(
        fetch_co2_forcing.requests,
        "get",
        lambda *_args, **_kwargs: response,
    )
    return response


@pytest.fixture
def mock_netcdf_response(fixture_dir: Path) -> Mock:
    """Return a mocked response using a real CO2 concentration NetCDF data."""
    response = Mock()
    response.content = (fixture_dir / "co2.nc").read_bytes()
    response.raise_for_status.return_value = None
    return response


@pytest.mark.usefixtures("mock_esgf_response")
def test_find_file_url_prefers_ceda() -> None:
    """Test that `find_file_url` prefers the CEDA ESGF HTTPServer node."""
    assert fetch_co2_forcing.find_file_url("ssp245") == (
        "https://esgf.ceda.ac.uk/thredds/fileServer/esg_cmip6/input4MIPs/CMIP6/"
        "ScenarioMIP/UoM/UoM-MESSAGE-GLOBIOM-ssp245-1-2-1/atmos/mon/mole-fraction-of-"
        "carbon-dioxide-in-air/gr1-GMNHSH/v20181127/mole-fraction-of-carbon-dioxide-"
        "in-air_input4MIPs_GHGConcentrations_ScenarioMIP_UoM-MESSAGE-GLOBIOM-ssp245-1-"
        "2-1_gr1-GMNHSH_201501-250012.nc"
    )


def test_find_file_url_raises_if_no_ceda(
    mock_esgf_response: Mock,
) -> None:
    """Test that `find_file_url` raises if no CEDA-hosted URL is found."""
    for doc in mock_esgf_response.json.return_value["response"]["docs"]:
        doc["url"] = [u for u in doc["url"] if "ceda.ac.uk" not in u]
    with pytest.raises(RuntimeError, match=r"No ceda.ac.uk-hosted download URL found"):
        fetch_co2_forcing.find_file_url("ssp245")


def test_fetch_co2_series_reads_requested_window(
    monkeypatch: pytest.MonkeyPatch,
    mock_netcdf_response: Mock,
) -> None:
    """Test that `fetch_co2_series` reads the requested time window correctly."""
    monkeypatch.setattr(
        fetch_co2_forcing,
        "find_file_url",
        lambda _: "https://test/co2.nc",
    )
    monkeypatch.setattr(
        fetch_co2_forcing.requests,
        "get",
        lambda *_args, **_kwargs: mock_netcdf_response,
    )
    start = pd.Timestamp("2020-01-01")
    end = pd.Timestamp("2020-06-01")
    actual = fetch_co2_forcing.fetch_co2_series("ssp245", start, end)
    assert list(actual.columns) == ["time", "co2_ppm", "scenario"]
    assert actual["time"].dtype == "datetime64[ns]"
    assert actual["co2_ppm"].notna().all()
    assert actual["scenario"].tolist() == ["ssp245"] * len(actual)
    expected_months = pd.period_range(start, end, freq="M")
    actual_months = actual["time"].dt.to_period("M")
    assert actual_months.tolist() == expected_months.tolist()
