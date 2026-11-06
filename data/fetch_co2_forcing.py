"""CLI producer: fetch input4MIPs CO2 concentration data for a given SSP scenario.

This data represents the CMIP6 GHG forcing dataset, served via ESGF.
Writes CSV-formatted monthly global-mean CO2 concentrations to an output file.

    python fetch_co2_forcing.py --start-date 2020-01-01 --end-date 2020-06-01 \
        --scenario ssp245 --output ~/data/co2/co2.csv

Coverage starts 2015-01 (ScenarioMIP data does not extend into the historical period).
"""

import argparse
import logging
import tempfile
from pathlib import Path

import pandas as pd
import requests
import xarray as xr

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
logger = logging.getLogger(__name__)

ESGF_SEARCH_URL = "https://esgf-data.dkrz.de/esg-search/search/"

SCENARIO_SOURCE_IDS = {
    "ssp126": "UoM-IMAGE-ssp126-1-2-1",
    "ssp245": "UoM-MESSAGE-GLOBIOM-ssp245-1-2-1",
    "ssp370": "UoM-AIM-ssp370-1-2-1",
    "ssp585": "UoM-REMIND-MAGPIE-ssp585-1-2-1",
}
"""Mapping of SSP scenario IDs to input4MIPs source IDs."""


def find_file_url(scenario: str) -> str:
    """Look up the download URL for a scenario's monthly global-mean CO2 file.

    input4MIPs CO2 forcing files are only mirrored at esgf.ceda.ac.uk. Will only
    retrieve URLs from this host suffixed "HTTPServer" in the response.

    Parameters
    ----------
    scenario : str
        SSP scenario ID (e.g., "ssp245").

    Returns
    -------
    str
        Download URL for the scenario's CO2 forcing file hosted on ceda.ac.uk.

    Raises
    ------
    HTTPError
        If the ESGF search request fails.
    RuntimeError
        If no ceda.ac.uk-hosted download URL is found for the given scenario.

    """
    params = {
        "project": "input4MIPs",
        "source_id": SCENARIO_SOURCE_IDS[scenario],
        "variable_id": "mole_fraction_of_carbon_dioxide_in_air",
        "grid_label": "gr1-GMNHSH",
        "frequency": "mon",
        "type": "File",
        "format": "application/solr+json",
        "limit": 1,
    }
    response = requests.get(ESGF_SEARCH_URL, params=params, timeout=30)
    response.raise_for_status()
    docs = response.json()["response"]["docs"]
    urls = [u.split("|")[0] for u in docs[0]["url"] if u.endswith("HTTPServer")]

    try:
        return next(u for u in urls if "ceda.ac.uk" in u)
    except StopIteration as e:
        msg = (
            f"No ceda.ac.uk-hosted download URL found: scenario={scenario}  "
            f"candidates={urls}"
        )
        raise RuntimeError(msg) from e


def fetch_co2_series(
    scenario: str,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> pd.DataFrame:
    """Load a scenario's monthly global-mean CO2 concentration series within a range.

    The full file spans 2015-2500 using cftime dates, which overflow pandas'
    nanosecond-precision Timestamp; slice by date *before* converting to pandas so only
    the (small) requested window round-trips through `pd.to_datetime`.

    Parameters
    ----------
    scenario : str
        SSP scenario ID (e.g., "ssp245").
    start : pd.Timestamp
        Inclusive start of the date range.
    end : pd.Timestamp
        Inclusive end of the date range.

    Returns
    -------
    df : pd.DataFrame
        DataFrame with columns "time" and "co2_ppm" containing the CO2 series within the
        specified range.

    Raises
    ------
    HTTPError
        If the CO2 series file request fails.

    """
    url = find_file_url(scenario)
    with tempfile.TemporaryDirectory() as temp_dir:
        response = requests.get(url, timeout=60)
        response.raise_for_status()
        tmp = Path(temp_dir) / "co2_series.nc"
        tmp.write_bytes(response.content)
        ds = xr.open_dataset(tmp)
        window = ds["mole_fraction_of_carbon_dioxide_in_air"].sel(
            sector=0,
            time=slice(start.strftime("%Y-%m"), end.strftime("%Y-%m")),
        )
        df = window.to_dataframe()
        ds.close()

    df = df.reset_index()[["time", "mole_fraction_of_carbon_dioxide_in_air"]]
    df = df.rename(columns={"mole_fraction_of_carbon_dioxide_in_air": "co2_ppm"})
    df["time"] = pd.to_datetime(df["time"].astype(str))
    df["scenario"] = scenario
    return df


def main() -> None:
    """Fetch CO2 forcing data for a given SSP scenario and date range."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--start-date",
        type=pd.Timestamp,
        required=True,
        help=(
            "Inclusive start date for fetching CO2 forcing data in YYYY-MM-DD format. "
            "Dates part way through a month will fetch the entire month's data."
        ),
    )
    parser.add_argument(
        "--end-date",
        type=pd.Timestamp,
        required=True,
        help=(
            "Inclusive end date for fetching CO2 forcing data in YYYY-MM-DD format. "
            "Dates part way through a month will fetch the entire month's data."
        ),
    )
    parser.add_argument(
        "--scenario",
        required=True,
        choices=sorted(SCENARIO_SOURCE_IDS),
        help="SSP scenario ID for which to fetch CO2 forcing data.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="File to which fetched CO2 forcing CSV data will be written.",
    )
    args = parser.parse_args()

    logger.info("Fetching input4MIPs CO2 series: scenario=%s", args.scenario)
    co2 = fetch_co2_series(args.scenario, args.start_date, args.end_date)

    logger.info("Writing %d row(s): path=%s", len(co2), args.output)
    co2.to_csv(args.output, index=False)

    logger.info("Done.")


if __name__ == "__main__":
    main()
