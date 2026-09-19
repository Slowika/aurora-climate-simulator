"""CLI producer: fetch input4MIPs CO2 concentration data (the CMIP6 GHG forcing dataset, served
via ESGF) for a given SSP scenario and register the requested date range as an Azure ML data
asset, e.g.:

    python -m data.fetch_co2_forcing --start-date 2020-01-01 --end-date 2020-06-01 \\
        --scenario ssp245 --output-path ~/data/co2/co2.csv --asset-name co2-forcing \\
        --subscription-id ... --resource-group ... --workspace-name ...

Coverage starts 2015-01 (ScenarioMIP data does not extend into the historical period).
"""

import argparse
import logging
import sys
import tempfile
from pathlib import Path

import pandas as pd
import requests
import xarray as xr
from azure.ai.ml.constants import AssetTypes

from data.azure_ml import get_ml_client, register_data_asset

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
logger = logging.getLogger(__name__)

ESGF_SEARCH_URL = "https://esgf-data.dkrz.de/esg-search/search/"

SCENARIO_SOURCE_IDS = {
    "ssp126": "UoM-IMAGE-ssp126-1-2-1",
    "ssp245": "UoM-MESSAGE-GLOBIOM-ssp245-1-2-1",
    "ssp370": "UoM-AIM-ssp370-1-2-1",
    "ssp585": "UoM-REMIND-MAGPIE-ssp585-1-2-1",
}


def find_file_url(scenario: str) -> str:
    """Look up the HTTPServer download URL for a scenario's monthly global-mean CO2 file."""
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
    docs = requests.get(ESGF_SEARCH_URL, params=params, timeout=30).json()["response"]["docs"]
    return next(u.split("|")[0] for u in docs[0]["url"] if u.endswith("HTTPServer"))


def fetch_co2_series(scenario: str, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    """Download a scenario's global-mean monthly CO2 concentration series for [start, end].

    The full file spans 2015-2500 using cftime dates, which overflow pandas' nanosecond-precision
    Timestamp; slice by date *before* converting to pandas so only the (small) requested window
    round-trips through `pd.to_datetime`.
    """
    url = find_file_url(scenario)
    with tempfile.NamedTemporaryFile(suffix=".nc") as tmp:
        tmp.write(requests.get(url, timeout=60).content)
        tmp.flush()
        ds = xr.open_dataset(tmp.name)
        window = ds["mole_fraction_of_carbon_dioxide_in_air"].sel(
            sector=0, time=slice(start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d"))
        )
        df = window.to_dataframe()
    df = df.reset_index()[["time", "mole_fraction_of_carbon_dioxide_in_air"]]
    df = df.rename(columns={"mole_fraction_of_carbon_dioxide_in_air": "co2_ppm"})
    df["time"] = pd.to_datetime(df["time"].astype(str))
    return df


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start-date", required=True)
    parser.add_argument("--end-date", required=True)
    parser.add_argument("--scenario", required=True, choices=sorted(SCENARIO_SOURCE_IDS))
    parser.add_argument("--output-path", required=True)
    parser.add_argument("--asset-name", required=True)
    parser.add_argument("--subscription-id", required=True)
    parser.add_argument("--resource-group", required=True)
    parser.add_argument("--workspace-name", required=True)
    args = parser.parse_args()

    try:
        start = pd.Timestamp(args.start_date)
        end = pd.Timestamp(args.end_date)

        logger.info(f"Fetching input4MIPs CO2 series for scenario '{args.scenario}'...")
        co2 = fetch_co2_series(args.scenario, start, end)

        output_path = Path(args.output_path).expanduser()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        co2.to_csv(output_path, index=False)
        logger.info(f"Wrote {len(co2)} row(s) to {output_path}")

        logger.info(f"Registering '{args.asset_name}' from {output_path}...")
        ml_client = get_ml_client(args.subscription_id, args.resource_group, args.workspace_name)
        register_data_asset(
            ml_client,
            name=args.asset_name,
            path=str(output_path),
            asset_type=AssetTypes.URI_FILE,
            description=f"input4MIPs CO2 ({args.scenario}), {args.start_date} to {args.end_date}",
        )
        logger.info("Done.")
    except Exception as e:
        print(e)
        sys.exit(1)


if __name__ == "__main__":
    main()
