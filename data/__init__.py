"""Data preparation scripts and helpers for Aurora Climate Simulator.

Modules
-------
azure_ml
    Shared Azure ML plumbing used by some data-producer scripts.
fetch_co2_forcing
    CLI producer: fetch input4MIPs CO2 concentration data (the CMIP6 GHG forcing
    dataset, served via ESGF) for a given SSP scenario and register the requested date
    range as an Azure ML data asset.
fetch_era5
    CLI producer: fetch ERA5 data from the Google Cloud Platform analysis-ready dataset.
"""
