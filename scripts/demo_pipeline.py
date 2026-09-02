"""Local demo of the full Aurora-CS pipeline: fetch real CO2 forcing and a real
HRES/ERA5-derived window, then run a short autoregressive training loop.

Not an Azure ML job -- a throwaway script to prove the pipeline mechanics end-to-end on
real data, run from the repo root, e.g.:

    python scripts/demo_pipeline.py --iters 5
"""

import argparse
import pickle
from dataclasses import replace
from datetime import datetime
from pathlib import Path

import gcsfs
import numpy as np
import pandas as pd
import requests
import torch
import torch.nn.functional as F
import xarray as xr
from huggingface_hub import hf_hub_download

from aurora import Batch, Metadata
from aurora_cs import AuroraCS

ZARR_URL = "gs://weatherbench2/datasets/hres_t0/2016-2022-6h-1440x721.zarr"
CO2_URL = "https://gml.noaa.gov/webdata/ccgg/trends/co2/co2_mm_mlo.txt"

SURF_VAR_MAP = {
    "2t": "2m_temperature",
    "10u": "10m_u_component_of_wind",
    "10v": "10m_v_component_of_wind",
    "msl": "mean_sea_level_pressure",
}
ATMOS_VAR_MAP = {
    "t": "temperature",
    "u": "u_component_of_wind",
    "v": "v_component_of_wind",
    "q": "specific_humidity",
    "z": "geopotential",
}


def fetch_co2_series(cache_path: Path) -> pd.DataFrame:
    """Download (and cache) NOAA GML's Mauna Loa monthly-mean CO2 record."""
    cache_path = cache_path.expanduser()
    if not cache_path.exists():
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(requests.get(CO2_URL, timeout=30).text)

    lines = [
        line
        for line in cache_path.read_text().splitlines()
        if line.strip() and not line.startswith("#")
    ]
    columns = [
        "year", "month", "decimal_date", "average", "deseasonalized", "ndays", "sdev", "unc",
    ]
    df = pd.DataFrame([line.split() for line in lines], columns=columns).astype(float)
    df["year"] = df["year"].astype(int)
    df["month"] = df["month"].astype(int)
    return df


def co2_ppm_for(co2: pd.DataFrame, when: datetime) -> float:
    """Look up the monthly-mean CO2 ppm for `when`, falling back to the nearest month."""
    exact = co2[(co2["year"] == when.year) & (co2["month"] == when.month)]
    if len(exact):
        return float(exact["average"].iloc[0])
    decimal_date = when.year + (when.month - 1) / 12
    idx = (co2["decimal_date"] - decimal_date).abs().idxmin()
    return float(co2.loc[idx, "average"])


def _prepare(x: np.ndarray) -> torch.Tensor:
    """Insert a batch dim and flip latitude, matching Aurora's expected orientation."""
    return torch.from_numpy(x[None][..., ::-1, :].copy())


def load_hres_window(
    start: datetime, num_history: int, num_future: int, cache_dir: Path, patch_size: int = 4,
) -> tuple[Batch, list[dict], list[datetime]]:
    """Download (and cache) `num_history + num_future` consecutive 6-hourly HRES/ERA5-derived
    steps starting at `start`, split into an initial history `Batch` plus a list of future
    ground-truth snapshots (one per future step) and the datetime of every step."""
    cache_dir = cache_dir.expanduser()
    cache_dir.mkdir(parents=True, exist_ok=True)

    n = num_history + num_future
    times = pd.date_range(start, periods=n, freq="6h")
    tag = f"{start:%Y%m%dT%H}_{n}"

    surf_path = cache_dir / f"{tag}-surface-level.nc"
    atmos_path = cache_dir / f"{tag}-atmospheric.nc"
    static_path = cache_dir / "static.nc"

    ds = None
    if not surf_path.exists() or not atmos_path.exists():
        fs = gcsfs.GCSFileSystem(token="anon")
        ds = xr.open_zarr(fs.get_mapper(ZARR_URL), chunks=None)

    if not surf_path.exists():
        ds[list(SURF_VAR_MAP.values())].sel(time=times).compute().to_netcdf(surf_path)
    if not atmos_path.exists():
        ds[list(ATMOS_VAR_MAP.values())].sel(time=times).compute().to_netcdf(atmos_path)
    if not static_path.exists():
        path = hf_hub_download(repo_id="microsoft/aurora", filename="aurora-0.25-static.pickle")
        with open(path, "rb") as f:
            static_vars = pickle.load(f)
        xr.Dataset(
            data_vars={k: (["latitude", "longitude"], v) for k, v in static_vars.items()},
            coords={
                "latitude": ("latitude", np.linspace(90, -90, 721)),
                "longitude": ("longitude", np.linspace(0, 360, 1440, endpoint=False)),
            },
        ).to_netcdf(static_path)

    surf_ds = xr.open_dataset(surf_path, engine="netcdf4")
    atmos_ds = xr.open_dataset(atmos_path, engine="netcdf4")
    static_ds = xr.open_dataset(static_path, engine="netcdf4")

    lat = torch.from_numpy(surf_ds.latitude.values[::-1].copy())
    lon = torch.from_numpy(surf_ds.longitude.values)
    atmos_levels = tuple(int(level) for level in atmos_ds.level.values)
    step_times = surf_ds.time.values.astype("datetime64[s]").tolist()

    # Build one Batch spanning the whole window (history + future stacked along the history
    # axis) and crop it once, so history and future end up on exactly the same cropped grid
    # that `Aurora.forward` will produce predictions on (it crops its input internally).
    full = Batch(
        surf_vars={
            short: _prepare(surf_ds[long].values) for short, long in SURF_VAR_MAP.items()
        },
        static_vars={k: torch.from_numpy(static_ds[k].values) for k in ("z", "slt", "lsm")},
        atmos_vars={
            short: _prepare(atmos_ds[long].values) for short, long in ATMOS_VAR_MAP.items()
        },
        metadata=Metadata(
            lat=lat,
            lon=lon,
            time=(step_times[num_history - 1],),
            atmos_levels=atmos_levels,
        ),
    ).crop(patch_size)

    history = replace(
        full,
        surf_vars={k: v[:, :num_history] for k, v in full.surf_vars.items()},
        atmos_vars={k: v[:, :num_history] for k, v in full.atmos_vars.items()},
    )
    future = [
        {
            "surf_vars": {k: v[:, i : i + 1] for k, v in full.surf_vars.items()},
            "atmos_vars": {k: v[:, i : i + 1] for k, v in full.atmos_vars.items()},
        }
        for i in range(num_history, n)
    ]

    return history, future, step_times


def advance(batch: Batch, pred: Batch) -> Batch:
    """Slide the history window: drop the oldest step, append the new prediction.

    Local reimplementation of `aurora.rollout._advance_batch`'s logic (kept local rather
    than importing a private, unexported symbol).
    """
    new_surf = {
        k: torch.cat([batch.surf_vars[k][:, 1:], v], dim=1)
        for k, v in pred.surf_vars.items()
        if k in batch.surf_vars
    }
    new_atmos = {
        k: torch.cat([batch.atmos_vars[k][:, 1:], v], dim=1)
        for k, v in pred.atmos_vars.items()
        if k in batch.atmos_vars
    }
    return replace(pred, surf_vars=new_surf, atmos_vars=new_atmos)


def rollout_loss(pred: Batch, target: dict) -> torch.Tensor:
    """Mean MSE across all surf/atmos variables between a prediction and a ground-truth
    snapshot (as produced by `load_hres_window`'s `future` list)."""
    losses = [
        F.mse_loss(pred.surf_vars[k], v.to(pred.surf_vars[k]))
        for k, v in target["surf_vars"].items()
    ] + [
        F.mse_loss(pred.atmos_vars[k], v.to(pred.atmos_vars[k]))
        for k, v in target["atmos_vars"].items()
    ]
    return torch.stack(losses).mean()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start-date", default="2021-06-01")
    parser.add_argument("--steps", type=int, default=2, help="Rollout length used for the loss.")
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--cache-dir", default="~/.cache/aurora-cs-demo")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    device = torch.device(args.device)
    print(f"Using device: {device}")
    cache_dir = Path(args.cache_dir)
    start = pd.Timestamp(args.start_date).to_pydatetime()
    num_history = 2  # Fixed: what the small pretrained checkpoint was trained with.

    print("Fetching CO2 series...")
    co2 = fetch_co2_series(cache_dir / "co2_mm_mlo.csv")

    print(f"Loading HRES window starting {start} ({num_history} history + {args.steps} future)...")
    init_batch, future, times = load_hres_window(start, num_history, args.steps, cache_dir)

    print("Building AuroraCS (AuroraSmallPretrained config) and loading its checkpoint...")
    model = AuroraCS(
        encoder_depths=(2, 6, 2),
        encoder_num_heads=(4, 8, 16),
        decoder_depths=(2, 6, 2),
        decoder_num_heads=(16, 8, 4),
        embed_dim=256,
        num_heads=8,
        use_lora=False,
    ).to(device)
    model.aurora.load_checkpoint(name="aurora-0.25-small-pretrained.ckpt")

    def rollout_once() -> torch.Tensor:
        batch = init_batch
        total = torch.zeros((), device=device)
        for step in range(args.steps):
            forcing = co2_ppm_for(co2, times[num_history + step])
            pred = model(batch, forcing=forcing)
            total = total + rollout_loss(pred, future[step])
            batch = advance(batch, pred)
        return total

    print("Baseline check: adapters are zero-init, so AuroraCS output must match raw Aurora.")
    with torch.no_grad():
        adapted_loss = rollout_once()

        batch = init_batch
        baseline_loss = torch.zeros((), device=device)
        for step in range(args.steps):
            pred = model.aurora(batch)
            baseline_loss = baseline_loss + rollout_loss(pred, future[step])
            batch = advance(batch, pred)

    print(f"  adapted (pre-training) loss = {adapted_loss.item():.6f}")
    print(f"  raw-Aurora loss            = {baseline_loss.item():.6f}")

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(trainable_params, lr=args.lr)

    print(f"Training for {args.iters} iterations...")
    for it in range(args.iters):
        optimizer.zero_grad()
        loss = rollout_once()
        loss.backward()
        optimizer.step()
        print(f"  iter {it}: loss = {loss.item():.6f}")


if __name__ == "__main__":
    main()
