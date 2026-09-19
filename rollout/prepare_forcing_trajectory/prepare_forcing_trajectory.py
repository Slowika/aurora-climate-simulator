"""Pipeline component: build the CO2 forcing trajectory for a rollout.

Looks up the input4MIPs-derived monthly CO2 concentration for each 6-hourly step of the
rollout (repeating the month's value across its steps, no interpolation), and saves the
result as a 1-D tensor.
"""

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd
import torch

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
logger = logging.getLogger(__name__)


def co2_ppm_for(co2: pd.DataFrame, when: pd.Timestamp) -> float:
    """Look up the monthly-mean CO2 ppm for `when`, falling back to the nearest month."""
    exact = co2[(co2["time"].dt.year == when.year) & (co2["time"].dt.month == when.month)]
    if len(exact):
        return float(exact["co2_ppm"].iloc[0])
    idx = (co2["time"] - when).abs().idxmin()
    return float(co2.loc[idx, "co2_ppm"])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--co2-data", required=True)
    parser.add_argument("--start-time", required=True)
    parser.add_argument("--num-steps", required=True, type=int)
    parser.add_argument("--forcing-trajectory", required=True)
    args = parser.parse_args()

    try:
        logger.info(
            f"Preparing {args.num_steps}-step forcing trajectory from {args.start_time} "
            f"using {args.co2_data}..."
        )
        co2 = pd.read_csv(args.co2_data, parse_dates=["time"])
        start_time = pd.Timestamp(args.start_time)
        step_times = [start_time + (i + 1) * pd.Timedelta(hours=6) for i in range(args.num_steps)]
        forcing = torch.tensor([co2_ppm_for(co2, when) for when in step_times], dtype=torch.float32)

        out_path = Path(args.forcing_trajectory)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(forcing, out_path)
        logger.info(f"Wrote {len(forcing)}-step forcing trajectory -> {out_path}")
    except Exception as e:
        print(e)
        sys.exit(1)


if __name__ == "__main__":
    main()
