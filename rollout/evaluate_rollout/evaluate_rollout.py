"""Pipeline component: evaluate a rollout against ground truth (and optionally a baseline).

Compares AuroraCS predictions to the corresponding ERA5 ground truth, and, if provided, to a
raw-Aurora baseline rollout, to quantify what the CO2 conditioning buys over the unconditioned
model.
"""

import argparse
import logging
import sys

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--era5-data", required=True)
    parser.add_argument("--baseline-predictions", required=False, default=None)
    parser.add_argument("--metrics", required=True)
    args = parser.parse_args()

    try:
        logger.info(f"Evaluating {args.predictions} against {args.era5_data}...")
        raise NotImplementedError(
            "TODO: compare predictions (and baseline_predictions, if given) against era5_data "
            "and write metrics to metrics."
        )
    except Exception as e:
        print(e)
        sys.exit(1)


if __name__ == "__main__":
    main()
