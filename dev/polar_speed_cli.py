#!/usr/bin/env -S uv run
# /// script
# requires-python = ">=3.9"
# dependencies = [
#   "numpy",
#   "pandas",
# ]
# ///
"""CLI helper that reports polar-derived boat speed from local source."""

from __future__ import annotations

import argparse
from pathlib import Path

import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Print polar boat speed for given TWS/TWA.")
    parser.add_argument("--tws", type=float, required=True, help="True wind speed in knots.")
    parser.add_argument("--twa", type=float, required=True, help="True wind angle in degrees.")
    parser.add_argument(
        "--polar",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "dev" / "polars" / "Small_polar.pol",
        help="Path to the polar file to use.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    from weather_router.polar import Polar

    polar = Polar(polarPath=str(args.polar))
    speed = polar.getSpeed(args.tws, args.twa)
    print(f"{speed:.6f}")


if __name__ == "__main__":
    main()
