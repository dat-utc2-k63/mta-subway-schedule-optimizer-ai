#!/usr/bin/env python3
"""Build weather_groups.json from training factors_hourly.csv."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "notebooks"))

from lib.ui_weather_groups import dump_weather_groups  # noqa: E402

FACTORS = ROOT / "datasets" / "factors_hourly.csv"
OUT = ROOT / "notebooks" / "outputs" / "default" / "ui_export" / "weather_groups.json"


def main() -> None:
    path = dump_weather_groups(OUT, FACTORS)
    print(f"Wrote {path}")


if __name__ == "__main__":
    main()
