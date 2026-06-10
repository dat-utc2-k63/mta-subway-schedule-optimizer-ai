#!/usr/bin/env python3
"""Crawl Open-Meteo (NYC) + calendar → factors_hourly.csv & factors_daily.csv."""

from __future__ import annotations

import argparse
import shutil
import sys
from datetime import date, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "notebooks"))

from lib.factor_builder import NYC_LAT, NYC_LON, build_factors  # noqa: E402

DATA_DIR = ROOT / "datasets"
DEFAULT_START = date(2023, 12, 1)
DEFAULT_END = date(2025, 11, 30)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Rebuild factors CSVs from Open-Meteo (NYC) + NY calendar rules."
    )
    parser.add_argument("--start", default=DEFAULT_START.isoformat(), help="YYYY-MM-DD")
    parser.add_argument("--end", default=DEFAULT_END.isoformat(), help="YYYY-MM-DD")
    parser.add_argument(
        "--to-today",
        action="store_true",
        help="Set end date to today (extend beyond ridership range).",
    )
    parser.add_argument("--no-backup", action="store_true", help="Skip .bak backup of existing CSVs.")
    args = parser.parse_args()

    start = date.fromisoformat(args.start)
    end = date.today() if args.to_today else date.fromisoformat(args.end)
    if end < start:
        raise SystemExit("end must be >= start")

    hourly_path = DATA_DIR / "factors_hourly.csv"
    daily_path = DATA_DIR / "factors_daily.csv"

    print(f"Location: NYC ({NYC_LAT}, {NYC_LON})")
    print(f"Range: {start} -> {end}")
    print("Fetching Open-Meteo + building calendar features...")

    hourly, daily = build_factors(start, end)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    if not args.no_backup:
        for p in (hourly_path, daily_path):
            if p.exists():
                bak = p.with_suffix(p.suffix + f".bak_{ts}")
                shutil.copy2(p, bak)
                print(f"Backup: {bak.name}")

    hourly.to_csv(hourly_path, index=False)
    daily.to_csv(daily_path, index=False)

    print(f"Wrote {hourly_path.name}: {len(hourly):,} rows")
    print(f"Wrote {daily_path.name}: {len(daily):,} rows")
    print("Done.")


if __name__ == "__main__":
    main()
