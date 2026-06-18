"""Lightweight API for station schedule (optional — run: uvicorn api:app)."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pandas as pd
from fastapi import FastAPI, HTTPException, Query

ROOT = Path(__file__).resolve().parent
NOTEBOOKS = ROOT / "notebooks"
sys.path.insert(0, str(NOTEBOOKS))

from lib.ui_station_schedule import schedule_by_station  # noqa: E402

UI_DIR = NOTEBOOKS / "outputs" / "default" / "ui_export"
SCHEDULE_DIR = ROOT / "datasets" / "schedule_current"
RIDERSHIP = ROOT / "datasets" / "ridership.csv"

app = FastAPI(title="MTA Schedule API", version="1.0")


def _load_schedule_json(scenario: str) -> pd.DataFrame:
    path = UI_DIR / f"schedule_{scenario}.json"
    if not path.exists():
        raise HTTPException(404, f"Không có schedule cho scenario={scenario}")
    with path.open(encoding="utf-8") as f:
        rows = json.load(f)
    return pd.DataFrame(rows)


@app.get("/schedule/by-station")
def get_schedule_by_station(
    scenario: str = Query("weekday_peak", description="weekday_peak | weekend | rainy_day"),
    route: str | None = Query(None, description="Lọc theo route_id"),
) -> list[dict]:
    """Giờ đến từng ga, có borough — đọc từ schedule export + GTFS offsets."""
    try:
        sched = _load_schedule_json(scenario)
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(500, str(exc)) from exc

    rid_m = os.path.getmtime(RIDERSHIP) if RIDERSHIP.exists() else 0.0
    sd_m = os.path.getmtime(SCHEDULE_DIR) if SCHEDULE_DIR.exists() else 0.0
    return schedule_by_station(
        sched,
        schedule_dir=str(SCHEDULE_DIR),
        ridership_path=str(RIDERSHIP),
        route_id=route,
        schedule_dir_mtime=sd_m,
        ridership_mtime=rid_m,
    )


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}
