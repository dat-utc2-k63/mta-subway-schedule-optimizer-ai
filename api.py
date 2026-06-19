"""MTA Schedule API + optimizer UI — run: uvicorn api:app --reload --port 8000"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any

import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

ROOT = Path(__file__).resolve().parent
NOTEBOOKS = ROOT / "notebooks"
sys.path.insert(0, str(NOTEBOOKS))

from lib.ui_constraints import ValidationError  # noqa: E402
from lib.ui_service import (  # noqa: E402
    OptimizeRequest,
    get_meta,
    get_tradeoff,
    load_artifacts,
    run_optimization,
)
from lib.ui_station_schedule import schedule_by_station  # noqa: E402

UI_DIR = NOTEBOOKS / "outputs" / "default" / "ui_export"
SCHEDULE_DIR = ROOT / "datasets" / "schedule_current"
RIDERSHIP = ROOT / "datasets" / "ridership.csv"
OPTIMIZER_UI = ROOT / "optimizer-ui"
LANDING_DIST = ROOT / "landingpage3d" / "dist"

app = FastAPI(title="MTA Schedule API", version="2.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class OptimizeBody(BaseModel):
    route_id: str
    tradeoff_preset: str = "balanced"
    lambda_cost: float | None = None
    input_mode: str = "date"
    selected_date: str | None = None
    weekday_weekend: str = "weekday"
    season: str = "summer"
    weather_group: str = "sunny"
    filter_holiday: bool = False
    holiday_name: str | None = None
    filter_major_event: bool = False
    use_route_fleet: bool = True
    manual_weather: dict[str, float | int] | None = None
    constraints: dict[str, Any] | None = None


def _load_schedule_json(scenario: str) -> pd.DataFrame:
    path = UI_DIR / f"schedule_{scenario}.json"
    if not path.exists():
        raise HTTPException(404, f"Không có schedule cho scenario={scenario}")
    with path.open(encoding="utf-8") as f:
        rows = json.load(f)
    return pd.DataFrame(rows)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/schedule/by-station")
def get_schedule_by_station(
    scenario: str = Query("weekday_peak", description="weekday_peak | weekend | rainy_day"),
    route: str | None = Query(None, description="Lọc theo route_id"),
) -> list[dict]:
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


@app.get("/api/optimizer/meta")
def optimizer_meta() -> dict[str, Any]:
    try:
        return get_meta()
    except FileNotFoundError as exc:
        raise HTTPException(503, str(exc)) from exc
    except Exception as exc:
        raise HTTPException(500, str(exc)) from exc


@app.get("/api/optimizer/tradeoff")
def optimizer_tradeoff(
    route_id: str = Query(...),
    preset: str = Query("balanced"),
) -> dict[str, Any]:
    try:
        return get_tradeoff(route_id, preset)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc


@app.post("/api/optimizer/run")
def optimizer_run(body: OptimizeBody) -> dict[str, Any]:
    if body.input_mode not in ("date", "scenario"):
        raise HTTPException(400, "input_mode must be date or scenario")
    req = OptimizeRequest(
        route_id=body.route_id,
        tradeoff_preset=body.tradeoff_preset,
        lambda_cost=body.lambda_cost,
        input_mode=body.input_mode,  # type: ignore[arg-type]
        selected_date=body.selected_date,
        weekday_weekend=body.weekday_weekend,
        season=body.season,
        weather_group=body.weather_group,
        filter_holiday=body.filter_holiday,
        holiday_name=body.holiday_name,
        filter_major_event=body.filter_major_event,
        use_route_fleet=body.use_route_fleet,
        manual_weather=body.manual_weather,
        constraints=body.constraints,
    )
    try:
        return run_optimization(req)
    except ValidationError as exc:
        raise HTTPException(422, str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    except FileNotFoundError as exc:
        raise HTTPException(503, str(exc)) from exc
    except Exception as exc:
        raise HTTPException(500, str(exc)) from exc


@app.post("/api/optimizer/reload")
def optimizer_reload() -> dict[str, str]:
    try:
        load_artifacts(reload=True)
        return {"status": "reloaded"}
    except Exception as exc:
        raise HTTPException(500, str(exc)) from exc


@app.get("/optimizer")
@app.get("/optimizer/")
def optimizer_page() -> FileResponse:
    index = OPTIMIZER_UI / "index.html"
    if not index.exists():
        raise HTTPException(404, "optimizer-ui not found — build or copy optimizer-ui/")
    return FileResponse(index)


if OPTIMIZER_UI.exists():
    app.mount(
        "/optimizer/static",
        StaticFiles(directory=OPTIMIZER_UI),
        name="optimizer-static",
    )


def _mount_landing() -> None:
    if not LANDING_DIST.exists():
        return
    assets = LANDING_DIST / "assets"
    if assets.is_dir():
        app.mount("/assets", StaticFiles(directory=assets), name="landing-assets")

    @app.get("/")
    def landing_index() -> FileResponse:
        return FileResponse(LANDING_DIST / "index.html")


_mount_landing()
