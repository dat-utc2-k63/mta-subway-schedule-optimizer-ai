"""MTA Schedule API + optimizer UI — run: uvicorn api:app --reload --port 8000"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any, Literal

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
    OptimizeV1Request,
    ScenarioV1,
    get_date_profile,
    get_meta_v1,
    run_optimization_v1,
)
from lib.ui_station_schedule import schedule_by_station  # noqa: E402

UI_DIR = NOTEBOOKS / "outputs" / "default" / "ui_export"
SCHEDULE_DIR = ROOT / "datasets" / "schedule_current"
RIDERSHIP = ROOT / "datasets" / "ridership.csv"
OPTIMIZER_UI = ROOT / "optimizer-ui"
LANDING_DIST = ROOT / "landingpage3d" / "dist"

app = FastAPI(title="MTA Schedule API", version="3.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class ScenarioBody(BaseModel):
    weekday_weekend: str = "weekday"
    season: str = "summer"
    weather_group: str = "sunny"
    filter_holiday: bool = False
    holiday_name: str | None = None
    filter_major_event: bool = False


class OptimizeV1Body(BaseModel):
    route_id: str
    preset: str = "balanced"
    mode: Literal["date", "scenario"] = "date"
    date: str | None = None
    scenario: ScenarioBody | None = None
    use_overrides: bool = False
    overrides: ScenarioBody | None = None


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


@app.get("/api/v1/meta")
def optimizer_meta_v1() -> dict[str, Any]:
    try:
        return get_meta_v1()
    except FileNotFoundError as exc:
        raise HTTPException(503, str(exc)) from exc
    except Exception as exc:
        raise HTTPException(500, str(exc)) from exc


@app.get("/api/v1/date-profile")
def optimizer_date_profile(
    date: str = Query(..., description="ISO date YYYY-MM-DD"),
) -> dict[str, Any]:
    try:
        return get_date_profile(date)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    except FileNotFoundError as exc:
        raise HTTPException(503, str(exc)) from exc
    except Exception as exc:
        raise HTTPException(500, str(exc)) from exc


@app.post("/api/v1/optimize")
def optimizer_run_v1(body: OptimizeV1Body) -> dict[str, Any]:
    if body.mode == "date" and not body.date:
        raise HTTPException(400, "date required when mode is date")
    if body.mode == "scenario" and body.scenario is None:
        body = body.model_copy(update={"scenario": ScenarioBody()})

    sc = body.scenario
    ov = body.overrides
    req = OptimizeV1Request(
        route_id=body.route_id,
        preset=body.preset,
        mode=body.mode,
        date=body.date,
        use_overrides=body.use_overrides,
        overrides=ScenarioV1(
            weekday_weekend=ov.weekday_weekend,
            season=ov.season,
            weather_group=ov.weather_group,
            filter_holiday=ov.filter_holiday,
            holiday_name=ov.holiday_name,
            filter_major_event=ov.filter_major_event,
        )
        if ov
        else None,
        scenario=ScenarioV1(
            weekday_weekend=sc.weekday_weekend,
            season=sc.season,
            weather_group=sc.weather_group,
            filter_holiday=sc.filter_holiday,
            holiday_name=sc.holiday_name,
            filter_major_event=sc.filter_major_event,
        )
        if sc
        else None,
    )
    try:
        return run_optimization_v1(req)
    except ValidationError as exc:
        raise HTTPException(422, str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    except FileNotFoundError as exc:
        raise HTTPException(503, str(exc)) from exc
    except Exception as exc:
        raise HTTPException(500, str(exc)) from exc


@app.get("/optimizer")
@app.get("/optimizer/")
def optimizer_page() -> FileResponse:
    index = OPTIMIZER_UI / "index.html"
    if not index.exists():
        raise HTTPException(404, "optimizer-ui not found")
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
