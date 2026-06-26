"""FastAPI backend: gọi pipeline tối ưu lịch trình MTA cho giao diện web (landingpage3d).

Chạy:  uvicorn demo.api.server:app --reload --port 8000
"""

from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from demo.config import (
    CAPACITY_PER_TRIP,
    COST_PER_VEHICLE_HOUR,
    LAMBDA_KNEE_FALLBACK,
    MAX_HEADWAY_MIN,
    MIN_HEADWAY_MIN,
    OPT_ROUTES,
    OVERNIGHT_MAX_HEADWAY_MIN,
    SMOOTHNESS_MAX_DELTA,
    TRIPS_DAYTIME_MAX_FACTOR,
    TRIPS_MIN_FACTOR,
    TRIPS_OVERNIGHT_MAX_FACTOR,
    TRIPS_OVERNIGHT_MIN_FACTOR,
    DEFAULT_MAX_DAY_AT_MAX_RATIO,
    DEFAULT_MAX_DAY_AT_MIN_RATIO,
    DEFAULT_MAX_NIGHT_AT_MAX_RATIO,
    DEFAULT_MAX_NIGHT_AT_MIN_RATIO,
)
from demo.api.serialize import serialize_results
from demo.core.optimizer import RunConfig, load_models_and_data, route_quick_info, run_optimization
from demo.core.scenario_builder import (
    WEATHER_PRESETS,
    WEEKDAY_REPRESENTATIVE_DOW,
    WEEKEND_REPRESENTATIVE_DOW,
    ScenarioConfig,
    build_scenario_hourly_factors,
)
from demo.core.weather_fetcher import fetch_openmeteo_for_date, weather_summary_line

# Mùa -> tháng đại diện (NYC) + nhãn tiếng Việt cho UI.
SEASONS: dict[str, dict[str, Any]] = {
    "spring": {"label": "🌸 Xuân", "month": 4},
    "summer": {"label": "☀️ Hè", "month": 7},
    "autumn": {"label": "🍂 Thu", "month": 10},
    "winter": {"label": "❄️ Đông", "month": 1},
}

app = FastAPI(title="MTA Schedule Optimizer API", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

_ASSETS: dict[str, Any] | None = None


def get_assets() -> dict[str, Any]:
    global _ASSETS
    if _ASSETS is None:
        _ASSETS = load_models_and_data()
    return _ASSETS


class OptimizeRequest(BaseModel):
    route_id: str = "1"
    mode: str = Field("scenario", description="'scenario' | 'date'")
    date: str | None = None

    day_type: str = Field("weekday", description="'weekday' | 'weekend'")
    season: str = "summer"
    month: int | None = None
    weather_preset: str | None = None
    is_holiday: bool = False
    is_event: bool = False

    w_low: float = 0.10
    w_high: float = 0.90

    min_headway_min: float = MIN_HEADWAY_MIN
    max_headway_min: float = MAX_HEADWAY_MIN
    overnight_max_headway_min: float = OVERNIGHT_MAX_HEADWAY_MIN
    capacity_per_trip: float = CAPACITY_PER_TRIP
    max_overflow_pct: float = 1.0
    smoothness_delta: int = SMOOTHNESS_MAX_DELTA
    route_fleet_limit: float = 0.0
    cost_per_vehicle_hour: float = COST_PER_VEHICLE_HOUR

    trips_daytime_max_factor: float = TRIPS_DAYTIME_MAX_FACTOR
    trips_min_factor: float = TRIPS_MIN_FACTOR
    trips_overnight_max_factor: float = TRIPS_OVERNIGHT_MAX_FACTOR
    trips_overnight_min_factor: float = TRIPS_OVERNIGHT_MIN_FACTOR
    max_over_ceiling_pct: float = 50.0

    use_capacity_constraint: bool = True
    use_smoothness_constraint: bool = True
    use_route_fleet_cap: bool = True


@app.get("/api/meta")
def meta() -> dict[str, Any]:
    return {
        "routes": OPT_ROUTES,
        "seasons": [{"id": k, "label": v["label"], "month": v["month"]} for k, v in SEASONS.items()],
        "weather_presets": list(WEATHER_PRESETS.keys()),
        "defaults": {
            "min_headway_min": MIN_HEADWAY_MIN,
            "max_headway_min": MAX_HEADWAY_MIN,
            "overnight_max_headway_min": OVERNIGHT_MAX_HEADWAY_MIN,
            "capacity_per_trip": CAPACITY_PER_TRIP,
            "max_overflow_pct": 1.0,
            "smoothness_delta": SMOOTHNESS_MAX_DELTA,
            "route_fleet_limit": 0.0,
            "cost_per_vehicle_hour": COST_PER_VEHICLE_HOUR,
            "trips_daytime_max_factor": TRIPS_DAYTIME_MAX_FACTOR,
            "trips_min_factor": TRIPS_MIN_FACTOR,
            "trips_overnight_max_factor": TRIPS_OVERNIGHT_MAX_FACTOR,
            "trips_overnight_min_factor": TRIPS_OVERNIGHT_MIN_FACTOR,
            "max_over_ceiling_pct": 50.0,
            "w_low": 0.10,
            "w_high": 0.90,
        },
    }


@app.get("/api/weather")
def weather(date: str) -> dict[str, Any]:
    try:
        target = date_from_iso(date)
        df = fetch_openmeteo_for_date(target)
        return {"summary": weather_summary_line(df)}
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=400, detail=f"Không lấy được thời tiết: {exc}") from exc


def date_from_iso(value: str) -> date:
    parts = [int(p) for p in value.split("-")]
    return date(parts[0], parts[1], parts[2])


def _build_hourly_factors(req: OptimizeRequest):
    if req.mode == "date":
        if not req.date:
            raise HTTPException(status_code=400, detail="Thiếu 'date' cho chế độ theo ngày thực tế.")
        return fetch_openmeteo_for_date(date_from_iso(req.date))

    is_weekend = 1 if req.day_type == "weekend" else 0
    dow = WEEKEND_REPRESENTATIVE_DOW if is_weekend else WEEKDAY_REPRESENTATIVE_DOW
    month = int(req.month) if req.month else int(SEASONS.get(req.season, SEASONS["summer"])["month"])
    preset_key = req.weather_preset if req.weather_preset in WEATHER_PRESETS else next(iter(WEATHER_PRESETS))
    weather = dict(WEATHER_PRESETS[preset_key])
    scenario = ScenarioConfig(
        is_weekend=is_weekend,
        day_of_week=int(dow),
        month=month,
        is_us_holiday=1 if req.is_holiday else 0,
        is_major_event_window=1 if req.is_event else 0,
        weather=weather,
    )
    return build_scenario_hourly_factors(scenario)


@app.post("/api/optimize")
def optimize(req: OptimizeRequest) -> dict[str, Any]:
    if req.route_id not in OPT_ROUTES:
        raise HTTPException(status_code=400, detail=f"Tuyến không hợp lệ: {req.route_id}")
    assets = get_assets()
    hourly_factors = _build_hourly_factors(req)
    lambda_fallback = int(assets["state"].get("lambda_knee_per_route", {}).get(req.route_id, LAMBDA_KNEE_FALLBACK))

    cfg = RunConfig(
        route_id=req.route_id,
        lambda_val=lambda_fallback,
        input_mode=req.mode,
        hourly_factors=hourly_factors,
        capacity_per_trip=float(req.capacity_per_trip),
        max_overflow_pct=float(req.max_overflow_pct),
        cost_per_vehicle_hour=float(req.cost_per_vehicle_hour),
        route_fleet_limit=float(req.route_fleet_limit),
        max_day_at_min_ratio=float(DEFAULT_MAX_DAY_AT_MIN_RATIO),
        max_day_at_max_ratio=float(DEFAULT_MAX_DAY_AT_MAX_RATIO),
        max_night_at_min_ratio=float(DEFAULT_MAX_NIGHT_AT_MIN_RATIO),
        max_night_at_max_ratio=float(DEFAULT_MAX_NIGHT_AT_MAX_RATIO),
        smoothness_delta=int(req.smoothness_delta),
        min_headway_min=float(req.min_headway_min),
        max_headway_min=float(req.max_headway_min),
        overnight_max_headway_min=float(req.overnight_max_headway_min),
        use_route_fleet_cap=bool(req.use_route_fleet_cap),
        use_system_fleet_cap=False,
        use_capacity_constraint=bool(req.use_capacity_constraint),
        use_smoothness_constraint=bool(req.use_smoothness_constraint),
        auto_knee=True,
        w_low=float(req.w_low),
        w_high=float(req.w_high),
        trips_min_factor=float(req.trips_min_factor),
        trips_overnight_min_factor=float(req.trips_overnight_min_factor),
        trips_daytime_max_factor=float(req.trips_daytime_max_factor),
        trips_overnight_max_factor=float(req.trips_overnight_max_factor),
        max_over_ceiling_pct=float(req.max_over_ceiling_pct),
        system_fleet_override=0,
    )

    try:
        results = run_optimization(cfg, assets)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=422, detail=f"Không tìm được lịch tối ưu: {exc}") from exc

    info = route_quick_info(req.route_id, assets)
    payload = serialize_results(results, threshold_pct=float(req.max_over_ceiling_pct))
    payload["n_directions"] = info["n_directions"]
    payload["active_hours"] = len(info["hours"])
    payload["predictor_ok"] = assets.get("predictor") is not None
    return payload


# Phục vụ bản build của landingpage3d (nếu đã `npm run build`) trên cùng origin (tùy chọn).
_DIST = Path(__file__).resolve().parents[2] / "landingpage3d" / "dist"
if _DIST.is_dir():
    app.mount("/", StaticFiles(directory=str(_DIST), html=True), name="site")
