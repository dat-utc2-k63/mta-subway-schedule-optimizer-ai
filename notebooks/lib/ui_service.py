"""FastAPI backend for schedule optimizer UI."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal

import joblib
import numpy as np
import pandas as pd

from .demand_runtime import DemandPredictor
from .ui_constraints import (
    ConstraintOverrides,
    ValidationError,
    default_constraint_panel,
    merge_overrides,
    validate_constraint_config,
)
from .ui_factors import load_feature_medians, prepare_hourly_factors_for_model
from .ui_lambda import (
    TRADEOFF_PRESET_ORDER,
    TRADEOFF_PRESETS,
    lambda_for_preset,
    route_lambda_knee,
    tradeoff_compact_label,
)
from .ui_nearest import resolve_to_nearest_training_day
from .ui_optimizer import optimize_route_day, resolve_lambda_ref
from .ui_scenario import (
    SEASONS,
    SEASON_VI,
    WEEKDAY_WEEKEND_OPTIONS,
    WEEKDAY_WEEKEND_VI,
    ScenarioSelection,
    build_scenario_hourly_factors,
    list_holiday_names,
)
from .ui_weather import PICKER_FUTURE_DAYS, build_hourly_factors
from .ui_weather_groups import (
    WEATHER_GROUP_KEYS,
    WEATHER_GROUP_VI,
    apply_weather_group_to_hourly,
    dump_weather_groups,
    load_weather_groups,
)

ROOT = Path(__file__).resolve().parents[2]
UI_DIR = ROOT / "notebooks" / "outputs" / "default" / "ui_export"
DATA_DIR = ROOT / "datasets"
FACTORS_HOURLY = DATA_DIR / "factors_hourly.csv"
FACTORS_DAILY = DATA_DIR / "factors_daily.csv"


def _jsonable(obj: Any) -> Any:
    if isinstance(obj, (np.integer, np.floating)):
        v = float(obj)
        return None if not np.isfinite(v) else v
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, pd.DataFrame):
        return obj.replace({np.nan: None}).to_dict(orient="records")
    if isinstance(obj, pd.Series):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, (datetime, date)):
        return obj.isoformat()
    if isinstance(obj, float) and not np.isfinite(obj):
        return None
    return obj


def apply_manual_weather(
    hourly: pd.DataFrame,
    manual: dict[str, float | int] | None,
) -> pd.DataFrame:
    if not manual:
        return hourly
    out = hourly.copy()
    for k, v in manual.items():
        if k in out.columns:
            out[k] = v
    if "apparent_temperature_c" in out.columns and "temperature_c" in manual:
        out["apparent_temperature_c"] = float(manual["temperature_c"]) - 2.0
    if "precipitation_mm" in out.columns and "rain_mm" in manual:
        out["precipitation_mm"] = float(manual["rain_mm"])
    return out


@dataclass
class ArtifactBundle:
    predictor: DemandPredictor
    ui_config: dict[str, Any]
    optimizer_state: dict[str, Any]
    routes: list[str]
    feature_medians: dict[str, Any]
    weather_groups: dict[str, Any]
    pareto_per_route: dict[str, dict[str, Any]]
    factors_mtime: float
    model_built_at: str


@lru_cache(maxsize=1)
def _load_artifacts_cached(model_mtime: float, factors_mtime: float) -> ArtifactBundle:
    predictor = DemandPredictor.load(UI_DIR)
    with (UI_DIR / "ui_config.json").open(encoding="utf-8") as f:
        ui_config = json.load(f)
    optimizer_state = joblib.load(UI_DIR / "optimizer_state.pkl")
    baseline_lookup = pd.read_json(UI_DIR / "baseline_lookup.json")
    routes = sorted(baseline_lookup["route_id"].astype(str).unique().tolist())
    ui_config["_model_built_at"] = datetime.fromtimestamp(model_mtime).strftime("%Y-%m-%d %H:%M")
    feature_medians = load_feature_medians(UI_DIR)
    wg_path = UI_DIR / "weather_groups.json"
    if not wg_path.exists():
        dump_weather_groups(wg_path, FACTORS_HOURLY)
    weather_groups = load_weather_groups(str(UI_DIR), os.path.getmtime(wg_path))
    pareto_per_route: dict[str, dict] = {}
    pareto_path = UI_DIR / "pareto_per_route.json"
    if pareto_path.exists():
        with pareto_path.open(encoding="utf-8") as f:
            pareto_per_route = {str(r["route_id"]): r for r in json.load(f)}
    return ArtifactBundle(
        predictor=predictor,
        ui_config=ui_config,
        optimizer_state=optimizer_state,
        routes=routes,
        feature_medians=feature_medians,
        weather_groups=weather_groups,
        pareto_per_route=pareto_per_route,
        factors_mtime=factors_mtime,
        model_built_at=ui_config["_model_built_at"],
    )


def load_artifacts(*, reload: bool = False) -> ArtifactBundle:
    model_path = UI_DIR / "optimizer_state.pkl"
    if not model_path.exists():
        raise FileNotFoundError(f"Missing optimizer artifacts at {UI_DIR}")
    model_mtime = os.path.getmtime(model_path)
    factors_mtime = os.path.getmtime(FACTORS_HOURLY)
    if reload:
        _load_artifacts_cached.cache_clear()
    return _load_artifacts_cached(model_mtime, factors_mtime)


def factors_date_bounds() -> dict[str, str]:
    fd = pd.read_csv(FACTORS_DAILY, parse_dates=["date"])
    min_d = fd["date"].min().date()
    max_d = fd["date"].max().date()
    picker_max = date.today() + timedelta(days=PICKER_FUTURE_DAYS)
    return {
        "min_date": min_d.isoformat(),
        "max_date": max_d.isoformat(),
        "picker_max_date": picker_max.isoformat(),
    }


def get_meta() -> dict[str, Any]:
    bundle = load_artifacts()
    cfg = bundle.ui_config
    tradeoffs = []
    for key in TRADEOFF_PRESET_ORDER:
        p = TRADEOFF_PRESETS[key]
        tradeoffs.append(
            {
                "key": key,
                "label": p["label_vi"],
                "hint": p["hint"],
                "factor": p["factor"],
            }
        )
    return _jsonable(
        {
            "routes": bundle.routes,
            "model_built_at": bundle.model_built_at,
            "use_per_route_lambda": bool(cfg.get("use_per_route_lambda")),
            "lambda_opt": float(cfg.get("lambda_opt", 92)),
            "lambda_knee_per_route": cfg.get("lambda_knee_per_route", {}),
            "cost_per_vehicle_hour": float(
                cfg.get("cost_per_vehicle_hour", cfg.get("lambda_cost_eval", 150))
            ),
            "tradeoff_presets": tradeoffs,
            "weather_groups": [
                {"key": k, "label": WEATHER_GROUP_VI[k]} for k in WEATHER_GROUP_KEYS
            ],
            "weekday_weekend": [
                {"key": k, "label": WEEKDAY_WEEKEND_VI[k]} for k in WEEKDAY_WEEKEND_OPTIONS
            ],
            "seasons": [{"key": k, "label": SEASON_VI[k]} for k in SEASONS],
            "holiday_names": list_holiday_names(str(FACTORS_DAILY)),
            "date_bounds": factors_date_bounds(),
            "default_constraints": default_constraint_panel(cfg),
        }
    )


def get_tradeoff(route_id: str, preset: str) -> dict[str, Any]:
    bundle = load_artifacts()
    if preset not in TRADEOFF_PRESETS:
        raise ValueError(f"Unknown preset: {preset}")
    row = bundle.pareto_per_route.get(str(route_id))
    lam = lambda_for_preset(route_id, preset, bundle.ui_config)
    knee = route_lambda_knee(route_id, bundle.ui_config)
    return _jsonable(
        {
            "route_id": str(route_id),
            "preset": preset,
            "lambda_cost": lam,
            "lambda_knee": knee,
            "label": tradeoff_compact_label(route_id, preset, bundle.ui_config, row),
        }
    )


@dataclass
class OptimizeRequest:
    route_id: str
    tradeoff_preset: str = "balanced"
    lambda_cost: float | None = None
    input_mode: Literal["date", "scenario"] = "date"
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


def _predict_route_hourly(
    predictor: DemandPredictor,
    route_id: str,
    hourly_factors: pd.DataFrame,
) -> pd.Series:
    feat = predictor.build_features_from_hourly_df([route_id], hourly_factors)
    pred = predictor.predict(feat)
    return pred.set_index("hour")["demand"].sort_index()


def run_optimization(req: OptimizeRequest) -> dict[str, Any]:
    bundle = load_artifacts()
    ui_config = bundle.ui_config
    preset = req.tradeoff_preset if req.tradeoff_preset in TRADEOFF_PRESETS else "balanced"
    lam = (
        float(req.lambda_cost)
        if req.lambda_cost is not None
        else lambda_for_preset(req.route_id, preset, ui_config)
    )

    panel = default_constraint_panel(ui_config)
    panel["use_route_fleet"] = req.use_route_fleet
    panel["use_system_fleet"] = False
    if req.constraints:
        panel.update({k: v for k, v in req.constraints.items() if v is not None})

    overrides = merge_overrides(ui_config, panel)
    validate_constraint_config(overrides, bundle.optimizer_state, route_id=req.route_id)

    scenario_warning: str | None = None
    factor_clip_note: str | None = None
    context_badges: list[str] = []

    if req.input_mode == "date":
        if not req.selected_date:
            raise ValueError("selected_date required for date mode")
        selected = date.fromisoformat(req.selected_date)
        query_hourly, query_src = build_hourly_factors(
            selected,
            factors_hourly_path=str(FACTORS_HOURLY),
            factors_daily_path=str(FACTORS_DAILY),
        )
        prepared = prepare_hourly_factors_for_model(
            query_hourly,
            feature_medians=bundle.feature_medians,
            factors_hourly_path=str(FACTORS_HOURLY),
            factors_hourly_mtime=bundle.factors_mtime,
        )
        hourly_factors = prepared.hourly_factors
        factor_clip_note = prepared.clip_note
        if req.weather_group != "sunny":
            hourly_factors = apply_weather_group_to_hourly(
                hourly_factors, req.weather_group, bundle.weather_groups
            )
        hourly_factors = apply_manual_weather(hourly_factors, req.manual_weather)
        nearest = resolve_to_nearest_training_day(
            hourly_factors,
            str(FACTORS_HOURLY),
            file_mtime=bundle.factors_mtime,
            query_date=pd.Timestamp(selected),
        )
        if not (nearest.is_self_match or nearest.distance < 1e-3):
            scenario_warning = (
                f"Profile ngày chọn không có trong training. "
                f"Dùng ngày gần nhất: {', '.join(nearest.nearest_dates)} ({nearest.note})."
            )
        is_weekend = int(hourly_factors["is_weekend"].median())
        source_label = query_src
        if req.weather_group != "sunny":
            source_label += f" · nhóm {WEATHER_GROUP_VI[req.weather_group]}"
        day_label = "Cuối tuần" if is_weekend else "Ngày thường"
        context_badges = [
            f"Tuyến {req.route_id}",
            selected.strftime("%d/%m/%Y") + f" · {day_label}",
            source_label,
        ]
        export_tag = selected.isoformat()
    else:
        scenario_sel = ScenarioSelection(
            weekday_weekend=req.weekday_weekend,
            season=req.season,
            weather="clear",
            filter_holiday=req.filter_holiday,
            holiday_name=req.holiday_name if req.filter_holiday else None,
            filter_major_event=req.filter_major_event,
        )
        built = build_scenario_hourly_factors(
            scenario_sel,
            factors_daily_path=str(FACTORS_DAILY),
            factors_hourly_path=str(FACTORS_HOURLY),
            file_mtime=bundle.factors_mtime,
        )
        hourly_factors = built.hourly_factors
        hourly_factors = apply_weather_group_to_hourly(
            hourly_factors, req.weather_group, bundle.weather_groups
        )
        hourly_factors = apply_manual_weather(hourly_factors, req.manual_weather)
        prepared = prepare_hourly_factors_for_model(
            hourly_factors,
            feature_medians=bundle.feature_medians,
            factors_hourly_path=str(FACTORS_HOURLY),
            factors_hourly_mtime=bundle.factors_mtime,
        )
        hourly_factors = prepared.hourly_factors
        factor_clip_note = prepared.clip_note
        is_weekend = built.is_weekend
        if built.nn_distance >= 1e-3:
            scenario_warning = (
                f"Kịch bản → ngày training gần nhất: "
                f"{', '.join(built.sample_dates)} ({built.match_note})."
            )
        source_label = f"Kịch bản: {built.label} · NN → {built.n_days} ngày · {built.match_note}"
        context_badges = [
            f"Tuyến {req.route_id}",
            f"Kịch bản · {built.label}",
            WEATHER_GROUP_VI[req.weather_group],
            f"{built.n_days} ngày · median/giờ",
        ]
        if scenario_sel.filter_holiday:
            context_badges.append(
                f"Ngày lễ: {built.holiday_name or 'tất cả'}"
            )
        export_tag = (
            "scenario_"
            + "_".join(
                [scenario_sel.weekday_weekend, scenario_sel.season, req.weather_group]
                + (
                    [scenario_sel.holiday_name.replace(" ", "_")]
                    if scenario_sel.holiday_name
                    else []
                )
                + (["major_event"] if scenario_sel.filter_major_event else [])
            )
        )

    demand_hourly = _predict_route_hourly(bundle.predictor, req.route_id, hourly_factors)
    cap = float(overrides.capacity_per_trip or ui_config.get("capacity_per_trip", 1200))
    result = optimize_route_day(
        req.route_id,
        demand_hourly,
        bundle.optimizer_state,
        lambda_cost=lam,
        capacity_per_trip=int(cap),
        is_weekend=is_weekend,
        lambda_ref=resolve_lambda_ref(req.route_id, ui_config),
        constraint_overrides=overrides,
        ui_config=ui_config,
    )

    base_m = result["baseline_metrics"]
    opt_m = result["optimized_metrics"]
    fleet_unit = float(
        ui_config.get("cost_per_vehicle_hour", ui_config.get("lambda_cost_eval", 150))
    )
    opt_vh = float(opt_m.get("total_vehicle_hours", opt_m["total_trips"]))
    base_vh = float(base_m.get("total_vehicle_hours", base_m["total_trips"]))

    hourly = result["hourly"]
    chart = {
        "hours": hourly["hour"].astype(int).tolist(),
        "baseline_demand": hourly["baseline_demand"].tolist(),
        "predicted_demand": hourly["predicted_demand"].tolist(),
        "baseline_headway": hourly["baseline_headway_min"].tolist(),
        "opt_headway": hourly["opt_headway_min"].tolist(),
    }

    schedule_rows = hourly[
        [
            "hour",
            "baseline_demand",
            "predicted_demand",
            "baseline_trips",
            "opt_trips",
            "baseline_headway_min",
            "opt_headway_min",
        ]
    ].to_dict(orient="records")

    return _jsonable(
        {
            "route_id": req.route_id,
            "lambda_cost": lam,
            "tradeoff_preset": preset,
            "export_tag": export_tag,
            "source_label": source_label,
            "context_badges": context_badges,
            "scenario_warning": scenario_warning,
            "factor_clip_note": factor_clip_note,
            "missing_hours": result.get("missing_hours", []),
            "metrics": {
                "wait": {
                    "optimized": float(opt_m["weighted_avg_wait_min"]),
                    "baseline": float(base_m["weighted_avg_wait_min"]),
                    "delta": float(base_m["weighted_avg_wait_min"] - opt_m["weighted_avg_wait_min"]),
                },
                "vehicle_hours": {
                    "optimized": opt_vh,
                    "baseline": base_vh,
                    "delta": opt_vh - base_vh,
                    "cost_usd": opt_vh * fleet_unit,
                },
                "overcrowding": {
                    "optimized": float(result["overcrowding_optimized"]),
                    "baseline": float(result["overcrowding_baseline"]),
                    "delta_pp": float(
                        result["overcrowding_baseline"] - result["overcrowding_optimized"]
                    ),
                },
                "queue": {
                    "max_queue_optimized": float(opt_m.get("max_queue_length", 0)),
                    "max_queue_baseline": float(base_m.get("max_queue_length", 0)),
                    "overflow_pct_optimized": float(opt_m.get("overflow_pct", 0)),
                    "overflow_pct_baseline": float(base_m.get("overflow_pct", 0)),
                },
                "fleet_utilization": {
                    "optimized": opt_m.get("fleet_utilization"),
                    "baseline": base_m.get("fleet_utilization"),
                },
            },
            "chart": chart,
            "schedule": schedule_rows,
            "constraint_binding": result.get("constraint_binding"),
            "csv_detail": result["detail"].assign(route=req.route_id, lambda_cost=lam).to_dict(
                orient="records"
            ),
        }
    )
