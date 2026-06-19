"""Trade-off λ helpers — per-route Pareto knee (queue wait vs vehicle-hours)."""

from __future__ import annotations

import math
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

LAMBDA_MIN = 15.0
LAMBDA_KNEE_DEFAULT = 92.0
LAMBDA_BALANCED = LAMBDA_KNEE_DEFAULT
LAMBDA_DEFAULT = LAMBDA_BALANCED
LAMBDA_WAIT = 46.0
LAMBDA_COST = 447.0

PRIORITY_COLORS: dict[str, str] = {
    "Wait-focused": "#2dd4bf",
    "Balanced": "#fbbf24",
    "Cost-focused": "#f87171",
}

PRIORITY_VI: dict[str, str] = {
    "Wait-focused": "Ưu tiên chờ",
    "Balanced": "Cân bằng",
    "Cost-focused": "Ưu tiên VH",
}

TRADEOFF_PRESET_ORDER: tuple[str, ...] = ("wait", "balanced", "vh_save", "min_vh")

TRADEOFF_PRESETS: dict[str, dict[str, Any]] = {
    "wait": {
        "label_vi": "Ưu tiên chờ",
        "factor": 0.5,
        "priority": "Wait-focused",
        "hint": "Giảm chờ tối đa; vehicle-hours tăng nhiều hơn.",
    },
    "balanced": {
        "label_vi": "Cân bằng (khuyến nghị)",
        "factor": 1.0,
        "priority": "Balanced",
        "hint": "Điểm knee Pareto riêng cho tuyến — trade-off chờ vs VH.",
    },
    "vh_save": {
        "label_vi": "Tiết kiệm vehicle-hours",
        "factor": 1.75,
        "priority": "Balanced",
        "hint": "Giảm VH ~30–40%; chấp nhận chờ cao hơn một chút.",
    },
    "min_vh": {
        "label_vi": "VH thấp nhất",
        "factor": 3.0,
        "priority": "Cost-focused",
        "hint": "Ưu tiên cắt vehicle-hours; cải thiện chờ giảm.",
    },
}


@dataclass(frozen=True)
class RouteParetoInfo:
    route_id: str
    lambda_knee: float
    f1_improve_pct: float | None = None
    f2_delta_pct: float | None = None
    f2_knee: float | None = None
    total_trips_knee: float | None = None

    @classmethod
    def from_config(
        cls,
        route_id: str,
        ui_config: dict[str, Any],
        pareto_row: dict[str, Any] | None = None,
    ) -> RouteParetoInfo:
        rid = str(route_id)
        lpr = ui_config.get("lambda_knee_per_route") or {}
        knee = float(lpr.get(rid, ui_config.get("lambda_opt", LAMBDA_KNEE_DEFAULT)))
        row = pareto_row or {}
        return cls(
            route_id=rid,
            lambda_knee=knee,
            f1_improve_pct=_opt_float(row.get("f1_improve_pct")),
            f2_delta_pct=_opt_float(row.get("f2_delta_pct")),
            f2_knee=_opt_float(row.get("f2_knee")),
            total_trips_knee=_opt_float(row.get("total_trips_knee")),
        )


def _opt_float(v: Any) -> float | None:
    if v is None:
        return None
    try:
        f = float(v)
        return f if math.isfinite(f) else None
    except (TypeError, ValueError):
        return None


def route_lambda_knee(route_id: str, ui_config: dict[str, Any]) -> float:
    lpr = ui_config.get("lambda_knee_per_route") or {}
    return float(lpr.get(str(route_id), ui_config.get("lambda_opt", LAMBDA_KNEE_DEFAULT)))


def lambda_for_preset(
    route_id: str,
    preset_key: str,
    ui_config: dict[str, Any],
) -> float:
    """λ trade-off = route knee × preset factor (clamped)."""
    knee = route_lambda_knee(route_id, ui_config)
    factor = float(TRADEOFF_PRESETS[preset_key]["factor"])
    return max(LAMBDA_MIN, knee * factor)


def lambda_priority(lam: float, *, knee: float | None = None) -> str:
    ref = float(knee if knee is not None else LAMBDA_KNEE_DEFAULT)
    if lam <= ref * 0.75:
        return "Wait-focused"
    if lam <= ref * 2.0:
        return "Balanced"
    return "Cost-focused"


def tradeoff_compact_label(
    route_id: str,
    preset_key: str,
    ui_config: dict[str, Any],
    pareto_row: dict[str, Any] | None = None,
) -> str:
    lam = lambda_for_preset(route_id, preset_key, ui_config)
    info = RouteParetoInfo.from_config(route_id, ui_config, pareto_row)
    if preset_key == "balanced" and info.f1_improve_pct is not None:
        vh = info.f2_delta_pct if info.f2_delta_pct is not None else 0.0
        return (
            f"λ={lam:.0f} · chờ −{info.f1_improve_pct:.0f}% · VH +{vh:.0f}%"
        )
    factors = {
        "wait": "chờ ↓↓",
        "balanced": "cân bằng",
        "vh_save": "VH ↓",
        "min_vh": "VH ↓↓",
    }
    return f"λ={lam:.0f} · {factors.get(preset_key, preset_key)}"


def priority_pill_text(lam: float, *, knee: float | None = None) -> tuple[str, str]:
    """Return (priority_vi, color_hex)."""
    key = lambda_priority(lam, knee=knee)
    return PRIORITY_VI[key], PRIORITY_COLORS[key]


# --- Legacy API (backward compat for tests / old callers) ---

PARETO_UI_LAMBDAS: tuple[float, ...] = (46.0, 92.0, 161.0, 276.0, 447.0)
PARETO_COUNT = len(PARETO_UI_LAMBDAS)
PARETO_ZONE_ORDER = TRADEOFF_PRESET_ORDER

PARETO_ZONES: dict[str, dict[str, Any]] = {
    k: {
        "label_vi": TRADEOFF_PRESETS[k]["label_vi"],
        "lambdas": (lambda_for_preset("1", k, {"lambda_knee_per_route": {"1": 162.0}, "lambda_opt": 92.0}),),
        "default": lambda_for_preset("1", k, {"lambda_knee_per_route": {"1": 162.0}, "lambda_opt": 92.0}),
        "hint": TRADEOFF_PRESETS[k]["hint"],
    }
    for k in TRADEOFF_PRESET_ORDER
}


@dataclass(frozen=True)
class ParetoPoint:
    lambda_equiv: float
    w: float = 0.5
    f1_improve_pct: float = 0.0
    f2_delta_pct: float = 0.0
    total_trips: float = 0.0
    tag: str = ""


def pareto_point_for_lambda(
    lam: float,
    route_id: str | None = None,
    ui_config: dict[str, Any] | None = None,
    pareto_row: dict[str, Any] | None = None,
    csv_path: str | None = None,
) -> ParetoPoint | None:
    _ = csv_path
    if ui_config is not None and route_id is not None:
        info = RouteParetoInfo.from_config(route_id, ui_config, pareto_row)
        tag = "Khuyến nghị" if abs(lam - info.lambda_knee) < 1.0 else ""
        return ParetoPoint(
            lambda_equiv=float(lam),
            f1_improve_pct=float(info.f1_improve_pct or 0.0),
            f2_delta_pct=float(info.f2_delta_pct or 0.0),
            total_trips=float(info.total_trips_knee or 0.0),
            tag=tag,
        )
    return ParetoPoint(lambda_equiv=float(lam))


def pareto_compact_label(
    lam: float,
    route_id: str | None = None,
    ui_config: dict[str, Any] | None = None,
    pareto_row: dict[str, Any] | None = None,
    csv_path: str | None = None,
) -> str:
    _ = csv_path
    p = pareto_point_for_lambda(lam, route_id, ui_config, pareto_row)
    if p is None:
        return f"λ={lam:.0f}"
    tag = f" — {p.tag}" if p.tag else ""
    if p.f1_improve_pct > 0:
        return (
            f"Chờ −{p.f1_improve_pct:.0f}%{tag} · "
            f"VH +{p.f2_delta_pct:.0f}% · λ={p.lambda_equiv:.0f}"
        )
    return f"λ={p.lambda_equiv:.0f}{tag}"


def lambda_to_pareto_index(lam: float, csv_path: str | None = None) -> int:
    _ = csv_path
    for i, v in enumerate(PARETO_UI_LAMBDAS, start=1):
        if abs(v - float(lam)) < 50:
            return i
    return 2


def analytical_trips_per_hour(
    demand: float,
    lambda_cost: float,
    *,
    cycle_time_min: float = 90.0,
) -> int:
    cyc = max(float(cycle_time_min), 1.0)
    t = math.sqrt(1800.0 * max(demand, 1e-9) / (max(float(lambda_cost), 1e-9) * cyc))
    return max(1, int(round(t)))
