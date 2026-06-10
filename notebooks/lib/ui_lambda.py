"""λ trade-off helpers for Streamlit UI."""

from __future__ import annotations

import math

LAMBDA_WAIT = 50.0
LAMBDA_BALANCED = 446.0  # Pareto knee (notebook §9b)
LAMBDA_COST = 1000.0

LAMBDA_SLIDER_MIN = 50
LAMBDA_SLIDER_MAX = 1000
LAMBDA_SLIDER_STEP = 50
LAMBDA_SLIDER_DEFAULT = 450  # gần Pareto knee 446 trên lưới bước 50

REFERENCE_DEMAND = 500.0

PRIORITY_COLORS: dict[str, str] = {
    "Wait-focused": "#2dd4bf",
    "Balanced": "#fbbf24",
    "Cost-focused": "#f87171",
}

PRIORITY_VI: dict[str, str] = {
    "Wait-focused": "Ưu tiên chờ",
    "Balanced": "Cân bằng",
    "Cost-focused": "Ưu tiên chi phí",
}


def lambda_priority(lam: float) -> str:
    """Phân vùng ưu tiên trên dải slider (446 ∈ Balanced)."""
    if lam <= 250:
        return "Wait-focused"
    if lam <= 650:
        return "Balanced"
    return "Cost-focused"


def lambda_slider_label(lam: float) -> str:
    return f"{PRIORITY_VI[lambda_priority(lam)]} · λ={lam:.0f}"


def analytical_trips_per_hour(demand: float, lambda_cost: float) -> int:
    t = math.sqrt(30.0 * max(demand, 1e-9) / max(lambda_cost, 1e-9))
    return max(1, int(round(t)))
