"""λ trade-off helpers — 20 điểm Pareto (notebook §9b, weekday_peak)."""

from __future__ import annotations

import math
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

LAMBDA_BALANCED = 446.0  # Pareto knee
LAMBDA_DEFAULT = LAMBDA_BALANCED
LAMBDA_WAIT = 195.0  # Pareto "Chờ max" (điểm đầu frontier)
LAMBDA_COST = 1479.0  # Pareto "Chi phí min" (điểm cuối frontier)

# Fallback nếu không đọc được CSV (outputs/default/pareto_frontier.csv)
_EMBEDDED_PARETO: list[dict[str, float]] = [
    {"lambda_equiv": 195, "w": 0.9, "f1_improve_pct": 31.46, "f2_delta_pct": 34.42, "total_trips": 11422},
    {"lambda_equiv": 264, "w": 0.858, "f1_improve_pct": 31.19, "f2_delta_pct": 32.31, "total_trips": 11242},
    {"lambda_equiv": 306, "w": 0.816, "f1_improve_pct": 31.00, "f2_delta_pct": 31.02, "total_trips": 11133},
    {"lambda_equiv": 356, "w": 0.774, "f1_improve_pct": 30.78, "f2_delta_pct": 29.83, "total_trips": 11032},
    {"lambda_equiv": 414, "w": 0.732, "f1_improve_pct": 30.48, "f2_delta_pct": 28.43, "total_trips": 10913},
    {"lambda_equiv": 446, "w": 0.689, "f1_improve_pct": 30.28, "f2_delta_pct": 27.56, "total_trips": 10839},
    {"lambda_equiv": 481, "w": 0.647, "f1_improve_pct": 30.08, "f2_delta_pct": 26.77, "total_trips": 10772},
    {"lambda_equiv": 558, "w": 0.605, "f1_improve_pct": 29.55, "f2_delta_pct": 24.93, "total_trips": 10615},
    {"lambda_equiv": 602, "w": 0.563, "f1_improve_pct": 29.20, "f2_delta_pct": 23.81, "total_trips": 10520},
    {"lambda_equiv": 649, "w": 0.521, "f1_improve_pct": 28.83, "f2_delta_pct": 22.74, "total_trips": 10429},
    {"lambda_equiv": 699, "w": 0.479, "f1_improve_pct": 28.40, "f2_delta_pct": 21.56, "total_trips": 10329},
    {"lambda_equiv": 753, "w": 0.437, "f1_improve_pct": 27.91, "f2_delta_pct": 20.34, "total_trips": 10225},
    {"lambda_equiv": 812, "w": 0.395, "f1_improve_pct": 27.20, "f2_delta_pct": 18.68, "total_trips": 10084},
    {"lambda_equiv": 875, "w": 0.353, "f1_improve_pct": 26.49, "f2_delta_pct": 17.15, "total_trips": 9954},
    {"lambda_equiv": 943, "w": 0.311, "f1_improve_pct": 25.63, "f2_delta_pct": 15.41, "total_trips": 9806},
    {"lambda_equiv": 1017, "w": 0.268, "f1_improve_pct": 24.49, "f2_delta_pct": 13.28, "total_trips": 9625},
    {"lambda_equiv": 1096, "w": 0.226, "f1_improve_pct": 23.22, "f2_delta_pct": 11.09, "total_trips": 9439},
    {"lambda_equiv": 1181, "w": 0.184, "f1_improve_pct": 21.75, "f2_delta_pct": 8.72, "total_trips": 9238},
    {"lambda_equiv": 1273, "w": 0.142, "f1_improve_pct": 20.11, "f2_delta_pct": 6.26, "total_trips": 9029},
    {"lambda_equiv": 1479, "w": 0.1, "f1_improve_pct": 16.23, "f2_delta_pct": 1.07, "total_trips": 8588},
]

PARETO_TAGS: dict[float, str] = {
    195.0: "Chờ max",
    446.0: "Knee",
    649.0: "Cân bằng",
    1479.0: "Chi phí min",
}

PARETO_COUNT = 20

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


@dataclass(frozen=True)
class ParetoPoint:
    lambda_equiv: float
    w: float
    f1_improve_pct: float
    f2_delta_pct: float
    total_trips: float
    tag: str = ""

    @classmethod
    def from_row(cls, row: dict[str, Any]) -> ParetoPoint:
        lam = float(row["lambda_equiv"])
        return cls(
            lambda_equiv=lam,
            w=float(row.get("w", 0)),
            f1_improve_pct=float(row.get("f1_improve_pct", 0)),
            f2_delta_pct=float(row.get("f2_delta_pct", 0)),
            total_trips=float(row.get("total_trips", 0)),
            tag=PARETO_TAGS.get(lam, ""),
        )


def _default_pareto_csv() -> Path:
    return Path(__file__).resolve().parents[1] / "outputs" / "default" / "pareto_frontier.csv"


def _rows_from_csv(path: Path) -> list[dict[str, float]] | None:
    if not path.is_file():
        return None
    try:
        import pandas as pd

        df = pd.read_csv(path).sort_values("lambda_equiv")
        return df.to_dict("records")
    except Exception:
        return None


@lru_cache(maxsize=4)
def load_pareto_points(csv_path: str | None = None) -> tuple[ParetoPoint, ...]:
    """20 điểm Pareo tốt nhất — đọc CSV notebook hoặc fallback embedded."""
    rows: list[dict[str, float]] | None = None
    if csv_path:
        rows = _rows_from_csv(Path(csv_path))
    if rows is None:
        rows = _rows_from_csv(_default_pareto_csv())
    if rows is None:
        rows = _EMBEDDED_PARETO
    points = tuple(ParetoPoint.from_row(r) for r in rows)
    if not points:
        raise ValueError("Không có điểm Pareto")
    return points


def pareto_lambda_values(csv_path: str | None = None) -> list[float]:
    return [p.lambda_equiv for p in load_pareto_points(csv_path)]


def default_pareto_index(csv_path: str | None = None) -> int:
    """Chỉ số 1..20 của điểm knee (λ≈446)."""
    for i, lam in enumerate(pareto_lambda_values(csv_path), start=1):
        if abs(lam - LAMBDA_BALANCED) < 0.5:
            return i
    return (PARETO_COUNT + 1) // 2


def lambda_at_pareto_index(index: int, csv_path: str | None = None) -> float:
    """Map slider 1..20 → λ Pareto."""
    i = int(index)
    if not 1 <= i <= PARETO_COUNT:
        raise ValueError(f"Pareto index phải từ 1 đến {PARETO_COUNT}, nhận {index}")
    return pareto_lambda_values(csv_path)[i - 1]


def pareto_point_at_index(index: int, csv_path: str | None = None) -> ParetoPoint:
    return load_pareto_points(csv_path)[int(index) - 1]


def pareto_point_for_lambda(lam: float, csv_path: str | None = None) -> ParetoPoint | None:
    for p in load_pareto_points(csv_path):
        if abs(p.lambda_equiv - lam) < 0.5:
            return p
    return None


def default_pareto_lambda(csv_path: str | None = None) -> float:
    values = pareto_lambda_values(csv_path)
    if LAMBDA_BALANCED in values:
        return LAMBDA_BALANCED
    return min(values, key=lambda v: abs(v - LAMBDA_BALANCED))


def pareto_option_label(lam: float, csv_path: str | None = None) -> str:
    """Nhãn selectbox: λ + tag + trade-off weekday_peak."""
    p = pareto_point_for_lambda(lam, csv_path)
    if p is None:
        return f"λ={lam:.0f}"
    tag = f" · {p.tag}" if p.tag else ""
    pri = PRIORITY_VI[lambda_priority(lam)]
    return (
        f"λ={p.lambda_equiv:.0f}{tag} · {pri} · "
        f"chờ −{p.f1_improve_pct:.1f}% · chi phí +{p.f2_delta_pct:.1f}% · "
        f"{p.total_trips:.0f} chuyến"
    )


def lambda_priority(lam: float) -> str:
    """Phân vùng theo 20 điểm Pareto (446 ∈ Balanced)."""
    if lam <= 306:
        return "Wait-focused"
    if lam <= 753:
        return "Balanced"
    return "Cost-focused"


def lambda_slider_label(lam: float) -> str:
    return pareto_option_label(lam)


def analytical_trips_per_hour(demand: float, lambda_cost: float) -> int:
    t = math.sqrt(30.0 * max(demand, 1e-9) / max(lambda_cost, 1e-9))
    return max(1, int(round(t)))


def pareto_candidates_for_export(csv_path: str | None = None) -> list[float]:
    """Danh sách λ cho ui_config.json / notebook export."""
    return pareto_lambda_values(csv_path)
