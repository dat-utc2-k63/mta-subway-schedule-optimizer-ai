"""λ trade-off helpers — 8 điểm Pareto curated (weekday_peak frontier)."""

from __future__ import annotations

import math
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

# 7 điểm UI (λ ≥ 264) — lấy từ pareto_frontier.csv
PARETO_UI_LAMBDAS: tuple[float, ...] = (264.0, 384.0, 602.0, 753.0, 1017.0, 1273.0, 1479.0)

LAMBDA_KNEE = 384.0
LAMBDA_BALANCED = LAMBDA_KNEE
LAMBDA_DEFAULT = LAMBDA_BALANCED
LAMBDA_WAIT = 264.0  # thấp nhất trên frontier UI (tham chiếu wait-focus)
LAMBDA_COST = 1479.0
LAMBDA_MIN_UI = 264.0

PARETO_COUNT = len(PARETO_UI_LAMBDAS)

PARETO_ZONE_ORDER: tuple[str, ...] = ("recommended", "fleet_save", "min_cost")

PARETO_ZONES: dict[str, dict[str, Any]] = {
    "recommended": {
        "label_vi": "Cân bằng (khuyến nghị)",
        "lambdas": (264.0, 384.0),
        "default": 384.0,
        "hint": "Gần mức cải thiện chờ tối đa (~27%). Mặc định: mức khuyến nghị (384).",
    },
    "fleet_save": {
        "label_vi": "Tiết kiệm xe",
        "lambdas": (602.0, 753.0),
        "default": 602.0,
        "hint": "Giảm 5–8% chi phí fleet, đổi lại mất ~1.5–2.5% cải thiện chờ so với mức khuyến nghị.",
    },
    "min_cost": {
        "label_vi": "Chi phí thấp nhất",
        "lambdas": (1017.0, 1273.0, 1479.0),
        "default": 1017.0,
        "hint": "Ưu tiên cắt số chuyến / fleet; cải thiện chờ giảm dần.",
    },
}

PARETO_TAGS: dict[float, str] = {
    264.0: "Thiên chờ",
    384.0: "Khuyến nghị",
    1479.0: "Tiết kiệm tối đa",
}

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

_EMBEDDED_PARETO: list[dict[str, float]] = [
    {"lambda_equiv": 264, "w": 0.858, "f1_improve_pct": 27.56, "f2_delta_pct": 31.60, "total_trips": 10691},
    {"lambda_equiv": 384, "w": 0.774, "f1_improve_pct": 27.02, "f2_delta_pct": 28.74, "total_trips": 10459},
    {"lambda_equiv": 602, "w": 0.563, "f1_improve_pct": 25.62, "f2_delta_pct": 23.97, "total_trips": 10071},
    {"lambda_equiv": 753, "w": 0.479, "f1_improve_pct": 24.44, "f2_delta_pct": 21.04, "total_trips": 9833},
    {"lambda_equiv": 1017, "w": 0.268, "f1_improve_pct": 20.71, "f2_delta_pct": 13.98, "total_trips": 9260},
    {"lambda_equiv": 1273, "w": 0.142, "f1_improve_pct": 15.86, "f2_delta_pct": 6.87, "total_trips": 8682},
    {"lambda_equiv": 1479, "w": 0.1, "f1_improve_pct": 11.43, "f2_delta_pct": 1.43, "total_trips": 8240},
]


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


def _filter_ui_lambdas(rows: list[dict[str, float]]) -> list[dict[str, float]]:
    by_lam = {float(r["lambda_equiv"]): r for r in rows}
    out: list[dict[str, float]] = []
    for lam in PARETO_UI_LAMBDAS:
        if lam in by_lam:
            out.append(by_lam[lam])
        else:
            for k, r in by_lam.items():
                if abs(k - lam) < 0.5:
                    out.append(r)
                    break
    return out


@lru_cache(maxsize=4)
def load_pareto_points(csv_path: str | None = None) -> tuple[ParetoPoint, ...]:
    """7 điểm Pareto curated (λ ≥ 264) — đọc CSV rồi lọc theo PARETO_UI_LAMBDAS."""
    rows: list[dict[str, float]] | None = None
    if csv_path:
        rows = _rows_from_csv(Path(csv_path))
    if rows is None:
        rows = _rows_from_csv(_default_pareto_csv())
    if rows is None:
        rows = _EMBEDDED_PARETO
    else:
        filtered = _filter_ui_lambdas(rows)
        rows = filtered if filtered else _EMBEDDED_PARETO
    points = tuple(ParetoPoint.from_row(r) for r in rows)
    if not points:
        raise ValueError("Không có điểm Pareto")
    return points


def pareto_lambda_values(csv_path: str | None = None) -> list[float]:
    return [p.lambda_equiv for p in load_pareto_points(csv_path)]


def lambda_to_pareto_index(lam: float, csv_path: str | None = None) -> int:
    """Map λ → chỉ số 1..7 trên frontier UI."""
    for i, v in enumerate(pareto_lambda_values(csv_path), start=1):
        if abs(v - float(lam)) < 0.5:
            return i
    raise ValueError(f"λ={lam} không nằm trong frontier UI {PARETO_UI_LAMBDAS}")


def default_pareto_index(csv_path: str | None = None) -> int:
    """Chỉ số của knee λ=384."""
    return lambda_to_pareto_index(LAMBDA_KNEE, csv_path)


def lambda_at_pareto_index(index: int, csv_path: str | None = None) -> float:
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
    return LAMBDA_KNEE


def pareto_zone_for_lambda(lam: float) -> str:
    for key, z in PARETO_ZONES.items():
        if any(abs(lam - float(v)) < 0.5 for v in z["lambdas"]):
            return key
    if lam <= 760:
        return "fleet_save" if lam >= 600 else "recommended"
    return "min_cost"


def pareto_short_label(index: int, csv_path: str | None = None) -> str:
    p = pareto_point_at_index(index, csv_path)
    tag = f" · {p.tag}" if p.tag else ""
    pri = PRIORITY_VI[lambda_priority(p.lambda_equiv)]
    return f"{pri}{tag} · λ={p.lambda_equiv:.0f}"


def pareto_option_label(lam: float, csv_path: str | None = None) -> str:
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


def pareto_compact_label(lam: float, csv_path: str | None = None) -> str:
    p = pareto_point_for_lambda(lam, csv_path)
    if p is None:
        return f"Mức {lam:.0f}"
    tag = f" — {p.tag}" if p.tag else ""
    return (
        f"Chờ giảm {p.f1_improve_pct:.1f}%{tag} · "
        f"chi phí +{p.f2_delta_pct:.1f}% · ~{p.total_trips:.0f} chuyến/ngày"
    )


def lambda_priority(lam: float) -> str:
    """Phân vùng theo frontier curated (knee 384 ∈ Balanced)."""
    if lam <= 384:
        return "Balanced"
    if lam <= 753:
        return "Balanced"
    return "Cost-focused"


def lambda_slider_label(lam: float) -> str:
    return pareto_option_label(lam)


def analytical_trips_per_hour(demand: float, lambda_cost: float) -> int:
    t = math.sqrt(30.0 * max(demand, 1e-9) / max(lambda_cost, 1e-9))
    return max(1, int(round(t)))


def pareto_candidates_for_export(csv_path: str | None = None) -> list[float]:
    return list(PARETO_UI_LAMBDAS)
