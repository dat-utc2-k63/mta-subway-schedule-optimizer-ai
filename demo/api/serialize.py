"""Chuyển kết quả run_optimization (numpy/pandas) → dict JSON-an-toàn cho web UI."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from demo.core.metrics import build_metrics_table


def _f(value: Any, default: float = 0.0) -> float:
    try:
        v = float(value)
        return v if np.isfinite(v) else default
    except (TypeError, ValueError):
        return default


def _over_ceiling(t_star: float, tmin: float, tmax: float) -> float:
    """% lệch so với ràng buộc đang chạm: dương = vượt trần, âm = chạm sàn, 0 = trong khoảng."""
    if tmax > 0 and t_star >= tmax:
        return (t_star / tmax - 1.0) * 100.0
    if tmin > 0 and t_star <= tmin:
        return (t_star / tmin - 1.0) * 100.0
    return 0.0


def serialize_results(results: dict[str, Any], *, threshold_pct: float) -> dict[str, Any]:
    slot_hour = np.asarray(results["slot_hour"]).astype(int)
    slot_dir = np.asarray(results["slot_dir"]).astype(int)
    demand = np.asarray(results["demand"], dtype=float)
    trips = np.asarray(results["trips"], dtype=float)
    baseline = np.asarray(results["baseline"], dtype=float)
    trips_max = np.asarray(results.get("trips_max", np.full(len(trips), np.nan)), dtype=float)
    trips_min = np.asarray(results.get("trips_min", np.zeros(len(trips))), dtype=float)
    t_star = np.asarray(results.get("t_star", trips), dtype=float)

    schedule = []
    for i in range(len(trips)):
        over = _over_ceiling(_f(t_star[i]), _f(trips_min[i]), _f(trips_max[i]))
        over_round = int(round(over))
        schedule.append(
            {
                "hour": int(slot_hour[i]),
                "direction": int(slot_dir[i]),
                "demand": int(round(_f(demand[i]))),
                "trips": int(round(_f(trips[i]))),
                "baseline": int(round(_f(baseline[i]))),
                "delta": int(round(_f(trips[i] - baseline[i]))),
                "headway_min": round(60.0 / _f(trips[i]), 1) if _f(trips[i]) > 0 else None,
                "over_ceiling_pct": over_round,
                "warn": bool(abs(over_round) > threshold_pct),
            }
        )
    schedule.sort(key=lambda r: (r["direction"], r["hour"]))

    cap = _f(results["config"].capacity_per_trip, 1200.0)
    demand_by_hour = [
        {
            "hour": int(slot_hour[i]),
            "direction": int(slot_dir[i]),
            "demand": int(round(_f(demand[i]))),
            "capacity": int(round(_f(trips[i]) * cap)),
        }
        for i in range(len(trips))
    ]
    demand_by_hour.sort(key=lambda r: (r["direction"], r["hour"]))

    m = results["metrics"]
    b = results["baseline_metrics"]
    kpis = {
        "total_trips": {"opt": int(_f(m["total_trips"])), "base": int(_f(b["total_trips"]))},
        "avg_wait_min": {"opt": round(_f(m["weighted_avg_wait_min"]), 2), "base": round(_f(b["weighted_avg_wait_min"]), 2)},
        "fleet_cost": {"opt": round(_f(m.get("fleet_cost"))), "base": round(_f(b.get("fleet_cost")))},
        "min_fleet": {
            "opt": round(_f(results.get("min_fleet_required")), 1),
            "base": round(_f(results.get("baseline_min_fleet_required")), 1),
        },
        "overflow_pct": {"opt": round(_f(m["overflow_pct"]), 1), "base": round(_f(b["overflow_pct"]), 1)},
    }

    pareto: list[dict[str, float]] = []
    pdf = results.get("pareto_df")
    if isinstance(pdf, pd.DataFrame) and not pdf.empty and {"f1", "f2"}.issubset(pdf.columns):
        sdf = pdf.sort_values("f2")
        lam = sdf["lambda_equiv"] if "lambda_equiv" in sdf.columns else pd.Series([np.nan] * len(sdf))
        for f1, f2, lv in zip(sdf["f1"], sdf["f2"], lam):
            pareto.append({"f1": round(_f(f1), 1), "f2": round(_f(f2), 1), "lambda": round(_f(lv), 0)})

    metric_table = build_metrics_table(
        results["metrics"],
        results["baseline_metrics"],
        results.get("min_fleet_required"),
        results.get("baseline_min_fleet_required"),
    ).to_dict(orient="records")

    n_warn = sum(1 for r in schedule if r["warn"])
    return {
        "route_id": str(results["config"].route_id),
        "lambda_used": int(_f(results.get("lambda_used"))),
        "w_range": [round(_f(results.get("w_range", (0.1, 0.9))[0]), 2), round(_f(results.get("w_range", (0.1, 0.9))[1]), 2)],
        "n_over_ceiling": int(n_warn),
        "schedule": schedule,
        "demand_by_hour": demand_by_hour,
        "kpis": kpis,
        "pareto": pareto,
        "metric_table": metric_table,
    }
