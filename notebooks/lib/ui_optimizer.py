"""Single-route analytical schedule optimization for Streamlit UI."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from . import single_route_pipeline as srp
from .ui_constraints import ConstraintOverrides, apply_post_opt_constraints, compute_binding_stats
from .ui_lambda import LAMBDA_BALANCED, LAMBDA_WAIT


def optimize_schedule_analytical(
    demand: np.ndarray,
    *,
    lambda_cost: float,
    trips_min: np.ndarray,
    trips_max: np.ndarray,
) -> np.ndarray:
    """Per-slot optimum: min D·30/t + λ·t  →  t* = sqrt(30·D/λ), clip bounds."""
    lam = float(lambda_cost)
    tmin = np.asarray(trips_min, dtype=float)
    tmax = np.asarray(trips_max, dtype=float)
    d = np.maximum(np.asarray(demand, dtype=float), 1e-9)
    trips_star = np.sqrt(30.0 * d / lam)
    return np.clip(np.round(trips_star), tmin, tmax).astype(int)


def optimize_schedule_analytical_anchored(
    slot_demand: np.ndarray,
    slot_baseline_demand: np.ndarray,
    baseline_trips: np.ndarray,
    *,
    lambda_cost: float,
    lambda_ref: float = LAMBDA_BALANCED,
    trips_min: np.ndarray,
    trips_max: np.ndarray,
) -> np.ndarray:
    """GTFS-anchored optimum: scale baseline trips by demand and λ vs reference.

    At predicted demand ≈ baseline demand and λ ≈ λ_ref, result ≈ GTFS baseline.
    Lower demand or higher λ → fewer trips than GTFS; higher demand or lower λ → more.
    """
    d_pred = np.maximum(np.asarray(slot_demand, dtype=float), 1e-9)
    d_base = np.maximum(np.asarray(slot_baseline_demand, dtype=float), 1e-9)
    base = np.maximum(np.asarray(baseline_trips, dtype=float), 1.0)
    lam = max(float(lambda_cost), 1e-9)
    lam_ref = max(float(lambda_ref), 1e-9)
    demand_scale = np.sqrt(d_pred / d_base)
    lambda_scale = np.sqrt(lam_ref / lam)
    trips_star = base * demand_scale * lambda_scale
    tmin = np.asarray(trips_min, dtype=float)
    tmax = np.asarray(trips_max, dtype=float)
    return np.clip(np.round(trips_star), tmin, tmax).astype(int)


def filter_route_slots(optimizer_state: dict[str, Any], route_id: str) -> dict[str, np.ndarray]:
    """Extract slot arrays for one route (route × direction × hour)."""
    mask = np.asarray(optimizer_state["slot_route"]) == str(route_id)
    n = int(mask.sum())

    if "direction_share" in optimizer_state:
        direction_share = np.asarray(optimizer_state["direction_share"], dtype=float)[mask]
    else:
        dirs = np.asarray(optimizer_state["slot_dir"])[mask]
        hours = np.asarray(optimizer_state["slot_hour"])[mask]
        direction_share = np.ones(n, dtype=float)
        for h in np.unique(hours):
            h_mask = hours == h
            n_dirs = int(h_mask.sum())
            if n_dirs > 0:
                direction_share[h_mask] = 1.0 / n_dirs

    return {
        "slot_dir": np.asarray(optimizer_state["slot_dir"])[mask],
        "slot_hour": np.asarray(optimizer_state["slot_hour"])[mask],
        "baseline_trips": np.asarray(optimizer_state["baseline_trips"], dtype=float)[mask],
        "trips_min": np.asarray(optimizer_state["TRIPS_MIN"], dtype=float)[mask],
        "trips_max": np.asarray(optimizer_state["TRIPS_MAX"], dtype=float)[mask],
        "direction_share": direction_share,
    }


def allocate_hourly_demand_to_slots(
    demand_by_hour: pd.Series,
    slots: dict[str, np.ndarray],
) -> tuple[np.ndarray, list[int]]:
    """Map route×hour demand → per-direction slot demand; report missing hours."""
    rh = demand_by_hour.copy()
    rh.index = rh.index.astype(int)
    out = np.zeros(len(slots["slot_hour"]), dtype=float)
    missing_hours: list[int] = []
    for i, h in enumerate(slots["slot_hour"]):
        h_int = int(h)
        if h_int in rh.index:
            base = float(rh.loc[h_int])
        else:
            base = float(rh.median())
            if h_int not in missing_hours:
                missing_hours.append(h_int)
        out[i] = base * float(slots["direction_share"][i])
    return out, missing_hours


def overcrowding_risk_index(
    demand: np.ndarray,
    trips: np.ndarray,
    *,
    capacity_per_trip: float,
    peak_hours: tuple[int, ...] = (7, 8, 9, 17, 18, 19),
    slot_hour: np.ndarray | None = None,
) -> float:
    """Fraction of peak-hour slots where effective load exceeds capacity (0–100)."""
    d = np.maximum(np.asarray(demand, dtype=float), 0.0)
    t = np.maximum(np.asarray(trips, dtype=float), 1.0)
    cap = t * float(capacity_per_trip)
    overload = np.clip(d - cap, 0.0, None) / np.maximum(d, 1.0)
    if slot_hour is None:
        return float(np.mean(overload) * 100.0)
    hrs = np.asarray(slot_hour, dtype=int)
    peak_mask = np.isin(hrs, peak_hours)
    if not peak_mask.any():
        return float(np.mean(overload) * 100.0)
    return float(np.mean(overload[peak_mask]) * 100.0)


def _lookup_hourly_baseline(
    route_id: str,
    hour: int,
    is_weekend: int,
    optimizer_state: dict[str, Any],
) -> float:
    """Median demand train (route×hour×weekend) từ baseline_lookup + fallback."""
    bl = optimizer_state["baseline_lookup"]
    fb = optimizer_state["fallback"]
    rid = str(route_id)
    wkd = float(is_weekend)
    try:
        return float(bl.loc[(rid, int(hour), wkd)])
    except (KeyError, TypeError):
        pass
    try:
        return float(fb.loc[(rid, int(hour))])
    except (KeyError, TypeError):
        pass
    if hasattr(bl, "median"):
        med = float(pd.to_numeric(bl, errors="coerce").median())
        if np.isfinite(med) and med > 0:
            return med
    return 100.0


def allocate_baseline_demand_to_slots(
    route_id: str,
    is_weekend: int,
    slots: dict[str, np.ndarray],
    optimizer_state: dict[str, Any],
) -> np.ndarray:
    """Chia baseline demand route×hour theo direction_share (giống predicted demand)."""
    out = np.zeros(len(slots["slot_hour"]), dtype=float)
    for i, h in enumerate(slots["slot_hour"]):
        base = _lookup_hourly_baseline(route_id, int(h), is_weekend, optimizer_state)
        out[i] = base * float(slots["direction_share"][i])
    return out


def optimize_route_day(
    route_id: str,
    demand_by_hour: pd.Series,
    optimizer_state: dict[str, Any],
    *,
    lambda_cost: float,
    capacity_per_trip: float,
    is_weekend: int = 0,
    lambda_ref: float = LAMBDA_BALANCED,
    constraint_overrides: ConstraintOverrides | None = None,
    ui_config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Run analytical optimization for one route; return slot-level + hourly rollup."""
    slots = filter_route_slots(optimizer_state, route_id)
    slot_demand, missing_hours = allocate_hourly_demand_to_slots(demand_by_hour, slots)
    slot_baseline_demand = allocate_baseline_demand_to_slots(
        route_id, is_weekend, slots, optimizer_state
    )
    n = len(slot_demand)
    baseline = np.round(slots["baseline_trips"]).astype(int)
    opt_kwargs = dict(
        slot_baseline_demand=slot_baseline_demand,
        baseline_trips=baseline,
        lambda_ref=lambda_ref,
        trips_min=slots["trips_min"],
        trips_max=slots["trips_max"],
    )

    lam = float(lambda_cost)
    if constraint_overrides is not None and constraint_overrides.lambda_cost is not None:
        lam = float(constraint_overrides.lambda_cost)

    opt_trips = optimize_schedule_analytical_anchored(
        slot_demand,
        lambda_cost=lam,
        **opt_kwargs,
    )
    wait_focus_trips = optimize_schedule_analytical_anchored(
        slot_demand,
        lambda_cost=LAMBDA_WAIT,
        **opt_kwargs,
    )

    overrides = constraint_overrides or ConstraintOverrides()
    cfg = ui_config or {}
    slots_full = {**slots, "slot_route": np.full(n, str(route_id), dtype=object), "baseline_trips": baseline}
    opt_trips = apply_post_opt_constraints(
        opt_trips,
        slot_demand,
        slots_full,
        overrides,
        optimizer_state,
        ui_config=cfg,
    )

    tmin, tmax = slots["trips_min"], slots["trips_max"]
    if ui_config is not None and constraint_overrides is not None:
        from .ui_constraints import recompute_trip_bounds

        tmin, tmax = recompute_trip_bounds(
            baseline.astype(float),
            slots["slot_hour"],
            overrides,
            optimizer_state,
            ui_config,
        )
    binding = compute_binding_stats(opt_trips, tmin, tmax, slots["slot_hour"])

    slot_route = np.full(n, str(route_id), dtype=object)
    base_metrics = srp.compute_wait_with_overflow(
        slot_demand,
        baseline,
        slot_route=slot_route,
        slot_dir=slots["slot_dir"],
        slot_hour=slots["slot_hour"],
        capacity_per_trip=capacity_per_trip,
        lambda_cost=lambda_cost,
    )
    opt_metrics = srp.compute_wait_with_overflow(
        slot_demand,
        opt_trips,
        slot_route=slot_route,
        slot_dir=slots["slot_dir"],
        slot_hour=slots["slot_hour"],
        capacity_per_trip=capacity_per_trip,
        lambda_cost=lambda_cost,
    )

    detail = pd.DataFrame(
        {
            "hour": slots["slot_hour"],
            "direction": slots["slot_dir"],
            "baseline_demand": slot_baseline_demand,
            "predicted_demand": slot_demand,
            "baseline_trips": baseline,
            "opt_trips": opt_trips,
            "baseline_headway_min": 60.0 / np.maximum(baseline, 1),
            "opt_headway_min": 60.0 / np.maximum(opt_trips, 1),
            "trips_min": slots["trips_min"].astype(int),
            "trips_max": slots["trips_max"].astype(int),
        }
    ).sort_values(["hour", "direction"])

    hourly = (
        detail.groupby("hour", as_index=False)
        .agg(
            baseline_demand=("baseline_demand", "sum"),
            predicted_demand=("predicted_demand", "sum"),
            baseline_trips=("baseline_trips", "sum"),
            opt_trips=("opt_trips", "sum"),
        )
        .assign(
            baseline_headway_min=lambda x: 60.0 / np.maximum(x["baseline_trips"], 1),
            opt_headway_min=lambda x: 60.0 / np.maximum(x["opt_trips"], 1),
        )
        .sort_values("hour")
    )

    hourly_by_dir = (
        detail.groupby(["hour", "direction"], as_index=False)
        .agg(
            baseline_demand=("baseline_demand", "sum"),
            predicted_demand=("predicted_demand", "sum"),
            baseline_trips=("baseline_trips", "sum"),
            opt_trips=("opt_trips", "sum"),
        )
        .assign(
            baseline_headway_min=lambda x: 60.0 / np.maximum(x["baseline_trips"], 1),
            opt_headway_min=lambda x: 60.0 / np.maximum(x["opt_trips"], 1),
        )
        .sort_values(["hour", "direction"])
    )

    return {
        "detail": detail,
        "hourly": hourly,
        "hourly_by_direction": hourly_by_dir,
        "missing_hours": missing_hours,
        "wait_focus_trips_total": float(wait_focus_trips.sum()),
        "lambda_trips_delta": float(wait_focus_trips.sum() - opt_trips.sum()),
        "baseline_metrics": base_metrics,
        "optimized_metrics": opt_metrics,
        "overcrowding_baseline": overcrowding_risk_index(
            slot_demand, baseline, capacity_per_trip=capacity_per_trip, slot_hour=slots["slot_hour"]
        ),
        "overcrowding_optimized": overcrowding_risk_index(
            slot_demand, opt_trips, capacity_per_trip=capacity_per_trip, slot_hour=slots["slot_hour"]
        ),
        "constraint_binding": binding,
    }
