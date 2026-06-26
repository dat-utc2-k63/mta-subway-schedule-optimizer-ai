from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import sys
from typing import Any

import joblib
import numpy as np
import pandas as pd
from functools import lru_cache

from demo.config import (
    CAPACITY_PER_TRIP,
    GTFS_DIR,
    LAMBDA_KNEE_FALLBACK,
    MAX_HEADWAY_MIN,
    MIN_HEADWAY_MIN,
    OVERNIGHT_MAX_HEADWAY_MIN,
    SMOOTHNESS_MAX_DELTA,
    TRIPS_DAYTIME_MAX_FACTOR,
    TRIPS_MIN_FACTOR,
    TRIPS_OVERNIGHT_MAX_FACTOR,
    TRIPS_OVERNIGHT_MIN_FACTOR,
    UI_EXPORT_DIR,
)

NOTEBOOKS_DIR = Path(__file__).resolve().parents[2] / "notebooks"
if str(NOTEBOOKS_DIR) not in sys.path:
    sys.path.insert(0, str(NOTEBOOKS_DIR))

from lib import demand_runtime  # noqa: E402
from lib import single_route_pipeline as srp  # noqa: E402


@dataclass
class RunConfig:
    route_id: str
    lambda_val: int
    input_mode: str
    hourly_factors: pd.DataFrame
    capacity_per_trip: float
    max_overflow_pct: float
    cost_per_vehicle_hour: float
    route_fleet_limit: float
    max_day_at_min_ratio: float
    max_day_at_max_ratio: float
    max_night_at_min_ratio: float
    max_night_at_max_ratio: float
    smoothness_delta: int
    min_headway_min: float
    max_headway_min: float
    overnight_max_headway_min: float
    use_route_fleet_cap: bool
    use_system_fleet_cap: bool
    use_capacity_constraint: bool
    use_smoothness_constraint: bool
    system_fleet_override: int
    auto_knee: bool = True
    w_low: float = 0.1
    w_high: float = 0.9
    trips_min_factor: float = TRIPS_MIN_FACTOR
    trips_overnight_min_factor: float = TRIPS_OVERNIGHT_MIN_FACTOR
    trips_daytime_max_factor: float = TRIPS_DAYTIME_MAX_FACTOR
    trips_overnight_max_factor: float = TRIPS_OVERNIGHT_MAX_FACTOR
    max_over_ceiling_pct: float = 50.0


@lru_cache(maxsize=1)
def load_models_and_data() -> dict[str, Any]:
    ui_dir = UI_EXPORT_DIR
    state = joblib.load(ui_dir / "optimizer_state.pkl")
    route_meta = json.loads((ui_dir / "route_meta.json").read_text(encoding="utf-8"))
    predictor = None
    try:
        predictor = demand_runtime.DemandPredictor.load(ui_dir)
        try:
            from .feature_imputer import build_feature_imputer

            predictor.lag_imputer = build_feature_imputer(predictor)
        except Exception:
            predictor.lag_imputer = None
    except Exception:
        predictor = None
    headway = srp.build_headway_from_gtfs(GTFS_DIR)
    return {"state": state, "route_meta": route_meta, "predictor": predictor, "headway": headway}


def route_quick_info(route_id: str, state: dict[str, Any]) -> dict[str, Any]:
    route_hours = state["route_meta"].get("route_dir_hours", {})
    dirs = []
    all_hours = []
    for key, hours in route_hours.items():
        rid, d = key.split("|")
        if rid == route_id:
            dirs.append(int(d))
            all_hours.extend(hours)
    return {"n_directions": len(set(dirs)), "hours": sorted(set(int(h) for h in all_hours))}


def _build_route_slots(route_id: str, state: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    slot_route = np.asarray(state["state"]["slot_route"]).astype(str)
    mask = slot_route == str(route_id)
    return (
        np.asarray(state["state"]["slot_route"])[mask].astype(str),
        np.asarray(state["state"]["slot_dir"])[mask].astype(int),
        np.asarray(state["state"]["slot_hour"])[mask].astype(int),
        mask,
    )


def _demand_by_slot(cfg: RunConfig, assets: dict[str, Any], slot_dir: np.ndarray, slot_hour: np.ndarray) -> tuple[np.ndarray, pd.DataFrame]:
    predictor = assets["predictor"]
    if predictor is not None:
        stations = predictor.stations_for_routes([cfg.route_id])
        feat = predictor.build_station_features_from_hourly_df(stations, cfg.hourly_factors)
        station_demand = predictor.predict_station(feat)
        rows = []
        for direction in sorted(set(slot_dir.tolist())):
            arr = srp.project_station_demand_to_departures(
                station_demand,
                predictor.station_rd_weights,
                predictor.travel_offsets,
                slot_route=np.asarray([cfg.route_id] * len(slot_hour)),
                slot_dir=np.asarray([direction] * len(slot_hour)),
                slot_hour=slot_hour,
                board_minute=float(assets["state"].get("board_minute", 30.0)),
            )
            rows.append(pd.DataFrame({"route_id": cfg.route_id, "direction": direction, "hour": slot_hour, "demand": arr}))
        demand_df = pd.concat(rows, ignore_index=True).groupby(["route_id", "direction", "hour"], as_index=False)["demand"].sum()
        station_or_route_df = station_demand.copy()
        if isinstance(predictor.station_rd_weights, pd.DataFrame):
            rw = predictor.station_rd_weights.copy()
            if {"station_complex_id", "route", "hour", "alloc_weight"}.issubset(rw.columns):
                rw = rw.loc[rw["route"].astype(str) == str(cfg.route_id)].copy()
                rw["station_complex_id"] = rw["station_complex_id"].astype(str)
                rw["hour"] = rw["hour"].astype(int)
                alloc = rw.groupby(["station_complex_id", "hour"], as_index=False)["alloc_weight"].sum()
                alloc["alloc_weight"] = alloc["alloc_weight"].clip(lower=0.0)
                station_or_route_df["station_complex_id"] = station_or_route_df["station_complex_id"].astype(str)
                station_or_route_df["hour"] = station_or_route_df["hour"].astype(int)
                station_or_route_df = station_or_route_df.merge(alloc, on=["station_complex_id", "hour"], how="left")
                station_or_route_df["alloc_weight"] = station_or_route_df["alloc_weight"].fillna(0.0)
                station_or_route_df["demand"] = station_or_route_df["demand"] * station_or_route_df["alloc_weight"]
                station_or_route_df = station_or_route_df.loc[station_or_route_df["demand"] > 0].copy()
                station_or_route_df = station_or_route_df[["station_complex_id", "hour", "demand"]]
    else:
        sched = pd.read_json(UI_EXPORT_DIR / "schedule_weekday_peak.json")
        demand_df = (
            sched.loc[sched["route"] == cfg.route_id, ["route", "direction", "hour", "demand_pred"]]
            .rename(columns={"route": "route_id", "demand_pred": "demand"})
            .copy()
        )
        station_or_route_df = demand_df.copy()
        scenario_station = assets["state"].get("scenario_station_demand")
        station_weights = assets["state"].get("station_rd_weights")
        if isinstance(scenario_station, dict):
            fallback_station = scenario_station.get("weekday_peak")
            if isinstance(fallback_station, pd.DataFrame) and {"station_complex_id", "hour", "demand"}.issubset(fallback_station.columns):
                station_or_route_df = fallback_station.copy()
                if isinstance(station_weights, pd.DataFrame) and {"station_complex_id", "route"}.issubset(station_weights.columns):
                    sw = station_weights.copy()
                    sw = sw.loc[sw["route"].astype(str) == str(cfg.route_id)].copy()
                    sw["station_complex_id"] = sw["station_complex_id"].astype(str)
                    if "hour" in sw.columns and "alloc_weight" in sw.columns:
                        sw["hour"] = sw["hour"].astype(int)
                        alloc = sw.groupby(["station_complex_id", "hour"], as_index=False)["alloc_weight"].sum()
                        alloc["alloc_weight"] = alloc["alloc_weight"].clip(lower=0.0)
                        station_or_route_df["station_complex_id"] = station_or_route_df["station_complex_id"].astype(str)
                        station_or_route_df["hour"] = station_or_route_df["hour"].astype(int)
                        station_or_route_df = station_or_route_df.merge(alloc, on=["station_complex_id", "hour"], how="left")
                        station_or_route_df["alloc_weight"] = station_or_route_df["alloc_weight"].fillna(0.0)
                        station_or_route_df["demand"] = station_or_route_df["demand"] * station_or_route_df["alloc_weight"]
                        station_or_route_df = station_or_route_df.loc[station_or_route_df["demand"] > 0].copy()
                        station_or_route_df = station_or_route_df[["station_complex_id", "hour", "demand"]]
                    else:
                        station_set = set(sw["station_complex_id"].unique().tolist())
                        if station_set:
                            station_or_route_df = station_or_route_df.loc[
                                station_or_route_df["station_complex_id"].astype(str).isin(station_set)
                            ].copy()
    demand_df["key"] = demand_df["direction"].astype(int).astype(str) + "|" + demand_df["hour"].astype(int).astype(str)
    key_to_demand = {k: float(v) for k, v in zip(demand_df["key"], demand_df["demand"])}
    demand = np.array([key_to_demand.get(f"{int(d)}|{int(h)}", 0.0) for d, h in zip(slot_dir, slot_hour)], dtype=float)
    return demand, station_or_route_df


def _compute_trip_bounds(
    baseline_trips: np.ndarray,
    slot_hour: np.ndarray,
    *,
    trips_min_factor: float,
    trips_overnight_min_factor: float,
    trips_daytime_max_factor: float,
    trips_overnight_max_factor: float,
    min_headway_min: float,
    max_headway_min: float,
    overnight_max_headway_min: float,
) -> tuple[np.ndarray, np.ndarray]:
    """TRIPS_MIN/MAX tu baseline theo 4 factor (gia tri tren UI) ∩ rang buoc headway.

    - Ban ngay: MAX = ceil(baseline × trips_daytime_max_factor), MIN = floor(baseline × trips_min_factor).
    - Ban dem (23-06): MAX = ceil(baseline × trips_overnight_max_factor) (it chuyen hon),
      MIN = floor(baseline × trips_overnight_min_factor).
    - Sau do giao voi tran/san suy tu headway (min/max headway, headway dem).
    """
    base = np.maximum(np.asarray(baseline_trips, dtype=float), 0.0)
    hrs = np.asarray(slot_hour, dtype=int)
    min_hw = max(float(min_headway_min), 0.5)
    day_max_hw = max(float(max_headway_min), min_hw)
    ovn_max_hw = max(float(overnight_max_headway_min), min_hw)
    max_trips_hw = int(np.floor(60.0 / min_hw))

    n = len(base)
    tmin = np.zeros(n, dtype=int)
    tmax = np.zeros(n, dtype=int)
    for i in range(n):
        h = int(hrs[i])
        is_overnight = srp.is_overnight_hour(h)
        min_f = float(trips_overnight_min_factor if is_overnight else trips_min_factor)
        max_f = float(trips_overnight_max_factor if is_overnight else trips_daytime_max_factor)
        max_hw = ovn_max_hw if is_overnight else day_max_hw
        min_trips_hw = int(np.ceil(60.0 / max_hw))
        floor_min = 1 if is_overnight else 2

        ti = max(int(np.floor(base[i] * min_f)), min_trips_hw, floor_min)
        ta = max(int(np.ceil(base[i] * max_f)), ti)
        ta = min(ta, max_trips_hw)
        if ta < ti:
            ta = ti
        tmin[i] = ti
        tmax[i] = ta
    return tmin, tmax


def _enforce_route_fleet_limit(
    trips: np.ndarray,
    *,
    demand: np.ndarray,
    trips_min: np.ndarray,
    slot_dir: np.ndarray,
    slot_hour: np.ndarray,
    cycle_time_min: np.ndarray,
    route_fleet_limit: float,
) -> np.ndarray:
    if route_fleet_limit <= 0:
        return trips
    t = np.asarray(trips, dtype=int).copy()
    tmin = np.asarray(trips_min, dtype=int)
    limit = float(route_fleet_limit)
    max_steps = int(np.maximum((t - tmin).sum(), 0))
    for _ in range(max_steps + 1):
        ok = True
        for h in sorted(set(slot_hour.tolist())):
            mask_h = slot_hour == h
            req = float(np.sum(t[mask_h] * cycle_time_min[mask_h] / 60.0))
            if req > limit + 1e-9:
                ok = False
                candidates = np.where(mask_h & (t > tmin))[0]
                if candidates.size == 0:
                    continue
                idx = int(candidates[np.argmin(demand[candidates])])
                t[idx] -= 1
        if ok:
            return t
    raise ValueError(f"Khong dat duoc gioi han fleet {limit:.1f} voi bo rang buoc hien tai.")


def _compute_live_knee(
    cfg: RunConfig,
    state: dict[str, Any],
    demand: np.ndarray,
    baseline: np.ndarray,
    cycle_time_min: np.ndarray,
    trips_min: np.ndarray,
    trips_max: np.ndarray,
    slot_route: np.ndarray,
    slot_dir: np.ndarray,
    slot_hour: np.ndarray,
    w_range: tuple[float, float] = (0.1, 0.9),
) -> tuple[pd.DataFrame, pd.Series | None]:
    """Quet Pareto frontier theo demand/rang buoc hien tai roi chon knee (max curvature)."""

    def _opt(d: np.ndarray, lam: float) -> np.ndarray:
        return srp.analytical_trips_per_slot(
            d, float(lam), cycle_time_min, trips_min=trips_min, trips_max=trips_max
        )

    def _eval(t: np.ndarray, d: np.ndarray, _lam: float | None = None) -> dict[str, Any]:
        return srp.compute_schedule_metrics(
            d,
            t,
            slot_route=slot_route,
            slot_dir=slot_dir,
            slot_hour=slot_hour,
            capacity_per_trip=cfg.capacity_per_trip,
            cycle_times=state["cycle_times"],
            cost_per_vehicle_hour=cfg.cost_per_vehicle_hour,
        )

    lambda_scan = np.unique(np.round(np.geomspace(20.0, 3000.0, 80)).astype(float))
    pareto_df = srp.generate_pareto_frontier(
        {"current": np.asarray(demand, dtype=float)},
        np.asarray(baseline, dtype=float),
        _opt,
        _eval,
        scenario="current",
        n_points=18,
        lambda_eval=cfg.cost_per_vehicle_hour,
        lambda_scan=lambda_scan,
        w_range=(float(w_range[0]), float(w_range[1])),
    )
    if pareto_df is None or len(pareto_df) == 0:
        return pd.DataFrame(), None
    knee = srp.find_knee_point(pareto_df)
    return pareto_df, knee


def run_optimization(cfg: RunConfig, assets: dict[str, Any]) -> dict[str, Any]:
    slot_route, slot_dir, slot_hour, mask = _build_route_slots(cfg.route_id, assets)
    state = assets["state"]
    baseline = np.asarray(state["baseline_trips"])[mask].astype(int)
    trips_min, trips_max = _compute_trip_bounds(
        baseline,
        slot_hour,
        trips_min_factor=cfg.trips_min_factor,
        trips_overnight_min_factor=cfg.trips_overnight_min_factor,
        trips_daytime_max_factor=cfg.trips_daytime_max_factor,
        trips_overnight_max_factor=cfg.trips_overnight_max_factor,
        min_headway_min=cfg.min_headway_min,
        max_headway_min=cfg.max_headway_min,
        overnight_max_headway_min=cfg.overnight_max_headway_min,
    )
    cycle_map = {(str(r), int(d)): float(c) for r, d, c in state["cycle_times"][["route_id", "direction_id", "cycle_time_min"]].to_numpy()}
    cycle_time_min = np.array([cycle_map.get((str(r), int(d)), 90.0) for r, d in zip(slot_route, slot_dir)], dtype=float)

    demand, station_or_route_demand_df = _demand_by_slot(cfg, assets, slot_dir, slot_hour)
    if demand.size == 0 or not np.isfinite(demand).all():
        raise ValueError("Khong the tao nhu cau hop le cho cau hinh hien tai.")

    lambda_knee_baked = int(state.get("lambda_knee_per_route", {}).get(cfg.route_id, LAMBDA_KNEE_FALLBACK))
    lambda_knee_live: int | None = None
    pareto_df = pd.DataFrame()
    if cfg.auto_knee:
        try:
            pareto_df, knee_row = _compute_live_knee(
                cfg, state, demand, baseline, cycle_time_min,
                trips_min, trips_max, slot_route, slot_dir, slot_hour,
                w_range=(float(cfg.w_low), float(cfg.w_high)),
            )
            if knee_row is not None:
                lambda_knee_live = int(round(float(knee_row["lambda_equiv"])))
        except Exception:
            lambda_knee_live = None
            pareto_df = pd.DataFrame()

    lambda_used = lambda_knee_live if (cfg.auto_knee and lambda_knee_live) else int(cfg.lambda_val)

    # Do bao hoa cua nghiem analytical tai lambda dang dung (truoc cac rang buoc khac):
    # cho biet λ con tac dung hay da bi tran/san chi phoi.
    t_star = np.sqrt(
        1800.0 * np.maximum(demand, 1e-9) / (max(float(lambda_used), 1e-9) * np.maximum(cycle_time_min, 1.0))
    )
    n_sat = max(len(t_star), 1)
    sat_at_max = float((t_star >= trips_max).sum()) / n_sat * 100.0
    sat_at_min = float((t_star <= trips_min).sum()) / n_sat * 100.0
    lambda_sat = {
        "at_max_pct": sat_at_max,
        "at_min_pct": sat_at_min,
        "interior_pct": max(0.0, 100.0 - sat_at_max - sat_at_min),
    }

    trips = srp.analytical_trips_per_slot(demand, lambda_used, cycle_time_min, trips_min=trips_min, trips_max=trips_max)
    # Service window GTFS (auto, khong hien thi UI): gioi han trips trong khung first/last departure.
    service_windows = state.get("service_windows")
    if isinstance(service_windows, pd.DataFrame) and not service_windows.empty:
        trips = srp.apply_service_window_constraints(
            trips,
            slot_route=slot_route,
            slot_dir=slot_dir,
            slot_hour=slot_hour,
            windows=service_windows,
            baseline_trips=baseline,
            trips_min=trips_min,
            trips_max=trips_max,
        )
    trips = srp.apply_optimizer_constraints(
        trips,
        demand,
        slot_route=slot_route,
        slot_dir=slot_dir,
        slot_hour=slot_hour,
        cycle_times=state["cycle_times"],
        fleet_by_route_dir=state["fleet_by_route_dir"],
        max_system_fleet=None,
        capacity_per_trip=cfg.capacity_per_trip,
        max_overflow_pct=cfg.max_overflow_pct,
        smoothness_max_delta=cfg.smoothness_delta,
        trips_min=trips_min,
        trips_max=trips_max,
        use_route_fleet=cfg.use_route_fleet_cap,
        use_system_fleet=False,
        use_capacity=cfg.use_capacity_constraint,
        use_smoothness=cfg.use_smoothness_constraint,
    )
    trips = _enforce_route_fleet_limit(
        trips,
        demand=demand,
        trips_min=trips_min,
        slot_dir=slot_dir,
        slot_hour=slot_hour,
        cycle_time_min=cycle_time_min,
        route_fleet_limit=cfg.route_fleet_limit,
    )
    if trips.size == 0 or not np.isfinite(trips).all() or np.any(trips < 0):
        raise ValueError("Toi uu khong tim duoc lich hop le voi bo rang buoc hien tai.")

    metrics = srp.compute_schedule_metrics(
        demand,
        trips,
        slot_route=slot_route,
        slot_dir=slot_dir,
        slot_hour=slot_hour,
        capacity_per_trip=cfg.capacity_per_trip,
        cycle_times=state["cycle_times"],
        cost_per_vehicle_hour=cfg.cost_per_vehicle_hour,
    )
    baseline_metrics = srp.compute_schedule_metrics(
        demand,
        baseline,
        slot_route=slot_route,
        slot_dir=slot_dir,
        slot_hour=slot_hour,
        capacity_per_trip=cfg.capacity_per_trip,
        cycle_times=state["cycle_times"],
        cost_per_vehicle_hour=cfg.cost_per_vehicle_hour,
    )
    bounds = srp.report_bound_status(
        trips,
        trips_min,
        trips_max,
        slot_hour,
        target_interior_pct=0.0,
        target_daytime_interior_pct=0.0,
        verbose=False,
    )
    by_group = bounds.get("by_hour_group")
    day_at_min_ratio = 0.0
    day_at_max_ratio = 0.0
    night_at_min_ratio = 0.0
    night_at_max_ratio = 0.0
    if isinstance(by_group, pd.DataFrame) and not by_group.empty:
        day = by_group.loc[by_group["hour_group"] == "daytime"]
        night = by_group.loc[by_group["hour_group"] == "overnight"]
        if not day.empty:
            day_at_min_ratio = float(day["at_min_pct"].iloc[0]) / 100.0
            day_at_max_ratio = float(day["at_max_pct"].iloc[0]) / 100.0
        if not night.empty:
            night_at_min_ratio = float(night["at_min_pct"].iloc[0]) / 100.0
            night_at_max_ratio = float(night["at_max_pct"].iloc[0]) / 100.0

    bound_checks = {
        "max_day_at_min_ratio": float(cfg.max_day_at_min_ratio),
        "max_day_at_max_ratio": float(cfg.max_day_at_max_ratio),
        "max_night_at_min_ratio": float(cfg.max_night_at_min_ratio),
        "max_night_at_max_ratio": float(cfg.max_night_at_max_ratio),
        "actual_day_at_min_ratio": day_at_min_ratio,
        "actual_day_at_max_ratio": day_at_max_ratio,
        "actual_night_at_min_ratio": night_at_min_ratio,
        "actual_night_at_max_ratio": night_at_max_ratio,
    }
    bound_checks["ok_day_at_min"] = bound_checks["actual_day_at_min_ratio"] <= bound_checks["max_day_at_min_ratio"]
    bound_checks["ok_day_at_max"] = bound_checks["actual_day_at_max_ratio"] <= bound_checks["max_day_at_max_ratio"]
    bound_checks["ok_night_at_min"] = bound_checks["actual_night_at_min_ratio"] <= bound_checks["max_night_at_min_ratio"]
    bound_checks["ok_night_at_max"] = bound_checks["actual_night_at_max_ratio"] <= bound_checks["max_night_at_max_ratio"]
    # Fleet toi thieu can de van hanh route: max over hour sum_dir(trips * cycle_time/60).
    cycle_lookup = (
        state["cycle_times"][["route_id", "direction_id", "cycle_time_min"]]
        .assign(route_id=lambda d: d["route_id"].astype(str), direction_id=lambda d: d["direction_id"].astype(int))
    )
    cycle_map = {
        (str(r), int(d)): float(c)
        for r, d, c in cycle_lookup[["route_id", "direction_id", "cycle_time_min"]].to_numpy()
    }

    def _fleet_required(trips_arr: np.ndarray) -> float:
        rows = []
        for h in sorted(set(slot_hour.tolist())):
            mask_h = slot_hour == h
            req_h = 0.0
            for t_i, r_i, d_i in zip(trips_arr[mask_h], slot_route[mask_h], slot_dir[mask_h]):
                req_h += float(t_i) * cycle_map.get((str(r_i), int(d_i)), 90.0) / 60.0
            rows.append(req_h)
        return float(max(rows) if rows else 0.0)

    min_fleet_required = _fleet_required(trips)
    baseline_min_fleet_required = _fleet_required(baseline)
    route_station_ids: list[str] = []
    station_weights = assets["state"].get("station_rd_weights")
    if isinstance(station_weights, pd.DataFrame) and {"route", "station_complex_id"}.issubset(station_weights.columns):
        route_station_ids = sorted(
            station_weights.loc[station_weights["route"].astype(str) == str(cfg.route_id), "station_complex_id"]
            .astype(str)
            .unique()
            .tolist()
        )
    return {
        "config": cfg,
        "slot_route": slot_route,
        "slot_dir": slot_dir,
        "slot_hour": slot_hour,
        "demand": demand,
        "demand_df": station_or_route_demand_df,
        "baseline": baseline,
        "trips": trips,
        "trips_min": trips_min,
        "trips_max": trips_max,
        "metrics": metrics,
        "baseline_metrics": baseline_metrics,
        "bounds": bounds,
        "bound_checks": bound_checks,
        "lambda_knee": lambda_knee_live if lambda_knee_live else lambda_knee_baked,
        "lambda_knee_live": lambda_knee_live,
        "lambda_knee_baked": lambda_knee_baked,
        "lambda_slider": int(cfg.lambda_val),
        "lambda_used": int(lambda_used),
        "auto_knee": bool(cfg.auto_knee),
        "w_range": (float(cfg.w_low), float(cfg.w_high)),
        "lambda_sat": lambda_sat,
        "t_star": t_star,
        "pareto_df": pareto_df,
        "route_station_ids": route_station_ids,
        "min_fleet_required": min_fleet_required,
        "baseline_min_fleet_required": baseline_min_fleet_required,
    }
