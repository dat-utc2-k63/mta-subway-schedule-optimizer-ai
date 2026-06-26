# -*- coding: utf-8 -*-
"""Pipeline MTA: paths (Kaggle/local), preset thí nghiệm, helpers NN/ML — xem LITERATURE.md."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.ensemble import HistGradientBoostingRegressor
from tensorflow.keras import Input, Model, layers

# --- Literature-backed defaults (ITM 2026 MTA LSTM; weather mixed-effects) ---
DEFAULT_LAG_COLS = ["log_lag_24h", "log_lag_168h", "log_rolling_7d"]
SEASON_ORDER = ("winter", "spring", "summer", "autumn")


def configure_gpu(*, mixed_precision: bool = True, memory_growth: bool = True) -> dict[str, Any]:
    """Tối ưu GPU cho Keras/TF (Kaggle): mixed-precision + memory growth. Gọi một lần ở đầu notebook."""
    info: dict[str, Any] = {
        "gpus": [],
        "mixed_precision": False,
        "memory_growth": False,
        "policy": "float32",
    }
    try:
        gpus = tf.config.list_physical_devices("GPU")
    except Exception as exc:  # pragma: no cover
        info["error"] = f"list_physical_devices: {exc}"
        return info

    info["gpus"] = [getattr(g, "name", str(g)) for g in gpus]
    if not gpus:
        return info

    if memory_growth:
        for g in gpus:
            try:
                tf.config.experimental.set_memory_growth(g, True)
                info["memory_growth"] = True
            except Exception:  # pragma: no cover
                pass

    try:
        from tensorflow.keras import mixed_precision as _mp

        if mixed_precision:
            _mp.set_global_policy("mixed_float16")
            info["mixed_precision"] = True
            info["policy"] = "mixed_float16"
        else:
            _mp.set_global_policy("float32")
            info["policy"] = "float32"
    except Exception as exc:  # pragma: no cover
        info["error"] = f"policy: {exc}"
    return info


def season_from_month(month: int) -> str:
    """Mùa khí tượng Bắc bán cầu (phù hợp NYC): Đông 12–2, Xuân 3–5, Hè 6–8, Thu 9–11."""
    m = int(month)
    if m in (12, 1, 2):
        return "winter"
    if m in (3, 4, 5):
        return "spring"
    if m in (6, 7, 8):
        return "summer"
    return "autumn"


def season_year_label(month: int, year: int) -> tuple[str, int]:
    """(mùa, nhãn năm) cho CV: 4 khối ~đều nhau; đông gắn năm T1/T2.

    - winter_2024 = 12/2023 → 02/2024 (T12 thuộc đông năm sau)
    - winter_2025 = 12/2024 → 02/2025
    - xuân/hè/thu: nhãn năm = năm lịch của ngày (T3–T11).
    """
    m, y = int(month), int(year)
    if m in (3, 4, 5):
        return "spring", y
    if m in (6, 7, 8):
        return "summer", y
    if m in (9, 10, 11):
        return "autumn", y
    # winter: Dec → winter_{y+1}; Jan/Feb → winter_{y}
    if m == 12:
        return "winter", y + 1
    return "winter", y


def dates_for_season_year(
    dates: pd.Series | np.ndarray | list,
    season: str,
    season_year: int,
) -> set:
    """Tập ngày thuộc (mùa, season_year) theo quy tắc CV mùa×năm."""
    if isinstance(dates, set):
        dates = sorted(dates)
    d = pd.to_datetime(pd.Series(dates).dropna().unique())
    out: set = set()
    for dt in d:
        s, yr = season_year_label(int(dt.month), int(dt.year))
        if s == season and yr == int(season_year):
            out.add(dt)
    return out


def season_year_blocks_in_dates(
    dates: pd.Series | np.ndarray | list | set,
) -> list[tuple[str, int, int]]:
    """Các khối (mùa, season_year, n_days) có trong tập ngày."""
    if isinstance(dates, (set, frozenset)):
        dates = sorted(dates)
    d = pd.to_datetime(pd.Series(dates).dropna().unique())
    if len(d) == 0:
        return []
    meta = pd.DataFrame({"date": d})
    meta["month"] = meta["date"].dt.month
    meta["cal_year"] = meta["date"].dt.year
    sy = meta.apply(lambda r: season_year_label(int(r["month"]), int(r["cal_year"])), axis=1)
    meta["season"] = sy.map(lambda x: x[0])
    meta["season_year"] = sy.map(lambda x: int(x[1]))
    agg = meta.groupby(["season", "season_year"], as_index=False)["date"].count()
    blocks = [
        (str(row["season"]), int(row["season_year"]), int(row["date"]))
        for _, row in agg.iterrows()
    ]
    return sorted(
        blocks,
        key=lambda x: (x[1], SEASON_ORDER.index(x[0]) if x[0] in SEASON_ORDER else 99),
    )


def auto_pick_holdout_season_years(
    dates: pd.Series | np.ndarray | list,
    *,
    test_season: str = "autumn",
    test_season_year: int = 2025,
    val_season: str = "summer",
    val_season_year: int = 2025,
) -> tuple[str, int, str, int]:
    """Chọn test/val từ dữ liệu có sẵn; ưu tiên preset nếu khối tồn tại."""
    blocks = season_year_blocks_in_dates(dates)
    if len(blocks) < 2:
        raise ValueError("Cần ít nhất 2 khối mùa×năm trong dates để hold-out")
    by_key = {(s, y): n for s, y, n in blocks}
    test_key = (str(test_season), int(test_season_year))
    val_key = (str(val_season), int(val_season_year))
    if test_key not in by_key:
        test_key = (blocks[-1][0], blocks[-1][1])
    if val_key not in by_key or val_key == test_key:
        rest = [(s, y) for s, y, _ in blocks if (s, y) != test_key]
        if not rest:
            raise ValueError("Không tìm được khối val khác test")
        val_key = rest[-1]
    return test_key[0], test_key[1], val_key[0], val_key[1]


def build_seasonal_holdout_splits(
    dates: pd.Series | np.ndarray | list,
    *,
    test_season: str = "autumn",
    test_season_year: int = 2025,
    val_season: str = "summer",
    val_season_year: int = 2025,
    auto_adjust: bool = False,
    return_meta: bool = False,
) -> tuple[set, set, set] | tuple[set, set, set, dict[str, Any]]:
    """Hold-out theo mùa: test = một khối mùa; val = khối mùa khác; còn lại train."""
    all_dates = set(pd.to_datetime(pd.Series(dates).dropna().unique()))
    auto_adjusted = False
    test_dates = dates_for_season_year(all_dates, test_season, test_season_year)
    val_dates = dates_for_season_year(all_dates, val_season, val_season_year)

    need_adjust = (
        not test_dates
        or not val_dates
        or bool(test_dates & val_dates)
    )
    split_mode = "season_year"
    if need_adjust and auto_adjust:
        try:
            test_season, test_season_year, val_season, val_season_year = auto_pick_holdout_season_years(
                sorted(all_dates),
                test_season=test_season,
                test_season_year=test_season_year,
                val_season=val_season,
                val_season_year=val_season_year,
            )
            test_dates = dates_for_season_year(all_dates, test_season, test_season_year)
            val_dates = dates_for_season_year(all_dates, val_season, val_season_year)
            auto_adjusted = True
        except ValueError:
            sorted_d = sorted(all_dates)
            n = len(sorted_d)
            n_test = max(1, int(n * 0.15))
            n_val = max(1, int(n * 0.15))
            test_dates = set(sorted_d[-n_test:])
            val_dates = set(sorted_d[-(n_test + n_val) : -n_test])
            train_dates = set(sorted_d[: -(n_test + n_val)])
            if not train_dates:
                raise ValueError("Không đủ ngày cho train/val/test (cần > 3 ngày)") from None
            meta = {
                "test_season": "temporal_tail",
                "test_season_year": 0,
                "val_season": "temporal_mid",
                "val_season_year": 0,
                "auto_adjusted": True,
                "split_mode": "temporal",
                "n_train": len(train_dates),
                "n_val": len(val_dates),
                "n_test": len(test_dates),
            }
            if return_meta:
                return train_dates, val_dates, test_dates, meta
            return train_dates, val_dates, test_dates

    if not test_dates:
        raise ValueError(f"Không có ngày cho test {test_season}_{test_season_year}")
    if not val_dates:
        raise ValueError(f"Không có ngày cho val {val_season}_{val_season_year}")
    overlap = test_dates & val_dates
    if overlap:
        raise ValueError(f"val và test trùng {len(overlap)} ngày")
    train_dates = all_dates - test_dates - val_dates
    if not train_dates:
        raise ValueError("Không còn ngày cho train sau khi tách val/test theo mùa")

    meta = {
        "test_season": str(test_season),
        "test_season_year": int(test_season_year),
        "val_season": str(val_season),
        "val_season_year": int(val_season_year),
        "auto_adjusted": auto_adjusted,
        "split_mode": split_mode,
        "n_train": len(train_dates),
        "n_val": len(val_dates),
        "n_test": len(test_dates),
    }
    if return_meta:
        return train_dates, val_dates, test_dates, meta
    return train_dates, val_dates, test_dates


def tune_blend_weight_mae(
    pred_mlp: np.ndarray,
    pred_gbm: np.ndarray,
    log_baseline: np.ndarray,
    demand_true: np.ndarray,
    resid_clip: tuple[float, float],
    *,
    candidates: np.ndarray | list | None = None,
) -> tuple[float, float]:
    """Chọn trọng số MLP (blend) tối thiểu MAE demand."""
    from sklearn.metrics import mean_absolute_error

    cands = np.linspace(0.0, 1.0, 11) if candidates is None else np.asarray(candidates, dtype=float)
    best_w, best_mae = 0.5, float("inf")
    for w in cands:
        pr = blend_residual_predictions(pred_mlp, pred_gbm, w)
        pred_d = residuals_to_demand(log_baseline, pr, resid_clip)
        mae = mean_absolute_error(demand_true, pred_d)
        if mae < best_mae:
            best_mae, best_w = mae, float(w)
    return best_w, best_mae


def is_overnight_hour(hour: int) -> bool:
    """Giờ đêm (23–06): ràng buộc TRIPS riêng, tách khỏi ban ngày."""
    h = int(hour)
    return h <= 6 or h >= 23


def hour_slot_group(hour: int) -> str:
    """Nhóm giờ cho báo cáo bound: daytime / overnight."""
    return "overnight" if is_overnight_hour(hour) else "daytime"


def detect_demand_peaks(
    demand: np.ndarray,
    *,
    slot_route: np.ndarray | None = None,
    peak_percentile: float = 75.0,
    overnight_percentile: float = 20.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Phát hiện cao điểm ĐỘNG theo nhu cầu (demand-driven), không dựa vào khung giờ tĩnh.

    Quét mảng nhu cầu đã dịch pha (trục xuất bến):
      - Peak: demand ≥ phân vị `peak_percentile` (tính riêng từng route nếu có slot_route).
      - Overnight (cầu rất thấp): demand ≤ phân vị `overnight_percentile`.
      - Còn lại: off-peak.

    Trả về (is_peak, is_overnight) — mảng bool theo slot.
    """
    d = np.asarray(demand, dtype=float)
    n = len(d)
    is_peak = np.zeros(n, dtype=bool)
    is_overnight = np.zeros(n, dtype=bool)
    if n == 0:
        return is_peak, is_overnight

    def _flag(idx: np.ndarray) -> None:
        vals = d[idx]
        finite = vals[np.isfinite(vals)]
        if finite.size == 0:
            return
        hi = float(np.percentile(finite, float(peak_percentile)))
        lo = float(np.percentile(finite, float(overnight_percentile)))
        is_peak[idx] = vals >= max(hi, 1e-9)
        is_overnight[idx] = vals <= lo

    if slot_route is None:
        _flag(np.arange(n))
    else:
        routes = np.asarray(slot_route)
        for r in np.unique(routes):
            _flag(np.flatnonzero(routes == r))
    # overnight và peak loại trừ nhau (peak ưu tiên)
    is_overnight &= ~is_peak
    return is_peak, is_overnight


def build_dynamic_bounds(
    baseline_trips: np.ndarray,
    slot_hour: np.ndarray,
    *,
    demand: np.ndarray | None = None,
    slot_route: np.ndarray | None = None,
    peak_percentile: float = 75.0,
    overnight_percentile: float = 20.0,
    daytime_factor: float | None = None,
    peak_factor: float = 1.40,
    offpeak_factor: float = 1.15,
    overnight_factor: float = 1.05,
    min_factor: float = 0.5,
    overnight_min_factor: float | None = None,
    max_delta: int = 3,
    absolute_max: int = 60,
    min_trips: int = 2,
    return_daytime_mask: bool = False,
    return_peak_mask: bool = False,
) -> tuple[np.ndarray, np.ndarray] | tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-slot TRIPS_MIN / TRIPS_MAX: một ràng buộc ban ngày + overnight riêng.

    - Ban ngày (07–22): cùng `daytime_factor` cho mọi slot không thuộc đêm.
    - Đêm (23–06): `overnight_factor` / `overnight_min_factor`.
    - Nếu truyền `demand`: phát hiện overnight theo phân vị cầu (theo route).
      Không còn tách peak/off-peak ban ngày.
    - `peak_factor` / `offpeak_factor`: legacy — chỉ dùng khi `daytime_factor` là None
      và hai factor khác nhau (notebook v1).
    """
    base = np.asarray(baseline_trips, dtype=float)
    hrs = np.asarray(slot_hour, dtype=int)

    if demand is not None:
        _, is_overnight = detect_demand_peaks(
            demand,
            slot_route=slot_route,
            peak_percentile=peak_percentile,
            overnight_percentile=overnight_percentile,
        )
    else:
        is_overnight = np.array([is_overnight_hour(h) for h in hrs], dtype=bool)

    is_daytime = ~is_overnight
    if daytime_factor is not None:
        factors = np.where(is_overnight, float(overnight_factor), float(daytime_factor))
    else:
        is_peak = np.isin(hrs, (7, 8, 9, 17, 18, 19)) & is_daytime
        factors = np.full(len(base), float(offpeak_factor))
        factors[is_peak] = float(peak_factor)
        factors[is_overnight] = float(overnight_factor)

    ovn_min = float(overnight_min_factor if overnight_min_factor is not None else min_factor)
    min_factors = np.full(len(base), float(min_factor))
    min_factors[is_overnight] = ovn_min
    min_trips_arr = np.where(is_overnight, 1, int(min_trips))

    trips_min = np.maximum(min_trips_arr, np.floor(base * min_factors)).astype(int)
    trips_max = np.minimum(absolute_max, np.floor(base * factors)).astype(int)
    trips_max = np.maximum(trips_max, base.astype(int) + int(max_delta))
    if return_daytime_mask or return_peak_mask:
        return trips_min, trips_max, is_daytime
    return trips_min, trips_max


def report_bound_status(
    opt_trips: np.ndarray,
    trips_min: np.ndarray,
    trips_max: np.ndarray,
    slot_hour: np.ndarray,
    *,
    target_interior_pct: float = 60.0,
    target_daytime_interior_pct: float = 40.0,
    target_peak_interior_pct: float | None = None,
    verbose: bool = True,
    label: str = "",
) -> dict[str, Any]:
    """% slot ở min / max / interior — tổng và theo nhóm giờ; kiểm tra target."""
    if target_peak_interior_pct is not None:
        target_daytime_interior_pct = float(target_peak_interior_pct)
    t = np.asarray(opt_trips, dtype=int)
    tmin = np.asarray(trips_min, dtype=int)
    tmax = np.asarray(trips_max, dtype=int)
    at_min = float((t <= tmin).mean()) * 100
    at_max = float((t >= tmax).mean()) * 100
    interior = 100.0 - at_min - at_max

    by_group = bound_status_by_hour_groups(t, tmin, tmax, slot_hour)
    daytime_interior = float("nan")
    if not by_group.empty and (by_group["hour_group"] == "daytime").any():
        daytime_interior = float(
            by_group.loc[by_group["hour_group"] == "daytime", "interior_pct"].iloc[0]
        )

    ok_total = interior >= target_interior_pct
    ok_daytime = np.isnan(daytime_interior) or daytime_interior >= target_daytime_interior_pct

    if verbose:
        prefix = f"[{label}] " if label else ""
        print(
            f"  {prefix}at TRIPS_MIN={at_min:.0f}% | TRIPS_MAX={at_max:.0f}% | "
            f"interior={interior:.0f}%"
            f" {'✓' if ok_total else '✗'} (target>{target_interior_pct:.0f}%)"
        )
        for _, row in by_group.iterrows():
            mark = ""
            if row["hour_group"] == "daytime":
                mark = " ✓" if row["interior_pct"] >= target_daytime_interior_pct else " ✗"
            print(
                f"    {row['hour_group']:10s} n={int(row['n_slots']):4d} | "
                f"min={row['at_min_pct']:.0f}% max={row['at_max_pct']:.0f}% "
                f"interior={row['interior_pct']:.0f}%{mark}"
            )

    return {
        "at_min_pct": at_min,
        "at_max_pct": at_max,
        "interior_pct": interior,
        "daytime_interior_pct": daytime_interior,
        "peak_interior_pct": daytime_interior,  # legacy alias
        "ok_total_interior": ok_total,
        "ok_daytime_interior": ok_daytime,
        "ok_peak_interior": ok_daytime,  # legacy alias
        "by_hour_group": by_group,
    }


def bound_status_jsonable(status: dict[str, Any]) -> dict[str, Any]:
    """Bản JSON-safe của report_bound_status (bỏ DataFrame by_hour_group)."""
    return {k: v for k, v in status.items() if k != "by_hour_group"}


def json_default(obj: Any) -> Any:
    """default= cho json.dump — numpy/pandas → native Python."""
    if obj is None or isinstance(obj, (str, bool, int, float)):
        return obj
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, pd.DataFrame):
        return obj.to_dict(orient="records")
    if isinstance(obj, pd.Series):
        return {k: json_default(v) for k, v in obj.items()}
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


def dump_json(path: Path | str, obj: Any, **kwargs: Any) -> None:
    """Ghi dict/list ra JSON; tự xử lý numpy/pandas lồng nhau."""
    p = Path(path)
    with p.open("w", encoding="utf-8") as f:
        json.dump(obj, f, default=json_default, **kwargs)


def bound_status_by_hour_groups(
    trips: np.ndarray,
    trips_min: np.ndarray,
    trips_max: np.ndarray,
    hours: np.ndarray,
) -> pd.DataFrame:
    """% slot ở min / max / interior theo nhóm giờ (daytime, overnight)."""
    t = np.asarray(trips, dtype=int)
    tmin = np.asarray(trips_min, dtype=int)
    tmax = np.asarray(trips_max, dtype=int)
    hrs = np.asarray(hours, dtype=int)
    groups = np.array([hour_slot_group(h) for h in hrs])
    rows = []
    for g in ("daytime", "overnight"):
        m = groups == g
        if not m.any():
            continue
        tg = t[m]
        tmin_g = tmin[m]
        tmax_g = tmax[m]
        at_min = float((tg <= tmin_g).mean()) * 100
        at_max = float((tg >= tmax_g).mean()) * 100
        interior = 100.0 - at_min - at_max
        rows.append(
            dict(
                hour_group=g,
                n_slots=int(m.sum()),
                at_min_pct=at_min,
                at_max_pct=at_max,
                interior_pct=interior,
            )
        )
    return pd.DataFrame(rows)


# NOTE: Cơ chế "demand spillover" / "transfer matrix" cũ (route-level) đã được loại bỏ
# triệt để — kiến trúc mới dùng dự báo cấp ga + Departure-Time Projection thay thế.


def cycle_times_to_map(
    cycle_times: pd.DataFrame | dict[tuple[str, int], float] | None,
    *,
    default_cycle_min: float = 90.0,
) -> dict[tuple[str, int], float]:
    """Lookup (route_id, direction_id) → cycle_time_min."""
    if cycle_times is None:
        return {}
    if isinstance(cycle_times, dict):
        return {k: float(v) for k, v in cycle_times.items()}
    out: dict[tuple[str, int], float] = {}
    for _, row in cycle_times.iterrows():
        out[(str(row["route_id"]), int(row["direction_id"]))] = float(row["cycle_time_min"])
    if not out:
        return {}
    return out


def compute_vehicle_hours(
    trips: np.ndarray,
    *,
    slot_route: np.ndarray,
    slot_dir: np.ndarray,
    cycle_times: pd.DataFrame | dict[tuple[str, int], float] | None = None,
    default_cycle_min: float = 90.0,
) -> tuple[np.ndarray, float]:
    """VehicleHours_t = Trips_t × CycleTime_t / 60 per slot; returns (per-slot, total)."""
    t = np.maximum(np.nan_to_num(np.asarray(trips, dtype=float), nan=0.0), 0.0)
    routes = np.asarray(slot_route)
    dirs = np.asarray(slot_dir, dtype=int)
    ct_map = cycle_times_to_map(cycle_times, default_cycle_min=default_cycle_min)
    slot_vh = np.zeros(len(t), dtype=float)
    for i in range(len(t)):
        key = (str(routes[i]), int(dirs[i]))
        cycle = float(ct_map.get(key, default_cycle_min))
        slot_vh[i] = t[i] * cycle / 60.0
    return slot_vh, float(slot_vh.sum())


def compute_fleet_utilization(
    total_vehicle_hours: float,
    system_fleet_capacity: float,
    *,
    hours_in_day: float = 24.0,
) -> float:
    """Fleet utilization = total VH / (peak system fleet × hours in day)."""
    denom = max(float(system_fleet_capacity) * float(hours_in_day), 1e-9)
    return float(total_vehicle_hours) / denom


def compute_schedule_metrics(
    demand: np.ndarray,
    trips: np.ndarray,
    *,
    slot_route: np.ndarray,
    slot_dir: np.ndarray,
    slot_hour: np.ndarray,
    capacity_per_trip: float = 1200.0,
    cycle_times: pd.DataFrame | dict[tuple[str, int], float] | None = None,
    default_cycle_min: float = 90.0,
    cost_per_vehicle_hour: float | None = None,
    system_fleet_capacity: float | None = None,
) -> dict[str, Any]:
    """Queue-based passenger wait + vehicle-hour operating cost (no weighted-sum objective).

    Queue propagates between consecutive hours of the same route-direction:
      Queue_t = max(0, Queue_{t-1} + Demand_t - Capacity_t)
      Served_t = min(Demand_t + Queue_{t-1}, Capacity_t)
      PassengerWait_t = Queue_{t-1} × Headway_t + Served_t × Headway_t / 2
    """
    d = np.clip(np.nan_to_num(np.asarray(demand, dtype=float), nan=0.0), 0.0, None)
    t = np.maximum(np.nan_to_num(np.asarray(trips, dtype=float), nan=1.0), 1.0)
    routes = np.asarray(slot_route)
    dirs = np.asarray(slot_dir, dtype=int)
    hours = np.asarray(slot_hour, dtype=int)
    n = len(d)
    cap_pt = float(capacity_per_trip)

    slot_wait = np.zeros(n, dtype=float)
    queue_out = np.zeros(n, dtype=float)
    queue_prev_arr = np.zeros(n, dtype=float)
    served_arr = np.zeros(n, dtype=float)

    groups: dict[tuple[str, int], list[int]] = {}
    for i in range(n):
        key = (str(routes[i]), int(dirs[i]))
        groups.setdefault(key, []).append(i)

    max_queue = 0.0
    for indices in groups.values():
        queue_prev = 0.0
        for i in sorted(indices, key=lambda idx: int(hours[idx])):
            capacity = t[i] * cap_pt
            demand_i = d[i]
            served = min(demand_i + queue_prev, capacity)
            headway = 60.0 / t[i]

            queue_prev_arr[i] = queue_prev
            served_arr[i] = served
            slot_wait[i] = queue_prev * headway + served * headway / 2.0
            queue_out[i] = max(0.0, queue_prev + demand_i - capacity)
            max_queue = max(max_queue, queue_out[i])
            queue_prev = queue_out[i]

    slot_vh, total_vh = compute_vehicle_hours(
        t, slot_route=routes, slot_dir=dirs,
        cycle_times=cycle_times, default_cycle_min=default_cycle_min,
    )
    total_wait = float(slot_wait.sum())
    total_demand = float(d.sum())
    overflow_slots = int((queue_out > 0).sum())
    fleet_cost = float(total_vh * cost_per_vehicle_hour) if cost_per_vehicle_hour is not None else None

    out: dict[str, Any] = dict(
        total_passenger_min_wait=total_wait,
        weighted_avg_wait_min=total_wait / max(total_demand, 1e-9),
        overflow_pct=float(overflow_slots / max(n, 1)) * 100.0,
        total_overflow_pax=float(queue_out.sum()),
        max_queue_length=float(max_queue),
        total_vehicle_hours=float(total_vh),
        total_trips=float(t.sum()),
        slot_wait=slot_wait,
        slot_vehicle_hours=slot_vh,
        overflow_out=queue_out,
        queue_out=queue_out,
        overflow_prev=queue_prev_arr,
        queue_prev=queue_prev_arr,
        served=served_arr,
        served_new=served_arr,
    )
    if fleet_cost is not None:
        out["fleet_cost"] = fleet_cost
        out["cost_per_vehicle_hour"] = float(cost_per_vehicle_hour)
    if system_fleet_capacity is not None and float(system_fleet_capacity) > 0:
        out["system_fleet_capacity"] = float(system_fleet_capacity)
        out["fleet_utilization"] = compute_fleet_utilization(
            total_vh, float(system_fleet_capacity)
        )
    return out


def compute_wait_with_overflow(
    demand: np.ndarray,
    trips: np.ndarray,
    *,
    slot_route: np.ndarray,
    slot_dir: np.ndarray,
    slot_hour: np.ndarray,
    capacity_per_trip: float = 1200.0,
    lambda_cost: float = 150.0,
    cycle_times: pd.DataFrame | dict[tuple[str, int], float] | None = None,
    default_cycle_min: float = 90.0,
) -> dict[str, Any]:
    """Backward-compatible alias → compute_schedule_metrics (queue model + vehicle hours)."""
    m = compute_schedule_metrics(
        demand, trips,
        slot_route=slot_route, slot_dir=slot_dir, slot_hour=slot_hour,
        capacity_per_trip=capacity_per_trip,
        cycle_times=cycle_times, default_cycle_min=default_cycle_min,
    )
    m["total_fleet_cost"] = m["total_vehicle_hours"]
    m["lambda_cost"] = float(lambda_cost)
    return m


def analytical_trips_per_slot(
    demand: np.ndarray,
    lambda_tradeoff: float,
    cycle_time_min: np.ndarray,
    *,
    trips_min: np.ndarray,
    trips_max: np.ndarray,
) -> np.ndarray:
    """Pareto scan helper: min D·30/t + λ·(t·cycle/60) → t* = sqrt(1800·D/(λ·cycle))."""
    lam = max(float(lambda_tradeoff), 1e-9)
    d = np.maximum(np.asarray(demand, dtype=float), 1e-9)
    cyc = np.maximum(np.asarray(cycle_time_min, dtype=float), 1.0)
    tmin = np.asarray(trips_min, dtype=float)
    tmax = np.asarray(trips_max, dtype=float)
    trips_star = np.sqrt(1800.0 * d / (lam * cyc))
    return np.clip(np.round(trips_star), tmin, tmax).astype(int)


def wait_totals_by_hour_groups(
    slot_wait: np.ndarray,
    slot_hour: np.ndarray,
) -> pd.DataFrame:
    """Tổng passenger-min wait theo nhóm giờ (daytime / overnight)."""
    sw = np.asarray(slot_wait, dtype=float)
    hrs = np.asarray(slot_hour, dtype=int)
    groups = np.array([hour_slot_group(int(h)) for h in hrs])
    rows = []
    for g in ("daytime", "overnight"):
        m = groups == g
        if not m.any():
            continue
        rows.append(dict(hour_group=g, n_slots=int(m.sum()), total_wait=float(sw[m].sum())))
    return pd.DataFrame(rows)


def chebyshev_scalarize(
    f1: float,
    f2: float,
    w: float,
    z_star: tuple[float, float],
) -> float:
    """Weighted Chebyshev: max(w·(f1−f1*), (1−w)·(f2−f2*)); w∈[0,1] trọng số f1."""
    z1, z2 = float(z_star[0]), float(z_star[1])
    return max(float(w) * (float(f1) - z1), (1.0 - float(w)) * (float(f2) - z2))


def filter_nondominated(
    df: pd.DataFrame,
    *,
    f1_col: str = "f1",
    f2_col: str = "f2",
) -> pd.DataFrame:
    """Giữ các điểm không bị dominate (minimize cả f1 và f2)."""
    if df.empty:
        return df.copy()
    rows = df.to_dict("records")
    keep: list[dict] = []
    for i, a in enumerate(rows):
        dominated = False
        for j, b in enumerate(rows):
            if i == j:
                continue
            if b[f1_col] <= a[f1_col] and b[f2_col] <= a[f2_col]:
                if b[f1_col] < a[f1_col] or b[f2_col] < a[f2_col]:
                    dominated = True
                    break
        if not dominated:
            keep.append(a)
    out = pd.DataFrame(keep)
    if out.empty:
        return df.copy()
    return out.reset_index(drop=True)


def find_knee_point(
    pareto_df: pd.DataFrame,
    *,
    f1_col: str = "f1",
    f2_col: str = "f2",
    method: str = "curvature",
    z_star: tuple[float, float] | None = None,
) -> pd.Series:
    """Chọn knee trên frontier: max curvature (mặc định) hoặc min khoảng cách tới utopia."""
    df = pareto_df.sort_values(f1_col).reset_index(drop=True)
    if len(df) == 0:
        raise ValueError("pareto_df rỗng")
    if len(df) == 1:
        return df.iloc[0]

    f1 = df[f1_col].to_numpy(dtype=float)
    f2 = df[f2_col].to_numpy(dtype=float)
    f1n = (f1 - f1.min()) / max(f1.max() - f1.min(), 1e-9)
    f2n = (f2 - f2.min()) / max(f2.max() - f2.min(), 1e-9)

    if method == "utopia":
        if z_star is None:
            z1n, z2n = 0.0, 0.0
        else:
            z1n = (float(z_star[0]) - f1.min()) / max(f1.max() - f1.min(), 1e-9)
            z2n = (float(z_star[1]) - f2.min()) / max(f2.max() - f2.min(), 1e-9)
        dist = np.sqrt((f1n - z1n) ** 2 + (f2n - z2n) ** 2)
        return df.iloc[int(np.argmin(dist))]

    dx = np.gradient(f1n)
    dy = np.gradient(f2n)
    ddx = np.gradient(dx)
    ddy = np.gradient(dy)
    curv = np.abs(dx * ddy - dy * ddx) / (dx**2 + dy**2 + 1e-12) ** 1.5
    curv = np.nan_to_num(curv, nan=0.0, posinf=0.0, neginf=0.0)
    if len(curv) > 2:
        curv[0] = 0.0
        curv[-1] = 0.0
    idx = int(np.argmax(curv))
    return df.iloc[idx]


def generate_pareto_frontier(
    scenario_demand: dict[str, np.ndarray],
    baseline_trips: np.ndarray,
    optimize_fn,
    evaluate_fn,
    *,
    scenario: str = "weekday_peak",
    n_points: int = 20,
    lambda_eval: float = 150.0,
    lambda_scan: np.ndarray | list[float] | None = None,
    w_range: tuple[float, float] = (0.1, 0.9),
) -> pd.DataFrame:
    """Sample Pareto frontier bằng Chebyshev scalarization; quét λ rồi lọc non-dominated."""
    if scenario not in scenario_demand:
        raise KeyError(f"scenario {scenario!r} không có trong scenario_demand")
    demand = np.asarray(scenario_demand[scenario], dtype=float)
    base = np.asarray(baseline_trips, dtype=float)
    lam_eval = float(lambda_eval)

    if lambda_scan is None:
        lambda_scan = np.unique(
            np.round(np.geomspace(20.0, 3000.0, 80)).astype(float)
        )
    else:
        lambda_scan = np.asarray(lambda_scan, dtype=float)

    base_m = evaluate_fn(base, demand, lam_eval)
    base_f1 = float(base_m["total_passenger_min_wait"])
    base_f2 = float(base_m.get("total_vehicle_hours", base_m.get("total_fleet_cost", 0.0)))

    candidates: list[dict[str, float]] = []
    for lam in lambda_scan:
        sol = np.asarray(optimize_fn(demand, float(lam)), dtype=int)
        m = evaluate_fn(sol, demand, lam_eval)
        f2 = float(m.get("total_vehicle_hours", m.get("total_fleet_cost", 0.0)))
        candidates.append(
            dict(
                lambda_equiv=float(lam),
                f1=float(m["total_passenger_min_wait"]),
                f2=f2,
                total_trips=float(m["total_trips"]),
                total_vehicle_hours=f2,
            )
        )
    cand_df = pd.DataFrame(candidates)
    z_star = (
        float(cand_df["f1"].min()),
        float(cand_df["f2"].min()),
    )

    weights = np.linspace(float(w_range[0]), float(w_range[1]), int(n_points))
    rows: list[dict[str, float]] = []
    for w in weights:
        best_score = float("inf")
        best_row: dict[str, float] | None = None
        for row in candidates:
            score = chebyshev_scalarize(row["f1"], row["f2"], float(w), z_star)
            if score < best_score:
                best_score = score
                best_row = {**row, "w": float(w), "chebyshev": score}
        if best_row is not None:
            rows.append(best_row)

    raw_df = pd.DataFrame(rows).drop_duplicates(
        subset=["f1", "f2", "lambda_equiv"], keep="first"
    )
    pareto_df = filter_nondominated(raw_df)
    pareto_df = pareto_df.sort_values("f1").reset_index(drop=True)
    pareto_df["f1_improve_pct"] = (base_f1 - pareto_df["f1"]) / max(base_f1, 1e-9) * 100.0
    pareto_df["f2_delta_pct"] = (pareto_df["f2"] - base_f2) / max(base_f2, 1e-9) * 100.0
    pareto_df["vehicle_hours_delta_pct"] = pareto_df["f2_delta_pct"]
    pareto_df["z_star_f1"] = z_star[0]
    pareto_df["z_star_f2"] = z_star[1]
    return pareto_df


def plot_pareto_frontier(
    pareto_df: pd.DataFrame,
    baseline_f1: float,
    baseline_f2: float,
    *,
    ax: Any | None = None,
    title: str = "Pareto frontier: vehicle-hours vs passenger-min wait",
    save_path: Path | str | None = None,
    annotate_all: bool = True,
) -> Any:
    """Scatter f2 vs f1; đánh dấu baseline, balanced (w≈0.5), knee."""
    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots(figsize=(8, 6))

    ax.scatter(
        pareto_df["f2"],
        pareto_df["f1"],
        c=pareto_df["w"],
        cmap="viridis",
        s=60,
        edgecolors="k",
        linewidths=0.4,
        zorder=3,
        label="Pareto",
    )
    ax.scatter(
        [baseline_f2],
        [baseline_f1],
        s=140,
        marker="s",
        c="grey",
        edgecolors="k",
        zorder=4,
        label="Baseline",
    )

    if "w" in pareto_df.columns and len(pareto_df) > 0:
        balanced = pareto_df.loc[(pareto_df["w"] - 0.5).abs().idxmin()]
        ax.scatter(
            [balanced["f2"]],
            [balanced["f1"]],
            s=160,
            marker="D",
            c="orange",
            edgecolors="k",
            zorder=5,
            label=f"Balanced (w={balanced['w']:.2f})",
        )

    knee = find_knee_point(pareto_df)
    ax.scatter(
        [knee["f2"]],
        [knee["f1"]],
        s=200,
        marker="*",
        c="crimson",
        edgecolors="k",
        zorder=6,
        label=f"Knee (λ={knee['lambda_equiv']:.0f})",
    )

    if annotate_all:
        for _, row in pareto_df.iterrows():
            ax.annotate(
                f"λ={row['lambda_equiv']:.0f}\n{row.get('total_vehicle_hours', row['f2']):.0f} vh",
                (row["f2"], row["f1"]),
                textcoords="offset points",
                xytext=(4, 4),
                fontsize=6,
                alpha=0.85,
            )

    ax.set_xlabel("f2: total vehicle-hours")
    ax.set_ylabel("f1: passenger-min wait")
    ax.set_title(title)
    ax.legend(fontsize=8, loc="best")
    ax.grid(True, alpha=0.3)

    if save_path is not None:
        fig = ax.figure
        fig.tight_layout()
        fig.savefig(save_path, dpi=120, bbox_inches="tight")

    return ax


def build_season_year_cv_folds(
    dates: pd.Series | np.ndarray | list,
    *,
    n_years: int = 2,
) -> list[tuple[set, set, str]]:
    """8-fold (4 mùa × n_years): mỗi fold = một (mùa, season_year) làm test."""
    d = pd.to_datetime(pd.Series(dates).dropna().unique())
    meta = pd.DataFrame({"date": d})
    meta["month"] = meta["date"].dt.month
    meta["cal_year"] = meta["date"].dt.year
    sy = meta.apply(lambda r: season_year_label(r["month"], r["cal_year"]), axis=1)
    meta["season"] = sy.map(lambda x: x[0])
    meta["season_year"] = sy.map(lambda x: x[1])

    years = sorted(meta["season_year"].unique())
    if len(years) > n_years:
        years = years[-n_years:]

    folds: list[tuple[set, set, str]] = []
    all_dates = set(meta["date"])
    for yr in years:
        for season in SEASON_ORDER:
            test_mask = (meta["season_year"] == yr) & (meta["season"] == season)
            if not test_mask.any():
                continue
            test_dates = set(meta.loc[test_mask, "date"])
            train_dates = all_dates - test_dates
            if not train_dates:
                continue
            label = f"{season}_{int(yr)}"
            folds.append((train_dates, test_dates, label))

    return folds


def derive_operating_hours_by_route(
    headway: pd.DataFrame,
    route_ids: list[str],
    *,
    directions: list[int] | None = None,
    min_trips: int = 1,
) -> dict[str, list[int]]:
    """Giờ khởi hành GTFS theo từng tuyến (full-network scope)."""
    return {
        str(r).strip(): derive_route_operating_hours(
            headway, r, directions=directions, min_trips=min_trips
        )
        for r in route_ids
    }


def derive_route_operating_hours(
    headway: pd.DataFrame,
    route_id: str,
    *,
    directions: list[int] | None = None,
    min_trips: int = 1,
) -> list[int]:
    """Giờ có ít nhất một chuyến khởi hành GTFS (tổng trip_count theo giờ)."""
    sub = headway.loc[headway["route_id"].astype(str) == str(route_id).strip()].copy()
    if directions is not None:
        sub = sub.loc[sub["direction_id"].isin(directions)]
    if sub.empty:
        return []
    agg = sub.groupby("hour", as_index=False)["trip_count"].sum()
    hours = agg.loc[agg["trip_count"] >= min_trips, "hour"].astype(int)
    return sorted(hours.tolist())


def derive_operating_hours_by_route_direction(
    headway: pd.DataFrame,
    route_ids: list[str],
    *,
    directions: list[int] | None = None,
    min_trips: int = 1,
) -> dict[tuple[str, int], list[int]]:
    """Giờ hoạt động theo (route, direction) — chính xác hơn gộp theo route."""
    dirs = directions
    if dirs is None:
        dirs = sorted(headway["direction_id"].dropna().unique().astype(int).tolist())
    out: dict[tuple[str, int], list[int]] = {}
    for rid in route_ids:
        rs = str(rid).strip()
        for d in dirs:
            sub = headway.loc[
                (headway["route_id"].astype(str) == rs) & (headway["direction_id"] == int(d))
            ]
            if sub.empty:
                continue
            agg = sub.groupby("hour")["trip_count"].sum()
            hours = sorted(agg.loc[agg >= min_trips].index.astype(int).tolist())
            if hours:
                out[(rs, int(d))] = hours
    return out


def gtfs_time_to_minutes(t: str) -> float:
    """GTFS departure_time HH:MM:SS (có thể >24h)."""
    if not isinstance(t, str) or ":" not in t:
        return float("nan")
    h, m, s = t.split(":")
    return int(h) * 60 + int(m) + int(s) / 60.0


def build_headway_from_gtfs(
    schedule_dir: Path,
    service_id: str = "Weekday",
) -> pd.DataFrame:
    """Dựng baseline headway (route × direction × hour) từ schedule_current/*.txt."""
    schedule_dir = Path(schedule_dir)
    trips = pd.read_csv(
        schedule_dir / "trips.txt",
        dtype={"trip_id": str, "route_id": str, "service_id": str},
    )
    stop_times = pd.read_csv(
        schedule_dir / "stop_times.txt",
        dtype={"trip_id": str, "stop_id": str, "departure_time": str, "arrival_time": str},
        usecols=lambda c: c in {"trip_id", "stop_sequence", "departure_time", "arrival_time"},
    )

    cal_path = schedule_dir / "calendar.txt"
    available_services = trips["service_id"].dropna().unique().tolist()
    if cal_path.exists():
        cal = pd.read_csv(cal_path, dtype={"service_id": str})
        available_services = cal["service_id"].dropna().unique().tolist() or available_services
    if service_id not in available_services and available_services:
        service_id = available_services[0]

    trips_f = trips[trips["service_id"] == service_id].copy()
    if trips_f.empty:
        trips_f = trips.copy()

    if "direction_id" not in trips_f.columns:
        trips_f["direction_id"] = 0
    trips_f["direction_id"] = (
        pd.to_numeric(trips_f["direction_id"], errors="coerce").fillna(0).astype(int)
    )

    first_stop = (
        stop_times.sort_values(["trip_id", "stop_sequence"])
        .groupby("trip_id", as_index=False)
        .first()[["trip_id", "departure_time", "arrival_time"]]
    )
    first_stop["dep_str"] = first_stop["departure_time"].fillna(first_stop["arrival_time"])
    first_stop["dep_min"] = first_stop["dep_str"].map(gtfs_time_to_minutes)
    first_stop = first_stop.dropna(subset=["dep_min"])

    merged = trips_f[["trip_id", "route_id", "direction_id"]].merge(
        first_stop[["trip_id", "dep_min"]], on="trip_id", how="inner"
    )
    merged["hour"] = (merged["dep_min"] // 60).astype(int) % 24
    merged["route_id"] = merged["route_id"].astype(str)

    rows = []
    for (route, direction, hour), g in merged.groupby(["route_id", "direction_id", "hour"]):
        times = np.sort(g["dep_min"].to_numpy() % (24 * 60))
        n = int(len(times))
        if n <= 0:
            continue
        avg_hw = 60.0 / n
        if n >= 2:
            deltas = np.diff(times)
            deltas = deltas[deltas > 0]
            min_hw = float(np.min(deltas)) if deltas.size else avg_hw
        else:
            min_hw = avg_hw
        rows.append({
            "route_id": route,
            "direction_id": int(direction),
            "hour": int(hour),
            "trip_count": n,
            "avg_headway_min": float(avg_hw),
            "min_headway_min": float(min_hw),
        })

    hw = pd.DataFrame(rows).sort_values(["route_id", "direction_id", "hour"]).reset_index(drop=True)
    hw.attrs["service_id"] = service_id
    return hw


def build_route_hour_trip_counts(
    headway: pd.DataFrame,
    *,
    min_trips: int = 1,
) -> pd.DataFrame:
    """Tổng trip_count theo (route_id, hour); chỉ giờ có ≥ min_trips chuyến."""
    agg = (
        headway.assign(route_id=headway["route_id"].astype(str))
        .groupby(["route_id", "hour"], as_index=False)["trip_count"]
        .sum()
    )
    agg = agg.loc[agg["trip_count"] >= int(min_trips)].copy()
    agg["hour"] = agg["hour"].astype(int)
    return agg


def build_station_route_hour_weights(
    station_to_routes: pd.DataFrame,
    route_hour_trips: pd.DataFrame,
    *,
    min_trips: int = 1,
    route_aliases: dict[str, str] | None = None,
) -> pd.DataFrame:
    """Trọng số phân bổ ridership ga→tuyến theo số chuyến GTFS trong từng giờ.

    weight_station(s,h) = Σ trips(r,h) cho mọi route r đi qua ga s và hoạt động giờ h.
    alloc_weight(r,s,h) = trips(r,h) / weight_station(s,h)
    """
    aliases = route_aliases or {}
    sr = station_to_routes[["station_complex_id", "route"]].drop_duplicates().copy()
    sr["route"] = sr["route"].astype(str).replace(aliases)
    sr["station_complex_id"] = sr["station_complex_id"].astype(str)

    rht = route_hour_trips.copy()
    rht["route_id"] = rht["route_id"].astype(str)

    merged = sr.merge(rht, left_on="route", right_on="route_id", how="inner")
    merged = merged.loc[merged["trip_count"] >= int(min_trips)].copy()
    if merged.empty:
        raise ValueError("Không có trọng số ga×tuyến×giờ từ GTFS")

    station_totals = (
        merged.groupby(["station_complex_id", "hour"], as_index=False)["trip_count"]
        .sum()
        .rename(columns={"trip_count": "weight_station"})
    )
    merged = merged.merge(station_totals, on=["station_complex_id", "hour"])
    merged["alloc_weight"] = merged["trip_count"] / merged["weight_station"]
    return merged[
        ["station_complex_id", "route", "hour", "trip_count", "weight_station", "alloc_weight"]
    ]


def aggregate_ridership_to_routes(
    ridership_station: pd.DataFrame,
    station_route_weights: pd.DataFrame,
    *,
    coverage_thr: float = 0.7,
) -> pd.DataFrame:
    """[DEPRECATED] Gộp ridership ga→tuyến (legacy route-level forecasting).

    Kiến trúc mới dự báo trực tiếp ở cấp GA (station-level) rồi dùng
    `project_station_demand_to_departures` để chiếu ngược về trục xuất bến.
    Hàm này chỉ giữ lại cho EDA/đối chiếu, không còn nằm trên luồng chính.
    """
    rs = ridership_station.copy()
    rs["station_complex_id"] = rs["station_complex_id"].astype(str)
    rs["hour"] = rs["hour"].astype(int)

    routed = rs.merge(
        station_route_weights[["station_complex_id", "route", "hour", "alloc_weight"]],
        on=["station_complex_id", "hour"],
        how="inner",
    )
    routed["ridership_route"] = routed["ridership"] * routed["alloc_weight"]

    route_stations_hour = (
        station_route_weights.groupby(["route", "hour"])["station_complex_id"]
        .nunique()
        .rename("n_stations_route")
        .reset_index()
    )

    agg = routed.groupby(["route", "date", "hour"]).agg(
        demand_observed=("ridership_route", "sum"),
        n_active=("station_complex_id", "nunique"),
    ).reset_index()
    agg = agg.merge(route_stations_hour, on=["route", "hour"], how="left")
    agg["n_stations_route"] = agg["n_stations_route"].fillna(1).clip(lower=1)
    agg["coverage"] = agg["n_active"] / agg["n_stations_route"]
    agg = agg.loc[agg["coverage"] >= float(coverage_thr)].copy()
    agg["demand"] = agg["demand_observed"] / agg["coverage"]

    out = agg[["route", "date", "hour", "demand", "coverage"]].rename(columns={"route": "route_id"})
    out["date"] = pd.to_datetime(out["date"])
    return out


# =====================================================================================
# STATION-LEVEL DEMAND FORECASTING + DEPARTURE-TIME PROJECTION (Không – Thời gian)
# -------------------------------------------------------------------------------------
# Luồng nhân quả vật lý:
#   1) Dự báo lượng quẹt thẻ vào TỪNG GA theo giờ  D_station(s, h_board).
#   2) Ma trận trọng số tần suất GTFS phân bổ cầu ga về cặp (Tuyến, Hướng):
#         D_rd(r, dir, s, h) = D_station(s, h) · w_freq(s, r, dir, h)
#   3) Chiếu ngược thời gian (khấu trừ T_travel từ bến đầu → ga s):
#         t0 = t_board − T_travel(r, dir, s)  → dồn cầu về trục xuất bến.
# =====================================================================================


def _haversine_km(lat1: np.ndarray, lon1: np.ndarray, lat2: np.ndarray, lon2: np.ndarray) -> np.ndarray:
    """Khoảng cách great-circle (km) — vector hoá NumPy."""
    r = 6371.0088
    p1 = np.radians(np.asarray(lat1, dtype=float))
    p2 = np.radians(np.asarray(lat2, dtype=float))
    dphi = p2 - p1
    dlmb = np.radians(np.asarray(lon2, dtype=float) - np.asarray(lon1, dtype=float))
    a = np.sin(dphi / 2.0) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dlmb / 2.0) ** 2
    return 2.0 * r * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))


def build_station_stop_map(
    station_coords: pd.DataFrame,
    stops: pd.DataFrame,
    *,
    station_id_col: str = "station_complex_id",
    lat_col: str = "latitude",
    lon_col: str = "longitude",
    max_dist_km: float = 1.5,
) -> pd.DataFrame:
    """Ánh xạ station_complex_id → parent_stop_id GTFS gần nhất (nearest neighbour theo toạ độ).

    Dữ liệu ridership không có stop_id GTFS; ta nối qua kinh/vĩ độ để lấy được
    `parent_stop_id` (cùng convention với `load_gtfs_stop_offset_templates`) — phục vụ
    tra cứu ma trận thời gian hành trình T_travel.

    Trả về: station_complex_id, parent_stop_id, stop_name, dist_km.
    """
    sc = station_coords[[station_id_col, lat_col, lon_col]].dropna().copy()
    sc[station_id_col] = sc[station_id_col].astype(str)
    sc = sc.drop_duplicates(subset=[station_id_col])

    st = stops.copy()
    st["stop_lat"] = pd.to_numeric(st.get("stop_lat"), errors="coerce")
    st["stop_lon"] = pd.to_numeric(st.get("stop_lon"), errors="coerce")
    st = st.dropna(subset=["stop_lat", "stop_lon"])
    parent = st.set_index("stop_id").get("parent_station")
    parent_map = parent.to_dict() if parent is not None else {}
    name_map = st.set_index("stop_id").get("stop_name", pd.Series(dtype=str)).to_dict()

    stop_lat = st["stop_lat"].to_numpy()
    stop_lon = st["stop_lon"].to_numpy()
    stop_ids = st["stop_id"].astype(str).to_numpy()

    rows: list[dict[str, Any]] = []
    for _, s in sc.iterrows():
        d = _haversine_km(float(s[lat_col]), float(s[lon_col]), stop_lat, stop_lon)
        j = int(np.argmin(d))
        dist = float(d[j])
        if dist > float(max_dist_km):
            continue
        sid = stop_ids[j]
        pst = parent_map.get(sid)
        parent_stop_id = str(pst) if isinstance(pst, str) and pst and pst != "nan" else str(sid)
        rows.append(
            dict(
                station_complex_id=str(s[station_id_col]),
                parent_stop_id=parent_stop_id,
                stop_name=str(name_map.get(sid, sid)),
                dist_km=round(dist, 4),
            )
        )
    return pd.DataFrame(rows)


def build_station_route_direction_weights(
    station_to_routes: pd.DataFrame,
    headway: pd.DataFrame,
    *,
    min_trips: int = 1,
    route_aliases: dict[str, str] | None = None,
) -> pd.DataFrame:
    """Ma trận trọng số tần suất: phân bổ cầu ga → cặp (route, direction) theo số chuyến GTFS.

    w_freq(s, r, dir, h) = trips(r, dir, h) / Σ_{r',dir' phục vụ ga s tại giờ h} trips(r', dir', h)

    Σ trọng số trên mỗi (station, hour) = 1 → bảo toàn tổng lượng cầu ga khi phân bổ.
    Đây là bản nâng cấp của `build_station_route_hour_weights` (thêm chiều direction).
    """
    aliases = route_aliases or {}
    sr = station_to_routes[["station_complex_id", "route"]].drop_duplicates().copy()
    sr["route"] = sr["route"].astype(str).replace(aliases)
    sr["station_complex_id"] = sr["station_complex_id"].astype(str)

    hw = headway.copy()
    hw["route_id"] = hw["route_id"].astype(str)
    hw["direction_id"] = pd.to_numeric(hw["direction_id"], errors="coerce").fillna(0).astype(int)
    hw["hour"] = hw["hour"].astype(int)
    hw = hw.loc[hw["trip_count"] >= int(min_trips), ["route_id", "direction_id", "hour", "trip_count"]]

    merged = sr.merge(hw, left_on="route", right_on="route_id", how="inner")
    if merged.empty:
        raise ValueError("Không dựng được trọng số ga×route×direction×giờ từ GTFS headway")

    station_totals = (
        merged.groupby(["station_complex_id", "hour"], as_index=False)["trip_count"]
        .sum()
        .rename(columns={"trip_count": "weight_station"})
    )
    merged = merged.merge(station_totals, on=["station_complex_id", "hour"])
    merged["alloc_weight"] = merged["trip_count"] / merged["weight_station"].clip(lower=1e-9)
    return merged[
        ["station_complex_id", "route", "direction_id", "hour", "trip_count", "weight_station", "alloc_weight"]
    ]


def build_station_travel_offsets(
    offset_templates: pd.DataFrame,
    station_stop_map: pd.DataFrame,
    *,
    run_delay_min: float = 0.0,
) -> pd.DataFrame:
    """Ma trận thời gian hành trình T_travel(route, dir, station) từ bến đầu → ga.

    `offset_templates` (từ `load_gtfs_stop_offset_templates`) cho median offset (phút) của
    từng stop so với departure đầu trip — chính là run-time lịch sử theo GTFS. Nối với
    `station_stop_map` (station_complex_id ↔ parent_stop_id) để quy về cấp ga.

    `run_delay_min`: đệm trễ chạy tàu lịch sử cộng thêm (mặc định 0 — GTFS offset đã gồm run-time).
    Trả về: station_complex_id, route_id, direction_id, stop_sequence, travel_offset_min.
    """
    ot = offset_templates.copy()
    ot["route_id"] = ot["route_id"].astype(str)
    ot["direction_id"] = pd.to_numeric(ot["direction_id"], errors="coerce").fillna(0).astype(int)
    ot["parent_stop_id"] = ot["parent_stop_id"].astype(str)

    sm = station_stop_map[["station_complex_id", "parent_stop_id"]].copy()
    sm["station_complex_id"] = sm["station_complex_id"].astype(str)
    sm["parent_stop_id"] = sm["parent_stop_id"].astype(str)

    merged = ot.merge(sm, on="parent_stop_id", how="inner")
    if merged.empty:
        return pd.DataFrame(
            columns=["station_complex_id", "route_id", "direction_id", "stop_sequence", "travel_offset_min"]
        )
    merged["travel_offset_min"] = merged["offset_min"].astype(float) + float(run_delay_min)
    out = (
        merged.groupby(["station_complex_id", "route_id", "direction_id"], as_index=False)
        .agg(travel_offset_min=("travel_offset_min", "median"), stop_sequence=("stop_sequence", "median"))
    )
    out["travel_offset_min"] = out["travel_offset_min"].clip(lower=0.0)
    return out


def project_station_demand_to_departures(
    station_demand: pd.DataFrame,
    station_rd_weights: pd.DataFrame,
    travel_offsets: pd.DataFrame,
    *,
    slot_route: np.ndarray,
    slot_dir: np.ndarray,
    slot_hour: np.ndarray,
    board_minute: float = 30.0,
    return_frame: bool = False,
) -> np.ndarray | tuple[np.ndarray, pd.DataFrame]:
    """Chiếu ngược Không–Thời gian: cầu ga (giờ quẹt thẻ) → cầu trục xuất bến (route, dir, giờ).

    Với mỗi hành khách quẹt thẻ vào ga s lúc giờ h_board:
      - Phân bổ về (route r, direction dir) theo trọng số tần suất w_freq(s, r, dir, h_board).
      - Khấu trừ thời gian hành trình: t0 = h_board·60 + board_minute − T_travel(r, dir, s).
      - Dồn lượng cầu về giờ xuất bến h0 = floor(t0 / 60) mod 24.

    Kết quả: mảng demand theo trục xuất bến, căn theo OPT_SCOPE (slot_route, slot_dir, slot_hour).
    Thay thế hoàn toàn cơ chế `direction_share` / `allocate_demand_to_slots` cũ.
    """
    sd = station_demand.copy()
    sd["station_complex_id"] = sd["station_complex_id"].astype(str)
    sd["hour"] = sd["hour"].astype(int)
    sd["demand"] = pd.to_numeric(sd["demand"], errors="coerce").fillna(0.0).clip(lower=0.0)

    w = station_rd_weights.copy()
    w["station_complex_id"] = w["station_complex_id"].astype(str)
    w["route"] = w["route"].astype(str)
    w["direction_id"] = pd.to_numeric(w["direction_id"], errors="coerce").fillna(0).astype(int)
    w["hour"] = w["hour"].astype(int)

    # B1: phân bổ cầu ga → (route, direction) tại giờ quẹt thẻ
    routed = sd.merge(w, on=["station_complex_id", "hour"], how="inner")
    routed["demand_rd"] = routed["demand"] * routed["alloc_weight"]

    # B2: nối ma trận thời gian hành trình → khấu trừ về trục xuất bến
    to = travel_offsets.rename(columns={"route_id": "route"}).copy()
    to["route"] = to["route"].astype(str)
    to["direction_id"] = pd.to_numeric(to["direction_id"], errors="coerce").fillna(0).astype(int)
    routed = routed.merge(
        to[["station_complex_id", "route", "direction_id", "travel_offset_min"]],
        on=["station_complex_id", "route", "direction_id"],
        how="left",
    )
    routed["travel_offset_min"] = routed["travel_offset_min"].fillna(0.0)

    board_min = routed["hour"].to_numpy() * 60.0 + float(board_minute)
    depart_min = board_min - routed["travel_offset_min"].to_numpy()
    routed["dep_hour"] = (np.floor(depart_min / 60.0).astype(int)) % 24

    # B3: dồn về (route, direction, giờ xuất bến)
    dep_demand = (
        routed.groupby(["route", "direction_id", "dep_hour"], as_index=False)["demand_rd"]
        .sum()
        .rename(columns={"demand_rd": "demand", "dep_hour": "hour"})
    )

    lookup = dep_demand.set_index(["route", "direction_id", "hour"])["demand"]
    n = len(slot_route)
    out = np.zeros(n, dtype=float)
    for i in range(n):
        key = (str(slot_route[i]), int(slot_dir[i]), int(slot_hour[i]))
        val = lookup.get(key)
        if val is not None and np.isfinite(val):
            out[i] = float(val)

    if return_frame:
        return out, dep_demand
    return out


def build_route_direction_departure_windows(
    schedule_dir: Path,
    *,
    service_id: str = "Weekday",
) -> pd.DataFrame:
    """First/last departure (phút, giờ bucket) theo (route_id, direction_id) từ GTFS."""
    schedule_dir = Path(schedule_dir)
    trips = pd.read_csv(
        schedule_dir / "trips.txt",
        dtype={"trip_id": str, "route_id": str, "service_id": str},
    )
    stop_times = pd.read_csv(
        schedule_dir / "stop_times.txt",
        dtype={"trip_id": str, "departure_time": str, "arrival_time": str},
        usecols=lambda c: c in {"trip_id", "stop_sequence", "departure_time", "arrival_time"},
    )
    cal_path = schedule_dir / "calendar.txt"
    available_services = trips["service_id"].dropna().unique().tolist()
    if cal_path.exists():
        cal = pd.read_csv(cal_path, dtype={"service_id": str})
        available_services = cal["service_id"].dropna().unique().tolist() or available_services
    if service_id not in available_services and available_services:
        service_id = available_services[0]

    trips_f = trips[trips["service_id"] == service_id].copy()
    if trips_f.empty:
        trips_f = trips.copy()
    if "direction_id" not in trips_f.columns:
        trips_f["direction_id"] = 0
    trips_f["direction_id"] = (
        pd.to_numeric(trips_f["direction_id"], errors="coerce").fillna(0).astype(int)
    )

    first_stop = (
        stop_times.sort_values(["trip_id", "stop_sequence"])
        .groupby("trip_id", as_index=False)
        .first()[["trip_id", "departure_time", "arrival_time"]]
    )
    first_stop["dep_str"] = first_stop["departure_time"].fillna(first_stop["arrival_time"])
    first_stop["dep_min"] = first_stop["dep_str"].map(gtfs_time_to_minutes)
    first_stop = first_stop.dropna(subset=["dep_min"])

    merged = trips_f[["trip_id", "route_id", "direction_id"]].merge(
        first_stop[["trip_id", "dep_min"]], on="trip_id", how="inner"
    )
    merged["route_id"] = merged["route_id"].astype(str)

    rows = []
    for (route, direction), g in merged.groupby(["route_id", "direction_id"]):
        dep = g["dep_min"].to_numpy(dtype=float)
        first_m, last_m = float(dep.min()), float(dep.max())
        rows.append(
            dict(
                route_id=str(route),
                direction_id=int(direction),
                first_dep_min=first_m,
                last_dep_min=last_m,
                first_dep_time=_minutes_to_gtfs_time(first_m),
                last_dep_time=_minutes_to_gtfs_time(last_m),
                first_hour=int(first_m // 60) % 24,
                last_hour=int(last_m // 60) % 24,
                n_trips=int(len(dep)),
            )
        )
    return pd.DataFrame(rows).sort_values(["route_id", "direction_id"]).reset_index(drop=True)


def _minutes_to_gtfs_time(minutes: float) -> str:
    m = int(round(float(minutes)))
    h, mm = divmod(m, 60)
    return f"{h:02d}:{mm:02d}:00"


def apply_service_window_constraints(
    trips: np.ndarray,
    *,
    slot_route: np.ndarray,
    slot_dir: np.ndarray,
    slot_hour: np.ndarray,
    windows: pd.DataFrame,
    baseline_trips: np.ndarray | None = None,
    trips_min: np.ndarray | None = None,
    trips_max: np.ndarray | None = None,
) -> np.ndarray:
    """Ràng buộc opt_trips trong khung first/last departure GTFS theo (route, direction).

    - Ngoài khung giờ: không tăng so baseline (giữ mức phục vụ tối thiểu).
    - Giờ first/last: trips >= max(baseline, TRIPS_MIN) để giữ điểm đầu/cuối dịch vụ.
    """
    t = np.asarray(trips, dtype=int).copy()
    win = windows.set_index(["route_id", "direction_id"])
    base = np.asarray(baseline_trips, dtype=int) if baseline_trips is not None else None
    tmin = np.asarray(trips_min, dtype=int) if trips_min is not None else None
    tmax = np.asarray(trips_max, dtype=int) if trips_max is not None else None

    for i in range(len(t)):
        key = (str(slot_route[i]), int(slot_dir[i]))
        if key not in win.index:
            continue
        row = win.loc[key]
        fh, lh = int(row["first_hour"]), int(row["last_hour"])
        h = int(slot_hour[i])
        in_window = (fh <= h <= lh) if fh <= lh else (h >= fh or h <= lh)

        if not in_window:
            floor = int(base[i]) if base is not None else (int(tmin[i]) if tmin is not None else 1)
            t[i] = floor
            continue

        if h == fh or h == lh:
            floor = int(base[i]) if base is not None else 1
            if tmin is not None:
                floor = max(floor, int(tmin[i]))
            t[i] = max(int(t[i]), floor)

        if tmax is not None:
            t[i] = min(int(t[i]), int(tmax[i]))
        if tmin is not None:
            t[i] = max(int(t[i]), int(tmin[i]))
    return t


# NYC subway terminal turnaround: ~3–5 min dwell each end (MTA ops guides; no GTFS field).
DEFAULT_TURNAROUND_BUFFER_MIN = 5.0


def compute_route_direction_cycle_times(
    schedule_dir: Path,
    *,
    service_id: str = "Weekday",
    turnaround_buffer_min: float = DEFAULT_TURNAROUND_BUFFER_MIN,
) -> pd.DataFrame:
    """Round-trip cycle time (phút) theo (route_id, direction_id) từ GTFS stop_times.

    one_way = arrival cuối − departure đầu trên mỗi trip.
    cycle_time = 2 × median(one_way) + 2 × turnaround_buffer (buffer mỗi đầu bến).
    """
    schedule_dir = Path(schedule_dir)
    trips = pd.read_csv(
        schedule_dir / "trips.txt",
        dtype={"trip_id": str, "route_id": str, "service_id": str},
    )
    stop_times = pd.read_csv(
        schedule_dir / "stop_times.txt",
        dtype={"trip_id": str, "departure_time": str, "arrival_time": str},
        usecols=lambda c: c in {"trip_id", "stop_sequence", "departure_time", "arrival_time"},
    )
    cal_path = schedule_dir / "calendar.txt"
    available_services = trips["service_id"].dropna().unique().tolist()
    if cal_path.exists():
        cal = pd.read_csv(cal_path, dtype={"service_id": str})
        available_services = cal["service_id"].dropna().unique().tolist() or available_services
    if service_id not in available_services and available_services:
        service_id = available_services[0]

    trips_f = trips[trips["service_id"] == service_id].copy()
    if trips_f.empty:
        trips_f = trips.copy()
    if "direction_id" not in trips_f.columns:
        trips_f["direction_id"] = 0
    trips_f["direction_id"] = (
        pd.to_numeric(trips_f["direction_id"], errors="coerce").fillna(0).astype(int)
    )

    st = stop_times.sort_values(["trip_id", "stop_sequence"])
    first = st.groupby("trip_id", as_index=False).first()
    last = st.groupby("trip_id", as_index=False).last()
    tt = first[["trip_id"]].merge(
        last[["trip_id", "arrival_time"]],
        on="trip_id",
        suffixes=("", "_last"),
    )
    tt["dep_str"] = first["departure_time"].fillna(first["arrival_time"]).values
    tt["arr_str"] = tt["arrival_time"].fillna(tt["dep_str"])
    tt["dep_min"] = tt["dep_str"].map(gtfs_time_to_minutes)
    tt["arr_min"] = tt["arr_str"].map(gtfs_time_to_minutes)
    tt["one_way_min"] = tt["arr_min"] - tt["dep_min"]
    tt = tt.dropna(subset=["one_way_min"])
    tt = tt.loc[tt["one_way_min"] > 0].copy()

    merged = trips_f[["trip_id", "route_id", "direction_id"]].merge(
        tt[["trip_id", "one_way_min"]], on="trip_id", how="inner"
    )
    merged["route_id"] = merged["route_id"].astype(str)
    buf = float(turnaround_buffer_min)

    rows = []
    for (route, direction), g in merged.groupby(["route_id", "direction_id"]):
        ow = g["one_way_min"].to_numpy(dtype=float)
        med_ow = float(np.median(ow))
        rows.append(
            dict(
                route_id=str(route),
                direction_id=int(direction),
                one_way_min_med=med_ow,
                cycle_time_min=float(2.0 * med_ow + 2.0 * buf),
                turnaround_buffer_min=buf,
                n_trips_sample=int(len(ow)),
            )
        )
    return pd.DataFrame(rows).sort_values(["route_id", "direction_id"]).reset_index(drop=True)


def build_headway_trip_bounds(
    slot_hour: np.ndarray,
    *,
    min_headway_min: float = 3.0,
    max_headway_min: float = 20.0,
    min_trips: int = 1,
    absolute_max: int = 60,
) -> tuple[np.ndarray, np.ndarray]:
    """Suy TRIPS_MIN/TRIPS_MAX từ MIN_HEADWAY / MAX_HEADWAY (phút).

    trips_max = floor(60 / min_headway), trips_min = ceil(60 / max_headway).
    """
    min_hw = max(float(min_headway_min), 1e-6)
    max_hw = max(float(max_headway_min), min_hw)
    t_max = int(np.floor(60.0 / min_hw))
    t_min = int(np.ceil(60.0 / max_hw))
    t_max = min(t_max, int(absolute_max))
    t_min = max(t_min, int(min_trips))
    n = len(slot_hour)
    return (
        np.full(n, t_min, dtype=int),
        np.full(n, t_max, dtype=int),
    )


def merge_trip_bounds(
    trips_min_a: np.ndarray,
    trips_max_a: np.ndarray,
    trips_min_b: np.ndarray,
    trips_max_b: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Giao hai bộ bound: min = max(mins), max = min(maxs), đảm bảo min ≤ max."""
    tmin = np.maximum(np.asarray(trips_min_a, dtype=int), np.asarray(trips_min_b, dtype=int))
    tmax = np.minimum(np.asarray(trips_max_a, dtype=int), np.asarray(trips_max_b, dtype=int))
    tmax = np.maximum(tmax, tmin)
    return tmin, tmax


def build_headway_trip_bounds_by_slot(
    slot_hour: np.ndarray,
    *,
    min_headway_min: float = 3.0,
    max_headway_min: float = 20.0,
    overnight_max_headway_min: float | None = None,
    min_trips: int = 1,
    absolute_max: int = 60,
) -> tuple[np.ndarray, np.ndarray]:
    """Headway bounds per slot; overnight có thể nới MAX_HEADWAY (sàn chuyến thấp hơn)."""
    hrs = np.asarray(slot_hour, dtype=int)
    is_overnight = (hrs <= 6) | (hrs >= 23)
    ovn_hw = float(overnight_max_headway_min if overnight_max_headway_min is not None else max_headway_min)
    max_hw = np.where(is_overnight, ovn_hw, float(max_headway_min))
    min_hw = max(float(min_headway_min), 1e-6)
    t_max_slot = np.minimum(int(absolute_max), np.floor(60.0 / min_hw).astype(int))
    t_min_slot = np.maximum(int(min_trips), np.ceil(60.0 / np.maximum(max_hw, 1e-6)).astype(int))
    return t_min_slot.astype(int), np.full(len(hrs), t_max_slot, dtype=int)


def compute_merged_trip_bounds(
    baseline_trips: np.ndarray,
    slot_hour: np.ndarray,
    *,
    demand: np.ndarray | None = None,
    slot_route: np.ndarray | None = None,
    peak_percentile: float = 75.0,
    overnight_percentile: float = 20.0,
    daytime_factor: float | None = None,
    peak_factor: float = 1.40,
    offpeak_factor: float = 1.15,
    overnight_factor: float = 1.05,
    min_factor: float = 0.5,
    overnight_min_factor: float | None = None,
    max_delta: int = 3,
    min_headway_min: float = 3.0,
    max_headway_min: float = 20.0,
    overnight_max_headway_min: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Dynamic baseline bounds ∩ headway bounds (một nguồn sự thật cho §5).

    Truyền `demand` (trục xuất bến) để bật phát hiện overnight động; nếu None → giờ tĩnh.
    """
    dyn_min, dyn_max = build_dynamic_bounds(
        baseline_trips,
        slot_hour,
        demand=demand,
        slot_route=slot_route,
        peak_percentile=peak_percentile,
        overnight_percentile=overnight_percentile,
        daytime_factor=daytime_factor,
        peak_factor=peak_factor,
        offpeak_factor=offpeak_factor,
        overnight_factor=overnight_factor,
        min_factor=min_factor,
        overnight_min_factor=overnight_min_factor,
        max_delta=max_delta,
    )
    if overnight_max_headway_min is not None:
        hw_min, hw_max = build_headway_trip_bounds_by_slot(
            slot_hour,
            min_headway_min=min_headway_min,
            max_headway_min=max_headway_min,
            overnight_max_headway_min=overnight_max_headway_min,
        )
    else:
        hw_min, hw_max = build_headway_trip_bounds(
            slot_hour,
            min_headway_min=min_headway_min,
            max_headway_min=max_headway_min,
        )
    return merge_trip_bounds(dyn_min, dyn_max, hw_min, hw_max)


def trips_bounds_for_scenario(
    baseline_trips: np.ndarray,
    slot_hour: np.ndarray,
    trips_min: np.ndarray,
    trips_max: np.ndarray,
    scenario: str,
    *,
    rainy_peak_hours: tuple[int, ...] = (),
    rainy_peak_min_factor: float = 0.65,
    trips_min_factor: float = 0.5,
    rainy_daytime_max_factor: float | None = None,
    rainy_offpeak_max_factor: float | None = None,
    peak_hours: tuple[int, ...] = (7, 8, 9, 17, 18, 19),
) -> tuple[np.ndarray, np.ndarray]:
    """Per-scenario TRIPS_MIN/TRIPS_MAX (rainy peak floor, rainy daytime ceiling)."""
    rainy_day_max = rainy_daytime_max_factor
    if rainy_day_max is None:
        rainy_day_max = rainy_offpeak_max_factor
    tmin = np.asarray(trips_min, dtype=int).copy()
    tmax = np.asarray(trips_max, dtype=int).copy()
    hrs = np.asarray(slot_hour, dtype=int)
    base = np.asarray(baseline_trips, dtype=float)
    if scenario == "rainy_day":
        if rainy_peak_hours and rainy_peak_min_factor > trips_min_factor:
            rainy_peak = np.isin(hrs, rainy_peak_hours)
            boosted_min = np.ceil(base[rainy_peak] * rainy_peak_min_factor).astype(int)
            tmin[rainy_peak] = np.maximum(tmin[rainy_peak], boosted_min)
        if rainy_day_max is not None:
            is_overnight = np.array([is_overnight_hour(h) for h in hrs], dtype=bool)
            rainy_day = ~is_overnight
            boosted_max = np.floor(base[rainy_day] * float(rainy_day_max)).astype(int)
            tmax[rainy_day] = np.maximum(tmax[rainy_day], boosted_max)
            tmax = np.maximum(tmax, tmin)
    return tmin, tmax


def compute_fleet_limits_from_baseline(
    baseline_trips: np.ndarray,
    slot_route: np.ndarray,
    slot_dir: np.ndarray,
    slot_hour: np.ndarray,
    cycle_times: pd.DataFrame,
) -> tuple[pd.DataFrame, float]:
    """Fleet size từ lịch baseline (Little's law).

    Per (route, direction): max_h trips[h] × cycle / 60.
    System: max_h Σ_{r,d} trips[r,d,h] × cycle[r,d] / 60.
  """
    ct = cycle_times.set_index(["route_id", "direction_id"])["cycle_time_min"]
    routes = np.asarray(slot_route)
    dirs = np.asarray(slot_dir, dtype=int)
    hours = np.asarray(slot_hour, dtype=int)
    base = np.asarray(baseline_trips, dtype=float)

    rd_vehicles: dict[tuple[str, int], float] = {}
    hour_system: dict[int, float] = {}

    for i in range(len(base)):
        key = (str(routes[i]), int(dirs[i]))
        cycle = float(ct.get(key, np.nan))
        if not np.isfinite(cycle) or cycle <= 0:
            cycle = float(ct.median()) if len(ct) else 90.0
        veh = base[i] * cycle / 60.0
        rd_vehicles[key] = max(rd_vehicles.get(key, 0.0), veh)
        h = int(hours[i])
        hour_system[h] = hour_system.get(h, 0.0) + veh

    rd_rows = [
        dict(route_id=k[0], direction_id=k[1], fleet_size=round(v, 3))
        for k, v in sorted(rd_vehicles.items())
    ]
    system_max = float(max(hour_system.values())) if hour_system else 0.0
    return pd.DataFrame(rd_rows), system_max


def max_trips_from_fleet(cycle_time_min: float, fleet_size: float) -> int:
    """Số chuyến/giờ tối đa: floor(fleet × 60 / cycle_time)."""
    cycle = max(float(cycle_time_min), 1e-6)
    fleet = max(float(fleet_size), 0.0)
    return max(1, int(np.floor(fleet * 60.0 / cycle)))


def apply_route_fleet_cap(
    trips: np.ndarray,
    *,
    slot_route: np.ndarray,
    slot_dir: np.ndarray,
    cycle_times: pd.DataFrame,
    fleet_by_route_dir: pd.DataFrame,
) -> np.ndarray:
    """Giới hạn trips theo fleet từng (route, direction) — ràng buộc cứng turnaround."""
    t = np.asarray(trips, dtype=int).copy()
    ct = cycle_times.set_index(["route_id", "direction_id"])["cycle_time_min"]
    fl = fleet_by_route_dir.set_index(["route_id", "direction_id"])["fleet_size"]
    for i in range(len(t)):
        key = (str(slot_route[i]), int(slot_dir[i]))
        if key not in fl.index:
            continue
        cycle = float(ct.get(key, 90.0))
        cap = max_trips_from_fleet(cycle, float(fl.loc[key]))
        t[i] = min(int(t[i]), cap)
    return t


def apply_system_fleet_cap(
    trips: np.ndarray,
    *,
    slot_route: np.ndarray,
    slot_dir: np.ndarray,
    slot_hour: np.ndarray,
    cycle_times: pd.DataFrame,
    max_system_fleet: float,
) -> np.ndarray:
    """Giới hạn tổng xe đồng thời toàn hệ thống theo từng giờ (scale đều nếu vượt cap)."""
    t = np.asarray(trips, dtype=int).copy()
    ct = cycle_times.set_index(["route_id", "direction_id"])["cycle_time_min"]
    cap = float(max_system_fleet)
    if cap <= 0:
        return t

    hours = np.asarray(slot_hour, dtype=int)
    for h in sorted(set(hours.tolist())):
        idx = np.where(hours == h)[0]
        vehicles = np.zeros(len(idx), dtype=float)
        for j, i in enumerate(idx):
            key = (str(slot_route[i]), int(slot_dir[i]))
            cycle = float(ct.get(key, 90.0))
            vehicles[j] = t[i] * cycle / 60.0
        total = float(vehicles.sum())
        if total <= cap + 1e-9:
            continue
        scale = cap / max(total, 1e-9)
        for j, i in enumerate(idx):
            key = (str(slot_route[i]), int(slot_dir[i]))
            cycle = float(ct.get(key, 90.0))
            new_trips = max(1, int(np.floor(t[i] * scale)))
            max_t = max_trips_from_fleet(cycle, cap)
            t[i] = min(new_trips, max_t)
    return t


def apply_capacity_floor(
    trips: np.ndarray,
    demand: np.ndarray,
    *,
    capacity_per_trip: float = 1200.0,
    max_overflow_pct: float | None = None,
    slot_route: np.ndarray | None = None,
    slot_dir: np.ndarray | None = None,
    slot_hour: np.ndarray | None = None,
) -> np.ndarray:
    """Nâng trips tối thiểu để đáp ứng demand (ceil(demand/capacity)); tùy chọn kiểm overflow."""
    t = np.asarray(trips, dtype=int).copy()
    d = np.clip(np.asarray(demand, dtype=float), 0.0, None)
    cap = max(float(capacity_per_trip), 1.0)
    floor = np.ceil(d / cap).astype(int)
    t = np.maximum(t, floor)

    if max_overflow_pct is not None and slot_route is not None:
        m = compute_schedule_metrics(
            d, t,
            slot_route=np.asarray(slot_route),
            slot_dir=np.asarray(slot_dir),
            slot_hour=np.asarray(slot_hour),
            capacity_per_trip=cap,
        )
        if float(m["overflow_pct"]) > float(max_overflow_pct):
            overflow_idx = np.where(np.asarray(m["overflow_out"]) > 0)[0]
            for i in overflow_idx:
                t[i] += 1
    return t


def apply_smoothness_constraint(
    trips: np.ndarray,
    *,
    slot_route: np.ndarray,
    slot_dir: np.ndarray,
    slot_hour: np.ndarray,
    max_delta_per_hour: int = 3,
    trips_min: np.ndarray | None = None,
    trips_max: np.ndarray | None = None,
) -> np.ndarray:
    """Giới hạn |trips[h] − trips[h−1]| ≤ max_delta trong cùng route×direction."""
    t = np.asarray(trips, dtype=int).copy()
    routes = np.asarray(slot_route)
    dirs = np.asarray(slot_dir, dtype=int)
    hours = np.asarray(slot_hour, dtype=int)
    delta = int(max_delta_per_hour)
    tmin = np.asarray(trips_min, dtype=int) if trips_min is not None else None
    tmax = np.asarray(trips_max, dtype=int) if trips_max is not None else None

    groups: dict[tuple[str, int], list[int]] = {}
    for i in range(len(t)):
        groups.setdefault((str(routes[i]), int(dirs[i])), []).append(i)

    for indices in groups.values():
        order = sorted(indices, key=lambda idx: int(hours[idx]))
        for _ in range(3):
            prev_val = None
            for i in order:
                if prev_val is not None:
                    lo, hi = prev_val - delta, prev_val + delta
                    t[i] = int(np.clip(t[i], lo, hi))
                if tmin is not None:
                    t[i] = max(int(t[i]), int(tmin[i]))
                if tmax is not None:
                    t[i] = min(int(t[i]), int(tmax[i]))
                prev_val = int(t[i])
            prev_val = None
            for i in reversed(order):
                if prev_val is not None:
                    lo, hi = prev_val - delta, prev_val + delta
                    t[i] = int(np.clip(t[i], lo, hi))
                if tmin is not None:
                    t[i] = max(int(t[i]), int(tmin[i]))
                if tmax is not None:
                    t[i] = min(int(t[i]), int(tmax[i]))
                prev_val = int(t[i])
    return t


def apply_optimizer_constraints(
    trips: np.ndarray,
    demand: np.ndarray,
    *,
    slot_route: np.ndarray,
    slot_dir: np.ndarray,
    slot_hour: np.ndarray,
    cycle_times: pd.DataFrame | None = None,
    fleet_by_route_dir: pd.DataFrame | None = None,
    max_system_fleet: float | None = None,
    capacity_per_trip: float = 1200.0,
    max_overflow_pct: float | None = None,
    smoothness_max_delta: int | None = None,
    trips_min: np.ndarray | None = None,
    trips_max: np.ndarray | None = None,
    use_route_fleet: bool = True,
    use_system_fleet: bool = True,
    use_capacity: bool = True,
    use_smoothness: bool = True,
) -> np.ndarray:
    """Áp dụng tuần tự các ràng buộc cứng sau bước analytical + service window."""
    t = np.asarray(trips, dtype=int).copy()
    if use_route_fleet and cycle_times is not None and fleet_by_route_dir is not None:
        t = apply_route_fleet_cap(
            t,
            slot_route=slot_route,
            slot_dir=slot_dir,
            cycle_times=cycle_times,
            fleet_by_route_dir=fleet_by_route_dir,
        )
    if use_system_fleet and cycle_times is not None and max_system_fleet is not None:
        t = apply_system_fleet_cap(
            t,
            slot_route=slot_route,
            slot_dir=slot_dir,
            slot_hour=slot_hour,
            cycle_times=cycle_times,
            max_system_fleet=float(max_system_fleet),
        )
    if use_capacity:
        t = apply_capacity_floor(
            t,
            demand,
            capacity_per_trip=capacity_per_trip,
            max_overflow_pct=max_overflow_pct,
            slot_route=slot_route,
            slot_dir=slot_dir,
            slot_hour=slot_hour,
        )
    if use_smoothness and smoothness_max_delta is not None:
        t = apply_smoothness_constraint(
            t,
            slot_route=slot_route,
            slot_dir=slot_dir,
            slot_hour=slot_hour,
            max_delta_per_hour=int(smoothness_max_delta),
            trips_min=trips_min,
            trips_max=trips_max,
        )
    if trips_min is not None:
        t = np.maximum(t, np.asarray(trips_min, dtype=int))
    if trips_max is not None:
        t = np.minimum(t, np.asarray(trips_max, dtype=int))
    return t.astype(int)


def format_keras_inputs(
    route_idx: np.ndarray,
    num_feat: np.ndarray,
    *,
    use_route_embedding: bool = False,
) -> dict[str, np.ndarray] | np.ndarray:
    """Chuẩn hóa input cho Keras (embedding cần route_idx shape (batch, 1), int32)."""
    nf = np.asarray(num_feat, dtype=np.float32)
    if nf.ndim == 1:
        nf = nf.reshape(1, -1)
    if not use_route_embedding:
        return nf
    ri = np.asarray(route_idx, dtype=np.int32).reshape(-1, 1)
    return {"route_idx": ri, "num_features": nf}


def add_cyclical_time_features(d: pd.DataFrame) -> pd.DataFrame:
    """Mã hóa chu kỳ giờ/tuần/tháng (phổ biến trong forecasting transit)."""
    out = d.copy()
    if "hour" in out.columns:
        out["hour_sin"] = np.sin(2 * np.pi * out["hour"] / 24)
        out["hour_cos"] = np.cos(2 * np.pi * out["hour"] / 24)
    if "day_of_week" in out.columns:
        out["dow_sin"] = np.sin(2 * np.pi * out["day_of_week"] / 7)
        out["dow_cos"] = np.cos(2 * np.pi * out["day_of_week"] / 7)
    if "month" in out.columns:
        m = pd.to_numeric(out["month"], errors="coerce").fillna(1).astype(int)
        out["month_sin"] = np.sin(2 * np.pi * (m - 1) / 12)
        out["month_cos"] = np.cos(2 * np.pi * (m - 1) / 12)
    return out


def _minutes_to_hhmm(minutes: float) -> str:
    m = int(round(float(minutes))) % (24 * 60)
    h, mm = divmod(m, 60)
    return f"{h:02d}:{mm:02d}"


def load_gtfs_stop_offset_templates(
    schedule_dir: Path,
    *,
    service_id: str = "Weekday",
) -> pd.DataFrame:
    """Median offset (phút) từ departure đầu trip → từng stop theo (route_id, direction_id).

    Reuses cùng logic parse GTFS như compute_route_direction_cycle_times.
  """
    schedule_dir = Path(schedule_dir)
    trips = pd.read_csv(
        schedule_dir / "trips.txt",
        dtype={"trip_id": str, "route_id": str, "service_id": str},
    )
    stop_times = pd.read_csv(
        schedule_dir / "stop_times.txt",
        dtype={"trip_id": str, "stop_id": str, "departure_time": str, "arrival_time": str},
        usecols=lambda c: c
        in {"trip_id", "stop_id", "stop_sequence", "departure_time", "arrival_time"},
    )
    stops_path = schedule_dir / "stops.txt"
    stops = (
        pd.read_csv(stops_path, dtype=str)
        if stops_path.exists()
        else pd.DataFrame(columns=["stop_id", "stop_name", "parent_station"])
    )

    cal_path = schedule_dir / "calendar.txt"
    available_services = trips["service_id"].dropna().unique().tolist()
    if cal_path.exists():
        cal = pd.read_csv(cal_path, dtype={"service_id": str})
        available_services = cal["service_id"].dropna().unique().tolist() or available_services
    if service_id not in available_services and available_services:
        service_id = available_services[0]

    trips_f = trips[trips["service_id"] == service_id].copy()
    if trips_f.empty:
        trips_f = trips.copy()
    if "direction_id" not in trips_f.columns:
        trips_f["direction_id"] = 0
    trips_f["direction_id"] = (
        pd.to_numeric(trips_f["direction_id"], errors="coerce").fillna(0).astype(int)
    )

    st = stop_times.sort_values(["trip_id", "stop_sequence"])
    first = st.groupby("trip_id", as_index=False).first()
    merged = trips_f[["trip_id", "route_id", "direction_id"]].merge(
        first[["trip_id", "departure_time", "arrival_time"]],
        on="trip_id",
        how="inner",
    )
    merged["dep_str"] = merged["departure_time"].fillna(merged["arrival_time"])
    merged["dep_min"] = merged["dep_str"].map(gtfs_time_to_minutes)
    merged = merged.dropna(subset=["dep_min"])

    st_full = st.merge(merged[["trip_id", "route_id", "direction_id", "dep_min"]], on="trip_id")
    st_full["time_str"] = st_full["arrival_time"].fillna(st_full["departure_time"])
    st_full["stop_min"] = st_full["time_str"].map(gtfs_time_to_minutes)
    st_full["offset_min"] = st_full["stop_min"] - st_full["dep_min"]
    st_full = st_full.dropna(subset=["offset_min"])
    st_full["route_id"] = st_full["route_id"].astype(str)

    if not stops.empty:
        parent = stops.set_index("stop_id")["parent_station"].to_dict()
        names = stops.set_index("stop_id")["stop_name"].to_dict()
        st_full["parent_stop_id"] = st_full["stop_id"].map(
            lambda s: parent.get(str(s), str(s)).replace("nan", str(s))
        )
        st_full["stop_name"] = st_full["stop_id"].map(lambda s: names.get(str(s), str(s)))
    else:
        st_full["parent_stop_id"] = st_full["stop_id"]
        st_full["stop_name"] = st_full["stop_id"]

    agg = (
        st_full.groupby(["route_id", "direction_id", "parent_stop_id", "stop_name", "stop_sequence"], as_index=False)[
            "offset_min"
        ]
        .median()
        .sort_values(["route_id", "direction_id", "stop_sequence"])
    )
    return agg


def expand_schedule_to_station_times(
    schedule: pd.DataFrame,
    schedule_dir: Path,
    *,
    offset_templates: pd.DataFrame | None = None,
    route_id: str | None = None,
) -> pd.DataFrame:
    """Mở rộng lịch route×direction×hour → giờ đến từng ga (offset từ GTFS).

    schedule columns: route (or route_id), direction (or direction_id), hour, opt_trips.
    """
    schedule_dir = Path(schedule_dir)
    templates = offset_templates if offset_templates is not None else load_gtfs_stop_offset_templates(schedule_dir)

    sch = schedule.copy()
    route_col = "route_id" if "route_id" in sch.columns else "route"
    dir_col = "direction_id" if "direction_id" in sch.columns else "direction"
    trips_col = "opt_trips" if "opt_trips" in sch.columns else "trips"

    if route_id is not None:
        sch = sch.loc[sch[route_col].astype(str) == str(route_id)]

    rows: list[dict[str, Any]] = []
    for _, slot in sch.iterrows():
        rid = str(slot[route_col])
        direction = int(slot[dir_col])
        hour = int(slot["hour"])
        n_trips = max(int(slot[trips_col]), 0)
        if n_trips <= 0:
            continue

        pattern = templates.loc[
            (templates["route_id"] == rid) & (templates["direction_id"] == direction)
        ]
        if pattern.empty:
            continue

        hour_start = hour * 60
        spacing = 60.0 / n_trips
        for trip_i in range(n_trips):
            first_dep = hour_start + (trip_i + 0.5) * spacing
            for _, stop in pattern.iterrows():
                arr_min = first_dep + float(stop["offset_min"])
                rows.append(
                    dict(
                        route=rid,
                        direction=direction,
                        hour=hour,
                        parent_stop_id=str(stop["parent_stop_id"]),
                        stop_name=str(stop["stop_name"]),
                        stop_sequence=int(stop["stop_sequence"]),
                        scheduled_time=_minutes_to_hhmm(arr_min),
                        scheduled_min=arr_min,
                    )
                )

    out = pd.DataFrame(rows)
    if out.empty:
        return out
    return out.sort_values(["route", "direction", "scheduled_min", "stop_sequence"]).reset_index(drop=True)


WEATHER_LABEL_CATEGORIES = [
    "clear",
    "mainly_clear",
    "partly_cloudy",
    "overcast",
    "light_drizzle",
    "drizzle",
    "dense_drizzle",
    "light_rain",
    "rain",
    "heavy_rain",
    "light_snow",
    "snow",
    "heavy_snow",
]


def encode_weather_label(
    d: pd.DataFrame,
    *,
    label_col: str = "weather_label",
    out_col: str = "weather_label_code",
    categories: list[str] | None = None,
) -> pd.DataFrame:
    """Mã hóa weather_label (chuỗi) → chỉ số nguyên cho mô hình."""
    out = d.copy()
    cats = categories or WEATHER_LABEL_CATEGORIES
    default_idx = cats.index("overcast") if "overcast" in cats else 0
    if label_col not in out.columns:
        out[out_col] = default_idx
        return out
    labels = out[label_col].astype(str).str.strip().str.lower()
    mapping = {c: i for i, c in enumerate(cats)}
    out[out_col] = labels.map(mapping).fillna(default_idx).astype(int)
    return out


def add_lag_features(
    d: pd.DataFrame,
    *,
    use_lags: bool = True,
    lag_cols: list[str] | None = None,
    entity_col: str = "route_id",
) -> pd.DataFrame:
    """Lag cùng giờ tại CÙNG thực thể: 24h, 168h, rolling 7d (ITM 2026 / LSTM MTA papers).

    `entity_col` = "station_complex_id" cho dự báo cấp ga, "route_id" cho legacy cấp tuyến.
    """
    lag_cols = lag_cols or DEFAULT_LAG_COLS
    if not use_lags:
        return d
    out = d.sort_values([entity_col, "hour", "date"]).copy()
    g = out.groupby([entity_col, "hour"])["demand"]
    out["demand_lag_24h"] = g.shift(1)
    out["demand_lag_168h"] = g.shift(7)
    out["demand_rolling_7d"] = g.transform(lambda s: s.shift(1).rolling(7, min_periods=1).mean())
    for raw, log_c in [
        ("demand_lag_24h", "log_lag_24h"),
        ("demand_lag_168h", "log_lag_168h"),
        ("demand_rolling_7d", "log_rolling_7d"),
    ]:
        out[log_c] = np.log1p(out[raw].clip(lower=0))
    return out.sort_values([entity_col, "date", "hour"]).reset_index(drop=True)


def fill_lag_from_train(
    d: pd.DataFrame,
    ref: pd.DataFrame,
    lag_cols: list[str] | None = None,
    entity_col: str = "route_id",
) -> pd.DataFrame:
    lag_cols = lag_cols or DEFAULT_LAG_COLS
    lag_medians = ref.groupby([entity_col, "hour"])[lag_cols].median().reset_index()
    out = d.merge(lag_medians, on=[entity_col, "hour"], how="left", suffixes=("", "_fill"))
    for c in lag_cols:
        fill_c = f"{c}_fill"
        if fill_c in out.columns:
            out[c] = out[c].fillna(out[fill_c])
            out = out.drop(columns=[fill_c])
        global_med = float(ref[c].median()) if c in ref.columns else 0.0
        out[c] = out[c].fillna(global_med)
    return out


def build_num_feature_list(
    hourly_factor_cols: list[str],
    *,
    use_lags: bool = True,
    lag_cols: list[str] | None = None,
) -> list[str]:
    """Danh sách cột numeric thống nhất train / scenario / CV."""
    lag_cols = lag_cols or DEFAULT_LAG_COLS
    feats = [
        "hour_sin",
        "hour_cos",
        "dow_sin",
        "dow_cos",
        "month_sin",
        "month_cos",
        "is_weekend",
        "is_us_holiday",
        *hourly_factor_cols,
        "log_baseline",
    ]
    if use_lags:
        feats.extend(lag_cols)
    return feats


def build_demand_model(
    n_routes: int,
    n_num_features: int,
    *,
    use_route_embedding: bool = False,
    embedding_dim: int | None = None,
    hidden: tuple[int, ...] = (64, 32),
    dropout: float = 0.25,
    use_batch_norm: bool = True,
) -> Model:
    """MLP residual log-demand (Huber) — BatchNorm theo best practice DL traffic.

    `n_routes` đóng vai trò là số thực thể có embedding (route HOẶC station). Với dự báo
    cấp ga (hàng trăm ga) nên dùng `embedding_dim` lớn hơn (vd. 8–16). Input vẫn tên
    "route_idx" để tương thích artifact/runtime; thực chất là chỉ số thực thể (entity index).

    Lớp output ép kiểu float32 (an toàn khi bật mixed-precision 'mixed_float16').
    """
    if embedding_dim is None:
        embedding_dim = 4 if n_routes <= 40 else int(min(16, max(6, round(n_routes ** 0.25 * 2))))
    inp_num = Input(shape=(n_num_features,), name="num_features")
    x = inp_num
    inputs: list[Any] = [inp_num]

    if use_route_embedding and n_routes > 1:
        inp_route = Input(shape=(1,), name="route_idx", dtype="int32")
        route_emb = layers.Embedding(
            n_routes,
            int(embedding_dim),
            name="route_emb",
            embeddings_regularizer=tf.keras.regularizers.l2(1e-3),
        )(inp_route)
        route_emb = layers.Flatten()(route_emb)
        x = layers.Concatenate()([route_emb, inp_num])
        inputs = [inp_route, inp_num]

    for i, units in enumerate(hidden):
        x = layers.Dense(
            units,
            activation="relu",
            kernel_regularizer=tf.keras.regularizers.l2(5e-3),
            name=f"dense_{i}",
        )(x)
        if use_batch_norm:
            x = layers.BatchNormalization(name=f"bn_{i}")(x)
        if dropout > 0:
            x = layers.Dropout(dropout, name=f"dropout_{i}")(x)

    out = layers.Dense(
        1, name="residual_log_demand", kernel_initializer="zeros", dtype="float32"
    )(x)
    model = Model(inputs, out)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(3e-4),
        loss=tf.keras.losses.Huber(delta=0.5),
        metrics=["mae"],
    )
    return model


def _normalize_date_set(dates: set | list | np.ndarray | pd.Series) -> set[pd.Timestamp]:
    """Chuẩn hóa tập ngày về midnight Timestamp (so khớp merge route×date×hour)."""
    if not dates:
        return set()
    return set(pd.to_datetime(pd.Series(list(dates))).dt.normalize())


def prepare_lstm_sequences(
    df: pd.DataFrame,
    seq_len: int,
    feature_cols: list[str],
    target_col: str,
    *,
    train_dates: set | list | np.ndarray | pd.Series | None = None,
    val_dates: set | list | np.ndarray | pd.Series | None = None,
    test_dates: set | list | np.ndarray | pd.Series | None = None,
    target_split: str = "train",
    meta_cols: list[str] | None = None,
    route_col: str = "route_id",
    date_col: str = "date",
    hour_col: str = "hour",
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame, dict[str, Any]]:
    """Tạo chuỗi LSTM theo route: mỗi mẫu = seq_len giờ trước → dự báo residual tại giờ hiện tại.

    Trả về X (n, seq_len, n_features), y (n,), meta (route/date/hour), stats.
    """
    split_map = {
        "train": _normalize_date_set(train_dates or set()),
        "val": _normalize_date_set(val_dates or set()),
        "test": _normalize_date_set(test_dates or set()),
    }
    if target_split not in split_map:
        raise ValueError(f"target_split phải là train|val|test, nhận {target_split!r}")
    target_dates = split_map[target_split]
    if not target_dates:
        raise ValueError(f"Không có ngày cho split {target_split!r}")

    def _in_target_dates(d: Any) -> bool:
        return pd.Timestamp(d).normalize() in target_dates

    missing = [c for c in feature_cols + [target_col, route_col, date_col, hour_col] if c not in df.columns]
    if missing:
        raise KeyError(f"Thiếu cột trong df: {missing}")

    work = df.copy()
    work[date_col] = pd.to_datetime(work[date_col]).dt.normalize()
    work = work.sort_values([route_col, date_col, hour_col]).reset_index(drop=True)

    meta_default = [route_col, date_col, hour_col]
    meta_keep = meta_default if meta_cols is None else list(dict.fromkeys(meta_default + list(meta_cols)))

    feat_arr = work[feature_cols].to_numpy(dtype=np.float32)
    target_arr = work[target_col].to_numpy(dtype=np.float32)
    dates_arr = work[date_col].to_numpy()
    routes_arr = work[route_col].to_numpy()

    xs: list[np.ndarray] = []
    ys: list[float] = []
    meta_rows: list[dict[str, Any]] = []
    skipped_short = 0
    skipped_nan = 0
    skipped_split = 0

    for route in work[route_col].unique():
        idx = np.flatnonzero(routes_arr == route)
        if len(idx) < seq_len + 1:
            skipped_short += len(idx)
            continue
        for pos in range(seq_len, len(idx)):
            i = int(idx[pos])
            if not _in_target_dates(dates_arr[i]):
                skipped_split += 1
                continue
            window = feat_arr[idx[pos - seq_len] : idx[pos]]
            if window.shape[0] != seq_len:
                skipped_short += 1
                continue
            y_val = float(target_arr[i])
            if np.isnan(window).any() or np.isnan(y_val):
                skipped_nan += 1
                continue
            xs.append(window)
            ys.append(y_val)
            meta_rows.append({c: work.iloc[i][c] for c in meta_keep})

    if not xs:
        raise ValueError(
            f"Không tạo được chuỗi LSTM cho split={target_split!r} "
            f"(seq_len={seq_len}, skipped_short={skipped_short}, skipped_nan={skipped_nan})"
        )

    X = np.stack(xs, axis=0).astype(np.float32)
    y = np.asarray(ys, dtype=np.float32)
    meta = pd.DataFrame(meta_rows).reset_index(drop=True)
    stats = {
        "split": target_split,
        "n_samples": int(len(y)),
        "seq_len": int(seq_len),
        "n_features": len(feature_cols),
        "n_routes": int(work[route_col].nunique()),
        "skipped_short_history": int(skipped_short),
        "skipped_nan": int(skipped_nan),
        "skipped_wrong_split": int(skipped_split),
    }
    return X, y, meta, stats


def build_lstm_demand_model(
    seq_len: int,
    n_features: int,
    *,
    hidden_units: int = 64,
    num_lstm_layers: int = 2,
    dropout: float = 0.25,
    use_batch_norm: bool = True,
) -> Model:
    """LSTM residual log-demand (Huber) — 2-layer LSTM theo ITM 2026 MTA paper."""
    _ = seq_len  # documented for callers; shape fixed via Input
    inp = Input(shape=(seq_len, n_features), name="lstm_sequence")
    x = inp
    for i in range(int(num_lstm_layers)):
        x = layers.LSTM(
            int(hidden_units),
            return_sequences=i < int(num_lstm_layers) - 1,
            name=f"lstm_{i}",
            dropout=float(dropout) if dropout > 0 else 0.0,
        )(x)
    if use_batch_norm:
        x = layers.BatchNormalization(name="bn_lstm")(x)
    head_units = max(int(hidden_units) // 2, 16)
    x = layers.Dense(
        head_units,
        activation="relu",
        kernel_regularizer=tf.keras.regularizers.l2(5e-3),
        name="dense_head",
    )(x)
    if dropout > 0:
        x = layers.Dropout(float(dropout), name="dropout_head")(x)
    out = layers.Dense(1, name="residual_log_demand", kernel_initializer="zeros")(x)
    model = Model(inp, out)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(3e-4),
        loss=tf.keras.losses.Huber(delta=0.5),
        metrics=["mae"],
    )
    return model


def fit_histgbm_demand(
    X_train: np.ndarray,
    y_train: np.ndarray,
    *,
    sample_weight: np.ndarray | None = None,
    max_depth: int = 8,
    learning_rate: float = 0.06,
    max_iter: int = 400,
) -> HistGradientBoostingRegressor:
    """Tree ensemble baseline (thay LightGBM khi không cài lightgbm)."""
    reg = HistGradientBoostingRegressor(
        max_depth=max_depth,
        learning_rate=learning_rate,
        max_iter=max_iter,
        early_stopping=True,
        validation_fraction=0.15,
        n_iter_no_change=20,
        random_state=42,
    )
    reg.fit(X_train, y_train, sample_weight=sample_weight)
    return reg


def blend_residual_predictions(
    pred_a: np.ndarray,
    pred_b: np.ndarray,
    weight_a: float = 0.5,
) -> np.ndarray:
    w = float(np.clip(weight_a, 0.0, 1.0))
    return w * pred_a + (1.0 - w) * pred_b


def residuals_to_demand(log_baseline: np.ndarray, resid: np.ndarray, clip: tuple[float, float]) -> np.ndarray:
    r = np.clip(resid, *clip)
    return np.expm1(log_baseline + r)


def predict_with_uncertainty_mcdropout(
    model: Model,
    X: dict[str, np.ndarray] | np.ndarray | list,
    *,
    n_samples: int = 50,
    training: bool = True,
) -> dict[str, np.ndarray]:
    """Deprecated: không dùng trong pipeline chính — giữ lại cho backward compat.

    MC Dropout: n forward passes với dropout active → mean/std/percentiles residual.
    """
    samples: list[np.ndarray] = []
    for _ in range(int(n_samples)):
        pred = model(X, training=training)
        arr = np.asarray(pred).reshape(-1)
        samples.append(arr)
    stack = np.stack(samples, axis=0)
    return dict(
        mean_pred=stack.mean(axis=0),
        std_pred=stack.std(axis=0),
        percentile_5=np.percentile(stack, 5, axis=0),
        percentile_95=np.percentile(stack, 95, axis=0),
    )


def fit_quantile_gbm(
    X_train: np.ndarray,
    y_train: np.ndarray,
    *,
    quantiles: list[float] | tuple[float, ...] = (0.05, 0.5, 0.95),
    sample_weight: np.ndarray | None = None,
    max_depth: int = 8,
    learning_rate: float = 0.06,
    max_iter: int = 400,
) -> dict[float, HistGradientBoostingRegressor]:
    """Fit một HistGBM riêng cho mỗi quantile (sklearn loss='quantile')."""
    models: dict[float, HistGradientBoostingRegressor] = {}
    for q in quantiles:
        reg = HistGradientBoostingRegressor(
            loss="quantile",
            quantile=float(q),
            max_depth=max_depth,
            learning_rate=learning_rate,
            max_iter=max_iter,
            early_stopping=True,
            validation_fraction=0.15,
            n_iter_no_change=20,
            random_state=42,
        )
        reg.fit(X_train, y_train, sample_weight=sample_weight)
        models[float(q)] = reg
    return models


def predict_quantile_gbm(
    models: dict[float, HistGradientBoostingRegressor],
    X: np.ndarray,
) -> dict[str, np.ndarray]:
    """Dự báo residual theo từng quantile đã fit."""
    out: dict[str, np.ndarray] = {}
    for q, m in sorted(models.items()):
        out[f"q{int(round(q * 100)):02d}"] = m.predict(X)
    if 0.5 in models:
        out["median"] = models[0.5].predict(X)
    if 0.05 in models:
        out["p05"] = models[0.05].predict(X)
    if 0.95 in models:
        out["p95"] = models[0.95].predict(X)
    if 0.75 in models:
        out["p75"] = models[0.75].predict(X)
    return out


def demand_from_quantile_gbm(
    quantile_models: dict[float, Any],
    X_num: np.ndarray,
    log_baseline: np.ndarray,
    *,
    quantile: float = 0.75,
    resid_clip: tuple[float, float] = (-0.55, 0.55),
) -> np.ndarray:
    """Demand conservative từ quantile GBM residual (vd. p75 cho scenario mưa/cực đoan)."""
    preds = predict_quantile_gbm(quantile_models, X_num)
    qk = f"q{int(round(float(quantile) * 100)):02d}"
    if qk in preds:
        resid = preds[qk]
    elif float(quantile) in quantile_models:
        resid = quantile_models[float(quantile)].predict(X_num)
    else:
        resid = preds.get("median", preds.get("q50", next(iter(preds.values()))))
    resid = np.clip(np.asarray(resid, dtype=float).reshape(-1), *resid_clip)
    return residuals_to_demand(log_baseline, resid, resid_clip)


def residual_interval_to_demand(
    log_baseline: np.ndarray,
    resid_center: np.ndarray,
    *,
    resid_low: np.ndarray | None = None,
    resid_high: np.ndarray | None = None,
    resid_std: np.ndarray | None = None,
    clip: tuple[float, float] = (-0.55, 0.55),
) -> dict[str, np.ndarray]:
    """Map khoảng residual (log) → demand; giữ nguyên clip cho điểm / bound."""
    lb = np.asarray(log_baseline, dtype=float)
    rc = np.clip(np.asarray(resid_center, dtype=float), *clip)
    demand_mean = np.expm1(lb + rc)
    out: dict[str, np.ndarray] = {"demand_mean": demand_mean}
    if resid_low is not None:
        out["demand_p05"] = np.expm1(lb + np.asarray(resid_low, dtype=float))
    if resid_high is not None:
        out["demand_p95"] = np.expm1(lb + np.asarray(resid_high, dtype=float))
    if resid_std is not None:
        std = np.asarray(resid_std, dtype=float)
        out["demand_std"] = demand_mean * std
        out["demand_p75"] = np.expm1(lb + rc + 0.6745 * std)
    if "demand_p05" in out and "demand_p95" in out:
        out["interval_width"] = out["demand_p95"] - out["demand_p05"]
    return out


def uncertainty_aware_optimize(
    demand_mean: np.ndarray,
    demand_std: np.ndarray,
    trips_min: np.ndarray,
    trips_max: np.ndarray,
    lambda_cost: float,
    risk_level: str = "moderate",
    *,
    demand_p75: np.ndarray | None = None,
    demand_p05: np.ndarray | None = None,
    demand_p95: np.ndarray | None = None,
) -> tuple[np.ndarray, tuple[np.ndarray, np.ndarray]]:
    """Lịch conservative theo uncertainty; confidence_band = (trips_low, trips_high)."""
    mean = np.asarray(demand_mean, dtype=float)
    std = np.asarray(demand_std, dtype=float)
    tmin = np.asarray(trips_min, dtype=float)
    tmax = np.asarray(trips_max, dtype=float)
    lam = float(lambda_cost)
    level = str(risk_level).strip().lower()

    if level == "aggressive":
        d_opt = mean.copy()
    elif level == "moderate":
        d_opt = mean + 0.5 * std
    elif level == "conservative":
        d_opt = (
            np.asarray(demand_p75, dtype=float)
            if demand_p75 is not None
            else mean + 0.6745 * std
        )
    else:
        raise ValueError(f"risk_level không hợp lệ: {risk_level!r}")

    def _opt(d: np.ndarray) -> np.ndarray:
        trips_star = np.sqrt(30.0 * np.maximum(d, 1e-9) / lam)
        return np.clip(np.round(trips_star), tmin, tmax).astype(int)

    trips_opt = _opt(d_opt)
    d_low = (
        np.asarray(demand_p05, dtype=float)
        if demand_p05 is not None
        else np.maximum(mean - std, 0.0)
    )
    d_high = np.asarray(demand_p95, dtype=float) if demand_p95 is not None else mean + std
    return trips_opt, (_opt(d_low), _opt(d_high))


def compute_interval_calibration(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    std_pred: np.ndarray,
    *,
    nominal_levels: np.ndarray | list[float] | None = None,
) -> pd.DataFrame:
    """Coverage thực tế vs nominal khi dùng Gaussian interval y_pred ± z·std."""
    from scipy import stats

    yt = np.asarray(y_true, dtype=float)
    yp = np.asarray(y_pred, dtype=float)
    sd = np.maximum(np.asarray(std_pred, dtype=float), 1e-9)
    levels = (
        np.linspace(0.5, 0.95, 10)
        if nominal_levels is None
        else np.asarray(nominal_levels, dtype=float)
    )
    rows = []
    for nom in levels:
        z = float(stats.norm.ppf(0.5 + nom / 2.0))
        lower = yp - z * sd
        upper = yp + z * sd
        covered = (yt >= lower) & (yt <= upper)
        rows.append(
            dict(
                nominal=float(nom),
                empirical=float(covered.mean()),
                z_score=z,
            )
        )
    return pd.DataFrame(rows)


def compute_quantile_calibration(
    y_true: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    *,
    nominal: float = 0.9,
) -> dict[str, float]:
    """Coverage một khoảng quantile cố định (vd. q05–q95 → nominal 0.9)."""
    yt = np.asarray(y_true, dtype=float)
    lo = np.asarray(lower, dtype=float)
    hi = np.asarray(upper, dtype=float)
    covered = (yt >= lo) & (yt <= hi)
    return dict(nominal=float(nominal), empirical=float(covered.mean()))


def plot_calibration_curve(
    calibration_df: pd.DataFrame,
    *,
    ax: Any | None = None,
    label: str = "",
    title: str = "Interval calibration",
    save_path: Path | str | None = None,
) -> Any:
    """Predicted interval coverage vs nominal (đường y=x = lý tưởng)."""
    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots(figsize=(5, 5))
    ax.plot(
        [0, 1],
        [0, 1],
        "k--",
        lw=1,
        alpha=0.6,
        label="Perfect",
    )
    ax.plot(
        calibration_df["nominal"],
        calibration_df["empirical"],
        "o-",
        label=label or "Model",
    )
    ax.set_xlabel("Nominal coverage")
    ax.set_ylabel("Empirical coverage")
    ax.set_title(title)
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    if save_path is not None:
        ax.figure.tight_layout()
        ax.figure.savefig(save_path, dpi=120, bbox_inches="tight")
    return ax


def plot_std_vs_forecast_error(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    std_pred: np.ndarray,
    *,
    ax: Any | None = None,
    title: str = "Predicted std vs |forecast error|",
    save_path: Path | str | None = None,
) -> Any:
    """Validate std có tương quan với |error| không."""
    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots(figsize=(6, 5))
    err = np.abs(np.asarray(y_true, dtype=float) - np.asarray(y_pred, dtype=float))
    sd = np.asarray(std_pred, dtype=float)
    ax.scatter(sd, err, alpha=0.35, s=14, edgecolors="none")
    if len(sd) > 2:
        corr = float(np.corrcoef(sd, err)[0, 1])
        ax.set_title(f"{title} (r={corr:.2f})")
    else:
        ax.set_title(title)
    ax.set_xlabel("Predicted std")
    ax.set_ylabel("|Forecast error|")
    ax.grid(True, alpha=0.3)
    if save_path is not None:
        ax.figure.tight_layout()
        ax.figure.savefig(save_path, dpi=120, bbox_inches="tight")
    return ax


def plot_interval_width_by_hour(
    hours: np.ndarray,
    interval_width: np.ndarray,
    *,
    ax: Any | None = None,
    title: str = "Prediction interval width by hour",
    save_path: Path | str | None = None,
) -> Any:
    """TB độ rộng interval theo giờ; highlight peak (7–9, 17–19)."""
    import matplotlib.pyplot as plt

    df = pd.DataFrame(
        {
            "hour": np.asarray(hours, dtype=int),
            "width": np.asarray(interval_width, dtype=float),
        }
    )
    agg = df.groupby("hour", as_index=False)["width"].mean()
    if ax is None:
        _, ax = plt.subplots(figsize=(9, 4))
    ax.bar(agg["hour"], agg["width"], color="#6baed6", edgecolor="#2171b5", linewidth=0.4)
    for h in (7, 8, 9, 17, 18, 19):
        ax.axvspan(h - 0.4, h + 0.4, color="orange", alpha=0.15, lw=0)
    ax.set_xlabel("Hour")
    ax.set_ylabel("Mean interval width (demand)")
    ax.set_title(title)
    ax.grid(True, axis="y", alpha=0.3)
    if save_path is not None:
        ax.figure.tight_layout()
        ax.figure.savefig(save_path, dpi=120, bbox_inches="tight")
    return ax


def smape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    denom = (np.abs(y_true) + np.abs(y_pred)) / 2.0
    mask = denom > 0
    if mask.sum() == 0:
        return float("nan")
    return float(np.mean(np.abs(y_true[mask] - y_pred[mask]) / denom[mask]) * 100)


def regression_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    mape_min_demand: float = 100.0,
) -> dict[str, float]:
    """MAE, RMSE, R², MAPE (%), SMAPE (%) trên demand thực."""
    from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, r2_score

    yt = np.asarray(y_true, dtype=float)
    yp = np.asarray(y_pred, dtype=float)
    mask = yt > mape_min_demand
    mape = (
        float(mean_absolute_percentage_error(yt[mask], yp[mask]) * 100)
        if mask.any()
        else float("nan")
    )
    return {
        "mae": float(mean_absolute_error(yt, yp)),
        "rmse": float(np.sqrt(np.mean((yt - yp) ** 2))),
        "r2": float(r2_score(yt, yp)),
        "mape_pct": mape,
        "smape_pct": smape(yt, yp),
    }


# --- Module 1–3: lịch chi tiết ga, fleet sweep-line, mô phỏng trễ ---


def build_station_schedule_for_route(
    opt_trips: np.ndarray,
    slot_route: np.ndarray,
    slot_dir: np.ndarray,
    slot_hour: np.ndarray,
    schedule_dir: Path,
    route_id: str,
    *,
    offset_templates: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Lấy slot của 1 route từ kết quả optimizer, build schedule rồi expand ra từng ga."""
    mask = np.asarray(slot_route) == str(route_id)
    sched_df = pd.DataFrame({
        "route_id": np.asarray(slot_route)[mask],
        "direction_id": np.asarray(slot_dir)[mask],
        "hour": np.asarray(slot_hour)[mask],
        "opt_trips": np.asarray(opt_trips)[mask],
    })
    return expand_schedule_to_station_times(
        sched_df,
        schedule_dir,
        offset_templates=offset_templates,
        route_id=str(route_id),
    )


def assign_trip_ids(station_schedule: pd.DataFrame) -> pd.DataFrame:
    """Gán trip_id (0,1,2,...) theo thứ tự khởi hành trong mỗi (route, direction)."""
    out = station_schedule.copy()
    if out.empty:
        out["trip_id"] = pd.Series(dtype=int)
        return out

    gcols = ["route", "direction", "hour"]
    min_seq = out.groupby(gcols)["stop_sequence"].transform("min")
    out["_is_first"] = out["stop_sequence"] == min_seq

    def _stop_offsets(sub: pd.DataFrame) -> pd.DataFrame:
        sub = sub.copy()
        first_dep = sub.loc[sub["stop_sequence"] == sub["stop_sequence"].min(), "scheduled_min"].iloc[0]
        sub["_offset"] = sub["scheduled_min"] - float(first_dep)
        return sub

    parts: list[pd.DataFrame] = []
    for _, sub in out.groupby(gcols, sort=False):
        parts.append(_stop_offsets(sub))
    out = pd.concat(parts).sort_index()
    out["_first_dep_key"] = (out["scheduled_min"] - out["_offset"]).round(4)

    trip_map = (
        out.drop_duplicates(["route", "direction", "_first_dep_key"])
        .sort_values(["route", "direction", "scheduled_min"])
        .assign(trip_id=lambda d: d.groupby(["route", "direction"]).cumcount())
        [["route", "direction", "_first_dep_key", "trip_id"]]
    )
    out = out.merge(trip_map, on=["route", "direction", "_first_dep_key"], how="left")
    drop_cols = ["_is_first", "_offset", "_first_dep_key"]
    out = out.drop(columns=[c for c in drop_cols if c in out.columns])
    return out.sort_values(["route", "direction", "trip_id", "stop_sequence"]).reset_index(drop=True)


def _generate_route_direction_departures(
    trips: np.ndarray,
    hours: np.ndarray,
) -> list[float]:
    """Sinh departure_min cho từng trip — cùng convention với expand_schedule_to_station_times."""
    departures: list[float] = []
    slot_trips = np.maximum(np.asarray(trips, dtype=float), 0.0)
    slot_hours = np.asarray(hours, dtype=int)
    for n_trips, hour in zip(slot_trips, slot_hours):
        n = int(round(n_trips))
        if n <= 0:
            continue
        hour_start = int(hour) * 60
        spacing = 60.0 / n
        for trip_i in range(n):
            departures.append(hour_start + (trip_i + 0.5) * spacing)
    return departures


def _fleet_sweep_max(departures: list[float], cycle_time_min: float) -> float:
    """Sweep-line: max số tàu đồng thời trên timeline [0, 1440) với wrap-around."""
    cycle = max(float(cycle_time_min), 1e-6)
    day_min = 1440.0
    events: list[tuple[float, int]] = []
    for dep in departures:
        d = float(dep)
        end = d + cycle
        events.append((d, 1))
        if end <= day_min:
            events.append((end, -1))
        else:
            events.append((0.0, 1))
            events.append((end - day_min, -1))
    if not events:
        return 0.0
    events.sort(key=lambda x: (x[0], x[1]))
    running = 0.0
    peak = 0.0
    for t, delta in events:
        if t > day_min:
            break
        running += delta
        peak = max(peak, running)
    return float(peak)


def compute_fleet_continuous_sweep(
    trips: np.ndarray,
    slot_route: np.ndarray,
    slot_dir: np.ndarray,
    slot_hour: np.ndarray,
    cycle_times: pd.DataFrame,
) -> tuple[pd.DataFrame, float]:
    """Fleet chính xác bằng sweep-line trên timeline liên tục (Module 2)."""
    ct = cycle_times.set_index(["route_id", "direction_id"])["cycle_time_min"]
    median_cycle = float(ct.median()) if len(ct) else 90.0

    lb_df, _lb_system = compute_fleet_limits_from_baseline(
        trips, slot_route, slot_dir, slot_hour, cycle_times
    )
    lb_map = lb_df.set_index(["route_id", "direction_id"])["fleet_size"]

    routes = np.asarray(slot_route)
    dirs = np.asarray(slot_dir, dtype=int)
    hours = np.asarray(slot_hour, dtype=int)
    trip_arr = np.asarray(trips, dtype=float)

    groups: dict[tuple[str, int], list[int]] = {}
    for i in range(len(trip_arr)):
        groups.setdefault((str(routes[i]), int(dirs[i])), []).append(i)

    rows: list[dict[str, Any]] = []
    all_events: list[tuple[float, int]] = []

    for (rid, direction), indices in sorted(groups.items()):
        dep_list = _generate_route_direction_departures(
            trip_arr[indices], hours[indices]
        )
        cycle = float(ct.get((rid, direction), np.nan))
        if not np.isfinite(cycle) or cycle <= 0:
            cycle = median_cycle
        fleet_cont = _fleet_sweep_max(dep_list, cycle)
        fleet_lb = float(lb_map.get((rid, direction), 0.0))
        pct_under = (
            (fleet_cont - fleet_lb) / max(fleet_cont, 1e-9) * 100.0
            if fleet_cont > 0
            else 0.0
        )
        rows.append(
            dict(
                route_id=rid,
                direction_id=direction,
                fleet_size=round(fleet_cont, 3),
                fleet_size_lower_bound=round(fleet_lb, 3),
                fleet_size_continuous=round(fleet_cont, 3),
                pct_underestimate=round(pct_under, 3),
            )
        )
        for dep in dep_list:
            end = dep + cycle
            all_events.append((dep, 1))
            if end <= 1440.0:
                all_events.append((end, -1))
            else:
                all_events.append((0.0, 1))
                all_events.append((end - 1440.0, -1))

    system_max = 0.0
    if all_events:
        all_events.sort(key=lambda x: (x[0], x[1]))
        running = 0.0
        for t, delta in all_events:
            if t > 1440.0:
                break
            running += delta
            system_max = max(system_max, running)

    return pd.DataFrame(rows), float(system_max)


def compare_fleet_estimates(
    baseline_trips: np.ndarray,
    slot_route: np.ndarray,
    slot_dir: np.ndarray,
    slot_hour: np.ndarray,
    cycle_times: pd.DataFrame,
) -> pd.DataFrame:
    """So sánh fleet lower-bound (Little's Law) vs continuous sweep-line."""
    cont_df, _ = compute_fleet_continuous_sweep(
        baseline_trips, slot_route, slot_dir, slot_hour, cycle_times
    )
    return (
        cont_df.rename(
            columns={
                "fleet_size_lower_bound": "fleet_lower_bound",
                "fleet_size_continuous": "fleet_continuous",
            }
        )[
            ["route_id", "direction_id", "fleet_lower_bound", "fleet_continuous", "pct_underestimate"]
        ]
        .sort_values("pct_underestimate", ascending=False)
        .reset_index(drop=True)
    )


def simulate_delay_propagation(
    station_schedule: pd.DataFrame,
    *,
    route_id: str,
    direction_id: int,
    delayed_trip_id: int,
    delay_min: float,
    delay_at_stop_sequence: int,
    min_headway_ratio: float = 0.6,
    max_propagation_trips: int = 3,
    dwell_recovery_min: float = 0.5,
) -> pd.DataFrame:
    """Mô phỏng lan truyền trễ tối thiểu trên lịch chi tiết ga (Module 3)."""
    if "trip_id" not in station_schedule.columns:
        raise ValueError("station_schedule cần cột trip_id — gọi assign_trip_ids() trước")

    out = station_schedule.copy()
    out["actual_min"] = out["scheduled_min"].astype(float)
    out["delay_applied_min"] = 0.0
    out["is_affected"] = False
    out["propagation_hop"] = np.nan

    route_col = "route" if "route" in out.columns else "route_id"
    dir_col = "direction" if "direction" in out.columns else "direction_id"
    mask = (
        out[route_col].astype(str) == str(route_id)
    ) & (out[dir_col].astype(int) == int(direction_id))
    if not mask.any():
        raise ValueError(f"Không có dữ liệu cho route={route_id!r} direction={direction_id}")

    sub_idx = out.index[mask]

    trips = sorted(out.loc[sub_idx, "trip_id"].unique())
    if delayed_trip_id not in trips:
        raise ValueError(f"delayed_trip_id={delayed_trip_id} không có trong lịch")

    def _apply_delay_to_trip(tid: int, base_delay: float, at_seq: int, hop: int) -> None:
        tmask = mask & (out["trip_id"] == tid)
        for idx in out.index[tmask]:
            seq = int(out.at[idx, "stop_sequence"])
            if seq < at_seq:
                continue
            extra = max(0.0, base_delay - dwell_recovery_min * (seq - at_seq))
            sched = float(out.at[idx, "scheduled_min"])
            out.at[idx, "actual_min"] = sched + extra
            out.at[idx, "delay_applied_min"] = extra
            out.at[idx, "is_affected"] = True
            out.at[idx, "propagation_hop"] = float(hop)

    _apply_delay_to_trip(
        int(delayed_trip_id), float(delay_min), int(delay_at_stop_sequence), hop=0
    )

    pos = trips.index(int(delayed_trip_id))
    for hop in range(1, int(max_propagation_trips) + 1):
        if pos + hop >= len(trips):
            break
        trip_before = int(trips[pos + hop - 1])
        trip_after = int(trips[pos + hop])

        stops_after = (
            out.loc[mask & (out["trip_id"] == trip_after)]
            .sort_values("stop_sequence")
        )
        violated = False
        for _, row in stops_after.iterrows():
            seq = int(row["stop_sequence"])
            before = out.loc[mask & (out["trip_id"] == trip_before) & (out["stop_sequence"] == seq)]
            if before.empty:
                continue
            headway_planned = float(row["scheduled_min"]) - float(before["scheduled_min"].iloc[0])
            if headway_planned <= 0:
                continue
            actual_before = float(before["actual_min"].iloc[0])
            headway_if_no_fix = float(row["scheduled_min"]) - actual_before
            if headway_if_no_fix < float(min_headway_ratio) * headway_planned:
                hold_min = float(min_headway_ratio) * headway_planned - headway_if_no_fix
                violated = True
                for idx in out.index[mask & (out["trip_id"] == trip_after) & (out["stop_sequence"] >= seq)]:
                    seq2 = int(out.at[idx, "stop_sequence"])
                    decayed = max(0.0, hold_min - dwell_recovery_min * (seq2 - seq))
                    sched = float(out.at[idx, "scheduled_min"])
                    out.at[idx, "actual_min"] = max(float(out.at[idx, "actual_min"]), sched + decayed)
                    applied = float(out.at[idx, "actual_min"]) - sched
                    out.at[idx, "delay_applied_min"] = applied
                    out.at[idx, "is_affected"] = True
                    out.at[idx, "propagation_hop"] = float(hop)
                break

        if not violated:
            break

    out["actual_time"] = out["actual_min"].map(_minutes_to_hhmm)
    if "delay_applied_min" not in out.columns:
        out["delay_applied_min"] = out["actual_min"] - out["scheduled_min"]
    return out.sort_values([route_col, dir_col, "trip_id", "stop_sequence"]).reset_index(drop=True)


def summarize_delay_impact(adjusted_schedule: pd.DataFrame) -> dict[str, Any]:
    """Tóm tắt ảnh hưởng mô phỏng trễ."""
    df = adjusted_schedule.copy()
    if "is_affected" not in df.columns:
        return dict(
            n_trips_affected=0,
            n_stops_affected=0,
            max_hold_min=0.0,
            total_hold_min=0.0,
            affected_trip_ids=[],
            worst_stop=None,
        )

    affected = df[df["is_affected"].astype(bool)].copy()
    delay_col = "delay_applied_min" if "delay_applied_min" in affected.columns else None
    if delay_col is None:
        affected["delay_applied_min"] = affected["actual_min"] - affected["scheduled_min"]
        delay_col = "delay_applied_min"

    worst_stop = None
    if not affected.empty:
        wi = affected["delay_applied_min"].idxmax()
        worst_stop = dict(
            parent_stop_id=str(affected.at[wi, "parent_stop_id"]) if "parent_stop_id" in affected.columns else "",
            stop_name=str(affected.at[wi, "stop_name"]) if "stop_name" in affected.columns else "",
            max_delay_min=float(affected.at[wi, delay_col]),
        )

    trip_ids = sorted({int(x) for x in affected["trip_id"].unique()}) if "trip_id" in affected.columns else []
    return dict(
        n_trips_affected=int(affected["trip_id"].nunique()) if "trip_id" in affected.columns else 0,
        n_stops_affected=int(len(affected)),
        max_hold_min=float(affected[delay_col].max()) if not affected.empty else 0.0,
        total_hold_min=float(affected[delay_col].sum()) if not affected.empty else 0.0,
        affected_trip_ids=trip_ids,
        worst_stop=worst_stop,
    )


def plot_delay_propagation(
    adjusted_schedule: pd.DataFrame,
    *,
    route_id: str,
    direction_id: int,
    ax: Any | None = None,
    title: str = "Delay propagation",
    save_path: Path | str | None = None,
) -> Any:
    """Time-distance diagram: actual_min vs stop_sequence theo trip."""
    import matplotlib.pyplot as plt

    route_col = "route" if "route" in adjusted_schedule.columns else "route_id"
    dir_col = "direction" if "direction" in adjusted_schedule.columns else "direction_id"

    sub = adjusted_schedule[
        (adjusted_schedule[route_col].astype(str) == str(route_id))
        & (adjusted_schedule[dir_col].astype(int) == int(direction_id))
    ].copy()
    if sub.empty:
        raise ValueError("Không có dữ liệu để vẽ")

    if ax is None:
        _, ax = plt.subplots(figsize=(10, 6))

    time_col = "actual_min" if "actual_min" in sub.columns else "scheduled_min"
    for tid, grp in sub.groupby("trip_id"):
        grp = grp.sort_values("stop_sequence")
        hop = grp["propagation_hop"].dropna()
        hop_val = float(hop.iloc[0]) if len(hop) else np.nan
        affected = bool(grp["is_affected"].any()) if "is_affected" in grp.columns else False
        if affected and hop_val == 0:
            color, lw, alpha = "#d62728", 2.0, 0.95
        elif affected and hop_val >= 1:
            color, lw, alpha = "#ff7f0e", 1.6, max(0.35, 0.85 - 0.15 * hop_val)
        else:
            color, lw, alpha = "#bdbdbd", 0.8, 0.45
        ax.plot(
            grp[time_col],
            grp["stop_sequence"],
            color=color,
            linewidth=lw,
            alpha=alpha,
            marker="o",
            markersize=3,
            label=f"trip {int(tid)}" if affected else None,
        )

    ax.set_xlabel("Thời gian (phút từ 00:00)")
    ax.set_ylabel("stop_sequence")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    if save_path is not None:
        ax.figure.tight_layout()
        ax.figure.savefig(save_path, dpi=120, bbox_inches="tight")
    return ax
