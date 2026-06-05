# -*- coding: utf-8 -*-
"""Pipeline MTA: paths (Kaggle/local), preset thí nghiệm, helpers NN/ML — xem LITERATURE.md."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.ensemble import HistGradientBoostingRegressor
from tensorflow.keras import Input, Model, layers

# --- Literature-backed defaults (ITM 2026 MTA LSTM; weather mixed-effects) ---
DEFAULT_LAG_COLS = ["log_lag_24h", "log_lag_168h", "log_rolling_7d"]
DEFAULT_SEQ_LEN = 7  # 7 ngày cùng giờ (chuỗi ngắn hạn)

SEASON_ORDER = ("winter", "spring", "summer", "autumn")


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


def hour_slot_group(hour: int) -> str:
    """Nhóm giờ cho báo cáo bound: peak / overnight / off_peak."""
    h = int(hour)
    if h in (7, 8, 9, 17, 18, 19):
        return "peak"
    if h <= 6 or h >= 23:
        return "overnight"
    return "off_peak"


def build_dynamic_bounds(
    baseline_trips: np.ndarray,
    slot_hour: np.ndarray,
    *,
    peak_factor: float = 1.15,
    offpeak_factor: float = 1.35,
    overnight_factor: float = 1.10,
    min_factor: float = 0.5,
    max_delta: int = 3,
    absolute_max: int = 60,
    min_trips: int = 2,
) -> tuple[np.ndarray, np.ndarray]:
    """Per-slot TRIPS_MIN / TRIPS_MAX theo nhóm giờ (peak / off-peak / overnight)."""
    base = np.asarray(baseline_trips, dtype=float)
    hrs = np.asarray(slot_hour, dtype=int)
    is_peak = np.isin(hrs, (7, 8, 9, 17, 18, 19))
    is_overnight = (hrs <= 6) | (hrs >= 23)
    factors = np.full(len(base), float(offpeak_factor))
    factors[is_peak] = float(peak_factor)
    factors[is_overnight] = float(overnight_factor)

    trips_min = np.maximum(min_trips, np.floor(base * min_factor)).astype(int)
    trips_max = np.minimum(absolute_max, np.floor(base * factors)).astype(int)
    trips_max = np.maximum(trips_max, base.astype(int) + int(max_delta))
    return trips_min, trips_max


def report_bound_status(
    opt_trips: np.ndarray,
    trips_min: np.ndarray,
    trips_max: np.ndarray,
    slot_hour: np.ndarray,
    *,
    target_interior_pct: float = 60.0,
    target_peak_interior_pct: float = 40.0,
    verbose: bool = True,
    label: str = "",
) -> dict[str, Any]:
    """% slot ở min / max / interior — tổng và theo nhóm giờ; kiểm tra target."""
    t = np.asarray(opt_trips, dtype=int)
    tmin = np.asarray(trips_min, dtype=int)
    tmax = np.asarray(trips_max, dtype=int)
    at_min = float((t <= tmin).mean()) * 100
    at_max = float((t >= tmax).mean()) * 100
    interior = 100.0 - at_min - at_max

    by_group = bound_status_by_hour_groups(t, tmin, tmax, slot_hour)
    peak_interior = float("nan")
    if not by_group.empty and (by_group["hour_group"] == "peak").any():
        peak_interior = float(by_group.loc[by_group["hour_group"] == "peak", "interior_pct"].iloc[0])

    ok_total = interior >= target_interior_pct
    ok_peak = np.isnan(peak_interior) or peak_interior >= target_peak_interior_pct

    if verbose:
        prefix = f"[{label}] " if label else ""
        print(
            f"  {prefix}at TRIPS_MIN={at_min:.0f}% | TRIPS_MAX={at_max:.0f}% | "
            f"interior={interior:.0f}%"
            f" {'✓' if ok_total else '✗'} (target>{target_interior_pct:.0f}%)"
        )
        for _, row in by_group.iterrows():
            mark = ""
            if row["hour_group"] == "peak":
                mark = " ✓" if row["interior_pct"] >= target_peak_interior_pct else " ✗"
            print(
                f"    {row['hour_group']:10s} n={int(row['n_slots']):4d} | "
                f"min={row['at_min_pct']:.0f}% max={row['at_max_pct']:.0f}% "
                f"interior={row['interior_pct']:.0f}%{mark}"
            )

    return {
        "at_min_pct": at_min,
        "at_max_pct": at_max,
        "interior_pct": interior,
        "peak_interior_pct": peak_interior,
        "ok_total_interior": ok_total,
        "ok_peak_interior": ok_peak,
        "by_hour_group": by_group,
    }


def bound_status_by_hour_groups(
    trips: np.ndarray,
    trips_min: np.ndarray,
    trips_max: np.ndarray,
    hours: np.ndarray,
) -> pd.DataFrame:
    """% slot ở min / max / interior theo nhóm giờ (peak, overnight, off_peak)."""
    t = np.asarray(trips, dtype=int)
    tmin = np.asarray(trips_min, dtype=int)
    tmax = np.asarray(trips_max, dtype=int)
    hrs = np.asarray(hours, dtype=int)
    groups = np.array([hour_slot_group(h) for h in hrs])
    rows = []
    for g in ("peak", "overnight", "off_peak"):
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


def build_transfer_matrix(
    routes_by_station_df: pd.DataFrame,
    route_list: list[str],
    *,
    route_col: str | None = None,
    route_aliases: dict[str, str] | None = None,
    min_weight: float = 0.1,
) -> pd.DataFrame:
    """Ma trận transfer N×N: shared stations / sqrt(n_r1 × n_r2); zero nếu ≤ min_weight."""
    raw = routes_by_station_df.copy()
    col = route_col
    if col is None:
        col = "gtfs_route_ids" if "gtfs_route_ids" in raw.columns else "route_ids_gtfs"
    aliases = route_aliases or {}

    station_routes = (
        raw.assign(route=raw[col].astype(str).str.split())
        .explode("route")
        .dropna(subset=["route"])
    )
    station_routes = station_routes[station_routes["route"] != ""]
    station_routes = station_routes[["station_complex_id", "route"]].drop_duplicates()
    station_routes["route"] = station_routes["route"].replace(aliases)

    routes = [str(r).strip() for r in route_list]
    routes = sorted(set(routes))
    route_station_sets: dict[str, set] = {r: set() for r in routes}
    for _, row in station_routes.iterrows():
        r = str(row["route"]).strip()
        if r in route_station_sets:
            route_station_sets[r].add(str(row["station_complex_id"]))

    n_stations = {r: max(len(route_station_sets[r]), 1) for r in routes}
    mat = pd.DataFrame(0.0, index=routes, columns=routes)
    for r1 in routes:
        s1 = route_station_sets[r1]
        for r2 in routes:
            shared = len(s1 & route_station_sets[r2])
            if shared <= 0:
                continue
            mat.loc[r1, r2] = shared / np.sqrt(n_stations[r1] * n_stations[r2])

    mat = mat.where(mat > float(min_weight), 0.0)
    return mat


def adjust_demand_with_spillover(
    demand_dict: dict[str, np.ndarray],
    transfer_matrix: pd.DataFrame,
    alpha: float = 0.05,
) -> dict[str, np.ndarray]:
    """Demand spillover: d_r[h] += alpha × Σ w[r,r2] × d_r2[h] (r2 ≠ r, w>0)."""
    a = float(alpha)
    adjusted: dict[str, np.ndarray] = {}
    for r, d in demand_dict.items():
        rs = str(r).strip()
        if rs not in transfer_matrix.index:
            adjusted[rs] = np.asarray(d, dtype=float).copy()
            continue
        d_r = np.asarray(d, dtype=float)
        spill = np.zeros_like(d_r, dtype=float)
        for r2 in transfer_matrix.columns:
            r2s = str(r2).strip()
            if r2s == rs or r2s not in demand_dict:
                continue
            w = float(transfer_matrix.loc[rs, r2s])
            if w <= 0.0:
                continue
            spill += w * np.asarray(demand_dict[r2s], dtype=float)
        adjusted[rs] = d_r + a * spill
    return adjusted


def route_hour_df_to_demand_dict(
    route_hour_df: pd.DataFrame,
    route_list: list[str],
    hours: list[int] | np.ndarray,
    *,
    demand_col: str = "demand",
) -> dict[str, np.ndarray]:
    """DataFrame (route_id, hour, demand) → {route: array(len(hours))}."""
    rh = route_hour_df.set_index(["route_id", "hour"])[demand_col]
    out: dict[str, np.ndarray] = {}
    hrs = [int(h) for h in hours]
    for r in route_list:
        rs = str(r).strip()
        out[rs] = np.array([float(rh.get((rs, h), 0.0)) for h in hrs], dtype=float)
    return out


def demand_dict_to_route_hour_df(
    demand_dict: dict[str, np.ndarray],
    hours: list[int] | np.ndarray,
) -> pd.DataFrame:
    """{route: demand[hours]} → long DataFrame route_id, hour, demand."""
    hrs = [int(h) for h in hours]
    rows = []
    for r, d in demand_dict.items():
        arr = np.asarray(d, dtype=float)
        for i, h in enumerate(hrs):
            rows.append(dict(route_id=str(r), hour=h, demand=float(arr[i])))
    return pd.DataFrame(rows)


def evaluate_spillover_forecast_mae(
    actual_dict: dict[str, np.ndarray],
    pred_dict: dict[str, np.ndarray],
    transfer_matrix: pd.DataFrame,
    alpha: float = 0.05,
) -> dict[str, float]:
    """MAE tổng trước/sau spillover adjustment trên demand_dict cùng format."""
    pred_adj = adjust_demand_with_spillover(pred_dict, transfer_matrix, alpha=alpha)
    abs_err_before: list[float] = []
    abs_err_after: list[float] = []
    for r, y_true in actual_dict.items():
        rs = str(r).strip()
        if rs not in pred_dict:
            continue
        yt = np.asarray(y_true, dtype=float)
        yb = np.asarray(pred_dict[rs], dtype=float)
        ya = np.asarray(pred_adj.get(rs, yb), dtype=float)
        n = min(len(yt), len(yb), len(ya))
        if n <= 0:
            continue
        abs_err_before.extend(np.abs(yt[:n] - yb[:n]))
        abs_err_after.extend(np.abs(yt[:n] - ya[:n]))
    mae_before = float(np.mean(abs_err_before)) if abs_err_before else float("nan")
    mae_after = float(np.mean(abs_err_after)) if abs_err_after else float("nan")
    return dict(
        mae_before=mae_before,
        mae_after=mae_after,
        mae_delta=mae_before - mae_after,
        alpha=float(alpha),
    )


def spillover_benefit_by_route(
    actual_dict: dict[str, np.ndarray],
    pred_dict: dict[str, np.ndarray],
    transfer_matrix: pd.DataFrame,
    alpha: float = 0.05,
) -> pd.DataFrame:
    """MAE trước/sau theo từng tuyến; benefit = mae_before − mae_after (dương = tốt hơn)."""
    pred_adj = adjust_demand_with_spillover(pred_dict, transfer_matrix, alpha=alpha)
    rows = []
    for r, y_true in actual_dict.items():
        rs = str(r).strip()
        if rs not in pred_dict:
            continue
        yt = np.asarray(y_true, dtype=float)
        yb = np.asarray(pred_dict[rs], dtype=float)
        ya = np.asarray(pred_adj.get(rs, yb), dtype=float)
        n = min(len(yt), len(yb), len(ya))
        if n <= 0:
            continue
        mae_b = float(np.mean(np.abs(yt[:n] - yb[:n])))
        mae_a = float(np.mean(np.abs(yt[:n] - ya[:n])))
        rows.append(
            dict(
                route_id=rs,
                mae_before=mae_b,
                mae_after=mae_a,
                benefit=mae_b - mae_a,
                n_hours=n,
            )
        )
    return pd.DataFrame(rows).sort_values("benefit", ascending=False).reset_index(drop=True)


def plot_transfer_heatmap(
    transfer_matrix: pd.DataFrame,
    *,
    top_n: int = 10,
    ax: Any | None = None,
    title: str = "Transfer matrix (top routes by connectivity)",
    save_path: Path | str | None = None,
) -> Any:
    """Heatmap top-N tuyến có tổng transfer weight lớn nhất."""
    import matplotlib.pyplot as plt
    import seaborn as sns

    degree = transfer_matrix.sum(axis=1).sort_values(ascending=False)
    top_routes = degree.head(int(top_n)).index.tolist()
    sub = transfer_matrix.loc[top_routes, top_routes]
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(
        sub,
        annot=True,
        fmt=".2f",
        cmap="YlOrRd",
        linewidths=0.4,
        cbar_kws={"label": "transfer weight"},
        ax=ax,
    )
    ax.set_title(title)
    if save_path is not None:
        ax.figure.tight_layout()
        ax.figure.savefig(save_path, dpi=120, bbox_inches="tight")
    return ax


def compute_wait_with_overflow(
    demand: np.ndarray,
    trips: np.ndarray,
    *,
    slot_route: np.ndarray,
    slot_dir: np.ndarray,
    slot_hour: np.ndarray,
    capacity_per_trip: float = 1200.0,
    lambda_cost: float = 150.0,
) -> dict[str, Any]:
    """Passenger-min wait có overflow theo chuỗi giờ (route × direction).

    Overflow từ giờ trước cộng vào demand giờ sau; hành khách kẹt chờ thêm ~1.5×headway.
    """
    d = np.clip(np.nan_to_num(np.asarray(demand, dtype=float), nan=0.0), 0.0, None)
    t = np.maximum(np.nan_to_num(np.asarray(trips, dtype=float), nan=1.0), 1.0)
    routes = np.asarray(slot_route)
    dirs = np.asarray(slot_dir, dtype=int)
    hours = np.asarray(slot_hour, dtype=int)
    n = len(d)
    cap_pt = float(capacity_per_trip)

    slot_wait = np.zeros(n, dtype=float)
    overflow_out = np.zeros(n, dtype=float)
    overflow_prev_arr = np.zeros(n, dtype=float)
    served_new_arr = np.zeros(n, dtype=float)

    groups: dict[tuple[str, int], list[int]] = {}
    for i in range(n):
        key = (str(routes[i]), int(dirs[i]))
        groups.setdefault(key, []).append(i)

    for indices in groups.values():
        overflow_prev = 0.0
        for i in sorted(indices, key=lambda idx: int(hours[idx])):
            capacity = t[i] * cap_pt
            demand_i = d[i]
            effective = demand_i + overflow_prev

            served_new = min(demand_i, max(0.0, capacity - overflow_prev))
            headway = 60.0 / t[i]
            wait_base = headway / 2.0
            wait_overflow = headway + wait_base

            overflow_prev_arr[i] = overflow_prev
            served_new_arr[i] = served_new
            slot_wait[i] = served_new * wait_base + overflow_prev * wait_overflow
            overflow_out[i] = max(0.0, effective - capacity)
            overflow_prev = overflow_out[i]

    total_wait = float(slot_wait.sum())
    total_demand = float(d.sum())
    overflow_slots = int((overflow_out > 0).sum())
    return dict(
        total_passenger_min_wait=total_wait,
        weighted_avg_wait_min=total_wait / max(total_demand, 1e-9),
        overflow_pct=float(overflow_slots / max(n, 1)) * 100.0,
        total_overflow_pax=float(overflow_out.sum()),
        total_fleet_cost=float(lambda_cost) * float(t.sum()),
        total_trips=float(t.sum()),
        objective=float(total_wait + float(lambda_cost) * float(t.sum())),
        lambda_cost=float(lambda_cost),
        slot_wait=slot_wait,
        overflow_out=overflow_out,
        overflow_prev=overflow_prev_arr,
        served_new=served_new_arr,
    )


def wait_totals_by_hour_groups(
    slot_wait: np.ndarray,
    slot_hour: np.ndarray,
) -> pd.DataFrame:
    """Tổng passenger-min wait theo nhóm giờ (peak / off_peak / overnight)."""
    sw = np.asarray(slot_wait, dtype=float)
    hrs = np.asarray(slot_hour, dtype=int)
    groups = np.array([hour_slot_group(int(h)) for h in hrs])
    rows = []
    for g in ("peak", "overnight", "off_peak"):
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
    base_f2 = float(base_m["total_fleet_cost"])

    candidates: list[dict[str, float]] = []
    for lam in lambda_scan:
        sol = np.asarray(optimize_fn(demand, float(lam)), dtype=int)
        m = evaluate_fn(sol, demand, lam_eval)
        candidates.append(
            dict(
                lambda_equiv=float(lam),
                f1=float(m["total_passenger_min_wait"]),
                f2=float(m["total_fleet_cost"]),
                total_trips=float(m["total_trips"]),
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
    pareto_df["z_star_f1"] = z_star[0]
    pareto_df["z_star_f2"] = z_star[1]
    return pareto_df


def plot_pareto_frontier(
    pareto_df: pd.DataFrame,
    baseline_f1: float,
    baseline_f2: float,
    *,
    ax: Any | None = None,
    title: str = "Pareto frontier: fleet cost vs passenger-min wait",
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
                f"λ={row['lambda_equiv']:.0f}\n{row['total_trips']:.0f} trips",
                (row["f2"], row["f1"]),
                textcoords="offset points",
                xytext=(4, 4),
                fontsize=6,
                alpha=0.85,
            )

    ax.set_xlabel("f2: fleet cost (λ × trips)")
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


def add_weather_interaction_features(d: pd.DataFrame) -> pd.DataFrame:
    """Tương tác thời tiết × peak (mixed-effects / tree papers trên MTA)."""
    out = d.copy()
    rain = pd.to_numeric(out.get("rain_mm", 0), errors="coerce").fillna(0)
    precip = pd.to_numeric(out.get("precipitation_mm", 0), errors="coerce").fillna(0)
    wind = pd.to_numeric(out.get("windspeed_kmh", 0), errors="coerce").fillna(0)
    out["log_rain_mm"] = np.log1p(rain.clip(lower=0))
    out["log_precip_mm"] = np.log1p(precip.clip(lower=0))
    peak = (
        (out.get("is_peak_morning", 0).fillna(0).astype(int) == 1)
        | (out.get("is_peak_evening", 0).fillna(0).astype(int) == 1)
    ).astype(float)
    out["rain_x_peak"] = rain * peak
    out["wind_x_rain"] = wind * out.get("is_rain", 0).fillna(0)
    return out


def add_lag_features(
    d: pd.DataFrame,
    *,
    use_lags: bool = True,
    lag_cols: list[str] | None = None,
) -> pd.DataFrame:
    """Lag cùng giờ: 24h, 168h, rolling 7d (ITM 2026 / LSTM MTA papers)."""
    lag_cols = lag_cols or DEFAULT_LAG_COLS
    if not use_lags:
        return d
    out = d.sort_values(["route_id", "hour", "date"]).copy()
    g = out.groupby(["route_id", "hour"])["demand"]
    out["demand_lag_24h"] = g.shift(1)
    out["demand_lag_168h"] = g.shift(7)
    out["demand_rolling_7d"] = g.transform(lambda s: s.shift(1).rolling(7, min_periods=1).mean())
    for raw, log_c in [
        ("demand_lag_24h", "log_lag_24h"),
        ("demand_lag_168h", "log_lag_168h"),
        ("demand_rolling_7d", "log_rolling_7d"),
    ]:
        out[log_c] = np.log1p(out[raw].clip(lower=0))
    return out.sort_values(["route_id", "date", "hour"]).reset_index(drop=True)


def fill_lag_from_train(
    d: pd.DataFrame,
    ref: pd.DataFrame,
    lag_cols: list[str] | None = None,
) -> pd.DataFrame:
    lag_cols = lag_cols or DEFAULT_LAG_COLS
    lag_medians = ref.groupby(["route_id", "hour"])[lag_cols].median().reset_index()
    out = d.merge(lag_medians, on=["route_id", "hour"], how="left", suffixes=("", "_fill"))
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
    extra_interactions: bool = True,
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
    if extra_interactions:
        feats.extend(["log_rain_mm", "log_precip_mm", "rain_x_peak", "wind_x_rain"])
    if use_lags:
        feats.extend(lag_cols)
    return feats


def prepare_lstm_sequences(
    frame: pd.DataFrame,
    feature_cols: list[str],
    *,
    seq_len: int = DEFAULT_SEQ_LEN,
    target_residual: bool = True,
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """Chuỗi seq_len ngày liên tiếp cùng (route, hour) — kiểu LSTM MTA hourly."""
    parts_X, parts_y, parts_meta = [], [], []
    for (_rid, _hour), g in frame.groupby(["route_id", "hour"]):
        g = g.sort_values("date").reset_index(drop=True)
        if len(g) <= seq_len:
            continue
        Xf = g[feature_cols].to_numpy(dtype=np.float32)
        if target_residual:
            y = (np.log1p(g["demand"].values) - g["log_baseline"].values).astype(np.float32)
        else:
            y = np.log1p(g["demand"].values).astype(np.float32)
        for i in range(seq_len, len(g)):
            parts_X.append(Xf[i - seq_len : i])
            parts_y.append(y[i])
            parts_meta.append(g.iloc[i])
    if not parts_X:
        raise ValueError("Không đủ chuỗi thời gian cho LSTM (cần > seq_len ngày / route×hour)")
    X_seq = np.stack(parts_X, axis=0)
    y_seq = np.array(parts_y, dtype=np.float32)
    meta = pd.DataFrame(parts_meta).reset_index(drop=True)
    return X_seq, y_seq, meta


def build_demand_model(
    n_routes: int,
    n_num_features: int,
    *,
    use_route_embedding: bool = False,
    hidden: tuple[int, ...] = (64, 32),
    dropout: float = 0.25,
    use_batch_norm: bool = True,
) -> Model:
    """MLP residual log-demand (Huber) — BatchNorm theo best practice DL traffic."""
    inp_num = Input(shape=(n_num_features,), name="num_features")
    x = inp_num
    inputs: list[Any] = [inp_num]

    if use_route_embedding and n_routes > 1:
        inp_route = Input(shape=(1,), name="route_idx", dtype="int32")
        route_emb = layers.Embedding(
            n_routes,
            4,
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

    out = layers.Dense(1, name="residual_log_demand", kernel_initializer="zeros")(x)
    model = Model(inputs, out)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(3e-4),
        loss=tf.keras.losses.Huber(delta=0.5),
        metrics=["mae"],
    )
    return model


def build_lstm_demand_model(
    seq_len: int,
    n_features: int,
    *,
    lstm_units: int = 32,
    dropout: float = 0.3,
) -> Model:
    """LSTM trên chuỗi ngắn (7×features) — theo hướng ITM 2026 MTA hourly."""
    inp = Input(shape=(seq_len, n_features), name="seq_features")
    x = layers.LSTM(lstm_units, name="lstm")(inp)
    x = layers.Dropout(dropout)(x)
    x = layers.Dense(16, activation="relu", name="dense_head")(x)
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
    """MC Dropout: n forward passes với dropout active → mean/std/percentiles residual."""
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
    ax.set_xlim(0, 0.5, 1.0)
    ax.set_ylim(0, 0.5, 1.0)
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


# --- Experiment presets (full-route notebook) ---
_CONFIG_KEYS = (
    "DEMAND_MODEL_TYPE",
    "USE_HISTGBM_BLEND",
    "CV_USE_BLEND",
    "CV_BLEND_TUNE_FRAC",
    "HOLDOUT_TEST_SEASON",
    "HOLDOUT_TEST_YEAR",
    "HOLDOUT_VAL_SEASON",
    "HOLDOUT_VAL_YEAR",
    "OPT_TARGET",
    "BALANCED_WEIGHTS",
    "LAMBDA_AUTO_CALIBRATE",
    "LAMBDA_COST",
    "LAMBDA_CANDIDATES",
    "TARGET_MAX_BIND_FRACTION",
    "USE_ROUTE_EMBEDDING",
    "USE_LAG_FEATURES",
    "NN_EPOCHS_MAIN",
    "NN_EPOCHS_CV",
    "LSTM_SEQ_LEN",
    "TUNE_RESID_CLIP_ON_VAL",
    "HISTGBM_BLEND_WEIGHT",
)

EXPERIMENTS: dict[str, dict[str, Any]] = {
    "default": {
        "title": "blend",
        "overrides": {
            "DEMAND_MODEL_TYPE": "blend",
            "USE_HISTGBM_BLEND": True,
            "CV_USE_BLEND": True,
            "TUNE_RESID_CLIP_ON_VAL": True,
            "LSTM_SEQ_LEN": 7,
            "HISTGBM_BLEND_WEIGHT": None,
        },
    },
    "model_mlp": {
        "title": "mlp",
        "overrides": {
            "DEMAND_MODEL_TYPE": "mlp",
            "USE_HISTGBM_BLEND": False,
            "CV_USE_BLEND": False,
            "TUNE_RESID_CLIP_ON_VAL": True,
            "LSTM_SEQ_LEN": 7,
        },
    },
    "model_lstm": {
        "title": "lstm",
        "overrides": {
            "DEMAND_MODEL_TYPE": "lstm",
            "USE_HISTGBM_BLEND": False,
            "CV_USE_BLEND": False,
            "TUNE_RESID_CLIP_ON_VAL": False,
            "LSTM_SEQ_LEN": 7,
            "NN_EPOCHS_MAIN": 120,
            "NN_EPOCHS_CV": 60,
        },
    },
}


def list_experiment_names() -> list[str]:
    return list(EXPERIMENTS.keys())


def apply_experiment(name: str, g: dict[str, Any]) -> None:
    """Áp preset theo RUN_EXPERIMENT (gồm DEMAND_MODEL_TYPE và flags liên quan)."""
    if name not in EXPERIMENTS:
        raise KeyError(f"Unknown experiment {name!r}. Use: {list_experiment_names()}")
    for key, val in (EXPERIMENTS[name].get("overrides") or {}).items():
        if key not in _CONFIG_KEYS:
            raise KeyError(f"Preset key not allowed: {key}")
        g[key] = val
