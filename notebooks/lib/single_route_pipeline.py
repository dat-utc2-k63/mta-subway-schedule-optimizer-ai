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


def build_seasonal_holdout_splits(
    dates: pd.Series | np.ndarray | list,
    *,
    test_season: str = "autumn",
    test_season_year: int = 2025,
    val_season: str = "summer",
    val_season_year: int = 2025,
) -> tuple[set, set, set]:
    """Hold-out theo mùa: test = một khối mùa; val = khối mùa khác; còn lại train."""
    all_dates = set(pd.to_datetime(pd.Series(dates).dropna().unique()))
    test_dates = dates_for_season_year(all_dates, test_season, test_season_year)
    val_dates = dates_for_season_year(all_dates, val_season, val_season_year)
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
    if h <= 5 or h >= 23:
        return "overnight"
    return "off_peak"


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
