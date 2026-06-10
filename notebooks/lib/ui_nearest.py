"""Tìm ngày training gần nhất trong không gian feature (chuẩn hóa L2)."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Any

import numpy as np
import pandas as pd

# Cột dùng cho vector ngày (24h × features) — khớp đầu vào demand model
VECTOR_COLS: tuple[str, ...] = (
    "temperature_c",
    "apparent_temperature_c",
    "precipitation_mm",
    "rain_mm",
    "snowfall_cm",
    "windspeed_kmh",
    "windgusts_kmh",
    "is_rain",
    "is_snow",
    "is_severe_wind",
    "is_peak_morning",
    "is_peak_evening",
    "is_overnight",
    "day_of_week",
    "is_weekend",
    "is_us_holiday",
    "month",
    "is_major_event_window",
)


@dataclass
class NearestDayResult:
    hourly_factors: pd.DataFrame
    nearest_dates: list[str]
    distance: float
    n_candidates: int
    is_self_match: bool
    note: str


@dataclass(frozen=True)
class _TrainingIndex:
    dates: tuple[pd.Timestamp, ...]
    matrix: np.ndarray
    mean: np.ndarray
    std: np.ndarray
    hourly: pd.DataFrame


def _normalize_dates(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["date"] = pd.to_datetime(out["date"]).dt.normalize()
    return out


def _fill_vector(values: np.ndarray, fill: np.ndarray) -> np.ndarray:
    out = values.astype(float).copy()
    bad = ~np.isfinite(out)
    if bad.any():
        out[bad] = fill[bad]
    return out


@lru_cache(maxsize=4)
def _load_training_index(factors_hourly_path: str, mtime: float) -> _TrainingIndex:
    hourly = _normalize_dates(pd.read_csv(factors_hourly_path, parse_dates=["date"]))
    dates = tuple(sorted(hourly["date"].unique()))
    rows: list[np.ndarray] = []
    for d in dates:
        day = hourly.loc[hourly["date"] == d].sort_values("hour")
        rows.append(_vectorize_hourly_day(day))
    matrix = np.vstack(rows)
    mean = np.nanmean(matrix, axis=0)
    std = np.nanstd(matrix, axis=0)
    std = np.where(std < 1e-9, 1.0, std)
    return _TrainingIndex(dates=dates, matrix=matrix, mean=mean, std=std, hourly=hourly)


def _vectorize_hourly_day(day_df: pd.DataFrame) -> np.ndarray:
    """Flatten 24 giờ × VECTOR_COLS thành một vector."""
    parts: list[float] = []
    for hour in range(24):
        hrows = day_df.loc[day_df["hour"] == hour]
        if hrows.empty:
            for _ in VECTOR_COLS:
                parts.append(np.nan)
            continue
        row = hrows.iloc[0]
        for col in VECTOR_COLS:
            if col not in row.index:
                parts.append(np.nan)
            else:
                parts.append(float(pd.to_numeric(row[col], errors="coerce")))
    return np.asarray(parts, dtype=float)


def vectorize_hourly_profile(hourly_factors: pd.DataFrame) -> np.ndarray:
    return _vectorize_hourly_day(hourly_factors.sort_values("hour"))


def _standardize(vec: np.ndarray, mean: np.ndarray, std: np.ndarray, fill: np.ndarray) -> np.ndarray:
    v = _fill_vector(vec, fill)
    return (v - mean) / std


def find_nearest_training_days(
    query_vector: np.ndarray,
    index: _TrainingIndex,
    *,
    tie_eps: float = 1e-4,
) -> tuple[list[pd.Timestamp], float]:
    """Trả về mọi ngày training có khoảng cách L2 chuẩn hóa tối thiểu."""
    fill = index.mean
    q = _standardize(query_vector, index.mean, index.std, fill)
    q = np.nan_to_num(q, nan=0.0)
    train = np.nan_to_num(
        (index.matrix - index.mean) / index.std,
        nan=0.0,
    )
    dists = np.linalg.norm(train - q, axis=1)
    best = float(dists.min())
    idx = np.where(dists <= best + tie_eps)[0]
    return [index.dates[int(i)] for i in idx], best


def _median_hourly_profile(hourly: pd.DataFrame, dates: list[pd.Timestamp]) -> pd.DataFrame:
    sub = hourly.loc[hourly["date"].isin(dates)].copy()
    if sub.empty:
        raise ValueError("Không có factors_hourly cho ngày gần nhất.")
    num_cols = [c for c in VECTOR_COLS if c in sub.columns]
    int_cols = [c for c in num_cols if c.startswith("is_") or c in ("day_of_week", "month")]
    float_cols = [c for c in num_cols if c not in int_cols]

    rows: list[dict[str, Any]] = []
    for hour in range(24):
        hrows = sub.loc[sub["hour"] == hour]
        if hrows.empty:
            raise ValueError(f"Thiếu giờ {hour} cho ngày gần nhất.")
        row: dict[str, Any] = {"hour": hour}
        for col in float_cols:
            row[col] = float(pd.to_numeric(hrows[col], errors="coerce").median())
        for col in int_cols:
            row[col] = int(round(float(pd.to_numeric(hrows[col], errors="coerce").median())))
        if "season" in hrows.columns:
            row["season"] = str(hrows["season"].mode().iloc[0])
        if "holiday_name" in hrows.columns:
            names = hrows["holiday_name"].dropna().astype(str)
            row["holiday_name"] = str(names.mode().iloc[0]) if not names.empty else ""
        rows.append(row)
    return pd.DataFrame(rows).sort_values("hour").reset_index(drop=True)


def resolve_to_nearest_training_day(
    query_hourly: pd.DataFrame,
    factors_hourly_path: str,
    *,
    file_mtime: float,
    query_date: pd.Timestamp | None = None,
) -> NearestDayResult:
    """
    Map profile truy vấn → ngày training gần nhất trong toàn bộ corpus (L2 chuẩn hóa).
  Dùng median 24h nếu có nhiều ngày cùng khoảng cách tối thiểu.
    """
    index = _load_training_index(factors_hourly_path, file_mtime)
    qvec = vectorize_hourly_profile(query_hourly)
    nearest_dates, dist = find_nearest_training_days(qvec, index)

    profile = _median_hourly_profile(index.hourly, nearest_dates)
    date_strs = [d.strftime("%Y-%m-%d") for d in nearest_dates]
    is_self = False
    if query_date is not None:
        qd = pd.Timestamp(query_date).normalize()
        is_self = any(pd.Timestamp(d).normalize() == qd for d in nearest_dates) and dist < 1e-3

    if len(nearest_dates) == 1:
        note = f"NN L2={dist:.4f} → {date_strs[0]}"
    else:
        note = f"NN L2={dist:.4f} → {len(nearest_dates)} ngày (median 24h)"

    return NearestDayResult(
        hourly_factors=profile,
        nearest_dates=date_strs,
        distance=dist,
        n_candidates=len(nearest_dates),
        is_self_match=is_self,
        note=note,
    )
