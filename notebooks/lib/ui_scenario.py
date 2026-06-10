"""Kịch bản tổng quát — query profile → NN trong không gian feature training."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Any

import numpy as np
import pandas as pd

from .ui_nearest import resolve_to_nearest_training_day

WEEKDAY_WEEKEND_OPTIONS: tuple[str, ...] = ("weekday", "weekend")
WEEKDAY_WEEKEND_VI: dict[str, str] = {
    "weekday": "Ngày thường",
    "weekend": "Cuối tuần",
}

SEASONS: tuple[str, ...] = ("winter", "spring", "summer", "fall")
SEASON_VI: dict[str, str] = {
    "winter": "Đông",
    "spring": "Xuân",
    "summer": "Hè",
    "fall": "Thu",
}

WEATHER_OPTIONS: tuple[str, ...] = ("clear", "rainy", "heavy_rain", "snow")
WEATHER_VI: dict[str, str] = {
    "clear": "Nắng / quang",
    "rainy": "Mưa",
    "heavy_rain": "Mưa lớn",
    "snow": "Tuyết",
}

HOURLY_NUMERIC = (
    "temperature_c",
    "apparent_temperature_c",
    "precipitation_mm",
    "rain_mm",
    "snowfall_cm",
    "windspeed_kmh",
    "windgusts_kmh",
)
HOURLY_FLAGS = (
    "is_rain",
    "is_snow",
    "is_severe_wind",
    "is_peak_morning",
    "is_peak_evening",
    "is_overnight",
    "is_major_event_window",
)
CALENDAR_FROM_HOURLY = (
    "day_of_week",
    "is_weekend",
    "is_us_holiday",
    "month",
    "season",
)
CALENDAR_STR_FROM_HOURLY = ("holiday_name",)


@dataclass(frozen=True)
class ScenarioSelection:
    weekday_weekend: str
    season: str
    weather: str
    filter_holiday: bool = False
    holiday_name: str | None = None
    filter_major_event: bool = False

    def active_filters(self) -> list[tuple[str, str]]:
        labels: list[tuple[str, str]] = [
            ("weekday_weekend", WEEKDAY_WEEKEND_VI[self.weekday_weekend]),
            ("season", SEASON_VI[self.season]),
            ("weather", WEATHER_VI[self.weather]),
        ]
        if self.filter_holiday:
            if self.holiday_name:
                labels.append(("holiday", self.holiday_name))
            else:
                labels.append(("holiday", "Ngày lễ"))
        if self.filter_major_event:
            labels.append(("major_event", "Có sự kiện lớn"))
        return labels


@dataclass
class ScenarioBuildResult:
    hourly_factors: pd.DataFrame
    label: str
    n_days: int
    date_min: str
    date_max: str
    sample_dates: list[str]
    is_weekend: int
    filters_applied: list[str]
    exact_match: bool
    match_note: str
    holiday_name: str | None = None
    nn_distance: float = 0.0


@lru_cache(maxsize=4)
def list_holiday_names(factors_daily_path: str) -> tuple[str, ...]:
    daily = pd.read_csv(factors_daily_path, parse_dates=["date"])
    if "holiday_name" not in daily.columns:
        return ()
    names = (
        daily.loc[daily["is_us_holiday"] == 1, "holiday_name"]
        .dropna()
        .astype(str)
        .unique()
        .tolist()
    )
    return tuple(sorted(names))


def _normalize_dates(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["date"] = pd.to_datetime(out["date"]).dt.normalize()
    return out


def _heavy_rain_threshold(daily: pd.DataFrame) -> float:
    rainy = daily.loc[daily["is_rainy_day"] == 1, "rain_mm"]
    rainy = pd.to_numeric(rainy, errors="coerce").dropna()
    if rainy.empty:
        raise ValueError("Không có ngày mưa trong factors_daily để định nghĩa 'Mưa lớn'.")
    return float(rainy.quantile(0.75))


def _median_int(series: pd.Series) -> int:
    val = float(pd.to_numeric(series, errors="coerce").median())
    if not np.isfinite(val):
        raise ValueError("Không có giá trị hợp lệ để lấy median.")
    return int(round(val))


def _mode_str(series: pd.Series) -> str:
    s = series.dropna().astype(str)
    s = s[s.str.len() > 0]
    if s.empty:
        return ""
    return str(s.mode().iloc[0])


def _weather_mask(d: pd.DataFrame, weather: str, heavy_thr: float) -> pd.Series:
    if weather == "clear":
        return (d["is_rainy_day"] == 0) & (d["is_snowy_day"] == 0)
    if weather == "rainy":
        return (d["is_rainy_day"] == 1) & (d["is_snowy_day"] == 0)
    if weather == "heavy_rain":
        return (d["is_rainy_day"] == 1) & (
            pd.to_numeric(d["rain_mm"], errors="coerce").fillna(0) >= heavy_thr
        )
    if weather == "snow":
        return d["is_snowy_day"] == 1
    raise ValueError(f"Weather không hợp lệ: {weather}")


def _pool_for_query(daily: pd.DataFrame, selection: ScenarioSelection, heavy_thr: float) -> pd.DataFrame:
    """Pool ngày để dựng profile truy vấn (nới lỏng nếu pool rỗng)."""
    d = _normalize_dates(daily)

    def core() -> pd.Series:
        m = d["season"] == selection.season
        m &= _weather_mask(d, selection.weather, heavy_thr)
        if selection.weekday_weekend == "weekday":
            m &= d["is_weekend"] == 0
        else:
            m &= d["is_weekend"] == 1
        return m

    masks: list[pd.Series] = []
    full = core()
    if selection.filter_holiday:
        full &= d["is_us_holiday"] == 1
    if selection.holiday_name:
        full &= d["holiday_name"].astype(str) == selection.holiday_name
    if selection.filter_major_event:
        full &= d["is_major_event_window"] == 1
    masks.append(full)

    if selection.filter_holiday and selection.holiday_name:
        m = core() & (d["is_us_holiday"] == 1)
        masks.append(m)

    if selection.filter_holiday:
        m = core() & (d["is_us_holiday"] == 1)
        masks.append(m)

    if selection.filter_major_event:
        m = core() & (d["is_major_event_window"] == 1)
        masks.append(m)

    masks.extend([core(), d["season"] == selection.season, pd.Series(True, index=d.index)])

    seen: set[int] = set()
    for mask in masks:
        key = int(mask.sum())
        if key in seen:
            continue
        seen.add(key)
        pool = d.loc[mask]
        if not pool.empty:
            return pool
    return d


def _median_hourly_from_dates(hourly: pd.DataFrame, dates: set[pd.Timestamp]) -> pd.DataFrame:
    sub = hourly.loc[hourly["date"].isin(dates)].copy()
    if sub.empty:
        raise ValueError("Không có factors_hourly cho pool truy vấn.")

    profile_rows: list[dict[str, Any]] = []
    for hour in range(24):
        hrows = sub.loc[sub["hour"] == hour]
        if hrows.empty:
            raise ValueError(f"Thiếu giờ {hour} trong pool truy vấn.")
        row: dict[str, Any] = {"hour": hour}
        for col in HOURLY_NUMERIC:
            if col not in hrows.columns:
                raise ValueError(f"factors_hourly thiếu cột {col}.")
            val = pd.to_numeric(hrows[col], errors="coerce").median()
            if not np.isfinite(val):
                raise ValueError(f"Giờ {hour}: không có giá trị hợp lệ cho {col}.")
            row[col] = float(val)
        for col in HOURLY_FLAGS:
            if col not in hrows.columns:
                raise ValueError(f"factors_hourly thiếu cột {col}.")
            row[col] = _median_int(hrows[col])
        for col in CALENDAR_FROM_HOURLY:
            if col not in hrows.columns:
                continue
            if col == "season":
                row[col] = _mode_str(hrows[col])
            else:
                row[col] = _median_int(hrows[col])
        for col in CALENDAR_STR_FROM_HOURLY:
            if col in hrows.columns:
                row[col] = _mode_str(hrows[col])
        profile_rows.append(row)
    return pd.DataFrame(profile_rows).sort_values("hour").reset_index(drop=True)


def build_scenario_query_hourly(
    selection: ScenarioSelection,
    daily: pd.DataFrame,
    hourly: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Profile truy vấn (median pool) + pool ngày dùng để dựng query."""
    heavy_thr = _heavy_rain_threshold(daily)
    pool = _pool_for_query(daily, selection, heavy_thr)
    dates = set(pool["date"])
    query = _median_hourly_from_dates(hourly, dates)
    return query, pool


def _scenario_label(selection: ScenarioSelection) -> str:
    return " · ".join(label for _, label in selection.active_filters())


def build_scenario_hourly_factors(
    selection: ScenarioSelection,
    *,
    factors_daily_path: str,
    factors_hourly_path: str,
    file_mtime: float,
) -> ScenarioBuildResult:
    daily = _normalize_dates(pd.read_csv(factors_daily_path, parse_dates=["date"]))
    hourly = _normalize_dates(pd.read_csv(factors_hourly_path, parse_dates=["date"]))

    query, pool = build_scenario_query_hourly(selection, daily, hourly)
    nearest = resolve_to_nearest_training_day(
        query,
        factors_hourly_path,
        file_mtime=file_mtime,
    )

    pool_dates = sorted(pool["date"].unique())
    filters_applied = [label for _, label in selection.active_filters()]

    return ScenarioBuildResult(
        hourly_factors=nearest.hourly_factors,
        label=_scenario_label(selection),
        n_days=nearest.n_candidates,
        date_min=nearest.nearest_dates[0],
        date_max=nearest.nearest_dates[-1],
        sample_dates=nearest.nearest_dates[:5],
        is_weekend=int(nearest.hourly_factors["is_weekend"].median())
        if "is_weekend" in nearest.hourly_factors
        else 0,
        filters_applied=filters_applied,
        exact_match=nearest.distance < 1e-3,
        match_note=nearest.note,
        holiday_name=selection.holiday_name,
        nn_distance=nearest.distance,
    )
