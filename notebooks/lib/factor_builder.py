"""Build factors_hourly.csv + factors_daily.csv from Open-Meteo (NYC) + calendar rules."""

from __future__ import annotations

import time
from datetime import date, datetime, timedelta
from typing import Any, Iterator

import numpy as np
import pandas as pd
import requests

NYC_LAT, NYC_LON = 40.7128, -74.0060
OPEN_METEO_TIMEOUT = 60
ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"

HOURLY_VARS = (
    "temperature_2m,apparent_temperature,precipitation,rain,snowfall,"
    "weathercode,wind_speed_10m,wind_gusts_10m,is_day"
)

WMO_LABELS: dict[int, str] = {
    0: "clear",
    1: "mainly_clear",
    2: "partly_cloudy",
    3: "overcast",
    45: "fog",
    48: "rime_fog",
    51: "light_drizzle",
    53: "drizzle",
    55: "dense_drizzle",
    56: "freezing_drizzle",
    57: "dense_freezing_drizzle",
    61: "light_rain",
    63: "rain",
    65: "heavy_rain",
    66: "freezing_rain",
    67: "heavy_freezing_rain",
    71: "light_snow",
    73: "snow",
    75: "heavy_snow",
    77: "snow_grains",
    80: "rain_showers",
    81: "heavy_rain_showers",
    82: "violent_rain_showers",
    85: "snow_showers",
    86: "heavy_snow_showers",
    95: "thunderstorm",
    96: "thunderstorm_hail",
    99: "thunderstorm_heavy_hail",
}


def meteorological_season(month: int) -> str:
    if month in (12, 1, 2):
        return "winter"
    if month in (3, 4, 5):
        return "spring"
    if month in (6, 7, 8):
        return "summer"
    return "fall"


def _ms_to_kmh(speed: float | None) -> float:
    if speed is None or not np.isfinite(speed):
        return 0.0
    return float(speed) * 3.6


def weather_label(code: int | float) -> str:
    return WMO_LABELS.get(int(code), f"code_{int(code)}")


def _ny_holidays(years: range) -> dict[date, str]:
    try:
        import holidays

        cal = holidays.US(state="NY", years=list(years))
        return {d: str(name) for d, name in cal.items()}
    except Exception:
        return {}


def _school_break_flags(d: date) -> tuple[int, int, int]:
    """NYC-style break windows inferred from existing training CSV."""
    y, m, day = d.year, d.month, d.day
    summer = int(m == 7 or m == 8)
    winter = int(m == 12 and day >= 24)
    spring = int(m == 3 and day >= 15)
    return summer, winter, spring


def _major_event_window(is_holiday: int, summer: int, winter: int, spring: int) -> int:
    return int(is_holiday or summer or winter or spring)


def _calendar_for_date(d: date, holiday_map: dict[date, str]) -> dict[str, Any]:
    dow = int(d.weekday())
    is_holiday = int(d in holiday_map)
    summer, winter, spring = _school_break_flags(d)
    return {
        "day_of_week": dow,
        "day_name": d.strftime("%A"),
        "is_weekend": int(dow >= 5),
        "is_monday": int(dow == 0),
        "is_friday": int(dow == 4),
        "month": d.month,
        "season": meteorological_season(d.month),
        "is_us_holiday": is_holiday,
        "holiday_name": holiday_map.get(d, "") if is_holiday else "",
        "is_summer_break": summer,
        "is_winter_break": winter,
        "is_spring_break": spring,
        "is_major_event_window": _major_event_window(is_holiday, summer, winter, spring),
    }


def _iter_chunks(start: date, end: date, *, days: int = 120) -> Iterator[tuple[date, date]]:
    cur = start
    while cur <= end:
        chunk_end = min(end, cur + timedelta(days=days - 1))
        yield cur, chunk_end
        cur = chunk_end + timedelta(days=1)


def fetch_open_meteo_range(start: date, end: date, *, pause_s: float = 0.3) -> pd.DataFrame:
    """Fetch hourly weather for NYC; archive for past, forecast API for recent days."""
    today = datetime.now().date()
    rows: list[dict[str, Any]] = []

    past_end = min(end, today - timedelta(days=1))
    if start <= past_end:
        for c0, c1 in _iter_chunks(start, past_end):
            rows.extend(_fetch_chunk(c0, c1, archive=True))
            if pause_s:
                time.sleep(pause_s)

    recent_start = max(start, today)
    if recent_start <= end:
        rows.extend(_fetch_recent(recent_start, end))

    if not rows:
        raise RuntimeError(f"No weather rows for {start} -> {end}")

    df = pd.DataFrame(rows)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["date"] = pd.to_datetime(df["date"]).dt.normalize()
    return df.sort_values(["date", "hour"]).reset_index(drop=True)


def _fetch_chunk(start: date, end: date, *, archive: bool) -> list[dict[str, Any]]:
    params = {
        "latitude": NYC_LAT,
        "longitude": NYC_LON,
        "start_date": start.isoformat(),
        "end_date": end.isoformat(),
        "hourly": HOURLY_VARS,
        "timezone": "America/New_York",
    }
    url = ARCHIVE_URL if archive else FORECAST_URL
    resp = requests.get(url, params=params, timeout=OPEN_METEO_TIMEOUT)
    resp.raise_for_status()
    return _parse_hourly_payload(resp.json())


def _fetch_recent(start: date, end: date) -> list[dict[str, Any]]:
    days_ahead = (end - datetime.now().date()).days
    params = {
        "latitude": NYC_LAT,
        "longitude": NYC_LON,
        "hourly": HOURLY_VARS,
        "timezone": "America/New_York",
        "forecast_days": max(1, min(16, days_ahead + 1)),
        "past_days": max(0, (datetime.now().date() - start).days),
    }
    resp = requests.get(FORECAST_URL, params=params, timeout=OPEN_METEO_TIMEOUT)
    resp.raise_for_status()
    rows = _parse_hourly_payload(resp.json())
    out = []
    for r in rows:
        d = pd.Timestamp(r["date"]).date()
        if start <= d <= end:
            out.append(r)
    return out


def _parse_hourly_payload(payload: dict[str, Any]) -> list[dict[str, Any]]:
    h = payload.get("hourly") or {}
    times = h.get("time") or []
    if not times:
        return []

    def col(name: str, i: int, default: float = 0.0) -> float:
        arr = h.get(name)
        if not arr or arr[i] is None:
            return default
        return float(arr[i])

    rows: list[dict[str, Any]] = []
    for i, ts in enumerate(times):
        d = pd.Timestamp(ts)
        hour = int(d.hour)
        temp = col("temperature_2m", i)
        app_temp = col("apparent_temperature", i, temp)
        precip = col("precipitation", i)
        rain = col("rain", i, precip)
        snow = col("snowfall", i)
        wind = _ms_to_kmh(col("wind_speed_10m", i))
        gust = _ms_to_kmh(col("wind_gusts_10m", i, wind * 1.25))
        wcode = int(col("weathercode", i))
        is_day = int(col("is_day", i, 1 if 6 <= hour <= 20 else 0))
        rows.append(
            {
                "timestamp": ts,
                "date": d.normalize(),
                "hour": hour,
                "temperature_c": temp,
                "apparent_temperature_c": app_temp,
                "precipitation_mm": precip,
                "rain_mm": rain,
                "snowfall_cm": snow,
                "windspeed_kmh": wind,
                "windgusts_kmh": gust,
                "weathercode": wcode,
                "weather_label": weather_label(wcode),
                "is_day": is_day,
                "is_rain": int(precip > 0 or rain > 0),
                "is_snow": int(snow > 0),
                "is_severe_wind": int(wind >= 50),
                "is_peak_morning": int(hour in (7, 8, 9)),
                "is_peak_evening": int(hour in (17, 18, 19)),
                "is_overnight": int(hour <= 5 or hour >= 23),
            }
        )
    return rows


def enrich_hourly_calendar(hourly: pd.DataFrame) -> pd.DataFrame:
    years = range(hourly["date"].dt.year.min(), hourly["date"].dt.year.max() + 1)
    holiday_map = _ny_holidays(years)
    cal_rows = []
    for d in sorted(hourly["date"].dt.normalize().unique()):
        dt = pd.Timestamp(d).date()
        cal_rows.append({"date": pd.Timestamp(d), **_calendar_for_date(dt, holiday_map)})
    cal = pd.DataFrame(cal_rows)
    out = hourly.merge(cal, on="date", how="left")
    return out.sort_values(["date", "hour"]).reset_index(drop=True)


def build_daily_from_hourly(hourly: pd.DataFrame) -> pd.DataFrame:
    g = hourly.groupby("date", as_index=False)
    daily = g.agg(
        day_of_week=("day_of_week", "first"),
        day_name=("day_name", "first"),
        is_weekend=("is_weekend", "first"),
        is_monday=("is_monday", "first"),
        is_friday=("is_friday", "first"),
        month=("month", "first"),
        season=("season", "first"),
        is_us_holiday=("is_us_holiday", "first"),
        holiday_name=("holiday_name", "first"),
        is_summer_break=("is_summer_break", "first"),
        is_winter_break=("is_winter_break", "first"),
        is_spring_break=("is_spring_break", "first"),
        is_major_event_window=("is_major_event_window", "first"),
        temp_max_c=("temperature_c", "max"),
        temp_min_c=("temperature_c", "min"),
        apparent_temp_max_c=("apparent_temperature_c", "max"),
        apparent_temp_min_c=("apparent_temperature_c", "min"),
        precipitation_mm=("precipitation_mm", "sum"),
        rain_mm=("rain_mm", "sum"),
        snowfall_cm=("snowfall_cm", "sum"),
        wind_max_kmh=("windspeed_kmh", "max"),
    )
    wc_mode = (
        hourly.groupby("date")["weathercode"]
        .agg(lambda s: int(s.mode().iloc[0]) if len(s.mode()) else int(s.iloc[0]))
        .reset_index()
    )
    daily = daily.merge(wc_mode, on="date")
    daily["weather_label"] = daily["weathercode"].map(lambda c: weather_label(int(c)))
    daily["is_rainy_day"] = (daily["rain_mm"] >= 1.0).astype(int)
    daily["is_snowy_day"] = (daily["snowfall_cm"] >= 1.0).astype(int)
    daily["is_extreme_heat"] = (daily["temp_max_c"] >= 32.2).astype(int)
    daily["is_extreme_cold"] = (daily["temp_min_c"] <= -5.2).astype(int)
    daily["date"] = pd.to_datetime(daily["date"]).dt.strftime("%Y-%m-%d")
    return daily


def build_factors(
    start: date,
    end: date,
    *,
    pause_s: float = 0.3,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    raw = fetch_open_meteo_range(start, end, pause_s=pause_s)
    hourly = enrich_hourly_calendar(raw)
    hourly["timestamp"] = hourly["timestamp"].dt.strftime("%Y-%m-%dT%H:%M:%S")
    hourly["date"] = pd.to_datetime(hourly["date"]).dt.strftime("%Y-%m-%d")
    hourly = hourly[
        [
            "timestamp",
            "date",
            "hour",
            "day_of_week",
            "day_name",
            "is_weekend",
            "is_monday",
            "is_friday",
            "month",
            "season",
            "is_us_holiday",
            "holiday_name",
            "is_summer_break",
            "is_winter_break",
            "is_spring_break",
            "is_major_event_window",
            "temperature_c",
            "apparent_temperature_c",
            "precipitation_mm",
            "rain_mm",
            "snowfall_cm",
            "windspeed_kmh",
            "windgusts_kmh",
            "weathercode",
            "weather_label",
            "is_day",
            "is_rain",
            "is_snow",
            "is_severe_wind",
            "is_peak_morning",
            "is_peak_evening",
            "is_overnight",
        ]
    ]
    daily = build_daily_from_hourly(enrich_hourly_calendar(raw))
    return hourly, daily
