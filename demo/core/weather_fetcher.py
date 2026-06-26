from __future__ import annotations

from datetime import date, datetime
from functools import lru_cache

import httpx
import pandas as pd

from demo.config import HOURLY_FACTOR_COLS, NYC_LAT, NYC_LON


def _weather_label_from_code(code: int) -> str:
    mapping = {
        0: "clear",
        1: "mainly_clear",
        2: "partly_cloudy",
        3: "overcast",
        45: "fog",
        48: "depositing_rime_fog",
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
        81: "rain_showers",
        82: "heavy_rain_showers",
        85: "snow_showers",
        86: "heavy_snow_showers",
        95: "thunderstorm",
        96: "thunderstorm_hail",
        99: "severe_thunderstorm_hail",
    }
    return mapping.get(int(code), "unknown")


def _season_from_month(month: int) -> str:
    m = int(month)
    if m in (12, 1, 2):
        return "winter"
    if m in (3, 4, 5):
        return "spring"
    if m in (6, 7, 8):
        return "summer"
    return "autumn"


@lru_cache(maxsize=64)
def fetch_openmeteo_for_date(target_date: date, lat: float = NYC_LAT, lon: float = NYC_LON) -> pd.DataFrame:
    hourly_fields = [
        "temperature_2m",
        "apparent_temperature",
        "precipitation",
        "rain",
        "snowfall",
        "windspeed_10m",
        "windgusts_10m",
        "weathercode",
    ]
    params = {
        "latitude": lat,
        "longitude": lon,
        "hourly": ",".join(hourly_fields),
        "start_date": target_date.isoformat(),
        "end_date": target_date.isoformat(),
        "timezone": "America/New_York",
    }
    with httpx.Client(timeout=15) as client:
        resp = client.get("https://api.open-meteo.com/v1/forecast", params=params)
        resp.raise_for_status()
        payload = resp.json()

    hourly = payload.get("hourly", {})
    df = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(hourly.get("time", [])),
            "temperature_c": hourly.get("temperature_2m", []),
            "apparent_temperature_c": hourly.get("apparent_temperature", []),
            "precipitation_mm": hourly.get("precipitation", []),
            "rain_mm": hourly.get("rain", []),
            "snowfall_cm": hourly.get("snowfall", []),
            "windspeed_kmh": hourly.get("windspeed_10m", []),
            "windgusts_kmh": hourly.get("windgusts_10m", []),
            "weathercode": hourly.get("weathercode", []),
        }
    )
    if df.empty:
        return pd.DataFrame(columns=["date", "hour"] + HOURLY_FACTOR_COLS)

    df["date"] = df["timestamp"].dt.date
    df["hour"] = df["timestamp"].dt.hour.astype(int)
    dt = pd.to_datetime(df["timestamp"])
    df["day_of_week"] = dt.dt.weekday.astype(int)
    df["day_name"] = dt.dt.day_name()
    df["month"] = dt.dt.month.astype(int)
    df["season"] = df["month"].map(_season_from_month)
    df["is_rain"] = (pd.to_numeric(df["precipitation_mm"], errors="coerce").fillna(0.0) > 0.5).astype(int)
    df["is_snow"] = (pd.to_numeric(df["snowfall_cm"], errors="coerce").fillna(0.0) > 0.1).astype(int)
    df["is_severe_wind"] = (pd.to_numeric(df["windgusts_kmh"], errors="coerce").fillna(0.0) > 50.0).astype(int)
    df["is_major_event_window"] = 0
    df["is_us_holiday"] = 0
    df["holiday_name"] = ""
    df["is_weekend"] = dt.dt.weekday.isin([5, 6]).astype(int)
    df["is_monday"] = (df["day_of_week"] == 0).astype(int)
    df["is_friday"] = (df["day_of_week"] == 4).astype(int)
    df["is_summer_break"] = 0
    df["is_winter_break"] = 0
    df["is_spring_break"] = 0
    df["weather_label"] = pd.to_numeric(df["weathercode"], errors="coerce").fillna(0).astype(int).map(_weather_label_from_code)
    df["is_day"] = df["hour"].between(6, 18).astype(int)
    df["is_peak_morning"] = df["hour"].isin([7, 8, 9]).astype(int)
    df["is_peak_evening"] = df["hour"].isin([17, 18, 19]).astype(int)
    df["is_overnight"] = (~df["hour"].between(5, 23)).astype(int)

    full_cols = [
        "timestamp", "date", "hour", "day_of_week", "day_name", "is_weekend", "is_monday", "is_friday",
        "month", "season", "is_us_holiday", "holiday_name", "is_summer_break", "is_winter_break",
        "is_spring_break", "is_major_event_window", "temperature_c", "apparent_temperature_c",
        "precipitation_mm", "rain_mm", "snowfall_cm", "windspeed_kmh", "windgusts_kmh",
        "weathercode", "weather_label", "is_day", "is_rain", "is_snow", "is_severe_wind",
        "is_peak_morning", "is_peak_evening", "is_overnight",
    ]
    for col in HOURLY_FACTOR_COLS:
        if col not in df.columns:
            df[col] = 0
    return df[full_cols].copy()


@lru_cache(maxsize=8)
def fetch_openmeteo_today(lat: float = NYC_LAT, lon: float = NYC_LON) -> pd.DataFrame:
    return fetch_openmeteo_for_date(date.today(), lat=lat, lon=lon)


def weather_summary_line(hourly_df: pd.DataFrame) -> str:
    if hourly_df.empty:
        return "Khong co du lieu thoi tiet hom nay."
    now_hour = datetime.now().hour
    row = hourly_df.loc[hourly_df["hour"] == now_hour]
    if row.empty:
        row = hourly_df.iloc[[0]]
    r = row.iloc[0]
    return (
        f"🌡️ {float(r['temperature_c']):.1f}°C (feels {float(r['apparent_temperature_c']):.1f}°C) | "
        f"🌧️ {float(r['precipitation_mm']):.1f}mm | "
        f"💨 {float(r['windspeed_kmh']):.1f} km/h | "
        f"Gusts {float(r['windgusts_kmh']):.1f} km/h"
    )
