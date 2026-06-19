"""Live / historical weather + calendar features for optimizer UI (Open-Meteo + holidays)."""

from __future__ import annotations

from datetime import date, datetime, timezone
from typing import Any

import numpy as np
import pandas as pd
import requests

NYC_LAT, NYC_LON = 40.7128, -74.0060
OPEN_METEO_TIMEOUT = 12
FORECAST_HORIZON_DAYS = 16
PICKER_FUTURE_DAYS = 365


def meteorological_season(month: int) -> str:
    if month in (12, 1, 2):
        return "winter"
    if month in (3, 4, 5):
        return "spring"
    if month in (6, 7, 8):
        return "summer"
    return "autumn"


def calendar_row_for_date(d: date, *, is_holiday: int) -> dict[str, Any]:
    dow = int(d.weekday())  # Mon=0 … Sun=6 — align factors_daily day_of_week
    return {
        "date": pd.Timestamp(d),
        "day_of_week": dow,
        "day_name": d.strftime("%A"),
        "is_weekend": int(dow >= 5),
        "is_monday": int(dow == 0),
        "is_friday": int(dow == 4),
        "month": d.month,
        "season": meteorological_season(d.month),
        "is_us_holiday": int(is_holiday),
        "is_major_event_window": 0,
    }


def ny_holiday_flag(d: date) -> int:
    try:
        import holidays

        us_ny = holidays.US(state="NY", years={d.year})
        return int(d in us_ny)
    except Exception:
        return 0


def _ms_to_kmh(speed_ms: float | None) -> float:
    if speed_ms is None or not np.isfinite(speed_ms):
        return 0.0
    return float(speed_ms) * 3.6


def fetch_open_meteo_hourly(target: date) -> pd.DataFrame:
    """Hourly weather for NYC; archive API for past dates, forecast for today/future."""
    today = datetime.now(timezone.utc).date()
    date_str = target.isoformat()
    days_ahead = (target - today).days
    if days_ahead > FORECAST_HORIZON_DAYS:
        raise ValueError(
            f"Open-Meteo forecast chỉ ~{FORECAST_HORIZON_DAYS} ngày; "
            f"ngày {date_str} quá xa — dùng climatology fallback."
        )
    hourly = (
        "temperature_2m,apparent_temperature,precipitation,rain,snowfall,"
        "wind_speed_10m,wind_gusts_10m,weathercode"
    )
    if target < today:
        url = "https://archive-api.open-meteo.com/v1/archive"
        params = {
            "latitude": NYC_LAT,
            "longitude": NYC_LON,
            "start_date": date_str,
            "end_date": date_str,
            "hourly": hourly,
            "timezone": "America/New_York",
        }
    else:
        url = "https://api.open-meteo.com/v1/forecast"
        params = {
            "latitude": NYC_LAT,
            "longitude": NYC_LON,
            "hourly": hourly,
            "timezone": "America/New_York",
            "forecast_days": min(FORECAST_HORIZON_DAYS, max(1, days_ahead + 1)),
        }

    resp = requests.get(url, params=params, timeout=OPEN_METEO_TIMEOUT)
    resp.raise_for_status()
    payload = resp.json()
    h = payload.get("hourly") or {}
    times = h.get("time") or []
    if not times:
        raise ValueError(f"Open-Meteo returned no hourly rows for {date_str}")

    rows = []
    for i, ts in enumerate(times):
        if not str(ts).startswith(date_str):
            continue
        hour = int(str(ts)[11:13])
        temp = float(h["temperature_2m"][i]) if h.get("temperature_2m") else 0.0
        app_temp = float(h["apparent_temperature"][i]) if h.get("apparent_temperature") else temp
        precip = float(h["precipitation"][i] or 0.0)
        rain = float(h["rain"][i] or precip)
        snow = float(h["snowfall"][i] or 0.0)
        wind = _ms_to_kmh(h["wind_speed_10m"][i] if h.get("wind_speed_10m") else 0.0)
        gust = _ms_to_kmh(h["wind_gusts_10m"][i] if h.get("wind_gusts_10m") else wind * 1.25)
        rows.append(
            {
                "hour": hour,
                "temperature_c": temp,
                "apparent_temperature_c": app_temp,
                "precipitation_mm": precip,
                "rain_mm": rain,
                "snowfall_cm": snow,
                "windspeed_kmh": wind,
                "windgusts_kmh": gust,
                "is_rain": int(precip > 0 or rain > 0),
                "is_snow": int(snow > 0),
                "is_severe_wind": int(wind >= 50),
                "is_peak_morning": int(hour in (7, 8, 9)),
                "is_peak_evening": int(hour in (17, 18, 19)),
                "is_overnight": int(hour <= 5 or hour >= 23),
            }
        )

    if not rows:
        raise ValueError(f"No hourly rows matched date {date_str} in Open-Meteo response")
    return pd.DataFrame(rows).sort_values("hour").reset_index(drop=True)


def load_reference_hourly(factors_hourly_path: str, target: date) -> pd.DataFrame | None:
    """Return 24h factors from training CSV when date exists."""
    try:
        fh = pd.read_csv(factors_hourly_path, parse_dates=["timestamp", "date"])
        if "hour" not in fh.columns:
            fh["hour"] = pd.to_datetime(fh["timestamp"]).dt.hour
        fh["date"] = pd.to_datetime(fh["date"]).dt.normalize()
        day = pd.Timestamp(target).normalize()
        sub = fh.loc[fh["date"] == day].copy()
        if sub.empty:
            return None
        return sub.sort_values("hour").reset_index(drop=True)
    except Exception:
        return None


def _build_climatology_hourly(
    target: date,
    cal: dict[str, Any],
    factors_hourly_path: str,
) -> pd.DataFrame:
    """Thời tiết theo median tháng×giờ từ training data (ngày tương lai xa)."""
    fh = pd.read_csv(factors_hourly_path, parse_dates=["date"])
    if "hour" not in fh.columns and "timestamp" in fh.columns:
        fh["hour"] = pd.to_datetime(fh["timestamp"]).dt.hour
    fh["month"] = pd.to_datetime(fh["date"]).dt.month
    month = int(cal["month"])
    sub = fh.loc[fh["month"] == month]
    if sub.empty:
        sub = fh

    num_cols = [
        c
        for c in (
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
        )
        if c in sub.columns
    ]
    prof = sub.groupby("hour")[num_cols].median().reset_index()
    prof["date"] = pd.Timestamp(target)
    prof["day_of_week"] = cal["day_of_week"]
    prof["is_weekend"] = cal["is_weekend"]
    prof["is_us_holiday"] = cal["is_us_holiday"]
    prof["month"] = month
    prof["is_major_event_window"] = cal.get("is_major_event_window", 0)
    return prof.sort_values("hour").reset_index(drop=True)


def build_hourly_factors(
    target: date,
    *,
    factors_hourly_path: str | None = None,
    factors_daily_path: str | None = None,
) -> tuple[pd.DataFrame, str]:
    """Build 24-row hourly factor frame; returns (dataframe, source_label)."""
    is_holiday = ny_holiday_flag(target)
    cal = calendar_row_for_date(target, is_holiday=is_holiday)
    today = datetime.now(timezone.utc).date()
    days_ahead = (target - today).days

    if factors_hourly_path:
        ref = load_reference_hourly(factors_hourly_path, target)
        if ref is not None:
            ref["is_us_holiday"] = cal["is_us_holiday"]
            ref["is_weekend"] = cal["is_weekend"]
            ref["day_of_week"] = cal["day_of_week"]
            ref["month"] = cal["month"]
            return ref, "factors_hourly.csv (reference)"

    if days_ahead > FORECAST_HORIZON_DAYS and factors_hourly_path:
        clim = _build_climatology_hourly(target, cal, factors_hourly_path)
        return clim, f"Climatology tháng {cal['month']} (ngày >{FORECAST_HORIZON_DAYS} ngày)"

    try:
        live = fetch_open_meteo_hourly(target)
        live["date"] = pd.Timestamp(target)
        live["day_of_week"] = cal["day_of_week"]
        live["is_weekend"] = cal["is_weekend"]
        live["is_us_holiday"] = cal["is_us_holiday"]
        live["month"] = cal["month"]
        live["is_major_event_window"] = cal["is_major_event_window"]
        src = "Open-Meteo forecast" if days_ahead >= 0 else "Open-Meteo archive"
        return live, src
    except Exception as api_err:
        if factors_hourly_path and days_ahead > 0:
            clim = _build_climatology_hourly(target, cal, factors_hourly_path)
            return clim, f"Climatology tháng {cal['month']} (API: {api_err})"
        if factors_daily_path:
            fd = pd.read_csv(factors_daily_path, parse_dates=["date"])
            fd["date"] = pd.to_datetime(fd["date"]).dt.normalize()
            row = fd.loc[fd["date"] == pd.Timestamp(target).normalize()]
            if not row.empty:
                from .demand_runtime import daily_factors_to_hourly

                r = row.iloc[0].to_dict()
                hours = list(range(24))
                rows = []
                for h in hours:
                    wh = daily_factors_to_hourly(r, h)
                    wh.update(
                        {
                            "hour": h,
                            "date": pd.Timestamp(target),
                            "day_of_week": cal["day_of_week"],
                            "is_weekend": cal["is_weekend"],
                            "is_us_holiday": cal["is_us_holiday"],
                            "month": cal["month"],
                            "is_major_event_window": cal["is_major_event_window"],
                        }
                    )
                    rows.append(wh)
                return (
                    pd.DataFrame(rows),
                    f"factors_daily.csv fallback (API: {api_err})",
                )
        raise RuntimeError(f"Weather unavailable for {target}: {api_err}") from api_err


# --- Scenario builder (synthetic, no API / no calendar date) ---

SEASON_TEMP_C: dict[str, float] = {
    "Spring": 12.0,
    "Summer": 26.0,
    "Autumn": 14.0,
    "Winter": -1.0,
}

SEASON_MONTH: dict[str, int] = {
    "Spring": 4,
    "Summer": 7,
    "Autumn": 10,
    "Winter": 1,
}

DAY_TYPE_CAL: dict[str, dict[str, int]] = {
    "Weekday": {"day_of_week": 1, "is_weekend": 0, "is_us_holiday": 0},
    "Weekend": {"day_of_week": 6, "is_weekend": 1, "is_us_holiday": 0},
    "Holiday": {"day_of_week": 1, "is_weekend": 0, "is_us_holiday": 1},
}


def _normalize_weather_label(weather: str) -> str:
    """Strip emoji / suffix from sidebar radio label."""
    w = weather.strip()
    for key in ("Clear", "Rainy", "Heavy rain / storm", "Snow"):
        if w.startswith(key):
            return key
    return w.split()[0] if w else "Clear"


def _weather_profile(weather: str, season: str) -> dict[str, float | int]:
    """Map scenario weather → hourly feature overrides."""
    w = _normalize_weather_label(weather)
    temp = SEASON_TEMP_C.get(season, 14.0)
    base = {
        "temperature_c": temp,
        "apparent_temperature_c": temp - 2.0,
        "precipitation_mm": 0.0,
        "rain_mm": 0.0,
        "snowfall_cm": 0.0,
        "windspeed_kmh": 15.0,
        "windgusts_kmh": 22.0,
        "is_rain": 0,
        "is_snow": 0,
        "is_severe_wind": 0,
    }
    if w == "Rainy":
        base.update(
            precipitation_mm=10.0,
            rain_mm=8.0,
            is_rain=1,
            windspeed_kmh=25.0,
            windgusts_kmh=35.0,
        )
    elif w == "Heavy rain / storm":
        base.update(
            precipitation_mm=30.0,
            rain_mm=25.0,
            is_rain=1,
            is_severe_wind=1,
            windspeed_kmh=55.0,
            windgusts_kmh=70.0,
            apparent_temperature_c=temp - 4.0,
        )
    elif w == "Snow":
        base.update(
            temperature_c=-2.0,
            apparent_temperature_c=-5.0,
            snowfall_cm=5.0,
            is_snow=1,
            windspeed_kmh=20.0,
            windgusts_kmh=28.0,
        )
    return base


def _load_training_medians(factors_hourly_path: str | None) -> pd.Series:
    """Global numeric medians from factors_hourly.csv (background when available)."""
    if not factors_hourly_path:
        return pd.Series(dtype=float)
    try:
        fh = pd.read_csv(factors_hourly_path)
        num_cols = [
            "temperature_c",
            "apparent_temperature_c",
            "precipitation_mm",
            "rain_mm",
            "snowfall_cm",
            "windspeed_kmh",
            "windgusts_kmh",
        ]
        present = [c for c in num_cols if c in fh.columns]
        if not present:
            return pd.Series(dtype=float)
        return fh[present].apply(pd.to_numeric, errors="coerce").median()
    except Exception:
        return pd.Series(dtype=float)


def build_scenario_factors(
    day_type: str,
    season: str,
    weather: str,
    event: bool,
    *,
    factors_hourly_path: str | None = None,
) -> pd.DataFrame:
    """
    Synthetic 24-row hourly factor frame for DemandPredictor.

    Uses training medians as background (when CSV available), then overrides
    calendar/weather flags per scenario — no API, no calendar date.
    """
    cal = DAY_TYPE_CAL.get(day_type, DAY_TYPE_CAL["Weekday"])
    month = SEASON_MONTH.get(season, 6)
    w_prof = _weather_profile(weather, season)
    med = _load_training_medians(factors_hourly_path)

    rows: list[dict[str, Any]] = []
    for hour in range(24):
        row: dict[str, Any] = {
            "hour": hour,
            "day_of_week": cal["day_of_week"],
            "is_weekend": cal["is_weekend"],
            "is_us_holiday": cal["is_us_holiday"],
            "month": month,
            "is_peak_morning": int(hour in (7, 8, 9)),
            "is_peak_evening": int(hour in (17, 18, 19)),
            "is_overnight": int(hour <= 5 or hour >= 23),
            "is_major_event_window": int(event and 16 <= hour <= 22),
        }
        for col in (
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
        ):
            row[col] = w_prof[col]
        for col in med.index:
            if col not in row and np.isfinite(med[col]):
                row[col] = float(med[col])
        rows.append(row)

    return pd.DataFrame(rows).sort_values("hour").reset_index(drop=True)
