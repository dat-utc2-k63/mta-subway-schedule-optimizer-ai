"""Weather group presets derived from training factors (weather_groups.json)."""

from __future__ import annotations

import json
from datetime import date
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .ui_factors import HOURLY_FACTOR_COLS, prepare_hourly_factors_for_model
from .ui_weather import calendar_row_for_date, ny_holiday_flag

WEATHER_GROUP_KEYS: tuple[str, ...] = (
    "sunny",
    "light_rain",
    "heavy_rain",
    "severe_storm",
    "snow",
    "heat_wave",
    "cold_snap",
)

WEATHER_GROUP_VI: dict[str, str] = {
    "sunny": "Nắng đẹp",
    "light_rain": "Mưa nhẹ",
    "heavy_rain": "Mưa lớn",
    "severe_storm": "Bão / gió mạnh",
    "snow": "Tuyết",
    "heat_wave": "Nóng cực đoan",
    "cold_snap": "Lạnh sâu",
}

WEATHER_NUMERIC = (
    "temperature_c",
    "apparent_temperature_c",
    "precipitation_mm",
    "rain_mm",
    "snowfall_cm",
    "windspeed_kmh",
    "windgusts_kmh",
)
WEATHER_FLAGS = ("is_rain", "is_snow", "is_severe_wind", "is_major_event_window")


def build_weather_groups(factors_hourly_path: str | Path) -> dict[str, Any]:
    """Compute group medians from training hourly factors (export step)."""
    fh = pd.read_csv(factors_hourly_path)
    for col in WEATHER_NUMERIC + WEATHER_FLAGS:
        if col in fh.columns:
            fh[col] = pd.to_numeric(fh[col], errors="coerce")

    temp = fh["temperature_c"].dropna()
    heat_thr = float(temp.quantile(0.90)) if not temp.empty else 30.0
    cold_thr = float(temp.quantile(0.10)) if not temp.empty else 0.0

    rainy = fh.loc[fh["is_rain"] == 1]
    rain_q75 = float(rainy["rain_mm"].quantile(0.75)) if not rainy.empty else 5.0

    masks: dict[str, pd.Series] = {
        "sunny": (fh["is_rain"] == 0) & (fh["is_snow"] == 0),
        "light_rain": (fh["is_rain"] == 1) & (fh["rain_mm"].fillna(0) < rain_q75),
        "heavy_rain": (fh["is_rain"] == 1) & (fh["rain_mm"].fillna(0) >= rain_q75),
        "severe_storm": (fh["is_severe_wind"] == 1)
        | ((fh["is_rain"] == 1) & (fh["windspeed_kmh"].fillna(0) >= 40)),
        "snow": fh["is_snow"] == 1,
        "heat_wave": fh["temperature_c"] >= heat_thr,
        "cold_snap": fh["temperature_c"] <= cold_thr,
    }

    groups: dict[str, dict[str, Any]] = {}
    meta: dict[str, Any] = {
        "source": str(factors_hourly_path),
        "rain_q75_mm": rain_q75,
        "heat_temp_c": heat_thr,
        "cold_temp_c": cold_thr,
    }

    for key, mask in masks.items():
        sub = fh.loc[mask]
        if sub.empty:
            sub = fh
        prof: dict[str, Any] = {"n_hours": int(len(sub))}
        for col in WEATHER_NUMERIC:
            prof[col] = float(sub[col].median()) if col in sub.columns else 0.0
        for col in WEATHER_FLAGS:
            prof[col] = int(round(float(sub[col].median()))) if col in sub.columns else 0
        groups[key] = prof

    return {"groups": groups, "meta": meta, "labels_vi": WEATHER_GROUP_VI}


def dump_weather_groups(path: str | Path, factors_hourly_path: str | Path) -> Path:
    out = Path(path)
    payload = build_weather_groups(factors_hourly_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    return out


@lru_cache(maxsize=4)
def load_weather_groups(ui_dir: str, mtime: float) -> dict[str, Any]:
    _ = mtime
    path = Path(ui_dir) / "weather_groups.json"
    if not path.exists():
        raise FileNotFoundError(f"Thiếu {path} — chạy scripts/build_weather_groups.py")
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def apply_weather_group_to_hourly(
    hourly: pd.DataFrame,
    group_key: str,
    weather_groups: dict[str, Any],
) -> pd.DataFrame:
    """Override weather columns; keep calendar columns from hourly."""
    groups = weather_groups["groups"]
    if group_key not in groups:
        raise ValueError(f"Nhóm thời tiết không hợp lệ: {group_key}")
    prof = groups[group_key]
    out = hourly.copy()
    for col in WEATHER_NUMERIC + WEATHER_FLAGS:
        if col in prof:
            out[col] = prof[col]
    return out


def build_hourly_with_weather_group(
    target: date,
    group_key: str,
    *,
    weather_groups: dict[str, Any],
    feature_medians: dict[str, float],
    factors_hourly_path: str | None = None,
    factors_hourly_mtime: float = 0.0,
) -> tuple[pd.DataFrame, str]:
    """Calendar from date + weather from training-derived group preset."""
    is_holiday = ny_holiday_flag(target)
    cal = calendar_row_for_date(target, is_holiday=is_holiday)
    prof = weather_groups["groups"][group_key]
    rows: list[dict[str, Any]] = []
    for hour in range(24):
        row: dict[str, Any] = {
            "hour": hour,
            "date": pd.Timestamp(target),
            "day_of_week": cal["day_of_week"],
            "is_weekend": cal["is_weekend"],
            "is_us_holiday": cal["is_us_holiday"],
            "month": cal["month"],
            "is_major_event_window": cal.get("is_major_event_window", 0),
            "is_peak_morning": int(hour in (7, 8, 9)),
            "is_peak_evening": int(hour in (17, 18, 19)),
            "is_overnight": int(hour <= 5 or hour >= 23),
        }
        for col in WEATHER_NUMERIC + WEATHER_FLAGS:
            row[col] = prof.get(col, feature_medians.get(col, 0.0))
        rows.append(row)
    raw = pd.DataFrame(rows)
    prepared = prepare_hourly_factors_for_model(
        raw,
        feature_medians=feature_medians,
        factors_hourly_path=factors_hourly_path,
        factors_hourly_mtime=factors_hourly_mtime,
    )
    label = f"Nhóm {WEATHER_GROUP_VI.get(group_key, group_key)} · {target.isoformat()}"
    return prepared.hourly_factors, label
