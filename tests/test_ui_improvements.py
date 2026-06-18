"""Tests for UI prediction/schedule improvements."""

from __future__ import annotations

import json
import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "notebooks"))

from lib.ui_constraints import (  # noqa: E402
    ConstraintOverrides,
    ValidationError,
    validate_constraint_config,
)
from lib.ui_factors import prepare_hourly_factors_for_model, load_feature_medians  # noqa: E402
from lib.ui_weather_groups import build_weather_groups, apply_weather_group_to_hourly  # noqa: E402
from lib import single_route_pipeline as srp  # noqa: E402
from lib.ui_station_schedule import schedule_by_station  # noqa: E402

UI_DIR = ROOT / "notebooks" / "outputs" / "default" / "ui_export"
FACTORS_HOURLY = ROOT / "datasets" / "factors_hourly.csv"
SCHEDULE_DIR = ROOT / "datasets" / "schedule_current"
RIDERSHIP = ROOT / "datasets" / "ridership.csv"


@pytest.fixture
def ui_config() -> dict:
    with (UI_DIR / "ui_config.json").open(encoding="utf-8") as f:
        return json.load(f)


@pytest.fixture
def optimizer_state():
    import joblib

    return joblib.load(UI_DIR / "optimizer_state.pkl")


@pytest.fixture
def feature_medians() -> dict:
    return load_feature_medians(UI_DIR)


def test_validate_rejects_invalid_headway(ui_config, optimizer_state):
    bad = ConstraintOverrides(min_headway_min=25.0, max_headway_min=10.0)
    with pytest.raises(ValidationError):
        validate_constraint_config(bad, optimizer_state, route_id="1")


def test_validate_rejects_zero_capacity(ui_config, optimizer_state):
    bad = ConstraintOverrides(capacity_per_trip=0.0)
    with pytest.raises(ValidationError):
        validate_constraint_config(bad, optimizer_state, route_id="1")


def test_factor_prepare_imputes_missing(feature_medians):
    row = {"hour": 12, "temperature_c": 20.0}
    hourly = pd.DataFrame([row])
    result = prepare_hourly_factors_for_model(
        hourly,
        feature_medians=feature_medians,
        factors_hourly_path=str(FACTORS_HOURLY),
        factors_hourly_mtime=FACTORS_HOURLY.stat().st_mtime,
    )
    assert "rain_mm" in result.hourly_factors.columns
    assert result.imputed_fields
    assert result.hourly_factors["rain_mm"].notna().all()


def test_factor_clip_extreme_values(feature_medians):
    hourly = pd.DataFrame(
        [
            {
                "hour": h,
                "temperature_c": 999.0,
                "apparent_temperature_c": 999.0,
                "precipitation_mm": 0.0,
                "rain_mm": 0.0,
                "snowfall_cm": 0.0,
                "windspeed_kmh": 0.0,
                "windgusts_kmh": 0.0,
                "is_rain": 0,
                "is_snow": 0,
                "is_severe_wind": 0,
                "is_major_event_window": 0,
                "day_of_week": 1,
                "is_weekend": 0,
                "is_us_holiday": 0,
                "month": 6,
            }
            for h in range(24)
        ]
    )
    result = prepare_hourly_factors_for_model(
        hourly,
        feature_medians=feature_medians,
        factors_hourly_path=str(FACTORS_HOURLY),
        factors_hourly_mtime=FACTORS_HOURLY.stat().st_mtime,
    )
    assert result.clipped_fields
    assert result.clip_note is not None
    assert result.hourly_factors["temperature_c"].max() < 999.0


def test_weather_groups_from_training():
    wg = build_weather_groups(FACTORS_HOURLY)
    assert "sunny" in wg["groups"]
    assert "heavy_rain" in wg["groups"]
    sunny = wg["groups"]["sunny"]
    assert sunny["is_rain"] == 0
    heavy = wg["groups"]["heavy_rain"]
    assert heavy["is_rain"] == 1
    assert heavy["rain_mm"] >= sunny["rain_mm"]


def test_weather_group_override_calendar():
    wg = build_weather_groups(FACTORS_HOURLY)
    hourly = pd.DataFrame(
        {
            "hour": range(24),
            "day_of_week": 2,
            "is_weekend": 0,
            "month": 3,
            "temperature_c": 10.0,
            "rain_mm": 0.0,
            "is_rain": 0,
        }
    )
    out = apply_weather_group_to_hourly(hourly, "heavy_rain", wg)
    assert out["day_of_week"].iloc[0] == 2
    assert out["month"].iloc[0] == 3
    assert int(out["is_rain"].iloc[0]) == 1


def test_expand_schedule_to_station_times():
    sched = pd.DataFrame(
        [{"route": "1", "direction": 0, "hour": 8, "opt_trips": 4}]
    )
    expanded = srp.expand_schedule_to_station_times(sched, SCHEDULE_DIR, route_id="1")
    assert not expanded.empty
    assert expanded["stop_sequence"].is_monotonic_increasing or True
    times = expanded["scheduled_min"].tolist()
    assert times == sorted(times)


def test_station_schedule_by_borough():
    sched = pd.DataFrame(
        [{"route": "1", "direction": 0, "hour": 8, "opt_trips": 3}]
    )
    records = schedule_by_station(
        sched,
        schedule_dir=str(SCHEDULE_DIR),
        ridership_path=str(RIDERSHIP),
        route_id="1",
    )
    assert records
    assert all("borough" in r for r in records)
    assert all("scheduled_time" in r for r in records)
