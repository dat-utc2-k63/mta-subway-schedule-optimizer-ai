from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from demo.config import HOURLY_FACTOR_COLS

# Mua -> thang dai dien (NYC). Dung de suy ra month_sin/month_cos cho mo hinh.
SEASONS: dict[str, int] = {
    "🌸 Mua xuan": 4,
    "☀️ Mua he": 7,
    "🍂 Mua thu": 10,
    "❄️ Mua dong": 1,
}

# Thu dai dien theo loai ngay (giua tuan / cuoi tuan).
WEEKDAY_REPRESENTATIVE_DOW = 2  # Thu Tu
WEEKEND_REPRESENTATIVE_DOW = 5  # Thu Bay


WEATHER_PRESETS = {
    "☀️ Nang dep (Clear)": dict(temperature_c=22.0, apparent_temperature_c=21.0, precipitation_mm=0.0, rain_mm=0.0, snowfall_cm=0.0, windspeed_kmh=10.0, windgusts_kmh=18.0, is_rain=0, is_snow=0, is_severe_wind=0),
    "🌦️ Mua nhe (Light Rain)": dict(temperature_c=16.0, apparent_temperature_c=14.0, precipitation_mm=3.5, rain_mm=3.5, snowfall_cm=0.0, windspeed_kmh=18.0, windgusts_kmh=30.0, is_rain=1, is_snow=0, is_severe_wind=0),
    "⛈️ Mua lon (Heavy Rain)": dict(temperature_c=14.0, apparent_temperature_c=11.0, precipitation_mm=25.0, rain_mm=25.0, snowfall_cm=0.0, windspeed_kmh=40.0, windgusts_kmh=65.0, is_rain=1, is_snow=0, is_severe_wind=1),
    "❄️ Tuyet nhe (Light Snow)": dict(temperature_c=-2.0, apparent_temperature_c=-6.0, precipitation_mm=4.0, rain_mm=0.0, snowfall_cm=5.0, windspeed_kmh=20.0, windgusts_kmh=35.0, is_rain=0, is_snow=1, is_severe_wind=0),
    "🌨️ Bao tuyet (Snowstorm)": dict(temperature_c=-8.0, apparent_temperature_c=-15.0, precipitation_mm=30.0, rain_mm=0.0, snowfall_cm=35.0, windspeed_kmh=55.0, windgusts_kmh=90.0, is_rain=0, is_snow=1, is_severe_wind=1),
    "🌤️ Nhieu may (Overcast)": dict(temperature_c=12.0, apparent_temperature_c=10.0, precipitation_mm=0.0, rain_mm=0.0, snowfall_cm=0.0, windspeed_kmh=15.0, windgusts_kmh=22.0, is_rain=0, is_snow=0, is_severe_wind=0),
    "🌬️ Gio manh kho (Severe Wind)": dict(temperature_c=10.0, apparent_temperature_c=5.0, precipitation_mm=0.0, rain_mm=0.0, snowfall_cm=0.0, windspeed_kmh=60.0, windgusts_kmh=85.0, is_rain=0, is_snow=0, is_severe_wind=1),
    "🌡️ Nang nong (Heatwave)": dict(temperature_c=38.0, apparent_temperature_c=43.0, precipitation_mm=0.0, rain_mm=0.0, snowfall_cm=0.0, windspeed_kmh=8.0, windgusts_kmh=14.0, is_rain=0, is_snow=0, is_severe_wind=0),
}


@dataclass
class ScenarioConfig:
    is_weekend: int
    day_of_week: int
    month: int
    is_us_holiday: int
    is_major_event_window: int
    weather: dict


def build_scenario_hourly_factors(config: ScenarioConfig) -> pd.DataFrame:
    rows = []
    for hour in range(24):
        row = {
            "hour": hour,
            "day_of_week": int(config.day_of_week),
            "is_weekend": int(config.is_weekend),
            "is_us_holiday": int(config.is_us_holiday),
            "month": int(config.month),
            "is_major_event_window": int(config.is_major_event_window),
        }
        row.update(config.weather)
        rows.append(row)
    df = pd.DataFrame(rows)
    for col in HOURLY_FACTOR_COLS:
        if col not in df.columns:
            df[col] = 0
    return df
