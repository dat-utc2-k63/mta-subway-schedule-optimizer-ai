from __future__ import annotations

from pathlib import Path

UI_EXPORT_DIR = Path("notebooks/outputs/default/ui_export")
GTFS_DIR = Path("datasets/schedule_current")

OPT_ROUTES = [
    "1", "2", "3", "4", "5", "6", "7", "A", "B", "C", "D", "E", "F", "G",
    "J", "L", "M", "N", "Q", "R", "SI", "W",
]
DIRECTIONS = [0, 1]

HOURLY_FACTOR_COLS = [
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
    "is_major_event_window",
]

LAMBDA_GRID = [100, 150, 200, 400, 600, 1000]
LAMBDA_KNEE_FALLBACK = 200

NYC_LAT = 40.7128
NYC_LON = -74.0060

CAPACITY_PER_TRIP = 1200.0
COST_PER_VEHICLE_HOUR = 150.0
MIN_HEADWAY_MIN = 3.0
MAX_HEADWAY_MIN = 20.0
OVERNIGHT_MAX_HEADWAY_MIN = 30.0
SMOOTHNESS_MAX_DELTA = 3
TRIPS_MIN_FACTOR = 0.5
TRIPS_OVERNIGHT_MIN_FACTOR = 0.32
TRIPS_DAYTIME_MAX_FACTOR = 1.25   # rang buoc TRIPS_MAX thong nhat ban ngay (07-22)
TRIPS_OVERNIGHT_MAX_FACTOR = 1.05

# Bound ratio mac dinh (% slot duoc phep cham TRIPS_MIN / TRIPS_MAX)
# Ban dem it chuyen hon -> cho phep nhieu slot o MIN hon, it slot o MAX hon.
DEFAULT_MAX_DAY_AT_MIN_RATIO = 0.45
DEFAULT_MAX_DAY_AT_MAX_RATIO = 0.55
DEFAULT_MAX_NIGHT_AT_MIN_RATIO = 1.0
DEFAULT_MAX_NIGHT_AT_MAX_RATIO = 0.30
