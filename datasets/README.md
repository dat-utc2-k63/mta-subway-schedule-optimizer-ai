# datasets — Winter 2023 → Fall 2025

| File | Range |
|------|--------|
| `ridership.csv` | **2023-12-01 → 2025-11-30** (~57.5M rows) |
| `factors_hourly.csv` | **2023-12-01 → 2025-11-30** (17,543 hours) |
| `factors_daily.csv` | **2023-12-01 → 2025-11-30** (731 days) |
| `routes.csv`, `routes_by_station_complex.csv`, `schedule_current/` | unchanged |

Join: `SUBSTR(ridership.transit_timestamp,1,19) = factors_hourly.timestamp`

Rebuild: `python trim_datasets_range.py`
