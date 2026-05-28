# datasets — bo danh gia chi tiet

| File | Vai tro |
|------|---------|
| `ridership.csv` | Du lieu nhu cau theo gio |
| `factors_hourly.csv` | Feature thoi tiet + lich theo gio |
| `routes_by_station_complex.csv` | Lo trinh/tuyen theo station complex |
| `routes.csv` | Data route (metadata route_id) |
| `schedule_current/*.txt` | Lich trinh GTFS hien tai |
| `train_manifest.json` | Khoa join (machine-readable) |

Ridership: co

`python consolidate_datasets.py --years 2024 2025 --hardlink`
