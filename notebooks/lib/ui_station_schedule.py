"""Station-level schedule expansion + borough mapping for UI/API."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import pandas as pd

from . import single_route_pipeline as srp

NYC_BOROUGHS: tuple[str, ...] = (
    "Manhattan",
    "Brooklyn",
    "Queens",
    "Bronx",
    "Staten Island",
)

BOROUGH_VI: dict[str, str] = {
    "Manhattan": "Manhattan",
    "Brooklyn": "Brooklyn",
    "Queens": "Queens",
    "Bronx": "Bronx",
    "Staten Island": "Staten Island",
}


@lru_cache(maxsize=2)
def _load_station_borough_map(ridership_path: str, mtime: float) -> pd.DataFrame:
    _ = mtime
    r = pd.read_csv(
        ridership_path,
        usecols=["station_complex_id", "station_complex", "borough"],
        dtype=str,
    )
    r = r.drop_duplicates("station_complex_id")
    r["station_key"] = (
        r["station_complex"]
        .str.lower()
        .str.replace(r" \(.*\)", "", regex=True)
        .str.strip()
    )
    return r


@lru_cache(maxsize=2)
def _load_gtfs_parent_names(schedule_dir: str, mtime: float) -> pd.DataFrame:
    _ = mtime
    stops = pd.read_csv(Path(schedule_dir) / "stops.txt", dtype=str)
    parents = stops.loc[stops["location_type"] == "1", ["stop_id", "stop_name"]].copy()
    parents["station_key"] = parents["stop_name"].str.lower().str.strip()
    return parents


def map_stops_to_borough(
    station_times: pd.DataFrame,
    *,
    ridership_path: str,
    schedule_dir: str,
    ridership_mtime: float = 0.0,
    schedule_mtime: float = 0.0,
) -> pd.DataFrame:
    """Gắn station_complex_id, station_name, borough qua tên ga (ridership không có borough trong routes CSV)."""
    out = station_times.copy()
    ridership = _load_station_borough_map(ridership_path, ridership_mtime)
    parents = _load_gtfs_parent_names(schedule_dir, schedule_mtime)
    out["station_key"] = out["stop_name"].str.lower().str.strip()
    out = out.merge(
        parents.rename(columns={"stop_id": "parent_stop_id", "stop_name": "gtfs_stop_name"}),
        on="parent_stop_id",
        how="left",
    )
    out["station_key"] = out["gtfs_stop_name"].fillna(out["stop_name"]).str.lower().str.strip()
    meta = ridership[["station_complex_id", "station_complex", "borough", "station_key"]]
    out = out.merge(meta, on="station_key", how="left")
    out["station_name"] = out["station_complex"].fillna(out["stop_name"])
    out["borough"] = out["borough"].fillna("Unknown")
    return out


def schedule_by_station(
    schedule_df: pd.DataFrame,
    *,
    schedule_dir: str,
    ridership_path: str,
    route_id: str | None = None,
    schedule_dir_mtime: float = 0.0,
    ridership_mtime: float = 0.0,
    offset_templates: pd.DataFrame | None = None,
) -> list[dict[str, Any]]:
    """Expand optimizer schedule → list records for API/UI."""
    expanded = srp.expand_schedule_to_station_times(
        schedule_df,
        Path(schedule_dir),
        offset_templates=offset_templates,
        route_id=route_id,
    )
    if expanded.empty:
        return []

    mapped = map_stops_to_borough(
        expanded,
        ridership_path=ridership_path,
        schedule_dir=schedule_dir,
        ridership_mtime=ridership_mtime,
        schedule_mtime=schedule_dir_mtime,
    )
    records: list[dict[str, Any]] = []
    for _, row in mapped.iterrows():
        records.append(
            {
                "station_complex_id": row.get("station_complex_id"),
                "station_name": row.get("station_name"),
                "borough": row.get("borough"),
                "route": str(row["route"]),
                "direction": int(row["direction"]),
                "hour": int(row["hour"]),
                "scheduled_time": row["scheduled_time"],
                "stop_sequence": int(row["stop_sequence"]),
            }
        )
    return records


def station_schedule_dataframe(records: list[dict[str, Any]]) -> pd.DataFrame:
    if not records:
        return pd.DataFrame(
            columns=[
                "station_complex_id",
                "station_name",
                "borough",
                "route",
                "direction",
                "hour",
                "scheduled_time",
            ]
        )
    return pd.DataFrame(records)
