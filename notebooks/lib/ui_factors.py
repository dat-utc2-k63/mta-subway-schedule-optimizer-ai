"""Align crawled / scenario hourly factors to training schema with impute + clip."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

# Weather + calendar columns used at train time (hourly factor subset).
HOURLY_FACTOR_COLS: tuple[str, ...] = (
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
    "day_of_week",
    "is_weekend",
    "is_us_holiday",
    "month",
)

CLIP_PERCENTILES = (0.01, 0.99)


@dataclass
class FactorPrepareResult:
    hourly_factors: pd.DataFrame
    clipped_fields: list[str]
    imputed_fields: list[str]
    clip_note: str | None


def _clip_bounds_from_training(
    factors_hourly_path: str,
    columns: list[str],
) -> dict[str, tuple[float, float]]:
    fh = pd.read_csv(factors_hourly_path)
    bounds: dict[str, tuple[float, float]] = {}
    for col in columns:
        if col not in fh.columns:
            continue
        s = pd.to_numeric(fh[col], errors="coerce").dropna()
        if s.empty:
            continue
        lo, hi = float(s.quantile(CLIP_PERCENTILES[0])), float(s.quantile(CLIP_PERCENTILES[1]))
        if np.isfinite(lo) and np.isfinite(hi) and lo <= hi:
            bounds[col] = (lo, hi)
    return bounds


@lru_cache(maxsize=4)
def load_feature_clip_ranges(factors_hourly_path: str, mtime: float) -> dict[str, tuple[float, float]]:
    """Percentile clip bounds from training factors_hourly.csv."""
    _ = mtime
    numeric = [c for c in HOURLY_FACTOR_COLS if not c.startswith("is_") and c not in ("day_of_week", "month")]
    return _clip_bounds_from_training(factors_hourly_path, numeric)


def prepare_hourly_factors_for_model(
    hourly: pd.DataFrame,
    *,
    feature_medians: dict[str, float],
    factors_hourly_path: str | None = None,
    factors_hourly_mtime: float = 0.0,
    required_cols: tuple[str, ...] | None = None,
) -> FactorPrepareResult:
    """Impute missing fields, enforce column set, clip extremes to training range."""
    cols = required_cols or HOURLY_FACTOR_COLS
    out = hourly.copy()
    imputed: list[str] = []
    clipped: list[str] = []

    for col in cols:
        if col not in out.columns:
            out[col] = feature_medians.get(col, 0.0)
            imputed.append(col)
        else:
            miss = out[col].isna()
            if miss.any():
                out.loc[miss, col] = feature_medians.get(col, 0.0)
                if col not in imputed:
                    imputed.append(col)

    bounds: dict[str, tuple[float, float]] = {}
    if factors_hourly_path:
        bounds = load_feature_clip_ranges(factors_hourly_path, factors_hourly_mtime)

    for col, (lo, hi) in bounds.items():
        if col not in out.columns:
            continue
        raw = pd.to_numeric(out[col], errors="coerce")
        below = raw < lo
        above = raw > hi
        if below.any() or above.any():
            out[col] = raw.clip(lo, hi)
            clipped.append(col)

    clip_note: str | None = None
    if clipped:
        clip_note = (
            "Một số giá trị đã được giới hạn theo phạm vi dữ liệu huấn luyện "
            f"({', '.join(sorted(set(clipped)))})."
        )

    return FactorPrepareResult(
        hourly_factors=out,
        clipped_fields=clipped,
        imputed_fields=imputed,
        clip_note=clip_note,
    )


def load_feature_medians(ui_dir: Path | str) -> dict[str, float]:
    import json

    path = Path(ui_dir) / "feature_medians.json"
    with path.open(encoding="utf-8") as f:
        return {k: float(v) for k, v in json.load(f).items()}
