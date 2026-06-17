"""Load exported demand-model artifacts and predict route×hour demand for API inference."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import tensorflow as tf

from . import single_route_pipeline as srp

DEFAULT_UI_DIR = Path(__file__).resolve().parents[1] / "outputs" / "default" / "ui_export"

# Map factors_daily.csv → hourly model columns (same day; no peak/overnight flags in train).
_DAILY_TO_HOURLY = {
    "temperature_c": lambda d, h: (float(d.get("temp_max_c", 15)) + float(d.get("temp_min_c", 10))) / 2.0,
    "apparent_temperature_c": lambda d, h: (
        float(d.get("apparent_temp_max_c", 14)) + float(d.get("apparent_temp_min_c", 9))
    ) / 2.0,
    "precipitation_mm": lambda d, h: float(d.get("precipitation_mm", 0)),
    "rain_mm": lambda d, h: float(d.get("rain_mm", 0)),
    "snowfall_cm": lambda d, h: float(d.get("snowfall_cm", 0)),
    "windspeed_kmh": lambda d, h: float(d.get("wind_max_kmh", 0)) * 0.7,
    "windgusts_kmh": lambda d, h: float(d.get("wind_max_kmh", 0)),
    "is_rain": lambda d, h: int(d.get("is_rainy_day", 0)),
    "is_snow": lambda d, h: int(d.get("is_snowy_day", 0)),
    "is_severe_wind": lambda d, h: int(float(d.get("wind_max_kmh", 0)) >= 50),
    "is_major_event_window": lambda d, h: int(d.get("is_major_event_window", 0)),
}


def daily_factors_to_hourly(daily: dict[str, Any] | pd.Series, hour: int) -> dict[str, float | int]:
    """Chuyển một dòng factors_daily → cột thời tiết theo giờ (HOURLY_FACTOR_COLS)."""
    d = dict(daily)
    return {col: fn(d, hour) for col, fn in _DAILY_TO_HOURLY.items()}


def load_factors_hourly_for_date(path: Path | str, date: str | pd.Timestamp) -> pd.DataFrame:
    """Đọc factors_hourly.csv cho một ngày (24 dòng date×hour)."""
    fh = pd.read_csv(path, parse_dates=["timestamp", "date"])
    if "hour" not in fh.columns:
        fh["hour"] = fh["timestamp"].dt.hour
    target = pd.to_datetime(date).normalize()
    fh["date"] = pd.to_datetime(fh["date"]).dt.normalize()
    return fh.loc[fh["date"] == target].copy()


def load_factors_daily_row(path: Path | str, date: str | pd.Timestamp) -> pd.Series:
    """Đọc một dòng factors_daily.csv."""
    fd = pd.read_csv(path, parse_dates=["date"])
    target = pd.to_datetime(date).normalize()
    fd["date"] = pd.to_datetime(fd["date"]).dt.normalize()
    row = fd.loc[fd["date"] == target]
    if row.empty:
        raise KeyError(f"No factors_daily row for date={target.date()}")
    return row.iloc[0]


@dataclass
class DemandPredictor:
    """Blend MLP + HistGBM residual demand model loaded from ui_export/."""

    keras_model: tf.keras.Model
    scaler: Any
    histgbm_model: Any | None
    blend_mlp_weight: float
    resid_clip: tuple[float, float]
    num_features: list[str]
    route_to_idx: dict[str, int]
    baseline_lookup: pd.Series
    fallback: pd.Series
    lag_feature_defaults: dict[str, dict[str, float]]
    feature_medians: dict[str, float]
    fb_route_weekend: pd.Series
    fb_hour_weekend: pd.Series
    fb_global: float
    use_route_embedding: bool
    use_lag_features: bool
    lag_feature_cols: list[str]
    hourly_factor_cols: list[str]

    @classmethod
    def load(cls, ui_dir: Path | str = DEFAULT_UI_DIR) -> DemandPredictor:
        root = Path(ui_dir)
        bundle_path = root / "demand_inference.pkl"
        if bundle_path.exists():
            bundle = joblib.load(bundle_path)
        else:
            bundle = _load_legacy_bundle(root)

        model = tf.keras.models.load_model(root / "demand_model.keras")
        return cls(keras_model=model, **bundle)

    def build_features(
        self,
        routes: list[str],
        hours: list[int],
        *,
        day_of_week: int,
        is_weekend: int,
        is_us_holiday: int = 0,
        month: int = 6,
        is_major_event_window: int = 0,
        weather_by_hour: dict[int, dict[str, Any]] | None = None,
        uniform_weather: dict[str, Any] | None = None,
    ) -> pd.DataFrame:
        """Dựng feature frame route×hour; weather có thể khác theo giờ hoặc đồng nhất cả ngày."""
        rows: list[dict[str, Any]] = []
        for route_id in routes:
            for hour in hours:
                row: dict[str, Any] = {
                    "route_id": str(route_id),
                    "hour": int(hour),
                    "day_of_week": int(day_of_week),
                    "is_weekend": int(is_weekend),
                    "is_us_holiday": int(is_us_holiday),
                    "month": int(month),
                    "is_major_event_window": int(is_major_event_window),
                }
                if weather_by_hour is not None and hour in weather_by_hour:
                    row.update(weather_by_hour[hour])
                elif uniform_weather is not None:
                    row.update(uniform_weather)
                rows.append(row)

        feat = pd.DataFrame(rows)
        feat = srp.add_cyclical_time_features(feat)
        if self.use_lag_features:
            for i, r in feat.iterrows():
                defaults = self.lag_feature_defaults.get(
                    f"{r['route_id']}|{int(r['hour'])}", {}
                )
                for c in self.lag_feature_cols:
                    feat.at[i, c] = float(defaults.get(c, self.feature_medians.get(c, 0.0)))
        feat["route_idx"] = feat["route_id"].map(self.route_to_idx).astype(int)
        return feat

    def build_features_from_daily(
        self,
        routes: list[str],
        hours: list[int],
        daily: dict[str, Any] | pd.Series,
    ) -> pd.DataFrame:
        """Dựng features từ một dòng factors_daily (mưa, nắng, lễ, …)."""
        d = dict(daily)
        weather_by_hour = {
            h: daily_factors_to_hourly(d, h)
            for h in hours
        }
        return self.build_features(
            routes,
            hours,
            day_of_week=int(d.get("day_of_week", 0)),
            is_weekend=int(d.get("is_weekend", 0)),
            is_us_holiday=int(d.get("is_us_holiday", 0)),
            month=int(d.get("month", 6)),
            is_major_event_window=int(d.get("is_major_event_window", 0)),
            weather_by_hour=weather_by_hour,
        )

    def build_features_from_hourly_df(
        self,
        routes: list[str],
        factors_hourly: pd.DataFrame,
    ) -> pd.DataFrame:
        """Dựng features khi đã có bảng factors_hourly cho ngày (join theo hour)."""
        fh = factors_hourly.set_index("hour")
        hours = sorted(int(h) for h in fh.index.unique())
        first = fh.iloc[0]
        weather_by_hour: dict[int, dict[str, Any]] = {}
        for hour in hours:
            row = fh.loc[hour]
            if isinstance(row, pd.DataFrame):
                row = row.iloc[0]
            w: dict[str, Any] = {}
            for col in self.hourly_factor_cols:
                if col in row.index:
                    w[col] = row[col]
            weather_by_hour[hour] = w
        return self.build_features(
            routes,
            hours,
            day_of_week=int(first.get("day_of_week", 0)),
            is_weekend=int(first.get("is_weekend", 0)),
            is_us_holiday=int(first.get("is_us_holiday", 0)),
            month=int(first.get("month", 6)),
            is_major_event_window=int(first.get("is_major_event_window", 0)),
            weather_by_hour=weather_by_hour,
        )

    def _attach_baseline(self, feat: pd.DataFrame) -> pd.DataFrame:
        bl = feat.merge(
            self.baseline_lookup.reset_index(),
            on=["route_id", "hour", "is_weekend"],
            how="left",
        )
        bl = bl.merge(
            self.fallback.reset_index(),
            on=["route_id", "hour"],
            how="left",
        )
        if self.fb_route_weekend is not None:
            bl = bl.merge(self.fb_route_weekend.reset_index(), on=["route_id", "is_weekend"], how="left")
        else:
            bl["fb_route_weekend"] = np.nan
        if self.fb_hour_weekend is not None:
            bl = bl.merge(self.fb_hour_weekend.reset_index(), on=["hour", "is_weekend"], how="left")
        else:
            bl["fb_hour_weekend"] = np.nan
        bl["baseline_demand"] = (
            pd.to_numeric(bl["baseline_demand"], errors="coerce")
            .fillna(bl["fallback_demand"])
            .fillna(bl["fb_route_weekend"])
            .fillna(bl["fb_hour_weekend"])
            .fillna(self.fb_global)
            .clip(lower=1e-6)
        )
        bl["log_baseline"] = np.log1p(bl["baseline_demand"])
        return bl

    def _prepare_matrix(self, bl: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        for c in self.num_features:
            if c == "log_baseline":
                continue
            if c not in bl.columns:
                bl[c] = self.feature_medians.get(c, 0.0)
            bl[c] = pd.to_numeric(bl[c], errors="coerce").fillna(
                self.feature_medians.get(c, 0.0)
            )
        x_num = self.scaler.transform(bl[self.num_features])
        route_idx = bl["route_idx"].values.reshape(-1, 1)
        log_baseline = bl["log_baseline"].values
        return route_idx, x_num, log_baseline

    def predict_residual(self, route_idx: np.ndarray, x_num: np.ndarray) -> np.ndarray:
        if self.use_route_embedding:
            pred_mlp = np.clip(
                self.keras_model.predict(
                    {"route_idx": route_idx, "num_features": x_num},
                    verbose=0,
                ).reshape(-1),
                *self.resid_clip,
            )
        else:
            pred_mlp = np.clip(
                self.keras_model.predict(x_num, verbose=0).reshape(-1),
                *self.resid_clip,
            )
        if self.histgbm_model is not None:
            pred_gbm = self.histgbm_model.predict(x_num)
            pred_resid = srp.blend_residual_predictions(
                pred_mlp, pred_gbm, self.blend_mlp_weight
            )
        else:
            pred_resid = pred_mlp
        return pred_resid

    def predict(self, feat: pd.DataFrame) -> pd.DataFrame:
        """Dự báo demand: exp(log_baseline + pred_residual) - 1."""
        bl = self._attach_baseline(feat)
        route_idx, x_num, log_baseline = self._prepare_matrix(bl)
        pred_resid = self.predict_residual(route_idx, x_num)
        demand = srp.residuals_to_demand(log_baseline, pred_resid, self.resid_clip)
        fill = float(np.nanmedian(bl["baseline_demand"]))
        demand = np.clip(np.nan_to_num(demand, nan=fill), 0.0, None)
        return bl[["route_id", "hour"]].assign(demand=demand)


def _load_legacy_bundle(root: Path) -> dict[str, Any]:
    """Fallback khi chưa có demand_inference.pkl (export cũ)."""
    import json

    scaler = joblib.load(root / "scaler.pkl")
    histgbm = joblib.load(root / "histgbm_blend.pkl") if (root / "histgbm_blend.pkl").exists() else None
    with (root / "ui_config.json").open(encoding="utf-8") as f:
        cfg = json.load(f)
    with (root / "route_meta.json").open(encoding="utf-8") as f:
        route_meta = json.load(f)
    opt = joblib.load(root / "optimizer_state.pkl")
    clip_hi = float(cfg.get("resid_clip", 0.55))
    fb_global = float(opt.get("fb_global", 100.0))
    feature_medians: dict[str, float] = {}
    med_path = root / "feature_medians.json"
    if med_path.exists():
        with med_path.open(encoding="utf-8") as f:
            feature_medians = {k: float(v) for k, v in json.load(f).items()}
    skip = {
        "hour_sin", "hour_cos", "dow_sin", "dow_cos", "month_sin", "month_cos",
        "is_weekend", "is_us_holiday", "log_baseline",
        "log_lag_24h", "log_lag_168h", "log_rolling_7d",
    }
    return {
        "scaler": scaler,
        "histgbm_model": histgbm,
        "blend_mlp_weight": float(cfg.get("blend_mlp_weight", 0.5)),
        "resid_clip": (-clip_hi, clip_hi),
        "num_features": list(cfg["num_features"]),
        "route_to_idx": {str(k): int(v) for k, v in route_meta.get("route_to_idx", {}).items()},
        "baseline_lookup": opt["baseline_lookup"],
        "fallback": opt["fallback"],
        "lag_feature_defaults": opt.get("lag_feature_defaults", {}),
        "feature_medians": feature_medians,
        "fb_route_weekend": opt.get("fb_route_weekend"),
        "fb_hour_weekend": opt.get("fb_hour_weekend"),
        "fb_global": fb_global,
        "use_route_embedding": bool(cfg.get("use_route_embedding", True)),
        "use_lag_features": bool(cfg.get("lag_feature_cols")),
        "lag_feature_cols": list(cfg.get("lag_feature_cols", [])),
        "hourly_factor_cols": [c for c in cfg["num_features"] if c not in skip],
    }
