"""Load exported station-level demand artifacts; predict boarding per station and
project to route×hour departure demand for API inference (Departure-Time Projection)."""

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
    """Blend MLP + HistGBM residual demand model (STATION-LEVEL) loaded from ui_export/.

    Dự báo lượng boarding tại từng GA theo giờ, sau đó CHIẾU NGƯỢC KHÔNG–THỜI GIAN
    (frequency weights + travel offsets) để tái dựng cầu theo (route, direction, giờ xuất bến).
    """

    keras_model: tf.keras.Model
    scaler: Any
    histgbm_model: Any | None
    blend_mlp_weight: float
    resid_clip: tuple[float, float]
    num_features: list[str]
    station_to_idx: dict[str, int]
    baseline_lookup: pd.Series
    fallback: pd.Series
    lag_feature_defaults: dict[str, dict[str, float]]
    feature_medians: dict[str, float]
    fb_station_weekend: pd.Series | None
    fb_hour_weekend: pd.Series | None
    fb_global: float
    use_entity_embedding: bool
    use_lag_features: bool
    lag_feature_cols: list[str]
    hourly_factor_cols: list[str]
    station_rd_weights: pd.DataFrame | None
    travel_offsets: pd.DataFrame | None
    board_minute: float
    lag_imputer: Any | None = None

    @classmethod
    def load(cls, ui_dir: Path | str = DEFAULT_UI_DIR) -> DemandPredictor:
        root = Path(ui_dir)
        bundle = joblib.load(root / "demand_inference.pkl")
        model = tf.keras.models.load_model(root / "demand_model.keras")

        entity_idx = bundle.get("station_to_idx") or bundle.get("route_to_idx") or {}
        return cls(
            keras_model=model,
            scaler=bundle["scaler"],
            histgbm_model=bundle.get("histgbm_model"),
            blend_mlp_weight=float(bundle.get("blend_mlp_weight", 0.5)),
            resid_clip=tuple(bundle.get("resid_clip", (-0.55, 0.55))),
            num_features=list(bundle["num_features"]),
            station_to_idx={str(k): int(v) for k, v in entity_idx.items()},
            baseline_lookup=bundle["baseline_lookup"],
            fallback=bundle["fallback"],
            lag_feature_defaults=bundle.get("lag_feature_defaults", {}),
            feature_medians=bundle.get("feature_medians", {}),
            fb_station_weekend=bundle.get("fb_station_weekend", bundle.get("fb_route_weekend")),
            fb_hour_weekend=bundle.get("fb_hour_weekend"),
            fb_global=float(bundle.get("fb_global", 100.0)),
            use_entity_embedding=bool(bundle.get("use_route_embedding", True)),
            use_lag_features=bool(bundle.get("use_lag_features")),
            lag_feature_cols=list(bundle.get("lag_feature_cols", [])),
            hourly_factor_cols=list(bundle.get("hourly_factor_cols", [])),
            station_rd_weights=bundle.get("station_rd_weights"),
            travel_offsets=bundle.get("travel_offsets"),
            board_minute=float(bundle.get("board_minute", 30.0)),
        )

    # ---- ánh xạ route ↔ ga ----
    def stations_for_routes(self, routes: list[str]) -> list[str]:
        """Các ga (có trong scope huấn luyện) phục vụ các route yêu cầu."""
        if self.station_rd_weights is None:
            return []
        rs = {str(r) for r in routes}
        w = self.station_rd_weights
        mask = w["route"].astype(str).isin(rs)
        stations = w.loc[mask, "station_complex_id"].astype(str).unique().tolist()
        return sorted(s for s in stations if s in self.station_to_idx)

    # ---- dựng feature cấp ga ----
    def build_station_features(
        self,
        stations: list[str],
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
        rows: list[dict[str, Any]] = []
        for station_id in stations:
            for hour in hours:
                row: dict[str, Any] = {
                    "station_complex_id": str(station_id),
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
                    f"{r['station_complex_id']}|{int(r['hour'])}", {}
                )
                missing = [c for c in self.lag_feature_cols if c not in defaults]
                imputed: dict[str, Any] = {}
                if missing and self.lag_imputer is not None:
                    ctx = {
                        "hour_sin": float(r.get("hour_sin", 0.0)),
                        "hour_cos": float(r.get("hour_cos", 0.0)),
                        "is_weekend": float(r.get("is_weekend", 0)),
                        "month_sin": float(r.get("month_sin", 0.0)),
                        "month_cos": float(r.get("month_cos", 0.0)),
                    }
                    imputed = self.lag_imputer.impute(ctx)
                for c in self.lag_feature_cols:
                    if c in defaults:
                        feat.at[i, c] = float(defaults[c])
                    elif c in imputed:
                        feat.at[i, c] = float(imputed[c])
                    else:
                        feat.at[i, c] = float(self.feature_medians.get(c, 0.0))
        feat["station_idx"] = feat["station_complex_id"].map(self.station_to_idx).fillna(0).astype(int)
        feat["route_idx"] = feat["station_idx"]
        return feat

    def build_station_features_from_hourly_df(
        self,
        stations: list[str],
        factors_hourly: pd.DataFrame,
    ) -> pd.DataFrame:
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
        return self.build_station_features(
            stations,
            hours,
            day_of_week=int(first.get("day_of_week", 0)),
            is_weekend=int(first.get("is_weekend", 0)),
            is_us_holiday=int(first.get("is_us_holiday", 0)),
            month=int(first.get("month", 6)),
            is_major_event_window=int(first.get("is_major_event_window", 0)),
            weather_by_hour=weather_by_hour,
        )

    def build_station_features_from_daily(
        self,
        stations: list[str],
        daily: dict[str, Any] | pd.Series,
    ) -> pd.DataFrame:
        d = dict(daily)
        hours = list(range(24))
        weather_by_hour = {h: daily_factors_to_hourly(d, h) for h in hours}
        return self.build_station_features(
            stations,
            hours,
            day_of_week=int(d.get("day_of_week", 0)),
            is_weekend=int(d.get("is_weekend", 0)),
            is_us_holiday=int(d.get("is_us_holiday", 0)),
            month=int(d.get("month", 6)),
            is_major_event_window=int(d.get("is_major_event_window", 0)),
            weather_by_hour=weather_by_hour,
        )

    def _attach_baseline(self, feat: pd.DataFrame) -> pd.DataFrame:
        bl = feat.merge(
            self.baseline_lookup.reset_index(),
            on=["station_complex_id", "hour", "is_weekend"],
            how="left",
        )
        bl = bl.merge(
            self.fallback.reset_index(),
            on=["station_complex_id", "hour"],
            how="left",
        )
        if self.fb_station_weekend is not None:
            bl = bl.merge(self.fb_station_weekend.reset_index(), on=["station_complex_id", "is_weekend"], how="left")
        else:
            bl["fb_station_weekend"] = np.nan
        if self.fb_hour_weekend is not None:
            bl = bl.merge(self.fb_hour_weekend.reset_index(), on=["hour", "is_weekend"], how="left")
        else:
            bl["fb_hour_weekend"] = np.nan
        bl["baseline_demand"] = (
            pd.to_numeric(bl["baseline_demand"], errors="coerce")
            .fillna(bl["fallback_demand"])
            .fillna(bl["fb_station_weekend"])
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
        entity_idx = bl["route_idx"].values.reshape(-1, 1)
        log_baseline = bl["log_baseline"].values
        return entity_idx, x_num, log_baseline

    def predict_residual(self, entity_idx: np.ndarray, x_num: np.ndarray) -> np.ndarray:
        if self.use_entity_embedding:
            pred_mlp = np.clip(
                self.keras_model.predict(
                    {"route_idx": entity_idx, "num_features": x_num},
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

    def predict_station(self, feat: pd.DataFrame) -> pd.DataFrame:
        """Dự báo boarding cấp GA: exp(log_baseline + residual) - 1 → (station, hour, demand)."""
        bl = self._attach_baseline(feat)
        entity_idx, x_num, log_baseline = self._prepare_matrix(bl)
        pred_resid = self.predict_residual(entity_idx, x_num)
        demand = srp.residuals_to_demand(log_baseline, pred_resid, self.resid_clip)
        fill = float(np.nanmedian(bl["baseline_demand"]))
        demand = np.clip(np.nan_to_num(demand, nan=fill), 0.0, None)
        return bl[["station_complex_id", "hour"]].assign(demand=demand)

    # backward-compat alias
    def predict(self, feat: pd.DataFrame) -> pd.DataFrame:
        return self.predict_station(feat)

    def predict_routes(
        self,
        routes: list[str],
        factors_hourly: pd.DataFrame,
    ) -> pd.DataFrame:
        """Cầu theo (route, hour) ở trục xuất bến = Departure-Time Projection cầu ga.

        Tổng hợp hai chiều → một dòng / (route, hour) để khớp hợp đồng API route×hour.
        """
        stations = self.stations_for_routes(routes)
        empty = pd.DataFrame(columns=["route_id", "hour", "demand"])
        if not stations or self.station_rd_weights is None or self.travel_offsets is None:
            return empty
        feat = self.build_station_features_from_hourly_df(stations, factors_hourly)
        station_demand = self.predict_station(feat)
        hours = sorted(int(h) for h in station_demand["hour"].unique())
        dirs = sorted(int(d) for d in pd.to_numeric(
            self.station_rd_weights["direction_id"], errors="coerce"
        ).dropna().unique())
        slot_route, slot_dir, slot_hour = [], [], []
        for r in routes:
            for d in dirs:
                for h in hours:
                    slot_route.append(str(r))
                    slot_dir.append(int(d))
                    slot_hour.append(int(h))
        arr = srp.project_station_demand_to_departures(
            station_demand,
            self.station_rd_weights,
            self.travel_offsets,
            slot_route=np.asarray(slot_route),
            slot_dir=np.asarray(slot_dir),
            slot_hour=np.asarray(slot_hour),
            board_minute=self.board_minute,
        )
        proj = pd.DataFrame({"route_id": slot_route, "hour": slot_hour, "demand": arr})
        return proj.groupby(["route_id", "hour"], as_index=False)["demand"].sum()
