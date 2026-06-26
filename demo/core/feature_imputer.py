"""K-Means feature imputer cho cac feature numeric con thieu khi suy luan.

Thay cho viec gan cung gia tri median toan cuc, ta phan cum (K-Means) cac profile
(ga, gio) tu lag_feature_defaults + baseline, roi voi mot dong thieu feature ta gan
no vao cum gan nhat theo context co san (gio, cuoi tuan, mua, baseline) va dien
feature thieu bang tam (centroid) cua cum do.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


def _cyc(value: float, period: float) -> tuple[float, float]:
    ang = 2.0 * math.pi * float(value) / float(period)
    return math.sin(ang), math.cos(ang)


class KMeansFeatureImputer:
    """Phan cum profile va dien feature thieu bang centroid cum gan nhat.

    - `fit(df, cols)`: hoc cum tren cac dong day du (khong NaN) cua `cols`.
    - `impute(values)`: voi dict feature (mot so co the thieu/NaN), gan cum gan nhat
      theo cac feature CO SAN, roi dien cac feature thieu tu centroid.
    """

    def __init__(self, n_clusters: int = 12, random_state: int = 42) -> None:
        self.n_clusters = int(n_clusters)
        self.random_state = int(random_state)
        self.cols: list[str] = []
        self.scaler: StandardScaler | None = None
        self.km: KMeans | None = None
        self.centroids: np.ndarray | None = None
        self.medians: dict[str, float] = {}

    def fit(self, df: pd.DataFrame, cols: list[str]) -> "KMeansFeatureImputer":
        self.cols = [c for c in cols if c in df.columns]
        if not self.cols:
            return self
        X = df[self.cols].apply(pd.to_numeric, errors="coerce").dropna()
        if len(X) == 0:
            self.medians = {c: 0.0 for c in self.cols}
            return self
        self.medians = {c: float(v) for c, v in X.median().items()}
        if len(X) < 2:
            return self
        self.scaler = StandardScaler().fit(X.to_numpy())
        Xs = self.scaler.transform(X.to_numpy())
        k = int(min(self.n_clusters, len(X)))
        self.km = KMeans(n_clusters=k, n_init=10, random_state=self.random_state).fit(Xs)
        self.centroids = self.scaler.inverse_transform(self.km.cluster_centers_)
        return self

    def _fill_from_median(self, out: dict[str, Any]) -> dict[str, Any]:
        for c in self.cols:
            if c not in out or pd.isna(out.get(c)):
                out[c] = self.medians.get(c, 0.0)
        return out

    def impute(self, values: dict[str, Any]) -> dict[str, Any]:
        out = dict(values)
        if self.km is None or self.scaler is None or self.centroids is None:
            return self._fill_from_median(out)
        present = [
            (i, c)
            for i, c in enumerate(self.cols)
            if c in values and values[c] is not None and not pd.isna(values[c])
        ]
        if not present:
            return self._fill_from_median(out)
        idx = [i for i, _ in present]
        v = np.array([float(values[c]) for _, c in present], dtype=float)
        vs = (v - self.scaler.mean_[idx]) / self.scaler.scale_[idx]
        centers = self.km.cluster_centers_[:, idx]
        cluster = int(np.argmin(np.linalg.norm(centers - vs, axis=1)))
        for i, c in enumerate(self.cols):
            if c not in out or pd.isna(out.get(c)):
                out[c] = float(self.centroids[cluster, i])
        return out


def build_feature_imputer(predictor: Any, n_clusters: int = 12) -> KMeansFeatureImputer | None:
    """Fit imputer tu artifact da export (lag_feature_defaults + baseline_lookup).

    Profile moi (ga, gio, is_weekend): context = gio (sin/cos) + cuoi tuan + log_baseline;
    target = 3 feature tre. Mua (month) duoc nhan o luc impute neu co.
    """
    lfd = getattr(predictor, "lag_feature_defaults", None) or {}
    if not lfd:
        return None
    lag_cols = list(getattr(predictor, "lag_feature_cols", []) or ["log_lag_24h", "log_lag_168h", "log_rolling_7d"])
    baseline_lookup = getattr(predictor, "baseline_lookup", None)

    base_map: dict[tuple[str, int, int], float] = {}
    if isinstance(baseline_lookup, pd.Series):
        for key, val in baseline_lookup.items():
            try:
                station, hour, wk = key
                base_map[(str(station), int(hour), int(float(wk)))] = float(val)
            except Exception:
                continue

    rows: list[dict[str, float]] = []
    for key, lag in lfd.items():
        try:
            station, hour_s = key.rsplit("|", 1)
            hour = int(hour_s)
        except Exception:
            continue
        hs, hc = _cyc(hour, 24)
        for wk in (0, 1):
            base = base_map.get((str(station), hour, wk))
            rows.append(
                {
                    "hour_sin": hs,
                    "hour_cos": hc,
                    "is_weekend": float(wk),
                    "log_baseline": math.log1p(base) if base is not None else float("nan"),
                    **{c: float(lag.get(c, float("nan"))) for c in lag_cols},
                }
            )
    if not rows:
        return None
    df = pd.DataFrame(rows)
    cols = ["hour_sin", "hour_cos", "is_weekend", "log_baseline", *lag_cols]
    return KMeansFeatureImputer(n_clusters=n_clusters).fit(df, cols)


def context_for_impute(hour: int, is_weekend: int, month: int | None = None, log_baseline: float | None = None) -> dict[str, float]:
    """Dung context goi impute: gio + cuoi tuan (+ mua, baseline neu co)."""
    hs, hc = _cyc(int(hour), 24)
    ctx: dict[str, float] = {"hour_sin": hs, "hour_cos": hc, "is_weekend": float(int(is_weekend))}
    if log_baseline is not None and not pd.isna(log_baseline):
        ctx["log_baseline"] = float(log_baseline)
    if month is not None:
        ms, mc = _cyc(int(month), 12)
        ctx["month_sin"] = ms
        ctx["month_cos"] = mc
    return ctx
