"""Apply full-scope + new dataset config to mta_schedule_optimization.ipynb."""
import json
from pathlib import Path

NB_PATH = Path(__file__).parent / "mta_schedule_optimization.ipynb"
nb = json.loads(NB_PATH.read_text(encoding="utf-8"))


def set_cell(idx, src: str):
    nb["cells"][idx]["source"] = [src]
    if nb["cells"][idx]["cell_type"] == "code":
        nb["cells"][idx]["outputs"] = []
        nb["cells"][idx]["execution_count"] = None


CELL_2 = r'''import os
import json
import warnings
import random
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, r2_score

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import tensorflow as tf
from tensorflow.keras import layers, Model, Input
from tensorflow.keras.callbacks import EarlyStopping

warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid", context="notebook")
plt.rcParams["figure.dpi"] = 110

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

DATA_DIR = Path("../datasets")
OUT_DIR = Path("outputs")
OUT_DIR.mkdir(exist_ok=True)

# --- Đường dẫn (theo train_manifest.json) ---
RIDERSHIP_FILE = DATA_DIR / "ridership.csv"
MANIFEST_FILE = DATA_DIR / "train_manifest.json"
ROUTES_BY_COMPLEX_FILE = DATA_DIR / "routes_by_station_complex.csv"
ROUTES_META_FILE = DATA_DIR / "routes.csv"

# --- Phạm vi tối ưu ---
OPT_HOURS = list(range(24))
OPT_ROUTE_MODE = "headway"  # "headway" | "ridership" | "intersection"
MIN_ROUTE_DEMAND_ROWS = 50
ROUTE_ALIASES = {"SIR": "SI"}

# --- Ridership / NN ---
COVERAGE_THR = 0.7
CHUNK_ROWS = 200_000
NN_EPOCHS_MAIN = 200
NN_EPOCHS_CV = 80
CV_SPLITS = 5

# --- Optimizer ---
LAMBDA_COST = 150.0
LAMBDA_GRID = [100, 150, 200]
USE_ANALYTICAL_OPT = True
GA_POP_SIZE = 80
GA_GENERATIONS = 120
TABU_ITERS = 400
GA_MAX_SLOTS = 200

with open(MANIFEST_FILE, encoding="utf-8") as f:
    train_manifest = json.load(f)

print("TensorFlow:", tf.__version__)
print("Data dir:", DATA_DIR.resolve())
print("Output dir:", OUT_DIR.resolve())
print("Manifest:", train_manifest.get("purpose", ""))
print("Route aliases:", train_manifest.get("route_aliases", ROUTE_ALIASES))
print(f"Scope: OPT_HOURS={len(OPT_HOURS)}h, OPT_ROUTE_MODE={OPT_ROUTE_MODE}")
print(f"Optimizer: USE_ANALYTICAL_OPT={USE_ANALYTICAL_OPT}")
'''

CELL_3 = """## 2. Load & Hợp nhất dữ liệu thực

Bộ dữ liệu (xem [`datasets/README.md`](../datasets/README.md) và `train_manifest.json`):

- `ridership.csv` — nhu cầu theo giờ × ga (tháng 2026-01, 31 ngày).
- `routes_by_station_complex.csv` — tuyến GTFS theo `station_complex_id` (`gtfs_route_ids`).
- `headway_by_route_hour.csv` — lịch GTFS **24 giờ** × 29 tuyến (`headway_source`).
- `factors_daily.csv` — thời tiết, lễ, lịch.

**Merge**: đọc ridership theo chunks → phân bổ đều theo tuyến tại ga → `(route_id, date, hour)` — **không lọc giờ** trước khi chọn scope."""

CELL_4 = r'''routes_complex = pd.read_csv(ROUTES_BY_COMPLEX_FILE)
routes_complex["station_complex_id"] = routes_complex["station_complex_id"].astype(str)

route_col = "gtfs_route_ids" if "gtfs_route_ids" in routes_complex.columns else "route_ids_gtfs"
station_to_routes = (
    routes_complex.assign(route=routes_complex[route_col].astype(str).str.split())
    .explode("route")
    .dropna(subset=["route"])
)
station_to_routes = station_to_routes[station_to_routes["route"] != ""]
station_to_routes = station_to_routes[["station_complex_id", "route"]].drop_duplicates()
station_to_routes["route"] = station_to_routes["route"].replace(ROUTE_ALIASES)

route_counts = station_to_routes.groupby("station_complex_id").size().rename("n_routes_at_station")
station_to_routes = station_to_routes.merge(route_counts, on="station_complex_id")
station_to_routes["weight"] = 1.0 / station_to_routes["n_routes_at_station"]

print("Tổng map station→route:", len(station_to_routes))
print("Số ga có map:", station_to_routes["station_complex_id"].nunique())
print("Số tuyến xuất hiện:", station_to_routes["route"].nunique())
station_to_routes.head()
'''

CELL_5 = r'''usecols = ["transit_timestamp", "transit_mode", "station_complex_id", "ridership"]
valid_stations = set(station_to_routes["station_complex_id"].unique())

agg_parts = []
for chunk in pd.read_csv(
    RIDERSHIP_FILE,
    usecols=usecols,
    chunksize=CHUNK_ROWS,
    parse_dates=["transit_timestamp"],
    dtype={"station_complex_id": str},
):
    chunk = chunk[chunk["transit_mode"] == "subway"]
    chunk = chunk[chunk["station_complex_id"].isin(valid_stations)]
    if chunk.empty:
        continue
    chunk["date"] = chunk["transit_timestamp"].dt.date
    chunk["hour"] = chunk["transit_timestamp"].dt.hour
    grp = (
        chunk.groupby(["station_complex_id", "date", "hour"], as_index=False)["ridership"]
        .sum()
    )
    agg_parts.append(grp)

ridership_station = pd.concat(agg_parts, ignore_index=True)
print(f"Aggregated (station,date,hour) rows: {len(ridership_station):,}")
print("Date range:", ridership_station["date"].min(), "→", ridership_station["date"].max())
'''

CELL_8 = r'''ridership_all = ridership_route.merge(factors, on="date", how="left")
ridership_all["day_of_week"] = ridership_all["day_of_week"].fillna(ridership_all["date"].dt.dayofweek)
ridership_all["is_weekend"] = ridership_all["is_weekend"].fillna(
    (ridership_all["date"].dt.dayofweek >= 5).astype(int)
)

num_cols = ["temp_max_c", "temp_min_c", "precipitation_mm", "rain_mm",
            "snowfall_cm", "wind_max_kmh"]
flag_cols = ["is_us_holiday", "is_rainy_day", "is_snowy_day",
             "is_extreme_heat", "is_extreme_cold", "is_major_event_window"]
for c in num_cols:
    ridership_all[c] = ridership_all[c].fillna(ridership_all[c].median())
for c in flag_cols:
    ridership_all[c] = ridership_all[c].fillna(0).astype(int)

print("Merged ridership (all hours) shape:", ridership_all.shape)
print("Hours:", sorted(ridership_all["hour"].unique()))
ridership_all.head()
'''

CELL_9 = """## 3. EDA & Chọn scope tuyến

Mục tiêu: mô tả demand theo giờ/tuyến, chọn **toàn bộ tuyến headway (29)** × **24 giờ** cho tối ưu (~1.392 slot)."""

CELL_10 = r'''HEADWAY_ROUTES = sorted(headway["route_id"].astype(str).unique())
route_row_counts = ridership_all.groupby("route_id").size()
RIDERSHIP_ROUTES = route_row_counts[route_row_counts >= MIN_ROUTE_DEMAND_ROWS].index.astype(str).tolist()

if OPT_ROUTE_MODE == "headway":
    OPT_ROUTES = HEADWAY_ROUTES
elif OPT_ROUTE_MODE == "intersection":
    OPT_ROUTES = sorted(set(HEADWAY_ROUTES) & set(RIDERSHIP_ROUTES))
else:
    OPT_ROUTES = RIDERSHIP_ROUTES

route_totals = (
    ridership_all.groupby("route_id")["demand"].mean().sort_values(ascending=False)
)
print("Top 12 tuyến theo demand TB/giờ:")
print(route_totals.head(12).round(0))
print(f"\nOPT_ROUTES ({len(OPT_ROUTES)} tuyến, mode={OPT_ROUTE_MODE}):", OPT_ROUTES)

df = ridership_all[ridership_all["route_id"].isin(OPT_ROUTES)].copy()
print(f"\nDataset NN: {len(df):,} bản ghi, {df['date'].nunique()} ngày, {df['hour'].nunique()} giờ")
print(f"Demand range: {df['demand'].min():.0f} – {df['demand'].max():.0f}")
'''

CELL_11 = r'''fig, axes = plt.subplots(1, 3, figsize=(16, 4))

top_plot = route_totals.head(12).index.tolist()
heat = (
    df.groupby(["route_id", "hour"])["demand"].mean().unstack().loc[top_plot]
)
sns.heatmap(heat, ax=axes[0], cmap="YlOrRd", annot=False)
axes[0].set_title("Demand TB: top-12 tuyến × 24h")

wk = df.groupby(["is_weekend", "hour"])["demand"].mean().unstack().T
wk.index = ["Weekday", "Weekend"]
wk.plot(kind="bar", ax=axes[1], color=["#3498DB", "#E74C3C"])
axes[1].set_title("Demand TB theo giờ: Weekday vs Weekend")
axes[1].set_xlabel("Hour"); axes[1].legend()

rain = df.groupby(["is_rainy_day", "hour"])["demand"].mean().unstack().T
rain.columns = ["Clear", "Rainy"]
rain.plot(kind="bar", ax=axes[2], color=["#27AE60", "#2C3E50"])
axes[2].set_title("Demand TB: Clear vs Rainy")
axes[2].set_xlabel("Hour"); axes[2].legend()

plt.tight_layout()
plt.savefig(OUT_DIR / "fig_eda.png", dpi=120, bbox_inches="tight")
plt.show()
'''

CELL_19 = r'''DIRECTIONS = [0, 1]
OPT_SCOPE = [(r, d, h) for r in OPT_ROUTES for d in DIRECTIONS for h in OPT_HOURS]
N_SLOTS = len(OPT_SCOPE)
print(f"Scope tối ưu: {len(OPT_ROUTES)} tuyến × {len(DIRECTIONS)} hướng × {len(OPT_HOURS)} giờ = {N_SLOTS} slots")

slot_to_idx = {s: i for i, s in enumerate(OPT_SCOPE)}
slot_route = np.array([s[0] for s in OPT_SCOPE])
slot_dir   = np.array([s[1] for s in OPT_SCOPE])
slot_hour  = np.array([s[2] for s in OPT_SCOPE])

hw_lookup = headway.set_index(["route_id", "direction_id", "hour"])["trip_count"]
hw_rd_median = headway.groupby(["route_id", "direction_id"])["trip_count"].median()
hw_rh_median = headway.groupby(["route_id", "hour"])["trip_count"].median()
hw_hour_median = headway.groupby("hour")["trip_count"].median()
hw_global_median = float(headway["trip_count"].median())

n_fallback = 0
fallback_tiers = []
baseline_trips = np.zeros(N_SLOTS)
for i, (r, d, h) in enumerate(OPT_SCOPE):
    if (r, d, h) in hw_lookup.index:
        baseline_trips[i] = hw_lookup.loc[(r, d, h)]
        fallback_tiers.append(0)
    elif (r, d) in hw_rd_median.index:
        baseline_trips[i] = hw_rd_median.loc[(r, d)]
        fallback_tiers.append(1)
        n_fallback += 1
    elif (r, h) in hw_rh_median.index:
        baseline_trips[i] = hw_rh_median.loc[(r, h)]
        fallback_tiers.append(2)
        n_fallback += 1
    elif h in hw_hour_median.index:
        baseline_trips[i] = hw_hour_median.loc[h]
        fallback_tiers.append(3)
        n_fallback += 1
    else:
        baseline_trips[i] = hw_global_median
        fallback_tiers.append(4)
        n_fallback += 1

print(f"Baseline trips: mean={baseline_trips.mean():.1f}, "
      f"min={baseline_trips.min():.0f}, max={baseline_trips.max():.0f}")
print(f"Headway fallback: {n_fallback}/{N_SLOTS} slots "
      f"(tier 0=exact GTFS, 1=route×dir, 2=route×hour, 3=hour, 4=global)")

TRIPS_MIN = np.maximum(2, (baseline_trips * 0.5)).astype(int)
TRIPS_MAX = np.minimum(60, (baseline_trips * 1.5).astype(int))
TRIPS_MAX = np.maximum(TRIPS_MAX, baseline_trips.astype(int) + 5)
print(f"λ mặc định LAMBDA_COST = {LAMBDA_COST}")
print(f"TRIPS_MIN range: [{TRIPS_MIN.min()}, {TRIPS_MIN.max()}]")
print(f"TRIPS_MAX range: [{TRIPS_MAX.min()}, {TRIPS_MAX.max()}]")
'''

CELL_20 = r'''def build_scenario_features(scenario: str) -> pd.DataFrame:
    """Tạo feature frame cho một kịch bản demand — cùng schema với train."""
    if scenario == "weekday_peak":
        dow, is_wkd = 1, 0
        rain, is_rain, is_holiday, is_heat = 0.0, 0, 0, 0
    elif scenario == "weekend":
        dow, is_wkd = 5, 1
        rain, is_rain, is_holiday, is_heat = 0.0, 0, 0, 0
    elif scenario == "rainy_day":
        dow, is_wkd = 1, 0
        rain, is_rain, is_holiday, is_heat = 12.0, 1, 0, 0
    else:
        raise ValueError(scenario)

    rows = []
    for (r, d, h) in OPT_SCOPE:
        rows.append(dict(
            route_id=r, direction_id=d, hour=h, day_of_week=dow,
            is_weekend=is_wkd, is_us_holiday=is_holiday,
            hour_sin=np.sin(2*np.pi*h/24), hour_cos=np.cos(2*np.pi*h/24),
            dow_sin=np.sin(2*np.pi*dow/7), dow_cos=np.cos(2*np.pi*dow/7),
            temp_max_c=float(df["temp_max_c"].median()),
            precipitation_mm=rain, rain_mm=rain,
            snowfall_cm=0.0, wind_max_kmh=float(df["wind_max_kmh"].median()),
            is_rainy_day=is_rain, is_snowy_day=0,
            is_extreme_heat=is_heat, is_extreme_cold=0,
        ))
    feat = pd.DataFrame(rows)
    feat["route_idx"] = feat["route_id"].map(route_to_idx).astype(int)
    return feat

def get_scenario_demand(scenario: str) -> np.ndarray:
    """Trả về demand dự báo (hành khách/giờ) cho từng slot."""
    feat = build_scenario_features(scenario)
    bl = feat.merge(baseline_lookup, on=["route_id", "hour", "is_weekend"], how="left")
    bl = bl.merge(fallback, on=["route_id", "hour"], how="left")
    bl["baseline_demand"] = bl["baseline_demand"].fillna(bl["fallback_demand"])
    bl["log_baseline"] = np.log1p(bl["baseline_demand"])

    Xn = scaler.transform(bl[NUM_FEATURES])
    return predict_demand(bl["route_idx"].values, Xn, bl["log_baseline"].values)

SCENARIOS = ["weekday_peak", "weekend", "rainy_day"]
scenario_demand = {s: get_scenario_demand(s) for s in SCENARIOS}

_chk = build_scenario_features("rainy_day").iloc[0]
assert _chk["is_rainy_day"] == 1 and _chk["is_extreme_heat"] == 0
print("✓ rainy_day: is_rainy_day=1, rain_mm=", _chk["rain_mm"])

scen_df = pd.DataFrame({s: scenario_demand[s] for s in SCENARIOS})
scen_df.insert(0, "hour", slot_hour)
scen_df.insert(0, "dir", slot_dir)
scen_df.insert(0, "route", slot_route)
hour_agg = scen_df.groupby("hour")[SCENARIOS].mean().round(0)
print("\nDemand dự báo TB theo giờ (passengers/hour, 3 kịch bản):")
print(hour_agg.head(8), "\n...", hour_agg.tail(3))
print("\nRainy / weekday_peak (TB):",
      round((scenario_demand["rainy_day"] / scenario_demand["weekday_peak"]).mean(), 3))
'''

CELL_21 = r'''def evaluate_schedule(
    trips: np.ndarray,
    demand: np.ndarray,
    lambda_cost: float | None = None,
) -> dict:
    """Tính chỉ số lịch trình; lambda_cost=None → dùng LAMBDA_COST global."""
    lam = LAMBDA_COST if lambda_cost is None else float(lambda_cost)
    trips = np.maximum(trips, 1)
    headway_min = 60.0 / trips
    avg_wait_min = headway_min / 2.0
    passenger_min_wait = demand * avg_wait_min
    total_wait = passenger_min_wait.sum()
    weighted_avg_wait = total_wait / max(demand.sum(), 1e-9)
    total_fleet_cost = lam * trips.sum()
    return dict(
        total_passenger_min_wait=float(total_wait),
        weighted_avg_wait_min=float(weighted_avg_wait),
        total_fleet_cost=float(total_fleet_cost),
        total_trips=float(trips.sum()),
        headway_std=float(headway_min.std()),
        objective=float(total_wait + total_fleet_cost),
        lambda_cost=lam,
    )

def objective(trips: np.ndarray, demand: np.ndarray, lambda_cost: float | None = None) -> float:
    return evaluate_schedule(trips, demand, lambda_cost=lambda_cost)["objective"]

def optimize_schedule_analytical(
    demand: np.ndarray,
    lambda_cost: float | None = None,
) -> np.ndarray:
    """Per-slot optimum: min D·30/t + λ·t  →  t* = sqrt(30·D/λ), clip bounds."""
    lam = LAMBDA_COST if lambda_cost is None else float(lambda_cost)
    trips = np.sqrt(30.0 * np.maximum(demand, 1e-9) / lam)
    trips = np.round(trips).astype(int)
    return np.clip(trips, TRIPS_MIN, TRIPS_MAX)

baseline_metrics = {s: evaluate_schedule(baseline_trips, scenario_demand[s])
                    for s in scenario_demand}
print("Baseline metrics (λ={}):".format(LAMBDA_COST))
for s, m in baseline_metrics.items():
    print(f"  {s:15s} | wait={m['total_passenger_min_wait']:>14,.0f} | "
          f"fleet={m['total_fleet_cost']:>12,.0f} | obj={m['objective']:,.0f} | trips={m['total_trips']:.0f}")
'''

# Insert new cell after 21 for optimization run (before GA markdown)
OPT_CELL = r'''opt_results = {}
import time

print(f"Optimizer: {'Analytical (per-slot)' if USE_ANALYTICAL_OPT else 'GA (see below)'} | N_SLOTS={N_SLOTS}")

for scen, dem in scenario_demand.items():
    t0 = time.time()
    if USE_ANALYTICAL_OPT:
        opt_results[scen] = optimize_schedule_analytical(dem)
    else:
        opt_results[scen], _ = genetic_algorithm(
            dem, generations=GA_GENERATIONS, pop_size=GA_POP_SIZE, seed=SEED
        )
    m = evaluate_schedule(opt_results[scen], dem)
    base_obj = baseline_metrics[scen]["objective"]
    impr = (base_obj - m["objective"]) / base_obj * 100
    label = "Analytical" if USE_ANALYTICAL_OPT else "GA"
    print(f"[{scen:14s}] {label} obj={m['objective']:>14,.0f} (impr {impr:+.2f}%) "
          f"trips={m['total_trips']:.0f} [{time.time()-t0:.2f}s]")

_sched = pd.DataFrame({
    "route": slot_route, "direction": slot_dir, "hour": slot_hour,
    "baseline_trips": baseline_trips,
    "opt_trips": opt_results["weekday_peak"],
})
_sched.to_csv(OUT_DIR / "schedule_weekday_peak.csv", index=False)
print("Đã lưu:", OUT_DIR / "schedule_weekday_peak.csv")
'''

CELL_29 = r'''OPT_METHOD = "Analytical" if USE_ANALYTICAL_OPT else "GA+Tabu"
BEST_LABEL = "Analytical" if USE_ANALYTICAL_OPT else "GA+Tabu"
best_results = opt_results if USE_ANALYTICAL_OPT else tabu_results

rows = []
for scen in scenario_demand:
    dem = scenario_demand[scen]
    base_wait = baseline_metrics[scen]["total_passenger_min_wait"]
    method_sols = [("Baseline", baseline_trips)]
    if USE_ANALYTICAL_OPT:
        method_sols.append(("Analytical", opt_results[scen]))
    else:
        method_sols.extend([("GA", ga_results[scen]), ("GA+Tabu", tabu_results[scen])])
    for method, sol in method_sols:
        m = evaluate_schedule(sol, dem)
        rows.append(dict(
            scenario=scen, method=method, n_slots=N_SLOTS,
            total_passenger_min_wait=m["total_passenger_min_wait"],
            total_fleet_cost=m["total_fleet_cost"],
            weighted_avg_wait_min=m["weighted_avg_wait_min"],
            headway_std=m["headway_std"],
            total_trips=m["total_trips"],
            objective=m["objective"],
            wait_saved_vs_baseline=base_wait - m["total_passenger_min_wait"],
            fleet_cost_delta_vs_baseline=m["total_fleet_cost"] - baseline_metrics[scen]["total_fleet_cost"],
        ))

results_df = pd.DataFrame(rows)
base_obj_per_scen = results_df[results_df["method"] == "Baseline"].set_index("scenario")["objective"]
base_wait_per_scen = results_df[results_df["method"] == "Baseline"].set_index("scenario")["total_passenger_min_wait"]

results_df["objective_improvement_pct"] = results_df.apply(
    lambda r: (base_obj_per_scen[r["scenario"]] - r["objective"]) / base_obj_per_scen[r["scenario"]] * 100,
    axis=1,
)
results_df["wait_improvement_pct"] = results_df.apply(
    lambda r: (base_wait_per_scen[r["scenario"]] - r["total_passenger_min_wait"])
              / base_wait_per_scen[r["scenario"]] * 100,
    axis=1,
)

print(f"=== Kết quả ({OPT_METHOD}, λ={LAMBDA_COST}, {N_SLOTS} slots) ===")
print(results_df.round(3).to_string(index=False))
results_df.to_csv(OUT_DIR / "results_summary.csv", index=False)

tradeoff_df = results_df[results_df["method"] == BEST_LABEL].copy()
tradeoff_show = tradeoff_df[[
    "scenario", "wait_saved_vs_baseline", "fleet_cost_delta_vs_baseline",
    "objective_improvement_pct", "total_trips",
]].rename(columns={
    "wait_saved_vs_baseline": "passenger_min_wait_saved",
    "fleet_cost_delta_vs_baseline": "fleet_cost_increase",
})
print(f"\n=== Trade-off {BEST_LABEL} vs Baseline ===")
print(tradeoff_show.round(0).to_string(index=False))
tradeoff_show.to_csv(OUT_DIR / "tradeoff_summary.csv", index=False)
print("\nĐã lưu:", OUT_DIR / "results_summary.csv", "|", OUT_DIR / "tradeoff_summary.csv")
'''

CELL_30 = r'''fig, axes = plt.subplots(1, 2, figsize=(14, 4.5))

best_label = "Analytical" if USE_ANALYTICAL_OPT else "GA+Tabu"
methods_impr = [best_label] if USE_ANALYTICAL_OPT else ["GA", "GA+Tabu"]
pivot = results_df.pivot(index="scenario", columns="method", values="wait_improvement_pct")
pivot = pivot[[c for c in methods_impr if c in pivot.columns]]
pivot.plot(kind="bar", ax=axes[0], color=["#2ECC71"] if USE_ANALYTICAL_OPT else ["#3498DB", "#2ECC71"])
axes[0].set_title("Cải thiện passenger-min chờ (%)")
axes[0].set_ylabel("Improvement (%)"); axes[0].legend(title="Method")

cols_wait = ["Baseline", best_label]
pivot2 = results_df.pivot(index="scenario", columns="method", values="weighted_avg_wait_min")[cols_wait]
pivot2.plot(kind="bar", ax=axes[1], color=["#95A5A6", "#2ECC71"])
axes[1].set_title("Thời gian chờ TB có trọng số (phút)")
axes[1].set_ylabel("Avg wait (min)"); axes[1].legend(title="Method")

plt.tight_layout()
plt.savefig(OUT_DIR / "fig_improvement.png", dpi=120, bbox_inches="tight")
plt.show()
'''

CELL_31 = """## 9b. Trade-off chờ ↔ fleet & phân tích độ nhạy λ

- **Trade-off**: giảm chờ bằng cách tăng chuyến (λ × trips).
- **Độ nhạy λ**: tái tối ưu `weekday_peak` với λ ∈ `LAMBDA_GRID` (analytical khi full scope)."""

CELL_32 = r'''fig, ax = plt.subplots(figsize=(7, 5))
best_label = "Analytical" if USE_ANALYTICAL_OPT else "GA+Tabu"
for scen in SCENARIOS:
    sub = results_df[(results_df["scenario"] == scen) & (results_df["method"].isin(["Baseline", best_label]))]
    for method in ["Baseline", best_label]:
        r = sub[sub["method"] == method].iloc[0]
        ax.scatter(r["total_fleet_cost"], r["total_passenger_min_wait"],
                   s=100, label=f"{scen} {method}")
ax.set_xlabel("Fleet cost (λ × trips)")
ax.set_ylabel("Passenger-min chờ")
ax.set_title(f"Trade-off (λ={LAMBDA_COST}, {N_SLOTS} slots)")
ax.legend(fontsize=7, ncol=2)
plt.tight_layout()
plt.savefig(OUT_DIR / "fig_tradeoff.png", dpi=120, bbox_inches="tight")
plt.show()

import time
lambda_rows = []
dem_peak = scenario_demand["weekday_peak"]
for lam in LAMBDA_GRID:
    t0 = time.time()
    sol_lam = optimize_schedule_analytical(dem_peak, lambda_cost=lam)
    m = evaluate_schedule(sol_lam, dem_peak, lambda_cost=lam)
    m_base = evaluate_schedule(baseline_trips, dem_peak, lambda_cost=lam)
    lambda_rows.append({
        "lambda": lam,
        "total_trips": m["total_trips"],
        "total_passenger_min_wait": m["total_passenger_min_wait"],
        "total_fleet_cost": m["total_fleet_cost"],
        "objective": m["objective"],
        "baseline_objective": m_base["objective"],
        "improvement_pct": (m_base["objective"] - m["objective"]) / m_base["objective"] * 100,
        "elapsed_s": time.time() - t0,
    })

lambda_sensitivity_df = pd.DataFrame(lambda_rows)
print("=== Độ nhạy λ (weekday_peak, analytical) ===")
print(lambda_sensitivity_df.round(2).to_string(index=False))
lambda_sensitivity_df.to_csv(OUT_DIR / "lambda_sensitivity.csv", index=False)

fig, axes = plt.subplots(1, 2, figsize=(11, 4))
axes[0].plot(lambda_sensitivity_df["lambda"], lambda_sensitivity_df["total_trips"], "o-", label="Optimal")
axes[0].axhline(baseline_trips.sum(), color="grey", ls="--", label="Baseline")
axes[0].set_xlabel("λ"); axes[0].set_ylabel("Total trips"); axes[0].legend()

axes[1].plot(lambda_sensitivity_df["lambda"], lambda_sensitivity_df["objective"], "o-", label="Opt obj")
axes[1].plot(lambda_sensitivity_df["lambda"], lambda_sensitivity_df["baseline_objective"], "s--", label="Baseline")
axes[1].set_xlabel("λ"); axes[1].set_ylabel("Objective"); axes[1].legend()
plt.tight_layout()
plt.savefig(OUT_DIR / "fig_lambda_sensitivity.png", dpi=120, bbox_inches="tight")
plt.show()
'''

CELL_33 = r'''def reshape_trip_matrix(sol: np.ndarray, direction: int, routes: list) -> pd.DataFrame:
    sub = pd.DataFrame({
        "route": slot_route, "dir": slot_dir, "hour": slot_hour, "trips": sol
    })
    sub = sub[sub["dir"] == direction]
    return sub.pivot(index="route", columns="hour", values="trips").reindex(routes)

HEATMAP_ROUTES = route_totals.head(8).index.tolist()
scen_to_show = "weekday_peak"
fig, axes = plt.subplots(1, 2, figsize=(14, 4))
for ax, (method, sol) in zip(axes, [
    ("Baseline GTFS", baseline_trips),
    ("Analytical" if USE_ANALYTICAL_OPT else "GA+Tabu", opt_results[scen_to_show]),
]):
    m = reshape_trip_matrix(sol, direction=0, routes=HEATMAP_ROUTES)
    sns.heatmap(m, annot=False, fmt=".0f", cmap="YlGnBu", ax=ax)
    ax.set_title(f"{method} — {scen_to_show} (dir 0, top-8 tuyến)")
    ax.set_xlabel("Hour"); ax.set_ylabel("Route")
plt.tight_layout()
plt.savefig(OUT_DIR / f"fig_schedule_{scen_to_show}.png", dpi=120, bbox_inches="tight")
plt.show()
'''

CELL_34 = r'''summary_metrics = {
    "n_slots": int(N_SLOTS),
    "n_routes": int(len(OPT_ROUTES)),
    "n_hours": int(len(OPT_HOURS)),
    "routes": list(OPT_ROUTES),
    "opt_method": "Analytical" if USE_ANALYTICAL_OPT else "GA+Tabu",
    "n_headway_fallback": int(n_fallback),
    "scenarios": list(scenario_demand.keys()),
    "lambda_cost_default": float(LAMBDA_COST),
    "baseline": {s: baseline_metrics[s] for s in scenario_demand},
    "optimized": {s: evaluate_schedule(opt_results[s], scenario_demand[s])
                  for s in scenario_demand},
    "nn_eval": nn_eval_metrics,
    "lambda_sensitivity_weekday_peak": lambda_sensitivity_df.to_dict(orient="records"),
}
with open(OUT_DIR / "baseline_metrics.json", "w", encoding="utf-8") as f:
    json.dump(summary_metrics, f, indent=2)
print("Đã ghi:", OUT_DIR / "baseline_metrics.json")
'''

CELL_35 = """## 10. Kết luận & Discussion

### Tổng kết

- **Phạm vi**: 29 tuyến × 2 hướng × 24 giờ (~1.392 slot); dữ liệu ridership tháng 2026-01 (31 ngày).
- **NN**: hold-out + 5-fold CV (`nn_eval_summary.csv`).
- **Tối ưu**: analytical per-slot (nhanh ở full mạng); GA+Tabu chỉ khi `USE_ANALYTICAL_OPT=False` và `N_SLOTS ≤ 200`.
- **Trade-off / λ**: `tradeoff_summary.csv`, `lambda_sensitivity.csv`.

### Hạn chế

- Mục tiêu tách từng slot (chưa ràng buộc fleet chung).
- Waiting time = headway/2.

### Hướng mở rộng

- RL (GTFS-RT), ràng buộc fleet/crew, transfer waiting."""

CELL_6_FIX = None  # cell 6 - remove local COVERAGE_THR if duplicated

# Apply cells
set_cell(2, CELL_2)
set_cell(3, CELL_3)
set_cell(4, CELL_4)
set_cell(5, CELL_5)
set_cell(8, CELL_8)
set_cell(9, CELL_9)
set_cell(10, CELL_10)
set_cell(11, CELL_11)
set_cell(19, CELL_19)
set_cell(20, CELL_20)
set_cell(21, CELL_21)

# Re-find indices after edits
def find_cell(substr):
    for i, c in enumerate(nb["cells"]):
        if substr in "".join(c.get("source", [])):
            return i
    return None

idx_29 = find_cell("OPT_METHOD = ")
idx_30 = find_cell("fig, axes = plt.subplots(1, 2, figsize=(14, 4.5))")
idx_32 = find_cell("lambda_sensitivity_df = pd.DataFrame(lambda_rows)")
idx_33 = find_cell("HEATMAP_ROUTES = ")
idx_34 = find_cell('"n_headway_fallback"')
idx_35 = find_cell("## 10. Kết luận")

if idx_29: set_cell(idx_29, CELL_29)
if idx_30: set_cell(idx_30, CELL_30)
if idx_32: set_cell(idx_32, CELL_32)
if idx_33: set_cell(idx_33, CELL_33)
if idx_34: set_cell(idx_34, CELL_34)
if idx_35: set_cell(idx_35, CELL_35)

# Update markdown cell 18 if exists - NN summary
idx_18 = find_cell("**Tổng kết Neural Network**")
if idx_18:
    set_cell(idx_18, """**Tổng kết Neural Network**:
- Ridership tháng đủ (31 ngày) × 29 tuyến × 24 giờ — ổn định hơn bộ 19 ngày cũ.
- Báo cáo hold-out và 5-fold CV (`nn_eval_summary.csv`).

## 5. Định nghĩa bài toán tối ưu hoá lịch trình

**Biến**: `trips[r,d,h]` cho mỗi slot trong `OPT_SCOPE` (~1.392).

**Mục tiêu**: min Σ (D·30/trips + λ·trips). **Analytical** khi `USE_ANALYTICAL_OPT=True`.

**Kịch bản**: weekday_peak, weekend, rainy_day (`is_rainy_day=1`).""")

# Update route_to_idx in cell 13 area
idx_13 = find_cell('route_to_idx = {r: i for i, r in enumerate(sorted(TOP_ROUTES))}')
if idx_13:
    src = "".join(nb["cells"][idx_13]["source"])
    src = src.replace("sorted(TOP_ROUTES)", "sorted(OPT_ROUTES)")
    src = src.replace("TOP_ROUTES", "OPT_ROUTES")
    set_cell(idx_13, src)

# Fix cell 12 markdown
idx_12 = find_cell("**Quan sát**")
if idx_12:
    set_cell(idx_12, """**Quan sát** (full 24h × top tuyến):
- Demand thay đổi rõ theo giờ; peak sáng/chiều vẫn nổi bật trên tuyến trục.
- Weekend và ngày mưa lệch so weekday — NN dùng cho kịch bản.""")

# Insert main optimization run after tabu_search is defined
if find_cell("opt_results = {}") is None:
    tabu_def_idx = find_cell("def tabu_search(")
    insert_at = (tabu_def_idx + 1) if tabu_def_idx is not None else find_cell("## 9. Đánh giá")
    if insert_at:
        nb["cells"].insert(insert_at, {
            "cell_type": "code",
            "metadata": {},
            "source": [OPT_CELL],
            "outputs": [],
            "execution_count": None,
            "id": "opt_analytical_cell",
        })

idx_ga = find_cell("ga_results = {}\nga_logs = {}")
if idx_ga is None:
    idx_ga = find_cell("ga_results = {}")
if idx_ga and "Bỏ qua GA" not in "".join(nb["cells"][idx_ga]["source"]):
    set_cell(idx_ga, '''# OPTIONAL — legacy GA (chỉ khi USE_ANALYTICAL_OPT=False và N_SLOTS nhỏ)
if not USE_ANALYTICAL_OPT and N_SLOTS <= GA_MAX_SLOTS:
    ga_results = {}
    ga_logs = {}
    import time
    for scen, dem in scenario_demand.items():
        t0 = time.time()
        sol, log = genetic_algorithm(dem, generations=GA_GENERATIONS, pop_size=GA_POP_SIZE, seed=SEED)
        ga_results[scen] = sol
        ga_logs[scen] = log
        m = evaluate_schedule(sol, dem)
        impr = (baseline_metrics[scen]["objective"] - m["objective"]) / baseline_metrics[scen]["objective"] * 100
        print(f"[{scen}] GA impr {impr:+.2f}% [{time.time()-t0:.1f}s]")
else:
    print("Bỏ qua GA — dùng analytical hoặc N_SLOTS > GA_MAX_SLOTS.")
''')

idx_tabu = find_cell("tabu_results = {}\ntabu_logs = {}")
if idx_tabu is None:
    idx_tabu = find_cell("tabu_results = {}")
if idx_tabu and "Bỏ qua Tabu" not in "".join(nb["cells"][idx_tabu]["source"]):
    set_cell(idx_tabu, '''# OPTIONAL — Tabu refine (sau GA)
if not USE_ANALYTICAL_OPT and N_SLOTS <= GA_MAX_SLOTS:
    tabu_results = {}
    tabu_logs = {}
    import time
    for scen, dem in scenario_demand.items():
        t0 = time.time()
        best, log = tabu_search(ga_results[scen], dem, iters=TABU_ITERS)
        tabu_results[scen] = best
        tabu_logs[scen] = log
        m = evaluate_schedule(best, dem)
        impr = (baseline_metrics[scen]["objective"] - m["objective"]) / baseline_metrics[scen]["objective"] * 100
        print(f"[{scen}] Tabu impr {impr:+.2f}% [{time.time()-t0:.1f}s]")
else:
    print("Bỏ qua Tabu.")
''')

NB_PATH.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
print("Patched", len(nb["cells"]), "cells")
