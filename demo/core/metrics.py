from __future__ import annotations

import pandas as pd


def build_metrics_table(
    metrics: dict,
    baseline_metrics: dict,
    min_fleet_required: float | None = None,
    baseline_min_fleet_required: float | None = None,
) -> pd.DataFrame:
    rows = [
        ("Tong thoi gian cho (pax-min)", metrics["total_passenger_min_wait"], baseline_metrics["total_passenger_min_wait"]),
        ("Avg wait / hanh khach (phut)", metrics["weighted_avg_wait_min"], baseline_metrics["weighted_avg_wait_min"]),
        ("Tong vehicle-hours", metrics["total_vehicle_hours"], baseline_metrics["total_vehicle_hours"]),
        ("Tong chi phi van hanh", metrics.get("fleet_cost", 0.0), baseline_metrics.get("fleet_cost", 0.0)),
        ("% Slots overflow", metrics["overflow_pct"], baseline_metrics["overflow_pct"]),
        ("Tong hanh khach tran", metrics["total_overflow_pax"], baseline_metrics["total_overflow_pax"]),
        ("Max queue length", metrics["max_queue_length"], baseline_metrics["max_queue_length"]),
        ("Tong so chuyen", metrics["total_trips"], baseline_metrics["total_trips"]),
    ]
    if min_fleet_required is not None and baseline_min_fleet_required is not None:
        rows.append(("So xe toi thieu can van hanh", min_fleet_required, baseline_min_fleet_required))
    data = []
    for name, opt_val, base_val in rows:
        delta = opt_val - base_val
        pct = (delta / base_val * 100.0) if base_val else 0.0
        data.append(
            {
                "Chi so": name,
                "Toi uu": round(float(opt_val), 3),
                "Baseline": round(float(base_val), 3),
                "Cai thien": f"{delta:+.2f} ({pct:+.1f}%)",
            }
        )
    return pd.DataFrame(data)
