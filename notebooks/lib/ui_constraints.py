"""Optional optimizer constraint overrides — validate + apply + binding stats."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from . import single_route_pipeline as srp


@dataclass
class ConstraintOverrides:
    use_route_fleet: bool = True
    use_system_fleet: bool = False
    use_capacity: bool = True
    use_smoothness: bool = True
    max_system_fleet: float | None = None
    capacity_per_trip: float | None = None
    smoothness_max_delta: int | None = None
    min_headway_min: float | None = None
    max_headway_min: float | None = None
    trips_peak_max_factor: float | None = None
    trips_offpeak_max_factor: float | None = None
    trips_overnight_max_factor: float | None = None
    trips_min_factor: float | None = None
    trips_overnight_min_factor: float | None = None
    lambda_cost: float | None = None
    opt_target: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {k: getattr(self, k) for k in self.__dataclass_fields__}


def default_constraint_panel(ui_config: dict[str, Any]) -> dict[str, Any]:
    """Defaults for UI — all values from ui_config.json."""
    return {
        "use_route_fleet": True,
        "use_system_fleet": False,
        "use_capacity": True,
        "use_smoothness": True,
        "max_system_fleet": float(ui_config["max_system_fleet"]),
        "capacity_per_trip": float(ui_config.get("capacity_per_trip", 1200)),
        "smoothness_max_delta": int(ui_config["smoothness_max_delta"]),
        "min_headway_min": float(ui_config["min_headway_min"]),
        "max_headway_min": float(ui_config["max_headway_min"]),
        "trips_peak_max_factor": float(ui_config["trips_peak_max_factor"]),
        "trips_offpeak_max_factor": float(ui_config["trips_offpeak_max_factor"]),
        "trips_overnight_max_factor": float(ui_config["trips_overnight_max_factor"]),
        "trips_min_factor": float(ui_config["trips_min_factor"]),
        "trips_overnight_min_factor": float(ui_config.get("trips_overnight_min_factor", ui_config["trips_min_factor"])),
        "lambda_cost": None,
        "opt_target": str(ui_config.get("opt_target", "objective")),
    }


def merge_overrides(ui_config: dict[str, Any], panel: dict[str, Any]) -> ConstraintOverrides:
    base = default_constraint_panel(ui_config)
    merged = {**base, **{k: v for k, v in panel.items() if v is not None}}
    return ConstraintOverrides(**{k: merged[k] for k in base})


class ValidationError(Exception):
    def __init__(self, message: str, hint: str = "") -> None:
        self.message = message
        self.hint = hint
        super().__init__(str(self))

    def __str__(self) -> str:
        if self.hint:
            return f"{self.message} — {self.hint}"
        return self.message


def _min_system_fleet_required(fleet_by_route_dir: pd.DataFrame) -> float:
    if fleet_by_route_dir is None or fleet_by_route_dir.empty:
        return 0.0
    return float(pd.to_numeric(fleet_by_route_dir["fleet_size"], errors="coerce").sum())


def validate_constraint_config(
    overrides: ConstraintOverrides,
    optimizer_state: dict[str, Any],
    *,
    route_id: str | None = None,
) -> None:
    """Server-side validation before running optimizer."""
    if overrides.capacity_per_trip is not None and overrides.capacity_per_trip <= 0:
        raise ValidationError(
            "Sức chứa mỗi chuyến phải > 0",
            "Tăng số chỗ ngồi mỗi chuyến lên ít nhất 1.",
        )

    min_hw = overrides.min_headway_min
    max_hw = overrides.max_headway_min
    if min_hw is not None and max_hw is not None and min_hw >= max_hw:
        raise ValidationError(
            f"Headway tối thiểu ({min_hw} phút) phải nhỏ hơn headway tối đa ({max_hw} phút)",
            "Giảm headway tối thiểu hoặc tăng headway tối đa.",
        )

    slot_route = np.asarray(optimizer_state["slot_route"])
    slot_hour = np.asarray(optimizer_state["slot_hour"])
    baseline = np.asarray(optimizer_state["baseline_trips"], dtype=float)

    if route_id is not None:
        mask = slot_route == str(route_id)
        baseline = baseline[mask]
        slot_hour = slot_hour[mask]

    tmin, tmax = recompute_trip_bounds(
        baseline,
        slot_hour,
        overrides,
        optimizer_state,
    )
    invalid = tmin > tmax
    if invalid.any():
        n_bad = int(invalid.sum())
        raise ValidationError(
            f"{n_bad} slot có trips_min > trips_max sau khi áp factor mới",
            "Giảm trips_min_factor hoặc tăng trips_*_max_factor / max_headway_min.",
        )

    if overrides.use_system_fleet and overrides.max_system_fleet is not None:
        fleet_df = optimizer_state.get("fleet_by_route_dir")
        min_fleet = _min_system_fleet_required(fleet_df)
        if overrides.max_system_fleet < min_fleet * 0.5:
            raise ValidationError(
                f"Giới hạn xe toàn mạng ({overrides.max_system_fleet:.0f}) quá thấp so với "
                f"fleet baseline tối thiểu (~{min_fleet:.0f})",
                "Tăng số xe tối đa hoặc tắt giới hạn toàn mạng.",
            )


def recompute_trip_bounds(
    baseline_trips: np.ndarray,
    slot_hour: np.ndarray,
    overrides: ConstraintOverrides,
    optimizer_state: dict[str, Any],
    ui_config: dict[str, Any] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Recompute TRIPS_MIN/MAX for slots using panel overrides."""
    cfg = ui_config or {}
    peak = overrides.trips_peak_max_factor or float(cfg.get("trips_peak_max_factor", 1.22))
    off = overrides.trips_offpeak_max_factor or float(cfg.get("trips_offpeak_max_factor", 1.35))
    ovn = overrides.trips_overnight_max_factor or float(cfg.get("trips_overnight_max_factor", 1.14))
    min_f = overrides.trips_min_factor or float(cfg.get("trips_min_factor", 0.5))
    ovn_min = overrides.trips_overnight_min_factor or float(cfg.get("trips_overnight_min_factor", min_f))
    min_hw = overrides.min_headway_min or float(cfg.get("min_headway_min", 3.0))
    max_hw = overrides.max_headway_min or float(cfg.get("max_headway_min", 20.0))
    ovn_hw = cfg.get("overnight_max_headway_min")
    max_delta = int(cfg.get("trips_max_delta", 3))

    return srp.compute_merged_trip_bounds(
        baseline_trips,
        slot_hour,
        peak_factor=peak,
        offpeak_factor=off,
        overnight_factor=ovn,
        min_factor=min_f,
        overnight_min_factor=ovn_min,
        max_delta=max_delta,
        min_headway_min=min_hw,
        max_headway_min=max_hw,
        overnight_max_headway_min=float(ovn_hw) if ovn_hw is not None else None,
    )


def apply_post_opt_constraints(
    opt_trips: np.ndarray,
    slot_demand: np.ndarray,
    slots: dict[str, np.ndarray],
    overrides: ConstraintOverrides,
    optimizer_state: dict[str, Any],
    *,
    ui_config: dict[str, Any],
) -> np.ndarray:
    """Apply apply_optimizer_constraints after analytical step."""
    max_fleet = overrides.max_system_fleet
    if max_fleet is None:
        max_fleet = float(optimizer_state.get("max_system_fleet", ui_config["max_system_fleet"]))

    cap = overrides.capacity_per_trip or float(ui_config.get("capacity_per_trip", 1200))
    smooth = overrides.smoothness_max_delta
    if smooth is None:
        smooth = int(ui_config["smoothness_max_delta"])

    tmin, tmax = recompute_trip_bounds(
        slots["baseline_trips"],
        slots["slot_hour"],
        overrides,
        optimizer_state,
        ui_config,
    )

    return srp.apply_optimizer_constraints(
        opt_trips,
        slot_demand,
        slot_route=slots.get("slot_route", np.asarray(optimizer_state["slot_route"])),
        slot_dir=slots["slot_dir"],
        slot_hour=slots["slot_hour"],
        cycle_times=optimizer_state.get("cycle_times"),
        fleet_by_route_dir=optimizer_state.get("fleet_by_route_dir"),
        max_system_fleet=max_fleet,
        capacity_per_trip=cap,
        max_overflow_pct=ui_config.get("max_overflow_pct"),
        smoothness_max_delta=smooth,
        trips_min=tmin,
        trips_max=tmax,
        use_route_fleet=overrides.use_route_fleet,
        use_system_fleet=overrides.use_system_fleet,
        use_capacity=overrides.use_capacity,
        use_smoothness=overrides.use_smoothness,
    )


def compute_binding_stats(
    trips: np.ndarray,
    trips_min: np.ndarray,
    trips_max: np.ndarray,
    slot_hour: np.ndarray,
) -> dict[str, Any]:
    """Binding fractions similar to notebook §6b ablation."""
    status = srp.report_bound_status(
        trips,
        trips_min,
        trips_max,
        slot_hour,
        verbose=False,
    )
    by_group = status["by_hour_group"]
    return {
        "at_min_pct": status["at_min_pct"],
        "at_max_pct": status["at_max_pct"],
        "interior_pct": status["interior_pct"],
        "by_hour_group": by_group.to_dict(orient="records") if not by_group.empty else [],
    }
