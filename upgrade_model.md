Review and refactor the entire transit demand forecasting and schedule optimization system.

Goal:
Transform the project into a scientifically defensible, operationally realistic, computationally efficient Predictive + Prescriptive Transit Scheduling framework suitable for academic research, thesis work, or publication.

The refactored system should prioritize:

1. Forecast accuracy
2. Operational realism
3. Interpretability
4. Computational speed
5. Simplicity (remove unnecessary heuristics)

==================================================
TARGET ARCHITECTURE
==================================================

RIDERSHIP
+
WEATHER
+
GTFS
        ↓
Residual Demand Forecast
(MLP + HistGBM Blend)
        ↓
Quantile Scenario Generator
        ↓
Analytical Schedule Optimizer
        ↓
Operational Constraints
    ├─ Capacity
    ├─ Fleet
    ├─ Headway
    ├─ Service Window
    └─ Turnaround
        ↓
Queue-Based Passenger Wait Evaluation
        ↓
Pareto Frontier
        ↓
Knee Point Selection
        ↓
Schedule Recommendation

==================================================
1. REMOVE SPILLOVER LAYER
==================================================

Completely remove:

- build_transfer_matrix()
- adjust_demand_with_spillover()
- evaluate_spillover_forecast_mae()
- spillover_benefit_by_route()
- plot_transfer_heatmap()

Reason:

- Shared stations are not actual transfer demand.
- No AFC/OMNY transfer-flow data exists.
- May artificially create demand.
- Adds complexity without proven forecasting benefit.

Spillover analysis can remain only as Future Work documentation.

All downstream logic should use direct route-level demand forecasts only.

==================================================
2. KEEP DEMAND FORECASTING FRAMEWORK
==================================================

Keep:

- Residual learning
- MLP model
- HistGradientBoosting model
- Blend optimization
- Seasonal holdout validation
- Seasonal cross-validation
- Quantile GBM scenarios

Forecast target remains:

Residual =
log(Demand)
-
log(Baseline)

Demand =
Baseline × exp(Residual)

Do not change this formulation.

==================================================
3. REPLACE OVERFLOW MODEL WITH QUEUE MODEL
==================================================

Remove all heuristic overflow penalties such as:

OverflowWait = 1.5 × Headway

or any equivalent fixed multiplier.

Replace with queue propagation.

Definitions:

Capacity_t =
Trips_t × CapacityPerTrip

Queue_t =
max(
0,
Queue_(t−1)
+
Demand_t
− Capacity_t
)

Served_t =
min(
Demand_t + Queue_(t−1),
Capacity_t
)

Headway_t =
60 / Trips_t

PassengerWait_t =
Queue_(t−1) × Headway_t
+
Served_t × Headway_t / 2

TotalPassengerWait =
Σ PassengerWait_t

The queue should propagate between consecutive hours of the same route-direction.

No arbitrary waiting multipliers should remain.

==================================================
4. REPLACE TRIP COST WITH VEHICLE-HOUR COST
==================================================

Current:

Cost = λ × Trips

Remove this formulation.

Replace with:

VehicleHours_t =
Trips_t × CycleTime_t / 60

TotalVehicleHours =
Σ VehicleHours_t

Optional:

FleetCost =
VehicleHours × CostPerVehicleHour

Use VehicleHours as the operational cost metric throughout the system.

Reason:

Trips alone do not represent operating effort.
Cycle time is required.

==================================================
5. REMOVE WEIGHTED-SUM OBJECTIVE
==================================================

Current:

Objective =
PassengerWait
+
λ × Cost

Remove this objective.

The optimizer should no longer depend on λ-weighted aggregation.

Instead:

Evaluate every solution using two objectives:

f1 = TotalPassengerWait

f2 = TotalVehicleHours

Optimization output should feed directly into Pareto analysis.

==================================================
6. KEEP PARETO OPTIMIZATION
==================================================

Keep:

- generate_pareto_frontier()
- chebyshev_scalarize()
- filter_nondominated()
- find_knee_point()

Update frontier dimensions:

Old:

PassengerWait vs Trips

New:

PassengerWait vs VehicleHours

All Pareto charts, reports, and exports must use VehicleHours.

==================================================
7. REVISE DYNAMIC BOUNDS
==================================================

Current implementation allows greater flexibility during off-peak than peak periods.

This is operationally counterintuitive.

Update suggested defaults:

Peak:

peak_factor = 1.40

Off-Peak:

offpeak_factor = 1.15

Overnight:

overnight_factor = 1.05

Lower bounds:

min_factor = 0.50

Keep:

max_delta protection
absolute_max protection

Document rationale for every bound.

==================================================
8. KEEP CAPACITY CONSTRAINT
==================================================

Retain:

Trips_min =
ceil(
Demand / CapacityPerTrip
)

This is a hard operational constraint.

Do not remove.

==================================================
9. KEEP FLEET CONSTRAINT
==================================================

Retain:

MaxTrips =
floor(
FleetSize × 60 / CycleTime
)

Retain:

compute_route_direction_cycle_times()

compute_fleet_limits_from_baseline()

apply_route_fleet_cap()

apply_system_fleet_cap()

These constraints are operationally valid.

==================================================
10. KEEP SERVICE WINDOW CONSTRAINTS
==================================================

Retain:

apply_service_window_constraints()

based on:

first departure
last departure

derived from GTFS.

No modifications required.

==================================================
11. KEEP TURNAROUND CONSTRAINTS
==================================================

Retain:

cycle_time =
2 × one_way_runtime
+
2 × turnaround_buffer

Keep turnaround enforcement.

==================================================
12. KEEP HEADWAY CONSTRAINTS
==================================================

Retain:

build_headway_trip_bounds()

build_headway_trip_bounds_by_slot()

merge_trip_bounds()

These provide realistic service frequency limits.

==================================================
13. REMOVE UNNECESSARY COMPLEXITY
==================================================

For every module classify:

A = Scientifically justified
B = Useful heuristic
C = Unnecessary complexity

Remove all Category C components.

Particularly review:

- redundant helper functions
- duplicate constraint logic
- duplicate objective calculations
- unused spillover utilities
- obsolete visualization code

==================================================
14. UPDATE REPORTING
==================================================

All evaluation outputs should report:

Forecast:
- MAE
- RMSE
- R²

Operations:
- Passenger Wait
- Average Wait
- Overflow Passengers
- Queue Length
- Vehicle Hours
- Fleet Utilization

Optimization:
- Pareto Frontier
- Knee Solution
- Wait Reduction (%)
- Vehicle Hour Change (%)

==================================================
15. FINAL DELIVERABLE
==================================================

Produce:

1. Refactored architecture diagram
2. Updated mathematical formulation
3. List of removed components
4. List of modified formulas
5. Updated optimization workflow
6. Updated objective definitions
7. Expected benefits
8. Trade-offs
9. Clean production-ready code

The final system should be a lightweight, interpretable, analytical transit scheduling framework without spillover heuristics or weighted-sum objectives, using queue-based passenger waiting and vehicle-hour-based operating costs.