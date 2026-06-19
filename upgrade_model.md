Tôi có 2 file:
- `single_route_pipeline.py` (thư viện helper)
- `mta_schedule_optimization.ipynb` (notebook chính)

Hãy thêm 3 module MỚI vào pipeline, KHÔNG sửa bất kỳ logic nào đã có trong
§1-§10 của notebook (demand model, Pareto, constraints giữ nguyên 100%).
Cả 3 module dưới đây độc lập với nhau nhưng có thể dùng nối tiếp:
GTFS station schedule → fleet chính xác → simulate delay trên lịch đó.

============================================================
MODULE 1 — Lịch chi tiết đến từng ga (đã có sẵn, chỉ cần dùng lại + bổ sung)
============================================================

Hàm `expand_schedule_to_station_times()` đã tồn tại trong pipeline — KHÔNG viết lại,
chỉ cần đảm bảo nó là input chuẩn cho Module 2 và Module 3 bên dưới.

Thêm 1 hàm tiện ích mới để build input cho nó từ opt_trips của TOÀN BỘ optimizer
(hiện hàm cần `schedule` DataFrame với cột route/direction/hour/opt_trips):

```python
def build_station_schedule_for_route(
    opt_trips: np.ndarray,
    slot_route: np.ndarray,
    slot_dir: np.ndarray,
    slot_hour: np.ndarray,
    schedule_dir: Path,
    route_id: str,
    *,
    offset_templates: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Lấy slot của 1 route từ kết quả optimizer toàn mạng, build DataFrame
    (route, direction, hour, opt_trips) rồi gọi expand_schedule_to_station_times().
    Tái sử dụng offset_templates nếu được truyền vào (tránh đọc lại GTFS nhiều lần
    khi gọi cho nhiều route)."""
    mask = slot_route == route_id
    sched_df = pd.DataFrame({
        "route_id": slot_route[mask],
        "direction_id": slot_dir[mask],
        "hour": slot_hour[mask],
        "opt_trips": opt_trips[mask],
    })
    return expand_schedule_to_station_times(
        sched_df, schedule_dir,
        offset_templates=offset_templates, route_id=route_id,
    )
```

Đồng thời cần thêm cột `trip_id` vào output của `expand_schedule_to_station_times()`
nếu chưa có — số thứ tự trip trong ngày, theo (route, direction), rank theo
scheduled_min tại ga có stop_sequence nhỏ nhất:

```python
def assign_trip_ids(station_schedule: pd.DataFrame) -> pd.DataFrame:
    """Gán trip_id (0,1,2,...) cho mỗi trip trong station_schedule, theo thứ tự
    khởi hành, riêng từng (route, direction)."""
    out = station_schedule.copy()
    first_stop = (
        out.sort_values("stop_sequence")
        .groupby(["route", "direction", "hour"])
        .apply(lambda g: g[g["stop_sequence"] == g["stop_sequence"].min()])
        .reset_index(drop=True)
    )
    # Cách đơn giản hơn: dùng scheduled_min tại stop_sequence nhỏ nhất mỗi trip
    # để rank — implement theo cách rõ ràng, dễ test, không cần quá tối ưu performance.
    ...
    return out
```
(Bạn tự chọn cách implement gọn nhất miễn đảm bảo trip_id tăng dần đúng theo
thời gian khởi hành thực tế trong mỗi route×direction.)

============================================================
MODULE 2 — Fleet chính xác bằng sweep-line continuous (thay vì Little's Law theo giờ)
============================================================

Vấn đề: `compute_fleet_limits_from_baseline()` hiện tại tính
`fleet = trips[h] × cycle_time / 60` rồi lấy max theo giờ — đây là LOWER BOUND,
bỏ qua phân bố trip không đều trong giờ và chuyển tiếp giữa các giờ.

Thêm hàm mới (KHÔNG sửa hàm cũ, giữ để so sánh):

```python
def compute_fleet_continuous_sweep(
    trips: np.ndarray,
    slot_route: np.ndarray,
    slot_dir: np.ndarray,
    slot_hour: np.ndarray,
    cycle_times: pd.DataFrame,
) -> tuple[pd.DataFrame, float]:
    """Fleet chính xác bằng sweep-line trên timeline liên tục cả ngày.

    Với mỗi (route, direction):
    1. Sinh departure_min cụ thể cho từng trip trong mỗi giờ:
       hour_start = hour * 60; spacing = 60 / n_trips_trong_gio
       departure_min[i] = hour_start + (i + 0.5) * spacing  (cùng convention
       với expand_schedule_to_station_times để nhất quán)
    2. Mỗi trip chiếm 1 tàu trong [departure_min, departure_min + cycle_time_min].
    3. Sweep-line: tạo events (departure_min, +1) và
       (departure_min + cycle_time_min, -1), sort theo thời gian
       (tie-break: xử lý -1 trước +1 nếu cùng thời điểm để không đếm thừa),
       quét và track running count → fleet_route_dir = max running count.
    4. Xử lý wrap-around qua nửa đêm: nếu cycle_time khiến departure +
       cycle_time vượt quá 1440 phút, coi như tiếp tục sang ngày hôm sau
       (giả định lịch lặp lại hàng ngày) — cộng dồn vào đầu ngày khi sweep.

    Trả về (per_route_dir_df, system_max) cùng format với
    compute_fleet_limits_from_baseline() để dễ so sánh, với thêm cột
    `fleet_size_lower_bound` (giá trị từ công thức Little's Law cũ) và
    `fleet_size_continuous` (giá trị mới), và `pct_underestimate` =
    (continuous - lower_bound) / continuous * 100.
    """
```

Thêm hàm so sánh:

```python
def compare_fleet_estimates(
    baseline_trips: np.ndarray,
    slot_route: np.ndarray,
    slot_dir: np.ndarray,
    slot_hour: np.ndarray,
    cycle_times: pd.DataFrame,
) -> pd.DataFrame:
    """Chạy cả compute_fleet_limits_from_baseline() và
    compute_fleet_continuous_sweep() trên cùng baseline_trips, trả về bảng
    so sánh per (route, direction): fleet_lower_bound, fleet_continuous,
    pct_underestimate, sorted theo pct_underestimate giảm dần
    (route nào bị underestimate nhiều nhất lên đầu)."""
```

Trong notebook §6 (chỗ đang gọi `compute_fleet_limits_from_baseline`),
thêm 1 cell mới ngay sau đó (KHÔNG thay thế cell cũ):

```python
# So sánh fleet lower-bound (Little's Law theo giờ) vs continuous (sweep-line)
fleet_comparison = srp.compare_fleet_estimates(
    baseline_trips, slot_route, slot_dir, slot_hour, cycle_times
)
print("Top 10 route×dir bị underestimate nhiều nhất (lower bound vs continuous):")
print(fleet_comparison.head(10).round(2).to_string(index=False))
fleet_comparison.to_csv(OUT_DIR / "fleet_estimate_comparison.csv", index=False)
print(f"\nTrung bình underestimate: {fleet_comparison['pct_underestimate'].mean():.1f}%")
```

`apply_route_fleet_cap()` hiện có vẫn dùng `fleet_by_route_dir` (lower bound) làm
ràng buộc — KHÔNG đổi, vì lower bound là lựa chọn AN TOÀN hơn cho constraint
(tránh overestimate khả năng đội tàu thật có). Module này chỉ để BIẾT mức độ
sai lệch, dùng cho báo cáo, không đổi hành vi optimizer.

============================================================
MODULE 3 — Simulate delay thủ công trên lịch chi tiết ga (what-if, không real-time)
============================================================

```python
def simulate_delay_propagation(
    station_schedule: pd.DataFrame,
    *,
    route_id: str,
    direction_id: int,
    delayed_trip_id: int,
    delay_min: float,
    delay_at_stop_sequence: int,
    min_headway_ratio: float = 0.6,
    max_propagation_trips: int = 3,
    dwell_recovery_min: float = 0.5,
) -> pd.DataFrame:
    """Mô phỏng 1 trip trễ delay_min phút tại 1 ga (delay_at_stop_sequence),
    lan truyền TỐI THIỂU sang tối đa max_propagation_trips trip liền sau cùng
    (route_id, direction_id). Input station_schedule cần có cột trip_id
    (dùng assign_trip_ids() ở Module 1 nếu chưa có).

    Logic:
    1. Lọc trip cùng (route_id, direction_id), sort theo trip_id.
    2. Trip delayed_trip_id: actual_min = scheduled_min + delay_min cho ga
       delay_at_stop_sequence và mọi ga SAU đó trên cùng trip, delay giảm dần
       max(0, delay_min - dwell_recovery_min * (stop_sequence - delay_at_stop_sequence)).
    3. Với mỗi trip kế tiếp (trip_id+1, +2, ..., tối đa max_propagation_trips):
       a. Tại mỗi ga, headway_planned = scheduled_min[trip_sau] - scheduled_min[trip_truoc]
       b. headway_neu_khong_chinh = scheduled_min[trip_sau] - actual_min[trip_truoc, ga đó]
       c. Nếu headway_neu_khong_chinh < min_headway_ratio * headway_planned:
          hold_min = min_headway_ratio * headway_planned - headway_neu_khong_chinh
          actual_min[trip_sau] = scheduled_min[trip_sau] + hold_min, áp dụng từ
          ga đó trở đi (decay dần qua dwell_recovery_min như bước 2)
       d. Nếu KHÔNG vi phạm ngưỡng ở MỌI ga của trip_sau: actual_min = scheduled_min
          (không đổi) VÀ DỪNG vòng lặp — không xét trip xa hơn (nguyên tắc lan
          truyền tối thiểu: delay đã được hấp thụ hết).
    4. Trả về bản copy station_schedule + cột: actual_min, actual_time (HH:MM),
       delay_applied_min, is_affected (bool), propagation_hop (0=trip gốc,
       1,2,...=số trip lan truyền, NaN nếu không ảnh hưởng).
    """


def summarize_delay_impact(adjusted_schedule: pd.DataFrame) -> dict[str, Any]:
    """Trả về dict: n_trips_affected, n_stops_affected, max_hold_min,
    total_hold_min, affected_trip_ids (list), worst_stop
    (dict: parent_stop_id, stop_name, max_delay_min tại đó)."""


def plot_delay_propagation(
    adjusted_schedule: pd.DataFrame,
    *,
    route_id: str,
    direction_id: int,
    ax: Any | None = None,
    title: str = "Delay propagation",
    save_path: Path | str | None = None,
) -> Any:
    """Time-distance diagram: trục x = thời gian (phút), trục y = thứ tự ga
    (stop_sequence). Mỗi trip = 1 đường nối các điểm (actual_min, stop_sequence).
    Trip gốc bị trễ: đỏ. Trip propagation_hop >= 1: cam, độ đậm giảm dần theo hop.
    Trip không ảnh hưởng: xám nhạt, mỏng. Dùng matplotlib line plot, không cần
    thư viện ngoài."""
```

============================================================
TÍCH HỢP 3 MODULE — Thêm section mới cuối notebook
============================================================

Thêm "## 11. Lịch chi tiết ga, fleet chính xác & mô phỏng trễ (demo)"
— section HOÀN TOÀN MỚI, không đụng §1-§10:

```python
# --- 11a. Build lịch chi tiết đến ga cho 1 route demo ---
demo_route = OPT_ROUTES[0]
offset_templates = srp.load_gtfs_stop_offset_templates(SCHEDULE_DIR, service_id=HEADWAY_SERVICE)

demo_station_schedule = srp.build_station_schedule_for_route(
    opt_trips, slot_route, slot_dir, slot_hour,
    SCHEDULE_DIR, demo_route, offset_templates=offset_templates,
)
demo_station_schedule = srp.assign_trip_ids(demo_station_schedule)
print(f"Lịch chi tiết route {demo_route}: {len(demo_station_schedule)} dòng "
      f"(trip × ga), {demo_station_schedule['trip_id'].nunique()} trips")
demo_station_schedule.to_csv(OUT_DIR / f"station_schedule_{demo_route}.csv", index=False)

# --- 11b. Fleet chính xác cho toàn mạng ---
fleet_comparison = srp.compare_fleet_estimates(
    baseline_trips, slot_route, slot_dir, slot_hour, cycle_times
)
fleet_comparison.to_csv(OUT_DIR / "fleet_estimate_comparison.csv", index=False)
print(f"\nFleet — trung bình underestimate của lower bound: "
      f"{fleet_comparison['pct_underestimate'].mean():.1f}%")
print(fleet_comparison.sort_values('pct_underestimate', ascending=False).head(5)
      .round(2).to_string(index=False))

# --- 11c. Mô phỏng trễ (what-if demo) trên lịch route đã build ở 11a ---
adjusted = srp.simulate_delay_propagation(
    demo_station_schedule,
    route_id=demo_route,
    direction_id=0,
    delayed_trip_id=5,
    delay_min=5.0,
    delay_at_stop_sequence=3,
    min_headway_ratio=0.6,
    max_propagation_trips=3,
    dwell_recovery_min=0.5,
)
impact = srp.summarize_delay_impact(adjusted)
print(f"\n[Demo] Trip 5 trễ 5 phút tại stop_sequence=3:")
print(f"  Trip bị ảnh hưởng: {impact['n_trips_affected']} | "
      f"Tổng hold: {impact['total_hold_min']:.1f} phút | "
      f"Max hold 1 trip: {impact['max_hold_min']:.1f} phút")

srp.plot_delay_propagation(
    adjusted, route_id=demo_route, direction_id=0,
    save_path=OUT_DIR / f"fig_delay_propagation_{demo_route}.png",
)
adjusted.to_csv(OUT_DIR / f"delay_simulation_{demo_route}.csv", index=False)
print("Đã lưu §11:", OUT_DIR / f"station_schedule_{demo_route}.csv", "|",
      OUT_DIR / "fleet_estimate_comparison.csv", "|",
      OUT_DIR / f"delay_simulation_{demo_route}.csv")
```

============================================================
RÀNG BUỘC CHUNG — áp dụng cho cả 3 module
============================================================

- KHÔNG sửa bất kỳ hàm/cell nào trong §1-§10 hiện có (demand model, Pareto,
  apply_optimizer_constraints, spillover...).
- `apply_route_fleet_cap()` vẫn dùng fleet lower-bound cũ (an toàn hơn cho
  ràng buộc cứng) — Module 2 chỉ bổ sung thông tin so sánh, KHÔNG thay đổi
  hành vi optimizer.
- Module 3 không dùng GTFS-Realtime, không dùng dữ liệu thời gian thực —
  input delay là tham số demo do người dùng tự chọn.
- Nguyên tắc "lan truyền tối thiểu" trong Module 3: dừng ngay khi 1 trip
  phía sau không còn vi phạm ngưỡng headway an toàn — đây là điều kiện
  dừng quan trọng nhất, cần kiểm tra kỹ logic break trong vòng lặp.
- Mọi hàm mới đặt cuối file `single_route_pipeline.py`, có docstring tiếng
  Việt theo đúng phong cách các hàm hiện có trong file.
- Sau khi viết xong, chạy thử với route đầu tiên trong OPT_ROUTES để xác
  nhận không có lỗi runtime trước khi coi là hoàn thành.