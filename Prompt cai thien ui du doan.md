# Prompt: Cải thiện UI dự đoán & lập lịch MTA

Dán prompt này vào agent code (Claude Code / Cursor / v.v.) đang có quyền truy cập repo UI/API thật của bạn. Prompt tự chứa đủ bối cảnh kỹ thuật từ pipeline hiện có (`single_route_pipeline.py` + notebook `all-route-mta.ipynb`) để agent không cần đoán field name.

## Bối cảnh dự án (đã có sẵn — KHÔNG re-train)

Notebook huấn luyện export toàn bộ artifact phục vụ UI/API vào `outputs/default/ui_export/`, gồm:

- `demand_model.keras`, `scaler.pkl`, `histgbm_blend.pkl`, `demand_inference.pkl` — pipeline dự báo demand (`Demand = exp(log_baseline + pred_residual) - 1`).
- `ui_config.json` — toàn bộ tham số cấu hình optimizer hiện tại: `lambda_opt`, `lambda_cost_eval`, `lambda_candidates`, `opt_target`, `trips_peak_max_factor`, `trips_offpeak_max_factor`, `trips_overnight_max_factor`, `trips_min_factor`, `trips_overnight_min_factor`, `min_headway_min`, `max_headway_min`, `turnaround_buffer_min`, `max_system_fleet`, `max_overflow_pct`, `smoothness_max_delta`, `capacity_per_trip`, `risk_scenarios`, `risk_demand_quantile`, `rainy_peak_trips_min_factor`, `rainy_hour_boost`, `num_features`, `scenarios`, v.v.
- `route_meta.json`, `optimizer_state.pkl` (gồm `service_windows`, `cycle_times`, `fleet_by_route_dir`, `TRIPS_MIN`/`TRIPS_MAX`, `transfer_matrix`...), `schedule_{scenario}.json`, `pareto_frontier.json`, `manifest.json`.
- Các hàm ràng buộc optimizer đã có trong `single_route_pipeline.py`: `apply_service_window_constraints`, `apply_route_fleet_cap`, `apply_system_fleet_cap`, `apply_capacity_floor`, `apply_smoothness_constraint`, gộp lại qua `apply_optimizer_constraints(..., use_route_fleet, use_system_fleet, use_capacity, use_smoothness)`.
- Feature thời tiết/lịch dùng để train (`HOURLY_FACTOR_COLS`): `temperature_c`, `apparent_temperature_c`, `precipitation_mm`, `rain_mm`, `snowfall_cm`, `windspeed_kmh`, `windgusts_kmh`, `is_rain`, `is_snow`, `is_severe_wind`, `is_major_event_window` — cộng `day_of_week`, `is_weekend`, `is_us_holiday`, `month`. Danh sách cột chính xác model cần (đúng thứ tự) nằm trong `demand_inference.pkl["num_features"]`; giá trị impute khi thiếu nằm trong `feature_medians.json`.
- GTFS gốc (`SCHEDULE_DIR`): `stop_times.txt`, `trips.txt`, `routes.txt`, có thể có `stops.txt`. Đã dùng để tính `cycle_times` (Little's law) và `service_windows` (giờ chạy đầu/cuối), nhưng **chưa** map ra giờ đến từng ga cụ thể.
- Map ga → tuyến: `routes_by_station_complex.csv` (cột `station_complex_id`, `gtfs_route_ids`/`route_ids_gtfs`) — **cần kiểm tra xem file này (hoặc `stops.txt`) có sẵn cột quận/borough chưa**; nếu không có, phải tự suy ra (xem mục 4).

## Mục tiêu: 4 cải tiến cho UI dự đoán/lập lịch hiện tại

### 1. Optional constraint controls (không bắt buộc, phải khả thi)

Thêm panel "Ràng buộc nâng cao (tuỳ chọn)" — mặc định thu gọn/tắt, không bắt người dùng phải hiểu optimizer mới dùng được:

- Toggle bật/tắt từng lớp ràng buộc, map trực tiếp 1-1 với `apply_optimizer_constraints`: `use_route_fleet`, `use_system_fleet`, `use_capacity`, `use_smoothness`.
- Input/slider cho: `max_system_fleet`, `capacity_per_trip`, `smoothness_max_delta`, `min_headway_min`/`max_headway_min`, các factor peak/offpeak/overnight, `lambda_cost`/chọn `opt_target`.
- Toàn bộ giá trị mặc định lấy từ `ui_config.json` (không hard-code lại trong frontend) để nút "Reset về mặc định" luôn đúng.
- **Bắt buộc validate phía server trước khi chạy optimizer**, không chỉ validate UI: `trips_min ≤ trips_max` sau khi áp factor mới, `max_system_fleet` không thấp hơn fleet tối thiểu cần để phục vụ baseline (so với `fleet_by_route_dir`/peak concurrent), `min_headway_min < max_headway_min`, `capacity_per_trip > 0`. Nếu cấu hình người dùng đưa vào làm bài toán vô nghĩa (vd. trips_max < trips_min cho mọi slot), trả lỗi rõ ràng kèm gợi ý sửa, **không** để optimizer chạy ra kết quả rác hoặc crash.
- Ghi log/trả về trong response phần "constraint nào đang bind" (% slot chạm min/max) tương tự bảng ablation đã có trong notebook (§6b), để người dùng hiểu vì sao kết quả không đổi khi họ chỉnh.

### 2. "Crawl factor theo ngày" phải tuân theo đúng feature schema lúc train

Khi UI lấy factor thời tiết/lịch cho một ngày cụ thể (gọi API thời tiết ngoài, hoặc đọc `factors_daily.csv`/`factors_hourly.csv` đã build sẵn):

- Map kết quả crawl đúng 1-1 vào các cột đã dùng lúc train (danh sách ở mục Bối cảnh). Không tự thêm cột mới, không đổi tên cột.
- Áp cùng pipeline tiền xử lý lúc train: thiếu trường nào thì impute bằng `feature_medians.json`; thứ tự & tập feature đưa vào `scaler.transform()` phải khớp `demand_inference.pkl["num_features"]`.
- Nếu giá trị crawl được nằm ngoài range train (vd. nhiệt độ/lượng mưa cực đoan chưa từng thấy trong `factors_hourly.csv`), clip về range hợp lý suy từ percentile của data train (đừng clip cứng theo số tự chọn) và hiển thị một dòng note nhỏ trong UI kiểu "giá trị đã được giới hạn theo phạm vi dữ liệu huấn luyện" — không âm thầm clip.
- Test case bắt buộc: 1 ngày có đủ factor, 1 ngày thiếu vài trường (kiểm tra fallback), 1 ngày giá trị input cực đoan (kiểm tra clip + cảnh báo).

### 3. Gom feature thời tiết thành "nhóm thời tiết" (weather group preset) thay vì chỉnh từng số

Thay UI hiện tại bắt người dùng chỉnh từng nhiệt độ/lượng mưa/gió rời rạc bằng các nhóm kịch bản, ví dụ:

- `sunny` (Nắng đẹp), `light_rain` (Mưa nhẹ), `heavy_rain` (Mưa lớn), `severe_storm` (Bão/gió mạnh), `snow` (Tuyết), `heat_wave` (Nóng cực đoan), `cold_snap` (Lạnh sâu).
- Giá trị đại diện mỗi nhóm **phải tính từ phân phối thật của data train** (`factors_hourly.csv`/`factors_daily.csv`), ví dụ: nhóm `heavy_rain` = median các feature trong các giờ có `is_rain=1` và `rain_mm` thuộc top quartile; không hard-code số tuỳ ý — để nhóm luôn "thích hợp với feature train" như yêu cầu.
- Build bảng mapping này một lần trong bước export (§11 của notebook) thành artifact mới, ví dụ `weather_groups.json`, để UI/API chỉ cần đọc, không phải tính lại từ raw data mỗi lần gọi.
- Giữ một mục "Tuỳ chỉnh thủ công (nâng cao)" thu gọn, cho phép chỉnh từng số nếu người dùng vẫn muốn — mặc định điền sẵn giá trị của nhóm đang chọn, không phải để trống.
- Khi người dùng chọn nhóm + ngày cụ thể, factor lịch (day_of_week, is_weekend, is_us_holiday, month) vẫn lấy theo ngày thật đã chọn; chỉ phần thời tiết được override theo nhóm.

### 4. In lịch chạy đến từng ga (từ GTFS) — UI dạng bảng chia theo quận

- Từ kết quả optimizer (`schedule_{scenario}.json`, đơn vị route×direction×hour), kết hợp `stop_times.txt`/`trips.txt` (đã parse trong `compute_route_direction_cycle_times`/`build_route_direction_departure_windows`) để suy ra giờ đến dự kiến tại **từng ga** dọc theo route/direction: chia đều số chuyến tối ưu trong giờ đó theo offset thời gian giữa các ga lấy từ `stop_sequence`/`stop_times` gốc (không bịa thời gian chạy giữa ga).
- Viết thêm 1 hàm dùng lại logic parse GTFS đã có trong `single_route_pipeline.py` (đừng parse `stop_times.txt` lại từ đầu ở tầng UI) — ví dụ thêm `expand_schedule_to_station_times()` vào cùng module.
- Expose qua endpoint mới, vd `GET /schedule/by-station?scenario=...&route=...`, trả về list `{station_complex_id, station_name, borough, route, direction, scheduled_time}`.
- UI hiển thị bảng, **chia theo quận/borough** (NYC: Manhattan, Brooklyn, Queens, Bronx, Staten Island) — mỗi quận một tab/section, bảng con sort theo giờ, có filter theo route.
- **Cần xác minh trước khi code**: `routes_by_station_complex.csv` hiện tại có cột borough/quận không. Nếu không có, phải tự suy borough — kiểm tra xem file ridership gốc (`ridership.csv`) hay metadata MTA có cột này không trước khi tự map theo tên ga/tọa độ; đừng giả định cột "borough" tồn tại sẵn.

## Tiêu chí hoàn thành

- Không phá vỡ contract API hiện có (các endpoint cũ vẫn hoạt động như trước).
- Không re-train model; mọi tính năng mới chỉ đọc artifact đã export hoặc bổ sung 1 bước export nhẹ ở §11.
- Có test cho: validate ràng buộc vô nghĩa bị chặn đúng, crawl factor thiếu/lệch range được xử lý đúng, mapping weather group khớp số liệu train, bảng lịch theo ga/quận trả đúng số ga và đúng thứ tự thời gian.