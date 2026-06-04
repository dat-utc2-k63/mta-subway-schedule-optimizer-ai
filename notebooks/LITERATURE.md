# Tài liệu tham khảo — MTA NYC ridership & lịch trình

Các bài dưới đây là cơ sở cho thiết kế feature và mô hình trong `lib/single_route_pipeline.py` / notebook single-route.

| # | Nguồn | Ý chính áp dụng vào pipeline |
|---|--------|------------------------------|
| 1 | Short-Term Passenger Flow Prediction for Urban Rail Transit Based on New York Subway Data (ITM Web of Conferences, 2026) | MTA hourly ridership; LSTM > ARIMA; peak weekday/weekend; lag + deep sequence |
| 2 | Mixed-Effects Modeling of NYC Subway Ridership Using MTA and Weather Data (arXiv:2505.02990) | Thời tiết (mưa, gió) ảnh hưởng demand; tương tác mùa |
| 3 | Deploying Robust Decision Support Systems for Transit Headway Control (arXiv:2509.08231) | GTFS + historical APC/demand; even-headway; objective chờ ↔ tần suất |
| 4 | May, A. D. (2023). *Transportation systems analysis and planning* | \(w = h/2\), \(h = 60/t\); passenger-min wait trong optimizer |
| 5 | LightGBM + weather trên MTA hourly (thực hành open data, e.g. GitHub nyc-subway-ridership-predictor) | Tree ensemble baseline so với NN |

**Không replicate đầy đủ:** GNN/hypergraph (MDPI MST-Hyper Trans) — cần đồ thị ga×tuyến; để hướng mở rộng sau.
