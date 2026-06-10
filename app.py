"""
MTA AI-Driven Transit Schedule Optimization — Streamlit dashboard.
"""

from __future__ import annotations

import json
import os
import sys
from datetime import date, datetime, timedelta
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

ROOT = Path(__file__).resolve().parent
NOTEBOOKS = ROOT / "notebooks"
sys.path.insert(0, str(NOTEBOOKS))

from lib.demand_runtime import DemandPredictor  # noqa: E402
from lib.ui_optimizer import optimize_route_day  # noqa: E402
from lib.ui_weather import (  # noqa: E402
    PICKER_FUTURE_DAYS,
    build_hourly_factors,
)
from lib.ui_nearest import resolve_to_nearest_training_day  # noqa: E402
from lib.ui_scenario import (  # noqa: E402
    SEASONS,
    SEASON_VI,
    WEATHER_OPTIONS,
    WEATHER_VI,
    WEEKDAY_WEEKEND_OPTIONS,
    WEEKDAY_WEEKEND_VI,
    ScenarioSelection,
    build_scenario_hourly_factors,
    list_holiday_names,
)
from lib.ui_lambda import (  # noqa: E402
    LAMBDA_BALANCED,
    PARETO_COUNT,
    PRIORITY_COLORS,
    PRIORITY_VI,
    default_pareto_index,
    lambda_at_pareto_index,
    lambda_priority,
)
UI_DIR = NOTEBOOKS / "outputs" / "default" / "ui_export"
DATA_DIR = ROOT / "datasets"
FACTORS_HOURLY = DATA_DIR / "factors_hourly.csv"
FACTORS_DAILY = DATA_DIR / "factors_daily.csv"

THEME_CSS = """
<style>
    @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600;700&display=swap');
    html, body, [class*="css"] { font-family: 'IBM Plex Sans', sans-serif; }
    .stApp { background: linear-gradient(160deg, #0f172a 0%, #1e293b 50%, #134e4a 100%); }
    [data-testid="stSidebar"] {
        background: #0f172a; border-right: 1px solid #334155;
    }
    .metric-card {
        background: rgba(15, 23, 42, 0.9); border: 1px solid #334155;
        border-radius: 10px; padding: 0.9rem 1.1rem;
    }
    .metric-card h3 { color: #94a3b8; font-size: 0.8rem; margin: 0 0 0.25rem 0; }
    .metric-card .value { color: #f8fafc; font-size: 1.5rem; font-weight: 700; }
    .metric-card .delta-good { color: #2dd4bf; font-size: 0.85rem; font-weight: 600; }
    .metric-card .delta-bad { color: #f87171; font-size: 0.85rem; font-weight: 600; }
    .hero-title { color: #e2e8f0; font-size: 1.4rem; font-weight: 700; margin-bottom: 0.15rem; }
    .priority-pill {
        display: inline-block; padding: 4px 12px; border-radius: 999px;
        font-size: 0.85rem; font-weight: 600; margin-top: 0.25rem;
    }
</style>
"""

BADGE_STYLE = (
    "background:#1e293b;border:1px solid #334155;border-radius:16px;"
    "padding:2px 10px;font-size:12px;color:#94a3b8"
)


def metric_card(title: str, value: str, delta: str | None = None, *, good: bool = True) -> str:
    css = "delta-good" if good else "delta-bad"
    delta_html = f'<div class="{css}">{delta}</div>' if delta else ""
    return f'<div class="metric-card"><h3>{title}</h3><div class="value">{value}</div>{delta_html}</div>'


def priority_pill(lam: float) -> str:
    key = lambda_priority(lam)
    color = PRIORITY_COLORS[key]
    label = PRIORITY_VI[key]
    return (
        f'<span class="priority-pill" style="background:{color}22;color:{color};'
        f'border:1px solid {color}55">{label} · λ={lam:.0f}</span>'
    )


@st.cache_data(show_spinner=False)
def factors_date_bounds() -> tuple[date, date, date]:
    fd = pd.read_csv(FACTORS_DAILY, parse_dates=["date"])
    min_d = fd["date"].min().date()
    max_d = fd["date"].max().date()
    picker_max = date.today() + timedelta(days=PICKER_FUTURE_DAYS)
    return min_d, max_d, picker_max


@st.cache_resource(show_spinner="Đang tải model…")
def load_artifacts(_model_mtime: float):
    predictor = DemandPredictor.load(UI_DIR)
    with (UI_DIR / "ui_config.json").open(encoding="utf-8") as f:
        ui_config = json.load(f)
    with (UI_DIR / "route_meta.json").open(encoding="utf-8") as f:
        route_meta = json.load(f)
    import joblib

    optimizer_state = joblib.load(UI_DIR / "optimizer_state.pkl")
    baseline_lookup = pd.read_json(UI_DIR / "baseline_lookup.json")
    routes = sorted(baseline_lookup["route_id"].astype(str).unique().tolist())
    ui_config["_model_built_at"] = datetime.fromtimestamp(_model_mtime).strftime(
        "%Y-%m-%d %H:%M"
    )
    return predictor, ui_config, route_meta, optimizer_state, baseline_lookup, routes


def predict_route_hourly_demand(
    predictor: DemandPredictor,
    route_id: str,
    hourly_factors: pd.DataFrame,
) -> pd.Series:
    feat = predictor.build_features_from_hourly_df([route_id], hourly_factors)
    pred = predictor.predict(feat)
    return pred.set_index("hour")["demand"].sort_index()


def factor_table_view(hourly_factors: pd.DataFrame) -> pd.DataFrame:
    """24 hàng factor đầu vào demand model, giờ lên trước."""
    if "hour" in hourly_factors.columns:
        view = hourly_factors.sort_values("hour").reset_index(drop=True)
    else:
        view = hourly_factors.copy()
    lead = [c for c in ("hour", "date", "timestamp") if c in view.columns]
    rest = [c for c in view.columns if c not in lead]
    return view[lead + rest]


def build_demand_headway_chart(hourly: pd.DataFrame) -> go.Figure:
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    if "baseline_demand" in hourly.columns:
        fig.add_trace(
            go.Bar(
                x=hourly["hour"],
                y=hourly["baseline_demand"],
                name="Demand baseline",
                marker_color="rgba(148, 163, 184, 0.4)",
            ),
            secondary_y=False,
        )
    fig.add_trace(
        go.Bar(
            x=hourly["hour"],
            y=hourly["predicted_demand"],
            name="Demand AI",
            marker_color="rgba(45, 212, 191, 0.55)",
        ),
        secondary_y=False,
    )
    fig.add_trace(
        go.Scatter(
            x=hourly["hour"],
            y=hourly["baseline_headway_min"],
            name="Headway GTFS",
            line=dict(color="#94a3b8", width=2, dash="dot"),
        ),
        secondary_y=True,
    )
    fig.add_trace(
        go.Scatter(
            x=hourly["hour"],
            y=hourly["opt_headway_min"],
            name="Headway AI",
            line=dict(color="#38bdf8", width=2.5),
        ),
        secondary_y=True,
    )
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(15,23,42,0.5)",
        height=360,
        margin=dict(l=40, r=40, t=40, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
    )
    fig.update_xaxes(title_text="Giờ", dtick=2)
    fig.update_yaxes(title_text="Hành khách/giờ", secondary_y=False)
    fig.update_yaxes(title_text="Headway (phút)", secondary_y=True, autorange="reversed")
    return fig


def render_results(
    *,
    route_id: str,
    lambda_cost: float,
    hourly_factors: pd.DataFrame,
    source_label: str,
    result: dict,
    ui_config: dict,
    export_tag: str,
    context_badges: list[str] | None = None,
    scenario_warning: str | None = None,
) -> None:
    badges = " ".join(
        f'<span style="{BADGE_STYLE}">{b}</span>' for b in (context_badges or [])
    )
    st.markdown(
        f'<div style="display:flex;gap:8px;flex-wrap:wrap;margin:0.5rem 0">{badges}</div>',
        unsafe_allow_html=True,
    )
    st.markdown(priority_pill(lambda_cost), unsafe_allow_html=True)

    if scenario_warning:
        st.warning(scenario_warning)

    if result.get("missing_hours"):
        st.warning(f"Thiếu {len(result['missing_hours'])} giờ — dùng median fallback.")

    base_m = result["baseline_metrics"]
    opt_m = result["optimized_metrics"]
    wait_delta = base_m["weighted_avg_wait_min"] - opt_m["weighted_avg_wait_min"]
    trips_delta = opt_m["total_trips"] - base_m["total_trips"]
    risk_delta = result["overcrowding_baseline"] - result["overcrowding_optimized"]
    fleet_unit_cost = float(ui_config.get("lambda_cost_eval", 150.0))
    opt_cost = opt_m["total_trips"] * fleet_unit_cost

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(
            metric_card(
                "Chờ TB (phút)",
                f"{opt_m['weighted_avg_wait_min']:.1f}",
                f"{'▼' if wait_delta > 0 else '▲'} {abs(wait_delta):.1f} vs GTFS",
                good=wait_delta > 0,
            ),
            unsafe_allow_html=True,
        )
    with c2:
        st.markdown(
            metric_card(
                "Tổng chuyến",
                f"{int(opt_m['total_trips'])}",
                f"{'▼' if trips_delta < 0 else '▲'} {abs(trips_delta):.0f} vs GTFS · ${opt_cost:,.0f}",
                good=trips_delta <= 0,
            ),
            unsafe_allow_html=True,
        )
    with c3:
        st.markdown(
            metric_card(
                "Nguy cơ quá tải (giờ cao điểm)",
                f"{result['overcrowding_optimized']:.0f}%",
                f"{'▼' if risk_delta > 0 else '▲'} {abs(risk_delta):.0f} pp vs GTFS",
                good=risk_delta > 0,
            ),
            unsafe_allow_html=True,
        )

    st.plotly_chart(build_demand_headway_chart(result["hourly"]), use_container_width=True)

    hourly = result["hourly"].copy()
    hourly["hour"] = hourly["hour"].astype(int)
    schedule_view = hourly[
        [
            "hour",
            "baseline_demand",
            "predicted_demand",
            "baseline_trips",
            "opt_trips",
            "baseline_headway_min",
            "opt_headway_min",
        ]
    ].rename(
        columns={
            "baseline_demand": "demand_gtfs",
            "predicted_demand": "demand_ai",
            "baseline_trips": "trips_gtfs",
            "opt_trips": "trips_ai",
            "baseline_headway_min": "headway_gtfs",
            "opt_headway_min": "headway_ai",
        }
    )
    st.dataframe(
        schedule_view.style.format(
            {
                "demand_gtfs": "{:,.0f}",
                "demand_ai": "{:,.0f}",
                "trips_gtfs": "{:.0f}",
                "trips_ai": "{:.0f}",
                "headway_gtfs": "{:.1f}",
                "headway_ai": "{:.1f}",
            }
        ),
        use_container_width=True,
        height=340,
    )

    with st.expander("Bảng factor (24h)", expanded=True):
        st.caption(f"Features đầu vào model · **{source_label}**")
        st.dataframe(factor_table_view(hourly_factors), use_container_width=True, height=420)

    with st.expander("Chi tiết theo hướng"):
        st.dataframe(result["hourly_by_direction"], use_container_width=True)

    export_df = result["detail"].copy()
    export_df["route"] = route_id
    export_df["run_context"] = export_tag
    export_df["priority"] = PRIORITY_VI[lambda_priority(lambda_cost)]
    export_df["lambda_cost"] = lambda_cost
    export_df = export_df.rename(columns={"predicted_demand": "ai_predicted_demand"})
    st.download_button(
        "Tải CSV",
        data=export_df.to_csv(index=False).encode("utf-8"),
        file_name=f"mta_{route_id}_{export_tag}_lam{lambda_cost}.csv",
        mime="text/csv",
        use_container_width=True,
    )


def main() -> None:
    st.set_page_config(
        page_title="MTA Schedule Optimizer",
        page_icon="🚌",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    st.markdown(THEME_CSS, unsafe_allow_html=True)

    model_mtime = os.path.getmtime(UI_DIR / "optimizer_state.pkl")
    factors_mtime = os.path.getmtime(FACTORS_HOURLY)
    min_date, _, picker_max_date = factors_date_bounds()

    try:
        predictor, ui_config, _, optimizer_state, _, routes = load_artifacts(model_mtime)
    except Exception as exc:
        st.error(f"Không tải được model từ `{UI_DIR}`: {exc}")
        st.stop()

    st.markdown('<p class="hero-title">MTA · Tối ưu lịch trình</p>', unsafe_allow_html=True)

    with st.sidebar:
        route_id = st.selectbox("Tuyến", routes, index=routes.index("1") if "1" in routes else 0)

        input_mode = st.radio(
            "Nguồn demand",
            ["Theo ngày cụ thể", "Kịch bản tổng quát"],
            help="Hai chế độ độc lập: chọn ngày thật hoặc profile median từ training.",
        )

        selected_date: date | None = None
        scenario_sel: ScenarioSelection | None = None

        if input_mode == "Theo ngày cụ thể":
            default_pick = min(max(date(2024, 6, 3), min_date), picker_max_date)
            selected_date = st.date_input(
                "Ngày",
                value=default_pick,
                min_value=min_date,
                max_value=picker_max_date,
            )
        else:
            st.markdown("**1 · Ngày thường / cuối tuần**")
            weekday_weekend = st.radio(
                "weekday_weekend",
                WEEKDAY_WEEKEND_OPTIONS,
                format_func=lambda x: WEEKDAY_WEEKEND_VI[x],
                horizontal=True,
                label_visibility="collapsed",
            )

            st.markdown("**2 · Mùa**")
            season = st.radio(
                "season",
                SEASONS,
                index=SEASONS.index("summer"),
                format_func=lambda x: SEASON_VI[x],
                horizontal=True,
                label_visibility="collapsed",
            )

            st.markdown("**3 · Thời tiết**")
            weather = st.radio(
                "weather",
                WEATHER_OPTIONS,
                format_func=lambda x: WEATHER_VI[x],
                horizontal=True,
                label_visibility="collapsed",
            )

            st.markdown("**4 · Ngày lễ**")
            filter_holiday = st.checkbox("Lọc theo ngày lễ", value=False)
            holiday_name: str | None = None
            if filter_holiday:
                names = list_holiday_names(str(FACTORS_DAILY))
                if not names:
                    st.warning("Không có tên ngày lễ trong factors_daily.")
                else:
                    pick = st.selectbox(
                        "Tên ngày lễ (tùy chọn)",
                        ["— Tất cả ngày lễ —", *names],
                        help="Bỏ trống = mọi ngày lễ trong training. Chọn tên = chỉ ngày lễ đó.",
                    )
                    if pick != "— Tất cả ngày lễ —":
                        holiday_name = pick

            st.markdown("**5 · Sự kiện lớn**")
            filter_major_event = st.checkbox("Lọc ngày có sự kiện lớn", value=False)

            scenario_sel = ScenarioSelection(
                weekday_weekend=weekday_weekend,
                season=season,
                weather=weather,
                filter_holiday=filter_holiday,
                holiday_name=holiday_name if filter_holiday else None,
                filter_major_event=filter_major_event,
            )

        st.markdown("**Ưu tiên chờ ↔ chi phí**")
        pareto_index = st.slider(
            "Mức Pareto",
            min_value=1,
            max_value=PARETO_COUNT,
            value=default_pareto_index(),
            step=1,
            help="1 = ưu tiên chờ · 20 = ưu tiên chi phí",
        )
        lambda_cost = lambda_at_pareto_index(pareto_index)
        st.markdown(f"**λ = {lambda_cost:.0f}**")
        st.markdown(priority_pill(lambda_cost), unsafe_allow_html=True)

        run_btn = st.button("Chạy tối ưu", type="primary", use_container_width=True)
        if st.button("Tải lại model", use_container_width=True):
            st.cache_resource.clear()
            st.rerun()

    if not run_btn:
        if input_mode == "Theo ngày cụ thể":
            st.info("Chọn tuyến, ngày, kéo mức Pareto (1–20) rồi bấm **Chạy tối ưu**.")
        else:
            st.info(
                "Chọn nhóm bắt buộc (ngày, mùa, thời tiết), tùy chọn lễ/sự kiện, "
                "mức Pareto (1–20) rồi **Chạy tối ưu**."
            )
        return

    with st.spinner("Đang dự báo demand & tối ưu…"):
        scenario_warning: str | None = None
        try:
            if input_mode == "Theo ngày cụ thể":
                assert selected_date is not None
                query_hourly, query_src = build_hourly_factors(
                    selected_date,
                    factors_hourly_path=str(FACTORS_HOURLY),
                    factors_daily_path=str(FACTORS_DAILY),
                )
                nearest = resolve_to_nearest_training_day(
                    query_hourly,
                    str(FACTORS_HOURLY),
                    file_mtime=factors_mtime,
                    query_date=pd.Timestamp(selected_date),
                )
                hourly_factors = nearest.hourly_factors
                is_weekend = int(hourly_factors["is_weekend"].median())
                if nearest.is_self_match or nearest.distance < 1e-3:
                    source_label = query_src
                else:
                    source_label = f"{query_src} → NN training: {', '.join(nearest.nearest_dates[:3])}"
                    scenario_warning = (
                        f"Profile ngày chọn không có trong training (hoặc khác feature). "
                        f"Dùng ngày gần nhất trong không gian feature: **{', '.join(nearest.nearest_dates)}** "
                        f"({nearest.note})."
                    )
                day_label = "Cuối tuần" if is_weekend else "Ngày thường"
                context_badges = [
                    f"Tuyến {route_id}",
                    selected_date.strftime("%d/%m/%Y") + f" · {day_label}",
                    source_label,
                ]
                export_tag = selected_date.isoformat()
            else:
                assert scenario_sel is not None
                built = build_scenario_hourly_factors(
                    scenario_sel,
                    factors_daily_path=str(FACTORS_DAILY),
                    factors_hourly_path=str(FACTORS_HOURLY),
                    file_mtime=factors_mtime,
                )
                hourly_factors = built.hourly_factors
                is_weekend = built.is_weekend
                scenario_warning = None
                if built.nn_distance >= 1e-3:
                    scenario_warning = (
                        f"Kịch bản → ngày training gần nhất trong không gian feature: "
                        f"**{', '.join(built.sample_dates)}** ({built.match_note})."
                    )
                source_label = (
                    f"Kịch bản: {built.label} · NN → {built.n_days} ngày · {built.match_note}"
                )
                context_badges = [
                    f"Tuyến {route_id}",
                    f"Kịch bản · {built.label}",
                    f"{built.n_days} ngày · median/giờ",
                ]
                if scenario_sel.filter_holiday:
                    if built.holiday_name:
                        context_badges.append(f"Ngày lễ: {built.holiday_name}")
                    else:
                        context_badges.append("Ngày lễ: tất cả")
                export_tag = (
                    "scenario_"
                    + "_".join(
                        [
                            scenario_sel.weekday_weekend,
                            scenario_sel.season,
                            scenario_sel.weather,
                        ]
                        + ([scenario_sel.holiday_name.replace(" ", "_")] if scenario_sel.holiday_name else [])
                        + (["major_event"] if scenario_sel.filter_major_event else [])
                    )
                )
        except Exception as exc:
            st.error(str(exc))
            st.stop()

        demand_hourly = predict_route_hourly_demand(predictor, route_id, hourly_factors)
        result = optimize_route_day(
            route_id,
            demand_hourly,
            optimizer_state,
            lambda_cost=float(lambda_cost),
            capacity_per_trip=int(ui_config.get("capacity_per_trip", 1200)),
            is_weekend=is_weekend,
            lambda_ref=float(ui_config.get("lambda_opt", LAMBDA_BALANCED)),
        )

    render_results(
        route_id=route_id,
        lambda_cost=float(lambda_cost),
        hourly_factors=hourly_factors,
        source_label=source_label,
        result=result,
        ui_config=ui_config,
        export_tag=export_tag,
        context_badges=context_badges,
        scenario_warning=scenario_warning,
    )


if __name__ == "__main__":
    main()
