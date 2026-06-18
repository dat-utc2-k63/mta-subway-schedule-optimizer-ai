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
    WEEKDAY_WEEKEND_OPTIONS,
    WEEKDAY_WEEKEND_VI,
    ScenarioSelection,
    build_scenario_hourly_factors,
    list_holiday_names,
)
from lib.ui_lambda import (  # noqa: E402
    LAMBDA_BALANCED,
    PARETO_ZONE_ORDER,
    PARETO_ZONES,
    PRIORITY_COLORS,
    PRIORITY_VI,
    lambda_priority,
    lambda_to_pareto_index,
    pareto_compact_label,
    pareto_point_for_lambda,
)
from lib.ui_factors import (  # noqa: E402
    load_feature_medians,
    prepare_hourly_factors_for_model,
)
from lib.ui_constraints import (  # noqa: E402
    ConstraintOverrides,
    ValidationError,
    default_constraint_panel,
    merge_overrides,
    validate_constraint_config,
)
from lib.ui_weather_groups import (  # noqa: E402
    WEATHER_GROUP_KEYS,
    WEATHER_GROUP_VI,
    apply_weather_group_to_hourly,
    load_weather_groups,
)
from lib.ui_station_schedule import (  # noqa: E402
    BOROUGH_VI,
    NYC_BOROUGHS,
    schedule_by_station,
    station_schedule_dataframe,
)
UI_DIR = NOTEBOOKS / "outputs" / "default" / "ui_export"
DATA_DIR = ROOT / "datasets"
FACTORS_HOURLY = DATA_DIR / "factors_hourly.csv"
FACTORS_DAILY = DATA_DIR / "factors_daily.csv"
SCHEDULE_DIR = DATA_DIR / "schedule_current"
RIDERSHIP = DATA_DIR / "ridership.csv"

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
    feature_medians = load_feature_medians(UI_DIR)
    wg_path = UI_DIR / "weather_groups.json"
    if not wg_path.exists():
        from lib.ui_weather_groups import dump_weather_groups

        dump_weather_groups(wg_path, FACTORS_HOURLY)
    wg_mtime = os.path.getmtime(wg_path)
    weather_groups = load_weather_groups(str(UI_DIR), wg_mtime)
    return (
        predictor,
        ui_config,
        route_meta,
        optimizer_state,
        baseline_lookup,
        routes,
        feature_medians,
        weather_groups,
    )


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


def render_pareto_selector() -> tuple[int, float]:
    """Ưu tiên chờ ↔ chi phí — 3 vùng + 8 điểm λ curated."""
    zone_labels = [PARETO_ZONES[z]["label_vi"] for z in PARETO_ZONE_ORDER]
    zone = st.radio(
        "Vùng Pareto",
        zone_labels,
        index=0,
        help="Frontier weekday_peak — 8 điểm λ đều, lồi tốt.",
    )
    zone_key = next(k for k in PARETO_ZONE_ORDER if PARETO_ZONES[k]["label_vi"] == zone)
    zconf = PARETO_ZONES[zone_key]
    lambdas = list(zconf["lambdas"])
    default_lam = float(zconf.get("default", lambdas[0]))
    default_idx = lambdas.index(default_lam) if default_lam in lambdas else 0

    lambda_cost = st.selectbox(
        "Mức λ",
        lambdas,
        index=default_idx,
        format_func=pareto_compact_label,
    )
    st.caption(zconf["hint"])

    with st.expander("λ=211 — vùng diminishing returns (không khuyến nghị)", expanded=False):
        st.caption(PARETO_ZONES["avoid"]["hint"])
        use_avoid = st.checkbox("Dùng λ=211 dù biết rủi ro", value=False)
        if use_avoid:
            lambda_cost = 211.0
            st.warning(
                "λ=211: tỉ lệ lợi/chi tệ nhất trên frontier — chỉ dùng khi cần thử nghiệm."
            )

    pareto_index = lambda_to_pareto_index(lambda_cost)
    p = pareto_point_for_lambda(lambda_cost)
    if p and p.tag == "Knee":
        st.caption("✓ Điểm knee — khuyến nghị mặc định.")
    st.markdown(priority_pill(lambda_cost), unsafe_allow_html=True)
    return pareto_index, float(lambda_cost)


def apply_manual_weather(hourly: pd.DataFrame, manual: dict[str, float | int] | None) -> pd.DataFrame:
    if not manual:
        return hourly
    out = hourly.copy()
    for k, v in manual.items():
        if k in out.columns:
            out[k] = v
    if "apparent_temperature_c" in out.columns and "temperature_c" in manual:
        out["apparent_temperature_c"] = float(manual["temperature_c"]) - 2.0
    if "precipitation_mm" in out.columns and "rain_mm" in manual:
        out["precipitation_mm"] = float(manual["rain_mm"])
    return out


def render_constraint_panel(ui_config: dict) -> dict:
    """Sidebar panel — defaults from ui_config.json."""
    defaults = default_constraint_panel(ui_config)
    with st.expander("Ràng buộc nâng cao (tuỳ chọn)", expanded=False):
        enabled = st.checkbox("Bật tùy chỉnh ràng buộc", value=False)
        if not enabled:
            return defaults
        c1, c2 = st.columns(2)
        with c1:
            use_route_fleet = st.checkbox("Fleet theo tuyến", value=defaults["use_route_fleet"])
            use_system_fleet = st.checkbox("Fleet toàn hệ", value=defaults["use_system_fleet"])
        with c2:
            use_capacity = st.checkbox("Sàn sức chứa", value=defaults["use_capacity"])
            use_smoothness = st.checkbox("Mượt theo giờ", value=defaults["use_smoothness"])
        max_system_fleet = st.number_input(
            "max_system_fleet",
            value=float(defaults["max_system_fleet"]),
            min_value=1.0,
            step=10.0,
        )
        capacity_per_trip = st.number_input(
            "capacity_per_trip",
            value=float(defaults["capacity_per_trip"]),
            min_value=1.0,
            step=50.0,
        )
        smoothness = st.slider(
            "smoothness_max_delta",
            1,
            10,
            int(defaults["smoothness_max_delta"]),
        )
        hw1, hw2 = st.columns(2)
        with hw1:
            min_headway = st.number_input(
                "min_headway_min",
                value=float(defaults["min_headway_min"]),
                min_value=1.0,
                step=0.5,
            )
        with hw2:
            max_headway = st.number_input(
                "max_headway_min",
                value=float(defaults["max_headway_min"]),
                min_value=2.0,
                step=1.0,
            )
        st.caption("Factor chuyến theo nhóm giờ")
        f1, f2, f3 = st.columns(3)
        with f1:
            peak_f = st.number_input("peak max", value=float(defaults["trips_peak_max_factor"]), step=0.05)
        with f2:
            off_f = st.number_input("off-peak max", value=float(defaults["trips_offpeak_max_factor"]), step=0.05)
        with f3:
            ovn_f = st.number_input("overnight max", value=float(defaults["trips_overnight_max_factor"]), step=0.05)
        opt_target = st.selectbox(
            "opt_target",
            ["objective", "wait", "cost"],
            index=["objective", "wait", "cost"].index(str(defaults["opt_target"])),
        )
        if st.button("Reset về mặc định", use_container_width=True):
            st.rerun()
        return {
            **defaults,
            "use_route_fleet": use_route_fleet,
            "use_system_fleet": use_system_fleet,
            "use_capacity": use_capacity,
            "use_smoothness": use_smoothness,
            "max_system_fleet": max_system_fleet,
            "capacity_per_trip": capacity_per_trip,
            "smoothness_max_delta": int(smoothness),
            "min_headway_min": min_headway,
            "max_headway_min": max_headway,
            "trips_peak_max_factor": peak_f,
            "trips_offpeak_max_factor": off_f,
            "trips_overnight_max_factor": ovn_f,
            "opt_target": opt_target,
        }


def render_station_schedule_by_borough(route_id: str, result: dict) -> None:
    """Bảng lịch theo ga, chia tab theo quận."""
    st.subheader("Lịch theo ga (GTFS)")
    detail = result["detail"][["hour", "direction", "opt_trips"]].rename(
        columns={"opt_trips": "opt_trips"}
    )
    detail["route"] = route_id
    rid_m = os.path.getmtime(RIDERSHIP) if RIDERSHIP.exists() else 0.0
    sd_m = os.path.getmtime(SCHEDULE_DIR) if SCHEDULE_DIR.exists() else 0.0
    records = schedule_by_station(
        detail,
        schedule_dir=str(SCHEDULE_DIR),
        ridership_path=str(RIDERSHIP),
        route_id=route_id,
        schedule_dir_mtime=sd_m,
        ridership_mtime=rid_m,
    )
    if not records:
        st.info("Không mở rộng được lịch theo ga cho tuyến này.")
        return
    df = station_schedule_dataframe(records)
    boroughs = [b for b in NYC_BOROUGHS if b in df["borough"].unique()]
    if not boroughs:
        boroughs = sorted(df["borough"].dropna().unique().tolist())
    tabs = st.tabs([BOROUGH_VI.get(b, b) for b in boroughs] + ["Tất cả"])
    for tab, borough in zip(tabs, boroughs):
        with tab:
            sub = df.loc[df["borough"] == borough].sort_values(["scheduled_time", "station_name"])
            st.dataframe(
                sub[["station_name", "route", "direction", "hour", "scheduled_time"]],
                use_container_width=True,
                height=360,
            )
    with tabs[-1]:
        st.dataframe(
            df.sort_values(["borough", "scheduled_time"]),
            use_container_width=True,
            height=400,
        )


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
    factor_clip_note: str | None = None,
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

    if factor_clip_note:
        st.caption(factor_clip_note)

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

    binding = result.get("constraint_binding")
    if binding:
        with st.expander("Ràng buộc đang bind", expanded=False):
            st.write(
                f"TRIPS_MIN: **{binding['at_min_pct']:.0f}%** · "
                f"TRIPS_MAX: **{binding['at_max_pct']:.0f}%** · "
                f"Interior: **{binding['interior_pct']:.0f}%**"
            )
            if binding.get("by_hour_group"):
                st.dataframe(pd.DataFrame(binding["by_hour_group"]), use_container_width=True)

    render_station_schedule_by_borough(route_id, result)

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
        (
            predictor,
            ui_config,
            _,
            optimizer_state,
            _,
            routes,
            feature_medians,
            weather_groups,
        ) = load_artifacts(model_mtime)
    except Exception as exc:
        st.error(f"Không tải được model từ `{UI_DIR}`: {exc}")
        st.stop()

    st.markdown('<p class="hero-title">MTA · Tối ưu lịch trình</p>', unsafe_allow_html=True)

    with st.sidebar:
        route_id = st.selectbox("Tuyến", routes, index=routes.index("1") if "1" in routes else 0)

        st.markdown("---")
        _, lambda_cost = render_pareto_selector()
        st.markdown("---")

        input_mode = st.radio(
            "Nguồn demand",
            ["Theo ngày cụ thể", "Kịch bản tổng quát"],
            help="Chọn ngày thật (crawl/API) hoặc profile median từ training.",
        )

        selected_date: date | None = None
        scenario_sel: ScenarioSelection | None = None
        weather_group = st.selectbox(
            "Nhóm thời tiết",
            WEATHER_GROUP_KEYS,
            format_func=lambda k: WEATHER_GROUP_VI[k],
            index=WEATHER_GROUP_KEYS.index("sunny"),
        )

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
            st.caption(f"Nhóm: **{WEATHER_GROUP_VI[weather_group]}** (chọn ở trên)")

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
                weather="clear",
                filter_holiday=filter_holiday,
                holiday_name=holiday_name if filter_holiday else None,
                filter_major_event=filter_major_event,
            )

        st.markdown("---")
        constraint_panel = render_constraint_panel(ui_config)

        manual_weather: dict[str, float | int] | None = None
        with st.expander("Tuỳ chỉnh thủ công (nâng cao)", expanded=False):
            prof = weather_groups["groups"].get(weather_group, {})
            manual_weather = {}
            c1, c2 = st.columns(2)
            manual_weather["temperature_c"] = c1.number_input(
                "Nhiệt độ (°C)", value=float(prof.get("temperature_c", 15.0))
            )
            manual_weather["rain_mm"] = c2.number_input(
                "Mưa (mm)", value=float(prof.get("rain_mm", 0.0))
            )
            c3, c4 = st.columns(2)
            manual_weather["windspeed_kmh"] = c3.number_input(
                "Gió (km/h)", value=float(prof.get("windspeed_kmh", 15.0))
            )
            manual_weather["snowfall_cm"] = c4.number_input(
                "Tuyết (cm)", value=float(prof.get("snowfall_cm", 0.0))
            )
            f1, f2, f3 = st.columns(3)
            manual_weather["is_rain"] = int(f1.checkbox("Có mưa", value=bool(prof.get("is_rain", 0))))
            manual_weather["is_snow"] = int(f2.checkbox("Có tuyết", value=bool(prof.get("is_snow", 0))))
            manual_weather["is_severe_wind"] = int(
                f3.checkbox("Gió mạnh", value=bool(prof.get("is_severe_wind", 0)))
            )

        run_btn = st.button("Chạy tối ưu", type="primary", use_container_width=True)
        if st.button("Tải lại model", use_container_width=True):
            st.cache_resource.clear()
            st.rerun()

    if not run_btn:
        st.info(
            "Chọn tuyến, ưu tiên Pareto, ngày/kịch bản rồi bấm **Chạy tối ưu**. "
            "Ràng buộc nâng cao là tuỳ chọn."
        )
        return

    factor_clip_note: str | None = None
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
                prepared = prepare_hourly_factors_for_model(
                    query_hourly,
                    feature_medians=feature_medians,
                    factors_hourly_path=str(FACTORS_HOURLY),
                    factors_hourly_mtime=factors_mtime,
                )
                hourly_factors = prepared.hourly_factors
                factor_clip_note = prepared.clip_note
                if weather_group != "sunny":
                    hourly_factors = apply_weather_group_to_hourly(
                        hourly_factors, weather_group, weather_groups
                    )
                hourly_factors = apply_manual_weather(hourly_factors, manual_weather)
                nearest = resolve_to_nearest_training_day(
                    hourly_factors,
                    str(FACTORS_HOURLY),
                    file_mtime=factors_mtime,
                    query_date=pd.Timestamp(selected_date),
                )
                if not (nearest.is_self_match or nearest.distance < 1e-3):
                    scenario_warning = (
                        f"Profile ngày chọn không có trong training. "
                        f"Dùng ngày gần nhất: **{', '.join(nearest.nearest_dates)}** ({nearest.note})."
                    )
                is_weekend = int(hourly_factors["is_weekend"].median())
                source_label = query_src
                if weather_group != "sunny":
                    source_label += f" · nhóm {WEATHER_GROUP_VI[weather_group]}"
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
                hourly_factors = apply_weather_group_to_hourly(
                    hourly_factors, weather_group, weather_groups
                )
                hourly_factors = apply_manual_weather(hourly_factors, manual_weather)
                prepared = prepare_hourly_factors_for_model(
                    hourly_factors,
                    feature_medians=feature_medians,
                    factors_hourly_path=str(FACTORS_HOURLY),
                    factors_hourly_mtime=factors_mtime,
                )
                hourly_factors = prepared.hourly_factors
                factor_clip_note = prepared.clip_note
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
                    WEATHER_GROUP_VI[weather_group],
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
                            weather_group,
                        ]
                        + ([scenario_sel.holiday_name.replace(" ", "_")] if scenario_sel.holiday_name else [])
                        + (["major_event"] if scenario_sel.filter_major_event else [])
                    )
                )
        except Exception as exc:
            st.error(str(exc))
            st.stop()

        overrides = merge_overrides(ui_config, constraint_panel)
        try:
            validate_constraint_config(overrides, optimizer_state, route_id=route_id)
        except ValidationError as exc:
            st.error(str(exc))
            st.stop()

        demand_hourly = predict_route_hourly_demand(predictor, route_id, hourly_factors)
        cap = float(overrides.capacity_per_trip or ui_config.get("capacity_per_trip", 1200))
        result = optimize_route_day(
            route_id,
            demand_hourly,
            optimizer_state,
            lambda_cost=float(lambda_cost),
            capacity_per_trip=int(cap),
            is_weekend=is_weekend,
            lambda_ref=float(ui_config.get("lambda_opt", LAMBDA_BALANCED)),
            constraint_overrides=overrides,
            ui_config=ui_config,
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
        factor_clip_note=factor_clip_note,
    )


if __name__ == "__main__":
    main()
