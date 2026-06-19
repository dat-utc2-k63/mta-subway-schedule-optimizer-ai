const API = "";

let meta = null;
let selectedPreset = "balanced";
let chart = null;
let lastCsvRows = null;

const $ = (id) => document.getElementById(id);

async function api(path, opts = {}) {
  const res = await fetch(`${API}${path}`, {
    headers: { "Content-Type": "application/json", ...(opts.headers || {}) },
    ...opts,
  });
  const text = await res.text();
  let data;
  try {
    data = text ? JSON.parse(text) : null;
  } catch {
    data = { detail: text };
  }
  if (!res.ok) {
    throw new Error(data?.detail || data?.message || `HTTP ${res.status}`);
  }
  return data;
}

function fmt(n, digits = 0) {
  if (n == null || Number.isNaN(n)) return "—";
  return Number(n).toLocaleString("vi-VN", {
    maximumFractionDigits: digits,
    minimumFractionDigits: digits,
  });
}

function setStatus(msg, kind = "idle") {
  const el = $("status");
  el.textContent = "";
  el.className = `status status--${kind}`;
  if (kind === "idle") {
    el.innerHTML = msg;
  } else {
    el.textContent = msg;
  }
}

function renderPresets() {
  const wrap = $("tradeoff-presets");
  wrap.innerHTML = "";
  for (const p of meta.tradeoff_presets) {
    const btn = document.createElement("button");
    btn.type = "button";
    btn.className = `preset-btn${p.key === selectedPreset ? " is-active" : ""}`;
    btn.dataset.preset = p.key;
    btn.innerHTML = `<strong>${p.label}</strong><small>${p.hint}</small>`;
    btn.addEventListener("click", () => selectPreset(p.key));
    wrap.appendChild(btn);
  }
  updateTradeoffHint();
}

async function updateTradeoffHint() {
  const route = $("route-id").value;
  if (!route) return;
  try {
    const t = await api(
      `/api/optimizer/tradeoff?route_id=${encodeURIComponent(route)}&preset=${selectedPreset}`,
    );
    $("tradeoff-hint").textContent = t.label;
  } catch {
    $("tradeoff-hint").textContent = "";
  }
}

function selectPreset(key) {
  selectedPreset = key;
  renderPresets();
}

function fillSelect(el, items, valueKey = "key", labelKey = "label") {
  el.innerHTML = "";
  for (const item of items) {
    const opt = document.createElement("option");
    opt.value = item[valueKey];
    opt.textContent = item[labelKey];
    el.appendChild(opt);
  }
}

function toggleInputMode() {
  const mode = $("input-mode").value;
  $("date-fields").hidden = mode !== "date";
  $("scenario-fields").hidden = mode !== "scenario";
}

function buildPayload() {
  const mode = $("input-mode").value;
  const payload = {
    route_id: $("route-id").value,
    tradeoff_preset: selectedPreset,
    input_mode: mode,
    weather_group: $("weather-group").value,
    use_route_fleet: $("use-route-fleet").checked,
  };
  if (mode === "date") {
    payload.selected_date = $("selected-date").value;
  } else {
    payload.weekday_weekend = $("weekday-weekend").value;
    payload.season = $("season").value;
    payload.filter_holiday = $("filter-holiday").checked;
    payload.filter_major_event = $("filter-major").checked;
    if ($("filter-holiday").checked && $("holiday-name").value) {
      payload.holiday_name = $("holiday-name").value;
    }
  }
  return payload;
}

function renderAlerts(data) {
  const wrap = $("alerts");
  wrap.innerHTML = "";
  if (data.context_badges?.length) {
    const badges = document.createElement("div");
    badges.className = "badges";
    for (const b of data.context_badges) {
      const span = document.createElement("span");
      span.className = "badge";
      span.textContent = b;
      badges.appendChild(span);
    }
    wrap.appendChild(badges);
  }
  if (data.scenario_warning) {
    const w = document.createElement("div");
    w.className = "alert alert--warn";
    w.textContent = data.scenario_warning;
    wrap.appendChild(w);
  }
  if (data.factor_clip_note) {
    const c = document.createElement("div");
    c.className = "alert";
    c.textContent = data.factor_clip_note;
    wrap.appendChild(c);
  }
}

function renderMetrics(m) {
  $("metrics").hidden = false;
  const wait = m.wait;
  $("m-wait").textContent = fmt(wait.optimized, 1);
  const waitGood = wait.delta > 0;
  $("m-wait-d").textContent = `${waitGood ? "▼" : "▲"} ${fmt(Math.abs(wait.delta), 1)} vs GTFS`;
  $("m-wait-d").className = `metric__delta ${waitGood ? "good" : "bad"}`;

  const vh = m.vehicle_hours;
  $("m-vh").textContent = fmt(vh.optimized, 0);
  const vhGood = vh.delta <= 0;
  let vhNote = `${vhGood ? "▼" : "▲"} ${fmt(Math.abs(vh.delta), 0)} vs GTFS · $${fmt(vh.cost_usd, 0)}`;
  const fu = m.fleet_utilization;
  if (fu?.optimized != null && fu?.baseline != null) {
    const dpp = (fu.optimized - fu.baseline) * 100;
    vhNote += ` · util ${fmt(fu.optimized * 100, 0)}% (${dpp >= 0 ? "+" : ""}${fmt(dpp, 1)}pp)`;
  }
  $("m-vh-d").textContent = vhNote;
  $("m-vh-d").className = `metric__delta ${vhGood ? "good" : "bad"}`;

  const risk = m.overcrowding;
  $("m-risk").textContent = `${fmt(risk.optimized, 0)}%`;
  const riskGood = risk.delta_pp > 0;
  $("m-risk-d").textContent = `${riskGood ? "▼" : "▲"} ${fmt(Math.abs(risk.delta_pp), 0)} pp vs GTFS`;
  $("m-risk-d").className = `metric__delta ${riskGood ? "good" : "bad"}`;
}

function renderChart(chartData) {
  $("chart-wrap").hidden = false;
  const ctx = $("main-chart").getContext("2d");
  if (chart) chart.destroy();
  chart = new Chart(ctx, {
    type: "bar",
    data: {
      labels: chartData.hours,
      datasets: [
        {
          label: "Demand GTFS",
          data: chartData.baseline_demand,
          backgroundColor: "rgba(139, 155, 180, 0.35)",
          yAxisID: "y",
          order: 2,
        },
        {
          label: "Demand AI",
          data: chartData.predicted_demand,
          backgroundColor: "rgba(61, 214, 198, 0.45)",
          yAxisID: "y",
          order: 1,
        },
        {
          label: "Headway GTFS",
          data: chartData.baseline_headway,
          type: "line",
          borderColor: "#8b9bb4",
          borderDash: [4, 4],
          pointRadius: 0,
          yAxisID: "y1",
        },
        {
          label: "Headway AI",
          data: chartData.opt_headway,
          type: "line",
          borderColor: "#3dd6c6",
          pointRadius: 0,
          yAxisID: "y1",
        },
      ],
    },
    options: {
      responsive: true,
      interaction: { mode: "index", intersect: false },
      plugins: {
        legend: { labels: { color: "#8b9bb4", boxWidth: 12 } },
      },
      scales: {
        x: { ticks: { color: "#8b9bb4" }, grid: { color: "rgba(255,255,255,0.05)" } },
        y: {
          position: "left",
          title: { display: true, text: "Hành khách/giờ", color: "#8b9bb4" },
          ticks: { color: "#8b9bb4" },
          grid: { color: "rgba(255,255,255,0.05)" },
        },
        y1: {
          position: "right",
          reverse: true,
          title: { display: true, text: "Headway (phút)", color: "#8b9bb4" },
          ticks: { color: "#8b9bb4" },
          grid: { drawOnChartArea: false },
        },
      },
    },
  });
}

function renderTable(rows) {
  $("table-wrap").hidden = false;
  const tbody = $("schedule-table").querySelector("tbody");
  tbody.innerHTML = "";
  for (const r of rows) {
    const tr = document.createElement("tr");
    tr.innerHTML = `
      <td>${r.hour}</td>
      <td>${fmt(r.baseline_demand, 0)}</td>
      <td>${fmt(r.predicted_demand, 0)}</td>
      <td>${fmt(r.baseline_trips, 0)}</td>
      <td>${fmt(r.opt_trips, 0)}</td>
      <td>${fmt(r.baseline_headway_min, 1)}</td>
      <td>${fmt(r.opt_headway_min, 1)}</td>`;
    tbody.appendChild(tr);
  }
}

function downloadCsv() {
  if (!lastCsvRows?.length) return;
  const cols = Object.keys(lastCsvRows[0]);
  const lines = [cols.join(",")];
  for (const row of lastCsvRows) {
    lines.push(cols.map((c) => JSON.stringify(row[c] ?? "")).join(","));
  }
  const blob = new Blob([lines.join("\n")], { type: "text/csv;charset=utf-8" });
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob);
  a.download = `mta_${lastCsvRows[0].route || "export"}.csv`;
  a.click();
  URL.revokeObjectURL(a.href);
}

async function init() {
  try {
    meta = await api("/api/optimizer/meta");
  } catch (err) {
    setStatus(`Không tải được model: ${err.message}`, "error");
    return;
  }

  $("model-meta").textContent = `Model · ${meta.model_built_at}`;

  fillSelect($("route-id"), meta.routes.map((r) => ({ key: r, label: `Tuyến ${r}` })));
  if (meta.routes.includes("1")) $("route-id").value = "1";

  fillSelect($("weekday-weekend"), meta.weekday_weekend);
  fillSelect($("season"), meta.seasons);
  fillSelect($("weather-group"), meta.weather_groups);

  const hol = $("holiday-name");
  for (const name of meta.holiday_names || []) {
    const opt = document.createElement("option");
    opt.value = name;
    opt.textContent = name;
    hol.appendChild(opt);
  }

  const bounds = meta.date_bounds;
  const dateInput = $("selected-date");
  dateInput.min = bounds.min_date;
  dateInput.max = bounds.picker_max_date;
  dateInput.value = "2024-06-03";

  renderPresets();

  $("input-mode").addEventListener("change", toggleInputMode);
  $("route-id").addEventListener("change", updateTradeoffHint);
  $("filter-holiday").addEventListener("change", () => {
    $("holiday-wrap").hidden = !$("filter-holiday").checked;
  });
  $("csv-btn").addEventListener("click", downloadCsv);

  $("opt-form").addEventListener("submit", async (e) => {
    e.preventDefault();
    const btn = $("run-btn");
    btn.disabled = true;
    setStatus("Đang dự báo demand & tối ưu…", "loading");
    $("metrics").hidden = true;
    $("chart-wrap").hidden = true;
    $("table-wrap").hidden = true;
    $("alerts").innerHTML = "";

    try {
      const data = await api("/api/optimizer/run", {
        method: "POST",
        body: JSON.stringify(buildPayload()),
      });
      setStatus(`λ = ${fmt(data.lambda_cost, 0)} · ${data.source_label}`, "idle");
      renderAlerts(data);
      renderMetrics(data.metrics);
      renderChart(data.chart);
      renderTable(data.schedule);
      lastCsvRows = data.csv_detail;
    } catch (err) {
      setStatus(err.message, "error");
    } finally {
      btn.disabled = false;
    }
  });

  toggleInputMode();
}

init();
