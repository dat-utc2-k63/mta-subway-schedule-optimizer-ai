const API = "";

let meta = null;
let mode = "date";
let selectedPreset = "balanced";
let chart = null;
let dateProfile = null;
let overridesOpen = false;
let profileAbort = null;

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
    throw new Error(data?.detail || `HTTP ${res.status}`);
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

function fillSelect(el, items, valueKey = "key", labelKey = "label") {
  el.innerHTML = "";
  for (const item of items) {
    const opt = document.createElement("option");
    opt.value = item[valueKey];
    opt.textContent = item[labelKey];
    el.appendChild(opt);
  }
}

function setOverrideFieldsEnabled(enabled, { dateMode = false } = {}) {
  const fields = [
    $("weather-group"),
    $("filter-major"),
    $("filter-holiday"),
    $("weekday-weekend"),
    $("season"),
  ];
  for (const el of fields) {
    if (!el) continue;
    if (dateMode) {
      el.disabled = !enabled || el.id === "weekday-weekend" || el.id === "season";
    } else {
      el.disabled = !enabled;
    }
  }
}

function applyInferredToOverrides(inferred) {
  if (!inferred) return;
  $("weekday-weekend").value = inferred.weekday_weekend;
  $("season").value = inferred.season;
  $("weather-group").value = inferred.weather_group;
  $("filter-holiday").checked = false;
  $("filter-major").checked = inferred.filter_major_event || false;
}

function setMode(next) {
  mode = next;
  document.querySelectorAll(".mode-tab").forEach((t) => {
    t.classList.toggle("is-active", t.dataset.mode === mode);
  });

  const isDate = mode === "date";
  $("date-field").hidden = !isDate;
  $("date-profile").hidden = isDate ? !dateProfile : true;
  $("profile-loading").hidden = true;

  if (isDate) {
    $("overrides-hint").textContent =
      "Ngày và mùa lấy từ lịch. Có thể đổi thời tiết hoặc sự kiện lớn để mô phỏng what-if.";
    $("ctrl-weekday").hidden = false;
    $("ctrl-season").hidden = false;
    if (dateProfile) {
      $("overrides-section").hidden = false;
      setOverrideFieldsEnabled(true, { dateMode: true });
      updateRunButton();
    } else {
      $("overrides-section").hidden = true;
      setOverrideFieldsEnabled(false);
    }
  } else {
    dateProfile = null;
    $("overrides-section").hidden = false;
    $("overrides-body").hidden = false;
    $("overrides-toggle").setAttribute("aria-expanded", "true");
    overridesOpen = true;
    $("overrides-hint").textContent = "Tự xây kịch bản không gắn ngày cụ thể.";
    setOverrideFieldsEnabled(true);
    $("ctrl-weekday").hidden = false;
    $("ctrl-season").hidden = false;
    updateRunButton();
  }
}

function updateRunButton() {
  const btn = $("run-btn");
  if (mode === "date") {
    btn.disabled = !dateProfile;
  } else {
    btn.disabled = false;
  }
}

function renderPresets() {
  const wrap = $("preset-btns");
  wrap.innerHTML = "";
  for (const p of meta.presets) {
    const btn = document.createElement("button");
    btn.type = "button";
    btn.textContent = p.label.replace(" (khuyến nghị)", "");
    btn.title = p.description;
    btn.classList.toggle("is-active", p.key === selectedPreset);
    btn.addEventListener("click", () => {
      selectedPreset = p.key;
      renderPresets();
    });
    wrap.appendChild(btn);
  }
}

function renderProfile(profile) {
  $("profile-title").textContent = profile.title;
  $("profile-source").textContent = `Nguồn: ${profile.source}`;

  const w = profile.weather || {};
  const stats = $("profile-stats");
  stats.innerHTML = "";
  const pills = [];
  if (w.temp_min_c != null && w.temp_max_c != null) {
    pills.push(`🌡 ${fmt(w.temp_min_c, 0)}–${fmt(w.temp_max_c, 0)}°C`);
  }
  if (w.rain_total_mm != null) {
    pills.push(`🌧 ${fmt(w.rain_total_mm, 1)} mm`);
  }
  if (w.wind_max_kmh != null) {
    pills.push(`💨 ${fmt(w.wind_max_kmh, 0)} km/h`);
  }
  for (const text of pills) {
    const span = document.createElement("span");
    span.className = "stat-pill";
    span.textContent = text;
    stats.appendChild(span);
  }

  const chips = $("profile-chips");
  chips.innerHTML = "";
  for (const label of profile.chips || []) {
    const span = document.createElement("span");
    span.className = "chip";
    span.textContent = label;
    chips.appendChild(span);
  }

  $("date-profile").hidden = false;
  $("empty-state").hidden = true;
}

async function fetchDateProfile(dateStr) {
  if (!dateStr) return;

  if (profileAbort) profileAbort.abort();
  profileAbort = new AbortController();

  dateProfile = null;
  updateRunButton();
  $("date-profile").hidden = true;
  $("overrides-section").hidden = true;
  $("profile-loading").hidden = false;
  setOverrideFieldsEnabled(false);

  try {
    const profile = await api(
      `/api/v1/date-profile?date=${encodeURIComponent(dateStr)}`,
      { signal: profileAbort.signal },
    );
    dateProfile = profile;
    renderProfile(profile);
    applyInferredToOverrides(profile.inferred);
    $("overrides-section").hidden = false;
    setOverrideFieldsEnabled(true, { dateMode: true });
    updateRunButton();
  } catch (err) {
    if (err.name === "AbortError") return;
    showError(`Không lấy được dữ liệu ngày: ${err.message}`);
    $("empty-state").hidden = false;
  } finally {
    $("profile-loading").hidden = true;
  }
}

function hasOverrideChanges() {
  if (!dateProfile?.inferred) return false;
  const inf = dateProfile.inferred;
  return (
    $("weather-group").value !== inf.weather_group ||
    $("filter-major").checked !== !!inf.filter_major_event
  );
}

function buildPayload() {
  const payload = {
    route_id: $("route-id").value,
    preset: selectedPreset,
    mode,
  };

  if (mode === "date") {
    payload.date = $("selected-date").value;
    if (hasOverrideChanges()) {
      payload.use_overrides = true;
      payload.overrides = {
        weekday_weekend: $("weekday-weekend").value,
        season: $("season").value,
        weather_group: $("weather-group").value,
        filter_holiday: $("filter-holiday").checked,
        filter_major_event: $("filter-major").checked,
      };
    }
  } else {
    payload.scenario = {
      weekday_weekend: $("weekday-weekend").value,
      season: $("season").value,
      weather_group: $("weather-group").value,
      filter_holiday: $("filter-holiday").checked,
      filter_major_event: $("filter-major").checked,
    };
  }
  return payload;
}

function renderContext(ctx) {
  $("ctx-title").textContent = ctx.title || "";
  $("ctx-sub").textContent = ctx.subtitle || "";
  const warn = $("ctx-warn");
  if (ctx.warning) {
    warn.textContent = ctx.warning;
    warn.hidden = false;
  } else {
    warn.hidden = true;
  }
}

function renderSummary(s) {
  $("kpi-wait").textContent = fmt(s.wait_min, 1);
  const waitGood = s.wait_delta > 0;
  $("kpi-wait-d").textContent = `${waitGood ? "▼" : "▲"} ${fmt(Math.abs(s.wait_delta), 1)} vs GTFS`;
  $("kpi-wait-d").className = `kpi-card__delta ${waitGood ? "good" : "bad"}`;

  $("kpi-vh").textContent = fmt(s.vehicle_hours, 0);
  const vhGood = s.vh_delta <= 0;
  $("kpi-vh-d").textContent = `${vhGood ? "▼" : "▲"} ${fmt(Math.abs(s.vh_delta), 0)} vs GTFS`;
  $("kpi-vh-d").className = `kpi-card__delta ${vhGood ? "good" : "bad"}`;

  $("kpi-risk").textContent = fmt(s.overcrowding_pct, 0);
  const riskGood = s.overcrowding_delta_pp > 0;
  $("kpi-risk-d").textContent = `${riskGood ? "▼" : "▲"} ${fmt(Math.abs(s.overcrowding_delta_pp), 0)} pp vs GTFS`;
  $("kpi-risk-d").className = `kpi-card__delta ${riskGood ? "good" : "bad"}`;
}

function renderChart(chartData) {
  const ctx = $("main-chart").getContext("2d");
  if (chart) chart.destroy();
  chart = new Chart(ctx, {
    type: "bar",
    data: {
      labels: chartData.hours,
      datasets: [
        {
          label: "Trips GTFS",
          data: chartData.trips_gtfs,
          backgroundColor: "rgba(148, 163, 184, 0.55)",
          borderRadius: 4,
          yAxisID: "y",
          order: 2,
        },
        {
          label: "Trips AI",
          data: chartData.trips_ai,
          backgroundColor: "rgba(37, 99, 235, 0.7)",
          borderRadius: 4,
          yAxisID: "y",
          order: 1,
        },
        {
          label: "Headway AI (phút)",
          data: chartData.headway_ai,
          type: "line",
          borderColor: "#059669",
          borderWidth: 2,
          backgroundColor: "transparent",
          pointRadius: 3,
          pointHoverRadius: 5,
          yAxisID: "y1",
          order: 0,
        },
      ],
    },
    options: {
      responsive: true,
      interaction: { mode: "index", intersect: false },
      plugins: {
        legend: {
          labels: { boxWidth: 12, font: { size: 12 }, padding: 16 },
        },
      },
      scales: {
        x: {
          title: { display: true, text: "Giờ", font: { size: 12 } },
          grid: { display: false },
        },
        y: {
          position: "left",
          title: { display: true, text: "Số chuyến/giờ", font: { size: 12 } },
          beginAtZero: true,
          grid: { color: "rgba(0,0,0,0.04)" },
        },
        y1: {
          position: "right",
          title: { display: true, text: "Headway (phút)", font: { size: 12 } },
          grid: { drawOnChartArea: false },
          reverse: true,
        },
      },
    },
  });
}

function renderTable(rows) {
  const tbody = $("hour-table").querySelector("tbody");
  tbody.innerHTML = "";
  for (const r of rows) {
    const tr = document.createElement("tr");
    tr.innerHTML = `
      <td>${r.hour}</td>
      <td>${fmt(r.trips_gtfs, 0)}</td>
      <td>${fmt(r.trips_ai, 0)}</td>
      <td>${fmt(r.headway_ai, 1)}</td>`;
    tbody.appendChild(tr);
  }
}

function showError(msg) {
  $("error-state").textContent = msg;
  $("error-state").hidden = false;
}

async function init() {
  try {
    meta = await api("/api/v1/meta");
  } catch (err) {
    showError(`Không tải được cấu hình: ${err.message}`);
    $("model-meta").textContent = "";
    return;
  }

  $("model-meta").textContent = `Model · ${meta.model_built_at}`;

  fillSelect($("route-id"), meta.routes.map((r) => ({ key: r, label: `Tuyến ${r}` })));
  if (meta.routes.includes("1")) $("route-id").value = "1";

  fillSelect($("weekday-weekend"), meta.weekday_weekend);
  fillSelect($("season"), meta.seasons);
  fillSelect($("weather-group"), meta.weather_groups);

  const bounds = meta.date_bounds;
  const dateInput = $("selected-date");
  dateInput.min = bounds.min_date;
  dateInput.max = bounds.picker_max_date;
  dateInput.value = bounds.max_date || bounds.min_date;

  renderPresets();

  document.querySelectorAll(".mode-tab").forEach((tab) => {
    tab.addEventListener("click", () => setMode(tab.dataset.mode));
  });

  $("overrides-toggle").addEventListener("click", () => {
    overridesOpen = !overridesOpen;
    $("overrides-body").hidden = !overridesOpen;
    $("overrides-toggle").setAttribute("aria-expanded", String(overridesOpen));
  });

  dateInput.addEventListener("change", () => {
    if (mode === "date") fetchDateProfile(dateInput.value);
  });

  $("opt-form").addEventListener("submit", async (e) => {
    e.preventDefault();
    const btn = $("run-btn");
    btn.disabled = true;
    btn.textContent = "Đang tối ưu…";
    $("error-state").hidden = true;

    try {
      const data = await api("/api/v1/optimize", {
        method: "POST",
        body: JSON.stringify(buildPayload()),
      });
      $("results").hidden = false;
      $("empty-state").hidden = true;
      renderContext(data.context);
      renderSummary(data.summary);
      renderChart(data.chart);
      renderTable(data.hours);
    } catch (err) {
      showError(err.message);
    } finally {
      btn.textContent = "Tối ưu lịch trình";
      updateRunButton();
    }
  });

  setMode("date");
  fetchDateProfile(dateInput.value);
}

init();
