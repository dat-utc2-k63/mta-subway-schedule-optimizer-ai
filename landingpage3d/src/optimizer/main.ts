import Chart from 'chart.js/auto';

const API_BASE = import.meta.env.VITE_API_BASE_URL?.trim() || '';

interface SeasonMeta { id: string; label: string; month: number; }
interface Meta {
  routes: string[];
  seasons: SeasonMeta[];
  weather_presets: string[];
  defaults: Record<string, number>;
}

interface ScheduleRow {
  hour: number; direction: number; demand: number; trips: number;
  baseline: number; delta: number; headway_min: number | null;
  over_ceiling_pct: number; warn: boolean;
}
interface DemandRow { hour: number; direction: number; demand: number; capacity: number; }
interface KpiPair { opt: number; base: number; }
interface OptimizeResult {
  route_id: string; lambda_used: number; w_range: [number, number];
  n_directions: number; active_hours: number; predictor_ok: boolean;
  n_over_ceiling: number;
  schedule: ScheduleRow[];
  demand_by_hour: DemandRow[];
  kpis: Record<string, KpiPair>;
  pareto: { f1: number; f2: number; lambda: number }[];
  metric_table: { 'Chi so': string; 'Toi uu': number; Baseline: number; 'Cai thien': string }[];
}

function $<T extends HTMLElement>(id: string): T {
  const el = document.getElementById(id);
  if (!el) throw new Error(`Thiếu phần tử #${id}`);
  return el as T;
}

let mode: 'scenario' | 'date' = 'scenario';
let dayType: 'weekday' | 'weekend' = 'weekday';
let demandChart: Chart | null = null;
let paretoChart: Chart | null = null;

function setStatus(text: string, kind: 'idle' | 'running' | 'ok' | 'error'): void {
  const pill = $('status-pill');
  pill.textContent = text;
  pill.className = `status-pill status-pill--${kind}`;
}

async function loadMeta(): Promise<Meta> {
  const res = await fetch(`${API_BASE}/api/meta`);
  if (!res.ok) throw new Error('Không tải được cấu hình từ máy chủ.');
  return res.json();
}

function fillSelect(id: string, items: { value: string; label: string }[]): void {
  const sel = $<HTMLSelectElement>(id);
  sel.innerHTML = '';
  for (const it of items) {
    const opt = document.createElement('option');
    opt.value = it.value;
    opt.textContent = it.label;
    sel.appendChild(opt);
  }
}

function applyDefaults(d: Record<string, number>): void {
  const map: Record<string, string> = {
    min_headway_min: 'min_headway_min',
    max_headway_min: 'max_headway_min',
    overnight_max_headway_min: 'overnight_max_headway_min',
    capacity_per_trip: 'capacity_per_trip',
    max_overflow_pct: 'max_overflow_pct',
    smoothness_delta: 'smoothness_delta',
    route_fleet_limit: 'route_fleet_limit',
    cost_per_vehicle_hour: 'cost_per_vehicle_hour',
    trips_daytime_max_factor: 'trips_daytime_max_factor',
    trips_min_factor: 'trips_min_factor',
    trips_overnight_max_factor: 'trips_overnight_max_factor',
    trips_overnight_min_factor: 'trips_overnight_min_factor',
    max_over_ceiling_pct: 'max_over_ceiling_pct',
  };
  for (const [key, id] of Object.entries(map)) {
    if (d[key] !== undefined) ($<HTMLInputElement>(id)).value = String(d[key]);
  }
  if (d.w_low !== undefined) ($<HTMLInputElement>('w_low')).value = String(d.w_low);
  if (d.w_high !== undefined) ($<HTMLInputElement>('w_high')).value = String(d.w_high);
  // Cập nhật lại thanh fill + nhãn sau khi nạp mặc định.
  ($<HTMLInputElement>('w_high')).dispatchEvent(new Event('input'));
}

function num(id: string): number {
  return parseFloat(($<HTMLInputElement>(id)).value);
}
function checked(id: string): boolean {
  return ($<HTMLInputElement>(id)).checked;
}

function collectRequest(): Record<string, unknown> {
  const monthVal = ($<HTMLSelectElement>('month')).value;
  return {
    route_id: ($<HTMLSelectElement>('route')).value,
    mode,
    date: ($<HTMLInputElement>('date')).value || null,
    day_type: dayType,
    season: ($<HTMLSelectElement>('season')).value,
    month: monthVal === 'auto' ? null : parseInt(monthVal, 10),
    weather_preset: ($<HTMLSelectElement>('weather')).value,
    is_holiday: checked('is_holiday'),
    is_event: checked('is_event'),
    w_low: num('w_low'),
    w_high: num('w_high'),
    min_headway_min: num('min_headway_min'),
    max_headway_min: num('max_headway_min'),
    overnight_max_headway_min: num('overnight_max_headway_min'),
    capacity_per_trip: num('capacity_per_trip'),
    max_overflow_pct: num('max_overflow_pct'),
    smoothness_delta: num('smoothness_delta'),
    route_fleet_limit: num('route_fleet_limit'),
    cost_per_vehicle_hour: num('cost_per_vehicle_hour'),
    trips_daytime_max_factor: num('trips_daytime_max_factor'),
    trips_min_factor: num('trips_min_factor'),
    trips_overnight_max_factor: num('trips_overnight_max_factor'),
    trips_overnight_min_factor: num('trips_overnight_min_factor'),
    max_over_ceiling_pct: num('max_over_ceiling_pct'),
    use_capacity_constraint: checked('use_capacity_constraint'),
    use_smoothness_constraint: checked('use_smoothness_constraint'),
    use_route_fleet_cap: checked('use_route_fleet_cap'),
  };
}

function fmt(n: number): string {
  return n.toLocaleString('vi-VN');
}

function kpiCard(label: string, value: string, deltaText: string, cls: string): string {
  return `<div class="kpi"><div class="kpi-label">${label}</div>` +
    `<div class="kpi-value">${value}</div>` +
    `<div class="kpi-delta ${cls}">${deltaText}</div></div>`;
}

function deltaClass(delta: number, lowerIsBetter: boolean): string {
  if (Math.abs(delta) < 1e-9) return 'flat';
  const good = lowerIsBetter ? delta < 0 : delta > 0;
  return good ? 'good' : 'bad';
}

function renderKpis(r: OptimizeResult): void {
  const k = r.kpis;
  const cards: string[] = [];
  const trips = k.total_trips;
  cards.push(kpiCard('Tổng chuyến/ngày', fmt(trips.opt), `${trips.opt - trips.base >= 0 ? '+' : ''}${fmt(trips.opt - trips.base)} so với hiện tại`, 'flat'));
  const wait = k.avg_wait_min;
  cards.push(kpiCard('Chờ trung bình (phút)', wait.opt.toFixed(2), `${(wait.opt - wait.base).toFixed(2)} phút`, deltaClass(wait.opt - wait.base, true)));
  const cost = k.fleet_cost;
  cards.push(kpiCard('Tổng chi phí', fmt(cost.opt), `${cost.opt - cost.base >= 0 ? '+' : ''}${fmt(cost.opt - cost.base)}`, deltaClass(cost.opt - cost.base, true)));
  const fleet = k.min_fleet;
  cards.push(kpiCard('Số tàu tối thiểu', fleet.opt.toFixed(1), `${(fleet.opt - fleet.base).toFixed(1)} tàu`, deltaClass(fleet.opt - fleet.base, true)));
  const ov = k.overflow_pct;
  cards.push(kpiCard('Khung giờ quá tải', `${ov.opt.toFixed(1)}%`, `hiện tại ${ov.base.toFixed(1)}%`, deltaClass(ov.opt - ov.base, true)));
  $('kpis').innerHTML = cards.join('');
}

function renderSchedule(r: OptimizeResult): void {
  const tbody = $<HTMLTableSectionElement>('schedule-table').querySelector('tbody');
  if (!tbody) return;
  const rows = r.schedule.map((s) => {
    const deltaCls = s.delta > 0 ? 'cell-pos' : s.delta < 0 ? 'cell-neg' : '';
    const sign = s.over_ceiling_pct > 0 ? '+' : '';
    const overTxt = `${s.warn ? '⚠️ ' : ''}${sign}${s.over_ceiling_pct}%`;
    const overCls = s.warn ? 'cell-warn' : '';
    return `<tr>
      <td>${String(s.hour).padStart(2, '0')}:00</td>
      <td>${s.direction}</td>
      <td>${fmt(s.demand)}</td>
      <td>${s.trips}</td>
      <td>${s.baseline}</td>
      <td class="${deltaCls}">${s.delta >= 0 ? '+' : ''}${s.delta}</td>
      <td>${s.headway_min !== null ? s.headway_min.toFixed(1) : '—'}</td>
      <td class="${overCls}">${overTxt}</td>
    </tr>`;
  });
  tbody.innerHTML = rows.join('');
  const note = $('over-note');
  note.textContent = r.n_over_ceiling > 0 ? `⚠️ ${r.n_over_ceiling} khung giờ vượt ngưỡng cho phép` : '';
}

function renderMetricTable(r: OptimizeResult): void {
  const tbody = $<HTMLTableSectionElement>('metric-table').querySelector('tbody');
  if (!tbody) return;
  tbody.innerHTML = r.metric_table
    .map((m) => `<tr><td>${m['Chi so']}</td><td>${fmt(m['Toi uu'])}</td><td>${fmt(m.Baseline)}</td><td>${m['Cai thien']}</td></tr>`)
    .join('');
}

function renderCharts(r: OptimizeResult): void {
  const dirs = Array.from(new Set(r.demand_by_hour.map((d) => d.direction))).sort();
  const hours = Array.from(new Set(r.demand_by_hour.map((d) => d.hour))).sort((a, b) => a - b);
  const palette = ['#f0c040', '#6db8ec', '#8ef0bf', '#ff9b95'];

  const demandSets = dirs.flatMap((dir, i) => {
    const sub = r.demand_by_hour.filter((d) => d.direction === dir);
    const byHour = (key: 'demand' | 'capacity') => hours.map((h) => sub.find((d) => d.hour === h)?.[key] ?? 0);
    return [
      { label: `Cầu H${dir}`, data: byHour('demand'), borderColor: palette[i % palette.length], backgroundColor: 'transparent', tension: 0.3, pointRadius: 2 },
      { label: `Sức chứa H${dir}`, data: byHour('capacity'), borderColor: palette[i % palette.length], borderDash: [5, 4], backgroundColor: 'transparent', tension: 0.3, pointRadius: 0 },
    ];
  });

  demandChart?.destroy();
  demandChart = new Chart($<HTMLCanvasElement>('demand-chart'), {
    type: 'line',
    data: { labels: hours.map((h) => `${h}h`), datasets: demandSets },
    options: chartOpts('Giờ', 'Hành khách'),
  });

  paretoChart?.destroy();
  const pts = r.pareto.map((p) => ({ x: p.f2, y: p.f1 }));
  const kneeIdx = r.pareto.findIndex((p) => Math.round(p.lambda) === r.lambda_used);
  paretoChart = new Chart($<HTMLCanvasElement>('pareto-chart'), {
    type: 'scatter',
    data: {
      datasets: [
        { label: 'Đường Pareto', data: pts, showLine: true, borderColor: '#6db8ec', backgroundColor: '#6db8ec', pointRadius: 4, tension: 0.2 },
        ...(kneeIdx >= 0 ? [{ label: `Điểm cân bằng (λ=${r.lambda_used})`, data: [pts[kneeIdx]], borderColor: '#f0c040', backgroundColor: '#f0c040', pointRadius: 7, pointStyle: 'star' as const }] : []),
      ],
    },
    options: chartOpts('Tổng giờ-tàu (chi phí)', 'Tổng phút chờ'),
  });
}

function chartOpts(xTitle: string, yTitle: string) {
  const grid = { color: 'rgba(248,245,240,0.08)' };
  const ticks = { color: 'rgba(248,245,240,0.6)' };
  return {
    responsive: true,
    interaction: { mode: 'nearest' as const, intersect: false },
    plugins: { legend: { labels: { color: 'rgba(248,245,240,0.8)', boxWidth: 12 } } },
    scales: {
      x: { title: { display: true, text: xTitle, color: 'rgba(248,245,240,0.6)' }, grid, ticks },
      y: { title: { display: true, text: yTitle, color: 'rgba(248,245,240,0.6)' }, grid, ticks },
    },
  };
}

function render(r: OptimizeResult): void {
  $('empty-state').hidden = true;
  $('results-body').hidden = false;
  $('result-route').textContent = `Tuyến ${r.route_id}`;
  const predNote = r.predictor_ok ? '' : ' · ⚠️ dùng dữ liệu mẫu (model chưa tải)';
  $('result-sub').textContent = `${r.n_directions} hướng · ${r.active_hours} giờ hoạt động · điểm cân bằng λ=${r.lambda_used}${predNote}`;
  renderKpis(r);
  renderSchedule(r);
  renderMetricTable(r);
  renderCharts(r);
}

async function runOptimize(ev: Event): Promise<void> {
  ev.preventDefault();
  const btn = $<HTMLButtonElement>('run-btn');
  const err = $('form-error');
  err.hidden = true;
  btn.disabled = true;
  $('empty-state').hidden = true;
  $('results-body').hidden = true;
  $('loading').hidden = false;
  setStatus('Đang chạy…', 'running');
  try {
    const res = await fetch(`${API_BASE}/api/optimize`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(collectRequest()),
    });
    const data = await res.json();
    if (!res.ok) throw new Error(data.detail || 'Lỗi máy chủ.');
    render(data as OptimizeResult);
    setStatus('Hoàn thành', 'ok');
  } catch (e) {
    const msg = e instanceof Error ? e.message : String(e);
    err.textContent = `Không tối ưu được: ${msg}`;
    err.hidden = false;
    $('empty-state').hidden = false;
    setStatus('Lỗi', 'error');
  } finally {
    btn.disabled = false;
    $('loading').hidden = true;
  }
}

async function refreshWeather(): Promise<void> {
  const dateVal = ($<HTMLInputElement>('date')).value;
  const out = $('weather-summary');
  if (!dateVal) { out.textContent = ''; return; }
  out.textContent = 'Đang lấy thời tiết…';
  try {
    const res = await fetch(`${API_BASE}/api/weather?date=${dateVal}`);
    const data = await res.json();
    out.textContent = res.ok ? data.summary : (data.detail || '');
  } catch {
    out.textContent = '';
  }
}

function wireSegments(): void {
  $('mode-seg').querySelectorAll<HTMLButtonElement>('.seg-btn').forEach((b) => {
    b.addEventListener('click', () => {
      mode = (b.dataset.mode as 'scenario' | 'date') ?? 'scenario';
      $('mode-seg').querySelectorAll('.seg-btn').forEach((x) => x.classList.remove('is-active'));
      b.classList.add('is-active');
      $('panel-scenario').hidden = mode !== 'scenario';
      $('panel-date').hidden = mode !== 'date';
    });
  });
  $('daytype-seg').querySelectorAll<HTMLButtonElement>('.seg-btn').forEach((b) => {
    b.addEventListener('click', () => {
      dayType = (b.dataset.daytype as 'weekday' | 'weekend') ?? 'weekday';
      $('daytype-seg').querySelectorAll('.seg-btn').forEach((x) => x.classList.remove('is-active'));
      b.classList.add('is-active');
    });
  });
}

function wireDualRange(): void {
  const low = $<HTMLInputElement>('w_low');
  const high = $<HTMLInputElement>('w_high');
  const fill = $('w-fill');
  const gap = 0.05;
  const sync = () => {
    let lo = parseFloat(low.value);
    let hi = parseFloat(high.value);
    if (lo > hi - gap) {
      // Đẩy tay kéo còn lại để hai tay không vượt nhau.
      if (document.activeElement === low) hi = Math.min(1, lo + gap);
      else lo = Math.max(0, hi - gap);
      low.value = String(lo);
      high.value = String(hi);
    }
    fill.style.left = `${lo * 100}%`;
    fill.style.right = `${(1 - hi) * 100}%`;
    $('wlow-out').textContent = lo.toFixed(2);
    $('whigh-out').textContent = hi.toFixed(2);
  };
  low.addEventListener('input', sync);
  high.addEventListener('input', sync);
  sync();
}

async function init(): Promise<void> {
  wireSegments();
  wireDualRange();
  $<HTMLInputElement>('date').addEventListener('change', refreshWeather);
  $('opt-form').addEventListener('submit', runOptimize);
  $<HTMLInputElement>('date').value = new Date().toISOString().slice(0, 10);

  try {
    const meta = await loadMeta();
    fillSelect('route', meta.routes.map((r) => ({ value: r, label: `Tuyến ${r}` })));
    fillSelect('season', meta.seasons.map((s) => ({ value: s.id, label: s.label })));
    if (meta.seasons[1]) ($<HTMLSelectElement>('season')).value = meta.seasons[1].id;
    fillSelect('month', [{ value: 'auto', label: 'Tự động theo mùa' }, ...Array.from({ length: 12 }, (_, i) => ({ value: String(i + 1), label: `Tháng ${i + 1}` }))]);
    fillSelect('weather', meta.weather_presets.map((w) => ({ value: w, label: w })));
    applyDefaults(meta.defaults);
    setStatus('Sẵn sàng', 'idle');
  } catch (e) {
    setStatus('Không kết nối được máy chủ', 'error');
    const err = $('form-error');
    err.textContent = `${e instanceof Error ? e.message : e}. Hãy chạy backend: uvicorn demo.api.server:app --port 8000`;
    err.hidden = false;
  }
}

init();
