"use strict";

const METRICS = [
  {key: "r_precision_top1", label: "R@1", higher: true, digits: 4},
  {key: "r_precision_top2", label: "R@2", higher: true, digits: 4},
  {key: "r_precision_top3", label: "R@3", higher: true, digits: 4},
  {key: "fid", label: "FID", lower: true, digits: 4},
  {key: "mm_dist", label: "MM-Dist", lower: true, digits: 4},
  {key: "control_error", label: "Control error", lower: true, digits: 3},
  {key: "foot_skating", label: "Foot skating", lower: true, digits: 4},
  {key: "jitter", label: "Jitter", lower: true, digits: 2},
  {key: "diversity", label: "Diversity", referenceStatistic: true, digits: 3},
];

const COLORS = ["#087d72", "#315f9d", "#c7563f", "#ad6d00"];
const state = {
  data: null,
  type: "all",
  density: "all",
  family: "all",
  settingId: null,
  metric: "r_precision_top3",
  selectedMethods: new Set(),
};
let barChart = null;
let radarChart = null;

const safe = (value) => String(value ?? "").replace(
  /[&<>"']/g,
  (char) => ({"&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;"})[char],
);

function metricSpec(key) {
  return METRICS.find((metric) => metric.key === key);
}

function settingLabel(setting) {
  const space = setting.axes ? setting.axes.toUpperCase() : "6D rotation";
  return `${setting.target_label} · ${setting.type} · ${setting.density} · ${space}`;
}

function generatedRows(setting) {
  return setting.methods.filter((row) => !row.is_reference);
}

function filteredSettings() {
  return state.data.settings.filter((setting) =>
    (state.type === "all" || setting.type === state.type)
    && (state.density === "all" || setting.density === state.density)
    && (state.family === "all" || setting.family === state.family));
}

function selectedSetting() {
  return state.data.settings.find((setting) => setting.id === state.settingId) ?? filteredSettings()[0];
}

function options(values, allLabel) {
  return [`<option value="all">${allLabel}</option>`, ...values.map(([value, label]) =>
    `<option value="${safe(value)}">${safe(label)}</option>`)].join("");
}

function configureFilters() {
  const types = [["rotation", "Rotation"], ["position", "Position"]];
  const densities = [["sparse", "Sparse"], ["dense", "Dense"]];
  const families = Array.from(new Map(state.data.settings.map((setting) => [setting.family, setting.family])).entries())
    .map(([value]) => [value, value.charAt(0).toUpperCase() + value.slice(1)]);
  document.getElementById("type-filter").innerHTML = options(types, "All control types");
  document.getElementById("density-filter").innerHTML = options(densities, "All densities");
  document.getElementById("family-filter").innerHTML = options(families, "All body targets");
  document.getElementById("metric-select").innerHTML = METRICS.filter((metric) => !metric.referenceStatistic)
    .map((metric) => `<option value="${metric.key}">${metric.label}</option>`).join("");
  document.getElementById("metric-select").value = state.metric;
}

function syncSettingSelect() {
  const settings = filteredSettings();
  if (!settings.some((setting) => setting.id === state.settingId)) state.settingId = settings[0]?.id ?? null;
  const select = document.getElementById("setting-select");
  select.innerHTML = settings.map((setting) => `<option value="${safe(setting.id)}">${safe(settingLabel(setting))}</option>`).join("");
  select.value = state.settingId ?? "";
}

function rankableValues(setting, metric) {
  return generatedRows(setting)
    .map((row) => row.metrics[metric.key])
    .filter((value) => Number.isFinite(Number(value)))
    .sort((a, b) => metric.lower ? a - b : b - a)
    .filter((value, index, values) => index === 0 || Math.abs(value - values[index - 1]) > 1e-8);
}

function resultClass(setting, row, metric) {
  if (row.is_reference || metric.referenceStatistic) return "";
  const values = rankableValues(setting, metric);
  const value = row.metrics[metric.key];
  if (!Number.isFinite(Number(value))) return "";
  if (Math.abs(value - values[0]) < 1e-8) return "best";
  if (values.length > 1 && Math.abs(value - values[1]) < 1e-8) return "second";
  return "";
}

function format(value, metric) {
  if (!Number.isFinite(Number(value))) return "–";
  return Number(value).toFixed(metric.digits);
}

function sortedGenerated(setting, metric) {
  return generatedRows(setting).slice().sort((a, b) => {
    const av = a.metrics[metric.key];
    const bv = b.metrics[metric.key];
    if (!Number.isFinite(Number(av))) return 1;
    if (!Number.isFinite(Number(bv))) return -1;
    return metric.lower ? av - bv : bv - av;
  });
}

function renderTable() {
  const setting = selectedSetting();
  if (!setting) return;
  const active = metricSpec(state.metric);
  const generated = sortedGenerated(setting, active);
  const rows = [setting.methods.find((row) => row.is_reference), ...generated].filter(Boolean);
  document.getElementById("table-head").innerHTML = `<tr><th>Method</th><th>Protocol</th>${METRICS.map((metric) => `<th>${metric.label}${metric.lower ? " ↓" : metric.higher ? " ↑" : ""}</th>`).join("")}<th>N</th></tr>`;
  document.getElementById("table-body").innerHTML = rows.map((row, index) => {
    const rank = row.is_reference ? "GT" : String(index);
    return `<tr class="${row.is_reference ? "reference-row" : ""}">
      <td><div class="method-cell"><span class="rank ${row.is_reference ? "reference" : ""}">${rank}</span><span><strong>${safe(row.method)}</strong><small>${safe(row.method_id)}</small></span></div></td>
      <td>${safe(row.protocol_status)}</td>
      ${METRICS.map((metric) => `<td class="${resultClass(setting, row, metric)}">${format(row.metrics[metric.key], metric)}</td>`).join("")}
      <td>${Number(row.artifacts.count).toLocaleString("en-US")}</td>
    </tr>`;
  }).join("");
  document.getElementById("results-title").textContent = settingLabel(setting);
  document.getElementById("results-subtitle").textContent = `${setting.id} · exact protocol comparison`;
  document.getElementById("result-count").textContent = `${generated.length} generated + GT`;
}

function normalizedScore(row, rows, metric) {
  const values = rows.map((item) => item.metrics[metric.key]).filter((value) => Number.isFinite(Number(value)));
  const value = row.metrics[metric.key];
  if (!Number.isFinite(Number(value)) || !values.length) return 0;
  const minimum = Math.min(...values);
  const maximum = Math.max(...values);
  if (Math.abs(maximum - minimum) < 1e-8) return 100;
  return metric.lower ? 100 * (maximum - value) / (maximum - minimum) : 100 * (value - minimum) / (maximum - minimum);
}

function syncMethodPicker(setting) {
  const names = generatedRows(setting).map((row) => row.method);
  state.selectedMethods = new Set(Array.from(state.selectedMethods).filter((name) => names.includes(name)));
  if (!state.selectedMethods.size) names.slice(0, 4).forEach((name) => state.selectedMethods.add(name));
  const picker = document.getElementById("method-picker");
  picker.innerHTML = names.map((name) => `<button type="button" data-method="${safe(name)}" aria-pressed="${state.selectedMethods.has(name)}">${safe(name)}</button>`).join("");
  picker.querySelectorAll("button").forEach((button) => button.addEventListener("click", () => {
    const name = button.dataset.method;
    if (state.selectedMethods.has(name)) state.selectedMethods.delete(name);
    else {
      if (state.selectedMethods.size >= 4) state.selectedMethods.delete(state.selectedMethods.values().next().value);
      state.selectedMethods.add(name);
    }
    syncMethodPicker(setting);
    renderCharts();
  }));
}

function renderCharts() {
  const setting = selectedSetting();
  if (!setting || typeof Chart === "undefined") return;
  const active = metricSpec(state.metric);
  const ranked = sortedGenerated(setting, active).filter((row) => Number.isFinite(Number(row.metrics[active.key])));
  document.getElementById("bar-title").textContent = `${active.label} · ${settingLabel(setting)}`;
  document.getElementById("bar-note").textContent = `${active.lower ? "Lower" : "Higher"} is better; generated methods only.`;
  barChart?.destroy();
  barChart = new Chart(document.getElementById("bar-chart"), {
    type: "bar",
    data: {labels: ranked.map((row) => row.method), datasets: [{data: ranked.map((row) => row.metrics[active.key]), backgroundColor: ranked.map((_, index) => index === 0 ? "#087d72" : index === 1 ? "#315f9d" : "#8ebfb6"), borderRadius: 3}]},
    options: {animation: false, indexAxis: "y", responsive: true, maintainAspectRatio: false, plugins: {legend: {display: false}}, scales: {x: {beginAtZero: true, grid: {color: "#e5ebe8"}}, y: {grid: {display: false}, ticks: {color: "#34413d", font: {size: 11, weight: "600"}}}}},
  });

  const generated = generatedRows(setting);
  const selected = generated.filter((row) => state.selectedMethods.has(row.method));
  const radarMetrics = METRICS.filter((metric) => ["r_precision_top3", "fid", "mm_dist", "control_error", "foot_skating"].includes(metric.key));
  radarChart?.destroy();
  radarChart = new Chart(document.getElementById("radar-chart"), {
    type: "radar",
    data: {labels: radarMetrics.map((metric) => metric.label), datasets: selected.map((row, index) => ({label: row.method, data: radarMetrics.map((metric) => normalizedScore(row, generated, metric)), borderColor: COLORS[index % COLORS.length], backgroundColor: `${COLORS[index % COLORS.length]}18`, borderWidth: 2, pointRadius: 2}))},
    options: {animation: false, responsive: true, maintainAspectRatio: false, scales: {r: {min: 0, max: 100, ticks: {display: false}, pointLabels: {color: "#46534f", font: {size: 11, weight: "600"}}, grid: {color: "#dfe6e2"}, angleLines: {color: "#dfe6e2"}}}, plugins: {legend: {position: "bottom", labels: {boxWidth: 11, boxHeight: 11, padding: 12}}}},
  });
}

function renderSummary() {
  const methods = new Set(state.data.settings.flatMap((setting) => generatedRows(setting).map((row) => row.method)));
  const measuredRows = state.data.settings.reduce((total, setting) => total + generatedRows(setting).length, 0);
  document.getElementById("setting-count").textContent = state.data.num_settings;
  document.getElementById("method-count").textContent = methods.size;
  document.getElementById("case-count").textContent = Number(state.data.num_cases).toLocaleString("en-US");
  document.getElementById("row-count").textContent = measuredRows;
  document.getElementById("snapshot").textContent = `${state.data.num_settings} settings · ${measuredRows} measured rows · GT reference included`;
  document.getElementById("updated").textContent = state.data.updated;
}

function renderAll() {
  syncSettingSelect();
  const setting = selectedSetting();
  if (!setting) return;
  state.settingId = setting.id;
  syncMethodPicker(setting);
  renderTable();
  renderCharts();
  history.replaceState(null, "", `?setting=${encodeURIComponent(setting.id)}`);
}

async function initialize() {
  const response = await fetch("body_part_condition_results.json", {cache: "no-cache"});
  if (!response.ok) throw new Error(`Results ${response.status}`);
  state.data = await response.json();
  const requested = new URLSearchParams(location.search).get("setting");
  const preferred = "pos_wrist_left_sparse_xyz";
  state.settingId = state.data.settings.some((setting) => setting.id === requested)
    ? requested
    : state.data.settings.some((setting) => setting.id === preferred)
      ? preferred
      : state.data.settings[0].id;
  configureFilters();
  renderSummary();
  renderAll();
  for (const [elementId, stateKey] of [["type-filter", "type"], ["density-filter", "density"], ["family-filter", "family"]]) {
    document.getElementById(elementId).addEventListener("change", (event) => {state[stateKey] = event.target.value; renderAll();});
  }
  document.getElementById("setting-select").addEventListener("change", (event) => {state.settingId = event.target.value; state.selectedMethods.clear(); renderAll();});
  document.getElementById("metric-select").addEventListener("change", (event) => {state.metric = event.target.value; renderTable(); renderCharts();});
}

initialize().catch((error) => {
  document.getElementById("snapshot").textContent = error.message;
  console.error(error);
});
