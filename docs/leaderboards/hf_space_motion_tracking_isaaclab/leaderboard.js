"use strict";

const COLORS = ["#087d72", "#315f9d", "#a5412e", "#956000"];
let data = null;
let activeSplit = null;
let activeMetric = "success_rate";
let barChart = null;
let radarChart = null;

const safe = (value) => String(value ?? "").replace(
  /[&<>"']/g,
  (char) => ({"&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;"})[char],
);
const splitInfo = () => data.splits.find((item) => item.id === activeSplit);
const metricInfo = (key) => data.metric_definitions.find((item) => item.key === key);
const measuredRows = () => data.rows.filter((row) => row.kind !== "reference" && row.splits[activeSplit]);
const rankedRows = () => measuredRows().filter((row) => row.rankable);
const metricValue = (row, key) => row.splits[activeSplit]?.metrics?.[key];

function displayValue(value, metric) {
  if (!Number.isFinite(Number(value))) return "–";
  if (metric.unit === "%") return `${(Number(value) * 100).toFixed(1)}%`;
  const digits = Math.abs(Number(value)) >= 100 ? 1 : Math.abs(Number(value)) >= 10 ? 2 : 3;
  return `${Number(value).toFixed(digits)}${metric.unit ? ` ${metric.unit}` : ""}`;
}

function orderedValues(metric) {
  const values = rankedRows().map((row) => metricValue(row, metric.key)).filter(Number.isFinite);
  values.sort((a, b) => metric.direction === "higher" ? b - a : a - b);
  return values.filter((value, index) => index === 0 || Math.abs(value - values[index - 1]) > 1e-10);
}

function resultClass(row, metric) {
  if (!row.rankable) return "";
  const values = orderedValues(metric);
  const value = metricValue(row, metric.key);
  if (Math.abs(value - values[0]) < 1e-10) return "best";
  if (values.length > 1 && Math.abs(value - values[1]) < 1e-10) return "second";
  return "";
}

function renderTable() {
  document.getElementById("table-head").innerHTML = `<tr><th>Method</th><th>Coverage</th>${data.metric_definitions.map((metric) => `<th>${safe(metric.label)} ${metric.direction === "higher" ? "↑" : "↓"}</th>`).join("")}</tr>`;
  document.getElementById("table-body").innerHTML = data.rows.map((row, index) => {
    const selected = row.splits[activeSplit];
    if (!selected) return "";
    const coverage = selected.coverage;
    const role = row.kind === "reference" ? "GT" : row.rankable ? String(index) : "UB";
    const roleText = row.kind === "reference" ? "Kinematic reference" : row.rankable ? "Physical controller" : "Per-reference upper bound · unranked";
    return `<tr class="${row.kind === "reference" ? "reference-row" : ""}">
      <td><div class="method-cell"><span class="rank ${row.kind === "reference" ? "reference" : ""}">${role}</span><span><strong>${safe(row.method)}</strong><small>${safe(roleText)}</small></span></div></td>
      <td>${coverage.evaluated.toLocaleString("en-US")} / ${coverage.population.toLocaleString("en-US")}</td>
      ${data.metric_definitions.map((metric) => `<td class="${resultClass(row, metric)}">${displayValue(selected.metrics[metric.key], metric)}</td>`).join("")}
    </tr>`;
  }).join("");
}

function normalized(row, rows, metric) {
  const values = rows.map((item) => metricValue(item, metric.key)).filter(Number.isFinite);
  const minimum = Math.min(...values);
  const maximum = Math.max(...values);
  if (Math.abs(maximum - minimum) < 1e-10) return 100;
  const value = metricValue(row, metric.key);
  return metric.direction === "higher"
    ? 100 * (value - minimum) / (maximum - minimum)
    : 100 * (maximum - value) / (maximum - minimum);
}

function chartNumber(value, metric) {
  return metric.unit === "%" ? value * 100 : value;
}

function renderCharts() {
  if (typeof Chart === "undefined") return;
  const metric = metricInfo(activeMetric);
  const rows = measuredRows().slice().sort((a, b) => {
    const av = metricValue(a, metric.key);
    const bv = metricValue(b, metric.key);
    return metric.direction === "higher" ? bv - av : av - bv;
  });
  document.getElementById("bar-title").textContent = `${metric.label}${metric.unit ? ` (${metric.unit})` : ""}`;
  document.getElementById("bar-note").textContent = `${metric.direction === "higher" ? "Higher" : "Lower"} is better; GT is excluded and upper bounds remain unranked.`;
  barChart?.destroy();
  barChart = new Chart(document.getElementById("bar-chart"), {
    type: "bar",
    data: {labels: rows.map((row) => row.method), datasets: [{data: rows.map((row) => chartNumber(metricValue(row, metric.key), metric)), backgroundColor: rows.map((row, index) => row.rankable ? COLORS[index % COLORS.length] : "#a9b2ae"), borderRadius: 3}]},
    options: {animation: false, indexAxis: "y", responsive: true, maintainAspectRatio: false, plugins: {legend: {display: false}}, scales: {x: {beginAtZero: true, grid: {color: "#e5ebe8"}}, y: {grid: {display: false}, ticks: {font: {size: 11, weight: "600"}, color: "#34413d"}}}},
  });

  const radarMetrics = data.metric_definitions.slice(0, 6);
  radarChart?.destroy();
  radarChart = new Chart(document.getElementById("radar-chart"), {
    type: "radar",
    data: {labels: radarMetrics.map((item) => item.label), datasets: rows.map((row, index) => ({label: `${row.method}${row.rankable ? "" : " · UB"}`, data: radarMetrics.map((item) => normalized(row, rows, item)), borderColor: COLORS[index % COLORS.length], backgroundColor: "transparent", fill: false, borderDash: row.rankable ? [] : [6, 4], borderWidth: 2, pointRadius: 2}))},
    options: {animation: false, responsive: true, maintainAspectRatio: false, scales: {r: {min: 0, max: 100, ticks: {display: false}, pointLabels: {font: {size: 11, weight: "600"}, color: "#46534f"}, grid: {color: "#dfe6e2"}, angleLines: {color: "#dfe6e2"}}}, plugins: {legend: {position: "bottom", labels: {boxWidth: 11, boxHeight: 3, padding: 12}}}},
  });
}

function bestRow(key) {
  const metric = metricInfo(key);
  return rankedRows().slice().sort((a, b) => metric.direction === "higher" ? metricValue(b, key) - metricValue(a, key) : metricValue(a, key) - metricValue(b, key))[0];
}

function renderSummary() {
  const split = splitInfo();
  const success = bestRow("success_rate");
  const mpjpe = bestRow("local_mpjpe_mm");
  document.getElementById("method-count").textContent = measuredRows().length;
  document.getElementById("best-success").textContent = displayValue(metricValue(success, "success_rate"), metricInfo("success_rate"));
  document.getElementById("best-success-note").textContent = success.method;
  document.getElementById("best-mpjpe").textContent = displayValue(metricValue(mpjpe, "local_mpjpe_mm"), metricInfo("local_mpjpe_mm"));
  document.getElementById("best-mpjpe-note").textContent = mpjpe.method;
  document.getElementById("population").textContent = split.population.toLocaleString("en-US");
  document.getElementById("population-note").textContent = split.label;
  document.getElementById("table-setting").textContent = `${split.label} · ${split.population} references`;
}

function renderStatic() {
  const splitTotal = data.splits.reduce((sum, item) => sum + item.population, 0);
  document.getElementById("subtitle").textContent = `${data.engine.name} ${data.engine.version} · Unitree G1 physical reference tracking with persisted measured rollouts.`;
  document.getElementById("badges").innerHTML = [data.engine.name, data.protocol_id, `${data.control_hz} Hz control`, `${splitTotal} references`, "Unitree G1 · 29 DOF"].map((value) => `<span class="badge">${safe(value)}</span>`).join("");
  document.getElementById("snapshot").textContent = `${data.rows.length - 1} measured methods · ${data.splits.length} split${data.splits.length > 1 ? "s" : ""} · GT included`;
  document.getElementById("viewer-coverage").textContent = `${splitTotal} cases · GT + 2 methods`;
  document.getElementById("updated").textContent = `Snapshot ${data.generated_at}`;
  document.getElementById("setting-label").value = `${data.engine.name} ${data.engine.version} · ${data.protocol_id}`;
  document.getElementById("protocol-engine").textContent = `${data.engine.name} ${data.engine.version}; Unitree G1; ${data.control_hz} Hz control. Results are ranked only inside this engine setting.`;
  document.getElementById("protocol-success").textContent = data.provenance.success || "A successful episode reaches the fixed horizon; completion reports the retained physical rollout fraction.";
  document.getElementById("protocol-aggregation").textContent = data.provenance.aggregation;
  document.getElementById("table-note-left").textContent = data.rows.some((row) => !row.rankable && row.kind !== "reference") ? "UB denotes the disclosed per-reference optimization upper bound." : "All controller rows cover the complete selected split.";
}

function renderSplit() {
  renderSummary();
  renderTable();
  renderCharts();
}

function syncViewer() {
  document.getElementById("case-viewer")?.contentWindow?.postMessage(
    {type: "motius:set-split", split: activeSplit},
    "*",
  );
}

async function initialize() {
  const response = await fetch("motion_tracking_results.json", {cache: "no-cache"});
  if (!response.ok) throw new Error(`Results ${response.status}`);
  data = await response.json();
  activeSplit = data.splits[0].id;
  const splitSelect = document.getElementById("split-select");
  splitSelect.innerHTML = data.splits.map((item) => `<option value="${safe(item.id)}">${safe(item.label)} · ${item.population}</option>`).join("");
  splitSelect.addEventListener("change", (event) => {activeSplit = event.target.value; renderSplit(); syncViewer();});
  const metricSelect = document.getElementById("metric-select");
  metricSelect.innerHTML = data.metric_definitions.map((metric) => `<option value="${safe(metric.key)}">${safe(metric.label)}</option>`).join("");
  metricSelect.value = activeMetric;
  metricSelect.addEventListener("change", (event) => {activeMetric = event.target.value; renderCharts();});
  renderStatic();
  renderSplit();
  document.getElementById("case-viewer").addEventListener("load", syncViewer);
  window.addEventListener("message", (event) => {
    if (event.source !== document.getElementById("case-viewer")?.contentWindow) return;
    if (event.data?.type !== "motius:viewer-split") return;
    if (!data.splits.some((item) => item.id === event.data.split)) return;
    activeSplit = event.data.split;
    splitSelect.value = activeSplit;
    renderSplit();
  });
}

initialize().catch((error) => {
  document.getElementById("snapshot").textContent = error.message;
  console.error(error);
});
