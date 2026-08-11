"use strict";

const METRICS = [
  {key: "pa_mpjpe_mm", label: "PA-MPJPE", digits: 2},
  {key: "mpjpe_mm", label: "MPJPE", digits: 2},
  {key: "pve_mm", label: "PVE", digits: 2},
  {key: "accel_mps2", label: "Accel. error", digits: 3},
];
const COLORS = ["#087d72", "#315f9d", "#c7563f", "#ad6d00"];
let data = null;
let activeMetric = "pa_mpjpe_mm";
let barChart = null;
let radarChart = null;

const safe = (value) => String(value ?? "").replace(
  /[&<>"']/g,
  (char) => ({"&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;"})[char],
);
const format = (value, digits) => Number.isFinite(Number(value)) ? Number(value).toFixed(digits) : "–";
const generatedRows = () => data.rows.filter((row) => !row.reference);

function methodMetadata(name) {
  return data.methods.find((method) => method.method === name);
}

function renderDemos() {
  document.getElementById("demo-grid").innerHTML = data.methods.map((method) => `<article class="demo-card">
    <div class="demo-head"><h3>${safe(method.method)}</h3><span>${safe(method.body_model)}</span></div>
    <video src="${safe(method.demo.video)}" controls playsinline preload="metadata"></video>
    <p>${safe(method.demo.source)}</p>
  </article>`).join("");
}

function renderTable() {
  document.getElementById("table-body").innerHTML = data.rows.map((row) => {
    const metadata = methodMetadata(row.method);
    const name = metadata?.model_card
      ? `<a href="${safe(metadata.model_card)}" target="_blank" rel="noopener noreferrer">${safe(row.method)}</a>`
      : safe(row.method);
    return `<tr class="${row.reference ? "reference-row" : ""}">
      <td><div class="method-cell"><span class="rank ${row.reference ? "reference" : ""}">${row.reference ? "GT" : "·"}</span><span><strong>${name}</strong><small>${safe(row.body_model)} · ${safe(row.version)}</small></span></div></td>
      <td>${safe(row.input_protocol)}</td><td>${format(row.coverage_percent, 2)}%</td>
      ${METRICS.map((metric) => `<td>${format(row.metrics[metric.key], metric.digits)}</td>`).join("")}
    </tr>`;
  }).join("");
}

function normalized(row, rows, metric) {
  const values = rows.map((item) => item.metrics[metric.key]).filter((value) => Number.isFinite(Number(value)));
  const value = row.metrics[metric.key];
  if (!Number.isFinite(Number(value)) || !values.length) return 0;
  const minimum = Math.min(...values);
  const maximum = Math.max(...values);
  if (Math.abs(maximum - minimum) < 1e-8) return 100;
  return 100 * (maximum - value) / (maximum - minimum);
}

function renderCharts() {
  if (typeof Chart === "undefined") return;
  const metric = METRICS.find((item) => item.key === activeMetric);
  const rows = generatedRows().filter((row) => Number.isFinite(Number(row.metrics[metric.key])))
    .slice().sort((a, b) => a.metrics[metric.key] - b.metrics[metric.key]);
  document.getElementById("bar-title").textContent = metric.label;
  barChart?.destroy();
  barChart = new Chart(document.getElementById("bar-chart"), {
    type: "bar",
    data: {labels: rows.map((row) => row.method), datasets: [{data: rows.map((row) => row.metrics[metric.key]), backgroundColor: rows.map((row) => row.input_protocol.includes("full-frame") ? "#315f9d" : "#087d72"), borderRadius: 3}]},
    options: {animation: false, indexAxis: "y", responsive: true, maintainAspectRatio: false, plugins: {legend: {display: false}}, scales: {x: {beginAtZero: true, grid: {color: "#e5ebe8"}}, y: {grid: {display: false}, ticks: {font: {size: 11, weight: "600"}, color: "#34413d"}}}},
  });

  const radarRows = generatedRows();
  const radarMetrics = METRICS.filter((metricItem) => radarRows.some((row) => Number.isFinite(Number(row.metrics[metricItem.key]))));
  radarChart?.destroy();
  radarChart = new Chart(document.getElementById("radar-chart"), {
    type: "radar",
    data: {labels: radarMetrics.map((metricItem) => metricItem.label), datasets: radarRows.map((row, index) => ({label: row.method, data: radarMetrics.map((metricItem) => normalized(row, radarRows, metricItem)), borderColor: COLORS[index], backgroundColor: `${COLORS[index]}14`, borderWidth: 2, pointRadius: 2}))},
    options: {animation: false, responsive: true, maintainAspectRatio: false, scales: {r: {min: 0, max: 100, ticks: {display: false}, pointLabels: {font: {size: 11, weight: "600"}, color: "#46534f"}, grid: {color: "#dfe6e2"}, angleLines: {color: "#dfe6e2"}}}, plugins: {legend: {position: "bottom", labels: {boxWidth: 11, boxHeight: 11, padding: 12}}}},
  });
}

function renderSummary() {
  const methods = data.methods;
  const targetCrop = data.rows.filter((row) => !row.reference && row.input_protocol === "official target crop").length;
  document.getElementById("method-count").textContent = methods.length;
  document.getElementById("crop-count").textContent = targetCrop;
  document.getElementById("full-frame-count").textContent = methods.length - targetCrop;
  document.getElementById("video-count").textContent = methods.filter((method) => method.demo?.video).length;
  document.getElementById("snapshot").textContent = `${methods.length} verified methods · ${methods.length} videos · GT reference included`;
  document.getElementById("updated").textContent = data.updated;
}

async function initialize() {
  const response = await fetch("monocular_capture_results.json", {cache: "no-cache"});
  if (!response.ok) throw new Error(`Results ${response.status}`);
  data = await response.json();
  const select = document.getElementById("metric-select");
  select.innerHTML = METRICS.filter((metric) => generatedRows().some((row) => Number.isFinite(Number(row.metrics[metric.key]))))
    .map((metric) => `<option value="${metric.key}">${metric.label}</option>`).join("");
  select.value = activeMetric;
  select.addEventListener("change", (event) => {activeMetric = event.target.value; renderCharts();});
  renderSummary();
  renderCharts();
  renderDemos();
  renderTable();
}

initialize().catch((error) => {
  document.getElementById("snapshot").textContent = error.message;
  console.error(error);
});
