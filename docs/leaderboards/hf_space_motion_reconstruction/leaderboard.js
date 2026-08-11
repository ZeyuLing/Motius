"use strict";

const METRICS = [
  {key: "rfid", label: "rFID", digits: 4},
  {key: "embedding_l2", label: "Emb-L2", digits: 3},
  {key: "mpjpe_mm", label: "MPJPE mm", digits: 2},
  {key: "pa_mpjpe_mm", label: "PA-MPJPE mm", digits: 2},
  {key: "mpjre_deg", label: "MPJRE deg", digits: 2},
  {key: "slide", label: "Slide", digits: 2},
  {key: "float", label: "Float", digits: 2},
  {key: "penetration", label: "Penetration", digits: 2},
  {key: "jitter", label: "Jitter", digits: 2},
];
const RADAR_KEYS = ["rfid", "embedding_l2", "mpjpe_mm", "pa_mpjpe_mm", "mpjre_deg", "jitter"];
const COLORS = ["#087d72", "#315f9d", "#ad6d00", "#9b4b62", "#55752f", "#75529b", "#b04d2c", "#397b90", "#6f6754"];
let data = null;
let activeMetric = "rfid";
let barChart = null;
let radarChart = null;

const safe = (value) => String(value ?? "").replace(
  /[&<>"']/g,
  (char) => ({"&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;"})[char],
);

const format = (value, digits) => Number.isFinite(Number(value)) ? Number(value).toFixed(digits) : "–";
const generatedRows = () => data.rows.filter((row) => !row.is_reference);

function uniqueSortedValues(metric) {
  return generatedRows().map((row) => row[metric.key]).filter(Number.isFinite).sort((a, b) => a - b)
    .filter((value, index, values) => index === 0 || Math.abs(value - values[index - 1]) > 1e-8);
}

function resultClass(row, metric) {
  if (row.is_reference) return "";
  const values = uniqueSortedValues(metric);
  if (Math.abs(row[metric.key] - values[0]) < 1e-8) return "best";
  if (values.length > 1 && Math.abs(row[metric.key] - values[1]) < 1e-8) return "second";
  return "";
}

function renderTable() {
  document.getElementById("table-head").innerHTML = `<tr><th>Method</th>${METRICS.map((metric) => `<th>${metric.label} ↓</th>`).join("")}<th>N geom.</th><th>Scope</th></tr>`;
  document.getElementById("table-body").innerHTML = data.rows.map((row, index) => `<tr class="${row.is_reference ? "reference-row" : ""}">
    <td><div class="method-cell"><span class="rank ${row.is_reference ? "reference" : ""}">${row.is_reference ? "GT" : index}</span><span><strong>${safe(row.method)}</strong><small>${safe(row.version)}</small></span></div></td>
    ${METRICS.map((metric) => `<td class="${resultClass(row, metric)}">${format(row[metric.key], metric.digits)}</td>`).join("")}
    <td>${Number(row.geometry_samples).toLocaleString("en-US")}</td><td>${safe(row.physical_scope)}</td>
  </tr>`).join("");
}

function normalized(row, rows, metric) {
  const values = rows.map((item) => item[metric.key]).filter(Number.isFinite);
  const minimum = Math.min(...values);
  const maximum = Math.max(...values);
  if (Math.abs(maximum - minimum) < 1e-8) return 100;
  return 100 * (maximum - row[metric.key]) / (maximum - minimum);
}

function renderCharts() {
  if (typeof Chart === "undefined") return;
  const metric = METRICS.find((item) => item.key === activeMetric);
  const rows = generatedRows().slice().sort((a, b) => a[metric.key] - b[metric.key]);
  document.getElementById("bar-title").textContent = metric.label;
  document.getElementById("bar-note").textContent = "Lower is better; generated methods only.";
  barChart?.destroy();
  barChart = new Chart(document.getElementById("bar-chart"), {
    type: "bar",
    data: {labels: rows.map((row) => row.method), datasets: [{data: rows.map((row) => row[metric.key]), backgroundColor: rows.map((_, index) => index === 0 ? "#087d72" : index === 1 ? "#315f9d" : "#8ebfb6"), borderRadius: 3}]},
    options: {animation: false, indexAxis: "y", responsive: true, maintainAspectRatio: false, plugins: {legend: {display: false}}, scales: {x: {beginAtZero: true, grid: {color: "#e5ebe8"}}, y: {grid: {display: false}, ticks: {font: {size: 11, weight: "600"}, color: "#34413d"}}}},
  });

  const radarMetrics = RADAR_KEYS.map((key) => METRICS.find((metricItem) => metricItem.key === key));
  radarChart?.destroy();
  radarChart = new Chart(document.getElementById("radar-chart"), {
    type: "radar",
    data: {labels: radarMetrics.map((metricItem) => metricItem.label), datasets: rows.map((row, index) => {const color = COLORS[index % COLORS.length]; return {label: row.method, data: radarMetrics.map((metricItem) => normalized(row, rows, metricItem)), borderColor: color, backgroundColor: `${color}18`, borderWidth: 2, pointRadius: 2};})},
    options: {animation: false, responsive: true, maintainAspectRatio: false, scales: {r: {min: 0, max: 100, ticks: {display: false}, pointLabels: {font: {size: 11, weight: "600"}, color: "#46534f"}, grid: {color: "#dfe6e2"}, angleLines: {color: "#dfe6e2"}}}, plugins: {legend: {position: "bottom", labels: {boxWidth: 11, boxHeight: 11, padding: 12}}}},
  });
}

function renderSummary() {
  const rows = generatedRows();
  const bestRfid = rows.slice().sort((a, b) => a.rfid - b.rfid)[0];
  const bestPa = rows.slice().sort((a, b) => a.pa_mpjpe_mm - b.pa_mpjpe_mm)[0];
  document.getElementById("method-count").textContent = rows.length;
  document.getElementById("best-rfid").textContent = format(bestRfid.rfid, 4);
  document.getElementById("best-rfid-note").textContent = bestRfid.method;
  document.getElementById("best-pa").textContent = `${format(bestPa.pa_mpjpe_mm, 2)} mm`;
  document.getElementById("best-pa-note").textContent = bestPa.method;
  document.getElementById("geometry-count").textContent = Number(rows[0].geometry_samples).toLocaleString("en-US");
  document.getElementById("snapshot").textContent = `${rows.length} generated methods · complete results · GT reference included`;
  document.getElementById("updated").textContent = data.updated;
}

async function initialize() {
  const response = await fetch("reconstruction_results.json", {cache: "no-cache"});
  if (!response.ok) throw new Error(`Results ${response.status}`);
  data = await response.json();
  const select = document.getElementById("metric-select");
  select.innerHTML = METRICS.map((metric) => `<option value="${metric.key}">${metric.label}</option>`).join("");
  select.value = activeMetric;
  select.addEventListener("change", (event) => {activeMetric = event.target.value; renderCharts();});
  renderSummary();
  renderTable();
  renderCharts();
}

initialize().catch((error) => {
  document.getElementById("snapshot").textContent = error.message;
  console.error(error);
});
