"use strict";

const METRICS = [
  {key: "utmr_r3", label: "uTMR R@3", digits: 4, higher: true},
  {key: "utmr_r1", label: "uTMR R@1", digits: 4, higher: true},
  {key: "utmr_m2m", label: "uTMR M2M", digits: 4, higher: true},
  {key: "mpjpe_cm", label: "MPJPE cm", digits: 3, higher: false},
  {key: "gb_mpjpe_cm", label: "GB-MPJPE cm", digits: 3, higher: false},
  {key: "rte_m", label: "RTE m", digits: 4, higher: false},
  {key: "accel_error", label: "Accel error", digits: 3, higher: false},
  {key: "jitter", label: "Jitter", digits: 3, higher: false},
  {key: "skating_ratio", label: "Skating", digits: 4, higher: false},
  {key: "penetration_ratio", label: "Penetration", digits: 4, higher: false},
];
const RADAR_KEYS = ["utmr_r3", "utmr_m2m", "mpjpe_cm", "gb_mpjpe_cm", "accel_error", "jitter"];
const COLORS = ["#087d72", "#315f9d", "#956000"];
let data = null;
let activeMetric = "utmr_r3";
let barChart = null;
let radarChart = null;

const safe = (value) => String(value ?? "").replace(
  /[&<>"']/g,
  (char) => ({"&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;"})[char],
);
const format = (value, digits) => Number.isFinite(Number(value)) ? Number(value).toFixed(digits) : "–";
const generatedRows = () => data.rows.filter((row) => !row.is_reference && !row.is_input);
const metricSort = (metric) => (a, b) => metric.higher ? b[metric.key] - a[metric.key] : a[metric.key] - b[metric.key];

function uniqueSortedValues(metric) {
  return generatedRows().map((row) => row[metric.key]).filter(Number.isFinite).sort(metric.higher ? (a, b) => b - a : (a, b) => a - b)
    .filter((value, index, values) => index === 0 || Math.abs(value - values[index - 1]) > 1e-8);
}

function resultClass(row, metric) {
  if (row.is_reference || row.is_input) return "";
  const values = uniqueSortedValues(metric);
  if (Math.abs(row[metric.key] - values[0]) < 1e-8) return "best";
  if (values.length > 1 && Math.abs(row[metric.key] - values[1]) < 1e-8) return "second";
  return "";
}

function renderTable() {
  const displayed = [
    ...data.rows.filter((row) => row.is_reference),
    ...data.rows.filter((row) => row.is_input),
    ...generatedRows().slice().sort(metricSort(METRICS[0])),
  ];
  document.getElementById("table-head").innerHTML = `<tr><th>Method</th><th>Support track</th>${METRICS.map((metric) => `<th>${metric.label} ${metric.higher ? "↑" : "↓"}</th>`).join("")}<th>N</th></tr>`;
  let rank = 0;
  document.getElementById("table-body").innerHTML = displayed.map((row) => {
    if (!row.is_reference && !row.is_input) rank += 1;
    const marker = row.is_reference ? "GT" : row.is_input ? "IN" : rank;
    return `<tr class="${row.is_reference || row.is_input ? "reference-row" : ""}">
      <td><div class="method-cell"><span class="rank ${row.is_reference || row.is_input ? "reference" : ""}">${marker}</span><span><strong>${safe(row.method)}</strong><small>${safe(row.version)}</small></span></div></td>
      <td>${safe(row.support)}</td>
      ${METRICS.map((metric) => `<td class="${resultClass(row, metric)}">${format(row[metric.key], metric.digits)}</td>`).join("")}
      <td>${Number(row.samples).toLocaleString("en-US")}</td>
    </tr>`;
  }).join("");
}

function normalized(row, rows, metric) {
  const values = rows.map((item) => item[metric.key]).filter(Number.isFinite);
  const minimum = Math.min(...values);
  const maximum = Math.max(...values);
  if (Math.abs(maximum - minimum) < 1e-8) return 100;
  return metric.higher
    ? 100 * (row[metric.key] - minimum) / (maximum - minimum)
    : 100 * (maximum - row[metric.key]) / (maximum - minimum);
}

function renderCharts() {
  if (typeof Chart === "undefined") return;
  const metric = METRICS.find((item) => item.key === activeMetric);
  const rows = generatedRows().slice().sort(metricSort(metric));
  document.getElementById("bar-title").textContent = metric.label;
  document.getElementById("bar-note").textContent = `${metric.higher ? "Higher" : "Lower"} is better; reference rows are excluded.`;
  barChart?.destroy();
  barChart = new Chart(document.getElementById("bar-chart"), {
    type: "bar",
    data: {labels: rows.map((row) => row.method), datasets: [{data: rows.map((row) => row[metric.key]), backgroundColor: rows.map((_, index) => index === 0 ? "#087d72" : index === 1 ? "#315f9d" : "#8ebfb6"), borderRadius: 3}]},
    options: {animation: false, indexAxis: "y", responsive: true, maintainAspectRatio: false, plugins: {legend: {display: false}}, scales: {x: {beginAtZero: true, grid: {color: "#e5ebe8"}}, y: {grid: {display: false}, ticks: {font: {size: 12, weight: "600"}, color: "#34413d"}}}},
  });

  const radarMetrics = RADAR_KEYS.map((key) => METRICS.find((metricItem) => metricItem.key === key));
  radarChart?.destroy();
  radarChart = new Chart(document.getElementById("radar-chart"), {
    type: "radar",
    data: {labels: radarMetrics.map((metricItem) => metricItem.label), datasets: rows.map((row, index) => {const color = COLORS[index % COLORS.length]; return {label: row.method, data: radarMetrics.map((metricItem) => normalized(row, rows, metricItem)), borderColor: color, backgroundColor: `${color}18`, borderWidth: 2, pointRadius: 2};})},
    options: {animation: false, responsive: true, maintainAspectRatio: false, scales: {r: {min: 0, max: 100, ticks: {display: false}, pointLabels: {font: {size: 12, weight: "600"}, color: "#46534f"}, grid: {color: "#dfe6e2"}, angleLines: {color: "#dfe6e2"}}}, plugins: {legend: {position: "bottom", labels: {boxWidth: 11, boxHeight: 11, padding: 14, font: {size: 12}}}}},
  });
}

function renderSummary() {
  const rows = generatedRows();
  const bestR3 = rows.slice().sort(metricSort(METRICS[0]))[0];
  const mpjpe = METRICS.find((metric) => metric.key === "mpjpe_cm");
  const bestMpjpe = rows.slice().sort(metricSort(mpjpe))[0];
  document.getElementById("method-count").textContent = rows.length;
  document.getElementById("best-r3").textContent = format(bestR3.utmr_r3, 4);
  document.getElementById("best-r3-note").textContent = bestR3.method;
  document.getElementById("best-mpjpe").textContent = `${format(bestMpjpe.mpjpe_cm, 3)} cm`;
  document.getElementById("best-mpjpe-note").textContent = bestMpjpe.method;
  document.getElementById("snapshot").textContent = `${rows.length} repair methods · 299 paired cases · GT and source included`;
  document.getElementById("updated").textContent = data.updated;
}

async function initialize() {
  const response = await fetch("motion_repair_results.json", {cache: "no-cache"});
  if (!response.ok) throw new Error(`Results ${response.status}`);
  data = await response.json();
  const select = document.getElementById("metric-select");
  select.innerHTML = METRICS.map((metric) => `<option value="${metric.key}">${metric.label}</option>`).join("");
  select.value = activeMetric;
  select.addEventListener("change", (event) => {
    activeMetric = event.target.value;
    renderCharts();
  });
  renderSummary();
  renderTable();
  renderCharts();
}

initialize().catch((error) => {
  document.getElementById("snapshot").textContent = error.message;
  console.error(error);
});
