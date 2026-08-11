const metricDefs = [
  { key: "bleu4", label: "BLEU-4", direction: "max" },
  { key: "rougeL", label: "ROUGE-L", direction: "max" },
  { key: "cider", label: "CIDEr", direction: "max" },
  { key: "bertRaw", label: "BERT raw", direction: "max" },
  { key: "bertF1", label: "BERT rescaled", direction: "max" },
  { key: "r1", label: "R@1", direction: "max" },
  { key: "r3", label: "R@3", direction: "max" },
  { key: "matching", label: "Matching", direction: "min" },
];

const colors = {
  tm2t: "#087d72",
  motiongpt: "#315f9d",
  motiongpt3: "#a5412e",
  vermo: "#956000",
};

let snapshot;
let selectedMetric = metricDefs[0];
let selectedReference = "token";

const formatValue = (value) => value == null || Number.isNaN(Number(value))
  ? "-"
  : Number(value).toFixed(3);

function metricsFor(method) {
  if (selectedReference === "raw" && method.rawReferenceMetrics) {
    return { ...method.metrics, ...method.rawReferenceMetrics };
  }
  return method.metrics || {};
}

function rankedMethods(key, direction) {
  return snapshot.methods
    .filter((method) => method.kind === "method" && method.status === "measured" && metricsFor(method)[key] != null)
    .sort((a, b) => direction === "min"
      ? metricsFor(a)[key] - metricsFor(b)[key]
      : metricsFor(b)[key] - metricsFor(a)[key]);
}

function rankingClass(method, key, direction) {
  if (method.kind !== "method") return "";
  const ranking = rankedMethods(key, direction);
  if (ranking[0]?.id === method.id) return "best";
  if (ranking[1]?.id === method.id) return "second";
  return "";
}

function renderSummary() {
  const benchmark = snapshot.benchmark;
  document.getElementById("sample-count").textContent = benchmark.num_samples.toLocaleString();
  document.getElementById("method-count").textContent = snapshot.methods.filter((item) => item.kind === "method").length;
  document.getElementById("reference-count").textContent = benchmark.reference_count;
  document.getElementById("chunk-size").textContent = benchmark.semantic_chunk_size;
  const measured = snapshot.methods.filter((item) => item.kind === "method" && item.status === "measured").length;
  document.getElementById("snapshot-status").textContent = measured === 4 ? "Complete metric snapshot" : `${measured}/4 baselines measured`;
  document.getElementById("snapshot-date").textContent = `Updated ${benchmark.updated_at} · ${benchmark.protocol}`;
}

function renderMetricTabs() {
  const tabs = document.getElementById("metric-tabs");
  tabs.replaceChildren(...metricDefs.map((metric) => {
    const button = document.createElement("button");
    button.className = "metric-tab";
    button.type = "button";
    button.role = "tab";
    button.textContent = metric.label;
    button.setAttribute("aria-selected", String(metric.key === selectedMetric.key));
    button.addEventListener("click", () => {
      selectedMetric = metric;
      renderMetricTabs();
      drawChart();
    });
    return button;
  }));
}

function renderReferenceTabs() {
  const tabs = document.getElementById("reference-tabs");
  const definitions = [
    { key: "token", label: "TM2T token refs" },
    { key: "raw", label: "Raw captions" },
  ];
  tabs.replaceChildren(...definitions.map((definition) => {
    const button = document.createElement("button");
    button.className = "reference-tab";
    button.type = "button";
    button.role = "tab";
    button.textContent = definition.label;
    button.setAttribute("aria-selected", String(definition.key === selectedReference));
    button.addEventListener("click", () => {
      selectedReference = definition.key;
      renderReferenceTabs();
      renderTable();
      drawChart();
    });
    return button;
  }));
}

function drawChart() {
  const canvas = document.getElementById("metric-chart");
  const empty = document.getElementById("chart-empty");
  const methods = rankedMethods(selectedMetric.key, selectedMetric.direction);
  const ratio = window.devicePixelRatio || 1;
  const width = Math.max(320, canvas.clientWidth);
  const height = Math.max(240, canvas.clientHeight);
  canvas.width = Math.round(width * ratio);
  canvas.height = Math.round(height * ratio);
  const ctx = canvas.getContext("2d");
  ctx.scale(ratio, ratio);
  ctx.clearRect(0, 0, width, height);
  document.getElementById("chart-direction").textContent = selectedMetric.direction === "min" ? "Lower is better" : "Higher is better";
  empty.hidden = methods.length > 0;
  const legend = document.getElementById("legend");
  legend.replaceChildren(...methods.map((method) => {
    const item = document.createElement("span");
    item.className = "legend-item";
    item.innerHTML = `<span class="legend-swatch" style="background:${colors[method.id]}"></span>${method.name}`;
    return item;
  }));
  if (!methods.length) return;

  const margin = { top: 18, right: 24, bottom: 48, left: 58 };
  const chartW = width - margin.left - margin.right;
  const chartH = height - margin.top - margin.bottom;
  const values = methods.map((method) => Number(metricsFor(method)[selectedMetric.key]));
  const max = Math.max(...values) * 1.12 || 1;
  ctx.font = '11px "IBM Plex Mono"';
  ctx.fillStyle = "#68756f";
  ctx.strokeStyle = "#d9e1dd";
  ctx.lineWidth = 1;
  for (let tick = 0; tick <= 4; tick += 1) {
    const value = max * tick / 4;
    const y = margin.top + chartH - chartH * tick / 4;
    ctx.beginPath(); ctx.moveTo(margin.left, y); ctx.lineTo(width - margin.right, y); ctx.stroke();
    ctx.textAlign = "right"; ctx.fillText(value.toFixed(2), margin.left - 9, y + 4);
  }
  const slot = chartW / methods.length;
  const barW = Math.min(96, slot * 0.56);
  methods.forEach((method, index) => {
    const value = Number(metricsFor(method)[selectedMetric.key]);
    const barH = chartH * value / max;
    const x = margin.left + slot * index + (slot - barW) / 2;
    const y = margin.top + chartH - barH;
    ctx.fillStyle = colors[method.id] || "#46534f";
    ctx.fillRect(x, y, barW, barH);
    ctx.fillStyle = "#17231f"; ctx.textAlign = "center";
    ctx.fillText(value.toFixed(3), x + barW / 2, Math.max(12, y - 7));
    ctx.fillText(method.name, x + barW / 2, margin.top + chartH + 25);
  });
}

function methodCell(method) {
  const links = [];
  if (method.paper) links.push(`<a href="${method.paper}" target="_blank" rel="noreferrer">paper</a>`);
  if (method.checkpoint) links.push(`<a href="${method.checkpoint}" target="_blank" rel="noreferrer">weights</a>`);
  return `<span class="method-name">${method.name}</span>${links.length ? `<span class="method-links">${links.join(" · ")}</span>` : ""}`;
}

function renderTable() {
  const body = document.getElementById("results-body");
  const columns = [
    ["bleu1", "max"], ["bleu4", "max"], ["rougeL", "max"], ["cider", "max"],
    ["bertRaw", "max"], ["bertF1", "max"], ["r1", "max"], ["r2", "max"], ["r3", "max"], ["matching", "min"],
  ];
  body.replaceChildren(...snapshot.methods.map((method) => {
    const row = document.createElement("tr");
    if (method.kind === "reference") row.className = "reference";
    const metrics = metricsFor(method);
    const metricCells = columns.map(([key, direction]) =>
      `<td class="${rankingClass(method, key, direction)}">${formatValue(metrics[key])}</td>`).join("");
    row.innerHTML = `<td>${methodCell(method)}</td><td><span class="status ${method.status}">${method.kind === "reference" ? "reference" : method.status}</span></td>${metricCells}`;
    return row;
  }));
  document.getElementById("table-protocol-note").textContent = selectedReference === "raw"
    ? "Raw HumanML3D captions; language metrics are a protocol-sensitivity diagnostic."
    : "TM2T token/lemma references; paper-compatible. GT and incomplete runs are not ranked.";
}

function renderAudit() {
  const grid = document.getElementById("audit-grid");
  if (!snapshot.qualitative?.length) {
    const empty = document.createElement("div");
    empty.className = "audit-empty";
    empty.textContent = "Qualitative records will appear after all four full-protocol runs complete.";
    grid.replaceChildren(empty);
    return;
  }
  grid.replaceChildren(...snapshot.qualitative.map((item) => {
    const article = document.createElement("article");
    article.className = "audit-case";
    const outputs = Object.entries(item.outputs).map(([method, caption]) =>
      `<div><strong>${method}</strong>${caption}</div>`).join("");
    article.innerHTML = `<div class="audit-id">${item.sample_id}</div><div class="audit-ref">${item.reference}</div><div class="audit-output">${outputs}</div>`;
    return article;
  }));
}

async function init() {
  const response = await fetch("m2t_results.json", { cache: "no-store" });
  if (!response.ok) throw new Error(`Unable to load results: ${response.status}`);
  snapshot = await response.json();
  renderSummary();
  renderReferenceTabs();
  renderMetricTabs();
  renderTable();
  renderAudit();
  drawChart();
}

window.addEventListener("resize", () => snapshot && drawChart());
init().catch((error) => {
  document.getElementById("snapshot-status").textContent = "Snapshot unavailable";
  document.getElementById("snapshot-date").textContent = error.message;
});
