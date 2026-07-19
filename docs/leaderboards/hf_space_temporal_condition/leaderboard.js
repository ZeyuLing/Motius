"use strict";

const TP2M_ROWS = Object.freeze([
  {method: "GT", settingId: "c1", settingLabel: "1-frame prefix", group: "c1", samples: 4042, r1: 0.7703, r2: 0.9030, r3: 0.9442, fid: 0.0000, mmDist: 14.8785, diversity: 27.7705, isReference: true},
  {method: "FlowMDM", settingId: "c1", settingLabel: "1-frame prefix", group: "c1", samples: 3968, r1: 0.4490, r2: 0.6300, r3: 0.7060, fid: 83.7730, mmDist: 19.8720, diversity: 26.3650},
  {method: "MotionStreamer", settingId: "c1", settingLabel: "1-frame prefix", group: "c1", samples: 3904, r1: 0.6170, r2: 0.7800, r3: 0.8500, fid: 12.4790, mmDist: 16.8490, diversity: 27.1340},
  {method: "KIMODO", settingId: "c1", settingLabel: "1-frame prefix", group: "c1", samples: 3968, r1: 0.5250, r2: 0.6900, r3: 0.7690, fid: 82.5600, mmDist: 19.3010, diversity: 26.1580},
  {method: "PRISM-KT", settingId: "c1", settingLabel: "1-frame prefix", group: "c1", samples: 4042, r1: 0.7798, r2: 0.9087, r3: 0.9482, fid: 16.2756, mmDist: 14.9304, diversity: 27.5360},
  {method: "GT", settingId: "c5", settingLabel: "5-frame prefix", group: "c5", samples: 4042, r1: 0.7703, r2: 0.9030, r3: 0.9442, fid: 0.0000, mmDist: 14.8785, diversity: 27.7705, isReference: true},
  {method: "FlowMDM", settingId: "c5", settingLabel: "5-frame prefix", group: "c5", samples: 3968, r1: 0.4810, r2: 0.6540, r3: 0.7290, fid: 75.8530, mmDist: 19.4560, diversity: 26.4670},
  {method: "MotionStreamer", settingId: "c5", settingLabel: "5-frame prefix", group: "c5", samples: 3904, r1: 0.6280, r2: 0.7860, r3: 0.8530, fid: 11.2140, mmDist: 16.5860, diversity: 27.1440},
  {method: "KIMODO", settingId: "c5", settingLabel: "5-frame prefix", group: "c5", samples: 3968, r1: 0.5380, r2: 0.6990, r3: 0.7750, fid: 80.3810, mmDist: 19.1990, diversity: 26.1540},
  {method: "PRISM-KT", settingId: "c5", settingLabel: "5-frame prefix", group: "c5", samples: 4042, r1: 0.7912, r2: 0.9107, r3: 0.9489, fid: 13.5449, mmDist: 14.7900, diversity: 27.4694},
  {method: "GT", settingId: "c9", settingLabel: "9-frame prefix", group: "c9", samples: 4042, r1: 0.7703, r2: 0.9030, r3: 0.9442, fid: 0.0000, mmDist: 14.8785, diversity: 27.7705, isReference: true},
  {method: "FlowMDM", settingId: "c9", settingLabel: "9-frame prefix", group: "c9", samples: 3968, r1: 0.4900, r2: 0.6640, r3: 0.7420, fid: 71.3380, mmDist: 19.2620, diversity: 26.6250},
  {method: "MotionStreamer", settingId: "c9", settingLabel: "9-frame prefix", group: "c9", samples: 3904, r1: 0.6330, r2: 0.7880, r3: 0.8560, fid: 11.0770, mmDist: 16.4860, diversity: 27.3810},
  {method: "KIMODO", settingId: "c9", settingLabel: "9-frame prefix", group: "c9", samples: 3968, r1: 0.5310, r2: 0.7040, r3: 0.7720, fid: 79.1220, mmDist: 19.1660, diversity: 26.2020},
  {method: "PRISM-KT", settingId: "c9", settingLabel: "9-frame prefix", group: "c9", samples: 4042, r1: 0.7867, r2: 0.9144, r3: 0.9529, fid: 12.5467, mmDist: 14.7775, diversity: 27.5227}
]);

const COMMON_COLUMNS = [
  {key: "method", label: "Method", type: "method"},
  {key: "settingLabel", label: "Setting", type: "setting"},
  {key: "samples", label: "Samples", type: "integer"},
  {key: "r1", label: "R@1", higher: true, digits: 4},
  {key: "r2", label: "R@2", higher: true, digits: 4},
  {key: "r3", label: "R@3", higher: true, digits: 4},
  {key: "fid", label: "FID", lower: true, digits: 4},
  {key: "mmDist", label: "MM-Dist", lower: true, digits: 4},
  {key: "diversity", label: "Diversity", referenceStatistic: true, digits: 4}
];

const CONTROL_COLUMNS = [
  ...COMMON_COLUMNS.slice(0, 2),
  {key: "text", label: "Text", type: "boolean"},
  ...COMMON_COLUMNS.slice(2, 8),
  {key: "conditionError", label: "Cond. Err (cm)", lower: true, digits: 3},
  {key: "fail20", label: "Fail@20", lower: true, digits: 4},
  {key: "fail50", label: "Fail@50", lower: true, digits: 4},
  {key: "skate", label: "Skate", lower: true, digits: 4},
  COMMON_COLUMNS[8]
];

const CONTROL_TABLE_COLUMNS = CONTROL_COLUMNS.filter((column) => !["settingLabel", "text"].includes(column.key));

const PROTOCOLS = {
  control: {
    label: "Temporal Control",
    subtitle: "Prediction, motion in-betweening, and sparse-keyframe control on the HumanML3D official test split.",
    badges: ["Prediction / MIB / Keyframe", "4,012 temporal cases", "Selected captions", "Joint-position evaluator", "Normalized-space FID"],
    columns: CONTROL_COLUMNS,
    defaultSetting: "Prediction::pre20",
    groups: [
      {id: "all", label: "All tasks"},
      {id: "Prediction", label: "Prediction"},
      {id: "MIB", label: "Motion in-betweening"},
      {id: "Keyframe", label: "Keyframe"}
    ],
    groupLabel: "Task family",
    chartMetrics: ["r3", "fid", "mmDist", "conditionError", "fail20", "skate"]
  },
  tp2m: {
    label: "TP2M Prefix",
    subtitle: "Caption-guided continuation from a short ground-truth motion prefix, scored in MotionStreamer-272 space.",
    badges: ["1 / 5 / 9 prefix frames", "HumanML3D official test", "Selected captions", "MotionStreamer-272 evaluator"],
    columns: COMMON_COLUMNS,
    defaultSetting: "c9",
    groups: [
      {id: "all", label: "All prefixes"},
      {id: "c1", label: "1 frame"},
      {id: "c5", label: "5 frames"},
      {id: "c9", label: "9 frames"}
    ],
    groupLabel: "Prefix length",
    chartMetrics: ["r1", "r2", "r3", "fid", "mmDist"]
  }
};

const state = {
  protocol: "control",
  rowsByProtocol: {control: [], tp2m: Array.from(TP2M_ROWS)},
  activeGroup: "all",
  chartSetting: null,
  query: "",
  sortKey: "r3",
  sortDir: "desc",
  selectedMethods: new Set()
};

const chartColors = ["#087d72", "#315f9d", "#c7563f", "#ad6d00"];
let barChart = null;
let radarChart = null;

function renderCaseExplorer() {
  const section = document.getElementById("case-explorer");
  const select = document.getElementById("case-explorer-setting");
  section.hidden = state.protocol !== "control";
  if (section.hidden) return;
  const selectedRows = selectedSettingRows();
  const textSetting = selectedRows.find((row) => row.text && !row.isReference)?.settingId;
  if (textSetting) select.value = textSetting.replace(/^temporal_/, "");
  document.getElementById("case-explorer-frame").src = `cases/${select.value}/index.html`;
}

function isRankable(row) {
  return !row.isReference && !row.paperOnly && !row.rankExcluded;
}

function activeConfig() {
  return PROTOCOLS[state.protocol];
}

function activeRows() {
  return state.rowsByProtocol[state.protocol];
}

function activeColumns() {
  return activeConfig().columns;
}

function activeSettings() {
  const seen = new Map();
  activeRows().forEach((row) => {
    const id = state.protocol === "control" ? row.settingGroupId : row.settingId;
    const label = state.protocol === "control" ? row.settingGroupLabel : row.settingLabel;
    if (!seen.has(id)) seen.set(id, {id, label, group: row.group});
  });
  return Array.from(seen.values());
}

function selectableSettings() {
  return activeSettings().filter((setting) => state.activeGroup === "all" || setting.group === state.activeGroup);
}

function selectedSettingRows() {
  return activeRows().filter((row) => state.protocol === "control"
    ? row.settingGroupId === state.chartSetting
    : row.settingId === state.chartSetting);
}

function chartRows() {
  const rows = selectedSettingRows();
  if (state.protocol !== "control") return rows;
  const textRows = rows.filter((row) => row.text);
  return textRows.length ? textRows : rows;
}

function generatedMethods() {
  return Array.from(new Set(activeRows().filter(isRankable).map((row) => row.method)));
}

function controlRows(data) {
  const gt = data.gt_reference;
  return data.settings.flatMap((setting) => {
    const suffix = setting.text ? "Text" : "No text";
    const label = `${setting.task} · ${setting.constraint} · ${suffix}`;
    const settingGroupId = `${setting.task}::${setting.condition_mode}`;
    const settingGroupLabel = `${setting.task} · ${setting.constraint}`;
    const reference = {
      method: gt.method,
      settingId: setting.id,
      settingLabel: label,
      settingGroupId,
      settingGroupLabel,
      group: setting.task,
      text: setting.text,
      samples: gt.samples,
      temporalSamples: gt.temporal_samples,
      r1: gt.metrics.r_precision_top1,
      r2: gt.metrics.r_precision_top2,
      r3: gt.metrics.r_precision_top3,
      fid: gt.metrics.fid,
      mmDist: gt.metrics.mm_dist,
      conditionError: gt.metrics.constraint_error_cm,
      fail20: gt.metrics.fail_20,
      fail50: gt.metrics.fail_50,
      skate: gt.metrics.foot_skating,
      diversity: gt.metrics.diversity,
      isReference: true
    };
    const methods = setting.methods.map((entry) => ({
      method: entry.method,
      settingId: setting.id,
      settingLabel: label,
      settingGroupId,
      settingGroupLabel,
      group: setting.task,
      text: setting.text,
      samples: entry.samples,
      r1: entry.metrics.r_precision_top1,
      r2: entry.metrics.r_precision_top2,
      r3: entry.metrics.r_precision_top3,
      fid: entry.metrics.fid,
      mmDist: entry.metrics.mm_dist,
      conditionError: entry.metrics.constraint_error_cm,
      fail20: entry.metrics.fail_20,
      fail50: entry.metrics.fail_50,
      skate: entry.metrics.foot_skating,
      diversity: entry.metrics.diversity
    }));
    return [reference, ...methods];
  });
}

function rowId(row) {
  return `${row.method}::${row.settingId}`;
}

function formatCell(row, column) {
  const value = row[column.key];
  if (value === null || value === undefined) return "-";
  if (column.type === "integer") return Number(value).toLocaleString("en-US");
  if (column.type === "boolean") return value ? "On" : "Off";
  if (column.type === "setting") return `<span class="setting-pill">${value}</span>`;
  if (typeof value === "number") return value.toFixed(column.digits ?? 3);
  return String(value);
}

function rankBadge(row, rank) {
  if (row.isReference) return '<span class="rank rank-ref">REF</span>';
  return `<span class="rank">${rank ?? "-"}</span>`;
}

function methodCell(row, rank) {
  return `<div class="method">${rankBadge(row, rank)}<strong>${row.method}</strong></div>`;
}

function compareRows(a, b, key, direction) {
  if (a.isReference !== b.isReference) return a.isReference ? -1 : 1;
  const av = a[key];
  const bv = b[key];
  const aMissing = av === null || av === undefined;
  const bMissing = bv === null || bv === undefined;
  if (aMissing !== bMissing) return aMissing ? 1 : -1;
  let delta = typeof av === "number" && typeof bv === "number"
    ? av - bv
    : String(av).localeCompare(String(bv));
  if (direction === "desc") delta *= -1;
  if (delta !== 0) return delta;
  const settingDelta = a.settingLabel.localeCompare(b.settingLabel);
  return settingDelta || a.method.localeCompare(b.method);
}

function rankMetric() {
  return activeColumns().find((column) => column.key === state.sortKey && (column.higher || column.lower))
    || activeColumns().find((column) => column.key === "r3");
}

function settingRankMap() {
  const ranks = new Map();
  const metric = rankMetric();
  new Set(activeRows().map((row) => row.settingId)).forEach((settingId) => {
    activeRows()
      .filter((row) => row.settingId === settingId && isRankable(row) && typeof row[metric.key] === "number")
      .slice()
      .sort((a, b) => compareRows(a, b, metric.key, metric.lower ? "asc" : "desc"))
      .forEach((row, index) => ranks.set(rowId(row), index + 1));
  });
  return ranks;
}

function bestClass(row, column) {
  if (!isRankable(row) || (!column.lower && !column.higher) || typeof row[column.key] !== "number") return "";
  const values = activeRows()
    .filter((item) => item.settingId === row.settingId && isRankable(item) && typeof item[column.key] === "number")
    .map((item) => item[column.key])
    .sort((a, b) => column.lower ? a - b : b - a)
    .filter((value, index, array) => index === 0 || Math.abs(value - array[index - 1]) > 1e-8);
  if (Math.abs(row[column.key] - values[0]) < 1e-8) return "best";
  if (values.length > 1 && Math.abs(row[column.key] - values[1]) < 1e-8) return "second";
  return "";
}

function filteredRows(rows = activeRows()) {
  const query = state.query.trim().toLowerCase();
  return rows
    .filter((row) => state.activeGroup === "all" || row.group === state.activeGroup)
    .filter((row) => `${row.method} ${row.settingLabel}`.toLowerCase().includes(query))
    .slice()
    .sort((a, b) => compareRows(a, b, state.sortKey, state.sortDir));
}

function renderProtocolHeader() {
  const config = activeConfig();
  document.getElementById("protocol-subtitle").textContent = config.subtitle;
  document.getElementById("protocol-badges").innerHTML = config.badges
    .map((badge) => `<span class="badge">${badge}</span>`).join("");
  document.querySelectorAll("[data-protocol]").forEach((button) => {
    button.setAttribute("aria-selected", String(button.dataset.protocol === state.protocol));
  });
  document.getElementById("protocol-snapshot").textContent =
    `${activeRows().length} result rows · ${generatedMethods().length} generated methods · GT reference included`;
}

function renderGroupTabs() {
  const tabs = document.getElementById("group-tabs");
  tabs.setAttribute("aria-label", activeConfig().groupLabel);
  tabs.innerHTML = activeConfig().groups.map((group) =>
    `<button class="tab" type="button" role="tab" data-group="${group.id}" aria-selected="${group.id === state.activeGroup}">${group.label}</button>`
  ).join("");
  tabs.querySelectorAll("button").forEach((button) => {
    button.addEventListener("click", () => {
      state.activeGroup = button.dataset.group;
      const settings = selectableSettings();
      if (!settings.some((setting) => setting.id === state.chartSetting)) state.chartSetting = settings[0]?.id ?? null;
      renderGroupTabs();
      renderChartControls();
      renderTable();
      renderCharts();
    });
  });
}

function renderOneTable(table, rows, columns, ranks) {
  table.querySelector("thead").innerHTML = `<tr>${columns.map((column) => {
    const active = state.sortKey === column.key;
    const indicator = active ? (state.sortDir === "asc" ? "↑" : "↓") : "";
    const hint = column.lower ? "lower is better" : column.higher ? "higher is better" : column.referenceStatistic ? "reference statistic" : "";
    return `<th title="${hint}"><button class="sortable" type="button" data-key="${column.key}"><span>${column.label}</span><span>${indicator}</span></button></th>`;
  }).join("")}</tr>`;
  table.querySelector("tbody").innerHTML = rows.length ? rows.map((row) =>
    `<tr class="${row.isReference ? "reference-row" : ""}">${columns.map((column) => {
      if (column.type === "method") return `<td>${methodCell(row, ranks.get(rowId(row)))}</td>`;
      return `<td class="${bestClass(row, column)}">${formatCell(row, column)}</td>`;
    }).join("")}</tr>`
  ).join("") : `<tr><td class="empty" colspan="${columns.length}">No rows match the current filters.</td></tr>`;
  table.querySelectorAll("button.sortable").forEach((button) => {
    button.addEventListener("click", () => {
      const column = columns.find((item) => item.key === button.dataset.key);
      if (state.sortKey === button.dataset.key) state.sortDir = state.sortDir === "asc" ? "desc" : "asc";
      else {
        state.sortKey = button.dataset.key;
        state.sortDir = column?.higher ? "desc" : "asc";
      }
      renderTable();
    });
  });
}

function renderTable() {
  const ranks = settingRankMap();
  const pair = document.getElementById("control-table-pair");
  const single = document.getElementById("single-table-shell");
  if (state.protocol === "control") {
    pair.hidden = false;
    single.hidden = true;
    const selectedRows = filteredRows(selectedSettingRows());
    const textRows = selectedRows.filter((row) => row.text);
    const motionRows = selectedRows.filter((row) => !row.text);
    renderOneTable(document.querySelector('[data-table="control-text"]'), textRows, CONTROL_TABLE_COLUMNS, ranks);
    renderOneTable(document.querySelector('[data-table="control-motion"]'), motionRows, CONTROL_TABLE_COLUMNS, ranks);
    document.getElementById("result-status").textContent = `${textRows.length} text-conditioned · ${motionRows.length} motion-only rows`;
    return;
  }
  pair.hidden = true;
  single.hidden = false;
  const rows = filteredRows();
  renderOneTable(document.querySelector('[data-table="temporal"]'), rows, activeColumns(), ranks);
  document.getElementById("result-status").textContent = `${rows.length} of ${activeRows().length} rows`;
}

function bestRow(rows, key, lower) {
  return rows.filter((row) => isRankable(row) && typeof row[key] === "number")
    .slice().sort((a, b) => lower ? a[key] - b[key] : b[key] - a[key])[0];
}

function renderSummaries() {
  const rows = chartRows();
  const bestR3 = bestRow(rows, "r3", false);
  const bestFid = bestRow(rows, "fid", true);
  const setting = activeSettings().find((item) => item.id === state.chartSetting);
  document.getElementById("summary-r3").textContent = bestR3?.method ?? "-";
  document.getElementById("summary-r3-note").textContent = bestR3 ? `R@3 ${bestR3.r3.toFixed(4)} · GT excluded` : "-";
  document.getElementById("summary-fid").textContent = bestFid?.method ?? "-";
  document.getElementById("summary-fid-note").textContent = bestFid ? `FID ${bestFid.fid.toFixed(4)} · GT excluded` : "-";
  document.getElementById("summary-methods").textContent = generatedMethods().length;
  document.getElementById("summary-methods-note").textContent = generatedMethods().join(", ");
  document.getElementById("summary-settings").textContent = activeSettings().length;
  document.getElementById("summary-settings-note").textContent = setting?.label ?? "-";
}

function renderChartControls() {
  const settings = selectableSettings();
  const settingSelect = document.getElementById("chart-setting");
  settingSelect.innerHTML = settings.map((setting) => `<option value="${setting.id}">${setting.label}</option>`).join("");
  settingSelect.value = state.chartSetting;
  const columns = activeColumns();
  const metricSelect = document.getElementById("bar-metric");
  metricSelect.innerHTML = activeConfig().chartMetrics.map((key) => {
    const metric = columns.find((column) => column.key === key);
    return `<option value="${key}">${metric.label}</option>`;
  }).join("");
  metricSelect.value = activeConfig().chartMetrics.includes(state.sortKey) ? state.sortKey : "r3";
}

function renderMethodPicker() {
  const methods = generatedMethods();
  const picker = document.getElementById("method-picker");
  picker.innerHTML = methods.map((method) => {
    const checked = state.selectedMethods.has(method) ? " checked" : "";
    const disabled = state.selectedMethods.size >= 4 && !state.selectedMethods.has(method) ? " disabled" : "";
    return `<label class="method-option"><input type="checkbox" value="${method}"${checked}${disabled}><span>${method}</span></label>`;
  }).join("");
  picker.querySelectorAll("input").forEach((input) => {
    input.addEventListener("change", () => {
      if (input.checked) state.selectedMethods.add(input.value);
      else state.selectedMethods.delete(input.value);
      renderMethodPicker();
      renderCharts();
    });
  });
}

function chartFont() {
  return {family: "IBM Plex Sans", size: 12};
}

function normalizedScore(row, rows, metric) {
  const values = rows.filter(isRankable).map((item) => item[metric.key]).filter((value) => typeof value === "number");
  if (typeof row[metric.key] !== "number" || !values.length) return 0;
  const min = Math.min(...values);
  const max = Math.max(...values);
  if (Math.abs(max - min) < 1e-12) return 100;
  return metric.lower ? 100 * (max - row[metric.key]) / (max - min) : 100 * (row[metric.key] - min) / (max - min);
}

function renderCharts() {
  renderSummaries();
  if (typeof Chart === "undefined") {
    document.getElementById("bar-note").textContent = "Chart.js could not be loaded; the table remains available.";
    return;
  }
  const columns = activeColumns();
  const metricKey = document.getElementById("bar-metric").value || "r3";
  const metric = columns.find((column) => column.key === metricKey);
  const rows = chartRows();
  const ranked = rows.filter((row) => isRankable(row) && typeof row[metric.key] === "number")
    .slice().sort((a, b) => compareRows(a, b, metric.key, metric.lower ? "asc" : "desc"));
  const setting = activeSettings().find((item) => item.id === state.chartSetting);
  document.getElementById("bar-title").textContent = `${metric.label} · ${setting?.label ?? ""}`;
  const conditionNote = state.protocol === "control" ? " · text-conditioned chart; both variants are tabulated below" : "";
  document.getElementById("bar-note").textContent = `Generated methods only · ${metric.lower ? "lower" : "higher"} is better${conditionNote}`;
  if (barChart) barChart.destroy();
  barChart = new Chart(document.getElementById("bar-chart"), {
    type: "bar",
    data: {
      labels: ranked.map((row) => row.method),
      datasets: [{
        data: ranked.map((row) => row[metric.key]),
        backgroundColor: ranked.map((_, index) => index === 0 ? "#176b48" : index === 1 ? "#315f9d" : "#8abeb7"),
        borderWidth: 0,
        borderRadius: 3
      }]
    },
    options: {
      indexAxis: "y", responsive: true, maintainAspectRatio: false, animation: {duration: 240},
      plugins: {legend: {display: false}},
      scales: {
        x: {beginAtZero: true, grid: {color: "#e5e9e7"}, ticks: {font: chartFont(), color: "#5b6864"}},
        y: {grid: {display: false}, ticks: {font: chartFont(), color: "#26332f"}}
      }
    }
  });
  const radarMetrics = activeConfig().chartMetrics.map((key) => columns.find((column) => column.key === key));
  const profileRows = rows.filter((row) => isRankable(row) && state.selectedMethods.has(row.method));
  if (radarChart) radarChart.destroy();
  radarChart = new Chart(document.getElementById("radar-chart"), {
    type: "radar",
    data: {
      labels: radarMetrics.map((item) => item.label),
      datasets: profileRows.map((row, index) => ({
        label: row.method,
        data: radarMetrics.map((item) => normalizedScore(row, rows, item)),
        borderColor: chartColors[index % chartColors.length],
        backgroundColor: `${chartColors[index % chartColors.length]}18`,
        borderWidth: 2, pointRadius: 2
      }))
    },
    options: {
      responsive: true, maintainAspectRatio: false, animation: {duration: 240},
      plugins: {legend: {position: "bottom", labels: {font: chartFont(), boxWidth: 10, boxHeight: 10}}},
      scales: {r: {min: 0, max: 100, ticks: {display: false}, grid: {color: "#dfe5e2"}, angleLines: {color: "#dfe5e2"}, pointLabels: {font: chartFont(), color: "#46534f"}}}
    }
  });
}

function renderProtocolDetails() {
  const control = state.protocol === "control";
  document.getElementById("protocol-details").innerHTML = control ? `
    <section><h3>Task families</h3><dl><dt>Prediction</dt><dd>First-frame or prefix completion.</dd><dt>MIB</dt><dd>Generate between observed start and end frames.</dd><dt>Keyframe</dt><dd>Generate around adaptive sparse full-body keyframes.</dd></dl></section>
    <section><h3>Evaluation</h3><dl><dt>Semantic</dt><dd>Motius Joint-Position Evaluator; normalized-space FID.</dd><dt>Condition</dt><dd>Pelvis-relative SMPL-22 error on constrained frames.</dd><dt>Physical</dt><dd>Foot-skating ratio from one deterministic pass.</dd></dl></section>
    <section><h3>GT reference</h3><dl><dt>Semantic</dt><dd>Reused verbatim from the T2M HumanML3D GT row.</dd><dt>Temporal</dt><dd>Condition error and failure rates are zero by identity.</dd><dt>Ranking</dt><dd>GT is visible but excluded from all ranks and charts.</dd></dl></section>` : `
    <section><h3>Task</h3><dl><dt>Input</dt><dd>Selected caption plus the first N ground-truth frames.</dd><dt>Output</dt><dd>Continuation for the remaining sequence.</dd><dt>Prefixes</dt><dd>1, 5, and 9 frames.</dd></dl></section>
    <section><h3>Evaluation</h3><dl><dt>Split</dt><dd>HumanML3D official test split.</dd><dt>Evaluator</dt><dd>MotionStreamer-272 retrieval and distribution metrics.</dd><dt>FID</dt><dd>Computed against the same official GT distribution.</dd></dl></section>
    <section><h3>GT reference</h3><dl><dt>Source</dt><dd>Same GT row as the T2M HumanML3D Leaderboard.</dd><dt>Visibility</dt><dd>Repeated for each prefix to keep comparisons local.</dd><dt>Ranking</dt><dd>GT is visible but excluded from all ranks and charts.</dd></dl></section>`;
}

function resetProtocolState() {
  const config = activeConfig();
  state.activeGroup = "all";
  state.chartSetting = config.defaultSetting;
  state.query = "";
  state.sortKey = "r3";
  state.sortDir = "desc";
  state.selectedMethods = new Set(generatedMethods().slice(0, 4));
  document.getElementById("search").value = "";
  renderProtocolHeader();
  renderGroupTabs();
  renderChartControls();
  renderMethodPicker();
  renderProtocolDetails();
  renderTable();
  renderCharts();
  renderCaseExplorer();
}

async function initialize() {
  const response = await fetch("temporal_control_results.json");
  if (!response.ok) throw new Error(`Failed to load Temporal Control data: ${response.status}`);
  state.rowsByProtocol.control = controlRows(await response.json());
  document.querySelectorAll("[data-protocol]").forEach((button) => {
    button.addEventListener("click", () => {
      state.protocol = button.dataset.protocol;
      resetProtocolState();
    });
  });
  document.getElementById("search").addEventListener("input", (event) => {
    state.query = event.target.value;
    renderTable();
  });
  document.getElementById("chart-setting").addEventListener("change", (event) => {
    state.chartSetting = event.target.value;
    renderTable();
    renderCharts();
    renderCaseExplorer();
  });
  document.getElementById("case-explorer-setting").addEventListener("change", (event) => {
    document.getElementById("case-explorer-frame").src = `cases/${event.target.value}/index.html`;
  });
  document.getElementById("bar-metric").addEventListener("change", renderCharts);
  resetProtocolState();
}

initialize().catch((error) => {
  document.getElementById("result-status").textContent = error.message;
  console.error(error);
});
