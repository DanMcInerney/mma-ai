const qs = (selector) => document.querySelector(selector);
const qsa = (selector) => [...document.querySelectorAll(selector)];
let activeEventJobId = null;
let activeMatchupJobId = null;
let activeDataJobId = null;
let selectedUpcomingNumber = 1;
let upcomingEventsCache = [];
let predictModelsAvailable = false;
let selectedEventHasMatchedFights = false;

async function api(path, options = {}) {
  const response = await fetch(path, {
    headers: { "Content-Type": "application/json" },
    ...options,
  });
  if (!response.ok) {
    const body = await response.json().catch(() => ({ detail: response.statusText }));
    throw new Error(apiErrorMessage(body.detail) || apiErrorMessage(body.error) || response.statusText);
  }
  return response.json();
}

function apiErrorMessage(detail) {
  if (!detail) return "";
  if (typeof detail === "string") return detail;
  if (Array.isArray(detail)) {
    return detail
      .map((item) => apiErrorMessage(item))
      .filter(Boolean)
      .join("; ");
  }
  if (typeof detail === "object") {
    return detail.msg || detail.message || detail.error || JSON.stringify(detail);
  }
  return String(detail);
}

function renderJson(target, value) {
  qs(target).textContent = typeof value === "string" ? value : JSON.stringify(value, null, 2);
}

function chartTitle(chart, index) {
  const title = chart?.layout?.title;
  if (typeof title === "string" && title.trim()) return title;
  if (title?.text) return title.text;
  return `Chart ${index + 1}`;
}

function normalizeAnalyticsCharts(result) {
  if (Array.isArray(result?.charts)) return result.charts.filter((chart) => chart?.data && chart?.layout);
  return [];
}

function renderPlotlyCharts(target, charts) {
  const element = qs(target);
  if (window.Plotly) {
    element.querySelectorAll(".analytics-plot").forEach((plot) => Plotly.purge(plot));
    if (element.classList.contains("js-plotly-plot")) Plotly.purge(element);
  }
  element.innerHTML = "";

  const normalizedCharts = (Array.isArray(charts) ? charts : charts ? [charts] : [])
    .filter((chart) => chart?.data && chart?.layout);
  if (!normalizedCharts.length) {
    element.classList.remove("has-charts");
    return;
  }

  element.classList.add("has-charts");
  normalizedCharts.forEach((chart, index) => {
    const title = chartTitle(chart, index);
    const card = document.createElement("article");
    card.className = "analytics-chart-card";
    card.innerHTML = `<div class="analytics-chart-title">${escapeHtml(title)}</div>`;
    const plot = document.createElement("div");
    plot.className = "analytics-plot";
    plot.setAttribute("aria-label", title);
    card.appendChild(plot);
    element.appendChild(card);
    Plotly.newPlot(plot, chart.data, chart.layout, { responsive: true, displaylogo: false });
  });
}

function renderAnalyticsRows(rows, columns) {
  if (!rows.length) return `<div class="analytics-empty">No rows returned for this query.</div>`;
  const visibleColumns = (columns.length ? columns : Object.keys(rows[0])).slice(0, 7);
  const extraColumns = Math.max((columns.length || Object.keys(rows[0]).length) - visibleColumns.length, 0);
  const visibleRows = rows.slice(0, 8);
  const header = visibleColumns.map((column) => `<th>${escapeHtml(column)}</th>`).join("");
  const body = visibleRows.map((row) => (
    `<tr>${visibleColumns.map((column) => `<td>${escapeHtml(row[column])}</td>`).join("")}</tr>`
  )).join("");
  const noteParts = [];
  if (rows.length > visibleRows.length) noteParts.push(`${rows.length - visibleRows.length} more rows`);
  if (extraColumns > 0) noteParts.push(`${extraColumns} more columns`);
  const note = noteParts.length ? `<div class="analytics-table-note">${escapeHtml(noteParts.join(", "))}</div>` : "";
  return `
    <div class="analytics-table-wrap">
      <table class="analytics-table">
        <thead><tr>${header}</tr></thead>
        <tbody>${body}</tbody>
      </table>
    </div>
    ${note}
  `;
}

function renderAnalyticsReport(result) {
  const output = qs("#analytics-output");
  const rows = Array.isArray(result?.rows) ? result.rows : [];
  const columns = Array.isArray(result?.columns) ? result.columns : [];
  const charts = normalizeAnalyticsCharts(result);
  const paragraphs = String(result?.answer || "Query executed.")
    .split(/\n{2,}/)
    .map((paragraph) => paragraph.trim())
    .filter(Boolean)
    .map((paragraph) => `<p>${escapeHtml(paragraph)}</p>`)
    .join("");
  const sql = result?.sql ? escapeHtml(result.sql) : "No SQL was returned.";
  output.innerHTML = `
    <div class="analytics-answer">
      <span class="analytics-kicker">Analysis</span>
      ${paragraphs || "<p>Query executed.</p>"}
    </div>
    <div class="analytics-metrics">
      <div><strong>${rows.length}</strong><span>Rows</span></div>
      <div><strong>${columns.length}</strong><span>Columns</span></div>
      <div><strong>${charts.length}</strong><span>Charts</span></div>
    </div>
    <details class="analytics-sql">
      <summary>SQL</summary>
      <pre>${sql}</pre>
    </details>
    ${renderAnalyticsRows(rows, columns)}
  `;
  renderPlotlyCharts("#analytics-chart", charts);
}

function renderAnalyticsError(message) {
  qs("#analytics-output").innerHTML = `<div class="analytics-error">${escapeHtml(message)}</div>`;
  renderPlotlyCharts("#analytics-chart", []);
}

function renderAnalyticsLoading() {
  qs("#analytics-output").innerHTML = `<div class="muted">Asking analytics...</div>`;
  renderPlotlyCharts("#analytics-chart", []);
}

function setAnalyticsBusy(isBusy) {
  const button = qs("#run-analytics");
  if (!button) return;
  button.disabled = Boolean(isBusy);
  const label = button.querySelector("span");
  if (label) label.textContent = isBusy ? "Asking" : "Ask";
}

function escapeHtml(value) {
  return String(value ?? "").replace(/[&<>"']/g, (char) => ({
    "&": "&amp;",
    "<": "&lt;",
    ">": "&gt;",
    '"': "&quot;",
    "'": "&#39;",
  })[char]);
}

function numberOrNull(value) {
  if (value === null || value === undefined || String(value).trim() === "") return null;
  const number = Number(value);
  return Number.isFinite(number) ? number : null;
}

function percentValue(value) {
  const number = numberOrNull(value);
  if (number === null) return null;
  return Math.abs(number) <= 1 ? number * 100 : number;
}

function formatEdge(value) {
  const number = numberOrNull(value);
  if (number === null) return "N/A";
  const sign = number > 0 ? "+" : "";
  return `${sign}${number.toFixed(1)} pp`;
}

function formatEv(value) {
  const number = numberOrNull(value);
  if (number === null) return "N/A";
  const sign = number > 0 ? "+" : "";
  return `${sign}${number.toFixed(1)}%`;
}

function modelEdge(aiProb, marketProb) {
  const ai = percentValue(aiProb);
  const market = percentValue(marketProb);
  return ai === null || market === null ? null : ai - market;
}

function modelMargin(row) {
  const fighter1 = percentValue(row.Fighter1_AI_Prob);
  const fighter2 = percentValue(row.Fighter2_AI_Prob);
  return fighter1 === null || fighter2 === null ? null : Math.abs(fighter1 - fighter2);
}

function formatOdds(value) {
  const raw = String(value ?? "").trim();
  if (!raw || raw.toUpperCase() === "N/A") return "N/A";
  const number = Number(raw);
  if (!Number.isFinite(number)) return raw;
  return number > 0 ? `+${number}` : String(number);
}

function americanOddsProfitPerDollar(value) {
  const raw = String(value ?? "").trim().replace(/^\+/, "");
  const odds = Number(raw);
  if (!Number.isFinite(odds) || Math.abs(odds) < 100) return null;
  return odds > 0 ? odds / 100 : 100 / Math.abs(odds);
}

function expectedValuePercent(aiProb, bookOdds) {
  const probability = percentValue(aiProb);
  const profit = americanOddsProfitPerDollar(bookOdds);
  if (probability === null || profit === null) return null;
  const p = probability / 100;
  return (p * profit - (1 - p)) * 100;
}

function probabilityToAmericanOdds(value) {
  const percent = percentValue(value);
  if (percent === null || percent <= 0 || percent >= 100) return "N/A";
  const probability = percent / 100;
  const decimalOdds = 1 / probability;
  const americanOdds = decimalOdds >= 2
    ? Math.round((decimalOdds - 1) * 100)
    : Math.round(-100 / (decimalOdds - 1));
  return formatOdds(americanOdds);
}

function eventDate(event) {
  const raw = event?.date || event?.fights?.[0]?.date;
  if (!raw) return null;
  const parsed = new Date(raw);
  return Number.isNaN(parsed.getTime()) ? null : parsed;
}

function formatEventDate(event) {
  const date = eventDate(event);
  return date ? date.toLocaleDateString(undefined, { month: "short", day: "numeric", year: "numeric" }) : "Date pending";
}

function todayDateInputValue() {
  const now = new Date();
  const month = String(now.getMonth() + 1).padStart(2, "0");
  const day = String(now.getDate()).padStart(2, "0");
  return `${now.getFullYear()}-${month}-${day}`;
}

function commaList(value) {
  const items = String(value ?? "")
    .split(",")
    .map((item) => item.trim())
    .filter(Boolean);
  return items.length ? items : null;
}

function setValue(selector, value) {
  const element = qs(selector);
  if (element) element.value = value ?? "";
}

function setChecked(selector, value) {
  const element = qs(selector);
  if (element) element.checked = Boolean(value);
}

function setDisabled(selector, value) {
  const element = qs(selector);
  if (element) element.disabled = Boolean(value);
}

function setLogText(selector, value) {
  const element = qs(selector);
  if (!element) return;
  element.textContent = value;
  element.scrollTop = element.scrollHeight;
}

async function copyText(value) {
  if (navigator.clipboard?.writeText) {
    try {
      await navigator.clipboard.writeText(value);
      return;
    } catch (_error) {
      // Fall through to the textarea fallback for browsers that expose the API
      // but block write permission in this context.
    }
  }
  const textArea = document.createElement("textarea");
  textArea.value = value;
  textArea.setAttribute("readonly", "");
  textArea.style.position = "fixed";
  textArea.style.left = "-9999px";
  document.body.appendChild(textArea);
  textArea.select();
  try {
    if (!document.execCommand("copy")) {
      throw new Error("Clipboard copy was blocked.");
    }
  } finally {
    textArea.remove();
  }
}

function listValue(value) {
  return Array.isArray(value) ? value.join(", ") : "";
}

function parseManualOdds(value) {
  const raw = String(value || "").trim();
  if (!raw) return null;
  if (raw.startsWith("{")) {
    const parsed = JSON.parse(raw);
    return Object.keys(parsed).length ? parsed : null;
  }
  const odds = {};
  raw.split(/\r?\n|,/).map((line) => line.trim()).filter(Boolean).forEach((line) => {
    const separator = line.includes("=") ? "=" : ":";
    const index = line.indexOf(separator);
    if (index < 1) throw new Error(`Invalid odds entry: ${line}`);
    const fighter = line.slice(0, index).trim();
    const oddsValue = Number(line.slice(index + 1).trim().replace(/^\+/, ""));
    if (!fighter || Number.isNaN(oddsValue)) throw new Error(`Invalid odds entry: ${line}`);
    odds[fighter] = oddsValue;
  });
  return Object.keys(odds).length ? odds : null;
}

function applyDashboardDefaults(defaults) {
  const data = defaults.data || {};
  const predict = defaults.predict || {};

  setValue("#analytics-max-rows", data.analytics_max_rows);

  setValue("#predict-model-type", predict.model_type);
  selectedUpcomingNumber = Number(predict.upcoming_number || 1);
  setChecked("#predict-odds", predict.odds);
  setChecked("#predict-flaresolverr", predict.flaresolverr);
  setChecked("#predict-calibrated", predict.use_calibrated);
  setChecked("#predict-shap", predict.shap);
}

async function loadDashboardDefaults() {
  applyDashboardDefaults(await api("/api/defaults"));
}

function failingReadinessChecks(payload) {
  const checks = payload?.checks || {};
  return Object.entries(checks)
    .filter(([, check]) => !check?.ok)
    .map(([name]) => name.replace(/_/g, " "));
}

function renderReadiness(payload) {
  const badge = qs("#readiness-badge");
  if (!badge) return;
  const ready = Boolean(payload?.ready);
  const failures = failingReadinessChecks(payload);
  badge.classList.remove("ready", "not-ready", "checking");
  badge.classList.add(ready ? "ready" : "not-ready");
  badge.textContent = ready ? "Ready" : "Setup incomplete";
  badge.title = ready
    ? "Ready for predictions: databases, processed CSVs, and starter model are available."
    : `Missing or unavailable: ${failures.join(", ") || "readiness checks"}`;
}

function renderAnalyticsStatus(payload) {
  const status = qs("#analytics-status");
  if (!status) return;
  status.classList.remove("ready", "blocked");
  status.classList.add(payload?.configured ? "ready" : "blocked");
  if (payload?.configured) {
    const endpoint = payload.base_url ? ` at ${payload.base_url}` : "";
    status.textContent = `LLM analytics ready: ${payload.provider}/${payload.model}${endpoint}.`;
  } else {
    status.textContent = "LLM analytics is not configured. You can still run read-only SQL analytics.";
    status.title = payload?.hint || "";
  }
}

async function refreshAnalyticsStatus() {
  try {
    renderAnalyticsStatus(await api("/api/data/analytics/status"));
  } catch (error) {
    renderAnalyticsStatus({ configured: false, hint: error.message });
  }
}

async function refreshReadiness() {
  const badge = qs("#readiness-badge");
  if (badge) {
    badge.classList.remove("ready", "not-ready");
    badge.classList.add("checking");
    badge.textContent = "Checking readiness";
    badge.title = "Checking data, model, and database readiness.";
  }
  try {
    const response = await fetch("/api/readiness");
    const body = await response.json().catch(() => ({}));
    renderReadiness(response.ok ? body : body.detail || body);
  } catch (error) {
    renderReadiness({ ready: false, checks: { web: { ok: false, error: error.message } } });
  }
}

function hasPositiveSideEv(edge, odds, marketProb) {
  const market = percentValue(marketProb);
  return edge !== null && edge > 0 && market !== null && market > 0 && formatOdds(odds) !== "N/A";
}

function sideState(name, bookOdds, aiProb, marketProb, isPick) {
  const edge = modelEdge(aiProb, marketProb);
  const ev = expectedValuePercent(aiProb, bookOdds);
  const positiveEv = ev === null ? hasPositiveSideEv(edge, bookOdds, marketProb) : ev > 0;
  return {
    name,
    bookOdds: formatOdds(bookOdds),
    aiOdds: probabilityToAmericanOdds(aiProb),
    edge,
    edgeText: formatEdge(edge),
    ev,
    evText: formatEv(ev),
    positiveEv,
    isPick,
  };
}

function renderFighterPickLine(side) {
  const badges = [side.isPick ? "Pick" : "", side.positiveEv ? "+EV" : ""].filter(Boolean).join(" ");
  return `
    <div class="fighter-pick-line${side.isPick ? " picked" : ""}${side.positiveEv ? " positive-ev" : ""}">
      <div class="fighter-pick-name">
        <strong>${escapeHtml(side.name || "N/A")}</strong>
        ${badges ? `<span>${escapeHtml(badges)}</span>` : ""}
      </div>
      <div class="pick-stat">
        <span>Odds</span>
        <strong>${escapeHtml(side.bookOdds)}</strong>
      </div>
      <div class="pick-stat">
        <span>AI Odds</span>
        <strong>${escapeHtml(side.aiOdds)}</strong>
      </div>
      <div class="pick-stat">
        <span>EV</span>
        <strong class="${side.positiveEv ? "ev-yes" : "ev-no"}">${escapeHtml(side.evText)}</strong>
      </div>
      <div class="pick-stat edge">
        <span>Pick Edge</span>
        <strong>${escapeHtml(side.edgeText)}</strong>
      </div>
    </div>`;
}

function renderPredictionGraphic(target, predictions) {
  if (!predictions || predictions.length === 0) {
    qs(target).innerHTML = `<div class="muted">No prediction rows were produced.</div>`;
    return;
  }
  qs(target).innerHTML = `
    <div class="prediction-set">${predictions.map((row) => {
    const evPositive = String(row.EV) === "1";
    const f1Winner = row.AI_Pick === row.Fighter1;
    const f2Winner = row.AI_Pick === row.Fighter2;
    const fighter1 = sideState(row.Fighter1, row.Fighter1_Odds, row.Fighter1_AI_Prob, row.Fighter1_Market_Prob, f1Winner);
    const fighter2 = sideState(row.Fighter2, row.Fighter2_Odds, row.Fighter2_AI_Prob, row.Fighter2_Market_Prob, f2Winner);
    const pickedSide = f1Winner ? fighter1 : fighter2;
    const pickHasValue = pickedSide.positiveEv || (pickedSide.ev === null && evPositive);
    const margin = modelMargin(row);
    return `
      <article class="prediction-result pick-card${pickHasValue ? " has-value" : ""}">
        <div class="pick-card-banner">
          <div class="pick-card-main">
            <span class="prediction-kicker">Pick</span>
            <strong>${escapeHtml(row.AI_Pick || "N/A")}</strong>
            <p>${escapeHtml(row.Fighter1)} vs ${escapeHtml(row.Fighter2)}</p>
          </div>
          <div class="pick-card-signals">
            <div class="signal-pill primary">
              <span>AI Margin</span>
              <strong>${escapeHtml(formatEdge(margin))}</strong>
            </div>
            <div class="signal-pill primary">
              <span>Pick Edge</span>
              <strong>${escapeHtml(pickedSide.edgeText)}</strong>
            </div>
            <div class="signal-pill ${pickHasValue ? "value" : "neutral"}">
              <span>EV</span>
              <strong>${escapeHtml(pickedSide.evText)}</strong>
            </div>
          </div>
        </div>
        <div class="fighter-pick-table" aria-label="Prediction odds and expected value">
          <div class="fighter-pick-head">
            <span>Fighter</span>
            <span>Odds</span>
            <span>AI Odds</span>
            <span>EV</span>
            <span>Pick Edge</span>
          </div>
          ${renderFighterPickLine(fighter1)}
          ${renderFighterPickLine(fighter2)}
        </div>
      </article>`;
  }).join("")}</div>`;
}

function renderUpcomingEvents(payload) {
  const events = [...(payload?.events || [])].sort((left, right) => {
    const leftDate = eventDate(left);
    const rightDate = eventDate(right);
    if (leftDate && rightDate) return leftDate - rightDate;
    if (leftDate) return -1;
    if (rightDate) return 1;
    return Number(left.upcoming_number || 0) - Number(right.upcoming_number || 0);
  });
  upcomingEventsCache = events;
  const select = qs("#predict-event");
  if (!events.length) {
    if (select) {
      select.innerHTML = `<option value="">No upcoming events found</option>`;
      select.disabled = true;
    }
    selectedUpcomingNumber = null;
    selectedEventHasMatchedFights = false;
    updatePredictionButtons();
    qs("#event-preview").innerHTML = `<div class="muted">${escapeHtml(payload?.warning || "No upcoming UFC events found.")}</div>`;
    qs("#events-output").innerHTML = "";
    return;
  }

  const existingSelection = select?.value ? Number(select.value) : null;
  selectedUpcomingNumber = events.some((event) => Number(event.upcoming_number) === existingSelection)
    ? existingSelection
    : Number(events[0].upcoming_number);
  if (select) {
    select.disabled = false;
    select.innerHTML = events.map((event) => `
      <option value="${event.upcoming_number}">${escapeHtml(event.name)}</option>`).join("");
    select.value = String(selectedUpcomingNumber);
  }

  updateEventPreview();
  qs("#events-output").innerHTML = payload.warning ? `<div class="muted">${escapeHtml(payload.warning)}</div>` : "";
}

function selectedUpcomingEvent() {
  return upcomingEventsCache.find((event) => Number(event.upcoming_number) === Number(selectedUpcomingNumber));
}

function updateEventPreview() {
  const event = selectedUpcomingEvent();
  if (!event) {
    selectedEventHasMatchedFights = false;
    updatePredictionButtons();
    qs("#event-preview").innerHTML = `<div class="muted">Choose an upcoming event to preview the matched fights.</div>`;
    return;
  }
  const fights = event.fights || [];
  selectedEventHasMatchedFights = fights.length > 0;
  updatePredictionButtons();
  const preview = fights.slice(0, 6)
    .map((fight) => `<span class="fight-chip">${escapeHtml(fight.fighter1)} vs ${escapeHtml(fight.fighter2)}</span>`)
    .join("");
  qs("#event-preview").innerHTML = `
    <div class="upcoming-event-summary">
      <div>
        <span class="prediction-kicker">Selected Event</span>
        <strong>${escapeHtml(event.name)}</strong>
        <p>${escapeHtml(formatEventDate(event))} | ${fights.length} matched fights</p>
      </div>
      <span class="event-number">#${escapeHtml(event.upcoming_number)}</span>
    </div>
    <div class="fight-chip-list">${preview || `<span class="muted">No matched fights yet.</span>`}</div>`;
}

function renderDataRefreshResult(result) {
  const counts = result?.scrape_counts || {};
  const status = result?.status || {};
  const rowDeltas = result?.row_deltas || {};
  const modelCsvs = status.model_csvs || {};
  const rawCsvs = status.raw_csvs || {};
  const rows = {
    raw_competitions: rawCsvs.competitions?.rows ?? "Missing",
    raw_individuals: rawCsvs.individuals?.rows ?? "Missing",
    prediction_data: modelCsvs.prediction_data?.rows ?? "Missing",
    training_data: modelCsvs.training_data?.rows ?? "Missing",
    training_data_dec: modelCsvs.training_data_dec?.rows ?? "Missing",
  };
  renderJson("#data-output", {
    status: "Data pipeline completed.",
    scrape_counts: counts,
    row_deltas: rowDeltas,
    rows,
  });
  renderDataMetrics(status, rowDeltas);
}

function formatMetric(value) {
  if (value === null || value === undefined || Number.isNaN(Number(value))) return "N/A";
  const number = Number(value);
  return Math.abs(number) <= 1 ? number.toFixed(3) : number.toFixed(2);
}

async function refreshStatus() {
  const status = await api("/api/data/status");
  qs("#status-line").textContent = `Raw: ${status.raw_data_dir} | Data: ${status.data_dir}`;
  renderDataMetrics(status);
}

function formatDelta(value) {
  if (typeof value !== "number") return "";
  if (value === 0) return "0";
  return value > 0 ? `+${value}` : String(value);
}

function renderDataMetrics(status, deltas = {}) {
  const metrics = [
    ["Fights CSV", status.raw_csvs.competitions.rows ?? "Missing", deltas.competitions],
    ["Fighters CSV", status.raw_csvs.individuals.rows ?? "Missing", deltas.individuals],
    ["Training Rows", status.model_csvs.training_data.rows ?? "Missing", deltas.training_data],
  ];
  qs("#data-metrics").innerHTML = metrics
    .map(([label, value, delta]) => `
      <div class="metric">
        <strong>${value}</strong>
        <span>${label}</span>
        ${formatDelta(delta) ? `<em>${escapeHtml(formatDelta(delta))} this run</em>` : ""}
      </div>`)
    .join("");
}

async function refreshJobs() {
  const { jobs } = await api("/api/jobs");
  qs("#job-strip").textContent = jobs.length
    ? jobs.slice(0, 4).map((job) => `${job.kind}: ${job.state}`).join(" | ")
    : "No background jobs yet";
  const dataJob = jobs.find((job) => job.id === activeDataJobId);
  if (activeDataJobId) await renderJobLog("#data-log", activeDataJobId);
  if (dataJob?.state === "succeeded") {
    renderDataRefreshResult(dataJob.result || {});
    activeDataJobId = null;
    await refreshStatus().catch(() => {});
    await refreshReadiness().catch(() => {});
    await loadUpcomingEventsWithStatus();
  } else if (dataJob?.state === "failed") {
    renderJson("#data-output", dataJob.error || "Data pipeline failed");
    activeDataJobId = null;
  }
  const eventJob = jobs.find((job) => job.id === activeEventJobId);
  if (activeEventJobId) await renderJobLog("#events-log", activeEventJobId);
  if (eventJob?.state === "succeeded") {
    renderPredictionGraphic("#events-output", eventJob.result?.predictions || []);
    activeEventJobId = null;
  } else if (eventJob?.state === "failed") {
    renderJson("#events-output", eventJob.error || "Prediction failed");
    activeEventJobId = null;
  }
  const matchupJob = jobs.find((job) => job.id === activeMatchupJobId);
  if (activeMatchupJobId) await renderJobLog("#prediction-log", activeMatchupJobId);
  if (matchupJob?.state === "succeeded") {
    renderPredictionGraphic("#prediction-output", matchupJob.result?.predictions || []);
    activeMatchupJobId = null;
  } else if (matchupJob?.state === "failed") {
    renderJson("#prediction-output", matchupJob.error || "Prediction failed");
    activeMatchupJobId = null;
  }
}

async function renderJobLog(target, jobId) {
  try {
    const payload = await api(`/api/jobs/${jobId}/log`);
    setLogText(target, payload.log || "No log output yet.");
  } catch (error) {
    setLogText(target, error.message);
  }
}

function wireTabs() {
  qsa(".tab").forEach((tab) => {
    tab.addEventListener("click", () => {
      qsa(".tab").forEach((item) => item.classList.remove("active"));
      qsa(".panel").forEach((item) => item.classList.remove("active"));
      tab.classList.add("active");
      qs(`#${tab.dataset.tab}`).classList.add("active");
    });
  });
}

function wireData() {
  qs("#refresh-status").addEventListener("click", async () => {
    await Promise.allSettled([refreshStatus(), refreshReadiness(), refreshAnalyticsStatus()]);
  });
  qs("#copy-analytics-system-prompt").addEventListener("click", async () => {
    const status = qs("#analytics-copy-status");
    try {
      const payload = await api("/api/data/analytics/system-prompt");
      await copyText(payload.system_prompt);
      status.textContent = "Analytics agent system prompt copied.";
    } catch (error) {
      status.textContent = `Could not copy prompt: ${error.message}`;
    }
  });
  qs("#run-data").addEventListener("click", async () => {
    try {
      const payload = {
        scrape: true,
        rebuild: true,
        reset_db: true,
        force_full: false,
        odds_features: true,
        odds: false,
      };
      const job = await api("/api/data/refresh", { method: "POST", body: JSON.stringify(payload) });
      activeDataJobId = job.job_id;
      qs("#data-log").textContent = "Queued...";
      renderJson("#data-output", job);
      await refreshJobs();
    } catch (error) {
      renderJson("#data-output", error.message);
    }
  });
  qs("#run-analytics").addEventListener("click", async () => {
    const question = qs("#analytics-question").value.trim();
    if (question.length < 3) {
      renderAnalyticsError("Enter an analytics question with at least 3 characters.");
      return;
    }
    setAnalyticsBusy(true);
    renderAnalyticsLoading();
    try {
      const result = await api("/api/data/analytics", {
        method: "POST",
        body: JSON.stringify({
          question,
          sql: qs("#analytics-sql").value || null,
          max_rows: Number(qs("#analytics-max-rows").value || 100),
        }),
      });
      renderAnalyticsReport(result);
    } catch (error) {
      renderAnalyticsError(error.message);
    } finally {
      setAnalyticsBusy(false);
    }
  });
}

async function refreshModels() {
  const modelType = qs("#predict-model-type").value || "win";
  const predictPayload = await api(`/api/predict/models?model_type=${encodeURIComponent(modelType)}`);
  const predictModels = predictPayload.models || [];
  qs("#predict-model").innerHTML = modelOptions(predictModels);
  renderPredictModelState(modelType, predictModels);
}

function modelOptions(models) {
  if (!models.length) return `<option value="">No models found</option>`;
  return `<option value="">Latest model</option>` +
    models.map((model) => `<option value="${escapeHtml(model.path)}">${escapeHtml(model.name)}</option>`).join("");
}

function renderPredictModelState(modelType, models) {
  predictModelsAvailable = models.length > 0;
  setDisabled("#predict-model", !predictModelsAvailable);
  updatePredictionButtons();
  const status = qs("#predict-model-status");
  if (!status) return;
  status.classList.remove("ready", "blocked");
  status.classList.add(predictModelsAvailable ? "ready" : "blocked");
  status.textContent = predictModelsAvailable
    ? `${models.length} ${modelType} model${models.length === 1 ? "" : "s"} available. Leave Model on Latest model to use the newest one.`
    : `No ${modelType} models found. Run setup again or provide a compatible ${modelType} model before predicting.`;
}

function updatePredictionButtons() {
  setDisabled("#run-event-predict", !predictModelsAvailable || !selectedEventHasMatchedFights);
  setDisabled("#run-matchup", !predictModelsAvailable);
}

function predictionDataCsv() {
  return qs("#predict-data-csv").value.trim() || null;
}

function trainingDataCsv() {
  return qs("#predict-training-csv").value.trim() || null;
}

function predictionOutputDir() {
  return qs("#predict-output-dir").value.trim() || null;
}

async function loadUpcomingEvents() {
  const select = qs("#predict-event");
  if (select) {
    select.disabled = true;
    select.innerHTML = `<option value="">Loading upcoming events...</option>`;
  }
  selectedEventHasMatchedFights = false;
  updatePredictionButtons();
  qs("#event-preview").innerHTML = `<div class="muted">Loading upcoming UFC events...</div>`;
  qs("#events-output").innerHTML = "";
  const params = new URLSearchParams();
  const predictionCsv = predictionDataCsv();
  if (predictionCsv) params.set("prediction_data_csv", predictionCsv);
  const query = params.toString();
  renderUpcomingEvents(await api(`/api/predict/upcoming${query ? `?${query}` : ""}`));
}

function renderUpcomingEventsError(message) {
  const select = qs("#predict-event");
  if (select) {
    select.disabled = true;
    select.innerHTML = `<option value="">Could not load upcoming events</option>`;
  }
  selectedUpcomingNumber = null;
  selectedEventHasMatchedFights = false;
  updatePredictionButtons();
  qs("#event-preview").innerHTML = `<div class="muted">${escapeHtml(message || "Could not load upcoming UFC events.")}</div>`;
  renderJson("#events-output", message || "Could not load upcoming UFC events.");
}

async function loadUpcomingEventsWithStatus() {
  try {
    await loadUpcomingEvents();
  } catch (error) {
    renderUpcomingEventsError(error.message);
  }
}

function wirePrediction() {
  setValue("#fight-date", qs("#fight-date").value || todayDateInputValue());
  qs("#predict-model-type").addEventListener("change", () => {
    refreshModels().catch(() => {});
  });
  qs("#predict-event").addEventListener("change", () => {
    selectedUpcomingNumber = Number(qs("#predict-event").value || 0) || null;
    updateEventPreview();
  });
  qs("#load-events").addEventListener("click", async () => {
    await loadUpcomingEventsWithStatus();
  });
  qs("#load-fighters").addEventListener("click", async () => {
    try {
      const params = new URLSearchParams();
      const predictionCsv = predictionDataCsv();
      if (predictionCsv) params.set("prediction_data_csv", predictionCsv);
      const query = params.toString();
      const { fighters } = await api(`/api/predict/fighters${query ? `?${query}` : ""}`);
      qs("#fighters-list").innerHTML = fighters.map((name) => `<option value="${escapeHtml(name)}"></option>`).join("");
      renderJson("#prediction-output", `${fighters.length} fighters loaded`);
    } catch (error) {
      renderJson("#prediction-output", error.message);
    }
  });
  qs("#run-event-predict").addEventListener("click", async () => {
    try {
      const upcomingNumber = selectedUpcomingNumber || Number(qs("#predict-event").value || 0);
      if (!predictModelsAvailable) {
        renderJson("#events-output", "No model is available for this target. Run setup again or provide a compatible model before predicting.");
        return;
      }
      if (!upcomingNumber) {
        renderJson("#events-output", "Choose an upcoming event before prediction.");
        return;
      }
      if (!selectedEventHasMatchedFights) {
        renderJson("#events-output", "The selected event has no matched fights yet. Run Update Data after setup or choose another event.");
        return;
      }
      const payload = {
        model_type: qs("#predict-model-type").value,
        model_path: qs("#predict-model").value || null,
        prediction_data_csv: predictionDataCsv(),
        training_data_csv: trainingDataCsv(),
        output_dir: predictionOutputDir(),
        upcoming_number: upcomingNumber,
        odds: qs("#predict-odds").checked,
        manual_odds: qs("#predict-odds").checked ? parseManualOdds(qs("#event-manual-odds").value) : null,
        flaresolverr: qs("#predict-flaresolverr").checked,
        use_calibrated: qs("#predict-calibrated").checked,
        shap: qs("#predict-shap").checked,
      };
      const job = await api("/api/predict/event", { method: "POST", body: JSON.stringify(payload) });
      activeEventJobId = job.job_id;
      setLogText("#events-log", "Queued...");
      qs("#events-output").innerHTML = `<div class="muted">Prediction results will appear here when the job finishes.</div>`;
      await refreshJobs();
    } catch (error) {
      renderJson("#events-output", error.message);
    }
  });
  qs("#run-matchup").addEventListener("click", async () => {
    try {
      const fighter1 = qs("#fighter1").value.trim();
      const fighter2 = qs("#fighter2").value.trim();
      if (!predictModelsAvailable) {
        renderJson("#prediction-output", "No model is available for this target. Run setup again or provide a compatible model before predicting.");
        return;
      }
      if (!fighter1 || !fighter2) {
        renderJson("#prediction-output", "Enter both fighter names before prediction.");
        return;
      }
      const payload = {
        model_type: qs("#predict-model-type").value,
        model_path: qs("#predict-model").value || null,
        prediction_data_csv: predictionDataCsv(),
        training_data_csv: trainingDataCsv(),
        output_dir: predictionOutputDir(),
        fighter1,
        fighter2,
        fight_date: qs("#fight-date").value || null,
        odds: qs("#predict-odds").checked,
        flaresolverr: qs("#predict-flaresolverr").checked,
        use_calibrated: qs("#predict-calibrated").checked,
        shap: qs("#predict-shap").checked,
      };
      const job = await api("/api/predict/matchup", { method: "POST", body: JSON.stringify(payload) });
      activeMatchupJobId = job.job_id;
      setLogText("#prediction-log", "Queued...");
      qs("#prediction-output").innerHTML = `<div class="muted">Prediction results will appear here when the job finishes.</div>`;
      await refreshJobs();
    } catch (error) {
      renderJson("#prediction-output", error.message);
    }
  });
}

wireTabs();
wireData();
wirePrediction();
loadDashboardDefaults().catch(() => {}).finally(() => loadUpcomingEventsWithStatus());
refreshStatus().catch(() => {});
refreshReadiness().catch(() => {});
refreshAnalyticsStatus().catch(() => {});
refreshModels().catch(() => {});
refreshJobs().catch(() => {});
setInterval(refreshJobs, 5000);
if (window.lucide) window.lucide.createIcons();
