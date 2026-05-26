const qs = (selector) => document.querySelector(selector);
const qsa = (selector) => [...document.querySelectorAll(selector)];
let activeEventJobId = null;
let activeMatchupJobId = null;
let activeTrainingJobId = null;
let activeDataJobId = null;
let selectedUpcomingNumber = 1;
let upcomingEventsCache = [];
let predictModelsAvailable = false;

async function api(path, options = {}) {
  const response = await fetch(path, {
    headers: { "Content-Type": "application/json" },
    ...options,
  });
  if (!response.ok) {
    const body = await response.json().catch(() => ({ detail: response.statusText }));
    throw new Error(body.detail || response.statusText);
  }
  return response.json();
}

function renderJson(target, value) {
  qs(target).textContent = typeof value === "string" ? value : JSON.stringify(value, null, 2);
}

function renderPlotlyChart(target, chart) {
  const element = qs(target);
  if (!chart) {
    if (window.Plotly) Plotly.purge(element);
    element.innerHTML = "";
    return;
  }
  Plotly.newPlot(element, chart.data, chart.layout, { responsive: true });
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

function formatPercent(value) {
  const number = percentValue(value);
  return number === null ? "N/A" : `${number.toFixed(1)}%`;
}

function probabilityWidth(value) {
  const number = percentValue(value);
  return number === null ? 0 : Math.max(0, Math.min(100, number));
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

function modelEdge(aiProb, marketProb) {
  const ai = percentValue(aiProb);
  const market = percentValue(marketProb);
  return ai === null || market === null ? null : ai - market;
}

function formatOdds(value) {
  const raw = String(value ?? "").trim();
  if (!raw || raw.toUpperCase() === "N/A") return "N/A";
  const number = Number(raw);
  if (!Number.isFinite(number)) return raw;
  return number > 0 ? `+${number}` : String(number);
}

function pickedEdge(row) {
  if (row.AI_Pick === row.Fighter1) return modelEdge(row.Fighter1_AI_Prob, row.Fighter1_Market_Prob);
  if (row.AI_Pick === row.Fighter2) return modelEdge(row.Fighter2_AI_Prob, row.Fighter2_Market_Prob);
  return null;
}

function eventDate(event) {
  const raw = event?.fights?.[0]?.date;
  if (!raw) return null;
  const parsed = new Date(raw);
  return Number.isNaN(parsed.getTime()) ? null : parsed;
}

function formatEventDate(event) {
  const date = eventDate(event);
  return date ? date.toLocaleDateString(undefined, { month: "short", day: "numeric", year: "numeric" }) : "Date pending";
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
  const train = defaults.train || {};
  const predict = defaults.predict || {};

  setValue("#analytics-max-rows", data.analytics_max_rows);

  setValue("#train-model-type", train.model_type);
  setValue("#train-preset", train.preset);
  setValue("#train-time-limit", train.time_limit);
  setValue("#train-split", train.split_strategy);
  setValue("#train-walk-windows", train.walkforward_n_windows);
  setValue("#train-walk-year", train.walkforward_initial_year);
  setChecked("#train-refit", train.refit_full);
  setChecked("#train-refit-all", train.refit_all);
  setValue("#train-test-size", train.test_size);
  setValue("#train-val-date", train.val_date);
  setValue("#train-start-date", train.start_date);
  setValue("#train-num-fights", train.num_fights);
  setChecked("#train-include-split-dec", train.include_split_dec);
  setValue("#train-decay", train.decay_rate);
  setValue("#train-normalize", train.normalize);
  setChecked("#train-recency", train.use_recency_weights);
  setChecked("#train-importance", train.calculate_importance);
  setValue("#train-feature-list", listValue(train.feature_list));
  setValue("#train-include-patterns", listValue(train.included_strings));
  setValue("#train-exclude-patterns", listValue(train.excluded_strings));
  setValue("#train-required-features", listValue(train.required_strings));
  setValue("#train-model-families", listValue(train.included_model_types));

  setValue("#predict-model-type", predict.model_type);
  selectedUpcomingNumber = Number(predict.upcoming_number || 1);
  setChecked("#predict-odds", predict.odds);
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

function renderPredictionGraphic(target, predictions) {
  if (!predictions || predictions.length === 0) {
    qs(target).innerHTML = `<div class="muted">No prediction rows were produced.</div>`;
    return;
  }
  qs(target).innerHTML = `<div class="prediction-set">${predictions.map((row) => {
    const evPositive = String(row.EV) === "1";
    const confidence = numberOrNull(row.Confidence);
    const f1Prob = probabilityWidth(row.Fighter1_AI_Prob);
    const f2Prob = probabilityWidth(row.Fighter2_AI_Prob);
    const valueSide = evPositive ? row.AI_Pick : "None";
    const f1Winner = row.AI_Pick === row.Fighter1;
    const f2Winner = row.AI_Pick === row.Fighter2;
    const f1Edge = modelEdge(row.Fighter1_AI_Prob, row.Fighter1_Market_Prob);
    const f2Edge = modelEdge(row.Fighter2_AI_Prob, row.Fighter2_Market_Prob);
    const pickEdge = pickedEdge(row);
    return `
      <article class="prediction-result pretty-prediction${evPositive ? " has-value" : ""}">
        <div class="prediction-hero">
          <div>
            <span class="prediction-kicker">Model Pick</span>
            <strong>${escapeHtml(row.AI_Pick || "N/A")}</strong>
            <p>${escapeHtml(row.Fighter1)} vs ${escapeHtml(row.Fighter2)}</p>
          </div>
          <div class="confidence-pill">
            <span>Confidence</span>
            <strong>${confidence === null ? "N/A" : `${confidence.toFixed(1)}%`}</strong>
          </div>
        </div>
        <div class="prediction-title">
          <div>
            <strong>${escapeHtml(row.Fighter1)} vs ${escapeHtml(row.Fighter2)}</strong>
            <p>Model edge on pick: ${escapeHtml(formatEdge(pickEdge))}</p>
          </div>
          <span class="${evPositive ? "ev-positive" : "ev-neutral"}">${evPositive ? "Positive EV" : "No positive EV"}</span>
        </div>
        <div class="fighter-prob-grid">
          <div class="fighter-prob-card${f1Winner ? " picked" : ""}">
            <div class="fighter-prob-top">
              <strong>${escapeHtml(row.Fighter1)}</strong>
              <span>${formatPercent(row.Fighter1_AI_Prob)}</span>
            </div>
            <div class="probability-track"><span style="width: ${f1Prob}%"></span></div>
            <div class="market-line">Market ${formatPercent(row.Fighter1_Market_Prob)} | Odds ${escapeHtml(formatOdds(row.Fighter1_Odds))}</div>
            <div class="edge-chip ${f1Edge !== null && f1Edge > 0 ? "edge-positive" : "edge-neutral"}">Model Edge ${escapeHtml(formatEdge(f1Edge))}</div>
          </div>
          <div class="fighter-prob-card${f2Winner ? " picked" : ""}">
            <div class="fighter-prob-top">
              <strong>${escapeHtml(row.Fighter2)}</strong>
              <span>${formatPercent(row.Fighter2_AI_Prob)}</span>
            </div>
            <div class="probability-track"><span style="width: ${f2Prob}%"></span></div>
            <div class="market-line">Market ${formatPercent(row.Fighter2_Market_Prob)} | Odds ${escapeHtml(formatOdds(row.Fighter2_Odds))}</div>
            <div class="edge-chip ${f2Edge !== null && f2Edge > 0 ? "edge-positive" : "edge-neutral"}">Model Edge ${escapeHtml(formatEdge(f2Edge))}</div>
          </div>
        </div>
        <div class="prediction-callout">
          <div>
            <span>Value Side</span>
            <strong>${escapeHtml(valueSide)}</strong>
          </div>
          <div>
            <span>AI Fair Line</span>
            <strong>${escapeHtml(formatOdds(row.AI_Odds))}</strong>
          </div>
          <div>
            <span>Pick Edge</span>
            <strong>${escapeHtml(formatEdge(pickEdge))}</strong>
          </div>
        </div>
        <div class="prediction-foot">
          <span>Pick: <strong>${escapeHtml(row.AI_Pick)}</strong></span>
          <span>Confidence: <strong>${confidence === null ? "N/A" : `${confidence.toFixed(1)}%`}</strong></span>
          <span>EV: <strong>${evPositive ? "Yes" : "No"}</strong></span>
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
    qs("#event-preview").innerHTML = `<div class="muted">Choose an upcoming event to preview the matched fights.</div>`;
    return;
  }
  const fights = event.fights || [];
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
    rows,
  });
}

function formatMetric(value) {
  if (value === null || value === undefined || Number.isNaN(Number(value))) return "N/A";
  const number = Number(value);
  return Math.abs(number) <= 1 ? number.toFixed(3) : number.toFixed(2);
}

function renderEvaluation(summary) {
  if (!summary?.available) {
    qs("#eval-output").innerHTML = `<div class="muted">${escapeHtml(summary?.message || "No evaluation artifacts found.")}</div>`;
    return;
  }
  const holdout = summary.metrics?.holdout_predictions || {};
  const train = summary.metrics?.training || {};
  const validation = summary.metrics?.validation || {};
  const holdoutScores = summary.metrics?.holdout || {};
  const metricCards = [
    ["Samples", holdout.samples],
    ["Holdout Accuracy", holdout.accuracy ?? holdoutScores.accuracy],
    ["Holdout Log Loss", holdout.log_loss ?? holdoutScores.log_loss],
    ["Brier Score", holdout.brier_score],
    ["ROC AUC", holdout.roc_auc],
    ["Train Accuracy", train.accuracy],
    ["Validation Accuracy", validation.accuracy],
    ["Mean Probability", holdout.mean_probability],
  ];
  const features = (summary.feature_importance || []).slice(0, 10);
  const weights = summary.model_weights || [];
  const checks = summary.best_practices || [];
  const reportPaths = summary.report_paths || {};
  qs("#eval-output").innerHTML = `
    <div>
      <h2>${escapeHtml(summary.model_name)}</h2>
      <p>${escapeHtml(summary.model_path)}</p>
      ${reportPaths.markdown ? `<p>Evaluation report: ${escapeHtml(reportPaths.markdown)}</p>` : ""}
    </div>
    <div class="eval-grid">
      ${metricCards.map(([label, value]) => `<div class="eval-card"><strong>${formatMetric(value)}</strong><span>${label}</span></div>`).join("")}
    </div>
    <div class="eval-chart-grid">
      <div id="eval-calibration" class="eval-chart"></div>
      <div id="eval-confidence" class="eval-chart"></div>
      <div id="eval-outcomes" class="eval-chart"></div>
    </div>
    <div class="eval-list">
      <details open>
        <summary>Best Practices Notes</summary>
        <ul>${(summary.notes || []).map((note) => `<li>${escapeHtml(note)}</li>`).join("")}</ul>
      </details>
      <details open>
        <summary>Best-Practice Checks</summary>
        <div class="check-list">
          ${checks.map((check) => `
            <div class="check-row">
              <span class="check-status ${escapeHtml(check.status)}">${escapeHtml(check.status)}</span>
              <div><strong>${escapeHtml(check.name)}</strong><p>${escapeHtml(check.detail)}</p></div>
            </div>`).join("") || "<p>No best-practice checks could be derived from this model.</p>"}
        </div>
      </details>
      <details>
        <summary>Top Features</summary>
        <ol>${features.map((row) => `<li>${escapeHtml(row.feature)}: ${formatMetric(row.importance)}</li>`).join("") || "<li>No feature importance artifact found.</li>"}</ol>
      </details>
      <details>
        <summary>Model Weights</summary>
        <ol>${weights.map((row) => `<li>${escapeHtml(row.model)}: ${formatMetric(row.weight)}</li>`).join("") || "<li>No ensemble weights found.</li>"}</ol>
      </details>
      <details>
        <summary>Artifacts</summary>
        <ul>${(summary.artifacts || []).map((artifact) => `<li>${escapeHtml(artifact.name)} (${artifact.size_bytes} bytes)</li>`).join("")}</ul>
      </details>
    </div>`;

  const charts = summary.charts || {};
  if (charts.calibration) Plotly.newPlot("eval-calibration", charts.calibration.data, charts.calibration.layout, { responsive: true });
  if (charts.confidence) Plotly.newPlot("eval-confidence", charts.confidence.data, charts.confidence.layout, { responsive: true });
  if (charts.outcomes) Plotly.newPlot("eval-outcomes", charts.outcomes.data, charts.outcomes.layout, { responsive: true });
}

async function refreshStatus() {
  const status = await api("/api/data/status");
  qs("#status-line").textContent = `Raw: ${status.raw_data_dir} | Data: ${status.data_dir}`;
  const metrics = [
    ["Fights CSV", status.raw_csvs.competitions.rows ?? "Missing"],
    ["Fighters CSV", status.raw_csvs.individuals.rows ?? "Missing"],
    ["Training Rows", status.model_csvs.training_data.rows ?? "Missing"],
  ];
  qs("#data-metrics").innerHTML = metrics
    .map(([label, value]) => `<div class="metric"><strong>${value}</strong><span>${label}</span></div>`)
    .join("");
}

async function refreshJobs() {
  const { jobs } = await api("/api/jobs");
  qs("#job-strip").textContent = jobs.length
    ? jobs.slice(0, 4).map((job) => `${job.kind}: ${job.state}`).join(" | ")
    : "No background jobs yet";
  qs("#train-jobs").textContent = jobs.length ? JSON.stringify(jobs, null, 2) : "No training jobs yet.";
  const trainingJob = jobs.find((job) => job.id === activeTrainingJobId);
  if (activeTrainingJobId) await renderJobLog("#train-log", activeTrainingJobId);
  if (trainingJob?.state === "succeeded") {
    renderEvaluation(trainingJob.result?.evaluation || { available: false, message: "Training finished without evaluation artifacts." });
    activateSubtab("train-evals");
    activeTrainingJobId = null;
  } else if (trainingJob?.state === "failed") {
    renderJson("#eval-output", trainingJob.error || "Training failed");
    activateSubtab("train-evals");
    activeTrainingJobId = null;
  }
  const dataJob = jobs.find((job) => job.id === activeDataJobId);
  if (activeDataJobId) await renderJobLog("#data-log", activeDataJobId);
  if (dataJob?.state === "succeeded") {
    renderDataRefreshResult(dataJob.result || {});
    activeDataJobId = null;
    await refreshStatus().catch(() => {});
    await refreshReadiness().catch(() => {});
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
    qs(target).textContent = payload.log || "No log output yet.";
  } catch (error) {
    qs(target).textContent = error.message;
  }
}

function activateSubtab(subtabId) {
  qsa(".subtab").forEach((item) => item.classList.toggle("active", item.dataset.subtab === subtabId));
  qsa(".subpanel").forEach((item) => item.classList.toggle("active", item.id === subtabId));
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
  qsa(".subtab").forEach((tab) => {
    tab.addEventListener("click", () => {
      activateSubtab(tab.dataset.subtab);
    });
  });
}

function wireData() {
  qs("#refresh-status").addEventListener("click", async () => {
    await Promise.allSettled([refreshStatus(), refreshReadiness()]);
  });
  qs("#run-data").addEventListener("click", async () => {
    try {
      const payload = {
        scrape: true,
        rebuild: true,
        reset_db: true,
        force_full: false,
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
    try {
      const result = await api("/api/data/analytics", {
        method: "POST",
        body: JSON.stringify({
          question: qs("#analytics-question").value,
          sql: qs("#analytics-sql").value || null,
          max_rows: Number(qs("#analytics-max-rows").value || 100),
        }),
      });
      renderJson("#analytics-output", { answer: result.answer, sql: result.sql, rows: result.rows });
      renderPlotlyChart("#analytics-chart", result.chart);
    } catch (error) {
      renderJson("#analytics-output", error.message);
      renderPlotlyChart("#analytics-chart", null);
    }
  });
}

function wireTraining() {
  qs("#run-train").addEventListener("click", async () => {
    try {
      const payload = {
        model_type: qs("#train-model-type").value,
        preset: qs("#train-preset").value,
        time_limit: Number(qs("#train-time-limit").value),
        split_strategy: qs("#train-split").value,
        walkforward_n_windows: Number(qs("#train-walk-windows").value),
        walkforward_initial_year: Number(qs("#train-walk-year").value),
        refit_full: qs("#train-refit").checked,
        refit_all: qs("#train-refit-all").checked,
        use_script_defaults: qs("#train-script-defaults").checked,
        test_size: qs("#train-test-size").value || null,
        val_date: qs("#train-val-date").value || null,
        start_date: qs("#train-start-date").value,
        num_fights: Number(qs("#train-num-fights").value),
        include_split_dec: qs("#train-include-split-dec").checked,
        decay_rate: Number(qs("#train-decay").value),
        normalize: qs("#train-normalize").value,
        use_recency_weights: qs("#train-recency").checked,
        calculate_importance: qs("#train-importance").checked,
        feature_list: commaList(qs("#train-feature-list").value),
        included_strings: commaList(qs("#train-include-patterns").value),
        excluded_strings: commaList(qs("#train-exclude-patterns").value),
        required_strings: commaList(qs("#train-required-features").value),
        included_model_types: commaList(qs("#train-model-families").value),
      };
      const job = await api("/api/train", { method: "POST", body: JSON.stringify(payload) });
      activeTrainingJobId = job.job_id;
      qs("#train-log").textContent = "Queued...";
      renderJson("#train-jobs", job);
      await refreshJobs();
    } catch (error) {
      renderJson("#train-jobs", error.message);
    }
  });
  qs("#load-eval").addEventListener("click", async () => {
    try {
      const modelPath = qs("#train-eval-model").value;
      const query = modelPath ? `?model_path=${encodeURIComponent(modelPath)}` : "";
      renderEvaluation(await api(`/api/train/evaluations${query}`));
    } catch (error) {
      renderJson("#eval-output", error.message);
    }
  });
}

async function refreshModels() {
  const modelType = qs("#predict-model-type").value || "win";
  const [predictPayload, allPayload] = await Promise.all([
    api(`/api/predict/models?model_type=${encodeURIComponent(modelType)}`),
    api("/api/predict/models"),
  ]);
  const predictModels = predictPayload.models || [];
  qs("#predict-model").innerHTML = modelOptions(predictModels);
  qs("#train-eval-model").innerHTML = modelOptions(allPayload.models || []);
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
  setDisabled("#run-event-predict", !predictModelsAvailable);
  setDisabled("#run-matchup", !predictModelsAvailable);
  const status = qs("#predict-model-status");
  if (!status) return;
  status.classList.remove("ready", "blocked");
  status.classList.add(predictModelsAvailable ? "ready" : "blocked");
  status.textContent = predictModelsAvailable
    ? `${models.length} ${modelType} model${models.length === 1 ? "" : "s"} available. Leave Model on Latest model to use the newest one.`
    : `No ${modelType} models found. Run setup again or train a ${modelType} model before predicting.`;
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
  qs("#event-preview").innerHTML = `<div class="muted">Loading upcoming UFC events...</div>`;
  qs("#events-output").innerHTML = "";
  const params = new URLSearchParams({ limit: "20" });
  const predictionCsv = predictionDataCsv();
  if (predictionCsv) params.set("prediction_data_csv", predictionCsv);
  renderUpcomingEvents(await api(`/api/predict/upcoming?${params.toString()}`));
}

function wirePrediction() {
  qs("#predict-model-type").addEventListener("change", () => {
    refreshModels().catch(() => {});
  });
  qs("#predict-event").addEventListener("change", () => {
    selectedUpcomingNumber = Number(qs("#predict-event").value || 0) || null;
    updateEventPreview();
  });
  qs("#load-events").addEventListener("click", async () => {
    try {
      await loadUpcomingEvents();
    } catch (error) {
      renderJson("#events-output", error.message);
    }
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
        renderJson("#events-output", "No model is available for this target. Run setup again or train a model before predicting.");
        return;
      }
      if (!upcomingNumber) {
        renderJson("#events-output", "Choose an upcoming event before prediction.");
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
        manual_odds: parseManualOdds(qs("#event-manual-odds").value),
        use_calibrated: qs("#predict-calibrated").checked,
        shap: qs("#predict-shap").checked,
      };
      const job = await api("/api/predict/event", { method: "POST", body: JSON.stringify(payload) });
      activeEventJobId = job.job_id;
      qs("#events-log").textContent = "Queued...";
      renderJson("#events-output", job);
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
        renderJson("#prediction-output", "No model is available for this target. Run setup again or train a model before predicting.");
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
        odds_fighter1: qs("#fighter1-odds").value ? Number(qs("#fighter1-odds").value) : null,
        odds_fighter2: qs("#fighter2-odds").value ? Number(qs("#fighter2-odds").value) : null,
        odds: qs("#matchup-odds").checked,
        use_calibrated: qs("#predict-calibrated").checked,
        shap: qs("#predict-shap").checked,
      };
      const job = await api("/api/predict/matchup", { method: "POST", body: JSON.stringify(payload) });
      activeMatchupJobId = job.job_id;
      qs("#prediction-log").textContent = "Queued...";
      renderJson("#prediction-output", job);
      await refreshJobs();
    } catch (error) {
      renderJson("#prediction-output", error.message);
    }
  });
}

wireTabs();
wireData();
wireTraining();
wirePrediction();
loadDashboardDefaults().catch(() => {}).finally(() => loadUpcomingEvents().catch(() => {}));
refreshStatus().catch(() => {});
refreshReadiness().catch(() => {});
refreshModels().catch(() => {});
refreshJobs().catch(() => {});
setInterval(refreshJobs, 5000);
if (window.lucide) window.lucide.createIcons();
