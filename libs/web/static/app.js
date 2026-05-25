const qs = (selector) => document.querySelector(selector);
const qsa = (selector) => [...document.querySelectorAll(selector)];
let activeEventJobId = null;
let activeMatchupJobId = null;
let activeTrainingJobId = null;
let activeDataJobId = null;
let selectedUpcomingNumber = 1;

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

function escapeHtml(value) {
  return String(value ?? "").replace(/[&<>"']/g, (char) => ({
    "&": "&amp;",
    "<": "&lt;",
    ">": "&gt;",
    '"': "&quot;",
    "'": "&#39;",
  })[char]);
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

  setChecked("#data-scrape", data.scrape);
  setChecked("#data-rebuild", data.rebuild);
  setChecked("#data-reset", data.reset_db);
  setChecked("#data-force", data.force_full);
  setChecked("#data-odds", data.odds);
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
  setValue("#predict-upcoming", predict.upcoming_number);
  selectedUpcomingNumber = Number(predict.upcoming_number || 1);
  setChecked("#predict-odds", predict.odds);
  setChecked("#predict-calibrated", predict.use_calibrated);
  setChecked("#predict-shap", predict.shap);
}

async function loadDashboardDefaults() {
  applyDashboardDefaults(await api("/api/defaults"));
}

function renderPredictionGraphic(target, predictions) {
  if (!predictions || predictions.length === 0) {
    qs(target).innerHTML = `<div class="muted">No prediction rows were produced.</div>`;
    return;
  }
  qs(target).innerHTML = predictions.map((row) => {
    const evPositive = String(row.EV) === "1";
    const confidence = Number(row.Confidence || 0);
    const f1Prob = Number(row.Fighter1_AI_Prob || 0);
    const f2Prob = Number(row.Fighter2_AI_Prob || 0);
    const f1Market = Number(row.Fighter1_Market_Prob || 0);
    const f2Market = Number(row.Fighter2_Market_Prob || 0);
    const valueSide = evPositive ? row.AI_Pick : "None";
    return `
      <article class="prediction-result">
        <div class="prediction-title">
          <strong>${escapeHtml(row.Fighter1)} vs ${escapeHtml(row.Fighter2)}</strong>
          <span class="${evPositive ? "ev-positive" : "ev-neutral"}">${evPositive ? "Positive EV" : "No positive EV"}</span>
        </div>
        <div class="fighter-row">
          <span>${escapeHtml(row.Fighter1)}</span>
          <meter min="0" max="100" value="${f1Prob}"></meter>
          <strong>${f1Prob.toFixed(1)}%</strong>
          <small>Market ${f1Market.toFixed(1)}% | ${escapeHtml(row.Fighter1_Odds || "N/A")}</small>
        </div>
        <div class="fighter-row">
          <span>${escapeHtml(row.Fighter2)}</span>
          <meter min="0" max="100" value="${f2Prob}"></meter>
          <strong>${f2Prob.toFixed(1)}%</strong>
          <small>Market ${f2Market.toFixed(1)}% | ${escapeHtml(row.Fighter2_Odds || "N/A")}</small>
        </div>
        <div class="value-strip">
          <span>Value Side</span>
          <strong>${escapeHtml(valueSide)}</strong>
        </div>
        <div class="prediction-foot">
          <span>Pick: <strong>${escapeHtml(row.AI_Pick)}</strong></span>
          <span>Confidence: <strong>${confidence.toFixed(1)}%</strong></span>
          <span>AI Odds: <strong>${escapeHtml(row.AI_Odds)}</strong></span>
        </div>
      </article>`;
  }).join("");
}

function renderUpcomingEvents(payload) {
  const events = payload?.events || [];
  if (!events.length) {
    qs("#events-output").innerHTML = `<div class="muted">${escapeHtml(payload?.warning || "No upcoming UFC events found.")}</div>`;
    return;
  }

  qs("#events-output").innerHTML = `
    <div class="event-list">
      ${events.map((event, index) => {
        const firstFight = event.fights?.[0];
        const date = firstFight?.date ? new Date(firstFight.date).toLocaleDateString() : "Date pending";
        const selected = Number(event.upcoming_number) === selectedUpcomingNumber || (selectedUpcomingNumber === null && index === 0);
        const preview = (event.fights || []).slice(0, 4)
          .map((fight) => `<li>${escapeHtml(fight.fighter1)} vs ${escapeHtml(fight.fighter2)}</li>`)
          .join("");
        return `
          <button class="event-card${selected ? " selected" : ""}" data-upcoming-number="${event.upcoming_number}">
            <span class="event-card-top">
              <strong>${escapeHtml(event.name)}</strong>
              <span>${escapeHtml(date)}</span>
            </span>
            <span class="event-card-meta">${event.fights?.length || 0} fights - event #${event.upcoming_number}</span>
            <ul>${preview || "<li>No matched fights yet</li>"}</ul>
          </button>`;
      }).join("")}
    </div>
    ${payload.warning ? `<div class="muted">${escapeHtml(payload.warning)}</div>` : ""}`;

  qsa(".event-card").forEach((card) => {
    card.addEventListener("click", () => {
      selectedUpcomingNumber = Number(card.dataset.upcomingNumber);
      qs("#predict-upcoming").value = String(selectedUpcomingNumber);
      qsa(".event-card").forEach((item) => item.classList.remove("selected"));
      card.classList.add("selected");
    });
  });
  if (!events.some((event) => Number(event.upcoming_number) === selectedUpcomingNumber)) {
    selectedUpcomingNumber = Number(events[0].upcoming_number);
    qs("#predict-upcoming").value = String(selectedUpcomingNumber);
    const firstCard = qs(".event-card");
    if (firstCard) firstCard.classList.add("selected");
  }
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
  qs("#eval-output").innerHTML = `
    <div>
      <h2>${escapeHtml(summary.model_name)}</h2>
      <p>${escapeHtml(summary.model_path)}</p>
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
    activeTrainingJobId = null;
  } else if (trainingJob?.state === "failed") {
    renderJson("#eval-output", trainingJob.error || "Training failed");
    activeTrainingJobId = null;
  }
  const dataJob = jobs.find((job) => job.id === activeDataJobId);
  if (activeDataJobId) await renderJobLog("#data-log", activeDataJobId);
  if (dataJob?.state === "succeeded") {
    renderDataRefreshResult(dataJob.result || {});
    activeDataJobId = null;
    await refreshStatus().catch(() => {});
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
      qsa(".subtab").forEach((item) => item.classList.remove("active"));
      qsa(".subpanel").forEach((item) => item.classList.remove("active"));
      tab.classList.add("active");
      qs(`#${tab.dataset.subtab}`).classList.add("active");
    });
  });
}

function wireData() {
  qs("#refresh-status").addEventListener("click", refreshStatus);
  qs("#run-data").addEventListener("click", async () => {
    try {
      const payload = {
        scrape: qs("#data-scrape").checked,
        rebuild: qs("#data-rebuild").checked,
        reset_db: qs("#data-reset").checked,
        force_full: qs("#data-force").checked,
        odds: qs("#data-odds").checked,
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
      if (result.chart) Plotly.newPlot("analytics-chart", result.chart.data, result.chart.layout, { responsive: true });
    } catch (error) {
      renderJson("#analytics-output", error.message);
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
  qs("#run-train-chat").addEventListener("click", async () => {
    try {
      const result = await api("/api/train/chat", {
        method: "POST",
        body: JSON.stringify({
          question: qs("#train-chat-question").value,
          model_path: qs("#train-eval-model").value || null,
        }),
      });
      renderJson("#train-chat-output", result.answer);
    } catch (error) {
      renderJson("#train-chat-output", error.message);
    }
  });
}

async function refreshModels() {
  const { models } = await api("/api/predict/models");
  const options = `<option value="">Latest ${models.length ? "" : "(none found)"}</option>` +
    models.map((model) => `<option value="${escapeHtml(model.path)}">${escapeHtml(model.name)}</option>`).join("");
  qs("#predict-model").innerHTML = options;
  qs("#train-eval-model").innerHTML = options;
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

function wirePrediction() {
  qs("#predict-upcoming").addEventListener("input", () => {
    selectedUpcomingNumber = Number(qs("#predict-upcoming").value || 1);
    qsa(".event-card").forEach((card) => {
      card.classList.toggle("selected", Number(card.dataset.upcomingNumber) === selectedUpcomingNumber);
    });
  });
  qs("#load-events").addEventListener("click", async () => {
    try {
      selectedUpcomingNumber = Number(qs("#predict-upcoming").value || 1);
      const params = new URLSearchParams({ limit: "5" });
      const predictionCsv = predictionDataCsv();
      if (predictionCsv) params.set("prediction_data_csv", predictionCsv);
      renderUpcomingEvents(await api(`/api/predict/upcoming?${params.toString()}`));
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
      const payload = {
        model_type: qs("#predict-model-type").value,
        model_path: qs("#predict-model").value || null,
        prediction_data_csv: predictionDataCsv(),
        training_data_csv: trainingDataCsv(),
        output_dir: predictionOutputDir(),
        upcoming_number: selectedUpcomingNumber || Number(qs("#predict-upcoming").value),
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
loadDashboardDefaults().catch(() => {});
refreshStatus().catch(() => {});
refreshModels().catch(() => {});
refreshJobs().catch(() => {});
setInterval(refreshJobs, 5000);
if (window.lucide) window.lucide.createIcons();
