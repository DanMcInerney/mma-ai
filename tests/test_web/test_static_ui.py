from pathlib import Path


STATIC_DIR = Path(__file__).resolve().parents[2] / "libs" / "web" / "static"


def test_prediction_card_renderer_exposes_value_and_market_context():
    html = (STATIC_DIR / "index.html").read_text(encoding="utf-8")
    app_js = (STATIC_DIR / "app.js").read_text(encoding="utf-8")
    styles = (STATIC_DIR / "styles.css").read_text(encoding="utf-8")

    assert 'src="/vendor/plotly.min.js"' in html
    assert 'src="/static/icons.js' in html
    assert 'src="/static/app.js?v=analytics-ask-20260605"' in html
    assert 'href="/static/styles.css?v=analytics-ask-20260605"' in html
    assert "cdn.plot.ly" not in html
    assert "unpkg.com" not in html
    assert "Pick" in app_js
    assert "Pick Edge" in app_js
    assert "AI Margin" in app_js
    assert "AI Odds" in app_js
    assert "+EV" in app_js
    assert "function probabilityToAmericanOdds(value)" in app_js
    assert "function expectedValuePercent(aiProb, bookOdds)" in app_js
    assert "function modelMargin(row)" in app_js
    assert "function renderFighterPickLine(" in app_js
    assert "function sideState(" in app_js
    assert "function formatOdds(value)" in app_js
    assert "Fighter1_Market_Prob" in app_js
    assert "Fighter2_Market_Prob" in app_js
    assert "Fighter1_Odds" in app_js
    assert "Fighter2_Odds" in app_js
    assert 'id="events-output" class="prediction-output"' in html
    assert 'id="prediction-output" class="prediction-output"' in html
    assert ".prediction-output" in styles
    assert ".pick-card" in styles
    assert ".pick-card-banner" in styles
    assert ".fighter-pick-line" in styles
    assert ".signal-pill" in styles
    assert ".ev-yes" in styles


def test_local_icon_bundle_covers_dashboard_icons():
    html = (STATIC_DIR / "index.html").read_text(encoding="utf-8")
    icons_js = (STATIC_DIR / "icons.js").read_text(encoding="utf-8")

    icon_names = {
        token.split('"', 1)[0]
        for token in html.split('data-lucide="')[1:]
    }

    assert icon_names
    for icon_name in icon_names:
        assert f'{icon_name}:' in icons_js or f'"{icon_name}":' in icons_js
    assert "window.lucide = { createIcons }" in icons_js


def test_primary_workflow_buttons_render_api_errors_in_place():
    app_js = (STATIC_DIR / "app.js").read_text(encoding="utf-8")

    assert "function apiErrorMessage(detail)" in app_js
    assert app_js.count("catch (error)") >= 7
    assert 'renderJson("#data-output", error.message)' in app_js
    assert 'renderJson("#events-output", error.message)' in app_js
    assert 'renderJson("#prediction-output", error.message)' in app_js


def test_data_ui_is_simplified_and_train_tab_is_removed():
    html = (STATIC_DIR / "index.html").read_text(encoding="utf-8")
    app_js = (STATIC_DIR / "app.js").read_text(encoding="utf-8")

    assert "<h2>Data</h2>" in html
    assert "<span>Update Data</span>" in html
    assert "Raw to Finalized Data" not in html
    assert "Pipeline Options" not in html
    assert "Training Chat" not in html
    assert "run-train-chat" not in html
    assert "run-train-chat" not in app_js
    assert 'data-tab="train"' not in html
    assert 'id="train"' not in html
    assert "Train Model" not in html
    assert "Advanced Training Knobs" not in html
    assert "run-train" not in html
    assert "wireTraining" not in app_js
    assert "/api/train" not in app_js
    assert "scrape: true" in app_js
    assert "rebuild: true" in app_js
    assert "reset_db: true" in app_js
    assert "odds_features: true" in app_js
    assert "row_deltas" in app_js
    assert "function renderDataMetrics(status, deltas = {})" in app_js
    assert "this run" in app_js


def test_manual_matchup_validates_required_fighter_names_before_api_call():
    html = (STATIC_DIR / "index.html").read_text(encoding="utf-8")
    app_js = (STATIC_DIR / "app.js").read_text(encoding="utf-8")

    assert 'qs("#fighter1").value.trim()' in app_js
    assert 'qs("#fighter2").value.trim()' in app_js
    assert 'id="fight-date" type="date"' in html
    assert "function todayDateInputValue()" in app_js
    assert 'setValue("#fight-date", qs("#fight-date").value || todayDateInputValue())' in app_js
    assert 'fight_date: qs("#fight-date").value || null' in app_js
    assert "Enter both fighter names before prediction." in app_js


def test_training_dashboard_controls_are_removed():
    html = (STATIC_DIR / "index.html").read_text(encoding="utf-8")
    app_js = (STATIC_DIR / "app.js").read_text(encoding="utf-8")

    for element_id in [
        "run-train",
        "train-log",
        "train-jobs",
        "train-eval-model",
        "train-feature-list",
        "train-include-patterns",
        "train-exclude-patterns",
        "train-required-features",
    ]:
        assert f'id="{element_id}"' not in html
        assert f'#{element_id}' not in app_js

    assert "activeTrainingJobId" not in app_js
    assert "renderEvaluation" not in app_js


def test_prediction_advanced_csv_controls_are_wired():
    html = (STATIC_DIR / "index.html").read_text(encoding="utf-8")
    app_js = (STATIC_DIR / "app.js").read_text(encoding="utf-8")

    for element_id in ["predict-data-csv", "predict-training-csv", "predict-output-dir"]:
        assert f'id="{element_id}"' in html
        assert f'qs("#{element_id}").value.trim()' in app_js

    assert "prediction_data_csv: predictionDataCsv()" in app_js
    assert "training_data_csv: trainingDataCsv()" in app_js
    assert "output_dir: predictionOutputDir()" in app_js
    assert 'params.set("prediction_data_csv", predictionCsv)' in app_js
    assert "/api/predict/fighters" in app_js
    assert "/api/predict/upcoming" in app_js


def test_predict_model_dropdown_filters_with_selected_target():
    html = (STATIC_DIR / "index.html").read_text(encoding="utf-8")
    app_js = (STATIC_DIR / "app.js").read_text(encoding="utf-8")
    styles = (STATIC_DIR / "styles.css").read_text(encoding="utf-8")

    assert 'id="predict-model-status"' in html
    assert '<label class="span-two">Model <select id="predict-model"></select></label>' in html
    assert '<input id="predict-odds" type="checkbox" /> Odds' in html
    assert '<label><input id="predict-shap" type="checkbox" /> Include SHAP</label>' in html
    assert "These live prediction odds are not passed into the model; they solely exist to calculate the Expected Value of each prediction." in html
    assert "Creates per-feature reasoning for why a fighter was picked" in html
    assert html.index("Advanced Prediction Knobs") < html.index('id="predict-model-type"')
    assert 'id="predict-flaresolverr"' in html
    assert "Use this only when BestFightOdds is blocking odds scraping." in html
    assert 'id="matchup-odds"' not in html
    assert "Matchup Odds" not in html
    assert 'id="fighter1-odds"' not in html
    assert 'id="fighter2-odds"' not in html
    assert 'odds: qs("#predict-odds").checked' in app_js
    assert 'flaresolverr: qs("#predict-flaresolverr").checked' in app_js
    assert "odds_fighter1" not in app_js
    assert "odds_fighter2" not in app_js
    assert 'id="run-event-predict" class="primary wide" disabled' in html
    assert 'id="run-matchup" class="primary wide" disabled' in html
    assert 'api(`/api/predict/models?model_type=${encodeURIComponent(modelType)}`)' in app_js
    assert 'qs("#predict-model-type").addEventListener("change"' in app_js
    assert "function modelOptions(models)" in app_js
    assert "function renderPredictModelState(modelType, models)" in app_js
    assert "No models found" in app_js
    assert "No model is available for this target" in app_js
    assert "function updatePredictionButtons()" in app_js
    assert 'setDisabled("#run-event-predict", !predictModelsAvailable || !selectedEventHasMatchedFights)' in app_js
    assert 'setDisabled("#run-matchup", !predictModelsAvailable)' in app_js
    assert ".model-status.blocked" in styles
    assert "button:disabled" in styles


def test_dynamic_select_options_escape_backend_values():
    app_js = (STATIC_DIR / "app.js").read_text(encoding="utf-8")

    assert '<option value="${escapeHtml(model.path)}">${escapeHtml(model.name)}</option>' in app_js
    assert '<option value="${escapeHtml(name)}"></option>' in app_js


def test_dashboard_controls_hydrate_from_defaults_endpoint():
    app_js = (STATIC_DIR / "app.js").read_text(encoding="utf-8")

    assert 'api("/api/defaults")' in app_js
    assert "function applyDashboardDefaults(defaults)" in app_js
    assert "setValue(\"#analytics-max-rows\", data.analytics_max_rows)" in app_js
    assert "const train = defaults.train" not in app_js
    assert "selectedUpcomingNumber = Number(predict.upcoming_number || 1)" in app_js
    assert 'setChecked("#predict-odds", predict.odds)' in app_js
    assert 'setChecked("#predict-flaresolverr", predict.flaresolverr)' in app_js
    assert "const existingSelection = select?.value ? Number(select.value) : null" in app_js
    assert ": Number(events[0].upcoming_number)" in app_js
    assert "loadDashboardDefaults().catch(() => {}).finally(() => loadUpcomingEventsWithStatus())" in app_js


def test_dashboard_surfaces_setup_readiness_state():
    html = (STATIC_DIR / "index.html").read_text(encoding="utf-8")
    app_js = (STATIC_DIR / "app.js").read_text(encoding="utf-8")
    styles = (STATIC_DIR / "styles.css").read_text(encoding="utf-8")

    assert 'id="readiness-badge"' in html
    assert 'aria-live="polite"' in html
    assert 'fetch("/api/readiness")' in app_js
    assert "function renderReadiness(payload)" in app_js
    assert "Setup incomplete" in app_js
    assert "Ready for predictions" in app_js
    assert "refreshReadiness().catch(() => {})" in app_js
    assert "Promise.allSettled([refreshStatus(), refreshReadiness(), refreshAnalyticsStatus()])" in app_js
    assert ".readiness-badge.ready" in styles
    assert ".readiness-badge.not-ready" in styles


def test_predict_tab_auto_loads_upcoming_event_dropdown_with_odds_context():
    html = (STATIC_DIR / "index.html").read_text(encoding="utf-8")
    app_js = (STATIC_DIR / "app.js").read_text(encoding="utf-8")

    assert 'id="predict-event"' in html
    assert 'id="event-preview"' in html
    assert "Loading upcoming events..." in html
    assert "These live event odds are not passed into the model" in html
    assert "async function loadUpcomingEvents()" in app_js
    assert "function renderUpcomingEventsError(message)" in app_js
    assert "async function loadUpcomingEventsWithStatus()" in app_js
    assert "function updateEventPreview()" in app_js
    assert "const raw = event?.date || event?.fights?.[0]?.date;" in app_js
    assert "const params = new URLSearchParams();" in app_js
    assert 'api(`/api/predict/upcoming${query ? `?${query}` : ""}`)' in app_js
    assert "Could not load upcoming events" in app_js
    assert '<option value="${event.upcoming_number}">${escapeHtml(event.name)}</option>' in app_js
    assert 'qs("#predict-event").addEventListener("change"' in app_js
    assert 'upcoming_number: upcomingNumber' in app_js
    assert "Choose an upcoming event before prediction." in app_js
    assert "The selected event has no matched fights yet" in app_js
    assert "selectedEventHasMatchedFights = fights.length > 0" in app_js
    assert 'qs("#event-preview").innerHTML = `<div class="muted">Loading upcoming UFC events...</div>`' in app_js
    assert "loadDashboardDefaults().catch(() => {}).finally(() => loadUpcomingEventsWithStatus())" in app_js


def test_analytics_options_expose_bounded_row_limit():
    html = (STATIC_DIR / "index.html").read_text(encoding="utf-8")
    app_js = (STATIC_DIR / "app.js").read_text(encoding="utf-8")
    styles = (STATIC_DIR / "styles.css").read_text(encoding="utf-8")

    assert 'id="analytics-max-rows" type="number" min="1" max="1000" value="100"' in html
    assert 'id="analytics-status"' in html
    assert 'id="analytics-copy-status"' in html
    assert 'id="copy-analytics-system-prompt"' in html
    assert "Copy Analytics Agent System Prompt" in html
    assert 'api("/api/data/analytics/system-prompt")' in app_js
    assert "function copyText(value)" in app_js
    assert "Analytics agent system prompt copied." in app_js
    assert "max_rows: Number(qs(\"#analytics-max-rows\").value || 100)" in app_js
    assert "function renderAnalyticsReport(result)" in app_js
    assert "function renderAnalyticsLoading()" in app_js
    assert "function setAnalyticsBusy(isBusy)" in app_js
    assert "function renderPlotlyCharts(target, charts)" in app_js
    assert "normalizeAnalyticsCharts(result)" in app_js
    assert "function renderPlotlyChart(target, chart)" not in app_js
    assert "function refreshAnalyticsStatus()" in app_js
    assert 'api("/api/data/analytics/status")' in app_js
    assert "LLM analytics ready" in app_js
    assert "Plotly.purge(element)" in app_js
    assert "renderAnalyticsReport(result)" in app_js
    assert "renderAnalyticsError(error.message)" in app_js
    assert "Enter an analytics question with at least 3 characters." in app_js
    assert "setAnalyticsBusy(true)" in app_js
    assert "renderAnalyticsLoading()" in app_js
    assert "setAnalyticsBusy(false)" in app_js
    assert 'id="analytics-output" class="result analytics-report"' in html
    assert 'id="analytics-chart" class="chart analytics-chart-grid"' in html
    assert ".analytics-status.ready" in styles
    assert ".analytics-status.blocked" in styles
    assert ".copy-status" in styles
    assert ".analytics-chart-card" in styles
    assert ".analytics-table" in styles


def test_debug_logs_and_manual_event_odds_are_wired():
    html = (STATIC_DIR / "index.html").read_text(encoding="utf-8")
    app_js = (STATIC_DIR / "app.js").read_text(encoding="utf-8")
    styles = (STATIC_DIR / "styles.css").read_text(encoding="utf-8")

    for element_id in ["data-log", "events-log", "prediction-log", "event-manual-odds"]:
        assert f'id="{element_id}"' in html

    assert "function parseManualOdds(value)" in app_js
    assert 'manual_odds: qs("#predict-odds").checked ? parseManualOdds(qs("#event-manual-odds").value) : null' in app_js
    assert "function setLogText(selector, value)" in app_js
    assert "async function renderJobLog(target, jobId)" in app_js
    assert "/api/jobs/${jobId}/log" in app_js
    assert "Prediction results will appear here when the job finishes." in app_js
    assert "Output Log" in html
    assert ".debug-log" in styles


def test_sticky_job_footer_does_not_cover_lower_predict_controls():
    styles = (STATIC_DIR / "styles.css").read_text(encoding="utf-8")

    assert "main {\n  padding: 24px 24px 72px;\n}" in styles
    assert "footer {\n  position: sticky;" in styles


def test_successful_background_jobs_refresh_dependent_dashboard_state():
    app_js = (STATIC_DIR / "app.js").read_text(encoding="utf-8")

    assert "await loadUpcomingEventsWithStatus();" in app_js
