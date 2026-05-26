from pathlib import Path


STATIC_DIR = Path(__file__).resolve().parents[2] / "libs" / "web" / "static"


def test_prediction_card_renderer_exposes_value_and_market_context():
    html = (STATIC_DIR / "index.html").read_text(encoding="utf-8")
    app_js = (STATIC_DIR / "app.js").read_text(encoding="utf-8")
    styles = (STATIC_DIR / "styles.css").read_text(encoding="utf-8")

    assert 'src="/vendor/plotly.min.js"' in html
    assert 'src="/static/icons.js"' in html
    assert "cdn.plot.ly" not in html
    assert "unpkg.com" not in html
    assert "Value Side" in app_js
    assert "Model Pick" in app_js
    assert "Model Edge" in app_js
    assert "Pick Edge" in app_js
    assert "function formatOdds(value)" in app_js
    assert "confidence-pill" in app_js
    assert "Fighter1_Market_Prob" in app_js
    assert "Fighter2_Market_Prob" in app_js
    assert "Fighter1_Odds" in app_js
    assert "Fighter2_Odds" in app_js
    assert ".edge-positive" in styles
    assert ".value-strip" in styles


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

    assert app_js.count("catch (error)") >= 8
    assert 'renderJson("#data-output", error.message)' in app_js
    assert 'renderJson("#train-jobs", error.message)' in app_js
    assert 'renderJson("#events-output", error.message)' in app_js
    assert 'renderJson("#prediction-output", error.message)' in app_js


def test_data_and_training_ui_are_simplified():
    html = (STATIC_DIR / "index.html").read_text(encoding="utf-8")
    app_js = (STATIC_DIR / "app.js").read_text(encoding="utf-8")

    assert "<h2>Data</h2>" in html
    assert "<span>Update Data</span>" in html
    assert "Raw to Finalized Data" not in html
    assert "Pipeline Options" not in html
    assert "Training Chat" not in html
    assert "run-train-chat" not in html
    assert "run-train-chat" not in app_js
    assert "scrape: true" in app_js
    assert "rebuild: true" in app_js
    assert "reset_db: true" in app_js
    assert html.index("Advanced Training Knobs") < html.index('id="train-model-type"')


def test_manual_matchup_validates_required_fighter_names_before_api_call():
    html = (STATIC_DIR / "index.html").read_text(encoding="utf-8")
    app_js = (STATIC_DIR / "app.js").read_text(encoding="utf-8")

    assert 'qs("#fighter1").value.trim()' in app_js
    assert 'qs("#fighter2").value.trim()' in app_js
    assert 'id="fight-date" type="date"' in html
    assert 'fight_date: qs("#fight-date").value || null' in app_js
    assert "Enter both fighter names before prediction." in app_js


def test_training_advanced_feature_filter_controls_are_wired():
    html = (STATIC_DIR / "index.html").read_text(encoding="utf-8")
    app_js = (STATIC_DIR / "app.js").read_text(encoding="utf-8")

    for element_id in [
        "train-feature-list",
        "train-include-patterns",
        "train-exclude-patterns",
        "train-required-features",
    ]:
        assert f'id="{element_id}"' in html
        assert f'qs("#{element_id}").value' in app_js

    assert "feature_list" in app_js
    assert "included_strings" in app_js
    assert "excluded_strings" in app_js
    assert "required_strings" in app_js


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
    assert 'id="run-event-predict" class="primary wide" disabled' in html
    assert 'id="run-matchup" class="primary wide" disabled' in html
    assert 'api(`/api/predict/models?model_type=${encodeURIComponent(modelType)}`)' in app_js
    assert 'qs("#predict-model-type").addEventListener("change"' in app_js
    assert "function modelOptions(models)" in app_js
    assert "function renderPredictModelState(modelType, models)" in app_js
    assert "No models found" in app_js
    assert "No model is available for this target" in app_js
    assert 'setDisabled("#run-event-predict", !predictModelsAvailable)' in app_js
    assert 'setDisabled("#run-matchup", !predictModelsAvailable)' in app_js
    assert 'api("/api/predict/models")' in app_js
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
    assert "setValue(\"#train-time-limit\", train.time_limit)" in app_js
    assert "setValue(\"#train-split\", train.split_strategy)" in app_js
    assert "setValue(\"#train-model-families\", listValue(train.included_model_types))" in app_js
    assert "setChecked(\"#train-refit\", train.refit_full)" in app_js
    assert "selectedUpcomingNumber = Number(predict.upcoming_number || 1)" in app_js
    assert "const existingSelection = select?.value ? Number(select.value) : null" in app_js
    assert ": Number(events[0].upcoming_number)" in app_js
    assert "loadDashboardDefaults().catch(() => {}).finally(() => loadUpcomingEvents().catch(() => {}))" in app_js


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
    assert "Promise.allSettled([refreshStatus(), refreshReadiness()])" in app_js
    assert ".readiness-badge.ready" in styles
    assert ".readiness-badge.not-ready" in styles


def test_predict_tab_auto_loads_upcoming_event_dropdown_with_odds_context():
    html = (STATIC_DIR / "index.html").read_text(encoding="utf-8")
    app_js = (STATIC_DIR / "app.js").read_text(encoding="utf-8")

    assert 'id="predict-event"' in html
    assert 'id="event-preview"' in html
    assert "Loading upcoming events..." in html
    assert "Odds are not included in the model" in html
    assert "async function loadUpcomingEvents()" in app_js
    assert "function updateEventPreview()" in app_js
    assert 'new URLSearchParams({ limit: "20" })' in app_js
    assert 'qs("#predict-event").addEventListener("change"' in app_js
    assert 'upcoming_number: upcomingNumber' in app_js
    assert "Choose an upcoming event before prediction." in app_js
    assert 'qs("#event-preview").innerHTML = `<div class="muted">Loading upcoming UFC events...</div>`' in app_js


def test_analytics_options_expose_bounded_row_limit():
    html = (STATIC_DIR / "index.html").read_text(encoding="utf-8")
    app_js = (STATIC_DIR / "app.js").read_text(encoding="utf-8")

    assert 'id="analytics-max-rows" type="number" min="1" max="1000" value="100"' in html
    assert "max_rows: Number(qs(\"#analytics-max-rows\").value || 100)" in app_js
    assert "function renderPlotlyChart(target, chart)" in app_js
    assert "Plotly.purge(element)" in app_js
    assert 'renderPlotlyChart("#analytics-chart", result.chart)' in app_js
    assert 'renderPlotlyChart("#analytics-chart", null)' in app_js


def test_debug_logs_and_manual_event_odds_are_wired():
    html = (STATIC_DIR / "index.html").read_text(encoding="utf-8")
    app_js = (STATIC_DIR / "app.js").read_text(encoding="utf-8")
    styles = (STATIC_DIR / "styles.css").read_text(encoding="utf-8")

    for element_id in ["data-log", "train-log", "events-log", "prediction-log", "event-manual-odds"]:
        assert f'id="{element_id}"' in html

    assert "function parseManualOdds(value)" in app_js
    assert 'manual_odds: parseManualOdds(qs("#event-manual-odds").value)' in app_js
    assert "async function renderJobLog(target, jobId)" in app_js
    assert "/api/jobs/${jobId}/log" in app_js
    assert ".debug-log" in styles


def test_sticky_job_footer_does_not_cover_lower_predict_controls():
    styles = (STATIC_DIR / "styles.css").read_text(encoding="utf-8")

    assert "main {\n  padding: 24px 24px 72px;\n}" in styles
    assert "footer {\n  position: sticky;" in styles
