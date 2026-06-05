import time

from fastapi.testclient import TestClient

from libs.web.app import create_app
from libs.web.models import TrainingRequest


def test_health_endpoint():
    client = TestClient(create_app())
    response = client.get("/api/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_app_startup_warms_up_upcoming_event_cache(monkeypatch):
    calls = []

    monkeypatch.setattr("libs.web.app.warm_up_upcoming_events", lambda: calls.append("warm"))

    with TestClient(create_app()):
        pass

    assert calls == ["warm"]


def test_readiness_endpoint_returns_ready_payload(monkeypatch):
    monkeypatch.setattr(
        "libs.web.app.get_readiness_status",
        lambda: {"status": "ok", "ready": True, "checks": {"database": {"ok": True}}},
    )
    client = TestClient(create_app())

    response = client.get("/api/readiness")

    assert response.status_code == 200
    assert response.json()["ready"] is True


def test_readiness_endpoint_returns_503_until_prediction_stack_is_ready(monkeypatch):
    monkeypatch.setattr(
        "libs.web.app.get_readiness_status",
        lambda: {"status": "not_ready", "ready": False, "checks": {"starter_model": {"ok": False}}},
    )
    client = TestClient(create_app())

    response = client.get("/api/readiness")

    assert response.status_code == 503
    assert response.json()["detail"]["checks"]["starter_model"]["ok"] is False


def test_plotly_vendor_bundle_is_served_locally():
    client = TestClient(create_app())

    response = client.get("/vendor/plotly.min.js")

    assert response.status_code == 200
    assert "javascript" in response.headers["content-type"]
    assert b"Plotly" in response.content[:5000]


def test_defaults_endpoint_includes_tabs():
    client = TestClient(create_app())
    response = client.get("/api/defaults")
    assert response.status_code == 200
    body = response.json()
    assert set(body) == {"data", "train", "predict"}
    assert body["data"]["analytics_max_rows"] == 100
    assert body["data"]["reset_db"] is True
    assert body["train"]["walkforward_n_windows"] == 4
    assert body["train"]["walkforward_initial_year"] == 2021
    assert body["train"]["refit_all"] is False
    assert body["train"]["feature_list"] is None
    assert body["train"]["included_strings"] is None
    assert body["predict"]["odds"] is True
    assert body["predict"]["flaresolverr"] is False


def test_train_defaults_are_valid_training_request_payload():
    client = TestClient(create_app())
    response = client.get("/api/defaults")
    assert response.status_code == 200

    request = TrainingRequest(**response.json()["train"])

    assert request.model_type == "win"
    assert request.time_limit == 3000
    assert request.split_strategy == "timeseries_split"
    assert request.included_model_types == ["TABICL", "MITRA", "TABM", "GBM_PREP", "CAT", "GBM", "REALTABPFN-V2"]


def test_analytics_endpoint_rejects_mutation_sql():
    client = TestClient(create_app())
    response = client.post(
        "/api/data/analytics",
        json={"question": "delete everything", "sql": "delete from features.fight_stats_fe"},
    )
    assert response.status_code == 400


def test_analytics_endpoint_reports_query_runtime_errors(monkeypatch):
    def fake_analytics(_question, _sql, _max_rows):
        raise RuntimeError("Database query failed and no finalized CSV fallback is available")

    monkeypatch.setattr("libs.web.app.run_analytics", fake_analytics)
    client = TestClient(create_app())

    response = client.post(
        "/api/data/analytics",
        json={"question": "show me rows", "sql": "select * from training_data", "max_rows": 10},
    )

    assert response.status_code == 400
    assert "Database query failed" in response.json()["detail"]


def test_analytics_status_endpoint_reports_non_secret_llm_status(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "local")
    monkeypatch.setenv("LLM_MODEL", "llama3.1")
    monkeypatch.delenv("LLM_API_KEY", raising=False)
    client = TestClient(create_app())

    response = client.get("/api/data/analytics/status")

    assert response.status_code == 200
    payload = response.json()
    assert payload["configured"] is True
    assert payload["provider"] == "local"
    assert payload["model"] == "llama3.1"
    assert payload["mode"] == "llm"
    assert "api_key" not in payload


def test_analytics_system_prompt_endpoint_returns_copyable_prompt():
    client = TestClient(create_app())

    response = client.get("/api/data/analytics/system-prompt")

    assert response.status_code == 200
    payload = response.json()
    assert payload["version"] == "2026-06-05"
    assert "MMA AI Data Tab analytics agent" in payload["system_prompt"]
    assert "_adjperf" in payload["system_prompt"]
    assert "features.odds" in payload["system_prompt"]
    assert "decimal odds" in payload["system_prompt"]
    assert "Treat it as post-fight" in payload["system_prompt"]


def test_data_refresh_endpoint_starts_background_job(monkeypatch):
    def fake_refresh(request):
        print("fake refresh stdout")
        return {
            "scrape_counts": {"fighters": 2},
            "status": {"model_csvs": {"training_data": {"rows": 10}}},
            "request": request.model_dump(),
        }

    monkeypatch.setattr("libs.web.app.run_data_refresh", fake_refresh)
    client = TestClient(create_app())

    response = client.post(
        "/api/data/refresh",
        json={"scrape": False, "rebuild": False, "force_full": True, "reset_db": False, "odds": True},
    )

    assert response.status_code == 200
    job_id = response.json()["job_id"]
    for _ in range(100):
        job_response = client.get(f"/api/jobs/{job_id}")
        if job_response.json()["state"] == "succeeded":
            break
        time.sleep(0.01)

    job = client.get(f"/api/jobs/{job_id}").json()
    assert job["state"] == "succeeded"
    assert job["result"]["scrape_counts"] == {"fighters": 2}
    assert job["result"]["request"]["force_full"] is True
    assert job["result"]["request"]["odds_features"] is True
    assert job["result"]["request"]["odds"] is True
    log_response = client.get(f"/api/jobs/{job_id}/log")
    assert log_response.status_code == 200
    assert "fake refresh stdout" in log_response.json()["log"]
    assert log_response.json()["log_path"]


def test_train_evaluations_endpoint_reports_missing_when_no_models(monkeypatch, tmp_path):
    monkeypatch.setenv("MMA_AI_MODELS_DIR", str(tmp_path / "models"))
    client = TestClient(create_app())

    response = client.get("/api/train/evaluations")

    assert response.status_code == 200
    assert response.json()["available"] is False


def test_predict_upcoming_endpoint_returns_events(monkeypatch):
    def fake_upcoming(prediction_data_csv=None, limit=5):
        return {
            "events": [
                {
                    "upcoming_number": 1,
                    "name": "UFC Test",
                    "fights": [{"date": "2026-06-01", "fighter1": "a", "fighter2": "b"}],
                }
            ],
            "warning": None,
        }

    monkeypatch.setattr("libs.web.app.list_upcoming_events", fake_upcoming)
    client = TestClient(create_app())

    response = client.get("/api/predict/upcoming?limit=1")

    assert response.status_code == 200
    assert response.json()["events"][0]["name"] == "UFC Test"


def test_predict_fighters_endpoint_rejects_unsafe_csv_path(monkeypatch, tmp_path):
    monkeypatch.setenv("MMA_AI_DATA_DIR", str(tmp_path / "data"))
    unsafe_csv = tmp_path / "outside" / "prediction_data.csv"
    unsafe_csv.parent.mkdir()
    unsafe_csv.write_text("fighter_name\nfighter one\n", encoding="utf-8")
    client = TestClient(create_app())

    response = client.get(f"/api/predict/fighters?prediction_data_csv={unsafe_csv}")

    assert response.status_code == 400
    assert "must be under" in response.json()["detail"]


def test_event_prediction_endpoint_rejects_unsafe_csv_path(monkeypatch, tmp_path):
    monkeypatch.setenv("MMA_AI_DATA_DIR", str(tmp_path / "data"))
    unsafe_csv = tmp_path / "outside" / "prediction_data.csv"
    unsafe_csv.parent.mkdir()
    unsafe_csv.write_text("fighter_name\nfighter one\n", encoding="utf-8")
    client = TestClient(create_app())

    response = client.post(
        "/api/predict/event",
        json={"prediction_data_csv": str(unsafe_csv), "upcoming_number": 1},
    )

    assert response.status_code == 400
    assert "must be under" in response.json()["detail"]


def test_event_prediction_endpoint_rejects_unsafe_output_dir(monkeypatch, tmp_path):
    data_dir = tmp_path / "data"
    outside = tmp_path / "outside" / "predictions"
    monkeypatch.setenv("MMA_AI_DATA_DIR", str(data_dir))
    data_dir.mkdir()
    client = TestClient(create_app())

    response = client.post(
        "/api/predict/event",
        json={"output_dir": str(outside), "upcoming_number": 1},
    )

    assert response.status_code == 400
    assert "output directory must be under" in response.json()["detail"]


def test_event_prediction_endpoint_rejects_invalid_manual_odds_before_job():
    client = TestClient(create_app())

    response = client.post(
        "/api/predict/event",
        json={"upcoming_number": 1, "manual_odds": {"fighter one": 0}},
    )

    assert response.status_code == 400
    assert "American odds" in response.json()["detail"]


def test_matchup_prediction_endpoint_rejects_unsafe_training_csv_path(monkeypatch, tmp_path):
    data_dir = tmp_path / "data"
    monkeypatch.setenv("MMA_AI_DATA_DIR", str(data_dir))
    data_dir.mkdir()
    prediction_csv = data_dir / "prediction_data.csv"
    prediction_csv.write_text("fighter_name\nfighter one\nfighter two\n", encoding="utf-8")
    unsafe_csv = tmp_path / "outside" / "training_data.csv"
    unsafe_csv.parent.mkdir()
    unsafe_csv.write_text("fighter1_name,y_true\nfighter one,1\n", encoding="utf-8")
    client = TestClient(create_app())

    response = client.post(
        "/api/predict/matchup",
        json={
            "prediction_data_csv": str(prediction_csv),
            "training_data_csv": str(unsafe_csv),
            "fighter1": "fighter one",
            "fighter2": "fighter two",
        },
    )

    assert response.status_code == 400
    assert "must be under" in response.json()["detail"]


def test_matchup_prediction_endpoint_rejects_blank_fighters_before_job(monkeypatch, tmp_path):
    data_dir = tmp_path / "data"
    monkeypatch.setenv("MMA_AI_DATA_DIR", str(data_dir))
    data_dir.mkdir()
    prediction_csv = data_dir / "prediction_data.csv"
    prediction_csv.write_text("fighter_name\nfighter one\nfighter two\n", encoding="utf-8")
    client = TestClient(create_app())

    response = client.post(
        "/api/predict/matchup",
        json={
            "prediction_data_csv": str(prediction_csv),
            "fighter1": " ",
            "fighter2": "fighter two",
        },
    )

    assert response.status_code == 400
    assert response.json()["detail"] == "Enter both fighter names before prediction."


def test_matchup_prediction_endpoint_rejects_invalid_fight_date_before_job(monkeypatch, tmp_path):
    data_dir = tmp_path / "data"
    monkeypatch.setenv("MMA_AI_DATA_DIR", str(data_dir))
    data_dir.mkdir()
    prediction_csv = data_dir / "prediction_data.csv"
    prediction_csv.write_text("fighter_name\nfighter one\nfighter two\n", encoding="utf-8")
    client = TestClient(create_app())

    response = client.post(
        "/api/predict/matchup",
        json={
            "prediction_data_csv": str(prediction_csv),
            "fighter1": "fighter one",
            "fighter2": "fighter two",
            "fight_date": "06/01/2026",
        },
    )

    assert response.status_code == 400
    assert response.json()["detail"] == "Fight date must use YYYY-MM-DD format."


def test_predict_models_endpoint_returns_starter_model(monkeypatch, tmp_path):
    models_dir = tmp_path / "AutogluonModels"
    starter_model = models_dir / "ag-20260304_110750-win-extreme"
    starter_model.mkdir(parents=True)
    for filename in ("predictor.pkl", "metadata.json", "feats.txt", "scaler.pkl"):
        (starter_model / filename).write_text("starter", encoding="utf-8")
    monkeypatch.setenv("MMA_AI_MODELS_DIR", str(models_dir))
    client = TestClient(create_app())

    response = client.get("/api/predict/models")

    assert response.status_code == 200
    models = response.json()["models"]
    assert [model["name"] for model in models] == ["ag-20260304_110750-win-extreme"]
    assert models[0]["has_features"] is True
    assert models[0]["has_scaler"] is True


def test_predict_models_endpoint_filters_by_model_type(monkeypatch, tmp_path):
    models_dir = tmp_path / "AutogluonModels"
    for model_name in ("ag-20260304_110750-win-extreme", "ag-20260304_110750-decision-best"):
        model_dir = models_dir / model_name
        model_dir.mkdir(parents=True)
        (model_dir / "feats.txt").write_text("feature\n", encoding="utf-8")
        (model_dir / "predictor.pkl").write_text("predictor", encoding="utf-8")
    monkeypatch.setenv("MMA_AI_MODELS_DIR", str(models_dir))
    client = TestClient(create_app())

    response = client.get("/api/predict/models?model_type=decision")

    assert response.status_code == 200
    assert [model["name"] for model in response.json()["models"]] == ["ag-20260304_110750-decision-best"]


def test_predict_models_endpoint_rejects_unknown_model_type():
    client = TestClient(create_app())

    response = client.get("/api/predict/models?model_type=style")

    assert response.status_code == 400
    assert response.json()["detail"] == "model_type must be win or decision."


def test_matchup_prediction_endpoint_rejects_missing_model_before_job(monkeypatch, tmp_path):
    data_dir = tmp_path / "data"
    models_dir = tmp_path / "models"
    monkeypatch.setenv("MMA_AI_DATA_DIR", str(data_dir))
    monkeypatch.setenv("MMA_AI_MODELS_DIR", str(models_dir))
    data_dir.mkdir()
    models_dir.mkdir()
    monkeypatch.setattr("libs.web.services.assert_prediction_runtime_dependencies", lambda: None)
    prediction_csv = data_dir / "prediction_data.csv"
    prediction_csv.write_text("fighter_name\nfighter one\nfighter two\n", encoding="utf-8")
    client = TestClient(create_app())

    response = client.post(
        "/api/predict/matchup",
        json={
            "prediction_data_csv": str(prediction_csv),
            "model_path": str(models_dir / "missing-model"),
            "fighter1": "fighter one",
            "fighter2": "fighter two",
        },
    )

    assert response.status_code == 400
    assert "Model directory not found" in response.json()["detail"]


def test_event_prediction_endpoint_rejects_missing_latest_model_before_job(monkeypatch, tmp_path):
    models_dir = tmp_path / "models"
    monkeypatch.setenv("MMA_AI_MODELS_DIR", str(models_dir))
    models_dir.mkdir()
    monkeypatch.setattr("libs.web.services.assert_prediction_runtime_dependencies", lambda: None)
    client = TestClient(create_app())

    response = client.post("/api/predict/event", json={"model_type": "win", "upcoming_number": 1})

    assert response.status_code == 400
    assert "No loadable win model found" in response.json()["detail"]
