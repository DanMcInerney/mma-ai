import time

from fastapi.testclient import TestClient

from libs.web.app import create_app
from libs.web.models import TrainingRequest


def test_health_endpoint():
    client = TestClient(create_app())
    response = client.get("/api/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


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


def test_train_chat_endpoint_uses_fallback(monkeypatch, tmp_path):
    for name in [
        "LLM_PROVIDER",
        "LLM_MODEL",
        "LLM_API_KEY",
        "LLM_BASE_URL",
        "OPENAI_API_KEY",
        "ANTHROPIC_API_KEY",
        "XAI_API_KEY",
        "GROK_API_KEY",
        "GEMINI_API_KEY",
        "GOOGLE_API_KEY",
    ]:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("MMA_AI_MODELS_DIR", str(tmp_path / "models"))
    client = TestClient(create_app())

    response = client.post("/api/train/chat", json={"question": "How do I avoid leakage?"})

    assert response.status_code == 200
    assert response.json()["used_llm"] is False


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


def test_matchup_prediction_endpoint_rejects_missing_model_before_job(monkeypatch, tmp_path):
    data_dir = tmp_path / "data"
    models_dir = tmp_path / "models"
    monkeypatch.setenv("MMA_AI_DATA_DIR", str(data_dir))
    monkeypatch.setenv("MMA_AI_MODELS_DIR", str(models_dir))
    data_dir.mkdir()
    models_dir.mkdir()
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
