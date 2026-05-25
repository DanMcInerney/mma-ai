import time

from fastapi.testclient import TestClient

from libs.web.app import create_app


def _wait_for_job(client: TestClient, job_id: str) -> dict:
    for _ in range(100):
        job = client.get(f"/api/jobs/{job_id}").json()
        if job["state"] in {"succeeded", "failed"}:
            return job
        time.sleep(0.01)
    raise AssertionError(f"Job did not finish: {job_id}")


def test_predict_tab_predicts_next_ufc_event(monkeypatch, tmp_path):
    """E2E smoke for the dashboard Predict tab's next-event workflow."""
    monkeypatch.setenv("MMA_AI_DATA_DIR", str(tmp_path / "data"))
    captured = {}

    def fake_upcoming(prediction_data_csv=None, limit=5):
        captured["upcoming"] = {"prediction_data_csv": prediction_data_csv, "limit": limit}
        return {
            "events": [
                {
                    "upcoming_number": 1,
                    "name": "UFC E2E Night",
                    "fights": [
                        {
                            "date": "2026-06-06T00:00:00",
                            "fighter1": "fighter one",
                            "fighter2": "fighter two",
                        }
                    ],
                }
            ],
            "warning": None,
        }

    def fake_event_prediction(request):
        captured["prediction_request"] = request.model_dump()
        print("predict-tab e2e fake prediction started")
        return {
            "output_dir": str(tmp_path / "data" / "predictions" / "latest"),
            "csv_path": str(tmp_path / "data" / "predictions" / "latest" / "fight_predictions.csv"),
            "predictions": [
                {
                    "Fighter1": "fighter one",
                    "Fighter2": "fighter two",
                    "Fighter1_Odds": "-120",
                    "Fighter2_Odds": "+100",
                    "Fighter1_AI_Prob": "55.0",
                    "Fighter2_AI_Prob": "45.0",
                    "Fighter1_Market_Prob": "52.0",
                    "Fighter2_Market_Prob": "48.0",
                    "AI_Pick": "fighter one",
                    "Confidence": "55.0",
                    "AI_Odds": "-122",
                    "EV": "1",
                }
            ],
        }

    monkeypatch.setattr("libs.web.app.list_upcoming_events", fake_upcoming)
    monkeypatch.setattr("libs.web.app.validate_event_prediction_request", lambda request: {"status": "ready"})
    monkeypatch.setattr("libs.web.app.run_event_prediction", fake_event_prediction)

    client = TestClient(create_app())

    dashboard = client.get("/")
    assert dashboard.status_code == 200
    assert 'data-tab="predict"' in dashboard.text
    assert 'id="load-events"' in dashboard.text
    assert 'id="run-event-predict"' in dashboard.text

    app_js = client.get("/static/app.js").text
    assert "/api/predict/upcoming" in app_js
    assert "/api/predict/event" in app_js

    upcoming = client.get("/api/predict/upcoming?limit=1")
    assert upcoming.status_code == 200
    event = upcoming.json()["events"][0]
    assert event["upcoming_number"] == 1
    assert event["name"] == "UFC E2E Night"

    response = client.post(
        "/api/predict/event",
        json={
            "model_type": "win",
            "upcoming_number": event["upcoming_number"],
            "odds": True,
            "manual_odds": {"fighter one": -120, "fighter two": 100},
            "shap": False,
        },
    )

    assert response.status_code == 200
    job = _wait_for_job(client, response.json()["job_id"])
    assert job["state"] == "succeeded"
    prediction = job["result"]["predictions"][0]
    assert prediction["AI_Pick"] == "fighter one"
    assert prediction["EV"] == "1"

    assert captured["upcoming"]["limit"] == 1
    assert captured["prediction_request"]["upcoming_number"] == 1
    assert captured["prediction_request"]["manual_odds"] == {"fighter one": -120, "fighter two": 100}

    log_response = client.get(f"/api/jobs/{job['id']}/log")
    assert log_response.status_code == 200
    assert "predict-tab e2e fake prediction started" in log_response.json()["log"]
