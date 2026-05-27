import os
import socket
from contextlib import closing
from threading import Thread
import time
from urllib.error import URLError
from urllib.request import urlopen

import pytest
from fastapi.testclient import TestClient

from libs.web.app import create_app


def _wait_for_job(client: TestClient, job_id: str) -> dict:
    for _ in range(100):
        job = client.get(f"/api/jobs/{job_id}").json()
        if job["state"] in {"succeeded", "failed"}:
            return job
        time.sleep(0.01)
    raise AssertionError(f"Job did not finish: {job_id}")


def _free_port() -> int:
    with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def _start_uvicorn(app, port: int):
    import uvicorn

    server = uvicorn.Server(
        uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning", lifespan="off")
    )
    thread = Thread(target=server.run, daemon=True)
    thread.start()
    base_url = f"http://127.0.0.1:{port}"
    for _ in range(100):
        try:
            with urlopen(f"{base_url}/api/health", timeout=0.5) as response:
                if response.status == 200:
                    return server, thread, base_url
        except (OSError, URLError):
            time.sleep(0.05)
    server.should_exit = True
    thread.join(timeout=5)
    raise AssertionError("Uvicorn test server did not start")


def _fake_status(tmp_path):
    raw_dir = tmp_path / "data" / "raw" / "ufcstats"
    data_dir = tmp_path / "data"
    return {
        "project_root": str(tmp_path),
        "database_url": "postgresql://postgres:***@localhost:5432/mma-ai",
        "raw_data_dir": str(raw_dir),
        "data_dir": str(data_dir),
        "raw_csvs": {
            "competitions": {"path": str(raw_dir / "competitions.csv"), "rows": 1},
            "individuals": {"path": str(raw_dir / "individuals.csv"), "rows": 1},
        },
        "model_csvs": {
            "prediction_data": {"path": str(data_dir / "prediction_data.csv"), "rows": 2},
            "training_data": {"path": str(data_dir / "training_data.csv"), "rows": 2},
            "training_data_dec": {"path": str(data_dir / "training_data_dec.csv"), "rows": 2},
        },
    }


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
    assert 'id="predict-event"' in dashboard.text
    assert 'id="run-event-predict"' in dashboard.text

    app_js = client.get("/static/app.js").text
    assert "/api/predict/upcoming" in app_js
    assert "/api/predict/event" in app_js
    assert "loadUpcomingEvents" in app_js

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


def test_predict_tab_browser_predicts_next_ufc_event(monkeypatch, tmp_path):
    """Real browser e2e for the Predict tab next-event workflow."""
    if os.getenv("MMA_AI_RUN_BROWSER_E2E") != "1":
        pytest.skip("Set MMA_AI_RUN_BROWSER_E2E=1 to run the Selenium browser e2e.")

    from selenium import webdriver
    from selenium.webdriver.chrome.options import Options
    from selenium.webdriver.common.by import By
    from selenium.webdriver.support import expected_conditions as ec
    from selenium.webdriver.support.select import Select
    from selenium.webdriver.support.ui import WebDriverWait

    captured = {}

    def fake_upcoming(prediction_data_csv=None, limit=None):
        captured["upcoming"] = {"prediction_data_csv": prediction_data_csv, "limit": limit}
        return {
            "events": [
                {
                    "upcoming_number": 1,
                    "name": "UFC Browser E2E",
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
        print("predict-tab browser e2e fake prediction started")
        return {
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
            ]
        }

    monkeypatch.setenv("MMA_AI_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setattr("libs.web.app.get_data_status", lambda: _fake_status(tmp_path))
    monkeypatch.setattr(
        "libs.web.app.get_readiness_status",
        lambda: {"status": "ok", "ready": True, "checks": {"starter_model": {"ok": True}}},
    )
    monkeypatch.setattr(
        "libs.web.app.list_models",
        lambda model_type=None: [
            {
                "name": "ag-browser-e2e-win-extreme",
                "path": str(tmp_path / "models" / "ag-browser-e2e-win-extreme"),
                "modified_at": 1,
                "has_features": True,
                "has_predictor": True,
                "is_ensemble": False,
                "has_scaler": True,
                "has_calibrator": False,
            }
        ],
    )
    monkeypatch.setattr("libs.web.app.list_upcoming_events", fake_upcoming)
    monkeypatch.setattr("libs.web.app.validate_event_prediction_request", lambda request: {"status": "ready"})
    monkeypatch.setattr("libs.web.app.run_event_prediction", fake_event_prediction)

    server, thread, base_url = _start_uvicorn(create_app(), _free_port())
    options = Options()
    options.add_argument("--headless=new")
    options.add_argument("--disable-gpu")
    options.add_argument("--no-sandbox")
    options.add_argument("--window-size=1280,900")
    driver = None
    try:
        driver = webdriver.Chrome(options=options)
        wait = WebDriverWait(driver, 15)
        driver.get(base_url)
        wait.until(ec.element_to_be_clickable((By.CSS_SELECTOR, '[data-tab="predict"]'))).click()
        wait.until(ec.presence_of_element_located((By.CSS_SELECTOR, '#predict-event option[value="1"]')))
        Select(driver.find_element(By.ID, "predict-event")).select_by_value("1")
        odds_summary = wait.until(ec.element_to_be_clickable((By.XPATH, "//summary[normalize-space()='Manual Event Odds']")))
        driver.execute_script("arguments[0].scrollIntoView({block: 'center'});", odds_summary)
        odds_summary.click()
        odds_input = wait.until(ec.element_to_be_clickable((By.ID, "event-manual-odds")))
        odds_input.clear()
        odds_input.send_keys("fighter one=-120\nfighter two=100")

        driver.find_element(By.ID, "run-event-predict").click()

        result = wait.until(ec.presence_of_element_located((By.CSS_SELECTOR, "#events-output .pick-card")))
        wait.until(ec.text_to_be_present_in_element((By.ID, "events-output"), "AI Odds"))
        wait.until(ec.text_to_be_present_in_element((By.ID, "events-log"), "predict-tab browser e2e fake prediction started"))

        assert "fighter one" in result.text
        assert "Odds" in result.text
        assert "AI Odds" in result.text
        assert "AI Margin" in result.text
        assert "+EV" in result.text
        assert "Pick Edge" in result.text
        assert "EV" in result.text
        assert captured["upcoming"]["limit"] is None
        assert captured["prediction_request"]["upcoming_number"] == 1
        assert captured["prediction_request"]["manual_odds"] == {"fighter one": -120, "fighter two": 100}
    finally:
        if driver is not None:
            driver.quit()
        server.should_exit = True
        thread.join(timeout=5)
