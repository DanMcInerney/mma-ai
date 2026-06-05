import os
import socket
import time
from contextlib import closing
from threading import Thread
from urllib.error import URLError
from urllib.request import urlopen

import pytest
from fastapi.testclient import TestClient

from libs.web.app import create_app


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


def test_data_tab_analytics_prompt_copy_smoke(monkeypatch, tmp_path):
    """Smoke-test the Data tab analytics prompt route and static wiring."""
    monkeypatch.setenv("MMA_AI_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setattr("libs.web.app.get_data_status", lambda: _fake_status(tmp_path))
    monkeypatch.setattr(
        "libs.web.app.get_readiness_status",
        lambda: {"status": "ok", "ready": True, "checks": {"starter_model": {"ok": True}}},
    )
    client = TestClient(create_app())

    dashboard = client.get("/")
    assert dashboard.status_code == 200
    assert 'id="copy-analytics-system-prompt"' in dashboard.text
    assert "Copy Analytics Agent System Prompt" in dashboard.text

    prompt_response = client.get("/api/data/analytics/system-prompt")
    assert prompt_response.status_code == 200
    prompt_payload = prompt_response.json()
    assert prompt_payload["version"] == "2026-06-05"
    assert "MMA AI Data Tab analytics agent" in prompt_payload["system_prompt"]
    assert "_adjperf" in prompt_payload["system_prompt"]
    assert "features.odds" in prompt_payload["system_prompt"]
    assert "Return strict JSON only" in prompt_payload["system_prompt"]

    app_js = client.get("/static/app.js").text
    icons_js = client.get("/static/icons.js").text
    assert 'api("/api/data/analytics/system-prompt")' in app_js
    assert "function copyText(value)" in app_js
    assert "Analytics agent system prompt copied." in app_js
    assert "copy:" in icons_js


def test_data_tab_browser_copies_analytics_system_prompt(monkeypatch, tmp_path):
    """Real browser e2e for the Data tab analytics prompt copy workflow."""
    if os.getenv("MMA_AI_RUN_BROWSER_E2E") != "1":
        pytest.skip("Set MMA_AI_RUN_BROWSER_E2E=1 to run the Selenium browser e2e.")

    from selenium import webdriver
    from selenium.webdriver.chrome.options import Options
    from selenium.webdriver.common.by import By
    from selenium.webdriver.support import expected_conditions as ec
    from selenium.webdriver.support.ui import WebDriverWait

    monkeypatch.setenv("MMA_AI_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setattr("libs.web.app.get_data_status", lambda: _fake_status(tmp_path))
    monkeypatch.setattr(
        "libs.web.app.get_readiness_status",
        lambda: {"status": "ok", "ready": True, "checks": {"starter_model": {"ok": True}}},
    )
    monkeypatch.setattr(
        "libs.web.app.get_analytics_status",
        lambda: {"configured": False, "mode": "sql_only", "hint": "test analytics status"},
    )

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

        copy_button = wait.until(ec.element_to_be_clickable((By.ID, "copy-analytics-system-prompt")))
        assert "Copy Analytics Agent System Prompt" in copy_button.text
        copy_button.click()

        status = wait.until(ec.presence_of_element_located((By.ID, "analytics-copy-status")))
        wait.until(ec.text_to_be_present_in_element((By.ID, "analytics-copy-status"), "copied"))
        assert status.text == "Analytics agent system prompt copied."

        prompt = driver.execute_script(
            "return fetch('/api/data/analytics/system-prompt').then(r => r.json()).then(j => j.system_prompt)"
        )
        assert "MMA AI Data Tab analytics agent" in prompt
        assert "_adjperf" in prompt
        assert "features.odds" in prompt
    finally:
        if driver is not None:
            driver.quit()
        server.should_exit = True
        thread.join(timeout=5)


def test_data_tab_browser_renders_llm_analytics_report_with_charts(monkeypatch, tmp_path):
    """Real browser e2e for asking an analytics question and rendering Plotly charts."""
    if os.getenv("MMA_AI_RUN_BROWSER_E2E") != "1":
        pytest.skip("Set MMA_AI_RUN_BROWSER_E2E=1 to run the Selenium browser e2e.")

    from selenium import webdriver
    from selenium.webdriver.chrome.options import Options
    from selenium.webdriver.common.by import By
    from selenium.webdriver.support import expected_conditions as ec
    from selenium.webdriver.support.ui import WebDriverWait

    monkeypatch.setenv("MMA_AI_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setattr("libs.web.app.get_data_status", lambda: _fake_status(tmp_path))
    monkeypatch.setattr(
        "libs.web.app.get_readiness_status",
        lambda: {"status": "ok", "ready": True, "checks": {"starter_model": {"ok": True}}},
    )
    monkeypatch.setattr(
        "libs.web.app.get_analytics_status",
        lambda: {"configured": True, "provider": "test", "model": "fake-analytics", "mode": "llm"},
    )

    def fake_run_analytics(question, sql=None, max_rows=100):
        assert "adjperf" in question.lower()
        assert sql is None
        assert max_rows == 100
        first_chart = {
            "data": [
                {
                    "type": "bar",
                    "x": ["Lightweight", "Welterweight", "Middleweight"],
                    "y": [0.48, 0.44, 0.41],
                    "name": "Accuracy",
                    "marker": {"color": ["#0f766e", "#2563eb", "#b42318"]},
                }
            ],
            "layout": {
                "title": {"text": "Significant strike accuracy by weight class"},
                "xaxis": {"title": {"text": "Weight class"}},
                "yaxis": {"title": {"text": "Accuracy"}},
                "template": "plotly_white",
            },
        }
        second_chart = {
            "data": [
                {
                    "type": "scatter",
                    "mode": "markers",
                    "x": [28, 31, 17],
                    "y": [0.48, 0.44, 0.41],
                    "name": "Fight sample",
                }
            ],
            "layout": {
                "title": {"text": "Sample size versus accuracy"},
                "xaxis": {"title": {"text": "Fights"}},
                "yaxis": {"title": {"text": "Accuracy"}},
                "template": "plotly_white",
            },
        }
        return {
            "answer": (
                "Positive opponent-adjusted striking performance clusters highest at lightweight. "
                "The plots separate ranking and sample-size context so the result is easier to trust."
            ),
            "sql": "select weightclass, avg_sig_str_acc, fight_count from features.sig_str_adjperf_summary",
            "rows": [
                {"weightclass": "Lightweight", "avg_sig_str_acc": 0.48, "fight_count": 28},
                {"weightclass": "Welterweight", "avg_sig_str_acc": 0.44, "fight_count": 31},
                {"weightclass": "Middleweight", "avg_sig_str_acc": 0.41, "fight_count": 17},
            ],
            "columns": ["weightclass", "avg_sig_str_acc", "fight_count"],
            "charts": [first_chart, second_chart],
        }

    monkeypatch.setattr("libs.web.app.run_analytics", fake_run_analytics)

    server, thread, base_url = _start_uvicorn(create_app(), _free_port())
    options = Options()
    options.add_argument("--headless=new")
    options.add_argument("--disable-gpu")
    options.add_argument("--no-sandbox")
    options.add_argument("--window-size=1280,1000")
    driver = None
    try:
        driver = webdriver.Chrome(options=options)
        wait = WebDriverWait(driver, 15)
        driver.get(base_url)

        question = wait.until(ec.presence_of_element_located((By.ID, "analytics-question")))
        question.clear()
        question.send_keys("Which adjperf striking features stand out by weight class?")
        wait.until(ec.element_to_be_clickable((By.ID, "run-analytics"))).click()

        wait.until(ec.text_to_be_present_in_element((By.ID, "analytics-output"), "Positive opponent-adjusted"))
        wait.until(ec.text_to_be_present_in_element((By.ID, "analytics-output"), "Lightweight"))
        wait.until(lambda active_driver: "features.sig_str_adjperf_summary" in active_driver.execute_script(
            "return document.querySelector('#analytics-output').textContent"
        ))
        wait.until(lambda active_driver: active_driver.execute_script(
            "return document.querySelectorAll('#analytics-chart .js-plotly-plot').length"
        ) == 2)

        assert driver.execute_script("return document.querySelectorAll('.analytics-chart-card').length") == 2
        assert "Significant strike accuracy by weight class" in driver.page_source
        assert "Sample size versus accuracy" in driver.page_source
        assert driver.execute_script("return document.querySelectorAll('#analytics-chart .main-svg').length") >= 2
        assert driver.execute_script("return document.querySelectorAll('#analytics-chart .barlayer .point').length") > 0
    finally:
        if driver is not None:
            driver.quit()
        server.should_exit = True
        thread.join(timeout=5)


def test_data_tab_browser_executes_sql_query_and_renders_plotly_chart(monkeypatch, tmp_path):
    """Real browser e2e for executing an analytics query and rendering the generated chart."""
    if os.getenv("MMA_AI_RUN_BROWSER_E2E") != "1":
        pytest.skip("Set MMA_AI_RUN_BROWSER_E2E=1 to run the Selenium browser e2e.")

    from selenium import webdriver
    from selenium.webdriver.chrome.options import Options
    from selenium.webdriver.common.by import By
    from selenium.webdriver.support import expected_conditions as ec
    from selenium.webdriver.support.ui import WebDriverWait

    data_dir = tmp_path / "data"
    data_dir.mkdir(parents=True)
    (data_dir / "training_data.csv").write_text(
        "\n".join(
            [
                "weightclass,avg_sig_str_acc,fight_count",
                "Lightweight,0.48,28",
                "Welterweight,0.44,31",
                "Middleweight,0.41,17",
            ]
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("MMA_AI_DATA_DIR", str(data_dir))
    monkeypatch.setenv("DATABASE_URL", "postgresql://postgres:postgres@127.0.0.1:1/missing")
    monkeypatch.setattr("libs.web.app.get_data_status", lambda: _fake_status(tmp_path))
    monkeypatch.setattr(
        "libs.web.app.get_readiness_status",
        lambda: {"status": "ok", "ready": True, "checks": {"starter_model": {"ok": True}}},
    )
    monkeypatch.setattr(
        "libs.web.app.get_analytics_status",
        lambda: {"configured": False, "mode": "sql_only", "hint": "test analytics status"},
    )

    server, thread, base_url = _start_uvicorn(create_app(), _free_port())
    options = Options()
    options.add_argument("--headless=new")
    options.add_argument("--disable-gpu")
    options.add_argument("--no-sandbox")
    options.add_argument("--window-size=1280,1000")
    driver = None
    try:
        driver = webdriver.Chrome(options=options)
        wait = WebDriverWait(driver, 15)
        driver.get(base_url)

        wait.until(ec.presence_of_element_located((By.ID, "analytics-question"))).send_keys(
            "Show significant strike accuracy by weight class."
        )
        wait.until(ec.presence_of_element_located((By.ID, "analytics-sql"))).send_keys(
            "select weightclass, avg_sig_str_acc, fight_count from training_data order by avg_sig_str_acc desc"
        )
        wait.until(ec.element_to_be_clickable((By.ID, "run-analytics"))).click()

        wait.until(ec.text_to_be_present_in_element((By.ID, "analytics-output"), "Query executed."))
        wait.until(ec.text_to_be_present_in_element((By.ID, "analytics-output"), "Lightweight"))
        wait.until(ec.text_to_be_present_in_element((By.ID, "analytics-output"), "3"))
        wait.until(lambda active_driver: active_driver.execute_script(
            "return document.querySelectorAll('#analytics-chart .js-plotly-plot').length"
        ) == 1)

        assert driver.execute_script("return document.querySelectorAll('.analytics-chart-card').length") == 1
        assert driver.execute_script("return document.querySelectorAll('#analytics-chart .main-svg').length") >= 1
        assert driver.execute_script("return document.querySelectorAll('#analytics-chart .barlayer .point').length") == 3
        assert "avg_sig_str_acc" in driver.execute_script(
            "return document.querySelector('#analytics-output').textContent"
        )
    finally:
        if driver is not None:
            driver.quit()
        server.should_exit = True
        thread.join(timeout=5)
