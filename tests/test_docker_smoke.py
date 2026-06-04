import io
import json
import subprocess
import urllib.error
from pathlib import Path

import pytest

from scripts import docker_smoke


def completed(args, returncode=0, stdout="", stderr=""):
    return subprocess.CompletedProcess(args, returncode, stdout=stdout, stderr=stderr)


class JsonResponse:
    def __init__(self, payload, status=200):
        self.payload = payload
        self.status = status

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def read(self):
        return json.dumps(self.payload).encode("utf-8")


def test_docker_smoke_runs_container_checks_health_deps_and_stops(monkeypatch, capsys):
    calls = []
    health_attempts = {"count": 0}

    def fake_run(args, **_kwargs):
        calls.append(args)
        if args[:3] == ["docker", "run", "-d"]:
            return completed(args, stdout="container-id\n")
        if args[:4] == ["docker", "exec", "smoke-test", "curl"]:
            url = args[-1]
            if url.endswith("/api/health"):
                health_attempts["count"] += 1
                if health_attempts["count"] == 1:
                    return completed(args, returncode=7, stderr="connection refused")
                return completed(args, stdout='{"status":"ok"}')
            if url.endswith("/vendor/plotly.min.js"):
                return completed(args, stdout="window.Plotly = Plotly;")
            if url.endswith("/static/icons.js"):
                return completed(args, stdout="window.lucide = { createIcons };")
            if url.endswith("/"):
                return completed(args, stdout="<title>MMA AI</title>")
        if args[:4] == ["docker", "exec", "smoke-test", "/app/.venv/bin/python"]:
            return completed(
                args,
                stdout="runtime dependency check ok: pytest absent, pytest_mock absent, kaleido 0.2.1, prediction imports present\n",
            )
        if args[:5] == ["docker", "exec", "smoke-test", "sh", "-lc"]:
            assert args[-1] == "test ! -e /app/tests"
            return completed(args)
        if args == ["docker", "stop", "smoke-test"]:
            return completed(args, stdout="smoke-test\n")
        raise AssertionError(f"unexpected command: {args}")

    monkeypatch.setattr(docker_smoke.subprocess, "run", fake_run)
    monkeypatch.setattr(docker_smoke.time, "sleep", lambda _seconds: None)

    docker_smoke.run_smoke("mma-ai-web:test", timeout_seconds=10, container_name="smoke-test")

    assert ["docker", "run", "-d", "--rm", "--name", "smoke-test", "mma-ai-web:test"] in calls
    assert ["docker", "stop", "smoke-test"] in calls
    assert health_attempts["count"] == 2
    output = capsys.readouterr().out
    assert "health ok" in output
    assert "dashboard assets ok" in output
    assert "runtime dependency check ok" in output
    assert "runtime source tree ok" in output
    assert "passed" in output


def test_deployed_smoke_checks_readiness_and_win_models(monkeypatch, capsys):
    requested_urls = []

    def fake_urlopen(request, **_kwargs):
        requested_urls.append(request.full_url)
        if request.full_url.endswith("/api/health"):
            return JsonResponse({"status": "ok"})
        if request.full_url.endswith("/api/readiness"):
            return JsonResponse({"ready": True, "checks": {"starter_model": {"ok": True}}})
        if request.full_url.endswith("/api/predict/models?model_type=win"):
            return JsonResponse({"models": [{"name": "ag-20260304_110750-win-extreme"}]})
        raise AssertionError(f"unexpected URL: {request.full_url}")

    monkeypatch.setattr(docker_smoke, "urlopen", fake_urlopen)

    docker_smoke.check_deployed_readiness("http://127.0.0.1:8001", model_type="win", timeout_seconds=5)

    assert requested_urls == [
        "http://127.0.0.1:8001/api/health",
        "http://127.0.0.1:8001/api/readiness",
        "http://127.0.0.1:8001/api/predict/models?model_type=win",
    ]
    output = capsys.readouterr().out
    assert "deployed health ok" in output
    assert "deployed readiness ok" in output
    assert "deployed win models ok: ag-20260304_110750-win-extreme" in output


def test_deployed_smoke_reports_failed_readiness_checks(monkeypatch):
    def fake_urlopen(request, **_kwargs):
        if request.full_url.endswith("/api/health"):
            return JsonResponse({"status": "ok"})
        if request.full_url.endswith("/api/readiness"):
            payload = {
                "detail": {
                    "ready": False,
                    "checks": {
                        "prediction_data_csv": {
                            "ok": False,
                            "path": "/app/data/prediction_data.csv",
                            "rows": None,
                            "missing_columns": ["fighter_name"],
                        },
                        "starter_model": {
                            "ok": False,
                            "expected": "ag-20260304_110750-win-extreme",
                            "models": [],
                        },
                    },
                }
            }
            body = io.BytesIO(json.dumps(payload).encode("utf-8"))
            raise urllib.error.HTTPError(request.full_url, 503, "Service Unavailable", hdrs=None, fp=body)
        raise AssertionError(f"unexpected URL: {request.full_url}")

    monkeypatch.setattr(docker_smoke, "urlopen", fake_urlopen)

    with pytest.raises(docker_smoke.SmokeError) as exc:
        docker_smoke.check_deployed_readiness("http://127.0.0.1:8001", timeout_seconds=5)

    message = str(exc.value)
    assert "Deployed readiness check failed" in message
    assert "prediction_data_csv" in message
    assert "missing columns: fighter_name" in message
    assert "starter_model" in message
    assert "expected ag-20260304_110750-win-extreme; discovered none" in message


def test_deployed_smoke_reports_empty_win_models(monkeypatch):
    def fake_urlopen(request, **_kwargs):
        if request.full_url.endswith("/api/health"):
            return JsonResponse({"status": "ok"})
        if request.full_url.endswith("/api/readiness"):
            return JsonResponse({"ready": True, "checks": {"starter_model": {"ok": True}}})
        if request.full_url.endswith("/api/predict/models?model_type=win"):
            return JsonResponse({"models": []})
        raise AssertionError(f"unexpected URL: {request.full_url}")

    monkeypatch.setattr(docker_smoke, "urlopen", fake_urlopen)

    with pytest.raises(docker_smoke.SmokeError, match="No win models found"):
        docker_smoke.check_deployed_readiness("http://127.0.0.1:8001", model_type="win", timeout_seconds=5)


def test_docker_smoke_cli_can_check_deployed_url(monkeypatch):
    captured = {}

    def fake_check(base_url, model_type, timeout_seconds):
        captured["base_url"] = base_url
        captured["model_type"] = model_type
        captured["timeout_seconds"] = timeout_seconds

    monkeypatch.setattr(docker_smoke, "check_deployed_readiness", fake_check)

    exit_code = docker_smoke.main(["--deployed-url", "http://localhost:8001", "--model-type", "decision", "--timeout", "12"])

    assert exit_code == 0
    assert captured == {
        "base_url": "http://localhost:8001",
        "model_type": "decision",
        "timeout_seconds": 12,
    }


def test_docker_smoke_uses_configured_timeout_for_first_container_start(monkeypatch):
    captured = {}

    def fake_run(args, **kwargs):
        if args[:3] == ["docker", "run", "-d"]:
            captured["docker_run_timeout"] = kwargs["timeout"]
            return completed(args, stdout="container-id\n")
        if args[:4] == ["docker", "exec", "smoke-test", "curl"]:
            url = args[-1]
            if url.endswith("/api/health"):
                return completed(args, stdout='{"status":"ok"}')
            if url.endswith("/vendor/plotly.min.js"):
                return completed(args, stdout="window.Plotly = Plotly;")
            if url.endswith("/static/icons.js"):
                return completed(args, stdout="window.lucide = { createIcons };")
            if url.endswith("/"):
                return completed(args, stdout="<title>MMA AI</title>")
        if args[:4] == ["docker", "exec", "smoke-test", "/app/.venv/bin/python"]:
            return completed(args, stdout="runtime dependency check ok\n")
        if args[:5] == ["docker", "exec", "smoke-test", "sh", "-lc"]:
            return completed(args)
        if args == ["docker", "stop", "smoke-test"]:
            return completed(args)
        raise AssertionError(f"unexpected command: {args}")

    monkeypatch.setattr(docker_smoke.subprocess, "run", fake_run)

    docker_smoke.run_smoke("mma-ai-web:test", timeout_seconds=180, container_name="smoke-test")

    assert captured["docker_run_timeout"] == 180


def test_docker_smoke_stops_container_when_asset_check_fails(monkeypatch):
    calls = []

    def fake_run(args, **_kwargs):
        calls.append(args)
        if args[:3] == ["docker", "run", "-d"]:
            return completed(args, stdout="container-id\n")
        if args[:4] == ["docker", "exec", "smoke-test", "curl"]:
            url = args[-1]
            if url.endswith("/api/health"):
                return completed(args, stdout='{"status":"ok"}')
            if url.endswith("/"):
                return completed(args, stdout="<title>MMA AI</title>")
            if url.endswith("/vendor/plotly.min.js"):
                return completed(args, stdout="missing bundled plotly")
        if args == ["docker", "stop", "smoke-test"]:
            return completed(args)
        raise AssertionError(f"unexpected command: {args}")

    monkeypatch.setattr(docker_smoke.subprocess, "run", fake_run)

    with pytest.raises(docker_smoke.SmokeError, match="Dashboard asset check failed"):
        docker_smoke.run_smoke("mma-ai-web:test", timeout_seconds=10, container_name="smoke-test")

    assert ["docker", "stop", "smoke-test"] in calls


def test_docker_smoke_stops_container_when_dependency_check_fails(monkeypatch):
    calls = []

    def fake_run(args, **_kwargs):
        calls.append(args)
        if args[:3] == ["docker", "run", "-d"]:
            return completed(args, stdout="container-id\n")
        if args[:4] == ["docker", "exec", "smoke-test", "curl"]:
            url = args[-1]
            if url.endswith("/api/health"):
                return completed(args, stdout='{"status":"ok"}')
            if url.endswith("/vendor/plotly.min.js"):
                return completed(args, stdout="window.Plotly = Plotly;")
            if url.endswith("/static/icons.js"):
                return completed(args, stdout="window.lucide = { createIcons };")
            if url.endswith("/"):
                return completed(args, stdout="<title>MMA AI</title>")
        if args[:4] == ["docker", "exec", "smoke-test", "/app/.venv/bin/python"]:
            return completed(args, returncode=1, stderr="test tooling present in runtime image: pytest")
        if args[:5] == ["docker", "exec", "smoke-test", "sh", "-lc"]:
            return completed(args)
        if args == ["docker", "stop", "smoke-test"]:
            return completed(args)
        raise AssertionError(f"unexpected command: {args}")

    monkeypatch.setattr(docker_smoke.subprocess, "run", fake_run)

    with pytest.raises(docker_smoke.SmokeError, match="test tooling present"):
        docker_smoke.run_smoke("mma-ai-web:test", timeout_seconds=10, container_name="smoke-test")

    assert ["docker", "stop", "smoke-test"] in calls


def test_docker_smoke_stops_container_when_prediction_imports_missing(monkeypatch):
    calls = []

    def fake_run(args, **_kwargs):
        calls.append(args)
        if args[:3] == ["docker", "run", "-d"]:
            return completed(args, stdout="container-id\n")
        if args[:4] == ["docker", "exec", "smoke-test", "curl"]:
            url = args[-1]
            if url.endswith("/api/health"):
                return completed(args, stdout='{"status":"ok"}')
            if url.endswith("/vendor/plotly.min.js"):
                return completed(args, stdout="window.Plotly = Plotly;")
            if url.endswith("/static/icons.js"):
                return completed(args, stdout="window.lucide = { createIcons };")
            if url.endswith("/"):
                return completed(args, stdout="<title>MMA AI</title>")
        if args[:4] == ["docker", "exec", "smoke-test", "/app/.venv/bin/python"]:
            return completed(args, returncode=1, stderr="prediction runtime dependencies missing: loguru (MITRA model loading)")
        if args[:5] == ["docker", "exec", "smoke-test", "sh", "-lc"]:
            return completed(args)
        if args == ["docker", "stop", "smoke-test"]:
            return completed(args)
        raise AssertionError(f"unexpected command: {args}")

    monkeypatch.setattr(docker_smoke.subprocess, "run", fake_run)

    with pytest.raises(docker_smoke.SmokeError, match="prediction runtime dependencies missing"):
        docker_smoke.run_smoke("mma-ai-web:test", timeout_seconds=10, container_name="smoke-test")

    assert ["docker", "stop", "smoke-test"] in calls


def test_docker_smoke_stops_container_when_tests_tree_is_present(monkeypatch):
    calls = []

    def fake_run(args, **_kwargs):
        calls.append(args)
        if args[:3] == ["docker", "run", "-d"]:
            return completed(args, stdout="container-id\n")
        if args[:4] == ["docker", "exec", "smoke-test", "curl"]:
            url = args[-1]
            if url.endswith("/api/health"):
                return completed(args, stdout='{"status":"ok"}')
            if url.endswith("/vendor/plotly.min.js"):
                return completed(args, stdout="window.Plotly = Plotly;")
            if url.endswith("/static/icons.js"):
                return completed(args, stdout="window.lucide = { createIcons };")
            if url.endswith("/"):
                return completed(args, stdout="<title>MMA AI</title>")
        if args[:4] == ["docker", "exec", "smoke-test", "/app/.venv/bin/python"]:
            return completed(
                args,
                stdout="runtime dependency check ok: pytest absent, pytest_mock absent, kaleido 0.2.1, prediction imports present\n",
            )
        if args[:5] == ["docker", "exec", "smoke-test", "sh", "-lc"]:
            return completed(args, returncode=1, stderr="/app/tests exists")
        if args == ["docker", "stop", "smoke-test"]:
            return completed(args)
        raise AssertionError(f"unexpected command: {args}")

    monkeypatch.setattr(docker_smoke.subprocess, "run", fake_run)

    with pytest.raises(docker_smoke.SmokeError, match="/app/tests exists"):
        docker_smoke.run_smoke("mma-ai-web:test", timeout_seconds=10, container_name="smoke-test")

    assert ["docker", "stop", "smoke-test"] in calls


def test_docker_smoke_timeout_includes_container_logs(monkeypatch):
    calls = []
    monotonic_values = iter([0, 1, 2, 3])

    def fake_run(args, **_kwargs):
        calls.append(args)
        if args[:3] == ["docker", "run", "-d"]:
            return completed(args)
        if args[:4] == ["docker", "exec", "smoke-test", "curl"]:
            return completed(args, returncode=7, stderr="connection refused")
        if args[:3] == ["docker", "logs", "--tail"]:
            return completed(args, stdout="uvicorn never started")
        if args == ["docker", "stop", "smoke-test"]:
            return completed(args)
        raise AssertionError(f"unexpected command: {args}")

    monkeypatch.setattr(docker_smoke.subprocess, "run", fake_run)
    monkeypatch.setattr(docker_smoke.time, "monotonic", lambda: next(monotonic_values))
    monkeypatch.setattr(docker_smoke.time, "sleep", lambda _seconds: None)

    with pytest.raises(docker_smoke.SmokeError, match="uvicorn never started"):
        docker_smoke.run_smoke("mma-ai-web:test", timeout_seconds=2, container_name="smoke-test")

    assert ["docker", "logs", "--tail", "120", "smoke-test"] in calls
    assert ["docker", "stop", "smoke-test"] in calls


def test_docker_smoke_cli_returns_nonzero_on_failure(monkeypatch, capsys):
    monkeypatch.setattr(
        docker_smoke,
        "run_smoke",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(docker_smoke.SmokeError("docker unavailable")),
    )

    exit_code = docker_smoke.main(["--image", "missing", "--timeout", "1"])

    assert exit_code == 1
    assert "docker unavailable" in capsys.readouterr().err


def test_docker_smoke_decodes_container_output_portably(monkeypatch):
    captured = {}

    def fake_run(args, **kwargs):
        captured.update(kwargs)
        return completed(args, stdout="ok")

    monkeypatch.setattr(docker_smoke.subprocess, "run", fake_run)

    result = docker_smoke._run(["docker", "exec", "smoke-test", "curl", "http://127.0.0.1:8000/vendor/plotly.min.js"])

    assert result.stdout == "ok"
    assert captured["text"] is True
    assert captured["encoding"] == "utf-8"
    assert captured["errors"] == "replace"


def test_docker_smoke_is_exposed_as_project_script():
    pyproject = Path(__file__).resolve().parents[1] / "pyproject.toml"

    assert 'mma-docker-smoke = "scripts.docker_smoke:main"' in pyproject.read_text(encoding="utf-8")
