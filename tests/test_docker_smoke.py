import subprocess
from pathlib import Path

import pytest

from scripts import docker_smoke


def completed(args, returncode=0, stdout="", stderr=""):
    return subprocess.CompletedProcess(args, returncode, stdout=stdout, stderr=stderr)


def test_docker_smoke_runs_container_checks_health_deps_and_stops(monkeypatch, capsys):
    calls = []
    health_attempts = {"count": 0}

    def fake_run(args, **_kwargs):
        calls.append(args)
        if args[:3] == ["docker", "run", "-d"]:
            return completed(args, stdout="container-id\n")
        if args[:4] == ["docker", "exec", "smoke-test", "curl"]:
            health_attempts["count"] += 1
            if health_attempts["count"] == 1:
                return completed(args, returncode=7, stderr="connection refused")
            return completed(args, stdout='{"status":"ok"}')
        if args[:4] == ["docker", "exec", "smoke-test", "/app/.venv/bin/python"]:
            return completed(
                args,
                stdout="runtime dependency check ok: pytest absent, pytest_mock absent, kaleido 0.2.1\n",
            )
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
    assert "runtime dependency check ok" in output
    assert "passed" in output


def test_docker_smoke_stops_container_when_dependency_check_fails(monkeypatch):
    calls = []

    def fake_run(args, **_kwargs):
        calls.append(args)
        if args[:3] == ["docker", "run", "-d"]:
            return completed(args, stdout="container-id\n")
        if args[:4] == ["docker", "exec", "smoke-test", "curl"]:
            return completed(args, stdout='{"status":"ok"}')
        if args[:4] == ["docker", "exec", "smoke-test", "/app/.venv/bin/python"]:
            return completed(args, returncode=1, stderr="test tooling present in runtime image: pytest")
        if args == ["docker", "stop", "smoke-test"]:
            return completed(args)
        raise AssertionError(f"unexpected command: {args}")

    monkeypatch.setattr(docker_smoke.subprocess, "run", fake_run)

    with pytest.raises(docker_smoke.SmokeError, match="test tooling present"):
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


def test_docker_smoke_is_exposed_as_project_script():
    pyproject = Path(__file__).resolve().parents[1] / "pyproject.toml"

    assert 'mma-docker-smoke = "scripts.docker_smoke:main"' in pyproject.read_text(encoding="utf-8")
