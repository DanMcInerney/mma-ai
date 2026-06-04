"""Smoke-test the built Docker web image in its runtime shape."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import textwrap
import time
import uuid
from dataclasses import dataclass
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen


DEFAULT_IMAGE = "mma-ai-web:latest"


@dataclass
class CommandResult:
    args: list[str]
    returncode: int
    stdout: str
    stderr: str


class SmokeError(RuntimeError):
    """Raised when the Docker smoke check cannot prove the runtime is healthy."""


def _run(args: list[str], *, timeout: int | None = None) -> CommandResult:
    try:
        completed = subprocess.run(
            args,
            capture_output=True,
            check=False,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
        )
    except FileNotFoundError as exc:
        raise SmokeError(f"Command not found: {args[0]}") from exc
    except subprocess.TimeoutExpired as exc:
        raise SmokeError(f"Command timed out: {subprocess.list2cmdline(args)}") from exc

    result = CommandResult(args=args, returncode=completed.returncode, stdout=completed.stdout, stderr=completed.stderr)
    if result.returncode != 0:
        raise SmokeError(_format_failure(result))
    return result


def _format_failure(result: CommandResult) -> str:
    details = [f"Command failed ({result.returncode}): {subprocess.list2cmdline(result.args)}"]
    if result.stdout.strip():
        details.append(f"stdout:\n{result.stdout.strip()}")
    if result.stderr.strip():
        details.append(f"stderr:\n{result.stderr.strip()}")
    return "\n".join(details)


def _run_allow_failure(args: list[str], *, timeout: int | None = None) -> CommandResult:
    try:
        completed = subprocess.run(
            args,
            capture_output=True,
            check=False,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired) as exc:
        return CommandResult(args=args, returncode=127, stdout="", stderr=str(exc))
    return CommandResult(args=args, returncode=completed.returncode, stdout=completed.stdout, stderr=completed.stderr)


def _docker_exec(container_name: str, command: list[str], *, timeout: int | None = None) -> CommandResult:
    return _run(["docker", "exec", container_name, *command], timeout=timeout)


def _docker_exec_allow_failure(container_name: str, command: list[str], *, timeout: int | None = None) -> CommandResult:
    return _run_allow_failure(["docker", "exec", container_name, *command], timeout=timeout)


def _join_url(base_url: str, path: str) -> str:
    return f"{base_url.rstrip('/')}/{path.lstrip('/')}"


def _fetch_json(base_url: str, path: str, *, timeout_seconds: int) -> tuple[int, dict]:
    url = _join_url(base_url, path)
    request = Request(url, headers={"Accept": "application/json"})
    try:
        with urlopen(request, timeout=timeout_seconds) as response:
            status = int(getattr(response, "status", 200))
            body = response.read().decode("utf-8", errors="replace")
    except HTTPError as exc:
        status = exc.code
        body = exc.read().decode("utf-8", errors="replace")
    except URLError as exc:
        raise SmokeError(f"Could not reach {url}: {exc}") from exc

    try:
        payload = json.loads(body)
    except json.JSONDecodeError as exc:
        raise SmokeError(f"Expected JSON from {url}, got: {body[:500]!r}") from exc
    if not isinstance(payload, dict):
        raise SmokeError(f"Expected JSON object from {url}, got: {type(payload).__name__}")
    return status, payload


def _readiness_payload(payload: dict) -> dict:
    detail = payload.get("detail")
    if isinstance(detail, dict) and "ready" in detail:
        return detail
    return payload


def _format_readiness_failure(name: str, check: object) -> str:
    if not isinstance(check, dict):
        return f"{name} (invalid check payload)"

    details: list[str] = []
    missing_columns = check.get("missing_columns") or []
    if missing_columns:
        details.append(f"missing columns: {', '.join(str(column) for column in missing_columns)}")

    if "rows" in check and not check.get("rows"):
        details.append(f"rows: {check.get('rows')}")

    missing_tables = check.get("missing_tables") or []
    if missing_tables:
        details.append(f"missing tables: {', '.join(str(table) for table in missing_tables)}")

    if check.get("error"):
        details.append(f"error: {check['error']}")

    if name == "starter_model":
        expected = check.get("expected") or "configured starter model"
        models = check.get("models") or []
        discovered = ", ".join(str(model) for model in models) if models else "none"
        details.append(f"expected {expected}; discovered {discovered}")

    suffix = "; ".join(details) if details else "not ok"
    return f"{name} ({suffix})"


def _failed_readiness_checks(readiness: dict) -> list[str]:
    checks = readiness.get("checks")
    if not isinstance(checks, dict):
        return ["readiness checks unavailable"]
    failures = [
        _format_readiness_failure(name, check)
        for name, check in checks.items()
        if not isinstance(check, dict) or check.get("ok") is not True
    ]
    return failures or ["ready=false"]


def check_deployed_readiness(base_url: str, model_type: str = "win", timeout_seconds: int = 30) -> None:
    """Verify a deployed dashboard has the artifacts needed for prediction."""
    health_status, health = _fetch_json(base_url, "/api/health", timeout_seconds=timeout_seconds)
    if health_status != 200 or health.get("status") != "ok":
        raise SmokeError(f"Deployed health check failed for {base_url}: HTTP {health_status} {health}")
    print("[docker-smoke] deployed health ok")

    readiness_status, readiness_response = _fetch_json(base_url, "/api/readiness", timeout_seconds=timeout_seconds)
    readiness = _readiness_payload(readiness_response)
    if readiness_status != 200 or readiness.get("ready") is not True:
        failures = "\n".join(f"- {failure}" for failure in _failed_readiness_checks(readiness))
        raise SmokeError(f"Deployed readiness check failed for {base_url}:\n{failures}")
    print("[docker-smoke] deployed readiness ok")

    query = urlencode({"model_type": model_type})
    models_status, models_payload = _fetch_json(
        base_url,
        f"/api/predict/models?{query}",
        timeout_seconds=timeout_seconds,
    )
    if models_status != 200:
        raise SmokeError(f"Deployed model discovery failed for {base_url}: HTTP {models_status} {models_payload}")
    models = models_payload.get("models") or []
    if not models:
        raise SmokeError(f"No {model_type} models found at {base_url}. Run setup again or provide a compatible model.")
    model_names = ", ".join(str(model.get("name", "<unnamed>")) for model in models[:5] if isinstance(model, dict))
    print(f"[docker-smoke] deployed {model_type} models ok: {model_names}")


def wait_for_health(container_name: str, timeout_seconds: int) -> None:
    """Wait for the web app to answer its internal health endpoint."""
    deadline = time.monotonic() + timeout_seconds
    last_result: CommandResult | None = None
    while time.monotonic() < deadline:
        last_result = _docker_exec_allow_failure(
            container_name,
            ["curl", "-fsS", "http://127.0.0.1:8000/api/health"],
            timeout=5,
        )
        if last_result.returncode == 0 and '"status":"ok"' in last_result.stdout.replace(" ", ""):
            print("[docker-smoke] health ok")
            return
        time.sleep(1)

    logs = _run_allow_failure(["docker", "logs", "--tail", "120", container_name], timeout=10)
    message = [
        f"Container did not become healthy within {timeout_seconds} seconds.",
        "last health attempt:",
        _format_failure(last_result) if last_result else "No health command was attempted.",
    ]
    if logs.stdout.strip() or logs.stderr.strip():
        message.append("container logs:")
        message.append((logs.stdout + logs.stderr).strip())
    raise SmokeError("\n".join(message))


def check_runtime_dependencies(container_name: str) -> None:
    """Verify the runtime venv has production dependencies and no test tooling."""
    code = textwrap.dedent(
        """
        import importlib.util
        import pickle
        import sys

        from libs.modeling.portable_artifacts import install_pathlib_pickle_compatibility
        from libs.modeling.runtime_dependencies import prediction_runtime_dependency_report

        present = [name for name in ("pytest", "pytest_mock") if importlib.util.find_spec(name) is not None]
        if present:
            raise SystemExit(f"test tooling present in runtime image: {', '.join(present)}")

        import kaleido
        version = getattr(kaleido, "__version__", None)
        if version != "0.2.1":
            raise SystemExit(f"unexpected kaleido version: {version!r}")

        prediction_runtime = prediction_runtime_dependency_report()
        if not prediction_runtime["ok"]:
            missing = ", ".join(
                f"{item['module']} ({item['reason']})" for item in prediction_runtime["missing"]
            )
            raise SystemExit(f"prediction runtime dependencies missing: {missing}")

        windows_path_pickle = (
            b"\\x80\\x04\\x953\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x8c\\x07pathlib\\x94"
            b"\\x8c\\x0bWindowsPath\\x94\\x93\\x94\\x8c\\x03C:\\\\\\x94\\x8c\\x06models"
            b"\\x94\\x8c\\x02ag\\x94\\x87\\x94R\\x94."
        )
        install_pathlib_pickle_compatibility()
        pickle.loads(windows_path_pickle)

        print("runtime dependency check ok: pytest absent, pytest_mock absent, kaleido 0.2.1, prediction imports present, cross-OS pathlib pickles load")
        """
    ).strip()
    result = _docker_exec(container_name, ["/app/.venv/bin/python", "-c", code], timeout=30)
    print(f"[docker-smoke] {result.stdout.strip()}")


def check_runtime_source_tree(container_name: str) -> None:
    """Verify dev-only source trees are not copied into the runtime image."""
    result = _docker_exec(container_name, ["sh", "-lc", "test ! -e /app/tests"], timeout=10)
    print("[docker-smoke] runtime source tree ok: tests absent")


def check_dashboard_assets(container_name: str) -> None:
    """Verify the runtime image serves the packaged dashboard and local JS assets."""
    checks = [
        ("/", "MMA AI"),
        ("/vendor/plotly.min.js", "Plotly"),
        ("/static/icons.js", "window.lucide"),
    ]
    for path, expected_text in checks:
        result = _docker_exec(
            container_name,
            ["curl", "-fsS", f"http://127.0.0.1:8000{path}"],
            timeout=10,
        )
        if expected_text not in result.stdout:
            raise SmokeError(f"Dashboard asset check failed for {path}: expected {expected_text!r}.")
    print("[docker-smoke] dashboard assets ok")


def run_smoke(image: str = DEFAULT_IMAGE, timeout_seconds: int = 90, container_name: str | None = None) -> None:
    """Run the Docker smoke test and clean up the container."""
    name = container_name or f"mma-ai-smoke-{uuid.uuid4().hex[:12]}"
    print(f"[docker-smoke] starting {image} as {name}")
    _run(["docker", "run", "-d", "--rm", "--name", name, image], timeout=timeout_seconds)
    try:
        wait_for_health(name, timeout_seconds)
        check_dashboard_assets(name)
        check_runtime_dependencies(name)
        check_runtime_source_tree(name)
        print("[docker-smoke] passed")
    finally:
        stopped = _run_allow_failure(["docker", "stop", name], timeout=30)
        if stopped.returncode == 0:
            print(f"[docker-smoke] stopped {name}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Smoke-test a built MMA AI Docker image or a deployed dashboard.")
    parser.add_argument("--image", default=DEFAULT_IMAGE, help=f"Docker image to run. Default: {DEFAULT_IMAGE}")
    parser.add_argument("--timeout", type=int, default=90, help="Seconds to wait for /api/health. Default: 90")
    parser.add_argument("--container-name", help="Optional explicit container name for debugging.")
    parser.add_argument("--deployed-url", help="Check an already deployed dashboard, for example http://127.0.0.1:8001.")
    parser.add_argument("--model-type", default="win", help="Prediction model type to require in deployed mode. Default: win")
    args = parser.parse_args(argv)

    try:
        if args.deployed_url:
            check_deployed_readiness(args.deployed_url, args.model_type, args.timeout)
        else:
            run_smoke(args.image, args.timeout, args.container_name)
    except SmokeError as exc:
        print(f"[docker-smoke] failed: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
