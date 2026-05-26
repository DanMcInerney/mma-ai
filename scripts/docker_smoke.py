"""Smoke-test the built Docker web image in its runtime shape."""

from __future__ import annotations

import argparse
import subprocess
import sys
import textwrap
import time
import uuid
from dataclasses import dataclass


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
    parser = argparse.ArgumentParser(description="Smoke-test a built MMA AI Docker web image.")
    parser.add_argument("--image", default=DEFAULT_IMAGE, help=f"Docker image to run. Default: {DEFAULT_IMAGE}")
    parser.add_argument("--timeout", type=int, default=90, help="Seconds to wait for /api/health. Default: 90")
    parser.add_argument("--container-name", help="Optional explicit container name for debugging.")
    args = parser.parse_args(argv)

    try:
        run_smoke(args.image, args.timeout, args.container_name)
    except SmokeError as exc:
        print(f"[docker-smoke] failed: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
