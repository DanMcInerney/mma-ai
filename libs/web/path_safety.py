"""Path validation helpers for dashboard-supplied filesystem paths."""

from __future__ import annotations

from pathlib import Path

from libs.paths import data_dir, data_file, models_dir


class PathPolicyError(ValueError):
    """Raised when a user-supplied path is outside the dashboard's safe roots."""


def resolve_data_csv(path_value: str | Path | None, default_name: str) -> Path:
    """Resolve a finalized-data CSV path under MMA_AI_DATA_DIR."""
    path = Path(path_value).expanduser().resolve() if path_value else data_file(default_name).resolve()
    _ensure_under_roots(path, [data_dir()], f"{default_name} path")
    if path.exists() and not path.is_file():
        raise PathPolicyError(f"{default_name} path is not a file: {path}")
    return path


def resolve_data_output_dir(path_value: str | Path | None, default_relative: str) -> Path:
    """Resolve a dashboard output directory under MMA_AI_DATA_DIR."""
    if path_value:
        raw_path = Path(path_value).expanduser()
        path = raw_path.resolve() if raw_path.is_absolute() else (data_dir() / raw_path).resolve()
    else:
        path = (data_dir() / default_relative).resolve()
    _ensure_under_roots(path, [data_dir()], "output directory")
    if path.exists() and not path.is_dir():
        raise PathPolicyError(f"output directory is not a directory: {path}")
    return path


def resolve_model_dir(path_value: str | Path | None) -> Path | None:
    """Resolve a model directory path under MMA_AI_MODELS_DIR."""
    if not path_value:
        return None
    path = Path(path_value).expanduser().resolve()
    _ensure_under_roots(path, [models_dir()], "model path")
    if not path.exists() or not path.is_dir():
        raise FileNotFoundError(f"Model directory not found: {path}")
    return path


def _ensure_under_roots(path: Path, roots: list[Path], label: str) -> None:
    resolved_roots = [root.expanduser().resolve() for root in roots]
    for root in resolved_roots:
        try:
            path.relative_to(root)
            return
        except ValueError:
            continue
    allowed = ", ".join(str(root) for root in resolved_roots)
    raise PathPolicyError(f"{label} must be under one of: {allowed}")
