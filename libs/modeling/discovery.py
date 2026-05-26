"""Shared helpers for discovering prediction-ready model directories."""

from __future__ import annotations

from pathlib import Path


def is_loadable_prediction_model_dir(path: Path) -> bool:
    """Return whether a model directory has the files needed for prediction."""
    if not path.is_dir() or not (path / "feats.txt").exists():
        return False
    if (path / "predictor.pkl").exists():
        return True
    if not (path / "ensemble_info.txt").exists():
        return False
    return (path / "final_model").is_dir() or any(child.is_dir() for child in path.glob("window_*"))
