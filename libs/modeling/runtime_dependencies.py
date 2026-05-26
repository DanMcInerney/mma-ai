"""Runtime dependency checks for prediction-ready AutoGluon model families."""

from __future__ import annotations

import importlib.util
from dataclasses import dataclass


@dataclass(frozen=True)
class RuntimeImport:
    module: str
    package_hint: str
    reason: str


PREDICTION_RUNTIME_IMPORTS = (
    RuntimeImport("autogluon.tabular", "autogluon.tabular", "AutoGluon predictor loading"),
    RuntimeImport("torch", "torch", "torch-backed AutoGluon models"),
    RuntimeImport("catboost", "catboost", "CAT model family"),
    RuntimeImport("lightgbm", "lightgbm", "GBM model family"),
    RuntimeImport("xgboost", "xgboost", "XGB-compatible model artifacts"),
    RuntimeImport("loguru", "loguru", "MITRA model loading"),
    RuntimeImport("einx", "einx", "MITRA tensor operations"),
    RuntimeImport("omegaconf", "omegaconf", "MITRA configuration loading"),
    RuntimeImport("transformers", "transformers", "MITRA Hugging Face model loading"),
    RuntimeImport("huggingface_hub", "huggingface_hub", "MITRA and TabPFN artifact loading"),
    RuntimeImport("einops", "einops", "MITRA tensor operations"),
    RuntimeImport("tabicl", "tabicl", "TABICL model family"),
    RuntimeImport("tabpfn", "tabpfn", "TabPFNv2 model family"),
)


def missing_prediction_runtime_imports() -> list[RuntimeImport]:
    """Return prediction dependencies that are absent from the active runtime."""
    missing = []
    for requirement in PREDICTION_RUNTIME_IMPORTS:
        if importlib.util.find_spec(requirement.module) is None:
            missing.append(requirement)
    return missing


def prediction_runtime_dependency_report() -> dict[str, object]:
    """Return a structured dependency status payload for smoke tests and APIs."""
    missing = missing_prediction_runtime_imports()
    return {
        "ok": not missing,
        "missing": [
            {
                "module": item.module,
                "package": item.package_hint,
                "reason": item.reason,
            }
            for item in missing
        ],
    }


def assert_prediction_runtime_dependencies() -> None:
    """Raise a concise setup error if the runtime cannot load default model families."""
    missing = missing_prediction_runtime_imports()
    if not missing:
        return
    modules = ", ".join(item.module for item in missing)
    packages = ", ".join(dict.fromkeys(item.package_hint for item in missing))
    raise RuntimeError(
        "Prediction runtime is missing dependencies for the configured AutoGluon model families. "
        f"Missing modules: {modules}. Rebuild or resync the environment so these packages are installed: {packages}."
    )
