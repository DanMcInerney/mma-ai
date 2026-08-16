"""One-fit/one-access retrospective evaluation state machine."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import random
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

from .protocol import (
    DEFAULT_SOURCE_CSV,
    FROZEN_FEATURE_SHA256,
    FROZEN_SOURCE_SHA256,
    canonical_json_bytes,
    canonical_sha256,
    file_sha256,
    verify_split,
)
from .registry import append_registry_record


RUN_ID = "80-10-10-evaluation"
ARTIFACT_SCOPE = "artifacts/02-one-shot-evaluation"
EXPECTED_PROFILE_SHA256 = "BCF50AE0C67A5B78B87BD6F44F17BD00B5F2351A8513EC508C3644BA92E23F76"
EXPECTED_PARTITION_SHA256 = "AC1D144D5745272611A92E00576854E510B10B08B43A1955465DD8EBB93D0C49"
EXPECTED_REGISTRY_PREFIX = "BAE04E82395EBFDBDA662170E83565E0D59AE91D5FDE3B338E64624866208305"
ROLLBACK_REVISION = "545441975b86caf0abb6136e099e44e6b93caf22"
ROLLBACK_TREE = "82305ddf6160338bfab8e1e8e4e6dc3b82efc7bf"
BASELINE_REVISION = "4ef43de12db79252355e5b6f5ecd58ccdb4c6a06"
EXPECTED_BASE_MODELS = (
    "RandomForestGini",
    "CatBoost",
    "TabICL",
    "ExtraTreesGini",
    "Mitra",
    "NeuralNetFastAI",
    "XGBoost",
    "PrepLightGBM",
    "LightGBM_r8",
    "RealMLP_r9",
)
SEEDS = {"python": 20260816, "numpy": 20260816, "torch": 20260816, "bootstrap": 20260816}
FORBIDDEN_MODEL_TOKENS = ("_FULL", "CONTEXT")


class EvaluationError(ValueError):
    pass


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _read_json(path: Path) -> Any:
    raw = Path(path).read_bytes()
    value = json.loads(raw)
    canonical = canonical_json_bytes(value)
    if raw not in (canonical, canonical + b"\n", canonical + b"\r\n"):
        raise EvaluationError(f"noncanonical JSON: {path}")
    return value


def _write_json(path: Path, value: Any) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical_json_bytes(value) + b"\n")
    return file_sha256(path)


def _append_jsonl(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("ab") as handle:
        handle.write(canonical_json_bytes(dict(value)) + b"\n")
        handle.flush()
        os.fsync(handle.fileno())


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not Path(path).exists():
        return []
    rows = []
    for number, raw in enumerate(Path(path).read_bytes().splitlines(), start=1):
        try:
            value = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise EvaluationError(f"invalid JSONL at {path}:{number}") from exc
        if raw != canonical_json_bytes(value):
            raise EvaluationError(f"noncanonical JSONL at {path}:{number}")
        rows.append(value)
    return rows


def tree_identity(root: Path) -> dict[str, Any]:
    root = Path(root)
    if not root.is_dir():
        raise EvaluationError(f"model tree is missing: {root}")
    files = []
    for path in sorted(candidate for candidate in root.rglob("*") if candidate.is_file()):
        files.append(
            {
                "path": path.relative_to(root).as_posix(),
                "size": path.stat().st_size,
                "sha256": file_sha256(path),
            }
        )
    if not files:
        raise EvaluationError("model tree cannot be empty")
    return {"file_count": len(files), "sha256": canonical_sha256(files), "files": files}


def compute_recency_weights(
    event_dates: pd.Series, *, decay_rate: float
) -> tuple[list[float], dict[str, Any]]:
    dates = pd.to_datetime(event_dates, errors="raise")
    if dates.empty:
        raise EvaluationError("sample weights require training rows")
    reference = dates.max()
    days_ago = (reference - dates).dt.days.to_numpy(dtype=float)
    raw = np.exp(-float(decay_rate) * days_ago / 365.25)
    weights = raw * len(raw) / raw.sum()
    values = [float(value) for value in weights]
    provenance = {
        "kind": "exponential-recency",
        "annual_decay": float(decay_rate),
        "normalization": "mean-one",
        "reference_date": reference.date().isoformat(),
        "row_count": len(values),
        "sum": float(weights.sum()),
        "minimum": float(weights.min()),
        "maximum": float(weights.max()),
    }
    return values, provenance


def resolve_candidate_names(predictor: Any) -> tuple[str, ...]:
    names = tuple(name for name in predictor.model_names() if not name.startswith("WeightedEnsemble"))
    if names != EXPECTED_BASE_MODELS:
        raise EvaluationError(f"resolved candidates differ from exact weighted-v8 portfolio: {names}")
    return names


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(item) for item in value]
    return str(value)


def _find_weights(value: Any) -> dict[str, float] | None:
    if isinstance(value, Mapping):
        candidate = value.get("model_weights")
        if isinstance(candidate, Mapping):
            return {str(key): float(weight) for key, weight in candidate.items()}
        for nested in value.values():
            found = _find_weights(nested)
            if found:
                return found
    elif isinstance(value, (list, tuple)):
        for nested in value:
            found = _find_weights(nested)
            if found:
                return found
    return None


def freeze_selection(
    predictor: Any,
    *,
    model_root: Path,
    scaler_path: Path,
    fixed_identities: Mapping[str, Any],
) -> dict[str, Any]:
    selected = str(predictor.model_best)
    all_names = tuple(str(name) for name in predictor.model_names())
    if any(token in name.upper() for name in all_names for token in FORBIDDEN_MODEL_TOKENS):
        raise EvaluationError("FULL/context nodes are forbidden from evaluation selection")
    base_models = resolve_candidate_names(predictor)
    if selected not in all_names:
        raise EvaluationError("selected node is absent from model graph")
    information = predictor.info().get("model_info", {})
    selected_info = information.get(selected, {})
    weights = _find_weights(selected_info) or ({selected: 1.0} if selected in base_models else None)
    if not weights:
        raise EvaluationError("selected ensemble weights could not be frozen")
    nodes = []
    for name in all_names:
        info = information.get(name, {})
        dependencies = info.get("ancestors") or info.get("dependencies") or []
        nodes.append(
            {
                "name": name,
                "model_type": str(info.get("model_type", "unknown")),
                "stack_level": info.get("stack_level"),
                "dependencies": _json_safe(dependencies),
            }
        )
    model_tree = tree_identity(Path(model_root))
    scaler_path = Path(scaler_path)
    return {
        "schema_version": 1,
        "state": "frozen-pre-test",
        "frozen_at_utc": utc_now(),
        **dict(fixed_identities),
        "classes": [int(value) if isinstance(value, (int, np.integer)) else str(value) for value in predictor.class_labels],
        "base_models": list(base_models),
        "model_graph": nodes,
        "selected_node": selected,
        "ensemble_dependencies": sorted(weights),
        "ensemble_weights": weights,
        "model_root": str(Path(model_root).resolve()),
        "model_tree": model_tree,
        "scaler_path": str(scaler_path.resolve()),
        "scaler_sha256": file_sha256(scaler_path),
        "selection_uses_full_or_context_metrics": False,
        "refit_full": False,
        "feature_importance_computed": False,
    }


def predict_without_labels(
    predictor: Any,
    manifest_rows: Sequence[Mapping[str, Any]],
    features: pd.DataFrame,
) -> list[dict[str, Any]]:
    if "fight_id" in features.columns:
        features = features.set_index(features["fight_id"].astype(str), drop=True)
    else:
        features = features.copy()
        features.index = features.index.astype(str)
    ids = [str(row["fight_id"]) for row in manifest_rows]
    missing = [fight_id for fight_id in ids if fight_id not in features.index]
    if missing:
        raise EvaluationError(f"test features are missing fight IDs: {missing[:3]}")
    protected = {"y_true", "label", "future_label", "outcome", "result"}
    clean = features.loc[ids].drop(columns=[name for name in features.columns if name.lower() in protected], errors="ignore")
    probabilities = predictor.predict_proba(clean)
    if hasattr(probabilities, "columns"):
        positive = probabilities[1] if 1 in probabilities.columns else probabilities.iloc[:, -1]
    else:
        array = np.asarray(probabilities)
        positive = array[:, -1] if array.ndim == 2 else array
    values = np.asarray(positive, dtype=float)
    if len(values) != len(ids) or not np.all(np.isfinite(values)) or np.any((values < 0) | (values > 1)):
        raise EvaluationError("predictor returned invalid test probabilities")
    return [
        {
            "fight_id": str(row["fight_id"]),
            "event_id": str(row["event_id"]),
            "event_date": str(row["event_date"])[:10],
            "probability": float(values[position]),
        }
        for position, row in enumerate(manifest_rows)
    ]


def assert_production_grammar(
    attempts: Sequence[Mapping[str, Any]],
    access: Sequence[Mapping[str, Any]],
    *,
    selection_frozen: bool,
    require_access: bool,
) -> None:
    launched = [row for row in attempts if row.get("state") == "launched"]
    exited = [row for row in attempts if row.get("state") == "exited"]
    if len(launched) != 1 or len(exited) != 1:
        raise EvaluationError("exactly one production fit marker pair is required")
    if launched[0].get("attempt_id") != exited[0].get("attempt_id"):
        raise EvaluationError("production fit marker identities differ")
    if access and not selection_frozen:
        raise EvaluationError("test access occurred before selection was frozen")
    expected = 1 if require_access else 0
    if len(access) != expected:
        raise EvaluationError(f"exactly one test access is required after scoring (found {len(access)})")
    if access and access[0].get("state") != "opened":
        raise EvaluationError("test access ledger contains an unsupported state")


def _paths(campaign_root: Path) -> dict[str, Path]:
    campaign_root = Path(campaign_root)
    run_root = campaign_root / "runs" / RUN_ID
    return {
        "campaign": campaign_root,
        "run": run_root,
        "artifact": campaign_root / ARTIFACT_SCOPE,
        "preflight": run_root / "preflight.json",
        "preregistration": run_root / "preregistration.json",
        "attempts": run_root / "attempts.jsonl",
        "access": run_root / "test-access.jsonl",
        "selection": run_root / "selection.json",
        "fit_failure": run_root / "fit-failure.json",
        "predictions": run_root / "test-predictions.jsonl",
        "result": run_root / "evaluation.json",
    }


def _git(*args: str, cwd: Path | None = None) -> str:
    completed = subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True, check=False)
    if completed.returncode:
        raise EvaluationError(completed.stderr.strip() or f"git {' '.join(args)} failed")
    return completed.stdout.strip()


def _canonical_registry_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest().upper()


def gpu_process_snapshot() -> dict[str, Any]:
    completed = subprocess.run(
        ["nvidia-smi", "--query-compute-apps=pid,process_name,used_gpu_memory", "--format=csv,noheader"],
        capture_output=True,
        text=True,
        check=False,
    )
    rows = [line.strip() for line in completed.stdout.splitlines() if line.strip()]
    own_prefix = f"{os.getpid()},"
    python_rows = [
        line
        for line in rows
        if "python" in line.lower() and not line.startswith(own_prefix)
    ]
    if completed.returncode or python_rows:
        raise EvaluationError(f"GPU process preflight failed; active Python rows: {python_rows}")
    return {"exit_code": completed.returncode, "rows": rows, "python_rows": python_rows}


def _original_checkout_identity() -> dict[str, Any]:
    root = Path(r"C:\Users\danhm\mma-ai\mma-ai")
    status = subprocess.check_output(["git", "status", "--porcelain=v1", "--untracked-files=all"], cwd=root)
    return {
        "path": str(root),
        "head": _git("rev-parse", "HEAD", cwd=root),
        "status_sha256": hashlib.sha256(status.replace(b"\r\n", b"\n")).hexdigest().upper(),
    }


def durable_preflight(campaign_root: Path, *, source_csv: Path = DEFAULT_SOURCE_CSV) -> dict[str, Any]:
    from libs.modeling.train import CUSTOM_HYPERPARAMETERS, training_runtime_preflight

    paths = _paths(campaign_root)
    if any(paths[name].exists() for name in ("preflight", "preregistration", "attempts", "access", "selection")):
        raise EvaluationError("evaluation preflight is append-once and already exists")
    if paths["artifact"].exists():
        raise EvaluationError("unique ignored evaluation artifact scope already exists")
    if file_sha256(Path(source_csv)) != FROZEN_SOURCE_SHA256:
        raise EvaluationError("sealed source CSV hash mismatch")
    split = verify_split(paths["campaign"], source_csv=Path(source_csv), strict=True)
    profile_path = paths["campaign"] / "profiles/evaluation.json"
    profile = _read_json(profile_path)
    if canonical_sha256(profile) != EXPECTED_PROFILE_SHA256 or split.profile_sha256 != EXPECTED_PROFILE_SHA256:
        raise EvaluationError("exact evaluation profile identity changed")
    if split.manifest_sha256 != EXPECTED_PARTITION_SHA256:
        raise EvaluationError("exact partition identity changed")
    if canonical_sha256(profile["features"]) != FROZEN_FEATURE_SHA256:
        raise EvaluationError("ordered feature identity changed")
    if _canonical_registry_sha(paths["campaign"] / "registry.jsonl") != EXPECTED_REGISTRY_PREFIX:
        raise EvaluationError("registry is not at the frozen two-record prefix")
    if tuple(CUSTOM_HYPERPARAMETERS["hybrid"].keys()) != (
        "CAT", "GBM", "XT", "RF", "FASTAI", "REALMLP", "XGB", "MITRA", "TABICL"
    ):
        raise EvaluationError("hybrid hyperparameter portfolio changed")
    runtime = training_runtime_preflight()
    gpu = gpu_process_snapshot()
    repo = Path.cwd().resolve()
    rollback_ref = _git("rev-parse", "codex/weighted-v8-67-baseline", cwd=repo)
    rollback_tree = _git("rev-parse", "codex/weighted-v8-67-baseline^{tree}", cwd=repo)
    if (rollback_ref, rollback_tree) != (ROLLBACK_REVISION, ROLLBACK_TREE):
        raise EvaluationError("immutable rollback branch moved")
    preregistration = {
        "schema_version": 1,
        "run_id": RUN_ID,
        "variant_bound": 1,
        "fit_attempt_bound": 1,
        "test_access_bound": 1,
        "train_rows": 2473,
        "validation_rows": 309,
        "test_rows": 307,
        "profile_path": "profiles/evaluation.json",
        "profile_sha256": EXPECTED_PROFILE_SHA256,
        "source_csv_path": str(Path(source_csv).resolve()),
        "source_csv_sha256": FROZEN_SOURCE_SHA256,
        "partition_manifest_sha256": EXPECTED_PARTITION_SHA256,
        "feature_sha256": FROZEN_FEATURE_SHA256,
        "expected_base_models": list(EXPECTED_BASE_MODELS),
        "artifact_scope": ARTIFACT_SCOPE,
        "selection_boundary": "Original",
        "full_metrics_admissible": False,
        "context_metrics_admissible": False,
        "refit_full": False,
        "bootstrap": {"block": "event_id", "iterations": 2000, "seed": SEEDS["bootstrap"]},
        "seeds": SEEDS,
    }
    _write_json(paths["preregistration"], preregistration)
    paths["attempts"].write_bytes(b"")
    paths["access"].write_bytes(b"")
    preflight = {
        "schema_version": 1,
        "state": "ready",
        "created_at_utc": utc_now(),
        "source_revision": _git("rev-parse", "HEAD", cwd=repo),
        "source_csv_sha256": FROZEN_SOURCE_SHA256,
        "source_csv_size": Path(source_csv).stat().st_size,
        "profile_file_sha256": file_sha256(profile_path),
        "profile_sha256": EXPECTED_PROFILE_SHA256,
        "partition_manifest_file_sha256": file_sha256(paths["campaign"] / "partitions/manifest.json"),
        "partition_manifest_sha256": EXPECTED_PARTITION_SHA256,
        "partition_hashes": split.partition_hashes,
        "partition_counts": split.partition_counts,
        "retired_count": split.retired_count,
        "retired_label_reads": 0,
        "registry_prefix_sha256": EXPECTED_REGISTRY_PREFIX,
        "attempt_count": 0,
        "test_access_count": 0,
        "active_training_processes": [],
        "gpu_snapshot": gpu,
        "runtime": runtime,
        "cache": {
            "hf_home": os.environ.get("HF_HOME"),
            "torch_home": os.environ.get("TORCH_HOME"),
            "transformers_offline": os.environ.get("TRANSFORMERS_OFFLINE"),
            "hf_hub_offline": os.environ.get("HF_HUB_OFFLINE"),
        },
        "candidate_inventory": list(EXPECTED_BASE_MODELS),
        "rollback": {"revision": rollback_ref, "tree": rollback_tree},
        "original_checkout": _original_checkout_identity(),
        "preservation": {
            "rollback_manifest_sha256": file_sha256(paths["campaign"] / "rollback-manifest.json"),
            "registry_prefix_file_sha256": file_sha256(paths["campaign"] / "registry.jsonl"),
            "source_csv_sha256": FROZEN_SOURCE_SHA256,
        },
        "database_access": False,
        "database_paths": [],
        "forbidden_tokens": [],
    }
    _write_json(paths["preflight"], preflight)
    return preflight


def _partition_rows(campaign_root: Path, name: str) -> list[dict[str, Any]]:
    document = _read_json(Path(campaign_root) / f"partitions/{name}.json")
    if document.get("partition") != name:
        raise EvaluationError(f"partition document mismatch: {name}")
    return list(document["rows"])


def _load_features(source_csv: Path, features: Sequence[str], ids: Sequence[str]) -> pd.DataFrame:
    columns = list(dict.fromkeys([*features, "fight_id", "event_id", "event_date", "method"]))
    frame = pd.read_csv(source_csv, usecols=columns)
    frame["fight_id"] = frame["fight_id"].astype(str)
    selected = frame[frame["fight_id"].isin(set(ids))].copy()
    by_id = selected.set_index("fight_id", drop=False)
    missing = [fight_id for fight_id in ids if fight_id not in by_id.index]
    if missing or len(selected) != len(ids):
        raise EvaluationError(f"source feature membership mismatch: {missing[:3]}")
    return by_id.loc[list(ids)].reset_index(drop=True)


def decode_labels_for_ids(source_csv: Path, ids: Sequence[str]) -> dict[str, int]:
    """Decode only labels whose fight IDs are explicitly authorized by the caller."""
    wanted = set(str(value) for value in ids)
    labels: dict[str, int] = {}
    with Path(source_csv).open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.reader(handle)
        header = next(reader)
        fight_index = header.index("fight_id")
        label_index = header.index("y_true")
        for row in reader:
            fight_id = row[fight_index]
            if fight_id in wanted:
                labels[fight_id] = int(float(row[label_index]))
    if set(labels) != wanted:
        raise EvaluationError("authorized label IDs do not resolve exactly in sealed source")
    return labels


def _seed_runtime() -> None:
    os.environ["PYTHONHASHSEED"] = str(SEEDS["python"])
    random.seed(SEEDS["python"])
    np.random.seed(SEEDS["numpy"])
    import torch

    torch.manual_seed(SEEDS["torch"])
    torch.cuda.manual_seed_all(SEEDS["torch"])


def fit_child(campaign_root: Path, *, source_csv: Path) -> dict[str, Any]:
    from autogluon.tabular import TabularPredictor
    from joblib import dump
    from sklearn.preprocessing import RobustScaler

    from libs.modeling.train import TrainingConfig, build_training_fit_kwargs, training_runtime_preflight

    paths = _paths(campaign_root)
    preflight = _read_json(paths["preflight"])
    prereg = _read_json(paths["preregistration"])
    if preflight.get("state") != "ready" or file_sha256(source_csv) != prereg["source_csv_sha256"]:
        raise EvaluationError("fit child inputs differ from durable preflight")
    if read_jsonl(paths["access"]):
        raise EvaluationError("fit child refuses any prior test access")
    runtime = training_runtime_preflight()
    _seed_runtime()
    profile = _read_json(paths["campaign"] / prereg["profile_path"])
    config = TrainingConfig(**profile)
    train_rows = _partition_rows(paths["campaign"], "train")
    validation_rows = _partition_rows(paths["campaign"], "validation")
    train_ids = [str(row["fight_id"]) for row in train_rows]
    validation_ids = [str(row["fight_id"]) for row in validation_rows]
    if (len(train_ids), len(validation_ids)) != (2473, 309):
        raise EvaluationError("fit requires exact 2,473/309 row partitions")
    features = list(profile["features"])
    frame = _load_features(Path(source_csv), features, [*train_ids, *validation_ids])
    by_id = frame.set_index("fight_id", drop=False)
    train_frame = by_id.loc[train_ids].copy()
    validation_frame = by_id.loc[validation_ids].copy()
    labels = decode_labels_for_ids(Path(source_csv), [*train_ids, *validation_ids])
    scale_columns = [name for name in features if name != "weightclass_encoded"]
    scaler = RobustScaler().fit(train_frame[scale_columns])
    X_train = train_frame[features].copy()
    X_validation = validation_frame[features].copy()
    X_train.loc[:, scale_columns] = scaler.transform(X_train[scale_columns])
    X_validation.loc[:, scale_columns] = scaler.transform(X_validation[scale_columns])
    weights, weight_provenance = compute_recency_weights(
        train_frame["event_date"], decay_rate=float(profile["decay_rate"])
    )
    train_data = X_train.copy()
    train_data["y_true"] = [labels[fight_id] for fight_id in train_ids]
    train_data["sample_weight"] = weights
    tuning_data = X_validation.copy()
    tuning_data["y_true"] = [labels[fight_id] for fight_id in validation_ids]
    tuning_data["sample_weight"] = 1.0
    model_root = paths["artifact"] / "model"
    scaler_path = model_root / "scaler.pkl"
    model_root.mkdir(parents=True, exist_ok=False)
    dump(scaler, scaler_path)
    predictor = TabularPredictor(
        label="y_true",
        eval_metric="log_loss",
        problem_type="binary",
        path=str(model_root),
        verbosity=2,
        sample_weight="sample_weight",
        weight_evaluation=False,
    )
    predictor.fit(**build_training_fit_kwargs(config, train_data=train_data, tuning_data=tuning_data))
    fixed = {
        "source_revision": preflight["source_revision"],
        "profile_sha256": prereg["profile_sha256"],
        "data_sha256": prereg["source_csv_sha256"],
        "partition_sha256": prereg["partition_manifest_sha256"],
        "feature_sha256": prereg["feature_sha256"],
        "train_rows": len(train_ids),
        "validation_rows": len(validation_ids),
        "train_fight_ids_sha256": canonical_sha256(train_ids),
        "validation_fight_ids_sha256": canonical_sha256(validation_ids),
        "sample_weight": weight_provenance,
        "runtime": runtime,
        "seeds": SEEDS,
        "artifact_scope": ARTIFACT_SCOPE,
    }
    selection = freeze_selection(
        predictor,
        model_root=model_root,
        scaler_path=scaler_path,
        fixed_identities=fixed,
    )
    _write_json(paths["selection"], selection)
    return selection


def launch_fit(campaign_root: Path, *, source_csv: Path = DEFAULT_SOURCE_CSV, timeout_seconds: int = 3300) -> dict[str, Any]:
    paths = _paths(campaign_root)
    prereg = _read_json(paths["preregistration"])
    preflight = _read_json(paths["preflight"])
    if preflight.get("state") != "ready" or read_jsonl(paths["attempts"]) or read_jsonl(paths["access"]):
        raise EvaluationError("fit launch requires unused durable preflight")
    if paths["artifact"].exists() or paths["selection"].exists():
        raise EvaluationError("fit launch refuses reused production destinations")
    if str(Path(source_csv).resolve()) != prereg["source_csv_path"] or file_sha256(source_csv) != FROZEN_SOURCE_SHA256:
        raise EvaluationError("fit launch source differs from preregistration")
    snapshot = gpu_process_snapshot()
    paths["artifact"].mkdir(parents=True, exist_ok=False)
    lock = paths["artifact"] / "gpu.lock"
    descriptor = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    stdout_path = paths["artifact"] / "fit.stdout.log"
    stderr_path = paths["artifact"] / "fit.stderr.log"
    attempt_id = "evaluation-fit-attempt-1"
    command = [
        sys.executable,
        "-m",
        "libs.modeling.split_refit_experiment",
        "fit-child",
        "--campaign",
        str(paths["campaign"]),
        "--source-csv",
        str(Path(source_csv).resolve()),
    ]
    launched = {
        "attempt_id": attempt_id,
        "state": "launched",
        "launched_at_utc": utc_now(),
        "launched_unix_ns": time.time_ns(),
        "command": command,
        "train_rows": 2473,
        "validation_rows": 309,
        "test_label_reads": 0,
        "gpu_lease": "single-exclusive-file",
        "prelaunch_gpu_processes": snapshot,
        "profile_sha256": prereg["profile_sha256"],
        "partition_sha256": prereg["partition_manifest_sha256"],
        "source_sha256": prereg["source_csv_sha256"],
    }
    _append_jsonl(paths["attempts"], launched)
    start = time.monotonic()
    exit_code = 124
    exit_reason = "timeout"
    try:
        with stdout_path.open("wb") as stdout, stderr_path.open("wb") as stderr:
            try:
                completed = subprocess.run(
                    command,
                    cwd=Path.cwd(),
                    stdout=stdout,
                    stderr=stderr,
                    timeout=timeout_seconds,
                    check=False,
                )
                exit_code = completed.returncode
                exit_reason = "process-exit"
            except subprocess.TimeoutExpired:
                stderr.write(f"\nfit exceeded authorized timeout of {timeout_seconds}s\n".encode())
    finally:
        os.close(descriptor)
        lock.unlink()
    exited = {
        "attempt_id": attempt_id,
        "state": "exited",
        "exited_at_utc": utc_now(),
        "exited_unix_ns": time.time_ns(),
        "exit_code": exit_code,
        "exit_reason": exit_reason,
        "elapsed_seconds": time.monotonic() - start,
        "stdout_path": str(stdout_path.resolve()),
        "stdout_sha256": file_sha256(stdout_path),
        "stderr_path": str(stderr_path.resolve()),
        "stderr_sha256": file_sha256(stderr_path),
        "test_label_reads": 0,
    }
    _append_jsonl(paths["attempts"], exited)
    if exit_code != 0 or not paths["selection"].is_file():
        failure = {"schema_version": 1, "state": "failed", "attempt": exited, "retry_authorized": False}
        failure_sha = _write_json(paths["fit_failure"], failure)
        append_registry_record(
            paths["campaign"],
            record_id="evaluation-fit-failure",
            payload={
                "kind": "evaluation-fit-failure",
                "artifact_path": f"runs/{RUN_ID}/fit-failure.json",
                "artifact_sha256": failure_sha,
            },
        )
        raise EvaluationError(f"sole production fit failed with exit code {exit_code}; retry is forbidden")
    selection_sha = file_sha256(paths["selection"])
    append_registry_record(
        paths["campaign"],
        record_id="evaluation-selection",
        payload={
            "kind": "evaluation-selection",
            "artifact_path": f"runs/{RUN_ID}/selection.json",
            "artifact_sha256": selection_sha,
            "model_tree_sha256": _read_json(paths["selection"])["model_tree"]["sha256"],
        },
    )
    return {"status": "complete", "selection_sha256": selection_sha, "attempt": exited}


def _selection_commit(paths: Mapping[str, Path]) -> str:
    repo = Path.cwd().resolve()
    relative = paths["selection"].resolve().relative_to(repo).as_posix()
    if subprocess.run(["git", "diff", "--quiet", "HEAD", "--", relative], cwd=repo).returncode:
        raise EvaluationError("selection manifest must be committed before test access")
    if subprocess.run(["git", "ls-files", "--error-unmatch", relative], cwd=repo, capture_output=True).returncode:
        raise EvaluationError("selection manifest must be tracked before test access")
    commit = _git("log", "-1", "--format=%H", "--", relative, cwd=repo)
    if not commit:
        raise EvaluationError("selection manifest has no committed identity")
    return commit


def _load_frozen_predictor(selection: Mapping[str, Any]):
    from autogluon.tabular import TabularPredictor
    from joblib import load

    predictor = TabularPredictor.load(selection["model_root"])
    scaler = load(selection["scaler_path"])
    if predictor.model_best != selection["selected_node"]:
        raise EvaluationError("loaded predictor selection differs from frozen manifest")
    return predictor, scaler


def _direct_event_intervals(records: Sequence[Mapping[str, Any]], *, iterations: int, seed: int) -> dict[str, Any]:
    from libs.modeling.experiment_campaign.metrics import reduce_predictions

    events = tuple(dict.fromkeys(str(row["event_id"]) for row in records))
    grouped = {event: [row for row in records if str(row["event_id"]) == event] for event in events}
    rng = np.random.default_rng(seed)
    samples = {name: [] for name in ("accuracy", "log_loss", "brier")}
    for _ in range(iterations):
        chosen = rng.choice(events, size=len(events), replace=True)
        sample = [row for event in chosen for row in grouped[str(event)]]
        reduced = reduce_predictions(sample)
        for name in samples:
            samples[name].append(float(getattr(reduced, name)))
    point = reduce_predictions(records)
    return {
        name: {
            "estimate": float(getattr(point, name)),
            "lower": float(np.quantile(values, 0.025)),
            "upper": float(np.quantile(values, 0.975)),
            "iterations": iterations,
            "seed": seed,
            "block": "event_id",
        }
        for name, values in samples.items()
    }


def score_evaluation(campaign_root: Path, *, source_csv: Path = DEFAULT_SOURCE_CSV) -> dict[str, Any]:
    from libs.modeling.experiment_campaign.metrics import reduce_predictions

    paths = _paths(campaign_root)
    attempts = read_jsonl(paths["attempts"])
    access = read_jsonl(paths["access"])
    selection = _read_json(paths["selection"])
    assert_production_grammar(attempts, access, selection_frozen=True, require_access=False)
    if attempts[-1].get("exit_code") != 0:
        raise EvaluationError("failed fit cannot be scored")
    if paths["predictions"].exists() or paths["result"].exists():
        raise EvaluationError("test scorer is append-once")
    selection_commit = _selection_commit(paths)
    selection_sha = file_sha256(paths["selection"])
    if tree_identity(Path(selection["model_root"])) != selection["model_tree"]:
        raise EvaluationError("model bytes changed before test access")
    prereg = _read_json(paths["preregistration"])
    if file_sha256(source_csv) != prereg["source_csv_sha256"]:
        raise EvaluationError("sealed source changed before scoring")
    test_rows = _partition_rows(paths["campaign"], "test")
    ids = [str(row["fight_id"]) for row in test_rows]
    if len(ids) != 307:
        raise EvaluationError("test scorer requires exactly 307 manifest IDs")
    source_features = _load_features(Path(source_csv), prereg["expected_base_models"][:0] or _read_json(paths["campaign"] / prereg["profile_path"])["features"], ids)
    predictor, scaler = _load_frozen_predictor(selection)
    features = list(_read_json(paths["campaign"] / prereg["profile_path"])["features"])
    scale_columns = [name for name in features if name != "weightclass_encoded"]
    scaled = source_features.copy()
    scaled.loc[:, scale_columns] = scaler.transform(scaled[scale_columns])
    prediction_frame = scaled.set_index("fight_id")[features]
    unlabeled = predict_without_labels(predictor, test_rows, prediction_frame)
    unlabeled_sha = canonical_sha256(unlabeled)
    access_record = {
        "access_id": "retrospective-test-access-1",
        "state": "opened",
        "opened_at_utc": utc_now(),
        "opened_unix_ns": time.time_ns(),
        "selection_sha256": selection_sha,
        "selection_commit": selection_commit,
        "unlabeled_prediction_sha256": unlabeled_sha,
        "row_count": 307,
        "fight_ids": ids,
        "label_source_sha256": FROZEN_SOURCE_SHA256,
        "label_decode_count": 307,
    }
    _append_jsonl(paths["access"], access_record)
    labels = decode_labels_for_ids(Path(source_csv), ids)
    source_by_id = source_features.set_index("fight_id", drop=False)
    records = []
    for row in unlabeled:
        source = source_by_id.loc[row["fight_id"]]
        records.append(
            {
                **row,
                "y_true": labels[row["fight_id"]],
                "boundary": "Original",
                "fit_scope": "prior-only",
                "fold": "retrospective-test",
                "weight_class": str(source.get("weightclass_encoded", "unknown")),
                "experience": "not-recomputed",
                "outcome_type": "decision" if "dec" in str(source.get("method", "")).lower() else "finish",
            }
        )
    with paths["predictions"].open("wb") as handle:
        for record in records:
            handle.write(canonical_json_bytes(record) + b"\n")
    metrics = reduce_predictions(records).as_dict()
    intervals = _direct_event_intervals(records, iterations=2000, seed=SEEDS["bootstrap"])
    result = {
        "schema_version": 1,
        "state": "complete",
        "retrospective": True,
        "row_count": 307,
        "prediction_sha256": file_sha256(paths["predictions"]),
        "unlabeled_prediction_sha256": unlabeled_sha,
        "selection_sha256": selection_sha,
        "selection_commit": selection_commit,
        "metrics": metrics,
        "event_block_intervals": intervals,
        "historical_boundaries": {
            "retrospective_test": 307,
            "accepted_tuning": 460,
            "nested_outer": 1108,
            "pooled": False,
            "claim": "all three are historical and selection-exposed to different degrees",
        },
        "post_test_adaptation": False,
        "retry_authorized": False,
        "model_tree_sha256": selection["model_tree"]["sha256"],
        "profile_sha256": selection["profile_sha256"],
    }
    result_sha = _write_json(paths["result"], result)
    append_registry_record(
        paths["campaign"],
        record_id="evaluation-result",
        payload={
            "kind": "evaluation-result",
            "artifact_path": f"runs/{RUN_ID}/evaluation.json",
            "artifact_sha256": result_sha,
            "predictions_path": f"runs/{RUN_ID}/test-predictions.jsonl",
            "predictions_sha256": result["prediction_sha256"],
            "selection_sha256": selection_sha,
        },
    )
    return result
