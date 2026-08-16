"""Append-once full-data deployment refit for the frozen weighted-v8 profile."""

from __future__ import annotations

import hashlib
import json
import os
import random
import subprocess
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from .protocol import FROZEN_FEATURE_SHA256, FROZEN_SOURCE_SHA256, canonical_sha256, file_sha256
from .registry import append_registry_record
from .runner import (
    EXPECTED_BASE_MODELS,
    ROLLBACK_REVISION,
    ROLLBACK_TREE,
    SEEDS,
    _append_jsonl,
    _find_weights,
    _git,
    _json_safe,
    _original_checkout_identity,
    _read_json,
    _write_json,
    gpu_process_snapshot,
    read_jsonl,
    tree_identity,
    utc_now,
)


RUN_ID = "full-data-refit"
ARTIFACT_SCOPE = "artifacts/03-full-data-refit"
PROFILE_NAME = "v8-hybrid-weighted"
EXPECTED_PROFILE_SHA256 = "55B750C16528AC07ECF0B9E8D9AD557308F4D9087A9A5DA86E24D8A62E8684A0"
EXPECTED_REGISTRY_PREFIX = "AF492E7F6C1EA7ED5AC0C4C8BA23930D33855235881FE950F0ACCEF2FFF95A2A"
EXPECTED_EVALUATION_SELECTION_SHA256 = "9376C1A934EE5D7139AC25D69C2770848E1C1C49744B17E4A7B388225C54E4AF"
BASELINE_REVISION = "70233a10c24cc240f84584cc6979717c46abf51e"
ACCEPTED_MODEL = Path(r"C:\Users\danhm\mma-ai\mma-ai\AutogluonModels\ag-20260815_090928-win-hybrid")
NO_RECENCY_MODEL = Path(r"C:\Users\danhm\mma-ai\mma-ai\AutogluonModels\ag-20260815_163858-win-hybrid")
TOP10_WORKTREE = Path(r"C:\Users\danhm\mma-ai\worktrees\top10-20260815")
PROBE_ROWS_PER_PARTITION = 32


class RefitError(ValueError):
    pass


def _canonical_registry_sha(path: Path) -> str:
    raw = path.read_bytes().replace(b"\r\n", b"\n")
    return hashlib.sha256(raw).hexdigest().upper()


def _registered_file_sha256(path: Path) -> str:
    raw = Path(path).read_bytes().replace(b"\r\n", b"\n")
    return hashlib.sha256(raw).hexdigest().upper()


def _paths(campaign_root: Path) -> dict[str, Path]:
    campaign = Path(campaign_root)
    run = campaign / "runs" / RUN_ID
    artifact = campaign / ARTIFACT_SCOPE
    return {
        "campaign": campaign,
        "run": run,
        "artifact": artifact,
        "model": artifact / "model",
        "preflight": run / "preflight.json",
        "preregistration": run / "preregistration.json",
        "attempts": run / "attempts.jsonl",
        "result": run / "refit.json",
        "failure": run / "fit-failure.json",
    }


def validate_named_profile(
    profile: Mapping[str, Any], *, expected_profile: Mapping[str, Any], expected_feature_count: int
) -> None:
    features = profile.get("features")
    if not isinstance(features, (list, tuple)) or len(features) != expected_feature_count:
        raise RefitError("named profile feature count changed")
    if len(features) != len(set(features)):
        raise RefitError("named profile features are duplicated")
    if len(profile) != 23 or dict(profile) != dict(expected_profile):
        raise RefitError("named profile differs from the exact frozen profile")


def validate_full_population(fight_ids: Sequence[Any]) -> list[str]:
    values = [str(value) for value in fight_ids]
    if len(values) != 3267 or len(values) != len(set(values)):
        raise RefitError("full refit requires exactly 3,267 unique eligible fights")
    return values


def build_refit_invocation(
    *,
    source_csv: str | Path,
    model_root: str | Path,
    source_rows: int,
    profile: Mapping[str, Any],
    destination_exists: bool | None = None,
) -> dict[str, Any]:
    if source_rows != 3267:
        raise RefitError("full refit invocation requires exactly 3,267 rows")
    model_root = Path(model_root)
    exists = model_root.exists() if destination_exists is None else destination_exists
    if exists:
        raise RefitError("full refit destination must be unique and absent")
    return {
        "call": 'libs.modeling.training_profiles.train_profile("v8-hybrid-weighted")',
        "call_ordinal": 1,
        "profile_name": PROFILE_NAME,
        "profile_delta_from_named": {},
        "source_csv": str(Path(source_csv).resolve()),
        "source_rows": source_rows,
        "model_root": str(model_root.resolve()),
        "refit_full": profile.get("refit_full"),
        "test_guided_config_delta": False,
    }


def assert_single_refit_attempt(
    attempts: Sequence[Mapping[str, Any]], *, require_success: bool
) -> None:
    launched = [row for row in attempts if row.get("state") == "launched"]
    exited = [row for row in attempts if row.get("state") == "exited"]
    if len(launched) != 1 or len(exited) != 1:
        if len(attempts) == 1:
            raise RefitError("refit attempt marker pair is incomplete")
        raise RefitError("exactly one refit production attempt is required")
    if launched[0].get("attempt_id") != exited[0].get("attempt_id"):
        raise RefitError("refit attempt marker identities differ")
    if require_success and exited[0].get("exit_code") != 0:
        raise RefitError("the sole refit attempt did not succeed")


def _node_weights(info: Mapping[str, Any]) -> dict[str, float]:
    if isinstance(info.get("weights"), Mapping):
        return {str(key): float(value) for key, value in info["weights"].items()}
    return _find_weights(info) or {}


def classify_saved_lineage(
    model_info: Mapping[str, Mapping[str, Any]],
    prediction_hashes: Mapping[str, str],
    *,
    total_rows: int,
) -> dict[str, dict[str, Any]]:
    if total_rows != 3267:
        raise RefitError("lineage requires the exact 3,267-row population")
    names = list(model_info)
    missing = [name for name in names if name not in prediction_hashes]
    if missing:
        raise RefitError(f"prediction identity is missing for saved nodes: {missing}")
    lineage: dict[str, dict[str, Any]] = {}
    for name in names:
        if name.endswith("_FULL"):
            continue
        info = model_info[name]
        lineage[name] = {
            "boundary": "Original",
            "origin": "internal-selection-fit",
            "fit_rows": int(info.get("num_samples", 0)),
            "dependencies": sorted(_node_weights(info)),
            "prediction_sha256": prediction_hashes[name],
            "metric_claim": "none",
            "context_contaminated": True,
        }
    pending = [name for name in names if name.endswith("_FULL")]
    pending.sort(key=lambda name: "WeightedEnsemble" in str(model_info[name].get("model_type", "")))
    for name in pending:
        info = model_info[name]
        parent = name[:-5]
        if parent not in model_info:
            raise RefitError(f"FULL node has no Original parent: {name}")
        rows = int(info.get("num_samples", 0))
        weights = _node_weights(info)
        dependencies = sorted(weights)
        is_ensemble = "Ensemble" in str(info.get("model_type", ""))
        if rows == total_rows:
            origin = "fresh-full-fit"
            effective_rows = total_rows
        elif is_ensemble and dependencies and all(dep in lineage for dep in dependencies):
            origin = "cloned-ensemble-wrapper"
            effective_rows = max(
                int(lineage[dep].get("effective_fit_rows", lineage[dep]["fit_rows"]))
                for dep in dependencies
            )
        elif prediction_hashes[name] == prediction_hashes[parent]:
            origin = "original-clone"
            effective_rows = rows
        else:
            raise RefitError(f"unsupported FULL fit boundary for {name}")
        lineage[name] = {
            "boundary": "FULL",
            "origin": origin,
            "fit_rows": rows,
            "effective_fit_rows": effective_rows,
            "dependencies": dependencies,
            "prediction_sha256": prediction_hashes[name],
            "parent_prediction_sha256": prediction_hashes[parent],
            "metric_claim": "none",
            "context_contaminated": True,
        }
    return lineage


def _load_population_ids(campaign: Path) -> list[str]:
    ids: list[str] = []
    for partition in ("train", "validation", "test", "retired"):
        document = _read_json(campaign / "partitions" / f"{partition}.json")
        ids.extend(str(value) for value in document["fight_ids"])
    return validate_full_population(ids)


def _repo_identity(root: Path) -> dict[str, str]:
    status = subprocess.check_output(
        ["git", "status", "--porcelain=v1", "--untracked-files=all"], cwd=root
    ).replace(b"\r\n", b"\n")
    return {
        "path": str(root.resolve()),
        "head": _git("rev-parse", "HEAD", cwd=root),
        "status_sha256": hashlib.sha256(status).hexdigest().upper(),
    }


def _preservation_snapshot(campaign: Path, selection: Mapping[str, Any]) -> dict[str, Any]:
    evaluation_root = Path(str(selection["model_root"]))
    evaluation_tree = tree_identity(evaluation_root)
    if evaluation_tree != selection.get("model_tree"):
        raise RefitError("frozen evaluation model tree changed")
    return {
        "rollback_revision": _git("rev-parse", "codex/weighted-v8-67-baseline"),
        "rollback_tree": _git("rev-parse", "codex/weighted-v8-67-baseline^{tree}"),
        "original_checkout": _original_checkout_identity(),
        "top10_worktree": _repo_identity(TOP10_WORKTREE),
        "accepted_model_tree": tree_identity(ACCEPTED_MODEL),
        "no_recency_model_tree": tree_identity(NO_RECENCY_MODEL),
        "evaluation_model_tree": evaluation_tree,
        "evaluation_selection_sha256": file_sha256(
            campaign / "runs/80-10-10-evaluation/selection.json"
        ),
        "evaluation_result_sha256": file_sha256(
            campaign / "runs/80-10-10-evaluation/evaluation.json"
        ),
        "evaluation_predictions_sha256": file_sha256(
            campaign / "runs/80-10-10-evaluation/test-predictions.jsonl"
        ),
        "rollback_manifest_sha256": file_sha256(campaign / "rollback-manifest.json"),
    }


def durable_refit_preflight(campaign_root: Path, *, source_csv: Path) -> dict[str, Any]:
    from libs.modeling.train import CUSTOM_HYPERPARAMETERS, training_runtime_preflight
    from libs.modeling.training_profiles import get_training_profile

    paths = _paths(campaign_root)
    if any(paths[key].exists() for key in ("preflight", "preregistration", "attempts", "result", "failure")):
        raise RefitError("full-data refit preflight is append-once and already exists")
    if paths["artifact"].exists():
        raise RefitError("unique full-data refit artifact destination already exists")
    source_csv = Path(source_csv).resolve()
    if file_sha256(source_csv) != FROZEN_SOURCE_SHA256:
        raise RefitError("sealed source CSV hash changed")
    registry_before = _canonical_registry_sha(paths["campaign"] / "registry.jsonl")
    if registry_before != EXPECTED_REGISTRY_PREFIX:
        raise RefitError("registry is not at the exact post-evaluation prefix")
    rollback = _read_json(paths["campaign"] / "rollback-manifest.json")
    profile = asdict(get_training_profile(PROFILE_NAME))
    expected_profile = rollback["reproduction"]["profile"]["fields"]
    validate_named_profile(profile, expected_profile=expected_profile, expected_feature_count=40)
    if canonical_sha256(profile) != EXPECTED_PROFILE_SHA256:
        raise RefitError("named profile canonical identity changed")
    if canonical_sha256(profile["features"]) != FROZEN_FEATURE_SHA256:
        raise RefitError("ordered feature identity changed")
    ids = _load_population_ids(paths["campaign"])
    selection_path = paths["campaign"] / "runs/80-10-10-evaluation/selection.json"
    selection = _read_json(selection_path)
    if _registered_file_sha256(selection_path) != EXPECTED_EVALUATION_SELECTION_SHA256:
        raise RefitError("evaluation selection identity changed")
    if tuple(selection.get("base_models", ())) != EXPECTED_BASE_MODELS:
        raise RefitError("evaluation ten-model resolver changed")
    if tuple(CUSTOM_HYPERPARAMETERS["hybrid"].keys()) != (
        "CAT", "GBM", "XT", "RF", "FASTAI", "REALMLP", "XGB", "MITRA", "TABICL"
    ):
        raise RefitError("hybrid ten-model resolver changed")
    runtime = training_runtime_preflight()
    gpu = gpu_process_snapshot()
    preservation = _preservation_snapshot(paths["campaign"], selection)
    if (
        preservation["rollback_revision"],
        preservation["rollback_tree"],
    ) != (ROLLBACK_REVISION, ROLLBACK_TREE):
        raise RefitError("immutable rollback identity changed")
    invocation = build_refit_invocation(
        source_csv=source_csv,
        model_root=paths["model"],
        source_rows=len(ids),
        profile=profile,
    )
    preregistration = {
        "schema_version": 1,
        "run_id": RUN_ID,
        "variant_bound": 1,
        "fit_attempt_bound": 1,
        "profile_name": PROFILE_NAME,
        "profile": profile,
        "profile_sha256": canonical_sha256(profile),
        "feature_sha256": canonical_sha256(profile["features"]),
        "source_csv_path": str(source_csv),
        "source_csv_sha256": FROZEN_SOURCE_SHA256,
        "source_rows": len(ids),
        "fight_ids_sha256": canonical_sha256(ids),
        "candidate_inventory": list(EXPECTED_BASE_MODELS),
        "invocation": invocation,
        "evaluation_selection_sha256": EXPECTED_EVALUATION_SELECTION_SHA256,
        "evaluation_result_is_config_input": False,
        "validation_claims_allowed": False,
        "database_access_allowed": False,
    }
    _write_json(paths["preregistration"], preregistration)
    paths["attempts"].parent.mkdir(parents=True, exist_ok=True)
    paths["attempts"].write_bytes(b"")
    preflight = {
        "schema_version": 1,
        "state": "ready",
        "created_at_utc": utc_now(),
        "source_revision": _git("rev-parse", "HEAD"),
        "registry_before_sha256": registry_before,
        "source_csv_sha256": FROZEN_SOURCE_SHA256,
        "source_csv_size": source_csv.stat().st_size,
        "source_rows": len(ids),
        "feature_count": len(profile["features"]),
        "profile_sha256": canonical_sha256(profile),
        "runtime": runtime,
        "gpu_snapshot": gpu,
        "cache": {
            "hf_home": os.environ.get("HF_HOME"),
            "torch_home": os.environ.get("TORCH_HOME"),
            "transformers_offline": os.environ.get("TRANSFORMERS_OFFLINE"),
            "hf_hub_offline": os.environ.get("HF_HUB_OFFLINE"),
        },
        "active_training_processes": [],
        "unique_destination": str(paths["model"].resolve()),
        "preservation": preservation,
        "database_access": False,
        "database_paths": [],
    }
    _write_json(paths["preflight"], preflight)
    return preflight


def _seed_runtime() -> None:
    os.environ["PYTHONHASHSEED"] = str(SEEDS["python"])
    random.seed(SEEDS["python"])
    np.random.seed(SEEDS["numpy"])
    import torch

    torch.manual_seed(SEEDS["torch"])
    torch.cuda.manual_seed_all(SEEDS["torch"])


def _standard_model_info(predictor: Any) -> dict[str, dict[str, Any]]:
    raw = predictor.info()["model_info"]
    result: dict[str, dict[str, Any]] = {}
    for name in predictor.model_names():
        info = raw[str(name)]
        result[str(name)] = {
            "model_type": str(info.get("model_type", "unknown")),
            "num_samples": int(info.get("num_samples", 0)),
            "weights": _find_weights(info) or {},
        }
    return result


def _probe_frame(model_root: Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    train = pd.read_pickle(model_root / "utils/data/X.pkl").head(PROBE_ROWS_PER_PARTITION)
    tune = pd.read_pickle(model_root / "utils/data/X_val.pkl").head(PROBE_ROWS_PER_PARTITION)
    probe = pd.concat([train, tune], axis=0).drop(columns=["sample_weight"], errors="ignore")
    return probe, {
        "train_rows": len(train),
        "tuning_rows": len(tune),
        "feature_sha256": canonical_sha256([str(value) for value in probe.columns]),
    }


def prediction_identities(predictor: Any, model_root: Path) -> tuple[dict[str, str], dict[str, Any]]:
    probe, identity = _probe_frame(model_root)
    hashes: dict[str, str] = {}
    for name in predictor.model_names():
        probabilities = predictor.predict_proba(probe, model=str(name), as_multiclass=False)
        values = [float(value) for value in np.asarray(probabilities, dtype=float)]
        if len(values) != len(probe) or not np.all(np.isfinite(values)):
            raise RefitError(f"invalid saved-node probe predictions: {name}")
        hashes[str(name)] = canonical_sha256(values)
    return hashes, {**identity, "row_count": len(probe)}


def _native_tree(model_root: Path) -> dict[str, Any]:
    included = []
    for path in sorted(candidate for candidate in model_root.rglob("*") if candidate.is_file()):
        relative = path.relative_to(model_root).as_posix()
        if relative in {"evals.txt", "feature_importance.csv", "training_data.csv", "feats.txt"}:
            continue
        included.append({"path": relative, "size": path.stat().st_size, "sha256": file_sha256(path)})
    return {"file_count": len(included), "sha256": canonical_sha256(included), "files": included}


def refit_child(campaign_root: Path, *, source_csv: Path) -> dict[str, Any]:
    from libs.modeling import training_profiles
    from libs.modeling.train import FileManager, training_runtime_preflight

    paths = _paths(campaign_root)
    preflight = _read_json(paths["preflight"])
    prereg = _read_json(paths["preregistration"])
    source_csv = Path(source_csv).resolve()
    if preflight.get("state") != "ready" or file_sha256(source_csv) != FROZEN_SOURCE_SHA256:
        raise RefitError("refit child inputs differ from durable preflight")
    if paths["model"].exists() or paths["result"].exists():
        raise RefitError("refit child refuses a reused result destination")
    profile = asdict(training_profiles.get_training_profile(PROFILE_NAME))
    validate_named_profile(profile, expected_profile=prereg["profile"], expected_feature_count=40)
    _seed_runtime()
    runtime = training_runtime_preflight()
    source_before = file_sha256(source_csv)
    old_data_dir = os.environ.get("MMA_AI_DATA_DIR")
    original_factory = FileManager.create_model_directory

    def exact_destination(model_type: str, preset: str, suffix: str = "") -> str:
        if (model_type, preset, suffix) != ("win", "hybrid", ""):
            raise RefitError("named profile attempted a different model destination grammar")
        paths["model"].mkdir(parents=True, exist_ok=False)
        return str(paths["model"].resolve())

    os.environ["MMA_AI_DATA_DIR"] = str(source_csv.parent)
    FileManager.create_model_directory = staticmethod(exact_destination)
    try:
        predictor = training_profiles.train_profile(PROFILE_NAME)
    finally:
        FileManager.create_model_directory = original_factory
        if old_data_dir is None:
            os.environ.pop("MMA_AI_DATA_DIR", None)
        else:
            os.environ["MMA_AI_DATA_DIR"] = old_data_dir
    if file_sha256(source_csv) != source_before:
        raise RefitError("sealed source changed during production invocation")
    saved_training = pd.read_csv(paths["model"] / "training_data.csv", usecols=["fight_id"])
    saved_ids = validate_full_population(saved_training["fight_id"].astype(str).tolist())
    if canonical_sha256(saved_ids) != prereg["fight_ids_sha256"]:
        raise RefitError("saved full-data population/order differs from preregistration")
    names = [str(name) for name in predictor.model_names()]
    originals = [name for name in names if not name.endswith("_FULL") and not name.startswith("WeightedEnsemble")]
    if tuple(originals) != EXPECTED_BASE_MODELS:
        raise RefitError("saved Original graph differs from ten-model weighted-v8 portfolio")
    if not any(name.endswith("_FULL") for name in names):
        raise RefitError("named production profile did not create FULL nodes")
    model_info = _standard_model_info(predictor)
    prediction_hashes, probe = prediction_identities(predictor, paths["model"])
    lineage = classify_saved_lineage(model_info, prediction_hashes, total_rows=len(saved_ids))
    graph = [
        {
            "name": name,
            "model_type": model_info[name]["model_type"],
            "dependencies": sorted(model_info[name]["weights"]),
            "weights": model_info[name]["weights"],
        }
        for name in names
    ]
    scaler_path = paths["model"] / "scaler.pkl"
    if not scaler_path.is_file():
        raise RefitError("full-data predictor scaler is missing")
    result = {
        "schema_version": 1,
        "state": "complete",
        "created_at_utc": utc_now(),
        "profile_name": PROFILE_NAME,
        "profile_sha256": prereg["profile_sha256"],
        "profile": profile,
        "feature_count": 40,
        "feature_sha256": prereg["feature_sha256"],
        "source_csv_path": str(source_csv),
        "source_sha256": source_before,
        "source_rows": len(saved_ids),
        "fight_ids_sha256": canonical_sha256(saved_ids),
        "fit_invocation_count": 1,
        "invocation": prereg["invocation"],
        "test_guided_config_delta": False,
        "evaluation_result_is_config_input": False,
        "model_root": str(paths["model"].resolve()),
        "selected_node": str(predictor.model_best),
        "classes": [_json_safe(value) for value in predictor.class_labels],
        "model_graph": graph,
        "lineage": lineage,
        "prediction_probe": probe,
        "prediction_hashes": prediction_hashes,
        "scaler_path": str(scaler_path.resolve()),
        "scaler_sha256": file_sha256(scaler_path),
        "runtime": runtime,
        "seeds": SEEDS,
        "complete_tree": tree_identity(paths["model"]),
        "native_tree": _native_tree(paths["model"]),
        "validation_claims": [],
        "full_and_context_metrics_admissible": False,
        "database_access": False,
        "registry_before_sha256": preflight["registry_before_sha256"],
        "preservation_before": preflight["preservation"],
    }
    _write_json(paths["result"], result)
    return result


def launch_refit(
    campaign_root: Path, *, source_csv: Path, timeout_seconds: int = 3900
) -> dict[str, Any]:
    paths = _paths(campaign_root)
    preflight = _read_json(paths["preflight"])
    prereg = _read_json(paths["preregistration"])
    if preflight.get("state") != "ready" or read_jsonl(paths["attempts"]):
        raise RefitError("refit launch requires an unused durable preflight")
    if paths["artifact"].exists() or paths["result"].exists():
        raise RefitError("refit launch refuses a reused production destination")
    source_csv = Path(source_csv).resolve()
    if str(source_csv) != prereg["source_csv_path"] or file_sha256(source_csv) != FROZEN_SOURCE_SHA256:
        raise RefitError("refit launch source differs from preregistration")
    snapshot = gpu_process_snapshot()
    paths["artifact"].mkdir(parents=True, exist_ok=False)
    lock = paths["artifact"] / "gpu.lock"
    descriptor = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    stdout_path = paths["artifact"] / "fit.stdout.log"
    stderr_path = paths["artifact"] / "fit.stderr.log"
    command = [
        sys.executable,
        "-m",
        "libs.modeling.split_refit_experiment",
        "refit-child",
        "--campaign",
        str(paths["campaign"]),
        "--source-csv",
        str(source_csv),
    ]
    attempt_id = "full-data-refit-attempt-1"
    launched = {
        "attempt_id": attempt_id,
        "state": "launched",
        "launched_at_utc": utc_now(),
        "launched_unix_ns": time.time_ns(),
        "command": command,
        "profile_name": PROFILE_NAME,
        "call_ordinal": 1,
        "source_rows": 3267,
        "gpu_lease": "single-exclusive-file",
        "prelaunch_gpu_processes": snapshot,
        "database_access": False,
    }
    _append_jsonl(paths["attempts"], launched)
    child_env = os.environ.copy()
    for key in list(child_env):
        if "DATABASE" in key.upper():
            child_env.pop(key)
    start = time.monotonic()
    exit_code = 124
    exit_reason = "timeout"
    try:
        with stdout_path.open("wb") as stdout, stderr_path.open("wb") as stderr:
            try:
                completed = subprocess.run(
                    command,
                    cwd=Path.cwd(),
                    env=child_env,
                    stdout=stdout,
                    stderr=stderr,
                    timeout=timeout_seconds,
                    check=False,
                )
                exit_code = completed.returncode
                exit_reason = "process-exit"
            except subprocess.TimeoutExpired:
                stderr.write(f"\nrefit exceeded authorized timeout of {timeout_seconds}s\n".encode())
    finally:
        os.close(descriptor)
        lock.unlink(missing_ok=True)
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
        "retry_authorized": False,
        "database_access": False,
    }
    _append_jsonl(paths["attempts"], exited)
    if exit_code != 0 or not paths["result"].is_file():
        failure = {
            "schema_version": 1,
            "state": "failed",
            "attempt": exited,
            "retry_authorized": False,
        }
        failure_sha = _write_json(paths["failure"], failure)
        append_registry_record(
            paths["campaign"],
            record_id="full-data-refit-failure",
            payload={
                "kind": "full-data-refit-failure",
                "artifact_path": f"runs/{RUN_ID}/fit-failure.json",
                "artifact_sha256": failure_sha,
            },
        )
        raise RefitError(f"sole full-data refit failed with exit code {exit_code}; retry is forbidden")
    result = _read_json(paths["result"])
    result_sha = file_sha256(paths["result"])
    append_registry_record(
        paths["campaign"],
        record_id="full-data-refit",
        payload={
            "kind": "full-data-refit",
            "artifact_path": f"runs/{RUN_ID}/refit.json",
            "artifact_sha256": result_sha,
            "model_tree_sha256": result["complete_tree"]["sha256"],
        },
    )
    return {
        "status": "complete",
        "result_sha256": result_sha,
        "registry_before_sha256": preflight["registry_before_sha256"],
        "registry_after_sha256": _canonical_registry_sha(paths["campaign"] / "registry.jsonl"),
        "attempt": exited,
    }


def verify_refit_attempt(campaign_root: Path, *, strict: bool) -> dict[str, Any]:
    if not strict:
        raise RefitError("refit attempt verification requires --strict")
    paths = _paths(campaign_root)
    attempts = read_jsonl(paths["attempts"])
    assert_single_refit_attempt(attempts, require_success=True)
    for noun in ("stdout", "stderr"):
        path = Path(attempts[-1][f"{noun}_path"])
        if file_sha256(path) != attempts[-1][f"{noun}_sha256"]:
            raise RefitError(f"refit {noun} log identity changed")
    return {"status": "PASS", "attempts": attempts}
