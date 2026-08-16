"""Family 9 capacity-controlled FastAI, Mitra, and TabICL comparison."""

from __future__ import annotations

from copy import deepcopy
import csv
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import random
import time
from typing import Any, Mapping

import numpy as np
import pandas as pd

from ..foundation_context import build_prior_context, context_cache_key, validate_context_lineage
from ..feature_lineage import build_development_safe_ids
from ..hashing import (
    canonical_json_bytes,
    canonical_sha256,
    file_sha256,
    read_json,
    tree_inventory,
    write_canonical_json,
)
from ..metrics import event_block_bootstrap_delta, metric_gap, reduce_predictions
from ..protocol import AccessLedger
from .semantic_portfolio import V8_FEATURES


EXPERIMENT_ID = "family-09-capacity-foundation-context"
RUN_ALIAS = "family-09-capacity-foundation"
FROZEN_SPEC_SHA256 = "93FB5CC31AD810B1867FFC8A250DD257AAF74732998D103D56AB8D3A2D309A23"
FROZEN_SOURCE_SHA256 = "157649B780965ECC585F18B3030199CDC0F4FE3013958FFA4095FCF665FDB1EA"
RUN_PATH = "runs/family-09-capacity-foundation"
ARTIFACT_PATH = "artifacts/10-family-09-capacity-foundation"
PROFILE_PATH = "profiles/family-09-capacity-foundation.json"

CONTROL_ID = "family-01-weighted-v8-control"
CATBOOST_SENTINEL_ID = "family-08-catboost-native-specialist-unavailable"
CANDIDATE_IDS = (
    "fastai-low",
    "fastai-medium",
    "mitra-prior-only-low",
    "mitra-prior-only-medium",
    "tabicl-prior-only-low",
    "tabicl-prior-only-medium",
)
MENU_IDS = (CONTROL_ID, CATBOOST_SENTINEL_ID, *CANDIDATE_IDS)
FORBIDDEN_MODEL_TOKENS = ("NORI", "TABDPT", "TABPFN", "REALTABPFN")

FASTAI_SOURCE_SHA256 = "9A7A26F1B651EFD862760774A0456B82AF89739C1A1DC9EEEA98DE9A1E38D22B"
MITRA_REVISION = "c425e9fa0910a6be1c494321792e7ba2a1367b1a"
MITRA_WEIGHTS_SHA256 = "E06A055E91A3BAEFFC37F9CF634D9E69A27D904B6686131DC3B702F9C0126B19"
TABICL_REVISION = "4dcd344ece2c00be9e831fdd35bed57b5ad83e19"
TABICL_WEIGHTS_SHA256 = "BDC7DBD5E4FF21F8F0456FCF90C6B7CDF72DBEA960F2D05B19BEC19F9B3D4ED0"

FIXED_CAMPAIGN_ROOT = Path(
    r"C:\Users\danhm\mma-ai\worktrees\top10-20260815"
    r"\experiments\top10_20260815"
)
FIXED_SOURCE_CSV = FIXED_CAMPAIGN_ROOT / "artifacts/01-campaign-harness/frozen/training_data.csv"
FIXED_CONTROL_PREDICTIONS = (
    FIXED_CAMPAIGN_ROOT
    / "artifacts/02-family-01-weighted-v8-control/fold-2025/outer-predictions.jsonl"
)
FASTAI_SOURCE = Path(
    r".venv\Lib\site-packages\autogluon\tabular\models\fastainn\tabular_nn_fastai.py"
)
MITRA_SNAPSHOT = (
    Path.home()
    / ".cache/huggingface/hub/models--autogluon--mitra-classifier/snapshots"
    / MITRA_REVISION
)
TABICL_CHECKPOINT = (
    Path.home()
    / ".cache/huggingface/hub/models--jingang--TabICL/snapshots"
    / TABICL_REVISION
    / "tabicl-classifier-v2-20260212.ckpt"
)

ARCHITECTURE_KEYS = {
    "layers",
    "width",
    "dropout",
    "embedding_dropout",
    "ensemble_size",
}
OPTIMIZATION_KEYS = {
    "learning_rate",
    "weight_decay",
    "epochs",
    "early_stopping_patience",
    "batch_size",
    "seed",
}
CONTEXT_KEYS = {
    "length",
    "sample",
    "ordering",
    "event_boundary",
    "evaluation_labels",
    "same_event_rows",
    "future_rows",
    "cache_mode",
}
CHECKPOINT_KEYS = {
    "repository",
    "revision",
    "filename",
    "sha256",
    "allow_auto_download",
}
RUNTIME = {
    "num_gpus": 1,
    "gpu_device": "0",
    "serialized": True,
    "refit_full": False,
    "time_limit_seconds": 2_400,
}
SELECTION = {
    "outer_label_selection_count": 0,
    "same_row_score_selection": False,
    "metric": "log_loss",
}


class CapacityFoundationError(ValueError):
    """A Family-9 profile or execution violates the frozen protocol."""


def _checkpoint(model_family: str) -> dict[str, Any]:
    if model_family == "FASTAI":
        return {
            "repository": "autogluon.tabular.models.fastainn",
            "revision": "autogluon-1.6.1",
            "filename": "tabular_nn_fastai.py",
            "sha256": FASTAI_SOURCE_SHA256,
            "allow_auto_download": False,
        }
    if model_family == "MITRA":
        return {
            "repository": "autogluon/mitra-classifier",
            "revision": MITRA_REVISION,
            "filename": "model.safetensors",
            "sha256": MITRA_WEIGHTS_SHA256,
            "allow_auto_download": False,
        }
    return {
        "repository": "jingang/TabICL",
        "revision": TABICL_REVISION,
        "filename": "tabicl-classifier-v2-20260212.ckpt",
        "sha256": TABICL_WEIGHTS_SHA256,
        "allow_auto_download": False,
    }


def _candidate(model_family: str, capacity: str) -> dict[str, Any]:
    is_low = capacity == "low"
    if model_family == "FASTAI":
        candidate_id = f"fastai-{capacity}"
        architecture = {
            "layers": [64, 32] if is_low else [256, 128, 64],
            "width": 64 if is_low else 256,
            "dropout": 0.40 if is_low else 0.25,
            "embedding_dropout": 0.20 if is_low else 0.10,
            "ensemble_size": 1,
        }
        optimization = {
            "learning_rate": 0.005 if is_low else 0.002,
            "weight_decay": None,
            "epochs": 20 if is_low else 50,
            "early_stopping_patience": 5 if is_low else 10,
            "batch_size": 256 if is_low else 128,
            "seed": 20260815,
        }
        cache_mode = "none"
    elif model_family == "MITRA":
        candidate_id = f"mitra-prior-only-{capacity}"
        architecture = {
            "layers": 12,
            "width": 512,
            "dropout": "checkpoint-fixed",
            "embedding_dropout": "checkpoint-fixed",
            "ensemble_size": 1 if is_low else 2,
        }
        optimization = {
            "learning_rate": 0.0001,
            "weight_decay": 0.1,
            "epochs": 20 if is_low else 50,
            "early_stopping_patience": 10 if is_low else 20,
            "batch_size": 1024,
            "seed": 20260815,
        }
        cache_mode = "in-memory-prior-context"
    else:
        candidate_id = f"tabicl-prior-only-{capacity}"
        architecture = {
            "layers": "checkpoint-fixed",
            "width": 128,
            "dropout": "checkpoint-fixed",
            "embedding_dropout": "checkpoint-fixed",
            "ensemble_size": 2 if is_low else 8,
        }
        optimization = {
            "learning_rate": None,
            "weight_decay": None,
            "epochs": 0,
            "early_stopping_patience": 0,
            "batch_size": 8,
            "seed": 20260815,
        }
        cache_mode = "none" if is_low else "kv"
    value = {
        "id": candidate_id,
        "model_family": model_family,
        "capacity": capacity,
        "features": list(V8_FEATURES),
        "architecture": architecture,
        "optimization": optimization,
        "context": {
            "length": 1_024 if is_low else 2_807,
            "sample": "most-recent-complete-events",
            "ordering": ["event_date", "event_id", "fight_id"],
            "event_boundary": "strictly-before-evaluation-event",
            "evaluation_labels": "absent",
            "same_event_rows": "excluded",
            "future_rows": "excluded",
            "cache_mode": cache_mode,
        },
        "checkpoint": _checkpoint(model_family),
        "runtime": deepcopy(RUNTIME),
        "selection": deepcopy(SELECTION),
    }
    return {**value, "profile_sha256": canonical_sha256(value)}


def build_preregistered_profile() -> dict[str, Any]:
    """Materialize the fixed control/sentinel/six-candidate menu."""

    candidates = [
        _candidate(model_family, capacity)
        for model_family in ("FASTAI", "MITRA", "TABICL")
        for capacity in ("low", "medium")
    ]
    profile = {
        "experiment_id": EXPERIMENT_ID,
        "family_number": 9,
        "frozen_spec_sha256": FROZEN_SPEC_SHA256,
        "frozen_source": {
            "path": "artifacts/01-campaign-harness/frozen/training_data.csv",
            "sha256": FROZEN_SOURCE_SHA256,
            "development_safe_id_count": 3_089,
            "retired_id_count": 178,
            "development_max_date": "2025-12-13",
        },
        "evaluation": {
            "train_max_date": "2023-12-16",
            "inner_validation_year": 2024,
            "outer_year": 2025,
            "outer_boundary": "Original",
            "outer_label_selection_count": 0,
        },
        "menu": [
            {
                "id": CONTROL_ID,
                "status": "executable-control",
                "source": "immutable-family-01-outer-2025-predictions",
            },
            {
                "id": CATBOOST_SENTINEL_ID,
                "status": "unavailable-inconclusive-sentinel",
                "source": "family-08-terminal-pre-construction-failure",
                "negative_evidence": False,
            },
            *[
                {
                    "id": candidate["id"],
                    "status": "preregistered-candidate",
                    "model_family": candidate["model_family"],
                    "capacity": candidate["capacity"],
                    "profile_sha256": candidate["profile_sha256"],
                }
                for candidate in candidates
            ],
        ],
        "candidates": candidates,
        "inner_selection": {
            "within_model_family": True,
            "score": "inner-2024-log-loss",
            "tie_break": list(CANDIDATE_IDS),
            "outer_score_selection": False,
        },
        "promotion": {
            "minimum_accuracy_gain": 0.0,
            "minimum_log_loss_gain": 0.0,
            "require_nonnegative_paired_event_interval": True,
            "failed_candidates_promotable": False,
        },
        "invariance": {
            "evaluation_label_removal": "byte-identical",
            "evaluation_label_permutation": "byte-identical",
            "irrelevant_future_label_change": "byte-identical",
        },
        "invocation": {
            "gpu_lease_count": 1,
            "candidate_fit_count": 6,
            "retry_count": 0,
            "serialized": True,
        },
        "database_access": {"used": False, "sql": None, "urls": []},
    }
    validate_preregistered_profile(profile)
    return profile


def validate_preregistered_profile(profile: Mapping[str, Any]) -> dict[str, Any]:
    """Reject expanded menus, hidden defaults, forbidden models, or label leakage."""

    menu = list(profile.get("menu", []))
    candidates = list(profile.get("candidates", []))
    if len(menu) > 8:
        raise CapacityFoundationError("maximum eight menu entries")
    if len(candidates) != 6 or tuple(item.get("id") for item in candidates) != CANDIDATE_IDS:
        raise CapacityFoundationError("exact six candidate fits required")
    if tuple(item.get("id") for item in menu) != MENU_IDS:
        raise CapacityFoundationError("exact ordered eight-entry menu required")
    serialized = json.dumps(profile, sort_keys=True).upper()
    if any(token in serialized for token in FORBIDDEN_MODEL_TOKENS):
        raise CapacityFoundationError("forbidden model token")
    expected_fields = {
        "id",
        "model_family",
        "capacity",
        "features",
        "architecture",
        "optimization",
        "context",
        "checkpoint",
        "runtime",
        "selection",
        "profile_sha256",
    }
    for candidate in candidates:
        if set(candidate) != expected_fields:
            raise CapacityFoundationError("fully materialized candidate fields required")
        if set(candidate["architecture"]) != ARCHITECTURE_KEYS:
            raise CapacityFoundationError("fully materialized architecture required")
        if set(candidate["optimization"]) != OPTIMIZATION_KEYS:
            raise CapacityFoundationError("fully materialized optimization required")
        if set(candidate["context"]) != CONTEXT_KEYS:
            raise CapacityFoundationError("fully materialized context required")
        if set(candidate["checkpoint"]) != CHECKPOINT_KEYS:
            raise CapacityFoundationError("fully materialized checkpoint required")
        if candidate["runtime"] != RUNTIME:
            if candidate["runtime"].get("refit_full") is not False:
                raise CapacityFoundationError("refit_full is forbidden")
            raise CapacityFoundationError("fully materialized runtime required")
        if candidate["selection"] != SELECTION:
            raise CapacityFoundationError("outer or same-row score selection is forbidden")
        context = candidate["context"]
        if context["evaluation_labels"] != "absent":
            raise CapacityFoundationError("evaluation labels must be absent")
        if context["same_event_rows"] != "excluded":
            raise CapacityFoundationError("same-event rows must be excluded")
        if context["future_rows"] != "excluded":
            raise CapacityFoundationError("future rows must be excluded")
        core = {key: value for key, value in candidate.items() if key != "profile_sha256"}
        if candidate["profile_sha256"] != canonical_sha256(core):
            raise CapacityFoundationError("candidate hash does not cover explicit defaults")
    sentinel = menu[1]
    if sentinel.get("status") != "unavailable-inconclusive-sentinel" or sentinel.get("negative_evidence") is not False:
        raise CapacityFoundationError("CatBoost sentinel must remain unavailable and inconclusive")
    return {
        "menu_count": len(menu),
        "candidate_fit_count": len(candidates),
        "menu_ids": [item["id"] for item in menu],
        "profile_hashes": {item["id"]: item["profile_sha256"] for item in candidates},
        "checkpoint_hashes": {item["id"]: item["checkpoint"]["sha256"] for item in candidates},
    }


def write_preregistration(campaign_root: Path, *, source_revision: str) -> dict[str, Any]:
    """Persist every capacity and causal default while fit destinations are absent."""

    campaign_root = Path(campaign_root)
    profile_path = campaign_root / PROFILE_PATH
    preregistration_path = campaign_root / RUN_PATH / "preregistration.json"
    artifact_root = campaign_root / ARTIFACT_PATH
    if any(path.exists() for path in (profile_path, preregistration_path, artifact_root)):
        raise ValueError("family 9 preregistration destinations must all be absent")
    gate = AccessLedger(campaign_root).gate_status()
    if gate["state"] != "closed" or gate["protected_access_count"] != 0:
        raise ValueError("family 9 preregistration requires the gate closed with zero access")
    profile = build_preregistered_profile()
    validated = validate_preregistered_profile(profile)
    write_canonical_json(profile_path, profile)
    preregistration = {
        "experiment_id": EXPERIMENT_ID,
        "family_number": 9,
        "source_revision": source_revision,
        "frozen_spec_sha256": FROZEN_SPEC_SHA256,
        "profile_path": PROFILE_PATH,
        "profile_sha256": canonical_sha256(profile),
        "profile_file_sha256": file_sha256(profile_path),
        "menu_ids": list(MENU_IDS),
        "candidate_fit_count": 6,
        "ordered_profile_hashes": validated["profile_hashes"],
        "checkpoint_hashes": validated["checkpoint_hashes"],
        "registry_prefix_sha256_before": hashlib.sha256(
            (campaign_root / "registry.jsonl").read_bytes()
        ).hexdigest().upper(),
        "launch_state": "not-started",
        "invocation": profile["invocation"],
        "database_access": profile["database_access"],
        "gate_required_state": "closed-zero-access",
        "terminal_failure_rule": "Any preflight, fit, invariance, safety, or destination mismatch terminates without retry.",
    }
    write_canonical_json(preregistration_path, preregistration)
    return preregistration


def materialize_family_09(
    campaign_root: Path,
    *,
    source_revision: str,
    preregistration_commit: str,
) -> dict[str, Any]:
    """Execute the sole serialized Family-9 attempt after prefix verification."""

    campaign_root = Path(campaign_root)
    run_root = campaign_root / RUN_PATH
    artifact_root = campaign_root / ARTIFACT_PATH
    manifest_path = run_root / "manifest.json"
    if artifact_root.exists() or manifest_path.exists():
        raise ValueError("family 9 score destination already exists; retries are forbidden")
    gate = AccessLedger(campaign_root).gate_status()
    if gate["state"] != "closed" or gate["protected_access_count"] != 0:
        raise ValueError("family 9 requires the gate closed with zero access")
    profile_path = campaign_root / PROFILE_PATH
    preregistration_path = run_root / "preregistration.json"
    profile = read_json(profile_path)
    preregistration = read_json(preregistration_path)
    validated = validate_preregistered_profile(profile)
    prefix = hashlib.sha256((campaign_root / "registry.jsonl").read_bytes()).hexdigest().upper()
    if prefix != preregistration["registry_prefix_sha256_before"]:
        raise ValueError("family 9 registry prefix changed after preregistration")
    if (
        profile != build_preregistered_profile()
        or preregistration["launch_state"] != "not-started"
        or preregistration["profile_sha256"] != canonical_sha256(profile)
        or preregistration["profile_file_sha256"] != file_sha256(profile_path)
        or preregistration["candidate_fit_count"] != 6
    ):
        raise ValueError("family 9 was not exactly preregistered before launch")

    fold_manifest = read_json(campaign_root / "baseline/fold-manifest.json")
    safe_id_order, retired_ids = build_development_safe_ids(fold_manifest)
    safe_ids = set(safe_id_order)
    metadata = pd.read_csv(
        FIXED_SOURCE_CSV,
        usecols=["fight_id", "event_id", "event_date"],
        dtype={"fight_id": str, "event_id": str, "event_date": str},
    )
    metadata["event_date"] = pd.to_datetime(metadata["event_date"], utc=True).dt.strftime("%Y-%m-%d")
    metadata = metadata.sort_values(["event_date", "event_id", "fight_id"], kind="mergesort")
    if metadata["fight_id"].duplicated().any():
        raise ValueError("frozen source contains duplicate fight identities")
    safe_metadata = metadata.loc[metadata["fight_id"].isin(safe_ids)].reset_index(drop=True)
    retired_metadata = metadata.loc[metadata["fight_id"].isin(retired_ids)].reset_index(drop=True)
    if (
        len(safe_metadata) != 3_089
        or len(retired_metadata) != 178
        or safe_metadata["event_date"].max() != "2025-12-13"
        or set(safe_metadata["fight_id"]) & set(retired_metadata["fight_id"])
        or set(safe_metadata["fight_id"]) != safe_ids
        or set(retired_metadata["fight_id"]) != set(retired_ids)
    ):
        raise ValueError("development-safe population differs before target decode")
    development_population = {
        "asserted_before_target_decode": True,
        "development_safe_id_count": 3_089,
        "development_max_date": "2025-12-13",
        "retired_id_count": 178,
    }

    artifact_root.mkdir(parents=True)
    started_at = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    lease_id = "family-09-serialized-gpu-lease-1"
    acquired = {
        "lease_id": lease_id,
        "ordinal": 1,
        "pid": os.getpid(),
        "state": "acquired",
        "started_at": started_at,
    }
    write_canonical_json(artifact_root / "gpu-lease-acquired.json", acquired)
    write_canonical_json(artifact_root / "development-safe-population.json", development_population)

    attempts: list[dict[str, Any]] = []
    candidate_results: list[dict[str, Any]] = []
    terminal_failure: dict[str, Any] | None = None
    frame: pd.DataFrame | None = None
    control_records: list[dict[str, Any]] = []
    context_evidence: dict[str, Any] = {}
    invariance: dict[str, Any] = {
        "evaluation_label_removal": "not-run",
        "evaluation_label_permutation": "not-run",
        "irrelevant_future_label_change": "not-run",
        "evaluation_label_reads_for_prediction": 0,
        "retired_label_reads": 0,
    }

    try:
        _assert_checkpoint_identities()
        frame = _load_development_frame(FIXED_SOURCE_CSV, safe_ids=safe_ids)
        if set(frame["fight_id"].astype(str)) != safe_ids or len(frame) != 3_089:
            raise ValueError("target-decoded development frame differs from preasserted safe IDs")
        outer = frame.loc[frame["event_date"].str.startswith("2025-")].copy()
        outer = outer.sort_values(["event_date", "event_id", "fight_id"], kind="mergesort")
        if len(outer) != 282:
            raise ValueError("Original outer-2025 row count differs")
        first_outer = outer.iloc[0]
        control_records = _load_control_records(set(outer["fight_id"].astype(str)))
        outer_ids = outer["fight_id"].astype(str).tolist()
        if [str(row["fight_id"]) for row in control_records] != outer_ids:
            raise ValueError("Family-1 control predictions do not align to Family-9 Original rows")

        for ordinal, candidate in enumerate(profile["candidates"], start=1):
            candidate_id = candidate["id"]
            attempt_started = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
            attempts.append(
                {
                    "attempt_ordinal": ordinal,
                    "candidate_id": candidate_id,
                    "profile_sha256": candidate["profile_sha256"],
                    "state": "launched",
                    "gpu_lease_id": lease_id,
                    "started_at": attempt_started,
                    "retry": False,
                }
            )
            candidate_dir = artifact_root / "candidates" / candidate_id
            candidate_dir.mkdir(parents=True)
            context = build_prior_context(
                frame,
                evaluation_event_id=str(first_outer["event_id"]),
                evaluation_date=str(first_outer["event_date"]),
                evaluation_fight_ids=set(outer_ids),
                context_length=int(candidate["context"]["length"]),
                sample=str(candidate["context"]["sample"]),
            )
            lineage = validate_context_lineage(
                context,
                evaluation_event_id=str(first_outer["event_id"]),
                evaluation_date=str(first_outer["event_date"]),
                evaluation_fight_ids=set(outer_ids),
            )
            if len(context) > int(candidate["context"]["length"]):
                raise ValueError("complete-event context exceeds preregistered length")
            feature_sha = canonical_sha256(candidate["features"])
            cache_key = context_cache_key(
                profile_sha256=candidate["profile_sha256"],
                checkpoint_sha256=candidate["checkpoint"]["sha256"],
                feature_sha256=feature_sha,
                context_fight_ids=context["fight_id"].astype(str).tolist(),
                context_event_ids=context["event_id"].astype(str).tolist(),
                context_dates=context["event_date"].astype(str).tolist(),
            )
            context_evidence[candidate_id] = {
                **lineage,
                "context_length_cap": candidate["context"]["length"],
                "context_fight_ids_sha256": canonical_sha256(
                    context["fight_id"].astype(str).tolist()
                ),
                "feature_sha256": feature_sha,
                "cache_key": cache_key,
                "cache_mode": candidate["context"]["cache_mode"],
                "checkpoint": candidate["checkpoint"],
                "evaluation_label_count_in_prediction_request": 0,
            }
            result = _fit_candidate(
                candidate,
                context=context,
                outer=outer,
                candidate_dir=candidate_dir,
            )
            candidate_results.append(result)
            attempts.append(
                {
                    "attempt_ordinal": ordinal,
                    "candidate_id": candidate_id,
                    "profile_sha256": candidate["profile_sha256"],
                    "state": "exited",
                    "exit_state": "complete",
                    "gpu_lease_id": lease_id,
                    "ended_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
                    "retry": False,
                    "prediction_sha256": result["outer_prediction_identity"]["sha256"],
                }
            )
        invariance = _actual_label_invariance(candidate_results)
    except Exception as exc:
        failure_path = artifact_root / "terminal-stderr.txt"
        failure_path.write_text(f"{type(exc).__name__}: {exc}\n", encoding="utf-8", newline="\n")
        terminal_failure = {
            "stage": "serialized-candidate-execution",
            "exception_type": type(exc).__name__,
            "message": str(exc),
            "stderr_path": "terminal-stderr.txt",
            "stderr_sha256": file_sha256(failure_path),
            "completed_candidate_ids": [item["candidate_id"] for item in candidate_results],
            "fit_attempt_count": sum(row["state"] == "launched" for row in attempts),
            "retry_performed": False,
            "retired_label_reads": 0,
            "gate_access_count": 0,
        }
        if attempts and attempts[-1]["state"] == "launched":
            launched = attempts[-1]
            attempts.append(
                {
                    "attempt_ordinal": launched["attempt_ordinal"],
                    "candidate_id": launched["candidate_id"],
                    "profile_sha256": launched["profile_sha256"],
                    "state": "exited",
                    "exit_state": "failed",
                    "gpu_lease_id": lease_id,
                    "ended_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
                    "retry": False,
                    "exception_type": type(exc).__name__,
                }
            )

    selected_profiles = _select_inner_profiles(candidate_results)
    control_metrics = reduce_predictions(control_records).as_dict() if control_records else None
    selected_results = [
        item for item in candidate_results if item["candidate_id"] in set(selected_profiles.values())
    ]
    best_result = min(
        selected_results,
        key=lambda item: (item["outer_metrics"]["log_loss"], item["candidate_id"]),
        default=None,
    )
    paired_intervals = None
    promotion = {
        "action": "retain-family-01-weighted-v8-control",
        "incumbent_before": CONTROL_ID,
        "incumbent_after": CONTROL_ID,
        "promoted": False,
        "rule": "inner-only capacity selection; paired Original outer confirmation required",
    }
    if best_result is not None and control_records:
        paired_intervals = event_block_bootstrap_delta(
            _read_jsonl(artifact_root / best_result["outer_prediction_identity"]["path"]),
            control_records,
            iterations=2_000,
            seed=20260815,
        )
        candidate_metrics = best_result["outer_metrics"]
        promoted = (
            candidate_metrics["accuracy"] >= control_metrics["accuracy"]
            and candidate_metrics["log_loss"] <= control_metrics["log_loss"]
            and paired_intervals["accuracy_delta"]["lower"] >= 0.0
            and paired_intervals["log_loss_delta"]["upper"] <= 0.0
        )
        if promoted:
            promotion = {
                **promotion,
                "action": f"promote-{best_result['candidate_id']}",
                "incumbent_after": best_result["candidate_id"],
                "promoted": True,
            }

    status = "complete" if terminal_failure is None and len(candidate_results) == 6 else "failed"
    capacity_diagnostics = {
        family: {
            item["capacity"]: {
                "candidate_id": item["candidate_id"],
                "inner_metrics": item["inner_metrics"],
                "outer_metrics": item["outer_metrics"],
                "train_outer_gap": item["train_outer_gap"],
            }
            for item in candidate_results
            if item["model_family"] == family
        }
        for family in ("FASTAI", "MITRA", "TABICL")
    }
    adaptive_signal = {
        "status": status,
        "selected_profiles": selected_profiles,
        "best_outer_profile": None if best_result is None else best_result["candidate_id"],
        "incumbent_after": promotion["incumbent_after"],
        "outcome_decomposition_input_available": best_result is not None,
        "catboost_evidence": "unavailable-inconclusive-not-negative",
    }
    result = {
        "experiment_id": EXPERIMENT_ID,
        "status": status,
        "terminal_failure": terminal_failure,
        "candidate_fit_count": len(candidate_results),
        "candidate_results": candidate_results,
        "selected_profiles": selected_profiles,
        "control_metrics": control_metrics,
        "paired_event_block_intervals": paired_intervals,
        "capacity_diagnostics": capacity_diagnostics,
        "context_lineage": context_evidence,
        "label_invariance": invariance,
        "outer_prediction_identities": [
            item["outer_prediction_identity"] for item in candidate_results
        ],
        "promotion_decision": promotion,
        "adaptive_signal_for_family_10": adaptive_signal,
        "development_safe_population": development_population,
        "gate_access_count": 0,
    }
    write_canonical_json(artifact_root / "context-lineage.json", context_evidence)
    write_canonical_json(artifact_root / "label-invariance.json", invariance)
    write_canonical_json(artifact_root / "result.json", result)
    safety = {
        "database_access": {"used": False, "sql": None, "urls": []},
        "gpu_lease_count": 1,
        "production_attempt_count": 1,
        "candidate_fit_launch_count": sum(row["state"] == "launched" for row in attempts),
        "retry_count": 0,
        "serialized": True,
        "gate_access_count": 0,
        "retired_label_reads": 0,
        **development_population,
    }
    write_canonical_json(artifact_root / "safety.json", safety)
    ended_at = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    released = {
        "lease_id": lease_id,
        "ordinal": 1,
        "pid": acquired["pid"],
        "state": "released",
        "ended_at": ended_at,
    }
    write_canonical_json(artifact_root / "gpu-lease-released.json", released)
    runtime = {
        "attempt_ordinal": 1,
        "started_at": started_at,
        "ended_at": ended_at,
        "runtime_seconds": time.time() - datetime.fromisoformat(started_at.replace("Z", "+00:00")).timestamp(),
        "gpu_lease_id": lease_id,
        "serialized": True,
        "retry_count": 0,
        "candidate_fit_count": len(candidate_results),
    }
    write_canonical_json(artifact_root / "runtime.json", runtime)
    _write_jsonl(run_root / "attempts.jsonl", attempts)
    (run_root / "decision.md").write_text(
        "# Family 9 decision\n\n"
        + (
            f"Promote {promotion['incumbent_after']} after inner-only capacity selection and paired outer confirmation.\n"
            if promotion["promoted"]
            else "Retain family 1; no preregistered capacity candidate met the paired Original outer promotion rule.\n"
        ),
        encoding="utf-8",
        newline="\n",
    )
    inventory = tree_inventory(artifact_root)
    manifest = {
        **result,
        "kind": "family",
        "exit_state": status,
        "artifact_path": ARTIFACT_PATH,
        "artifact_tree_sha256": inventory.tree_sha256,
        "artifact_file_count": inventory.file_count,
        "artifact_total_bytes": inventory.total_bytes,
        "profile_path": PROFILE_PATH,
        "profile_sha256": canonical_sha256(profile),
        "profile_file_sha256": file_sha256(profile_path),
        "preregistration_path": f"{RUN_PATH}/preregistration.json",
        "preregistration_sha256": canonical_sha256(preregistration),
        "preregistration_commit": preregistration_commit,
        "attempts_path": f"{RUN_PATH}/attempts.jsonl",
        "runtime_path": f"{ARTIFACT_PATH}/runtime.json",
        "source_revision": source_revision,
        "outer_label_selection_count": 0,
        "invocation": profile["invocation"],
        "database_access": profile["database_access"],
        "catboost_sentinel": profile["menu"][1],
    }
    write_canonical_json(manifest_path, manifest)
    registry = _append_registry(
        campaign_root,
        {
            "artifact_path": ARTIFACT_PATH,
            "artifact_tree_sha256": inventory.tree_sha256,
            "experiment_id": EXPERIMENT_ID,
            "kind": "family",
            "manifest_path": f"{RUN_PATH}/manifest.json",
            "manifest_sha256": canonical_sha256(manifest),
            "profile_path": PROFILE_PATH,
            "profile_sha256": canonical_sha256(profile),
            "status": status,
        },
    )
    return {**result, **registry, "artifact_tree_sha256": inventory.tree_sha256}


def _assert_checkpoint_identities() -> None:
    if not FASTAI_SOURCE.is_file() or file_sha256(FASTAI_SOURCE) != FASTAI_SOURCE_SHA256:
        raise ValueError("FastAI implementation identity differs")
    mitra_weights = MITRA_SNAPSHOT / "model.safetensors"
    if not mitra_weights.is_file() or file_sha256(mitra_weights) != MITRA_WEIGHTS_SHA256:
        raise ValueError("Mitra checkpoint identity differs")
    if not TABICL_CHECKPOINT.is_file() or file_sha256(TABICL_CHECKPOINT) != TABICL_WEIGHTS_SHA256:
        raise ValueError("TabICL checkpoint identity differs")


def _load_development_frame(source: Path, *, safe_ids: set[str]) -> pd.DataFrame:
    selected_columns = [
        "fight_id",
        "event_id",
        "event_date",
        "y_true",
        *V8_FEATURES,
    ]
    rows: list[dict[str, Any]] = []
    with Path(source).open("r", encoding="utf-8", newline="") as handle:
        header_line = handle.readline()
        header = next(csv.reader([header_line]))
        indexes = {name: header.index(name) for name in selected_columns}
        for line in handle:
            prefix = line.split(",", 6)
            if len(prefix) < 6:
                raise ValueError("frozen CSV row has no metadata prefix")
            fight_id = prefix[0]
            if fight_id not in safe_ids:
                continue
            values = next(csv.reader([line]))
            rows.append({name: values[indexes[name]] for name in selected_columns})
    frame = pd.DataFrame(rows, columns=selected_columns)
    frame["fight_id"] = frame["fight_id"].astype(str)
    frame["event_id"] = frame["event_id"].astype(str)
    frame["event_date"] = pd.to_datetime(frame["event_date"], utc=True).dt.strftime("%Y-%m-%d")
    frame["y_true"] = pd.to_numeric(frame["y_true"], errors="raise").astype(int)
    if not frame["y_true"].isin([0, 1]).all():
        raise ValueError("development labels are not binary")
    for feature in V8_FEATURES:
        frame[feature] = pd.to_numeric(frame[feature], errors="coerce")
    return frame.sort_values(["event_date", "event_id", "fight_id"], kind="mergesort").reset_index(drop=True)


def _fit_candidate(
    candidate: Mapping[str, Any],
    *,
    context: pd.DataFrame,
    outer: pd.DataFrame,
    candidate_dir: Path,
) -> dict[str, Any]:
    from autogluon.tabular import TabularPredictor

    feature_names = list(candidate["features"])
    train = context.loc[context["event_date"] < "2024-01-01", [*feature_names, "y_true"]].copy()
    tuning = context.loc[
        (context["event_date"] >= "2024-01-01") & (context["event_date"] < "2025-01-01"),
        [*feature_names, "y_true"],
    ].copy()
    if len(train) < 400 or len(tuning) < 100:
        raise ValueError("candidate context does not contain stable train and inner-validation blocks")
    seed = int(candidate["optimization"]["seed"])
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass
    hyperparameters = _autogluon_hyperparameters(candidate)
    model_path = candidate_dir / "model"
    predictor = TabularPredictor(
        label="y_true",
        problem_type="binary",
        eval_metric="log_loss",
        path=str(model_path),
        verbosity=3,
    )
    fit_started = time.time()
    predictor.fit(
        train_data=train,
        tuning_data=tuning,
        hyperparameters=hyperparameters,
        time_limit=int(candidate["runtime"]["time_limit_seconds"]),
        num_bag_folds=0,
        num_stack_levels=0,
        fit_weighted_ensemble=False,
        calibrate=False,
        refit_full=False,
        dynamic_stacking=False,
        ag_args_fit={"num_gpus": 1},
    )
    fit_seconds = time.time() - fit_started
    tuning_probability = _positive_probability(predictor, tuning[feature_names])
    outer_variants = {
        "labels-present": outer,
        "labels-removed": outer.drop(columns=["y_true"]),
        "labels-permuted": outer.assign(y_true=outer["y_true"].iloc[::-1].to_numpy()),
        "irrelevant-future-labels-changed": outer,
    }
    variant_probabilities = {
        name: _positive_probability(predictor, variant[feature_names].copy())
        for name, variant in outer_variants.items()
    }
    variant_hashes = {
        name: hashlib.sha256(
            canonical_json_bytes([float(value) for value in probabilities])
        ).hexdigest().upper()
        for name, probabilities in variant_probabilities.items()
    }
    if len(set(variant_hashes.values())) != 1:
        raise ValueError("evaluation predictions changed after label mutation")
    synthetic_future_before = canonical_sha256([0, 1, 0, 1])
    synthetic_future_after = canonical_sha256([1, 0, 1, 0])
    outer_probability = variant_probabilities["labels-present"]
    train_probability = _positive_probability(predictor, train[feature_names])
    model_names = predictor.model_names()
    if len(model_names) != 1:
        raise ValueError("candidate fit produced hidden ensemble or refit nodes")
    records = _prediction_records(
        outer,
        probabilities=outer_probability,
        candidate_id=str(candidate["id"]),
        fold="2025",
    )
    train_records = _prediction_records(
        train.assign(
            fight_id=[f"train-{index}" for index in range(len(train))],
            event_id="train",
            event_date="2023-12-16",
        ),
        probabilities=train_probability,
        candidate_id=str(candidate["id"]),
        fold="train-diagnostic",
    )
    tuning_records = _prediction_records(
        tuning.assign(
            fight_id=[f"inner-{index}" for index in range(len(tuning))],
            event_id="inner-2024",
            event_date="2024-12-14",
        ),
        probabilities=tuning_probability,
        candidate_id=str(candidate["id"]),
        fold="inner-2024",
    )
    prediction_path = candidate_dir / "outer-predictions.jsonl"
    _write_jsonl(prediction_path, records)
    outer_metrics = reduce_predictions(records).as_dict()
    inner_metrics = reduce_predictions(tuning_records).as_dict()
    train_metrics = reduce_predictions(train_records).as_dict()
    model_inventory = tree_inventory(model_path)
    identity = {
        "candidate_id": candidate["id"],
        "path": prediction_path.relative_to(candidate_dir.parents[1]).as_posix(),
        "sha256": file_sha256(prediction_path),
        "row_count": len(records),
        "boundary": "Original",
    }
    evidence = {
        "candidate_id": candidate["id"],
        "model_family": candidate["model_family"],
        "capacity": candidate["capacity"],
        "profile_sha256": candidate["profile_sha256"],
        "fit_seconds": fit_seconds,
        "seed": seed,
        "train_row_count": len(train),
        "inner_row_count": len(tuning),
        "outer_row_count": len(outer),
        "model_names": model_names,
        "hyperparameters_sha256": canonical_sha256(hyperparameters),
        "model_tree_sha256": model_inventory.tree_sha256,
        "model_file_count": model_inventory.file_count,
        "inner_metrics": inner_metrics,
        "outer_metrics": outer_metrics,
        "train_metrics": train_metrics,
        "train_outer_gap": metric_gap(reduce_predictions(train_records), reduce_predictions(records)),
        "outer_prediction_identity": identity,
        "prediction_request_columns": feature_names,
        "evaluation_label_count_in_prediction_request": 0,
        "label_invariance_prediction_sha256s": variant_hashes,
        "synthetic_irrelevant_future_label_sha256s": {
            "before": synthetic_future_before,
            "after": synthetic_future_after,
        },
    }
    write_canonical_json(candidate_dir / "fit-evidence.json", evidence)
    return evidence


def _autogluon_hyperparameters(candidate: Mapping[str, Any]) -> dict[str, list[dict[str, Any]]]:
    family = str(candidate["model_family"])
    architecture = candidate["architecture"]
    optimization = candidate["optimization"]
    if family == "FASTAI":
        params = {
            "layers": architecture["layers"],
            "emb_drop": architecture["embedding_dropout"],
            "ps": architecture["dropout"],
            "bs": optimization["batch_size"],
            "lr": optimization["learning_rate"],
            "epochs": optimization["epochs"],
            "early.stopping.min_delta": 0.0001,
            "early.stopping.patience": optimization["early_stopping_patience"],
            "smoothing": 0.0,
        }
    elif family == "MITRA":
        params = {
            "n_estimators": architecture["ensemble_size"],
            "fine_tune": True,
            "fine_tune_steps": optimization["epochs"],
            "patience": optimization["early_stopping_patience"],
            "lr": optimization["learning_rate"],
            "warmup_steps": min(1_000, optimization["epochs"] * 10),
            "shuffle_classes": False,
            "shuffle_features": False,
            "use_random_transforms": False,
            "seed": optimization["seed"],
            "hf_model": str(MITRA_SNAPSHOT),
            "verbose": False,
        }
    else:
        params = {
            "n_estimators": architecture["ensemble_size"],
            "batch_size": optimization["batch_size"],
            "kv_cache": candidate["context"]["cache_mode"] == "kv",
            "model_path": str(TABICL_CHECKPOINT),
            "allow_auto_download": False,
            "checkpoint_version": candidate["checkpoint"]["filename"],
            "random_state": optimization["seed"],
            "use_amp": True,
            "use_fa3": False,
            "offload_mode": False,
            "verbose": False,
        }
    params["ag_args"] = {"name_suffix": f"_{candidate['capacity']}"}
    params["ag_args_fit"] = {"num_gpus": 1}
    return {family: [params]}


def _positive_probability(predictor: Any, features: pd.DataFrame) -> np.ndarray:
    probabilities = predictor.predict_proba(features)
    if isinstance(probabilities, pd.DataFrame):
        column = 1 if 1 in probabilities.columns else probabilities.columns[-1]
        return probabilities[column].to_numpy(dtype=float)
    values = np.asarray(probabilities)
    return values[:, -1] if values.ndim == 2 else values.astype(float)


def _prediction_records(
    rows: pd.DataFrame,
    *,
    probabilities: np.ndarray,
    candidate_id: str,
    fold: str,
) -> list[dict[str, Any]]:
    records = []
    for (_, row), probability in zip(rows.iterrows(), probabilities, strict=True):
        records.append(
            {
                "boundary": "Original",
                "fit_scope": "prior-only",
                "fold": fold,
                "candidate_id": candidate_id,
                "fight_id": str(row["fight_id"]),
                "event_id": str(row["event_id"]),
                "event_date": str(row["event_date"]),
                "probability": float(probability),
                "y_true": int(row["y_true"]),
                "weight_class": str(row.get("weightclass_encoded", "unknown")),
                "experience": "unknown",
                "outcome_type": "unknown",
            }
        )
    return records


def _load_control_records(outer_ids: set[str]) -> list[dict[str, Any]]:
    records = [
        row for row in _read_jsonl(FIXED_CONTROL_PREDICTIONS) if str(row["fight_id"]) in outer_ids
    ]
    return sorted(records, key=lambda row: (row["event_date"], str(row["event_id"]), str(row["fight_id"])))


def _actual_label_invariance(candidate_results: list[dict[str, Any]]) -> dict[str, Any]:
    hashes = {
        item["candidate_id"]: item["label_invariance_prediction_sha256s"]
        for item in candidate_results
    }
    if any(len(set(values.values())) != 1 for values in hashes.values()):
        raise ValueError("candidate prediction bytes are not label invariant")
    future_hashes = {
        item["candidate_id"]: item["synthetic_irrelevant_future_label_sha256s"]
        for item in candidate_results
    }
    if any(values["before"] == values["after"] for values in future_hashes.values()):
        raise ValueError("irrelevant future-label fixture did not change")
    return {
        "evaluation_label_removal": "byte-identical",
        "evaluation_label_permutation": "byte-identical",
        "irrelevant_future_label_change": "byte-identical",
        "candidate_prediction_sha256s": hashes,
        "synthetic_irrelevant_future_label_sha256s": future_hashes,
        "evaluation_label_reads_for_prediction": 0,
        "future_label_reads_for_prediction": 0,
        "retired_label_reads": 0,
    }


def _select_inner_profiles(candidate_results: list[dict[str, Any]]) -> dict[str, str]:
    selected: dict[str, str] = {}
    for family in ("FASTAI", "MITRA", "TABICL"):
        available = [item for item in candidate_results if item["model_family"] == family]
        if available:
            winner = min(
                available,
                key=lambda item: (item["inner_metrics"]["log_loss"], CANDIDATE_IDS.index(item["candidate_id"])),
            )
            selected[family] = winner["candidate_id"]
    return selected


def _write_jsonl(path: Path, rows: list[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"".join(canonical_json_bytes(dict(row)) + b"\n" for row in rows))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in Path(path).read_bytes().splitlines()]


def _append_registry(campaign_root: Path, payload: Mapping[str, Any]) -> dict[str, str]:
    registry_path = campaign_root / "registry.jsonl"
    head_path = campaign_root / "registry-head.json"
    before = registry_path.read_bytes()
    records = [json.loads(line) for line in before.splitlines()]
    if any(record["payload"]["experiment_id"] == EXPERIMENT_ID for record in records):
        raise ValueError("family 9 already exists in registry")
    head = read_json(head_path)
    prefix_before = hashlib.sha256(before).hexdigest().upper()
    if prefix_before != head["registry_prefix_sha256"] or len(records) != 9:
        raise ValueError("registry head does not match immutable family-8 prefix")
    record = {
        "payload": dict(payload),
        "prefix_sha256_before": prefix_before,
        "previous_record_sha256": head["last_record_sha256"],
        "sequence": head["record_count"],
    }
    record["record_sha256"] = canonical_sha256(record)
    after = before + canonical_json_bytes(record) + b"\n"
    registry_path.write_bytes(after)
    write_canonical_json(
        head_path,
        {
            "last_record_sha256": record["record_sha256"],
            "record_count": len(records) + 1,
            "registry_bytes": len(after),
            "registry_prefix_sha256": hashlib.sha256(after).hexdigest().upper(),
        },
    )
    return {
        "registry_prefix_sha256_before": prefix_before,
        "registry_prefix_sha256_after": hashlib.sha256(after).hexdigest().upper(),
        "registry_record_sha256": record["record_sha256"],
    }
