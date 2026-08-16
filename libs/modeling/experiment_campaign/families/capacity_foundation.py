"""Family 9 capacity-controlled FastAI, Mitra, and TabICL comparison."""

from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

from ..hashing import canonical_sha256, file_sha256, write_canonical_json
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
    preregistration_path = campaign_root / RUN_PATH / "preregistration.json"
    preregistration = json.loads(preregistration_path.read_text(encoding="utf-8"))
    prefix = hashlib.sha256((campaign_root / "registry.jsonl").read_bytes()).hexdigest().upper()
    if prefix != preregistration["registry_prefix_sha256_before"]:
        raise ValueError("family 9 registry prefix changed after preregistration")
    raise NotImplementedError(
        f"family 9 scorer not launched: {source_revision} after {preregistration_commit}"
    )
