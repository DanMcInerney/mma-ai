"""Family 8 CatBoost-first representation and capacity preregistration."""

from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

from ..feature_lineage import build_development_safe_ids
from ..hashing import (
    canonical_json_bytes,
    canonical_sha256,
    file_sha256,
    read_json,
    tree_inventory,
    write_canonical_json,
)
from ..protocol import AccessLedger
from .semantic_portfolio import V8_FEATURES


EXPERIMENT_ID = "family-08-catboost-native-specialist"
RUN_ALIAS = "family-08-catboost-specialist"
FROZEN_SPEC_SHA256 = "93FB5CC31AD810B1867FFC8A250DD257AAF74732998D103D56AB8D3A2D309A23"
FROZEN_SOURCE_SHA256 = "157649B780965ECC585F18B3030199CDC0F4FE3013958FFA4095FCF665FDB1EA"
RUN_PATH = "runs/family-08-catboost-specialist"
ARTIFACT_PATH = "artifacts/09-family-08-catboost-specialist"
DATA_PATH = "data/experiments/top10_20260815/family-08-catboost-specialist"
FIXED_FAMILY_7_ARTIFACT = Path(
    r"C:\Users\danhm\mma-ai\worktrees\top10-20260815"
    r"\experiments\top10_20260815\artifacts\08-family-07-matchup-geometry"
)

PROFILE_IDS = (
    "shared-normalized-ordered-low",
    "shared-normalized-ordered-medium",
    "raw-count-exposure-ordered-low",
    "raw-count-exposure-ordered-medium",
    "native-categorical-ordered-low",
    "native-categorical-ordered-medium",
    "native-categorical-clustered-l2",
    "native-categorical-clustered-random",
)
FOLD_IDS = ("outer-2022", "outer-2023", "outer-2024", "outer-2025")

CATBOOST_PARAMETER_KEYS = (
    "loss_function",
    "eval_metric",
    "iterations",
    "learning_rate",
    "depth",
    "l2_leaf_reg",
    "random_strength",
    "bagging_temperature",
    "bootstrap_type",
    "boosting_type",
    "grow_policy",
    "border_count",
    "one_hot_max_size",
    "random_seed",
    "task_type",
    "devices",
    "thread_count",
    "early_stopping_rounds",
    "use_best_model",
    "allow_writing_files",
    "verbose",
    "nan_mode",
    "leaf_estimation_method",
    "leaf_estimation_iterations",
)

RAW_NUMERIC_FEATURES = (
    "fighter1_prior_win_count",
    "fighter2_prior_win_count",
    "fighter1_prior_ko_count",
    "fighter2_prior_ko_count",
    "fighter1_prior_submission_count",
    "fighter2_prior_submission_count",
    "fighter1_prior_fight_exposure",
    "fighter2_prior_fight_exposure",
    "fighter1_prior_sig_str_land_count",
    "fighter2_prior_sig_str_land_count",
    "fighter1_prior_sig_str_attempt_exposure",
    "fighter2_prior_sig_str_attempt_exposure",
    "fighter1_prior_control_seconds",
    "fighter2_prior_control_seconds",
    "fighter1_prior_fight_seconds_exposure",
    "fighter2_prior_fight_seconds_exposure",
    "fighter1_age_asof",
    "fighter2_age_asof",
    "fighter1_days_since_prior_fight",
    "fighter2_days_since_prior_fight",
)

RATE_FEATURES = (
    {
        "name": "fighter1_prior_win_rate",
        "numerator": "fighter1_prior_win_count",
        "exposure": "fighter1_prior_fight_exposure",
        "missing_semantics": "missing iff exposure is zero; CatBoost numeric NaN",
    },
    {
        "name": "fighter2_prior_win_rate",
        "numerator": "fighter2_prior_win_count",
        "exposure": "fighter2_prior_fight_exposure",
        "missing_semantics": "missing iff exposure is zero; CatBoost numeric NaN",
    },
    {
        "name": "fighter1_prior_sig_str_accuracy",
        "numerator": "fighter1_prior_sig_str_land_count",
        "exposure": "fighter1_prior_sig_str_attempt_exposure",
        "missing_semantics": "missing iff exposure is zero; CatBoost numeric NaN",
    },
    {
        "name": "fighter2_prior_sig_str_accuracy",
        "numerator": "fighter2_prior_sig_str_land_count",
        "exposure": "fighter2_prior_sig_str_attempt_exposure",
        "missing_semantics": "missing iff exposure is zero; CatBoost numeric NaN",
    },
)

NATIVE_CATEGORICAL_FEATURES = (
    "weightclass_encoded_as_category",
    "fighter1_prior_experience_bucket",
    "fighter2_prior_experience_bucket",
    "fighter1_prior_activity_bucket",
    "fighter2_prior_activity_bucket",
)

FORBIDDEN_FEATURE_TOKENS = (
    "fighter1_id",
    "fighter2_id",
    "fighter_id",
    "opponent_id",
    "fighter1_name",
    "fighter2_name",
    "fighter_url",
    "event_url",
    "y_true",
    "target",
    "label",
    "winner",
    "result",
    "method",
    "post_fight",
    "future",
    "odds",
    "market",
)


class CatBoostSpecialistError(ValueError):
    """The CatBoost menu or representation violates the frozen protocol."""


def _parameters(*, capacity: str, l2: float, random_strength: float) -> dict[str, Any]:
    if capacity == "low":
        iterations, learning_rate, depth = 600, 0.035, 5
    else:
        iterations, learning_rate, depth = 1_000, 0.025, 7
    return {
        "loss_function": "Logloss",
        "eval_metric": "Logloss",
        "iterations": iterations,
        "learning_rate": learning_rate,
        "depth": depth,
        "l2_leaf_reg": l2,
        "random_strength": random_strength,
        "bagging_temperature": 1.0,
        "bootstrap_type": "Bayesian",
        "boosting_type": "Ordered",
        "grow_policy": "SymmetricTree",
        "border_count": 128,
        "one_hot_max_size": 2,
        "random_seed": 20260815,
        "task_type": "GPU",
        "devices": "0",
        "thread_count": 8,
        "early_stopping_rounds": 100,
        "use_best_model": True,
        "allow_writing_files": False,
        "verbose": False,
        "nan_mode": "Min",
        "leaf_estimation_method": "Newton",
        "leaf_estimation_iterations": 10,
    }


def _representations() -> dict[str, dict[str, Any]]:
    raw_numeric = [*RAW_NUMERIC_FEATURES, *(item["name"] for item in RATE_FEATURES)]
    raw = {
        "availability": "prediction-as-of-time",
        "numeric_feature_names": raw_numeric,
        "categorical_feature_names": [],
        "rate_features": deepcopy(list(RATE_FEATURES)),
        "normalization": {
            "fit_scope": "fold-training-only",
            "method": "none-native-numeric",
            "missing_numeric": "CatBoost-NaN-with-explicit-exposure",
        },
        "lineage": "fighter-prior-only-counts-and-exposures",
    }
    return {
        "shared-normalized": {
            "availability": "prediction-as-of-time",
            "numeric_feature_names": list(V8_FEATURES),
            "categorical_feature_names": [],
            "rate_features": [],
            "normalization": {
                "fit_scope": "outer-train-only",
                "method": "robust",
                "missing_numeric": "outer-training-median",
            },
            "lineage": "frozen-family-01-v8-control",
        },
        "raw-count-exposure": raw,
        "native-categorical": {
            **deepcopy(raw),
            "categorical_feature_names": list(NATIVE_CATEGORICAL_FEATURES),
            "categorical_statistics": {
                "fit_scope": "fold-training-only",
                "target_statistics": "CatBoost-ordered-within-fit-only",
                "unseen_value": "CatBoost-native-unseen-category",
                "missing_value": "__MISSING__",
            },
            "lineage": "fighter-prior-only-counts-exposures-and-asof-categories",
        },
    }


def _profile(
    profile_id: str,
    representation_id: str,
    capacity: str,
    *,
    l2: float,
    random_strength: float,
) -> dict[str, Any]:
    categorical = (
        list(NATIVE_CATEGORICAL_FEATURES)
        if representation_id == "native-categorical"
        else []
    )
    value = {
        "id": profile_id,
        "representation_id": representation_id,
        "capacity": capacity,
        "fold_ids": list(FOLD_IDS),
        "categorical_feature_names": categorical,
        "catboost_parameters": _parameters(
            capacity=capacity,
            l2=l2,
            random_strength=random_strength,
        ),
        "refit_full": False,
    }
    return {**value, "profile_sha256": canonical_sha256(value)}


def build_preregistered_profile() -> dict[str, Any]:
    """Materialize the exact shared/raw/native CatBoost comparison before score."""

    profiles = [
        _profile(PROFILE_IDS[0], "shared-normalized", "low", l2=5.0, random_strength=1.0),
        _profile(PROFILE_IDS[1], "shared-normalized", "medium", l2=5.0, random_strength=1.0),
        _profile(PROFILE_IDS[2], "raw-count-exposure", "low", l2=5.0, random_strength=1.0),
        _profile(PROFILE_IDS[3], "raw-count-exposure", "medium", l2=5.0, random_strength=1.0),
        _profile(PROFILE_IDS[4], "native-categorical", "low", l2=5.0, random_strength=1.0),
        _profile(PROFILE_IDS[5], "native-categorical", "medium", l2=5.0, random_strength=1.0),
        _profile(PROFILE_IDS[6], "native-categorical", "low", l2=12.0, random_strength=1.0),
        _profile(PROFILE_IDS[7], "native-categorical", "medium", l2=5.0, random_strength=2.5),
    ]
    profile = {
        "experiment_id": EXPERIMENT_ID,
        "family_number": 8,
        "frozen_spec_sha256": FROZEN_SPEC_SHA256,
        "frozen_source": {
            "path": "artifacts/01-campaign-harness/frozen/training_data.csv",
            "sha256": FROZEN_SOURCE_SHA256,
            "development_safe_id_count": 3_089,
            "retired_id_count": 178,
            "development_max_date": "2025-12-13",
        },
        "dependency": {
            "experiment_id": "family-07-matchup-swap-geometry",
            "run_alias": "family-07-matchup-geometry",
            "required_data_path": "data/experiments/top10_20260815/family-07-matchup-geometry",
            "fallback": "terminal-failure-before-row-decode",
        },
        "representations": _representations(),
        "profiles": profiles,
        "fold_ids": list(FOLD_IDS),
        "outer_years": [2022, 2023, 2024, 2025],
        "inner_validation_year_count": 3,
        "selection": {
            "fit_scope": "prior-inner-only",
            "score": "mean-inner-log-loss",
            "tie_break": list(PROFILE_IDS),
            "outer_label_selection_count": 0,
            "odds_selection": False,
        },
        "database_access": {"used": False, "sql": None, "urls": []},
        "invocation": {"gpu_lease_count": 1, "retry_count": 0, "serialized": True},
    }
    validate_preregistered_profile(profile)
    return profile


def _feature_names(representation: Mapping[str, Any]) -> list[str]:
    return [
        *(str(value) for value in representation.get("numeric_feature_names", [])),
        *(str(value) for value in representation.get("categorical_feature_names", [])),
    ]


def validate_preregistered_profile(profile: Mapping[str, Any]) -> dict[str, Any]:
    """Reject incomplete capacity, causal, representation, or fit declarations."""

    profiles = list(profile.get("profiles", []))
    if len(profiles) > 8:
        raise CatBoostSpecialistError("maximum eight CatBoost profiles")
    if len(profiles) != 8 or tuple(item.get("id") for item in profiles) != PROFILE_IDS:
        raise CatBoostSpecialistError("exact ordered eight-profile menu required")
    representations = profile.get("representations", {})
    if set(representations) != {
        "shared-normalized",
        "raw-count-exposure",
        "native-categorical",
    }:
        raise CatBoostSpecialistError("shared, raw, and native representations are required")
    shared = representations["shared-normalized"]
    if shared.get("numeric_feature_names") != list(V8_FEATURES):
        raise CatBoostSpecialistError("shared-control inputs must reproduce exactly")
    for representation_id, representation in representations.items():
        normalization = representation.get("normalization", {})
        if normalization.get("fit_scope") == "global":
            raise CatBoostSpecialistError("global preprocessing is forbidden")
        for name in _feature_names(representation):
            lowered = name.lower()
            if any(token in lowered for token in FORBIDDEN_FEATURE_TOKENS):
                raise CatBoostSpecialistError(
                    "identity, target, post-fight, future, or odds feature leakage"
                )
        if representation_id in {"raw-count-exposure", "native-categorical"}:
            for rate in representation.get("rate_features", []):
                if not all(rate.get(key) for key in ("name", "numerator", "exposure", "missing_semantics")):
                    raise CatBoostSpecialistError("raw rate has missing exposure semantics")
        if representation_id == "native-categorical":
            statistics = representation.get("categorical_statistics", {})
            if statistics.get("fit_scope") != "fold-training-only":
                raise CatBoostSpecialistError("native categorical statistics must be fold-local")
    parameter_keys = set(CATBOOST_PARAMETER_KEYS)
    for item in profiles:
        parameters = item.get("catboost_parameters", {})
        if set(parameters) != parameter_keys:
            raise CatBoostSpecialistError("fully materialized CatBoost parameters required")
        if item.get("refit_full") is not False:
            raise CatBoostSpecialistError("refit_full is forbidden")
        representation_id = item.get("representation_id")
        if representation_id not in representations:
            raise CatBoostSpecialistError("unknown representation")
        if item.get("fold_ids") != list(FOLD_IDS):
            raise CatBoostSpecialistError("profiles must use identical fold IDs")
        if representation_id == "native-categorical":
            if parameters.get("boosting_type") != "Ordered":
                raise CatBoostSpecialistError("native profiles require ordered boosting")
            if item.get("categorical_feature_names") != list(NATIVE_CATEGORICAL_FEATURES):
                raise CatBoostSpecialistError("native categorical declaration differs")
        core = {key: value for key, value in item.items() if key != "profile_sha256"}
        if item.get("profile_sha256") != canonical_sha256(core):
            raise CatBoostSpecialistError("profile hash does not cover all explicit defaults")
    if profile.get("selection", {}).get("odds_selection") is not False:
        raise CatBoostSpecialistError("odds-derived selection is forbidden")
    if profile.get("selection", {}).get("outer_label_selection_count") != 0:
        raise CatBoostSpecialistError("outer-label capacity selection is forbidden")
    return {
        "profile_count": len(profiles),
        "profile_ids": [item["id"] for item in profiles],
        "profile_hashes": {item["id"]: item["profile_sha256"] for item in profiles},
        "representation_hashes": {
            key: canonical_sha256(value) for key, value in representations.items()
        },
    }


def validate_fold_fit_evidence(
    evidence: Sequence[Mapping[str, Any]],
    *,
    profile: Mapping[str, Any],
) -> dict[str, Any]:
    """Require every comparison to use the same fold-local fitting boundary."""

    validate_preregistered_profile(profile)
    if tuple(item.get("profile_id") for item in evidence) != PROFILE_IDS:
        raise CatBoostSpecialistError("fit evidence differs from preregistered profiles")
    for item in evidence:
        if item.get("fold_ids") != list(FOLD_IDS):
            raise CatBoostSpecialistError("all representations require identical fold IDs")
        if any(
            item.get(key) != "fold-training-only"
            for key in (
                "numeric_fit_scope",
                "categorical_statistics_scope",
                "missing_value_scope",
            )
        ):
            raise CatBoostSpecialistError("preprocessing and categorical statistics must be fold-local")
        if item.get("outer_label_selection_count") != 0:
            raise CatBoostSpecialistError("outer-label selection is forbidden")
    return {"fold_ids": list(FOLD_IDS), "profile_count": len(evidence)}


def write_preregistration(campaign_root: Path, *, source_revision: str) -> dict[str, Any]:
    """Persist the complete menu while every score destination is absent."""

    campaign_root = Path(campaign_root)
    profile_path = campaign_root / "profiles/family-08-catboost-specialist.json"
    preregistration_path = campaign_root / RUN_PATH / "preregistration.json"
    artifact_root = campaign_root / ARTIFACT_PATH
    data_root = campaign_root.parents[1] / DATA_PATH
    if any(path.exists() for path in (profile_path, preregistration_path, artifact_root, data_root)):
        raise ValueError("family 8 preregistration destinations must all be absent")
    gate = AccessLedger(campaign_root).gate_status()
    if gate["state"] != "closed" or gate["protected_access_count"] != 0:
        raise ValueError("family 8 preregistration requires the gate closed with zero access")
    profile = build_preregistered_profile()
    validated = validate_preregistered_profile(profile)
    write_canonical_json(profile_path, profile)
    preregistration = {
        "experiment_id": EXPERIMENT_ID,
        "family_number": 8,
        "source_revision": source_revision,
        "frozen_spec_sha256": FROZEN_SPEC_SHA256,
        "profile_path": "profiles/family-08-catboost-specialist.json",
        "profile_sha256": canonical_sha256(profile),
        "profile_file_sha256": file_sha256(profile_path),
        "preregistered_profile_ids": list(PROFILE_IDS),
        "ordered_profile_hashes": validated["profile_hashes"],
        "representation_hashes": validated["representation_hashes"],
        "registry_prefix_sha256_before": hashlib.sha256(
            (campaign_root / "registry.jsonl").read_bytes()
        ).hexdigest().upper(),
        "scoring_state": "not-started",
        "selection": profile["selection"],
        "database_access": profile["database_access"],
        "invocation": profile["invocation"],
        "gate_required_state": "closed-zero-access",
        "terminal_failure_rule": (
            "Any dependency, safe-population, representation, fold, safety, or destination "
            "mismatch terminates without retry."
        ),
    }
    write_canonical_json(preregistration_path, preregistration)
    return preregistration


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"".join(canonical_json_bytes(dict(row)) + b"\n" for row in rows))


def _append_registry(campaign_root: Path, payload: Mapping[str, Any]) -> dict[str, str]:
    registry_path = campaign_root / "registry.jsonl"
    head_path = campaign_root / "registry-head.json"
    before = registry_path.read_bytes()
    records = [json.loads(line) for line in before.splitlines()]
    if any(record["payload"]["experiment_id"] == EXPERIMENT_ID for record in records):
        raise ValueError("family 8 already exists in the registry")
    head = read_json(head_path)
    prefix_before = hashlib.sha256(before).hexdigest().upper()
    if prefix_before != head["registry_prefix_sha256"] or len(records) != 8:
        raise ValueError("registry head does not match the immutable family-7 prefix")
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
        "record_sha256": record["record_sha256"],
        "registry_prefix_sha256_before": prefix_before,
        "registry_prefix_sha256_after": hashlib.sha256(after).hexdigest().upper(),
    }


def materialize_family_08(
    campaign_root: Path,
    *,
    source_revision: str,
    preregistration_commit: str,
) -> dict[str, Any]:
    """Record one no-retry dependency failure before any row or target decode."""

    campaign_root = Path(campaign_root)
    run_root = campaign_root / RUN_PATH
    artifact_root = campaign_root / ARTIFACT_PATH
    data_root = campaign_root.parents[1] / DATA_PATH
    manifest_path = run_root / "manifest.json"
    if artifact_root.exists() or data_root.exists() or manifest_path.exists():
        raise ValueError("family 8 score destination already exists; retries are forbidden")
    inherited_database = [
        name for name in ("DATABASE_URL", "ODDS_DATABASE_URL") if os.environ.get(name)
    ]
    if inherited_database:
        raise ValueError("family 8 refuses inherited database URLs")
    gate = AccessLedger(campaign_root).gate_status()
    if gate["state"] != "closed" or gate["protected_access_count"] != 0:
        raise ValueError("family 8 requires the gate closed with zero access")
    profile_path = campaign_root / "profiles/family-08-catboost-specialist.json"
    preregistration_path = run_root / "preregistration.json"
    profile = read_json(profile_path)
    preregistration = read_json(preregistration_path)
    validated = validate_preregistered_profile(profile)
    registry_before = hashlib.sha256((campaign_root / "registry.jsonl").read_bytes()).hexdigest().upper()
    if (
        profile != build_preregistered_profile()
        or preregistration["scoring_state"] != "not-started"
        or preregistration["profile_sha256"] != canonical_sha256(profile)
        or preregistration["profile_file_sha256"] != file_sha256(profile_path)
        or preregistration["registry_prefix_sha256_before"] != registry_before
    ):
        raise ValueError("family 8 was not exactly preregistered before launch")

    # Identity-only subtraction is the mandatory first data operation. No CSV row is opened here.
    fold_manifest = read_json(campaign_root / "baseline/fold-manifest.json")
    safe_ids, retired_ids = build_development_safe_ids(fold_manifest)
    development_max_date = str(fold_manifest["folds"][-1]["outer"]["test_date_range"][1])
    if len(safe_ids) != 3_089 or len(retired_ids) != 178 or development_max_date != "2025-12-13":
        raise ValueError("family 8 development-safe population differs before target decode")
    development_population = {
        "asserted_before_row_or_target_decode": True,
        "development_safe_id_count": len(safe_ids),
        "development_max_date": development_max_date,
        "retired_id_count": len(retired_ids),
    }

    artifact_root.mkdir(parents=True)
    started_at = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    acquired = {
        "lease_id": "family-08-serialized-gpu-lease-1",
        "ordinal": 1,
        "pid": os.getpid(),
        "state": "acquired",
        "started_at": started_at,
    }
    write_canonical_json(artifact_root / "gpu-lease-acquired.json", acquired)

    dependency_manifest = read_json(campaign_root / "runs/family-07-matchup-geometry/manifest.json")
    dependency_inventory = tree_inventory(FIXED_FAMILY_7_ARTIFACT)
    dependency_predictions = [
        *dependency_manifest["outer_original_prediction_identities"],
        *dependency_manifest["outer_swapped_prediction_identities"],
    ]
    if (
        dependency_manifest["exit_state"] != "failed"
        or dependency_manifest["data_path"] is not None
        or dependency_predictions
        or dependency_inventory.tree_sha256 != dependency_manifest["artifact_tree_sha256"]
    ):
        raise ValueError("family 7 dependency evidence differs from the frozen terminal failure")
    dependency_evidence = {
        "experiment_id": dependency_manifest["experiment_id"],
        "exit_state": dependency_manifest["exit_state"],
        "artifact_tree_sha256": dependency_inventory.tree_sha256,
        "artifact_file_count": dependency_inventory.file_count,
        "data_path": dependency_manifest["data_path"],
        "outer_prediction_identities": dependency_predictions,
        "adaptive_signal": dependency_manifest["adaptive_signal_for_family_08"],
    }
    failure_text = (
        "Family 8 terminated before row or target decode: the immutable family 7 dependency "
        "failed before construction and produced no matchup table or Original predictions."
    )
    stderr_path = artifact_root / "terminal-stderr.txt"
    stderr_path.write_text(failure_text + "\n", encoding="utf-8", newline="\n")
    terminal_failure = {
        "attempt_ordinal": 1,
        "stage": "pre-construction-dependency-resolution",
        "exception_type": "DependencyUnavailable",
        "message": failure_text,
        "stderr_path": "terminal-stderr.txt",
        "stderr_sha256": file_sha256(stderr_path),
        "construction_started": False,
        "row_decode_started": False,
        "target_decode_started": False,
        "fit_started": False,
        "outer_labels_scored": False,
        "retry_performed": False,
    }
    representation_comparison = {
        "status": "unavailable",
        "reason": "pre-construction-dependency-failure",
        "shared_profile_ids": list(PROFILE_IDS[:2]),
        "raw_count_exposure_profile_ids": list(PROFILE_IDS[2:4]),
        "native_categorical_profile_ids": list(PROFILE_IDS[4:]),
        "identical_fold_ids": list(FOLD_IDS),
        "native_versus_shared_deltas": None,
    }
    promotion = {
        "action": "retain-family-01-weighted-v8-control",
        "incumbent_before": "family-01-weighted-v8-control",
        "incumbent_after": "family-01-weighted-v8-control",
        "promoted": False,
        "rule": "failed candidates cannot be promoted",
    }
    adaptive_signal = {
        "status": "family-08-failed-before-catboost-construction",
        "selected_profiles": [],
        "selected_representation_hashes": [],
        "native_categorical_evidence_available": False,
    }
    result = {
        "experiment_id": EXPERIMENT_ID,
        "status": "failed",
        "terminal_failure": terminal_failure,
        "metrics": None,
        "incumbent_metrics": None,
        "metric_deltas": None,
        "paired_event_block_intervals": None,
        "calibration_gaps": None,
        "subgroup_gaps": None,
        "train_gaps": None,
        "capacity_diagnostics": None,
        "representation_comparison": representation_comparison,
        "representation_schema_and_lineage_hashes": validated["representation_hashes"],
        "outer_prediction_identities": [],
        "promotion_decision": promotion,
        "adaptive_signal_for_family_09": adaptive_signal,
        "development_safe_population": development_population,
        "dependency_evidence": dependency_evidence,
        "gate_access_count": gate["protected_access_count"],
    }
    write_canonical_json(artifact_root / "failure.json", terminal_failure)
    write_canonical_json(artifact_root / "dependency-evidence.json", dependency_evidence)
    write_canonical_json(
        artifact_root / "representation-schema.json",
        {
            "representations": profile["representations"],
            "representation_hashes": validated["representation_hashes"],
            "fold_ids": list(FOLD_IDS),
        },
    )
    write_canonical_json(
        artifact_root / "safety.json",
        {
            "database_access": profile["database_access"],
            "gpu_lease_count": 1,
            "production_attempt_count": 1,
            "retry_count": 0,
            "serialized": True,
            "gate_access_count": gate["protected_access_count"],
            **development_population,
        },
    )
    ended_at = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    released = {
        "lease_id": acquired["lease_id"],
        "ordinal": 1,
        "pid": acquired["pid"],
        "state": "released-after-terminal-failure",
        "ended_at": ended_at,
    }
    write_canonical_json(artifact_root / "gpu-lease-released.json", released)
    runtime = {
        "attempt_ordinal": 1,
        "started_at": started_at,
        "ended_at": ended_at,
        "gpu_lease_id": acquired["lease_id"],
        "serialized": True,
        "retry_count": 0,
        "terminal_stage": terminal_failure["stage"],
    }
    write_canonical_json(artifact_root / "runtime.json", runtime)
    write_canonical_json(artifact_root / "result.json", result)
    inventory = tree_inventory(artifact_root)

    _write_jsonl(
        run_root / "attempts.jsonl",
        [
            {
                "attempt_ordinal": 1,
                "state": "failed",
                "stage": terminal_failure["stage"],
                "exception_type": terminal_failure["exception_type"],
                "gpu_lease_id": acquired["lease_id"],
                "retry": False,
            }
        ],
    )
    (run_root / "decision.md").write_text(
        "# Family 8 decision\n\nRetain family 1: the sole serialized attempt stopped before row decode because family 7 produced no matchup table or predictions.\n",
        encoding="utf-8",
        newline="\n",
    )
    manifest = {
        **result,
        "kind": "family",
        "exit_state": "failed",
        "artifact_path": ARTIFACT_PATH,
        "artifact_tree_sha256": inventory.tree_sha256,
        "artifact_file_count": inventory.file_count,
        "data_path": None,
        "data_sha256": None,
        "profile_path": "profiles/family-08-catboost-specialist.json",
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
            "profile_path": manifest["profile_path"],
            "profile_sha256": manifest["profile_sha256"],
            "status": "failed",
        },
    )
    return {**result, **registry, "artifact_tree_sha256": inventory.tree_sha256}
