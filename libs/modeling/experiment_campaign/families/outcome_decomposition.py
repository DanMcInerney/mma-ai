"""Family 10 preregistration and serialized outcome-decomposition execution."""

from __future__ import annotations

from copy import deepcopy
import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import time
from typing import Any, Mapping

import numpy as np
import pandas as pd

from ..feature_lineage import build_development_safe_ids, decode_development_rows
from ..hashing import (
    canonical_json_bytes,
    canonical_sha256,
    file_sha256,
    read_json,
    tree_inventory,
    write_canonical_json,
)
from ..metrics import event_block_bootstrap_delta, metric_gap, reduce_predictions
from ..outcome_decomposition import (
    OutcomeDecompositionError,
    blend_probability,
    build_combined_records,
    shrink_probability,
)
from ..protocol import AccessLedger
from .semantic_portfolio import V8_FEATURES


EXPERIMENT_ID = "family-10-outcome-decomposition"
RUN_ALIAS = "family-10-outcome-decomposition"
RUN_PATH = "runs/family-10-outcome-decomposition"
ARTIFACT_PATH = "artifacts/11-family-10-outcome-decomposition"
PROFILE_PATH = "profiles/family-10-outcome-decomposition.json"
FROZEN_SPEC_SHA256 = "93FB5CC31AD810B1867FFC8A250DD257AAF74732998D103D56AB8D3A2D309A23"
FROZEN_SOURCE_SHA256 = "157649B780965ECC585F18B3030199CDC0F4FE3013958FFA4095FCF665FDB1EA"

FIXED_CAMPAIGN_ROOT = Path(
    r"C:\Users\danhm\mma-ai\worktrees\top10-20260815"
    r"\experiments\top10_20260815"
)
FIXED_SOURCE_CSV = FIXED_CAMPAIGN_ROOT / "artifacts/01-campaign-harness/frozen/training_data.csv"
FIXED_CONTROL_ROOT = (
    FIXED_CAMPAIGN_ROOT / "artifacts/02-family-01-weighted-v8-control"
)

VARIANT_IDS = (
    "direct-incumbent-control",
    "three-component",
    "shrinkage-gated-three-component",
    "decision-finish-specialist-mixture",
    "support-trimmed-specialist-mixture",
    "constant-prior-fallback",
)
COMPONENT_IDS = ("decision", "decision-win", "finish-win")

_MODEL = {
    "library": "catboost",
    "class": "CatBoostClassifier",
    "iterations": 300,
    "depth": 4,
    "learning_rate": 0.03,
    "l2_leaf_reg": 8.0,
    "loss_function": "Logloss",
    "random_seed": 20260815,
    "task_type": "CPU",
    "thread_count": 6,
    "allow_writing_files": False,
    "verbose": False,
}


def _variant(identifier: str, **parameters: Any) -> dict[str, Any]:
    value = {"id": identifier, **parameters}
    value["profile_sha256"] = canonical_sha256(value)
    return value


def build_preregistered_profile() -> dict[str, Any]:
    """Materialize every fit, support, prior, clipping, and fallback default."""

    variants = [
        _variant("direct-incumbent-control", kind="control", component_ids=[]),
        _variant(
            "three-component",
            kind="law-of-total-probability",
            component_ids=list(COMPONENT_IDS),
            clipping=[0.02, 0.98],
        ),
        _variant(
            "shrinkage-gated-three-component",
            kind="law-of-total-probability-shrunk",
            component_ids=list(COMPONENT_IDS),
            shrinkage_strength=80.0,
            clipping=[0.02, 0.98],
        ),
        _variant(
            "decision-finish-specialist-mixture",
            kind="decomposition-control-mixture",
            component_ids=list(COMPONENT_IDS),
            decomposition_weight=0.75,
            clipping=[0.02, 0.98],
        ),
        _variant(
            "support-trimmed-specialist-mixture",
            kind="support-trimmed-mixture",
            component_ids=list(COMPONENT_IDS),
            minimum_conditional_support=120,
            decomposition_weight_supported=0.8,
            decomposition_weight_sparse=0.35,
            clipping=[0.02, 0.98],
        ),
        _variant(
            "constant-prior-fallback",
            kind="constant-prior-decomposition",
            component_ids=list(COMPONENT_IDS),
            clipping=[0.02, 0.98],
        ),
    ]
    return {
        "experiment_id": EXPERIMENT_ID,
        "frozen_spec_sha256": FROZEN_SPEC_SHA256,
        "source": {
            "sha256": "157649B780965ECC585F18B3030199CDC0F4FE3013958FFA4095FCF665FDB1EA",
            "development_safe_id_count": 3_089,
            "development_max_date": "2025-12-13",
            "retired_id_count": 178,
            "safe_ids_asserted_before_label_decode": True,
        },
        "outer_folds": [2022, 2023, 2024, 2025],
        "outer_row_count": 1_108,
        "features": list(V8_FEATURES),
        "components": [
            {"id": "decision", "label": "method-contains-dec", "training_subset": "all-prior"},
            {"id": "decision-win", "label": "fighter1-win", "training_subset": "prior-decisions"},
            {"id": "finish-win", "label": "fighter1-win", "training_subset": "prior-finishes"},
        ],
        "model": deepcopy(_MODEL),
        "sample_weight": {"kind": "exponential-recency", "annual_decay": 0.15, "normalize": True},
        "fallbacks": [{"id": "constant-prior", "registered": True, "prior": 0.5}],
        "support": {"minimum_specialist_rows": 120, "sparse_action": "registered-constant-prior"},
        "gate": {"kind": "fixed", "outer_label_reads": 0},
        "selection": {
            "metric": "log_loss",
            "outer_label_component_fit_count": 0,
            "promotion_requires_full_denominator": True,
        },
        "runtime": {
            "production_process_count": 1,
            "lease_count": 1,
            "serialized": True,
            "retry_count": 0,
            "refit_full": False,
            "database_access": False,
        },
        "variants": variants,
    }


def validate_preregistered_profile(profile: Mapping[str, Any]) -> dict[str, Any]:
    """Validate the exact bounded menu and all contamination barriers."""

    if profile.get("experiment_id") != EXPERIMENT_ID:
        raise OutcomeDecompositionError("family 10 experiment identity differs")
    variants = list(profile.get("variants", ()))
    if len(variants) > 6:
        raise OutcomeDecompositionError("family 10 may preregister at most six variants")
    if tuple(item.get("id") for item in variants) != VARIANT_IDS:
        raise OutcomeDecompositionError("family 10 variant menu differs")
    if profile.get("outer_folds") != [2022, 2023, 2024, 2025] or profile.get("outer_row_count") != 1_108:
        raise OutcomeDecompositionError("family 10 full development denominator differs")
    if profile.get("source", {}).get("safe_ids_asserted_before_label_decode") is not True:
        raise OutcomeDecompositionError("safe IDs must be asserted before label decode")
    if profile.get("gate") != {"kind": "fixed", "outer_label_reads": 0}:
        raise OutcomeDecompositionError("learned gate lineage differs")
    fallbacks = list(profile.get("fallbacks", ()))
    if fallbacks != [{"id": "constant-prior", "registered": True, "prior": 0.5}]:
        raise OutcomeDecompositionError("exact registered fallback is required")
    if profile.get("runtime", {}).get("refit_full") is not False:
        raise OutcomeDecompositionError("refit_full is forbidden")
    for item in variants:
        core = {key: value for key, value in item.items() if key != "profile_sha256"}
        if item.get("profile_sha256") != canonical_sha256(core):
            raise OutcomeDecompositionError("variant hash does not cover its defaults")
    return {
        "variant_count": len(variants),
        "variant_ids": [item["id"] for item in variants],
        "variant_hashes": {item["id"]: item["profile_sha256"] for item in variants},
    }


def write_preregistration(campaign_root: Path, *, source_revision: str) -> dict[str, Any]:
    """Commit the exact six-variant menu while every launch destination is absent."""

    campaign_root = Path(campaign_root)
    profile_path = campaign_root / PROFILE_PATH
    preregistration_path = campaign_root / RUN_PATH / "preregistration.json"
    artifact_root = campaign_root / ARTIFACT_PATH
    if any(path.exists() for path in (profile_path, preregistration_path, artifact_root)):
        raise ValueError("family 10 preregistration destinations must all be absent")
    gate = AccessLedger(campaign_root).gate_status()
    if gate["state"] != "closed" or gate["protected_access_count"] != 0:
        raise ValueError("family 10 preregistration requires the gate closed with zero access")
    profile = build_preregistered_profile()
    validated = validate_preregistered_profile(profile)
    write_canonical_json(profile_path, profile)
    preregistration = {
        "experiment_id": EXPERIMENT_ID,
        "family_number": 10,
        "source_revision": source_revision,
        "frozen_spec_sha256": FROZEN_SPEC_SHA256,
        "profile_path": PROFILE_PATH,
        "profile_sha256": canonical_sha256(profile),
        "profile_file_sha256": file_sha256(profile_path),
        "variant_count": validated["variant_count"],
        "variant_ids": validated["variant_ids"],
        "variant_hashes": validated["variant_hashes"],
        "registry_prefix_sha256_before": hashlib.sha256(
            (campaign_root / "registry.jsonl").read_bytes()
        ).hexdigest().upper(),
        "launch_state": "not-started",
        "production_process_count": 1,
        "retry_count": 0,
        "database_access": {"used": False, "sql": None, "urls": []},
        "gate_required_state": "closed-zero-access",
        "terminal_failure_rule": (
            "Any safe-population, label, component fit, denominator, safety, or destination "
            "mismatch terminates the single attempt without retry."
        ),
    }
    write_canonical_json(preregistration_path, preregistration)
    return preregistration


def _write_jsonl(path: Path, rows: list[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"".join(canonical_json_bytes(dict(row)) + b"\n" for row in rows))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in Path(path).read_bytes().splitlines()]


def _assert_safe_metadata(
    source: Path,
    *,
    safe_ids: tuple[str, ...],
    retired_ids: frozenset[str],
) -> dict[str, Any]:
    metadata = pd.read_csv(
        source,
        usecols=["fight_id", "event_id", "event_date"],
        dtype={"fight_id": str, "event_id": str, "event_date": str},
    )
    metadata["event_date"] = pd.to_datetime(metadata["event_date"], utc=True).dt.strftime("%Y-%m-%d")
    if metadata["fight_id"].duplicated().any():
        raise ValueError("frozen source contains duplicate fight identities")
    safe = metadata.loc[metadata["fight_id"].isin(set(safe_ids))]
    retired = metadata.loc[metadata["fight_id"].isin(retired_ids)]
    if (
        len(safe) != 3_089
        or len(retired) != 178
        or safe["event_date"].max() != "2025-12-13"
        or set(safe["fight_id"]) != set(safe_ids)
        or set(retired["fight_id"]) != set(retired_ids)
        or set(safe["fight_id"]) & set(retired["fight_id"])
    ):
        raise ValueError("development-safe population differs before target/outcome decode")
    return {
        "asserted_before_target_decode": True,
        "development_safe_id_count": 3_089,
        "development_max_date": "2025-12-13",
        "retired_id_count": 178,
    }


def _load_development_frame(
    source: Path,
    *,
    safe_ids: tuple[str, ...],
    retired_ids: frozenset[str],
    features: list[str],
) -> pd.DataFrame:
    columns = ["fight_id", "event_id", "event_date", "method", "y_true", *features]
    with Path(source).open("rb") as handle:
        header = next(csv.reader([handle.readline().decode("utf-8")]))
        indices = [header.index(name) for name in columns]
        decoded = decode_development_rows(
            handle,
            safe_ids=safe_ids,
            retired_ids=retired_ids,
            indices=indices,
        )
    frame = pd.DataFrame(decoded, columns=columns)
    frame["fight_id"] = frame["fight_id"].astype(str)
    frame["event_id"] = frame["event_id"].astype(str)
    frame["event_date"] = pd.to_datetime(frame["event_date"], utc=True).dt.strftime("%Y-%m-%d")
    frame["y_true"] = pd.to_numeric(frame["y_true"], errors="raise").astype(int)
    if len(frame) != 3_089 or set(frame["fight_id"]) != set(safe_ids) or not frame["y_true"].isin([0, 1]).all():
        raise ValueError("decoded development labels differ from the asserted safe roster")
    for feature in features:
        frame[feature] = pd.to_numeric(frame[feature], errors="coerce")
    frame["is_decision"] = frame["method"].astype(str).str.lower().str.contains("dec").astype(int)
    return frame.sort_values(["event_date", "event_id", "fight_id"], kind="mergesort").reset_index(drop=True)


def _control_records(fold_manifest: Mapping[str, Any]) -> list[dict[str, Any]]:
    expected_ids = [
        str(fight_id)
        for fold in fold_manifest["folds"]
        for fight_id in fold["outer"]["test_fight_ids"]
    ]
    by_id: dict[str, dict[str, Any]] = {}
    for year in (2022, 2023, 2024, 2025):
        path = FIXED_CONTROL_ROOT / f"fold-{year}/outer-predictions.jsonl"
        for record in _read_jsonl(path):
            by_id[str(record["fight_id"])] = record
    if len(expected_ids) != 1_108 or len(set(expected_ids)) != 1_108 or set(by_id) != set(expected_ids):
        raise ValueError("Family-1 control denominator differs from the frozen outer folds")
    records = [by_id[fight_id] for fight_id in expected_ids]
    if len(records) != 1_108:
        raise ValueError("Family-1 control is not the exact 1,108-row denominator")
    return records


def _model_parameters(profile: Mapping[str, Any]) -> dict[str, Any]:
    excluded = {"library", "class"}
    return {key: value for key, value in profile["model"].items() if key not in excluded}


def _fit_component(
    profile: Mapping[str, Any],
    *,
    component_id: str,
    train: pd.DataFrame,
    outer: pd.DataFrame,
    feature_names: list[str],
    artifact_dir: Path,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    from catboost import CatBoostClassifier

    if component_id == "decision":
        fit = train
        target = fit["is_decision"].to_numpy(dtype=int)
    elif component_id == "decision-win":
        fit = train.loc[train["is_decision"] == 1]
        target = fit["y_true"].to_numpy(dtype=int)
    elif component_id == "finish-win":
        fit = train.loc[train["is_decision"] == 0]
        target = fit["y_true"].to_numpy(dtype=int)
    else:
        raise ValueError(f"unknown component: {component_id}")
    support = len(fit)
    prior = float(np.mean(target)) if support else 0.5
    minimum = int(profile["support"]["minimum_specialist_rows"])
    fallback = support < minimum or len(set(target.tolist())) < 2
    if fallback:
        outer_probability = np.full(len(outer), prior if support else 0.5, dtype=float)
        train_probability = np.full(len(train), prior if support else 0.5, dtype=float)
        model_identity = None
    else:
        model = CatBoostClassifier(**_model_parameters(profile))
        dates = pd.to_datetime(fit["event_date"], utc=True)
        days = (dates.max() - dates).dt.total_seconds() / 86_400.0
        decay = float(profile["sample_weight"]["annual_decay"])
        weights = np.exp(-decay * days / 365.25)
        weights = weights * len(weights) / weights.sum()
        model.fit(fit[feature_names], target, sample_weight=weights)
        outer_probability = model.predict_proba(outer[feature_names])[:, 1]
        train_probability = model.predict_proba(train[feature_names])[:, 1]
        model_path = artifact_dir / "model.cbm"
        model_path.parent.mkdir(parents=True, exist_ok=True)
        model.save_model(model_path)
        model_identity = {"path": model_path.name, "sha256": file_sha256(model_path)}
    return outer_probability, train_probability, {
        "component_id": component_id,
        "support": support,
        "positive_count": int(np.sum(target)),
        "negative_count": int(support - np.sum(target)),
        "prior": prior,
        "fallback_used": fallback,
        "model_identity": model_identity,
    }


def _component_records(
    outer: pd.DataFrame,
    *,
    probabilities: np.ndarray,
    component_id: str,
    fold: str,
    fit_max_date: str,
    embargo_days: int,
) -> list[dict[str, Any]]:
    return [
        {
            "component_id": component_id,
            "fight_id": str(row.fight_id),
            "fold": fold,
            "event_id": str(row.event_id),
            "event_date": str(row.event_date),
            "probability": float(probability),
            "fit_scope": "prior-only",
            "fit_max_date": fit_max_date,
            "embargo_days": embargo_days,
            "outer_label_reads": 0,
        }
        for row, probability in zip(outer.itertuples(index=False), probabilities, strict=True)
    ]


def _variant_records(
    profile: Mapping[str, Any],
    *,
    control: list[dict[str, Any]],
    components: Mapping[str, list[dict[str, Any]]],
    fold_evidence: Mapping[str, Mapping[str, Any]],
) -> dict[str, list[dict[str, Any]]]:
    by_fold: dict[str, list[int]] = {}
    for index, row in enumerate(control):
        by_fold.setdefault(str(row["fold"]), []).append(index)
    variants: dict[str, list[dict[str, Any]]] = {
        "direct-incumbent-control": [
            {**row, "candidate_id": "direct-incumbent-control", "outer_label_reads": 0}
            for row in control
        ]
    }
    base = build_combined_records(
        control,
        components["decision"],
        components["decision-win"],
        components["finish-win"],
        variant_id="three-component",
        clipping=(0.02, 0.98),
    )
    variants["three-component"] = base
    shrunk: list[dict[str, Any]] = []
    constant: list[dict[str, Any]] = []
    for fold, indices in by_fold.items():
        component_rows: dict[str, list[dict[str, Any]]] = {}
        constant_rows: dict[str, list[dict[str, Any]]] = {}
        for component_id in COMPONENT_IDS:
            evidence = fold_evidence[fold][component_id]
            component_rows[component_id] = [
                {
                    **components[component_id][index],
                    "probability": shrink_probability(
                        components[component_id][index]["probability"],
                        evidence["prior"],
                        evidence["support"],
                        80.0,
                    ),
                }
                for index in indices
            ]
            constant_rows[component_id] = [
                {**components[component_id][index], "probability": evidence["prior"]}
                for index in indices
            ]
        template = [control[index] for index in indices]
        shrunk.extend(build_combined_records(
            template,
            component_rows["decision"],
            component_rows["decision-win"],
            component_rows["finish-win"],
            variant_id="shrinkage-gated-three-component",
            clipping=(0.02, 0.98),
        ))
        constant.extend(build_combined_records(
            template,
            constant_rows["decision"],
            constant_rows["decision-win"],
            constant_rows["finish-win"],
            variant_id="constant-prior-fallback",
            clipping=(0.02, 0.98),
        ))
    variants["shrinkage-gated-three-component"] = shrunk
    variants["constant-prior-fallback"] = constant
    variants["decision-finish-specialist-mixture"] = [
        {
            **candidate,
            "candidate_id": "decision-finish-specialist-mixture",
            "probability": blend_probability(candidate["probability"], control_row["probability"], 0.75),
        }
        for candidate, control_row in zip(base, control, strict=True)
    ]
    trimmed = []
    for candidate, control_row in zip(base, control, strict=True):
        evidence = fold_evidence[str(candidate["fold"])]
        supported = min(evidence["decision-win"]["support"], evidence["finish-win"]["support"]) >= 120
        weight = 0.8 if supported else 0.35
        trimmed.append({
            **candidate,
            "candidate_id": "support-trimmed-specialist-mixture",
            "probability": blend_probability(candidate["probability"], control_row["probability"], weight),
            "specialist_support_status": "supported" if supported else "sparse",
        })
    variants["support-trimmed-specialist-mixture"] = trimmed
    if tuple(variants) != VARIANT_IDS or any(len(rows) != 1_108 for rows in variants.values()):
        raise ValueError("variant output does not cover the exact preregistered denominator")
    return variants


def _metric_records(
    frame: pd.DataFrame,
    probabilities: np.ndarray,
    labels: np.ndarray,
    *,
    fold: str,
) -> list[dict[str, Any]]:
    return [
        {
            "boundary": "Original",
            "fit_scope": "prior-only",
            "fold": fold,
            "fight_id": str(row.fight_id),
            "event_id": str(row.event_id),
            "event_date": str(row.event_date),
            "probability": float(probability),
            "y_true": int(label),
            "weight_class": str(row.weightclass_encoded),
            "experience": "unknown",
            "outcome_type": "component",
        }
        for row, probability, label in zip(
            frame.itertuples(index=False), probabilities, labels, strict=True
        )
    ]


def _append_registry(campaign_root: Path, payload: Mapping[str, Any]) -> dict[str, str]:
    registry_path = campaign_root / "registry.jsonl"
    head_path = campaign_root / "registry-head.json"
    before = registry_path.read_bytes()
    records = [json.loads(line) for line in before.splitlines()]
    if any(record["payload"]["experiment_id"] == EXPERIMENT_ID for record in records):
        raise ValueError("family 10 already exists in registry")
    head = read_json(head_path)
    prefix_before = hashlib.sha256(before).hexdigest().upper()
    if prefix_before != head["registry_prefix_sha256"] or len(records) != 10:
        raise ValueError("registry head does not match immutable family-9 prefix")
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


def materialize_family_10(
    campaign_root: Path,
    *,
    source_revision: str,
    preregistration_commit: str,
) -> dict[str, Any]:
    """Run the sole serialized full-denominator Family-10 production attempt."""

    campaign_root = Path(campaign_root)
    artifact_root = campaign_root / ARTIFACT_PATH
    run_root = campaign_root / RUN_PATH
    manifest_path = run_root / "manifest.json"
    if artifact_root.exists() or manifest_path.exists():
        raise ValueError("family 10 production destination exists; retries are forbidden")
    gate = AccessLedger(campaign_root).gate_status()
    if gate["state"] != "closed" or gate["protected_access_count"] != 0:
        raise ValueError("family 10 requires the gate closed with zero access")
    profile_path = campaign_root / PROFILE_PATH
    preregistration_path = run_root / "preregistration.json"
    profile = read_json(profile_path)
    preregistration = read_json(preregistration_path)
    validate_preregistered_profile(profile)
    prefix = hashlib.sha256((campaign_root / "registry.jsonl").read_bytes()).hexdigest().upper()
    if (
        profile != build_preregistered_profile()
        or preregistration["profile_sha256"] != canonical_sha256(profile)
        or preregistration["registry_prefix_sha256_before"] != prefix
        or preregistration["launch_state"] != "not-started"
        or preregistration["variant_count"] != 6
    ):
        raise ValueError("family 10 is not the exact committed preregistration")
    if file_sha256(FIXED_SOURCE_CSV) != FROZEN_SOURCE_SHA256:
        raise ValueError("frozen source identity differs")

    fold_manifest = read_json(campaign_root / "baseline/fold-manifest.json")
    safe_ids, retired_ids = build_development_safe_ids(fold_manifest)
    development_population = _assert_safe_metadata(
        FIXED_SOURCE_CSV,
        safe_ids=safe_ids,
        retired_ids=retired_ids,
    )
    expected_outer_counts = [len(fold["outer"]["test_fight_ids"]) for fold in fold_manifest["folds"]]
    if expected_outer_counts != [282, 251, 293, 282] or sum(expected_outer_counts) != 1_108:
        raise ValueError("outer folds do not define the exact 1,108-row denominator")

    artifact_root.mkdir(parents=True)
    started_at = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    lease = {
        "lease_id": "family-10-serialized-production-lease-1",
        "ordinal": 1,
        "pid": os.getpid(),
        "state": "acquired",
        "started_at": started_at,
        "model_task_type": profile["model"]["task_type"],
    }
    write_canonical_json(artifact_root / "production-lease-acquired.json", lease)
    write_canonical_json(artifact_root / "development-safe-population.json", development_population)

    attempts: list[dict[str, Any]] = []
    component_lineage: dict[str, dict[str, Any]] = {}
    component_records = {component_id: [] for component_id in COMPONENT_IDS}
    component_train_outer_gaps: dict[str, dict[str, Any]] = {}
    variant_results: list[dict[str, Any]] = []
    prediction_identities: dict[str, dict[str, Any]] = {}
    terminal_failure: dict[str, Any] | None = None
    status = "complete"
    variants: dict[str, list[dict[str, Any]]] = {}
    control: list[dict[str, Any]] = []
    try:
        frame = _load_development_frame(
            FIXED_SOURCE_CSV,
            safe_ids=safe_ids,
            retired_ids=retired_ids,
            features=list(profile["features"]),
        )
        control = _control_records(fold_manifest)
        frame_by_id = frame.set_index("fight_id", drop=False)
        if [str(row["fight_id"]) for row in control] != [
            str(fight_id)
            for fold in fold_manifest["folds"]
            for fight_id in fold["outer"]["test_fight_ids"]
        ]:
            raise ValueError("control rows do not follow the frozen outer-fold order")
        for fold in fold_manifest["folds"]:
            year = str(fold["test_year"])
            train_ids = [str(value) for value in fold["outer"]["train_fight_ids"]]
            outer_ids = [str(value) for value in fold["outer"]["test_fight_ids"]]
            train = frame_by_id.loc[train_ids].reset_index(drop=True)
            outer = frame_by_id.loc[outer_ids].reset_index(drop=True)
            if len(outer) != expected_outer_counts[int(year) - 2022]:
                raise ValueError(f"outer-{year} row count differs")
            embargo_days = int(fold["embargo_days"])
            fit_max_date = str(train["event_date"].max())
            if pd.Timestamp(fit_max_date) > pd.Timestamp(outer["event_date"].min()) - pd.Timedelta(days=embargo_days):
                raise ValueError(f"outer-{year} component context violates embargo")
            component_lineage[year] = {}
            for component_id in COMPONENT_IDS:
                ordinal = len(attempts) // 2 + 1
                launched_at = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
                attempts.append({
                    "attempt_ordinal": ordinal,
                    "fold": year,
                    "component_id": component_id,
                    "state": "launched",
                    "started_at": launched_at,
                    "production_lease_id": lease["lease_id"],
                    "retry": False,
                    "profile_sha256": canonical_sha256(profile),
                })
                component_dir = artifact_root / "components" / component_id / f"fold-{year}"
                outer_probability, train_probability, evidence = _fit_component(
                    profile,
                    component_id=component_id,
                    train=train,
                    outer=outer,
                    feature_names=list(profile["features"]),
                    artifact_dir=component_dir,
                )
                records = _component_records(
                    outer,
                    probabilities=outer_probability,
                    component_id=component_id,
                    fold=year,
                    fit_max_date=fit_max_date,
                    embargo_days=embargo_days,
                )
                prediction_path = component_dir / "outer-predictions.jsonl"
                _write_jsonl(prediction_path, records)
                component_records[component_id].extend(records)
                if component_id == "decision":
                    train_mask = np.ones(len(train), dtype=bool)
                    outer_mask = np.ones(len(outer), dtype=bool)
                    train_labels = train["is_decision"].to_numpy(dtype=int)
                    outer_labels = outer["is_decision"].to_numpy(dtype=int)
                else:
                    desired = 1 if component_id == "decision-win" else 0
                    train_mask = train["is_decision"].to_numpy(dtype=int) == desired
                    outer_mask = outer["is_decision"].to_numpy(dtype=int) == desired
                    train_labels = train.loc[train_mask, "y_true"].to_numpy(dtype=int)
                    outer_labels = outer.loc[outer_mask, "y_true"].to_numpy(dtype=int)
                train_metric = reduce_predictions(_metric_records(
                    train.loc[train_mask], train_probability[train_mask], train_labels, fold=f"train-{year}"
                ))
                outer_metric = reduce_predictions(_metric_records(
                    outer.loc[outer_mask], outer_probability[outer_mask], outer_labels, fold=year
                ))
                gap = metric_gap(train_metric, outer_metric)
                evidence.update({
                    "fold": year,
                    "fit_scope": "prior-only",
                    "fit_max_date": fit_max_date,
                    "outer_min_date": str(outer["event_date"].min()),
                    "embargo_days": embargo_days,
                    "outer_prediction_identity": {
                        "path": prediction_path.relative_to(artifact_root).as_posix(),
                        "sha256": file_sha256(prediction_path),
                        "row_count": len(records),
                    },
                    "train_metrics": train_metric.as_dict(),
                    "outer_component_metrics": outer_metric.as_dict(),
                    "train_outer_gap": gap,
                    "outer_label_fit_count": 0,
                })
                component_lineage[year][component_id] = evidence
                component_train_outer_gaps[f"{year}:{component_id}"] = gap
                write_canonical_json(component_dir / "fit-evidence.json", evidence)
                attempts.append({
                    "attempt_ordinal": ordinal,
                    "fold": year,
                    "component_id": component_id,
                    "state": "exited",
                    "exit_code": 0,
                    "ended_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
                    "production_lease_id": lease["lease_id"],
                    "retry": False,
                })
        if any(len(records) != 1_108 for records in component_records.values()):
            raise ValueError("component predictions do not cover all 1,108 outer rows")
        variants = _variant_records(
            profile,
            control=control,
            components=component_records,
            fold_evidence=component_lineage,
        )
        control_metrics = reduce_predictions(variants["direct-incumbent-control"]).as_dict()
        paired_intervals: dict[str, Any] = {}
        for variant_id in VARIANT_IDS:
            rows = variants[variant_id]
            path = artifact_root / "variants" / variant_id / "outer-predictions.jsonl"
            _write_jsonl(path, rows)
            metrics = reduce_predictions(rows).as_dict()
            intervals = None if variant_id == "direct-incumbent-control" else event_block_bootstrap_delta(
                rows,
                variants["direct-incumbent-control"],
                iterations=2_000,
                seed=20260815,
            )
            if intervals is not None:
                paired_intervals[variant_id] = intervals
            identity = {
                "path": path.relative_to(artifact_root).as_posix(),
                "sha256": file_sha256(path),
                "row_count": len(rows),
                "boundary": "Original",
                "outer_years": [2022, 2023, 2024, 2025],
            }
            prediction_identities[variant_id] = identity
            variant_results.append({
                "variant_id": variant_id,
                "profile_sha256": next(
                    item["profile_sha256"] for item in profile["variants"] if item["id"] == variant_id
                ),
                "metrics": metrics,
                "paired_event_block_intervals": intervals,
                "prediction_identity": identity,
            })
        best = min(
            (item for item in variant_results if item["variant_id"] != "direct-incumbent-control"),
            key=lambda item: (item["metrics"]["log_loss"], VARIANT_IDS.index(item["variant_id"])),
        )
        best_interval = paired_intervals[best["variant_id"]]
        promoted = (
            best["metrics"]["log_loss"] < control_metrics["log_loss"]
            and best_interval["log_loss_delta"]["upper"] < 0.0
            and best["metrics"]["brier"] <= control_metrics["brier"]
            and best["metrics"]["accuracy"] >= control_metrics["accuracy"] - 0.005
        )
        incumbent_after = best["variant_id"] if promoted else "family-01-weighted-v8-control"
        promotion = {
            "action": "promote-family-10" if promoted else "retain-family-01-weighted-v8-control",
            "incumbent_before": "family-01-weighted-v8-control",
            "incumbent_after": incumbent_after,
            "promoted": promoted,
            "selected_decomposition_variant": best["variant_id"],
            "rule": (
                "full-1,108-row log loss improvement with paired 95% upper bound below zero, "
                "non-worse Brier, and accuracy no more than 0.5 percentage points lower"
            ),
        }
        development_final = {
            "incumbent_id": incumbent_after,
            "candidate_prediction_identity": (
                prediction_identities[best["variant_id"]] if promoted else prediction_identities["direct-incumbent-control"]
            ),
            "development_metrics": best["metrics"] if promoted else control_metrics,
            "sealed": False,
            "gate_access_count": 0,
        }
    except Exception as exc:
        status = "failed"
        error_path = artifact_root / "terminal-error.txt"
        error_path.write_text(f"{type(exc).__name__}: {exc}\n", encoding="utf-8", newline="\n")
        terminal_failure = {
            "type": type(exc).__name__,
            "message": str(exc),
            "stderr_path": error_path.relative_to(artifact_root).as_posix(),
            "stderr_sha256": file_sha256(error_path),
            "retry": False,
        }
        if attempts and attempts[-1].get("state") == "launched":
            launched = attempts[-1]
            attempts.append({
                "attempt_ordinal": launched["attempt_ordinal"],
                "fold": launched["fold"],
                "component_id": launched["component_id"],
                "state": "exited",
                "exit_code": 1,
                "ended_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
                "production_lease_id": lease["lease_id"],
                "retry": False,
            })
        control_metrics = None
        paired_intervals = {}
        promotion = {
            "action": "retain-family-01-weighted-v8-control",
            "incumbent_before": "family-01-weighted-v8-control",
            "incumbent_after": "family-01-weighted-v8-control",
            "promoted": False,
            "selected_decomposition_variant": None,
            "rule": "terminal component failure cannot promote",
        }
        development_final = {
            "incumbent_id": "family-01-weighted-v8-control",
            "candidate_prediction_identity": None,
            "development_metrics": None,
            "sealed": False,
            "gate_access_count": 0,
        }

    _write_jsonl(run_root / "attempts.jsonl", attempts)
    write_canonical_json(artifact_root / "component-lineage.json", component_lineage)
    result = {
        "experiment_id": EXPERIMENT_ID,
        "status": status,
        "terminal_failure": terminal_failure,
        "variant_results": variant_results,
        "component_fit_lineage_and_support": component_lineage,
        "component_train_outer_gaps": component_train_outer_gaps,
        "component_prediction_identities": {
            component_id: [component_lineage[str(year)][component_id]["outer_prediction_identity"]
                           for year in (2022, 2023, 2024, 2025)
                           if str(year) in component_lineage]
            for component_id in COMPONENT_IDS
        },
        "combined_prediction_identities": prediction_identities,
        "control_metrics": control_metrics,
        "paired_event_block_intervals": paired_intervals,
        "promotion_decision": promotion,
        "development_final_incumbent_identity": development_final,
        "development_safe_population": development_population,
        "comparison_scope": {
            "outer_years": [2022, 2023, 2024, 2025],
            "outer_row_count": 1_108,
            "family_1_comparable": True,
            "family_9_predictions_used": False,
            "development_only": True,
        },
        "gate_access_count": 0,
    }
    write_canonical_json(artifact_root / "result.json", result)
    safety = {
        "database_access": {"used": False, "sql": None, "urls": []},
        "production_lease_count": 1,
        "production_process_count": 1,
        "gpu_fit_count": 0,
        "component_fit_launch_count": sum(row["state"] == "launched" for row in attempts),
        "retry_count": 0,
        "serialized": True,
        "gate_access_count": 0,
        "retired_label_reads": 0,
        **development_population,
    }
    write_canonical_json(artifact_root / "safety.json", safety)
    ended_at = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    write_canonical_json(artifact_root / "production-lease-released.json", {
        "lease_id": lease["lease_id"],
        "ordinal": 1,
        "pid": lease["pid"],
        "state": "released",
        "ended_at": ended_at,
    })
    write_canonical_json(artifact_root / "runtime.json", {
        "started_at": started_at,
        "ended_at": ended_at,
        "runtime_seconds": time.time() - datetime.fromisoformat(started_at.replace("Z", "+00:00")).timestamp(),
        "production_lease_id": lease["lease_id"],
        "serialized": True,
        "retry_count": 0,
        "component_fit_count": sum(row["state"] == "launched" for row in attempts),
    })
    (run_root / "decision.md").write_text(
        "# Family 10 development-only decision\n\n"
        f"{promotion['action']}. The historically exposed 2026 cohort remains retired; "
        "the software gate remains closed with zero access.\n",
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
        "outer_label_component_fit_count": 0,
        "database_access": {"used": False, "sql": None, "urls": []},
    }
    write_canonical_json(manifest_path, manifest)
    registry = _append_registry(campaign_root, {
        "artifact_path": ARTIFACT_PATH,
        "artifact_tree_sha256": inventory.tree_sha256,
        "experiment_id": EXPERIMENT_ID,
        "kind": "family",
        "manifest_path": f"{RUN_PATH}/manifest.json",
        "manifest_sha256": canonical_sha256(manifest),
        "profile_path": PROFILE_PATH,
        "profile_sha256": canonical_sha256(profile),
        "status": status,
    })
    return {**result, **registry, "artifact_tree_sha256": inventory.tree_sha256}


def _main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--campaign", type=Path, required=True)
    parser.add_argument("--source-revision", required=True)
    parser.add_argument("--preregistration-commit", required=True)
    args = parser.parse_args()
    result = materialize_family_10(
        args.campaign,
        source_revision=args.source_revision,
        preregistration_commit=args.preregistration_commit,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["status"] == "complete" else 1


if __name__ == "__main__":
    raise SystemExit(_main())
