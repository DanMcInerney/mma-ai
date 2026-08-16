"""Family 7 matchup-interaction and fighter-swap preregistration."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Mapping

from ..feature_lineage import build_development_safe_ids
from ..hashing import (
    canonical_sha256,
    file_sha256,
    read_json,
    tree_inventory,
    write_canonical_json,
)
from ..matchup_geometry import validate_preregistered_matchup_profiles
from ..protocol import AccessLedger


EXPERIMENT_ID = "family-07-matchup-swap-geometry"
RUN_ALIAS = "family-07-matchup-geometry"
FROZEN_SPEC_SHA256 = "93FB5CC31AD810B1867FFC8A250DD257AAF74732998D103D56AB8D3A2D309A23"
FROZEN_SOURCE_SHA256 = "157649B780965ECC585F18B3030199CDC0F4FE3013958FFA4095FCF665FDB1EA"
RUN_PATH = "runs/family-07-matchup-geometry"
ARTIFACT_PATH = "artifacts/08-family-07-matchup-geometry"
DATA_PATH = "data/experiments/top10_20260815/family-07-matchup-geometry"
FIXED_FAMILY_6_ARTIFACT = Path(
    r"C:\Users\danhm\mma-ai\worktrees\top10-20260815"
    r"\experiments\top10_20260815\artifacts\07-family-06-fighter-states"
)

PROFILE_IDS = (
    "retained-incumbent-control",
    "striking-offense-versus-defense",
    "grappling-weakness-versus-exploitation",
    "opponent-style-matchmaking-tendency",
    "damage-versus-durability",
    "all-directional-interactions",
    "fighter-swap-averaged-prediction",
    "hard-complementary-antisymmetric-geometry",
)

INTERACTION_DEFINITIONS = (
    {
        "name": "striking_offense_vs_defense",
        "left": "high_count_striking_state",
        "right": "striking_defense_state",
        "formula": "cross-difference",
        "minimum_support": 2.0,
        "fallback": 0.0,
        "swap_rule": "negate",
        "lineage": "both-fighters-prior-only",
    },
    {
        "name": "grappling_exploitation_vs_weakness",
        "left": "grappling_exploitation_state",
        "right": "grappling_weakness_state",
        "formula": "cross-difference",
        "minimum_support": 2.0,
        "fallback": 0.0,
        "swap_rule": "negate",
        "lineage": "both-fighters-prior-only",
    },
    {
        "name": "style_matchmaking_tendency",
        "left": "opponent_style_success_state",
        "right": "style_susceptibility_state",
        "formula": "cross-difference",
        "minimum_support": 3.0,
        "fallback": 0.0,
        "swap_rule": "negate",
        "lineage": "both-fighters-prior-only",
    },
    {
        "name": "damage_vs_durability",
        "left": "damage_output_state",
        "right": "durability_weakness_state",
        "formula": "cross-difference",
        "minimum_support": 2.0,
        "fallback": 0.0,
        "swap_rule": "negate",
        "lineage": "both-fighters-prior-only",
    },
)


def build_preregistered_profile() -> dict[str, Any]:
    interaction_names = [item["name"] for item in INTERACTION_DEFINITIONS]
    groups = (
        (),
        (interaction_names[0],),
        (interaction_names[1],),
        (interaction_names[2],),
        (interaction_names[3],),
        tuple(interaction_names),
        tuple(interaction_names),
        tuple(interaction_names),
    )
    geometries = (
        "original-only-control",
        "original-only",
        "original-only",
        "original-only",
        "original-only",
        "original-only",
        "original-swapped-average",
        "hard-complementary-antisymmetric",
    )
    profiles = []
    for profile_id, names, geometry in zip(PROFILE_IDS, groups, geometries, strict=True):
        names = list(names)
        profiles.append(
            {
                "id": profile_id,
                "interaction_names": names,
                "prediction_geometry": geometry,
                "ordered_interaction_sha256": canonical_sha256(
                    {"interaction_names": names, "prediction_geometry": geometry}
                ),
            }
        )
    profile = {
        "experiment_id": EXPERIMENT_ID,
        "family_number": 7,
        "frozen_spec_sha256": FROZEN_SPEC_SHA256,
        "frozen_source": {
            "path": "artifacts/01-campaign-harness/frozen/training_data.csv",
            "sha256": FROZEN_SOURCE_SHA256,
            "cutoff": "2025-12-31",
            "development_safe_id_count": 3_089,
            "retired_id_count": 178,
            "development_max_date": "2025-12-13",
        },
        "dependency": {
            "experiment_id": "family-06-multiscale-count-aware-state",
            "run_alias": "family-06-fighter-states",
            "required_data_path": "data/experiments/top10_20260815/family-06-fighter-states/matched-state-table.csv",
            "fallback": "terminal-failure-before-row-decode",
        },
        "role_swap": {
            "paired_metadata": ["id", "name", "url", "label", "odds", "market_probability"],
            "paired_payloads": ["features", "lineage"],
            "target_rule": "binary-complement",
            "antisymmetric_rule": "negate",
        },
        "interaction_definitions": [dict(item) for item in INTERACTION_DEFINITIONS],
        "normalization": {"fit_scope": "outer-train-only", "method": "training-median-standard-scale"},
        "profiles": profiles,
        "outer_years": [2022, 2023, 2024, 2025],
        "inner_validation_year_count": 3,
        "selection": {
            "fit_scope": "prior-inner-only",
            "score": "mean-inner-log-loss",
            "tie_break": list(PROFILE_IDS),
            "outer_label_selection_count": 0,
        },
        "geometry": {
            "probability_sum": 1.0,
            "tolerance": 1e-12,
            "store_original_and_swapped": True,
            "store_invariance_residual": True,
        },
        "database_access": {"used": False, "sql": None, "urls": []},
        "invocation": {"gpu_lease_count": 1, "retry_count": 0, "serialized": True},
    }
    validate_preregistered_matchup_profiles(profile)
    return profile


def write_preregistration(campaign_root: Path, *, source_revision: str) -> dict[str, Any]:
    """Persist the exact eight-profile menu while score destinations are absent."""

    campaign_root = Path(campaign_root)
    profile_path = campaign_root / "profiles/family-07-matchup-geometry.json"
    preregistration_path = campaign_root / RUN_PATH / "preregistration.json"
    artifact_root = campaign_root / ARTIFACT_PATH
    data_root = campaign_root.parents[1] / DATA_PATH
    if any(path.exists() for path in (profile_path, preregistration_path, artifact_root, data_root)):
        raise ValueError("family 7 preregistration destinations must all be absent")
    gate = AccessLedger(campaign_root).gate_status()
    if gate["state"] != "closed" or gate["protected_access_count"] != 0:
        raise ValueError("family 7 preregistration requires the gate closed with zero access")
    profile = build_preregistered_profile()
    write_canonical_json(profile_path, profile)
    preregistration = {
        "experiment_id": EXPERIMENT_ID,
        "family_number": 7,
        "source_revision": source_revision,
        "frozen_spec_sha256": FROZEN_SPEC_SHA256,
        "profile_path": "profiles/family-07-matchup-geometry.json",
        "profile_sha256": canonical_sha256(profile),
        "profile_file_sha256": file_sha256(profile_path),
        "preregistered_profile_ids": list(PROFILE_IDS),
        "ordered_profile_hashes": {
            item["id"]: item["ordered_interaction_sha256"] for item in profile["profiles"]
        },
        "registry_prefix_sha256_before": hashlib.sha256(
            (campaign_root / "registry.jsonl").read_bytes()
        ).hexdigest().upper(),
        "scoring_state": "not-started",
        "selection": profile["selection"],
        "database_access": profile["database_access"],
        "invocation": profile["invocation"],
        "gate_required_state": "closed-zero-access",
        "terminal_failure_rule": "Any dependency, safe-population, role-swap, lineage, support, geometry, safety, or destination mismatch terminates without retry.",
    }
    write_canonical_json(preregistration_path, preregistration)
    return preregistration


def _write_jsonl(path: Path, rows: list[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(
        b"".join(
            json.dumps(dict(row), sort_keys=True, separators=(",", ":")).encode() + b"\n"
            for row in rows
        )
    )


def _append_registry(campaign_root: Path, payload: Mapping[str, Any]) -> dict[str, str]:
    registry_path = campaign_root / "registry.jsonl"
    head_path = campaign_root / "registry-head.json"
    before = registry_path.read_bytes()
    records = [json.loads(line) for line in before.splitlines()]
    if any(record["payload"]["experiment_id"] == EXPERIMENT_ID for record in records):
        raise ValueError("family 7 already exists in the registry")
    head = read_json(head_path)
    prefix_before = hashlib.sha256(before).hexdigest().upper()
    if prefix_before != head["registry_prefix_sha256"] or len(records) != 7:
        raise ValueError("registry head does not match the immutable family-6 prefix")
    record = {
        "payload": dict(payload),
        "prefix_sha256_before": prefix_before,
        "previous_record_sha256": head["last_record_sha256"],
        "sequence": head["record_count"],
    }
    record["record_sha256"] = canonical_sha256(record)
    after = before + json.dumps(record, sort_keys=True, separators=(",", ":")).encode() + b"\n"
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


def materialize_family_07(
    campaign_root: Path,
    *,
    source_revision: str,
    preregistration_commit: str,
) -> dict[str, Any]:
    """Record the sole terminal attempt without decoding rows when family 6 has no state table."""

    campaign_root = Path(campaign_root)
    run_root = campaign_root / RUN_PATH
    artifact_root = campaign_root / ARTIFACT_PATH
    manifest_path = run_root / "manifest.json"
    data_root = campaign_root.parents[1] / DATA_PATH
    if artifact_root.exists() or data_root.exists() or manifest_path.exists():
        raise ValueError("family 7 score destination already exists; retries are forbidden")
    inherited_database = [
        key for key in ("DATABASE_URL", "ODDS_DATABASE_URL") if os.environ.get(key)
    ]
    if inherited_database:
        raise ValueError("family 7 refuses inherited database URLs")
    gate = AccessLedger(campaign_root).gate_status()
    if gate["state"] != "closed" or gate["protected_access_count"] != 0:
        raise ValueError("family 7 requires the gate closed with zero access")
    profile_path = campaign_root / "profiles/family-07-matchup-geometry.json"
    preregistration_path = run_root / "preregistration.json"
    profile = read_json(profile_path)
    preregistration = read_json(preregistration_path)
    validate_preregistered_matchup_profiles(profile)
    registry_before = hashlib.sha256((campaign_root / "registry.jsonl").read_bytes()).hexdigest().upper()
    if (
        profile != build_preregistered_profile()
        or preregistration["scoring_state"] != "not-started"
        or preregistration["profile_file_sha256"] != file_sha256(profile_path)
        or preregistration["profile_sha256"] != canonical_sha256(profile)
        or preregistration["registry_prefix_sha256_before"] != registry_before
    ):
        raise ValueError("family 7 was not exactly preregistered before score")

    fold_manifest = read_json(campaign_root / "baseline/fold-manifest.json")
    safe_ids, retired_ids = build_development_safe_ids(fold_manifest)
    development_max_date = str(fold_manifest["folds"][-1]["outer"]["test_date_range"][1])
    if len(safe_ids) != 3_089 or len(retired_ids) != 178 or development_max_date != "2025-12-13":
        raise ValueError("family 7 development-safe population differs before row decode")
    development_population = {
        "asserted_before_row_or_target_decode": True,
        "development_safe_id_count": len(safe_ids),
        "development_max_date": development_max_date,
        "retired_id_count": len(retired_ids),
    }

    artifact_root.mkdir(parents=True)
    acquired = {
        "lease_id": "family-07-serialized-lease-1",
        "ordinal": 1,
        "pid": os.getpid(),
        "state": "acquired",
    }
    write_canonical_json(artifact_root / "gpu-lease-acquired.json", acquired)

    dependency_manifest = read_json(campaign_root / "runs/family-06-fighter-states/manifest.json")
    dependency_inventory = tree_inventory(FIXED_FAMILY_6_ARTIFACT)
    dependency_available = (
        dependency_manifest["exit_state"] == "complete"
        and dependency_manifest["data_path"] is not None
        and dependency_manifest["outer_prediction_identities"]
        and dependency_inventory.tree_sha256 == dependency_manifest["artifact_tree_sha256"]
    )
    if dependency_available:
        raise ValueError("family 7 scorer is not authorized for an unregistered dependency state")
    if (
        dependency_manifest["exit_state"] != "failed"
        or dependency_manifest["data_path"] is not None
        or dependency_manifest["outer_prediction_identities"]
        or dependency_inventory.tree_sha256 != dependency_manifest["artifact_tree_sha256"]
    ):
        raise ValueError("family 6 dependency evidence differs from the frozen terminal failure")
    dependency_evidence = {
        "experiment_id": dependency_manifest["experiment_id"],
        "exit_state": dependency_manifest["exit_state"],
        "artifact_tree_sha256": dependency_inventory.tree_sha256,
        "artifact_file_count": dependency_inventory.file_count,
        "data_path": dependency_manifest["data_path"],
        "outer_prediction_identities": dependency_manifest["outer_prediction_identities"],
        "adaptive_signal": dependency_manifest["adaptive_signal_for_family_07"],
    }
    failure_text = (
        "Family 7 terminated before row or target decode: the immutable family 6 dependency "
        "failed before construction and produced no matched fighter-state table or outer predictions."
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
    swap_evidence = {
        "status": "unavailable",
        "reason": "pre-construction-dependency-failure",
        "registered_mapping": profile["role_swap"],
        "original_prediction_count": 0,
        "swapped_prediction_count": 0,
        "invariance_residual_count": 0,
        "complementarity_failure_count": 0,
    }
    promotion = {
        "action": "retain-family-01-weighted-v8-control",
        "incumbent_before": "family-01-weighted-v8-control",
        "incumbent_after": "family-01-weighted-v8-control",
        "promoted": False,
        "rule": "failed candidates cannot be promoted",
    }
    adaptive_signal = {
        "status": "family-07-failed-before-matchup-construction",
        "selected_profiles": [],
        "selected_interaction_hashes": [],
        "swap_geometry_available": False,
    }
    result = {
        "experiment_id": EXPERIMENT_ID,
        "status": "failed",
        "terminal_failure": terminal_failure,
        "metrics": None,
        "paired_event_block_intervals": None,
        "slice_metrics": None,
        "symmetry_diagnostics": None,
        "swap_mapping_and_invariance_evidence": swap_evidence,
        "outer_original_prediction_identities": [],
        "outer_swapped_prediction_identities": [],
        "promotion_decision": promotion,
        "adaptive_signal_for_family_08": adaptive_signal,
        "development_safe_population": development_population,
        "dependency_evidence": dependency_evidence,
        "gate_access_count": gate["protected_access_count"],
    }
    write_canonical_json(artifact_root / "failure.json", terminal_failure)
    write_canonical_json(artifact_root / "dependency-evidence.json", dependency_evidence)
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
    write_canonical_json(
        artifact_root / "gpu-lease-released.json",
        {
            "lease_id": acquired["lease_id"],
            "ordinal": 1,
            "pid": acquired["pid"],
            "state": "released-after-terminal-failure",
        },
    )
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
                "retry": False,
            }
        ],
    )
    (run_root / "decision.md").write_text(
        "# Family 7 decision\n\nRetain family 1: the sole attempt stopped before row decode because family 6 produced no fighter-state table.\n",
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
        "profile_path": "profiles/family-07-matchup-geometry.json",
        "profile_sha256": canonical_sha256(profile),
        "profile_file_sha256": file_sha256(profile_path),
        "preregistration_path": f"{RUN_PATH}/preregistration.json",
        "preregistration_sha256": canonical_sha256(preregistration),
        "preregistration_commit": preregistration_commit,
        "attempts_path": f"{RUN_PATH}/attempts.jsonl",
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
