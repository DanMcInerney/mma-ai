"""Experiment-zero construction from immutable, byte-copied evidence."""

from __future__ import annotations

import csv
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import pandas as pd

from .hashing import (
    canonical_sha256,
    file_sha256,
    tree_inventory,
    write_canonical_json,
)
from .protocol import AccessLedger, GATE_END, GATE_START, build_development_folds, initialize_gate
from .registry import (
    CAMPAIGN_FAMILY_IDS,
    RESOLVED_PROFILE_FIELDS,
    append_registry_record,
    initialize_registry,
    validate_registry,
    validate_resolved_profile,
)


SAFE_ROSTER_COLUMNS = ("fight_id", "event_id", "event_date")


@dataclass(frozen=True)
class BaselineSources:
    frozen_csv: Path
    accepted_model: Path
    no_recency_model: Path
    accepted_evidence: Path
    no_recency_evidence: Path


@dataclass(frozen=True)
class BootstrapResult:
    experiment_id: str
    baseline_manifest_sha256: str
    fold_manifest_sha256: str
    artifact_tree_sha256: str
    registry_prefix_sha256: str


def _source_paths(sources: BaselineSources) -> dict[str, Path]:
    return {
        "frozen_csv": sources.frozen_csv,
        "accepted_evidence/direct-evaluation.json": sources.accepted_evidence / "training" / "direct-evaluation.json"
        if (sources.accepted_evidence / "training" / "direct-evaluation.json").exists()
        else sources.accepted_evidence / "direct-evaluation.json",
        "accepted_evidence/final-reverification.md": sources.accepted_evidence / "final-reverification.md",
        "no_recency_evidence/direct-evaluation.json": sources.no_recency_evidence / "training" / "direct-evaluation.json"
        if (sources.no_recency_evidence / "training" / "direct-evaluation.json").exists()
        else sources.no_recency_evidence / "direct-evaluation.json",
        "no_recency_evidence/final-verification.md": sources.no_recency_evidence / "final-verification.md",
    }


def _preflight_sources(
    sources: BaselineSources,
    expected_source_hashes: Mapping[str, str],
) -> dict[str, Any]:
    if not sources.frozen_csv.is_file():
        raise ValueError(f"missing frozen CSV: {sources.frozen_csv}")
    for directory in (
        sources.accepted_model,
        sources.no_recency_model,
        sources.accepted_evidence,
        sources.no_recency_evidence,
    ):
        if not directory.is_dir():
            raise ValueError(f"missing baseline source directory: {directory}")
    concrete = _source_paths(sources)
    actual_hashes = {}
    for name, expected in expected_source_hashes.items():
        if name not in concrete or not concrete[name].is_file():
            raise ValueError(f"missing fixed source identity: {name}")
        actual = file_sha256(concrete[name])
        if actual != expected.upper():
            raise ValueError(f"source hash mismatch for {name}: {actual}")
        actual_hashes[name] = actual
    return {
        "fixed_file_hashes": actual_hashes,
        "accepted_model": tree_inventory(sources.accepted_model).as_dict(),
        "no_recency_model": tree_inventory(sources.no_recency_model).as_dict(),
        "accepted_evidence": tree_inventory(sources.accepted_evidence).as_dict(),
        "no_recency_evidence": tree_inventory(sources.no_recency_evidence).as_dict(),
    }


def _copy_sources(sources: BaselineSources, artifact_root: Path) -> None:
    if artifact_root.exists():
        raise ValueError(f"artifact destination already exists: {artifact_root}")
    artifact_root.parent.mkdir(parents=True, exist_ok=True)
    (artifact_root / "frozen").mkdir(parents=True)
    shutil.copy2(sources.frozen_csv, artifact_root / "frozen" / "training_data.csv")
    shutil.copytree(sources.accepted_model, artifact_root / "models" / "accepted", copy_function=shutil.copy2)
    shutil.copytree(sources.no_recency_model, artifact_root / "models" / "no-recency", copy_function=shutil.copy2)
    shutil.copytree(sources.accepted_evidence, artifact_root / "evidence" / "accepted", copy_function=shutil.copy2)
    shutil.copytree(sources.no_recency_evidence, artifact_root / "evidence" / "no-recency", copy_function=shutil.copy2)


def _verify_copy_component(source_inventory: Mapping[str, Any], artifact_inventory: Mapping[str, Any], prefix: str) -> None:
    source_files = {
        entry["path"]: (entry["bytes"], entry["sha256"])
        for entry in source_inventory["files"]
    }
    copy_files = {
        entry["path"].removeprefix(prefix): (entry["bytes"], entry["sha256"])
        for entry in artifact_inventory["files"]
        if entry["path"].startswith(prefix)
    }
    if source_files != copy_files:
        raise ValueError(f"artifact copy mismatch for {prefix}")


def _safe_roster(filtered_training_csv: Path) -> list[dict[str, str]]:
    # The parser projects identifiers/dates only. It never requests or exposes y_true.
    frame = pd.read_csv(filtered_training_csv, usecols=list(SAFE_ROSTER_COLUMNS), dtype="string")
    frame["event_date"] = pd.to_datetime(frame["event_date"], errors="raise")
    if frame[list(SAFE_ROSTER_COLUMNS)].isna().any().any():
        raise ValueError("filtered baseline roster has missing identifiers/dates")
    if frame["fight_id"].duplicated().any():
        raise ValueError("filtered baseline roster has duplicate fight IDs")
    frame = frame.sort_values(["event_date", "event_id", "fight_id"])
    return [
        {
            "fight_id": str(row.fight_id),
            "event_id": str(row.event_id),
            "event_date": row.event_date.date().isoformat(),
        }
        for row in frame.itertuples(index=False)
    ]


def _csv_schema(path: Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        header = next(csv.reader(handle))
    with Path(path).open("rb") as handle:
        row_count = sum(1 for _ in handle) - 1
    return {
        "column_count": len(header),
        "columns_sha256": canonical_sha256(header),
        "row_count": row_count,
    }


def _population_counts(roster: Sequence[Mapping[str, str]]) -> dict[str, int]:
    dates = [pd.Timestamp(row["event_date"]).date() for row in roster]
    return {
        "total": len(dates),
        "pre_2025": sum(value < pd.Timestamp("2025-01-01").date() for value in dates),
        "from_2025": sum(value >= pd.Timestamp("2025-01-01").date() for value in dates),
        "gate": sum(GATE_START <= value <= GATE_END for value in dates),
    }


def bootstrap_experiment_zero(
    campaign_root: Path,
    artifact_root: Path,
    *,
    sources: BaselineSources,
    source_revision: str,
    working_profile: Mapping[str, Any],
    no_recency_profile: Mapping[str, Any],
    expected_population: Mapping[str, int],
    expected_source_hashes: Mapping[str, str],
    expected_model_identities: Mapping[str, Any],
) -> BootstrapResult:
    campaign_root = Path(campaign_root)
    artifact_root = Path(artifact_root)
    if len(source_revision) != 40:
        raise ValueError("source revision must be a full Git object ID")
    if campaign_root.exists():
        raise ValueError(f"campaign destination already exists: {campaign_root}")
    if artifact_root.exists():
        raise ValueError(f"artifact destination already exists: {artifact_root}")

    weighted_profile = dict(working_profile)
    no_weight_profile = dict(no_recency_profile)
    weighted_profile["features"] = list(weighted_profile["features"])
    no_weight_profile["features"] = list(no_weight_profile["features"])
    weighted_profile_sha = validate_resolved_profile(weighted_profile)
    no_weight_profile_sha = validate_resolved_profile(no_weight_profile)
    source_inventory = _preflight_sources(sources, expected_source_hashes)
    roster = _safe_roster(sources.accepted_model / "training_data.csv")
    observed_population = _population_counts(roster)
    if observed_population != dict(expected_population):
        raise ValueError(
            f"filtered population mismatch: observed={observed_population}, expected={dict(expected_population)}"
        )
    folds = build_development_folds(roster)

    _copy_sources(sources, artifact_root)
    artifact_inventory = tree_inventory(artifact_root).as_dict()
    artifact_inventory["root"] = artifact_root.relative_to(campaign_root).as_posix()
    _verify_copy_component(source_inventory["accepted_model"], artifact_inventory, "models/accepted/")
    _verify_copy_component(source_inventory["no_recency_model"], artifact_inventory, "models/no-recency/")
    _verify_copy_component(source_inventory["accepted_evidence"], artifact_inventory, "evidence/accepted/")
    _verify_copy_component(source_inventory["no_recency_evidence"], artifact_inventory, "evidence/no-recency/")
    frozen_copy_hash = file_sha256(artifact_root / "frozen" / "training_data.csv")
    source_csv_hash = file_sha256(sources.frozen_csv)
    if frozen_copy_hash != source_csv_hash:
        raise ValueError("frozen CSV copy hash mismatch")

    campaign_root.mkdir(parents=True, exist_ok=True)
    # Registry/manifests are canonical byte streams; disable checkout newline rewriting.
    (campaign_root / ".gitattributes").write_bytes(b"* -text\n")
    initialize_gate(campaign_root, expected_family_ids=CAMPAIGN_FAMILY_IDS)
    ledger = AccessLedger(campaign_root)
    roster_dates = [pd.Timestamp(row["event_date"]).date() for row in roster]
    ledger.record(
        purpose="baseline-fold-and-gate-membership",
        columns=SAFE_ROSTER_COLUMNS,
        min_date=min(roster_dates),
        max_date=max(roster_dates),
        protected_gate_labels=False,
    )

    profiles_dir = campaign_root / "profiles"
    weighted_profile_path = profiles_dir / "experiment-zero.json"
    no_recency_profile_path = profiles_dir / "experiment-zero-no-recency.json"
    write_canonical_json(weighted_profile_path, weighted_profile)
    write_canonical_json(no_recency_profile_path, no_weight_profile)

    gate_roster = [
        row
        for row in roster
        if GATE_START <= pd.Timestamp(row["event_date"]).date() <= GATE_END
    ]
    fold_manifest = {
        "protocol": "expanding-whole-event-calendar-year",
        "years": [2022, 2023, 2024, 2025],
        "embargo_days": 7,
        "safe_columns": list(SAFE_ROSTER_COLUMNS),
        "label_columns_accessed": [],
        "population_fight_ids": [row["fight_id"] for row in roster],
        "gate_roster": gate_roster,
        "folds": [fold.as_dict() for fold in folds],
    }
    fold_manifest_path = campaign_root / "baseline" / "fold-manifest.json"
    fold_manifest_sha = write_canonical_json(fold_manifest_path, fold_manifest)

    feature_order = weighted_profile["features"]
    feature_sha = canonical_sha256(feature_order)
    gate_state = ledger.gate_status()
    baseline_manifest = {
        "experiment_id": "experiment-zero",
        "campaign_id": "top10_20260815",
        "source_revision": source_revision,
        "dirty_state": "clean",
        "profiles": {
            "working": {
                "path": weighted_profile_path.relative_to(campaign_root).as_posix(),
                "field_count": len(RESOLVED_PROFILE_FIELDS),
                "sha256": weighted_profile_sha,
            },
            "no_recency": {
                "path": no_recency_profile_path.relative_to(campaign_root).as_posix(),
                "field_count": len(RESOLVED_PROFILE_FIELDS),
                "sha256": no_weight_profile_sha,
            },
        },
        "features": {
            "count": len(feature_order),
            "ordered_names": feature_order,
            "ordered_sha256": feature_sha,
        },
        "population": observed_population,
        "frozen_csv": {
            "source_path": str(sources.frozen_csv.resolve()),
            "copy_path": "artifacts/01-campaign-harness/frozen/training_data.csv",
            "sha256": source_csv_hash,
            "schema": _csv_schema(sources.frozen_csv),
        },
        "fold_manifest": {
            "path": fold_manifest_path.relative_to(campaign_root).as_posix(),
            "sha256": fold_manifest_sha,
            "years": [2022, 2023, 2024, 2025],
            "outer_counts": [len(fold.test_ids) for fold in folds],
        },
        "model_identities": dict(expected_model_identities),
        "model_graph": {
            "weighted_original": {"Mitra": 1.0},
            "no_recency_original": {"Mitra": 8 / 9, "XGBoost": 1 / 9},
            "selection_boundary": "Original",
            "full_metrics_select": False,
        },
        "runtime": {
            "autogluon": "1.6.1",
            "python": "3.12.4",
            "torch": "2.10.0+cu130",
            "cuda": "13.0",
            "device": "NVIDIA GeForce RTX 5090 Laptop GPU",
        },
        "invocation": {
            "baseline_profile": "v8-hybrid-weighted",
            "time_limit": 3000,
            "refit_full": True,
            "campaign_training_launched": False,
        },
        "source_inventory": source_inventory,
        "artifact_inventory": artifact_inventory,
        "gate": {
            "gate_id": gate_state["gate_id"],
            "state": gate_state["state"],
            "expected_rows": expected_population["gate"],
            "protected_access_count": gate_state["protected_access_count"],
        },
    }
    baseline_manifest_path = campaign_root / "baseline" / "manifest.json"
    baseline_manifest_sha = write_canonical_json(baseline_manifest_path, baseline_manifest)

    run_manifest = {
        "experiment_id": "experiment-zero",
        "kind": "admission",
        "exit_state": "complete",
        "source_revision": source_revision,
        "profile_sha256": weighted_profile_sha,
        "baseline_manifest_path": baseline_manifest_path.relative_to(campaign_root).as_posix(),
        "baseline_manifest_sha256": baseline_manifest_sha,
        "fold_manifest_sha256": fold_manifest_sha,
        "artifact_path": artifact_root.relative_to(campaign_root).as_posix(),
        "artifact_tree_sha256": artifact_inventory["tree_sha256"],
        "training_launched": False,
        "gate_state": "closed",
    }
    run_manifest_path = campaign_root / "runs" / "experiment-zero" / "manifest.json"
    run_manifest_sha = write_canonical_json(run_manifest_path, run_manifest)
    (run_manifest_path.parent / "decision.md").write_text(
        "# Experiment zero\n\nAdmission only. No family fit or campaign-gate score was launched.\n",
        encoding="utf-8",
    )

    initialize_registry(campaign_root)
    append_registry_record(
        campaign_root,
        {
            "experiment_id": "experiment-zero",
            "kind": "experiment-zero",
            "status": "complete",
            "profile_path": weighted_profile_path.relative_to(campaign_root).as_posix(),
            "profile_sha256": weighted_profile_sha,
            "manifest_path": run_manifest_path.relative_to(campaign_root).as_posix(),
            "manifest_sha256": run_manifest_sha,
            "artifact_path": artifact_root.relative_to(campaign_root).as_posix(),
            "artifact_tree_sha256": artifact_inventory["tree_sha256"],
        },
    )
    return BootstrapResult(
        experiment_id="experiment-zero",
        baseline_manifest_sha256=baseline_manifest_sha,
        fold_manifest_sha256=fold_manifest_sha,
        artifact_tree_sha256=artifact_inventory["tree_sha256"],
        registry_prefix_sha256=validate_registry(
            campaign_root, strict=True
        ).registry_prefix_sha256,
    )
