"""Verification and strict campaign seams for family experiment tracers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, Mapping

from .hashing import canonical_sha256, file_sha256, read_json, tree_inventory
from .metrics import event_block_bootstrap_delta, metric_gap, reduce_predictions
from .protocol import AccessLedger
from .registry import CAMPAIGN_FAMILY_IDS, RegistryError, validate_registry


FIXED_CAMPAIGN_ARTIFACT = Path(
    r"C:\Users\danhm\mma-ai\worktrees\top10-20260815"
    r"\experiments\top10_20260815\artifacts\01-campaign-harness"
)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for line_number, raw in enumerate(Path(path).read_bytes().splitlines(), start=1):
        try:
            record = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid JSONL at {path}:{line_number}") from exc
        records.append(record)
    return records


def _assert_equal(actual: Any, expected: Any, noun: str) -> None:
    if canonical_sha256(actual) != canonical_sha256(expected):
        raise ValueError(f"{noun} does not recompute from fixed artifacts")


def _promotion_replay(manifest: Mapping[str, Any], status: str) -> dict[str, Any]:
    expected = {
        "action": "establish-development-control" if status == "complete" else "retain-experiment-zero",
        "incumbent_before": "experiment-zero",
        "incumbent_after": (
            "family-01-weighted-v8-control" if status == "complete" else "experiment-zero"
        ),
        "frozen_replacement_predicate_applicable": False,
        "reason": (
            "experiment zero has no aligned four-fold development predictions; "
            "family 1 is the preregistered control denominator"
            if status == "complete"
            else "the only authorized control launch ended in a preserved terminal failure"
        ),
    }
    _assert_equal(manifest["promotion_decision"], expected, "promotion decision")
    return expected


def verify_family_run(
    campaign_root: Path,
    experiment_id: str,
    *,
    recompute_all: bool,
) -> dict[str, Any]:
    campaign_root = Path(campaign_root)
    if experiment_id != CAMPAIGN_FAMILY_IDS[0]:
        raise ValueError("this verifier owns only family 1")
    manifest_path = campaign_root / "runs" / experiment_id / "manifest.json"
    manifest = read_json(manifest_path)
    if manifest.get("experiment_id") != experiment_id:
        raise ValueError("run manifest experiment ID mismatch")
    profile = read_json(campaign_root / manifest["profile_path"])
    if canonical_sha256(profile) != manifest["profile_sha256"]:
        raise ValueError("resolved profile hash mismatch")
    artifact_root = campaign_root / manifest["artifact_path"]
    inventory = tree_inventory(artifact_root)
    if inventory.tree_sha256 != manifest["artifact_tree_sha256"]:
        raise ValueError("family artifact tree hash mismatch")
    if inventory.file_count != manifest["artifact_file_count"]:
        raise ValueError("family artifact file count mismatch")

    attempts = read_jsonl(campaign_root / manifest["attempts_path"])
    launched = [record for record in attempts if record.get("state") == "launched"]
    exited = [record for record in attempts if record.get("state") == "exited"]
    if len(launched) != len(exited):
        raise ValueError("every launched fit must have one durable exit record")
    if [row["attempt_id"] for row in launched] != [row["attempt_id"] for row in exited]:
        raise ValueError("attempt launch/exit identities are not aligned")
    gate = AccessLedger(campaign_root).gate_status()
    if gate["state"] != "closed" or gate["protected_access_count"] != 0:
        raise ValueError("family verification requires the gate closed with zero access")

    status = manifest["exit_state"]
    decision = _promotion_replay(manifest, status)
    common = {
        "experiment_id": experiment_id,
        "status": status,
        "gate_access_count": gate["protected_access_count"],
        "artifact_tree_sha256": inventory.tree_sha256,
        "attempt_count": len(launched),
        "promotion_decision": decision,
    }
    if status == "failed":
        failure = manifest["terminal_failure"]
        stderr_path = artifact_root / failure["stderr_path"]
        if file_sha256(stderr_path) != failure["stderr_sha256"]:
            raise ValueError("terminal stderr identity mismatch")
        return {**common, "terminal_failure": failure}
    if status != "complete":
        raise ValueError(f"unsupported family exit state: {status}")

    fold_paths = [artifact_root / item["path"] for item in manifest["fold_predictions"]]
    if [item["year"] for item in manifest["fold_predictions"]] != [2022, 2023, 2024, 2025]:
        raise ValueError("family 1 must contain exactly the four frozen outer folds")
    predictions = [record for path in fold_paths for record in read_jsonl(path)]
    from .families.weighted_v8 import validate_prediction_chronology

    validate_prediction_chronology(predictions)
    metrics = reduce_predictions(predictions).as_dict()
    baseline = read_jsonl(artifact_root / manifest["baseline_predictions_path"])
    candidate_2025 = [row for row in predictions if row["fold"] == "2025"]
    intervals = event_block_bootstrap_delta(
        candidate_2025,
        baseline,
        iterations=manifest["bootstrap"]["iterations"],
        seed=manifest["bootstrap"]["seed"],
    )
    baseline_metrics = reduce_predictions(baseline).as_dict()
    if recompute_all:
        _assert_equal(metrics, manifest["metrics"], "family metrics")
        _assert_equal(intervals, manifest["paired_event_block_intervals"], "paired intervals")
        _assert_equal(baseline_metrics, manifest["historical_baseline_2025_metrics"], "baseline metrics")
    train_gap = manifest["train_outer_gap"]
    if train_gap.get("status") == "computed":
        tuning = read_jsonl(artifact_root / train_gap["source_path"])
        recomputed_gap = metric_gap(reduce_predictions(tuning), reduce_predictions(predictions))
        _assert_equal(recomputed_gap, train_gap["metrics"], "train/outer gap")
    elif train_gap.get("status") != "inadmissible":
        raise ValueError("train/outer gap must be computed or explicitly inadmissible")
    return {
        **common,
        "outer_years": [2022, 2023, 2024, 2025],
        "fold_prediction_count": len(fold_paths),
        "metrics": metrics,
        "historical_baseline_2025_metrics": baseline_metrics,
        "paired_event_block_intervals": intervals,
        "train_outer_gap": train_gap,
    }


def validate_terminal_campaign(
    campaign_root: Path,
    *,
    expect_terminal_through: int | None,
    require_gate_closed: bool,
) -> dict[str, Any]:
    campaign_root = Path(campaign_root)
    registry = validate_registry(campaign_root, strict=True)
    if expect_terminal_through is not None:
        expected_ids = CAMPAIGN_FAMILY_IDS[:expect_terminal_through]
        if registry.family_ids != expected_ids or registry.record_count != expect_terminal_through + 1:
            raise RegistryError("registry does not contain the exact terminal family prefix")
    baseline = read_json(campaign_root / "baseline" / "manifest.json")
    fold_path = campaign_root / baseline["fold_manifest"]["path"]
    if canonical_sha256(read_json(fold_path)) != baseline["fold_manifest"]["sha256"]:
        raise RegistryError("baseline fold manifest changed")
    local_artifact = campaign_root / baseline["artifact_inventory"]["root"]
    artifact_root = local_artifact if local_artifact.is_dir() else FIXED_CAMPAIGN_ARTIFACT
    inventory = tree_inventory(artifact_root)
    if inventory.tree_sha256 != baseline["artifact_inventory"]["tree_sha256"]:
        raise RegistryError("fixed baseline artifact tree changed")
    gate = AccessLedger(campaign_root).gate_status()
    if require_gate_closed and (gate["state"] != "closed" or gate["protected_access_count"] != 0):
        raise RegistryError("gate is not closed with zero protected access")
    return {
        "record_count": registry.record_count,
        "family_ids": list(registry.family_ids),
        "registry_prefix_sha256": registry.registry_prefix_sha256,
        "baseline_artifact_tree_sha256": inventory.tree_sha256,
        "gate_state": gate["state"],
        "protected_gate_access_count": gate["protected_access_count"],
    }
