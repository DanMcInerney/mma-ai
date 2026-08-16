"""Verification and strict campaign seams for family experiment tracers."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Iterable, Mapping

from .hashing import canonical_sha256, file_sha256, read_json, tree_inventory
from .metrics import event_block_bootstrap_delta, metric_gap, reduce_predictions
from .protocol import AccessLedger
from .registry import (
    CAMPAIGN_FAMILY_IDS,
    FAMILY_2_VARIANT_IDS,
    RegistryError,
    validate_registry,
    validate_resolved_profile,
)


FIXED_CAMPAIGN_ARTIFACT = Path(
    r"C:\Users\danhm\mma-ai\worktrees\top10-20260815"
    r"\experiments\top10_20260815\artifacts\01-campaign-harness"
)
FIXED_FAMILY_1_ARTIFACT = Path(
    r"C:\Users\danhm\mma-ai\worktrees\top10-20260815"
    r"\experiments\top10_20260815\artifacts\02-family-01-weighted-v8-control"
)
FIXED_FAMILY_2_ARTIFACT = Path(
    r"C:\Users\danhm\mma-ai\worktrees\top10-20260815"
    r"\experiments\top10_20260815\artifacts\03-family-02-horizon-recency"
)
FROZEN_REGISTRY_PREFIX_BEFORE_FAMILY_3 = (
    "C5F8E37AEC82E0AEFDAAE6EECF7A89E55EFDC04788884FFA504105F131C752BB"
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


def _validate_campaign_registry(campaign_root: Path):
    registry_path = campaign_root / "registry.jsonl"
    raw_lines = registry_path.read_bytes().splitlines(keepends=True)
    if len(raw_lines) <= 3:
        return validate_registry(campaign_root, strict=True)
    records = [json.loads(line) for line in raw_lines]
    expected_ids = ["experiment-zero", *CAMPAIGN_FAMILY_IDS[:3]]
    if [row["payload"]["experiment_id"] for row in records] != expected_ids:
        raise RegistryError("registry does not contain the exact completed family-3 prefix")
    if hashlib.sha256(b"".join(raw_lines[:3])).hexdigest().upper() != FROZEN_REGISTRY_PREFIX_BEFORE_FAMILY_3:
        raise RegistryError("the frozen registry prefix before family 3 changed")
    previous_record = "0" * 64
    prefix_bytes = b""
    for sequence, (raw, record) in enumerate(zip(raw_lines, records, strict=True)):
        expected_prefix = hashlib.sha256(prefix_bytes).hexdigest().upper()
        if (
            record["sequence"] != sequence
            or record["prefix_sha256_before"] != expected_prefix
            or record["previous_record_sha256"] != previous_record
        ):
            raise RegistryError("registry chain or prefix identity mismatch")
        recorded_sha = record["record_sha256"]
        hashed = {key: value for key, value in record.items() if key != "record_sha256"}
        if canonical_sha256(hashed) != recorded_sha:
            raise RegistryError("registry record hash mismatch")
        payload = record["payload"]
        manifest_path = campaign_root / payload["manifest_path"]
        profile_path = campaign_root / payload["profile_path"]
        if canonical_sha256(read_json(manifest_path)) != payload["manifest_sha256"]:
            raise RegistryError("registered manifest identity mismatch")
        profile = read_json(profile_path)
        if canonical_sha256(profile) != payload["profile_sha256"]:
            raise RegistryError("registered profile identity mismatch")
        manifest = read_json(manifest_path)
        if (
            manifest["experiment_id"] != payload["experiment_id"]
            or manifest["artifact_path"] != payload["artifact_path"]
            or manifest["artifact_tree_sha256"] != payload["artifact_tree_sha256"]
            or manifest["profile_sha256"] != payload["profile_sha256"]
            or manifest["exit_state"] != payload["status"]
        ):
            raise RegistryError("registered payload and manifest differ")
        previous_record = recorded_sha
        prefix_bytes += raw
    head = read_json(campaign_root / "registry-head.json")
    prefix_sha256 = hashlib.sha256(prefix_bytes).hexdigest().upper()
    if (
        head["last_record_sha256"] != previous_record
        or head["record_count"] != len(records)
        or head["registry_bytes"] != len(prefix_bytes)
        or head["registry_prefix_sha256"] != prefix_sha256
    ):
        raise RegistryError("registry head differs from the chained registry bytes")
    return SimpleNamespace(
        record_count=len(records),
        family_ids=tuple(expected_ids[1:]),
        registry_prefix_sha256=prefix_sha256,
    )


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
    if experiment_id == CAMPAIGN_FAMILY_IDS[2]:
        return _verify_family_3_run(campaign_root, recompute_all=recompute_all)
    if experiment_id == CAMPAIGN_FAMILY_IDS[1]:
        return _verify_family_2_run(campaign_root, recompute_all=recompute_all)
    if experiment_id != CAMPAIGN_FAMILY_IDS[0]:
        raise ValueError("verifier owns only the completed campaign family prefix")
    manifest_path = campaign_root / "runs" / experiment_id / "manifest.json"
    manifest = read_json(manifest_path)
    if manifest.get("experiment_id") != experiment_id:
        raise ValueError("run manifest experiment ID mismatch")
    profile = read_json(campaign_root / manifest["profile_path"])
    if canonical_sha256(profile) != manifest["profile_sha256"]:
        raise ValueError("resolved profile hash mismatch")
    local_artifact = campaign_root / manifest["artifact_path"]
    artifact_root = local_artifact if local_artifact.is_dir() else FIXED_FAMILY_1_ARTIFACT
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


def _verify_family_2_run(campaign_root: Path, *, recompute_all: bool) -> dict[str, Any]:
    from datetime import date

    from .families.horizon_recency import (
        FIXED_INCUMBENT_ARTIFACT,
        _promotion_decision,
        select_joint_variant,
    )
    from .families.weighted_v8 import validate_prediction_chronology

    experiment_id = CAMPAIGN_FAMILY_IDS[1]
    manifest = read_json(campaign_root / f"runs/{experiment_id}/manifest.json")
    if manifest.get("experiment_id") != experiment_id:
        raise ValueError("family 2 manifest experiment ID mismatch")
    profile = read_json(campaign_root / manifest["profile_path"])
    validate_resolved_profile(profile)
    if canonical_sha256(profile) != manifest["profile_sha256"]:
        raise ValueError("family 2 resolved profile hash mismatch")
    local_artifact = campaign_root / manifest["artifact_path"]
    artifact_root = local_artifact if local_artifact.is_dir() else FIXED_FAMILY_2_ARTIFACT
    inventory = tree_inventory(artifact_root)
    if (
        inventory.tree_sha256 != manifest["artifact_tree_sha256"]
        or inventory.file_count != manifest["artifact_file_count"]
    ):
        raise ValueError("family 2 artifact inventory mismatch")

    attempts = read_jsonl(campaign_root / manifest["attempts_path"])
    launched = [record for record in attempts if record.get("state") == "launched"]
    terminal = [record for record in attempts if record.get("state") != "launched"]
    if len(launched) != len(terminal):
        raise ValueError("every family 2 launch must have one terminal record")
    launched_ids = [record["attempt_id"] for record in launched]
    terminal_ids = [record["attempt_id"] for record in terminal]
    if launched_ids != terminal_ids or len(set(launched_ids)) != len(launched_ids):
        raise ValueError("family 2 attempt identities are retried, duplicated, or misaligned")
    allowed_terminal = {"succeeded", "failed", "cancelled", "limited"}
    if any(record["state"] not in allowed_terminal for record in terminal):
        raise ValueError("family 2 attempt has an unsupported terminal state")
    if any(record["variant_id"] not in FAMILY_2_VARIANT_IDS for record in launched):
        raise ValueError("family 2 launched an unregistered variant")
    for launch, end in zip(launched, terminal, strict=True):
        if end["attempt_elapsed_seconds"] > profile["per_fit_time_cap_seconds"] + 1:
            raise ValueError("family 2 attempt exceeded its preregistered cap")
        if file_sha256(artifact_root / launch["stdout_path"]) != end["stdout_sha256"]:
            raise ValueError("family 2 stdout identity mismatch")
        if file_sha256(artifact_root / launch["stderr_path"]) != end["stderr_sha256"]:
            raise ValueError("family 2 stderr identity mismatch")
    acquired = read_json(artifact_root / "gpu-lease-acquired.json")
    released = read_json(artifact_root / "gpu-lease-released.json")
    if acquired["pid"] != released["pid"] or released["elapsed_seconds"] > profile["family_deadline_seconds"]:
        raise ValueError("family 2 GPU lease evidence violates the frozen bound")
    gate = AccessLedger(campaign_root).gate_status()
    if gate["state"] != "closed" or gate["protected_access_count"] != 0:
        raise ValueError("family 2 verification requires the gate closed with zero access")

    status = manifest["exit_state"]
    common = {
        "experiment_id": experiment_id,
        "status": status,
        "gate_access_count": gate["protected_access_count"],
        "artifact_tree_sha256": inventory.tree_sha256,
        "artifact_file_count": inventory.file_count,
        "attempt_count": len(launched),
        "gpu_lease_elapsed_seconds": released["elapsed_seconds"],
    }
    if status != "complete":
        failure = manifest["terminal_failure"]
        if not failure or failure["exit_state"] not in {"failed", "cancelled", "limited"}:
            raise ValueError("family 2 non-complete result lacks explicit terminal evidence")
        return {**common, "terminal_failure": failure}
    if len(launched) != len(FAMILY_2_VARIANT_IDS) * 4 or any(
        record["state"] != "succeeded" for record in terminal
    ):
        raise ValueError("complete family 2 result requires all 32 unique fits to succeed")
    expected_pairs = {
        (year, variant_id) for year in (2022, 2023, 2024, 2025) for variant_id in FAMILY_2_VARIANT_IDS
    }
    if {(record["fold"], record["variant_id"]) for record in launched} != expected_pairs:
        raise ValueError("family 2 launch matrix differs from the frozen menu")

    predictions: list[dict[str, Any]] = []
    selected_variants = manifest["selected_variants"]
    if len(selected_variants) != 4:
        raise ValueError("family 2 requires four inner-selected variants")
    for year, selection in zip((2022, 2023, 2024, 2025), selected_variants, strict=True):
        replayed = select_joint_variant(
            selection["inner_scores"],
            outer_min_date=date.fromisoformat(selection["inner_scores"][0]["outer_min_date"]),
            embargo_days=profile["embargo_days"],
        )
        _assert_equal(replayed, {key: selection[key] for key in replayed}, f"family 2 fold {year} selection")
        entry = next(item for item in manifest["fold_predictions"] if item["year"] == year)
        path = artifact_root / entry["path"]
        if file_sha256(path) != entry["sha256"]:
            raise ValueError("family 2 outer prediction identity mismatch")
        rows = read_jsonl(path)
        if len(rows) != entry["row_count"] or any(row["selected_variant"] != replayed["variant_id"] for row in rows):
            raise ValueError("family 2 selected outer set is inconsistent")
        predictions.extend(rows)
    validate_prediction_chronology(predictions)
    metrics = reduce_predictions(predictions).as_dict()
    incumbent = [
        row
        for year in (2022, 2023, 2024, 2025)
        for row in read_jsonl(FIXED_INCUMBENT_ARTIFACT / f"fold-{year}/outer-predictions.jsonl")
    ]
    incumbent_metrics = reduce_predictions(incumbent).as_dict()
    intervals = event_block_bootstrap_delta(
        predictions,
        incumbent,
        iterations=profile["bootstrap"]["iterations"],
        seed=profile["bootstrap"]["seed"],
    )
    fold_losses = metrics["fold_metrics"]
    drift = {
        "fold_log_loss": {str(year): fold_losses[str(year)]["log_loss"] for year in (2022, 2023, 2024, 2025)},
        "year_over_year_log_loss_delta": {
            str(year): fold_losses[str(year)]["log_loss"] - fold_losses[str(year - 1)]["log_loss"]
            for year in (2023, 2024, 2025)
        },
    }
    decision = _promotion_decision(metrics, intervals, complete=True)
    selected_inner = {
        "status": "selection-context",
        "fold_log_loss": {
            str(year): selected_variants[index]["inner_log_loss"]
            for index, year in enumerate((2022, 2023, 2024, 2025))
        },
        "row_count": sum(selection["inner_scores"][0]["row_count"] for selection in selected_variants),
    }
    if recompute_all:
        _assert_equal(metrics, manifest["metrics"], "family 2 metrics")
        _assert_equal(incumbent_metrics, manifest["incumbent_metrics"], "family 2 incumbent metrics")
        _assert_equal(intervals, manifest["paired_event_block_intervals"], "family 2 paired intervals")
        _assert_equal(drift, manifest["drift_summary"], "family 2 drift summary")
        _assert_equal(selected_inner, manifest["selected_inner_metrics"], "family 2 inner selection summary")
        _assert_equal(decision, manifest["promotion_decision"], "family 2 promotion decision")
    return {
        **common,
        "outer_years": [2022, 2023, 2024, 2025],
        "fold_prediction_count": 4,
        "selected_variants": [selection["variant_id"] for selection in selected_variants],
        "metrics": metrics,
        "incumbent_metrics": incumbent_metrics,
        "paired_event_block_intervals": intervals,
        "drift_summary": drift,
        "selected_inner_metrics": selected_inner,
        "inner_outer_gap": manifest["inner_outer_gap"],
        "promotion_decision": decision,
        "adaptive_signal_for_family_03": manifest["adaptive_signal_for_family_03"],
    }


def _verify_family_3_run(campaign_root: Path, *, recompute_all: bool) -> dict[str, Any]:
    from .calibration import CALIBRATION_VARIANT_IDS
    from .families.temporal_calibration import promotion_decision
    from .families.weighted_v8 import validate_prediction_chronology

    experiment_id = CAMPAIGN_FAMILY_IDS[2]
    manifest = read_json(campaign_root / f"runs/{experiment_id}/manifest.json")
    if manifest.get("experiment_id") != experiment_id or manifest.get("exit_state") != "complete":
        raise ValueError("family 3 manifest is not a completed result")
    profile = read_json(campaign_root / manifest["profile_path"])
    if canonical_sha256(profile) != manifest["profile_sha256"]:
        raise ValueError("family 3 profile hash mismatch")
    if tuple(item["id"] for item in profile["calibration_variants"]) != CALIBRATION_VARIANT_IDS:
        raise ValueError("family 3 profile differs from the exact four-variant menu")
    preregistration = read_json(campaign_root / manifest["preregistration_path"])
    lineage_preregistration = read_json(campaign_root / manifest["lineage_preregistration_path"])
    profile_file_sha256 = file_sha256(campaign_root / manifest["profile_path"])
    if preregistration["profile_sha256"] != profile_file_sha256:
        raise ValueError("family 3 preregistration profile identity mismatch")
    if lineage_preregistration["variant_menu_profile_sha256"] != profile_file_sha256:
        raise ValueError("family 3 lineage preregistration profile identity mismatch")
    if preregistration["scoring_state"] != "not-started" or lineage_preregistration["scoring_state"] != "not-started":
        raise ValueError("family 3 was not preregistered before score")
    artifact_root = campaign_root / manifest["artifact_path"]
    inventory = tree_inventory(artifact_root)
    if (
        inventory.tree_sha256 != manifest["artifact_tree_sha256"]
        or inventory.file_count != manifest["artifact_file_count"]
    ):
        raise ValueError("family 3 artifact inventory mismatch")
    gate = AccessLedger(campaign_root).gate_status()
    if gate["state"] != "closed" or gate["protected_access_count"] != 0:
        raise ValueError("family 3 verification requires the gate closed with zero access")
    lineage = read_json(artifact_root / manifest["lineage_audit_path"])
    negative_control = lineage["negative_control"]
    if len(negative_control) != 4 or any(
        item["status"] != "ineligible"
        or item["variant_fit_count"] != 0
        or item["variant_score_count"] != 0
        or item["same_fit_row_count"] == 0
        for item in negative_control
    ):
        raise ValueError("family 3 negative-control lineage was not rejected before score")
    if lineage["base_model_retrain_count"] != 0:
        raise ValueError("family 3 must not retrain a base model")

    predictions = []
    selections = manifest["selections"]
    if [entry["year"] for entry in manifest["fold_predictions"]] != [2022, 2023, 2024, 2025]:
        raise ValueError("family 3 requires exactly four outer folds")
    if selections[0]["variant_id"] != "identity" or selections[0]["fit_row_count"] != 0:
        raise ValueError("family 3 2022 fold must be identity-only with no fit")
    for entry, selection in zip(manifest["fold_predictions"], selections, strict=True):
        path = artifact_root / entry["path"]
        if file_sha256(path) != entry["sha256"]:
            raise ValueError("family 3 outer prediction identity mismatch")
        rows = read_jsonl(path)
        if len(rows) != entry["row_count"] or any(
            row["boundary"] != "Original"
            or row["selected_calibration_variant"] != selection["variant_id"]
            for row in rows
        ):
            raise ValueError("family 3 outer rows violate calibrated Original lineage")
        if entry["year"] == 2022 and any(
            row["probability"] != row["original_probability"] for row in rows
        ):
            raise ValueError("family 3 2022 identity probabilities changed")
        predictions.extend(rows)
    validate_prediction_chronology(predictions)
    metrics = reduce_predictions(predictions).as_dict()
    incumbent = [
        row
        for year in (2022, 2023, 2024, 2025)
        for row in read_jsonl(FIXED_FAMILY_2_ARTIFACT / f"fold-{year}/outer-predictions.jsonl")
    ]
    incumbent_metrics = reduce_predictions(incumbent).as_dict()
    intervals = event_block_bootstrap_delta(
        predictions,
        incumbent,
        iterations=profile["bootstrap"]["iterations"],
        seed=profile["bootstrap"]["seed"],
    )
    metric_deltas = {
        name: metrics[name] - incumbent_metrics[name]
        for name in ("log_loss", "brier", "calibration_intercept", "calibration_slope", "ece", "accuracy")
    }
    decision = promotion_decision(metric_deltas, intervals)
    if recompute_all:
        _assert_equal(metrics, manifest["metrics"], "family 3 metrics")
        _assert_equal(incumbent_metrics, manifest["incumbent_metrics"], "family 3 incumbent metrics")
        _assert_equal(metric_deltas, manifest["metric_deltas"], "family 3 metric deltas")
        _assert_equal(intervals, manifest["paired_event_block_intervals"], "family 3 intervals")
        _assert_equal(decision, manifest["promotion_decision"], "family 3 promotion decision")
    return {
        "experiment_id": experiment_id,
        "status": "complete",
        "gate_access_count": gate["protected_access_count"],
        "artifact_tree_sha256": inventory.tree_sha256,
        "artifact_file_count": inventory.file_count,
        "outer_years": [2022, 2023, 2024, 2025],
        "selected_variants": [selection["variant_id"] for selection in selections],
        "metrics": metrics,
        "incumbent_metrics": incumbent_metrics,
        "metric_deltas": metric_deltas,
        "paired_event_block_intervals": intervals,
        "promotion_decision": decision,
        "adaptive_signal_for_family_04": manifest["adaptive_signal_for_family_04"],
        "base_model_retrain_count": lineage["base_model_retrain_count"],
        "negative_control_count": len(negative_control),
    }


def replay_campaign_decisions(campaign_root: Path, *, through: str) -> dict[str, Any]:
    campaign_root = Path(campaign_root)
    if through not in CAMPAIGN_FAMILY_IDS[:3]:
        raise ValueError("decision replay is bounded to the completed family prefix")
    registry = _validate_campaign_registry(campaign_root)
    through_index = CAMPAIGN_FAMILY_IDS.index(through) + 1
    expected = CAMPAIGN_FAMILY_IDS[:through_index]
    if registry.family_ids[:through_index] != expected:
        raise ValueError("registry does not contain the requested replay prefix")
    decisions = []
    for experiment_id in expected:
        verified = verify_family_run(campaign_root, experiment_id, recompute_all=True)
        decisions.append({
            "experiment_id": experiment_id,
            "status": verified["status"],
            "promotion_decision": verified["promotion_decision"],
            "metrics": verified.get("metrics"),
            "paired_event_block_intervals": verified.get("paired_event_block_intervals"),
            "drift_summary": verified.get("drift_summary"),
            "adaptive_signal_for_next_family": verified.get(
                "adaptive_signal_for_family_03", verified.get("adaptive_signal_for_family_04")
            ),
        })
    return {
        "through": through,
        "registry_prefix_sha256": registry.registry_prefix_sha256,
        "decisions": decisions,
        "incumbent_after": decisions[-1]["promotion_decision"]["incumbent_after"],
        "gate_access_count": AccessLedger(campaign_root).gate_status()["protected_access_count"],
    }


def validate_terminal_campaign(
    campaign_root: Path,
    *,
    expect_terminal_through: int | None,
    require_gate_closed: bool,
) -> dict[str, Any]:
    campaign_root = Path(campaign_root)
    registry = _validate_campaign_registry(campaign_root)
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
