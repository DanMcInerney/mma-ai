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
FIXED_FAMILY_3_ARTIFACT = Path(
    r"C:\Users\danhm\mma-ai\worktrees\top10-20260815"
    r"\experiments\top10_20260815\artifacts\04-family-03-temporal-calibration"
)
FIXED_FAMILY_4_ARTIFACT = Path(
    r"C:\Users\danhm\mma-ai\worktrees\top10-20260815"
    r"\experiments\top10_20260815\artifacts\05-family-04-oof-ensemble"
)
FIXED_FAMILY_5_ARTIFACT = Path(
    r"C:\Users\danhm\mma-ai\worktrees\top10-20260815"
    r"\experiments\top10_20260815\artifacts\06-family-05-semantic-portfolio"
)
FIXED_FAMILY_6_ARTIFACT = Path(
    r"C:\Users\danhm\mma-ai\worktrees\top10-20260815"
    r"\experiments\top10_20260815\artifacts\07-family-06-fighter-states"
)
FIXED_FAMILY_7_ARTIFACT = Path(
    r"C:\Users\danhm\mma-ai\worktrees\top10-20260815"
    r"\experiments\top10_20260815\artifacts\08-family-07-matchup-geometry"
)
FIXED_FAMILY_8_ARTIFACT = Path(
    r"C:\Users\danhm\mma-ai\worktrees\top10-20260815"
    r"\experiments\top10_20260815\artifacts\09-family-08-catboost-specialist"
)
FIXED_FAMILY_9_ARTIFACT = Path(
    r"C:\Users\danhm\mma-ai\worktrees\top10-20260815"
    r"\experiments\top10_20260815\artifacts\10-family-09-capacity-foundation"
)
FIXED_FAMILY_10_ARTIFACT = Path(
    r"C:\Users\danhm\mma-ai\worktrees\top10-20260815"
    r"\experiments\top10_20260815\artifacts\11-family-10-outcome-decomposition"
)
FAMILY_5_RUN_ALIAS = "family-05-semantic-portfolio"
FAMILY_6_RUN_ALIAS = "family-06-fighter-states"
FAMILY_7_RUN_ALIAS = "family-07-matchup-geometry"
FAMILY_8_RUN_ALIAS = "family-08-catboost-specialist"
FAMILY_9_RUN_ALIAS = "family-09-capacity-foundation"
FAMILY_10_RUN_ALIAS = "family-10-outcome-decomposition"


def _canonical_family_id(experiment_id: str) -> str:
    if experiment_id == FAMILY_5_RUN_ALIAS:
        return CAMPAIGN_FAMILY_IDS[4]
    if experiment_id == FAMILY_6_RUN_ALIAS:
        return CAMPAIGN_FAMILY_IDS[5]
    if experiment_id == FAMILY_7_RUN_ALIAS:
        return CAMPAIGN_FAMILY_IDS[6]
    if experiment_id == FAMILY_8_RUN_ALIAS:
        return CAMPAIGN_FAMILY_IDS[7]
    if experiment_id == FAMILY_9_RUN_ALIAS:
        return CAMPAIGN_FAMILY_IDS[8]
    if experiment_id == FAMILY_10_RUN_ALIAS:
        return CAMPAIGN_FAMILY_IDS[9]
    return experiment_id


def _canonical_checkout_text_sha256(path: Path) -> str:
    normalized = Path(path).read_bytes().replace(b"\r\n", b"\n")
    if b"\r" in normalized:
        raise ValueError("canonical text identity rejects bare carriage returns")
    return hashlib.sha256(normalized).hexdigest().upper()
FROZEN_REGISTRY_PREFIX_BEFORE_FAMILY_3 = (
    "C5F8E37AEC82E0AEFDAAE6EECF7A89E55EFDC04788884FFA504105F131C752BB"
)
FROZEN_REGISTRY_PREFIX_BEFORE_FAMILY_4 = (
    "BAB7A9FBBFCFEA0D024DBBA8F338805CB1EBAB1B578AE663CF5E3862239C7C7D"
)
FROZEN_REGISTRY_PREFIX_BEFORE_FAMILY_5 = (
    "62EF72477F4451E42533CF2F4D7DA0FB29F4774739019569E40D6FDE16256D40"
)
FROZEN_REGISTRY_PREFIX_BEFORE_FAMILY_6 = (
    "AB670F8B67B5408B949C264A808F9B9F788494D23C9F785FD0E41BCC17A63B25"
)
FROZEN_REGISTRY_PREFIX_BEFORE_FAMILY_7 = (
    "16E71698CFF77558EDB91085DB5CD3F0A5C6186DE65D69C13E722C05805C7592"
)
FROZEN_REGISTRY_PREFIX_BEFORE_FAMILY_8 = (
    "21033B96E86D9D7317DE070CE66E84082D1C25B828D70FC5928D25651BA30F71"
)
FROZEN_REGISTRY_PREFIX_BEFORE_FAMILY_9 = (
    "5DC5D278A65C8633A3ECF2C72FE886604B202FCBF29AEE2A8C4B04DC9AD02E0F"
)
FROZEN_REGISTRY_PREFIX_BEFORE_FAMILY_10 = (
    "A1A7FC3F8486D8A07C82ED73F77626B93381A57F81CD8907796BC88ED10CB622"
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
    if len(records) > 11:
        raise RegistryError("registry extends beyond the completed family-10 prefix")
    expected_ids = ["experiment-zero", *CAMPAIGN_FAMILY_IDS[: len(records) - 1]]
    if [row["payload"]["experiment_id"] for row in records] != expected_ids:
        raise RegistryError("registry does not contain the exact completed family prefix")
    if hashlib.sha256(b"".join(raw_lines[:3])).hexdigest().upper() != FROZEN_REGISTRY_PREFIX_BEFORE_FAMILY_3:
        raise RegistryError("the frozen registry prefix before family 3 changed")
    if (
        len(raw_lines) > 4
        and hashlib.sha256(b"".join(raw_lines[:4])).hexdigest().upper()
        != FROZEN_REGISTRY_PREFIX_BEFORE_FAMILY_4
    ):
        raise RegistryError("the frozen registry prefix before family 4 changed")
    if (
        len(raw_lines) > 5
        and hashlib.sha256(b"".join(raw_lines[:5])).hexdigest().upper()
        != FROZEN_REGISTRY_PREFIX_BEFORE_FAMILY_5
    ):
        raise RegistryError("the frozen registry prefix before family 5 changed")
    if (
        len(raw_lines) > 6
        and hashlib.sha256(b"".join(raw_lines[:6])).hexdigest().upper()
        != FROZEN_REGISTRY_PREFIX_BEFORE_FAMILY_6
    ):
        raise RegistryError("the frozen registry prefix before family 6 changed")
    if (
        len(raw_lines) > 7
        and hashlib.sha256(b"".join(raw_lines[:7])).hexdigest().upper()
        != FROZEN_REGISTRY_PREFIX_BEFORE_FAMILY_7
    ):
        raise RegistryError("the frozen registry prefix before family 7 changed")
    if (
        len(raw_lines) > 8
        and hashlib.sha256(b"".join(raw_lines[:8])).hexdigest().upper()
        != FROZEN_REGISTRY_PREFIX_BEFORE_FAMILY_8
    ):
        raise RegistryError("the frozen registry prefix before family 8 changed")
    if (
        len(raw_lines) > 9
        and hashlib.sha256(b"".join(raw_lines[:9])).hexdigest().upper()
        != FROZEN_REGISTRY_PREFIX_BEFORE_FAMILY_9
    ):
        raise RegistryError("the frozen registry prefix before family 9 changed")
    if (
        len(raw_lines) > 10
        and hashlib.sha256(b"".join(raw_lines[:10])).hexdigest().upper()
        != FROZEN_REGISTRY_PREFIX_BEFORE_FAMILY_10
    ):
        raise RegistryError("the frozen registry prefix before family 10 changed")
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
    experiment_id = _canonical_family_id(experiment_id)
    if experiment_id == CAMPAIGN_FAMILY_IDS[9]:
        return _verify_family_10_run(campaign_root, recompute_all=recompute_all)
    if experiment_id == CAMPAIGN_FAMILY_IDS[8]:
        return _verify_family_9_run(campaign_root, recompute_all=recompute_all)
    if experiment_id == CAMPAIGN_FAMILY_IDS[7]:
        return _verify_family_8_run(campaign_root, recompute_all=recompute_all)
    if experiment_id == CAMPAIGN_FAMILY_IDS[6]:
        return _verify_family_7_run(campaign_root, recompute_all=recompute_all)
    if experiment_id == CAMPAIGN_FAMILY_IDS[5]:
        return _verify_family_6_run(campaign_root, recompute_all=recompute_all)
    if experiment_id == CAMPAIGN_FAMILY_IDS[4]:
        return _verify_family_5_run(campaign_root, recompute_all=recompute_all)
    if experiment_id == CAMPAIGN_FAMILY_IDS[3]:
        return _verify_family_4_run(campaign_root, recompute_all=recompute_all)
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
    local_artifact = campaign_root / manifest["artifact_path"]
    artifact_root = local_artifact if local_artifact.is_dir() else FIXED_FAMILY_3_ARTIFACT
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


def _verify_family_4_run(campaign_root: Path, *, recompute_all: bool) -> dict[str, Any]:
    from .ensemble import ENSEMBLE_VARIANT_IDS
    from .families.oof_ensemble import (
        FIXED_ARTIFACT_BASE,
        _metric_gaps,
        promotion_decision,
        select_recipe_for_outer,
    )

    experiment_id = CAMPAIGN_FAMILY_IDS[3]
    manifest = read_json(campaign_root / "runs/family-04-oof-ensemble/manifest.json")
    if manifest.get("experiment_id") != experiment_id or manifest.get("exit_state") != "complete":
        raise ValueError("family 4 manifest is not a completed result")
    profile_path = campaign_root / manifest["profile_path"]
    profile = read_json(profile_path)
    if (
        canonical_sha256(profile) != manifest["profile_sha256"]
        or file_sha256(profile_path) != manifest["profile_file_sha256"]
        or tuple(profile["recipe_menu"]) != ENSEMBLE_VARIANT_IDS
        or tuple(profile["selection_tie_break"]) != ENSEMBLE_VARIANT_IDS
    ):
        raise ValueError("family 4 profile differs from the preregistered exact menu")
    preregistration = read_json(campaign_root / manifest["preregistration_path"])
    if (
        preregistration["profile_sha256"] != file_sha256(profile_path)
        or preregistration["scoring_state"] != "not-started"
        or tuple(preregistration["preregistered_recipe_ids"]) != ENSEMBLE_VARIANT_IDS
        or preregistration["solver"] != profile["solver"]
        or preregistration["regularization_shrinkage"] != profile["regularization_shrinkage"]
        or preregistration["foundation_aggregate_cap"] != profile["foundation_aggregate_cap"]
    ):
        raise ValueError("family 4 menu or optimizer was not preregistered")
    local_artifact = campaign_root / manifest["artifact_path"]
    artifact_root = local_artifact if local_artifact.is_dir() else FIXED_FAMILY_4_ARTIFACT
    inventory = tree_inventory(artifact_root)
    if (
        inventory.tree_sha256 != manifest["artifact_tree_sha256"]
        or inventory.file_count != manifest["artifact_file_count"]
    ):
        raise ValueError("family 4 artifact inventory mismatch")
    gate = AccessLedger(campaign_root).gate_status()
    if gate["state"] != "closed" or gate["protected_access_count"] != 0:
        raise ValueError("family 4 verification requires the gate closed with zero access")
    attempts = read_jsonl(campaign_root / manifest["attempts_path"])
    if (
        len(attempts) != 20
        or any(record["recipe_id"] not in ENSEMBLE_VARIANT_IDS for record in attempts)
        or any(
            {record["recipe_id"] for record in attempts if record["fold"] == year}
            != set(ENSEMBLE_VARIANT_IDS)
            for year in (2022, 2023, 2024, 2025)
        )
    ):
        raise ValueError("family 4 attempts differ from the exact maximum-five menu")
    lineage = read_json(artifact_root / manifest["constituent_lineage_path"])
    if (
        lineage["outer_label_fit_count"] != 0
        or lineage["full_prediction_node_count"] != 0
        or lineage["gate_access_count"] != 0
        or lineage["foundation_aggregate_cap"] != profile["foundation_aggregate_cap"]
    ):
        raise ValueError("family 4 lineage records a forbidden fit or access")

    sources = {
        item["id"]: FIXED_ARTIFACT_BASE / Path(item["artifact_path"]).name
        for item in profile["constituents"]
    }
    history = {name: [] for name in profile["constituent_ids"]}
    predictions: list[dict[str, Any]] = []
    incumbent: list[dict[str, Any]] = []
    replayed_selections = []
    for index, year in enumerate(profile["outer_years"]):
        outer = {
            name: read_jsonl(sources[name] / f"fold-{year}/outer-predictions.jsonl")
            for name in profile["constituent_ids"]
        }
        lineage_fold = lineage["folds"][index]
        for name in profile["constituent_ids"]:
            source_path = sources[name] / f"fold-{year}/outer-predictions.jsonl"
            source_identity = lineage_fold["constituents"][name]
            if (
                source_identity["sha256"] != file_sha256(source_path)
                or source_identity["row_count"] != len(outer[name])
            ):
                raise ValueError("family 4 constituent source identity mismatch")
        replayed = select_recipe_for_outer(
            history if any(history.values()) else {},
            outer,
            outer_year=year,
            profile=profile,
        )
        selection = replayed["selection"]
        _assert_equal(selection, manifest["selections"][index], f"family 4 fold {year} selection")
        _assert_equal(selection, read_json(artifact_root / f"fold-{year}/selection.json"), f"family 4 fold {year} selection artifact")
        entry = manifest["fold_predictions"][index]
        prediction_path = artifact_root / entry["path"]
        if file_sha256(prediction_path) != entry["sha256"]:
            raise ValueError("family 4 outer prediction identity mismatch")
        stored = read_jsonl(prediction_path)
        if (
            len(stored) != entry["row_count"]
            or any(row["boundary"] != "Original" for row in stored)
        ):
            raise ValueError("family 4 requires four Original outer prediction sets")
        _assert_equal(stored, replayed["predictions"], f"family 4 fold {year} predictions")
        predictions.extend(stored)
        incumbent.extend(outer[profile["current_constituent_id"]])
        replayed_selections.append(selection)
        for name in profile["constituent_ids"]:
            history[name].extend(outer[name])

    metrics = reduce_predictions(predictions).as_dict()
    incumbent_metrics = reduce_predictions(incumbent).as_dict()
    intervals = event_block_bootstrap_delta(
        predictions,
        incumbent,
        iterations=int(profile["bootstrap"]["iterations"]),
        seed=int(profile["bootstrap"]["seed"]),
    )
    metric_deltas = {
        name: float(metrics[name]) - float(incumbent_metrics[name])
        for name in ("log_loss", "brier", "accuracy")
    }
    calibration_gaps, subgroup_gaps = _metric_gaps(metrics, incumbent_metrics)
    decision = promotion_decision({"log_loss_delta": metric_deltas["log_loss"]}, intervals)
    adaptive_signal = {
        "selected_recipes": [selection["selected_recipe_id"] for selection in replayed_selections],
        "foundation_weights": [
            sum(selection["weights"][name] for name in profile["foundation_constituent_ids"])
            for selection in replayed_selections
        ],
        "pooled_log_loss_delta": metric_deltas["log_loss"],
        "pooled_ece_delta": calibration_gaps["ece"],
    }
    if recompute_all:
        _assert_equal(metrics, manifest["metrics"], "family 4 metrics")
        _assert_equal(incumbent_metrics, manifest["incumbent_metrics"], "family 4 incumbent metrics")
        _assert_equal(metric_deltas, manifest["metric_deltas"], "family 4 metric deltas")
        _assert_equal(calibration_gaps, manifest["calibration_gaps"], "family 4 calibration gaps")
        _assert_equal(subgroup_gaps, manifest["subgroup_gaps"], "family 4 subgroup gaps")
        _assert_equal(intervals, manifest["paired_event_block_intervals"], "family 4 intervals")
        _assert_equal(decision, manifest["promotion_decision"], "family 4 promotion decision")
        _assert_equal(adaptive_signal, manifest["adaptive_signal_for_family_05"], "family 4 adaptive signal")
    return {
        "experiment_id": experiment_id,
        "status": "complete",
        "gate_access_count": gate["protected_access_count"],
        "artifact_tree_sha256": inventory.tree_sha256,
        "artifact_file_count": inventory.file_count,
        "outer_years": profile["outer_years"],
        "selected_recipes": adaptive_signal["selected_recipes"],
        "weights": [selection["weights"] for selection in replayed_selections],
        "metrics": metrics,
        "incumbent_metrics": incumbent_metrics,
        "metric_deltas": metric_deltas,
        "calibration_gaps": calibration_gaps,
        "subgroup_gaps": subgroup_gaps,
        "paired_event_block_intervals": intervals,
        "promotion_decision": decision,
        "adaptive_signal_for_family_05": adaptive_signal,
        "preregistration_commit": manifest["preregistration_commit"],
    }


def _verify_family_5_run(campaign_root: Path, *, recompute_all: bool) -> dict[str, Any]:
    from .families.semantic_portfolio import (
        EXPERIMENT_ID,
        FROZEN_SOURCE,
        V8_ORDERED_FEATURE_SHA256,
        _header_sha256,
        _metric_gaps,
        _source_header,
        promotion_decision,
    )
    from .semantic_portfolio import (
        MEASUREMENT_GROUP_IDS,
        select_stable_features,
        validate_preregistered_profile,
    )

    manifest = read_json(campaign_root / "runs/family-05-semantic-portfolio/manifest.json")
    if manifest.get("experiment_id") != EXPERIMENT_ID or manifest.get("exit_state") != "complete":
        raise ValueError("family 5 manifest is not a completed result")
    profile_path = campaign_root / manifest["profile_path"]
    profile = read_json(profile_path)
    header = _source_header(Path(profile["frozen_source"]["absolute_path"]))
    validated_profile = validate_preregistered_profile(profile, source_header=header)
    expected_profile_ids = ("v8-control", *MEASUREMENT_GROUP_IDS)
    if (
        canonical_sha256(profile) != manifest["profile_sha256"]
        or file_sha256(profile_path) != manifest["profile_file_sha256"]
        or tuple(item["id"] for item in profile["measurement_profiles"]) != expected_profile_ids
        or len(profile["measurement_profiles"]) != 8
        or profile["v8_ordered_feature_sha256"] != V8_ORDERED_FEATURE_SHA256
    ):
        raise ValueError("family 5 profile differs from the preregistered exact menu")
    preregistration = read_json(campaign_root / manifest["preregistration_path"])
    if (
        preregistration["scoring_state"] != "not-started"
        or preregistration["profile_file_sha256"] != file_sha256(profile_path)
        or preregistration["profile_sha256"] != canonical_sha256(profile)
        or tuple(preregistration["preregistered_profile_ids"]) != expected_profile_ids
        or preregistration["source_header_sha256"] != _header_sha256(header)
    ):
        raise ValueError("family 5 menu, source, or selection rule was not preregistered")
    local_artifact = campaign_root / manifest["artifact_path"]
    artifact_root = local_artifact if local_artifact.is_dir() else FIXED_FAMILY_5_ARTIFACT
    inventory = tree_inventory(artifact_root)
    if (
        inventory.tree_sha256 != manifest["artifact_tree_sha256"]
        or inventory.file_count != manifest["artifact_file_count"]
    ):
        raise ValueError("family 5 artifact inventory mismatch")
    gate = AccessLedger(campaign_root).gate_status()
    if gate["state"] != "closed" or gate["protected_access_count"] != 0:
        raise ValueError("family 5 verification requires the gate closed with zero access")
    attempts = read_jsonl(campaign_root / manifest["attempts_path"])
    if (
        len(attempts) != 32
        or any(record["profile_id"] not in expected_profile_ids for record in attempts)
        or any(
            {record["profile_id"] for record in attempts if record["fold"] == year}
            != set(expected_profile_ids)
            for year in profile["outer_years"]
        )
    ):
        raise ValueError("family 5 attempts differ from the exact maximum-eight menu")
    lineage = read_json(artifact_root / manifest["source_lineage_path"])
    data_path = campaign_root.parents[1] / manifest["data_path"]
    data_sha256 = _canonical_checkout_text_sha256(data_path)
    if (
        lineage["source_file_sha256"] != profile["frozen_source"]["sha256"]
        or lineage["source_header_sha256"] != _header_sha256(header)
        or lineage["candidate_feature_sha256"] != V8_ORDERED_FEATURE_SHA256
        or lineage["outer_label_selection_count"] != 0
        or lineage["gate_selection_count"] != 0
        or lineage["combined_row_importance_used"] is not False
        or data_sha256 != manifest["data_sha256"]
        or data_sha256 != lineage["development_table_sha256"]
    ):
        raise ValueError("family 5 source or selection lineage mismatch")

    candidate_predictions: list[dict[str, Any]] = []
    incumbent_predictions: list[dict[str, Any]] = []
    train_predictions: list[dict[str, Any]] = []
    replayed_selections = []
    for fold in manifest["folds"]:
        year = int(fold["year"])
        evidence_path = artifact_root / fold["evidence_path"]
        selection_path = artifact_root / fold["selection_path"]
        prediction_path = artifact_root / fold["prediction_path"]
        train_path = artifact_root / fold["train_prediction_path"]
        model_path = artifact_root / fold["model_path"]
        if (
            file_sha256(evidence_path) != fold["evidence_sha256"]
            or canonical_sha256(read_json(selection_path)) != fold["selection_sha256"]
            or file_sha256(prediction_path) != fold["prediction_sha256"]
            or file_sha256(train_path) != fold["train_prediction_sha256"]
            or canonical_sha256(read_json(model_path)) != fold["model_sha256"]
        ):
            raise ValueError(f"family 5 fold {year} artifact identity mismatch")
        evidence = read_jsonl(evidence_path)
        if len(evidence) != 240:
            raise ValueError(f"family 5 fold {year} inner evidence is incomplete")
        scores = {}
        stable = {}
        for profile_id in expected_profile_ids:
            rows = [row for row in evidence if row["profile_id"] == profile_id]
            stable[profile_id] = select_stable_features(rows, profile=profile, outer_year=year)
            losses = {
                int(row["fold"]): float(row["validation_log_loss"])
                for row in rows
            }
            if len(losses) != profile["inner_validation_year_count"]:
                raise ValueError(f"family 5 fold {year} profile evidence support mismatch")
            scores[profile_id] = sum(losses[fold_year] for fold_year in sorted(losses)) / len(losses)
        eligible = [profile_id for profile_id in expected_profile_ids if stable[profile_id]["selected_features"]]
        selected_profile = min(
            eligible,
            key=lambda profile_id: (scores[profile_id], expected_profile_ids.index(profile_id)),
        )
        replayed = {
            **stable[selected_profile],
            "selected_profile_id": selected_profile,
            "profile_scores": scores,
            "eligible_profile_ids": eligible,
            "scored_profile_count": len(scores),
            "selection_basis": "minimum mean inner log-loss among stability-eligible profiles",
        }
        selection = read_json(selection_path)
        _assert_equal(replayed, selection, f"family 5 fold {year} inner selection")
        if (
            selection["outer_label_selection_count"] != 0
            or selection["combined_row_importance_used"] is not False
            or selection["selected_feature_sha256"] != fold["selected_feature_sha256"]
        ):
            raise ValueError("family 5 selection records forbidden evidence")
        predictions = read_jsonl(prediction_path)
        trains = read_jsonl(train_path)
        if (
            len(predictions) != fold["prediction_row_count"]
            or len(trains) != fold["train_prediction_row_count"]
            or any(
                row.get("boundary") != "Original"
                or row.get("fit_scope") != "prior-only"
                or int(str(row["event_date"])[:4]) != year
                or row.get("selected_measurement_profile") != selected_profile
                for row in predictions
            )
        ):
            raise ValueError("family 5 requires four Original outer prediction sets")
        incumbent = read_jsonl(FIXED_FAMILY_1_ARTIFACT / f"fold-{year}/outer-predictions.jsonl")
        candidate_predictions.extend(predictions)
        incumbent_predictions.extend(incumbent)
        train_predictions.extend(trains)
        replayed_selections.append(selection)

    metrics = reduce_predictions(candidate_predictions).as_dict()
    incumbent_metrics = reduce_predictions(incumbent_predictions).as_dict()
    train_metric_result = reduce_predictions(train_predictions)
    train_metrics = train_metric_result.as_dict()
    metric_deltas = {
        name: float(metrics[name]) - float(incumbent_metrics[name])
        for name in ("log_loss", "brier", "accuracy")
    }
    intervals = event_block_bootstrap_delta(
        candidate_predictions,
        incumbent_predictions,
        iterations=int(profile["bootstrap"]["iterations"]),
        seed=int(profile["bootstrap"]["seed"]),
    )
    calibration_gaps, subgroup_gaps = _metric_gaps(metrics, incumbent_metrics)
    train_gaps = metric_gap(train_metric_result, reduce_predictions(candidate_predictions))
    decision = promotion_decision(metric_deltas, intervals)
    adaptive_signal = {
        "selected_profiles": [item["selected_profile_id"] for item in replayed_selections],
        "selected_feature_hashes": [item["selected_feature_sha256"] for item in replayed_selections],
        "selected_feature_counts": [len(item["selected_features"]) for item in replayed_selections],
        "pooled_log_loss_delta": metric_deltas["log_loss"],
        "pooled_ece_delta": calibration_gaps["ece"],
    }
    if recompute_all:
        _assert_equal(metrics, manifest["metrics"], "family 5 metrics")
        _assert_equal(incumbent_metrics, manifest["incumbent_metrics"], "family 5 incumbent metrics")
        _assert_equal(train_metrics, manifest["train_metrics"], "family 5 train metrics")
        _assert_equal(metric_deltas, manifest["metric_deltas"], "family 5 metric deltas")
        _assert_equal(calibration_gaps, manifest["calibration_gaps"], "family 5 calibration gaps")
        _assert_equal(subgroup_gaps, manifest["subgroup_gaps"], "family 5 subgroup gaps")
        _assert_equal(train_gaps, manifest["train_gaps"], "family 5 train gaps")
        _assert_equal(intervals, manifest["paired_event_block_intervals"], "family 5 intervals")
        _assert_equal(decision, manifest["promotion_decision"], "family 5 promotion decision")
        _assert_equal(adaptive_signal, manifest["adaptive_signal_for_family_06"], "family 5 adaptive signal")
    return {
        "experiment_id": EXPERIMENT_ID,
        "status": "complete",
        "gate_access_count": gate["protected_access_count"],
        "artifact_tree_sha256": inventory.tree_sha256,
        "artifact_file_count": inventory.file_count,
        "profile_count": validated_profile["profile_count"],
        "outer_years": profile["outer_years"],
        "selected_profiles": adaptive_signal["selected_profiles"],
        "selected_features_by_fold": [item["selected_features"] for item in replayed_selections],
        "selected_feature_hashes": adaptive_signal["selected_feature_hashes"],
        "metrics": metrics,
        "incumbent_metrics": incumbent_metrics,
        "train_metrics": train_metrics,
        "metric_deltas": metric_deltas,
        "calibration_gaps": calibration_gaps,
        "subgroup_gaps": subgroup_gaps,
        "train_gaps": train_gaps,
        "paired_event_block_intervals": intervals,
        "promotion_decision": decision,
        "adaptive_signal_for_family_06": adaptive_signal,
        "outer_label_selection_count": lineage["outer_label_selection_count"],
        "combined_row_importance_used": lineage["combined_row_importance_used"],
        "preregistration_commit": manifest["preregistration_commit"],
    }


def _verify_family_6_run(campaign_root: Path, *, recompute_all: bool) -> dict[str, Any]:
    from .families.fighter_states import (
        EXPERIMENT_ID,
        PROFILE_IDS,
        build_preregistered_profile,
    )
    from .fighter_states import validate_preregistered_profiles

    manifest = read_json(campaign_root / "runs/family-06-fighter-states/manifest.json")
    profile_path = campaign_root / manifest["profile_path"]
    profile = read_json(profile_path)
    validated = validate_preregistered_profiles(profile)
    preregistration = read_json(campaign_root / manifest["preregistration_path"])
    if (
        manifest.get("experiment_id") != EXPERIMENT_ID
        or profile != build_preregistered_profile()
        or tuple(validated["profile_ids"]) != PROFILE_IDS
        or canonical_sha256(profile) != manifest["profile_sha256"]
        or file_sha256(profile_path) != manifest["profile_file_sha256"]
        or preregistration["scoring_state"] != "not-started"
        or canonical_sha256(preregistration) != manifest["preregistration_sha256"]
    ):
        raise ValueError("family 6 profile or preregistration identity mismatch")
    local_artifact = campaign_root / manifest["artifact_path"]
    artifact_root = local_artifact if local_artifact.is_dir() else FIXED_FAMILY_6_ARTIFACT
    inventory = tree_inventory(artifact_root)
    if (
        inventory.tree_sha256 != manifest["artifact_tree_sha256"]
        or inventory.file_count != manifest["artifact_file_count"]
    ):
        raise ValueError("family 6 artifact inventory mismatch")
    gate = AccessLedger(campaign_root).gate_status()
    if gate["state"] != "closed" or gate["protected_access_count"] != 0:
        raise ValueError("family 6 verification requires the gate closed with zero access")
    status = manifest["exit_state"]
    if status != "failed":
        raise ValueError("family 6 verifier expected the frozen terminal failure result")
    failure = manifest["terminal_failure"]
    stderr_path = artifact_root / failure["stderr_path"]
    attempts = read_jsonl(campaign_root / manifest["attempts_path"])
    acquired = read_json(artifact_root / "gpu-lease-acquired.json")
    released = read_json(artifact_root / "gpu-lease-released.json")
    safety = read_json(artifact_root / "safety.json")
    if (
        failure["attempt_ordinal"] != 1
        or failure["construction_started"] is not False
        or failure["fit_started"] is not False
        or failure["outer_labels_scored"] is not False
        or failure["retry_performed"] is not False
        or file_sha256(stderr_path) != failure["stderr_sha256"]
        or len(attempts) != 1
        or attempts[0].get("retry") is not False
        or acquired["lease_id"] != released["lease_id"]
        or acquired["pid"] != released["pid"]
        or safety["gpu_lease_count"] != 1
        or safety["production_attempt_count"] != 1
        or safety["retry_count"] != 0
        or safety["database_access"] != {"used": False, "sql": None, "urls": []}
    ):
        raise ValueError("family 6 terminal failure or invocation evidence differs")
    if recompute_all:
        _assert_equal(read_json(artifact_root / "failure.json"), failure, "family 6 failure")
        result = read_json(artifact_root / "result.json")
        for key in (
            "status",
            "terminal_failure",
            "metrics",
            "paired_event_block_intervals",
            "support_summary",
            "promotion_decision",
            "adaptive_signal_for_family_07",
            "gate_access_count",
        ):
            _assert_equal(result[key], manifest[key], f"family 6 {key}")
    return {
        "experiment_id": EXPERIMENT_ID,
        "status": status,
        "gate_access_count": gate["protected_access_count"],
        "artifact_tree_sha256": inventory.tree_sha256,
        "artifact_file_count": inventory.file_count,
        "profile_count": validated["profile_count"],
        "attempt_count": 1,
        "retry_count": 0,
        "terminal_failure": failure,
        "metrics": None,
        "paired_event_block_intervals": None,
        "support_summary": manifest["support_summary"],
        "promotion_decision": manifest["promotion_decision"],
        "adaptive_signal_for_family_07": manifest["adaptive_signal_for_family_07"],
        "outer_prediction_identities": manifest["outer_prediction_identities"],
        "preregistration_commit": manifest["preregistration_commit"],
    }


def _verify_family_7_run(campaign_root: Path, *, recompute_all: bool) -> dict[str, Any]:
    from .families.matchup_geometry import (
        EXPERIMENT_ID,
        PROFILE_IDS,
        build_preregistered_profile,
    )
    from .feature_lineage import build_development_safe_ids
    from .matchup_geometry import validate_preregistered_matchup_profiles

    manifest = read_json(campaign_root / "runs/family-07-matchup-geometry/manifest.json")
    profile_path = campaign_root / manifest["profile_path"]
    profile = read_json(profile_path)
    validated = validate_preregistered_matchup_profiles(profile)
    preregistration = read_json(campaign_root / manifest["preregistration_path"])
    if (
        manifest.get("experiment_id") != EXPERIMENT_ID
        or profile != build_preregistered_profile()
        or tuple(validated["profile_ids"]) != PROFILE_IDS
        or canonical_sha256(profile) != manifest["profile_sha256"]
        or file_sha256(profile_path) != manifest["profile_file_sha256"]
        or preregistration["scoring_state"] != "not-started"
        or canonical_sha256(preregistration) != manifest["preregistration_sha256"]
    ):
        raise ValueError("family 7 profile or preregistration identity mismatch")
    local_artifact = campaign_root / manifest["artifact_path"]
    artifact_root = local_artifact if local_artifact.is_dir() else FIXED_FAMILY_7_ARTIFACT
    inventory = tree_inventory(artifact_root)
    if (
        inventory.tree_sha256 != manifest["artifact_tree_sha256"]
        or inventory.file_count != manifest["artifact_file_count"]
    ):
        raise ValueError("family 7 artifact inventory mismatch")
    gate = AccessLedger(campaign_root).gate_status()
    if gate["state"] != "closed" or gate["protected_access_count"] != 0:
        raise ValueError("family 7 verification requires the gate closed with zero access")
    safe_ids, retired_ids = build_development_safe_ids(
        read_json(campaign_root / "baseline/fold-manifest.json")
    )
    if (
        len(safe_ids) != 3_089
        or len(retired_ids) != 178
        or manifest["development_safe_population"]["development_max_date"] != "2025-12-13"
        or manifest["development_safe_population"]["asserted_before_row_or_target_decode"] is not True
    ):
        raise ValueError("family 7 development-safe population evidence differs")
    if manifest["exit_state"] != "failed":
        raise ValueError("family 7 verifier expected the frozen terminal failure result")
    failure = manifest["terminal_failure"]
    attempts = read_jsonl(campaign_root / manifest["attempts_path"])
    acquired = read_json(artifact_root / "gpu-lease-acquired.json")
    released = read_json(artifact_root / "gpu-lease-released.json")
    safety = read_json(artifact_root / "safety.json")
    dependency = read_json(artifact_root / "dependency-evidence.json")
    dependency_inventory = tree_inventory(FIXED_FAMILY_6_ARTIFACT)
    if (
        failure["attempt_ordinal"] != 1
        or failure["stage"] != "pre-construction-dependency-resolution"
        or failure["construction_started"] is not False
        or failure["row_decode_started"] is not False
        or failure["target_decode_started"] is not False
        or failure["fit_started"] is not False
        or failure["outer_labels_scored"] is not False
        or failure["retry_performed"] is not False
        or file_sha256(artifact_root / failure["stderr_path"]) != failure["stderr_sha256"]
        or len(attempts) != 1
        or attempts[0].get("retry") is not False
        or acquired["lease_id"] != released["lease_id"]
        or acquired["pid"] != released["pid"]
        or safety["gpu_lease_count"] != 1
        or safety["production_attempt_count"] != 1
        or safety["retry_count"] != 0
        or safety["database_access"] != {"used": False, "sql": None, "urls": []}
        or dependency["exit_state"] != "failed"
        or dependency["data_path"] is not None
        or dependency["outer_prediction_identities"]
        or dependency["artifact_tree_sha256"] != dependency_inventory.tree_sha256
    ):
        raise ValueError("family 7 terminal failure, dependency, or invocation evidence differs")
    if manifest["outer_original_prediction_identities"] or manifest["outer_swapped_prediction_identities"]:
        raise ValueError("family 7 pre-construction failure unexpectedly recorded predictions")
    if recompute_all:
        _assert_equal(read_json(artifact_root / "failure.json"), failure, "family 7 failure")
        _assert_equal(dependency, manifest["dependency_evidence"], "family 7 dependency evidence")
        result = read_json(artifact_root / "result.json")
        for key in (
            "status",
            "terminal_failure",
            "metrics",
            "paired_event_block_intervals",
            "slice_metrics",
            "symmetry_diagnostics",
            "swap_mapping_and_invariance_evidence",
            "outer_original_prediction_identities",
            "outer_swapped_prediction_identities",
            "promotion_decision",
            "adaptive_signal_for_family_08",
            "development_safe_population",
            "dependency_evidence",
            "gate_access_count",
        ):
            _assert_equal(result[key], manifest[key], f"family 7 {key}")
    return {
        "experiment_id": EXPERIMENT_ID,
        "status": "failed",
        "gate_access_count": gate["protected_access_count"],
        "artifact_tree_sha256": inventory.tree_sha256,
        "artifact_file_count": inventory.file_count,
        "profile_count": validated["profile_count"],
        "attempt_count": 1,
        "retry_count": 0,
        "terminal_failure": failure,
        "metrics": None,
        "paired_event_block_intervals": None,
        "slice_metrics": None,
        "symmetry_diagnostics": None,
        "swap_mapping_and_invariance_evidence": manifest["swap_mapping_and_invariance_evidence"],
        "outer_original_prediction_identities": [],
        "outer_swapped_prediction_identities": [],
        "promotion_decision": manifest["promotion_decision"],
        "adaptive_signal_for_family_08": manifest["adaptive_signal_for_family_08"],
        "development_safe_population": manifest["development_safe_population"],
        "preregistration_commit": manifest["preregistration_commit"],
    }


def _verify_family_8_run(campaign_root: Path, *, recompute_all: bool) -> dict[str, Any]:
    from .families.catboost_specialist import (
        EXPERIMENT_ID,
        FOLD_IDS,
        PROFILE_IDS,
        build_preregistered_profile,
        validate_preregistered_profile,
    )
    from .feature_lineage import build_development_safe_ids

    manifest = read_json(campaign_root / "runs/family-08-catboost-specialist/manifest.json")
    profile_path = campaign_root / manifest["profile_path"]
    profile = read_json(profile_path)
    validated = validate_preregistered_profile(profile)
    preregistration = read_json(campaign_root / manifest["preregistration_path"])
    if (
        manifest.get("experiment_id") != EXPERIMENT_ID
        or profile != build_preregistered_profile()
        or tuple(validated["profile_ids"]) != PROFILE_IDS
        or canonical_sha256(profile) != manifest["profile_sha256"]
        or file_sha256(profile_path) != manifest["profile_file_sha256"]
        or preregistration["scoring_state"] != "not-started"
        or preregistration["ordered_profile_hashes"] != validated["profile_hashes"]
        or preregistration["representation_hashes"] != validated["representation_hashes"]
        or canonical_sha256(preregistration) != manifest["preregistration_sha256"]
    ):
        raise ValueError("family 8 profile or preregistration identity mismatch")
    local_artifact = campaign_root / manifest["artifact_path"]
    artifact_root = local_artifact if local_artifact.is_dir() else FIXED_FAMILY_8_ARTIFACT
    inventory = tree_inventory(artifact_root)
    if (
        inventory.tree_sha256 != manifest["artifact_tree_sha256"]
        or inventory.file_count != manifest["artifact_file_count"]
    ):
        raise ValueError("family 8 artifact inventory mismatch")
    gate = AccessLedger(campaign_root).gate_status()
    if gate["state"] != "closed" or gate["protected_access_count"] != 0:
        raise ValueError("family 8 verification requires the gate closed with zero access")
    safe_ids, retired_ids = build_development_safe_ids(
        read_json(campaign_root / "baseline/fold-manifest.json")
    )
    development = manifest["development_safe_population"]
    if (
        len(safe_ids) != 3_089
        or len(retired_ids) != 178
        or development["development_safe_id_count"] != 3_089
        or development["retired_id_count"] != 178
        or development["development_max_date"] != "2025-12-13"
        or development["asserted_before_row_or_target_decode"] is not True
    ):
        raise ValueError("family 8 development-safe population evidence differs")
    if manifest["exit_state"] != "failed":
        raise ValueError("family 8 verifier expected the frozen terminal failure result")
    failure = manifest["terminal_failure"]
    attempts = read_jsonl(campaign_root / manifest["attempts_path"])
    acquired = read_json(artifact_root / "gpu-lease-acquired.json")
    released = read_json(artifact_root / "gpu-lease-released.json")
    runtime = read_json(artifact_root / "runtime.json")
    safety = read_json(artifact_root / "safety.json")
    dependency = read_json(artifact_root / "dependency-evidence.json")
    dependency_inventory = tree_inventory(FIXED_FAMILY_7_ARTIFACT)
    schema = read_json(artifact_root / "representation-schema.json")
    if (
        failure["attempt_ordinal"] != 1
        or failure["stage"] != "pre-construction-dependency-resolution"
        or any(
            failure[key] is not False
            for key in (
                "construction_started",
                "row_decode_started",
                "target_decode_started",
                "fit_started",
                "outer_labels_scored",
                "retry_performed",
            )
        )
        or file_sha256(artifact_root / failure["stderr_path"]) != failure["stderr_sha256"]
        or len(attempts) != 1
        or attempts[0].get("retry") is not False
        or acquired["lease_id"] != released["lease_id"]
        or acquired["lease_id"] != runtime["gpu_lease_id"]
        or acquired["pid"] != released["pid"]
        or safety["gpu_lease_count"] != 1
        or safety["production_attempt_count"] != 1
        or safety["retry_count"] != 0
        or safety["serialized"] is not True
        or safety["database_access"] != {"used": False, "sql": None, "urls": []}
        or dependency["exit_state"] != "failed"
        or dependency["data_path"] is not None
        or dependency["outer_prediction_identities"]
        or dependency["artifact_tree_sha256"] != dependency_inventory.tree_sha256
        or schema["representation_hashes"] != validated["representation_hashes"]
        or schema["fold_ids"] != list(FOLD_IDS)
    ):
        raise ValueError("family 8 failure, dependency, representation, or invocation evidence differs")
    if manifest["outer_prediction_identities"]:
        raise ValueError("family 8 pre-construction failure unexpectedly recorded predictions")
    if recompute_all:
        _assert_equal(read_json(artifact_root / "failure.json"), failure, "family 8 failure")
        _assert_equal(dependency, manifest["dependency_evidence"], "family 8 dependency")
        result = read_json(artifact_root / "result.json")
        for key in (
            "status",
            "terminal_failure",
            "metrics",
            "incumbent_metrics",
            "metric_deltas",
            "paired_event_block_intervals",
            "calibration_gaps",
            "subgroup_gaps",
            "train_gaps",
            "capacity_diagnostics",
            "representation_comparison",
            "representation_schema_and_lineage_hashes",
            "outer_prediction_identities",
            "promotion_decision",
            "adaptive_signal_for_family_09",
            "development_safe_population",
            "dependency_evidence",
            "gate_access_count",
        ):
            _assert_equal(result[key], manifest[key], f"family 8 {key}")
    return {
        "experiment_id": EXPERIMENT_ID,
        "status": "failed",
        "gate_access_count": gate["protected_access_count"],
        "artifact_tree_sha256": inventory.tree_sha256,
        "artifact_file_count": inventory.file_count,
        "profile_count": validated["profile_count"],
        "profile_hashes": validated["profile_hashes"],
        "representation_hashes": validated["representation_hashes"],
        "attempt_count": 1,
        "retry_count": 0,
        "terminal_failure": failure,
        "metrics": None,
        "paired_event_block_intervals": None,
        "calibration_gaps": None,
        "subgroup_gaps": None,
        "train_gaps": None,
        "capacity_diagnostics": None,
        "representation_comparison": manifest["representation_comparison"],
        "outer_prediction_identities": [],
        "promotion_decision": manifest["promotion_decision"],
        "adaptive_signal_for_family_09": manifest["adaptive_signal_for_family_09"],
        "development_safe_population": development,
        "preregistration_commit": manifest["preregistration_commit"],
    }


def _verify_family_9_run(campaign_root: Path, *, recompute_all: bool) -> dict[str, Any]:
    from .families.capacity_foundation import (
        CANDIDATE_IDS,
        build_preregistered_profile,
        context_cache_key,
        validate_preregistered_profile,
    )

    campaign_root = Path(campaign_root)
    experiment_id = CAMPAIGN_FAMILY_IDS[8]
    manifest = read_json(campaign_root / "runs/family-09-capacity-foundation/manifest.json")
    if manifest["experiment_id"] != experiment_id:
        raise ValueError("family 9 manifest experiment ID differs")
    profile_path = campaign_root / manifest["profile_path"]
    profile = read_json(profile_path)
    validated = validate_preregistered_profile(profile)
    if (
        profile != build_preregistered_profile()
        or canonical_sha256(profile) != manifest["profile_sha256"]
        or file_sha256(profile_path) != manifest["profile_file_sha256"]
        or manifest["preregistration_commit"] == ""
    ):
        raise ValueError("family 9 preregistration identity differs")
    local_artifact = campaign_root / manifest["artifact_path"]
    artifact_root = local_artifact if local_artifact.is_dir() else FIXED_FAMILY_9_ARTIFACT
    inventory = tree_inventory(artifact_root)
    if (
        inventory.tree_sha256 != manifest["artifact_tree_sha256"]
        or inventory.file_count != manifest["artifact_file_count"]
        or inventory.total_bytes != manifest["artifact_total_bytes"]
    ):
        raise ValueError("family 9 artifact inventory differs")
    attempts = read_jsonl(campaign_root / manifest["attempts_path"])
    launched = [row for row in attempts if row.get("state") == "launched"]
    exited = [row for row in attempts if row.get("state") == "exited"]
    if (
        len(launched) != len(exited)
        or len(launched) > 6
        or [row["candidate_id"] for row in launched] != [row["candidate_id"] for row in exited]
        or any(row.get("retry") is not False for row in attempts)
        or tuple(row["candidate_id"] for row in launched) != CANDIDATE_IDS[: len(launched)]
    ):
        raise ValueError("family 9 attempt serialization or fit bound differs")
    acquired = read_json(artifact_root / "gpu-lease-acquired.json")
    released = read_json(artifact_root / "gpu-lease-released.json")
    safety = read_json(artifact_root / "safety.json")
    if (
        acquired["lease_id"] != released["lease_id"]
        or acquired["pid"] != released["pid"]
        or safety["gpu_lease_count"] != 1
        or safety["production_attempt_count"] != 1
        or safety["candidate_fit_launch_count"] != len(launched)
        or safety["retry_count"] != 0
        or safety["serialized"] is not True
        or safety["retired_label_reads"] != 0
        or safety["database_access"] != {"used": False, "sql": None, "urls": []}
    ):
        raise ValueError("family 9 safety evidence differs")
    gate = AccessLedger(campaign_root).gate_status()
    if gate["state"] != "closed" or gate["protected_access_count"] != 0:
        raise ValueError("family 9 requires the gate closed with zero access")
    result = read_json(artifact_root / "result.json")
    result_keys = (
        "status",
        "terminal_failure",
        "candidate_fit_count",
        "candidate_results",
        "selected_profiles",
        "control_metrics",
        "paired_event_block_intervals",
        "capacity_diagnostics",
        "comparison_scope",
        "context_lineage",
        "label_invariance",
        "outer_prediction_identities",
        "promotion_decision",
        "adaptive_signal_for_family_10",
        "development_safe_population",
        "gate_access_count",
    )
    for key in result_keys:
        _assert_equal(result[key], manifest[key], f"family 9 {key}")
    if manifest["development_safe_population"] != {
        "asserted_before_target_decode": True,
        "development_safe_id_count": 3_089,
        "development_max_date": "2025-12-13",
        "retired_id_count": 178,
    }:
        raise ValueError("family 9 safe population evidence differs")
    candidate_results = manifest["candidate_results"]
    if len(candidate_results) != manifest["candidate_fit_count"]:
        raise ValueError("family 9 completed fit count differs")
    recomputed_results = []
    for item in candidate_results:
        prediction_path = artifact_root / item["outer_prediction_identity"]["path"]
        records = read_jsonl(prediction_path)
        metrics = reduce_predictions(records).as_dict()
        if (
            metrics != item["outer_metrics"]
            or file_sha256(prediction_path) != item["outer_prediction_identity"]["sha256"]
            or len(records) != item["outer_prediction_identity"]["row_count"] != 282
            or item["evaluation_label_count_in_prediction_request"] != 0
            or len(set(item["label_invariance_prediction_sha256s"].values())) != 1
            or item["synthetic_irrelevant_future_label_sha256s"]["before"]
            == item["synthetic_irrelevant_future_label_sha256s"]["after"]
        ):
            raise ValueError("family 9 Original prediction reduction differs")
        model_root = prediction_path.parent / "model"
        model_inventory = tree_inventory(model_root)
        if (
            model_inventory.tree_sha256 != item["model_tree_sha256"]
            or model_inventory.file_count != item["model_file_count"]
            or len(item["model_names"]) != 1
        ):
            raise ValueError("family 9 model or hidden-node evidence differs")
        lineage = manifest["context_lineage"][item["candidate_id"]]
        candidate = next(
            value for value in profile["candidates"] if value["id"] == item["candidate_id"]
        )
        if (
            lineage["context_row_count"] > lineage["context_length_cap"]
            or lineage["max_context_date"] >= "2025-01-01"
            or lineage["evaluation_label_count_in_prediction_request"] != 0
            or lineage["checkpoint"] != candidate["checkpoint"]
        ):
            raise ValueError("family 9 context lineage differs")
        recomputed_results.append(item)
    selected = {}
    for family in ("FASTAI", "MITRA", "TABICL"):
        available = [item for item in recomputed_results if item["model_family"] == family]
        if available:
            selected[family] = min(
                available,
                key=lambda item: (
                    item["inner_metrics"]["log_loss"],
                    CANDIDATE_IDS.index(item["candidate_id"]),
                ),
            )["candidate_id"]
    _assert_equal(selected, manifest["selected_profiles"], "family 9 inner selection")
    if manifest["comparison_scope"] != {
        "candidate_outer_years": [2025],
        "candidate_row_count": 282,
        "family_1_full_development_row_count": 1_108,
        "full_development_comparable": False,
        "campaign_promotion_eligible": False,
        "use": "bounded-2025-capacity-probe-for-family-10",
    }:
        raise ValueError("family 9 bounded comparison scope differs")
    if (
        manifest["promotion_decision"]["promoted"] is not False
        or manifest["promotion_decision"]["campaign_promotion_eligible"] is not False
        or manifest["promotion_decision"]["incumbent_after"]
        != "family-01-weighted-v8-control"
    ):
        raise ValueError("family 9 bounded probe cannot promote")
    invariance = manifest["label_invariance"]
    if (
        invariance["evaluation_label_removal"] != "byte-identical"
        or invariance["evaluation_label_permutation"] != "byte-identical"
        or invariance["irrelevant_future_label_change"] != "byte-identical"
        or invariance["evaluation_label_reads_for_prediction"] != 0
        or invariance["retired_label_reads"] != 0
    ):
        raise ValueError("family 9 label invariance differs")
    status = manifest["exit_state"]
    if status == "complete" and len(candidate_results) != 6:
        raise ValueError("complete family 9 result lacks six fits")
    if status == "failed" and manifest["terminal_failure"] is None:
        raise ValueError("failed family 9 result lacks terminal evidence")
    return {
        "experiment_id": experiment_id,
        "status": status,
        "gate_access_count": 0,
        "artifact_tree_sha256": inventory.tree_sha256,
        "artifact_file_count": inventory.file_count,
        "menu_count": validated["menu_count"],
        "candidate_fit_count": len(candidate_results),
        "attempt_count": len(launched),
        "retry_count": 0,
        "terminal_failure": manifest["terminal_failure"],
        "metrics": None if not candidate_results else min(
            candidate_results, key=lambda item: item["outer_metrics"]["log_loss"]
        )["outer_metrics"],
        "paired_event_block_intervals": manifest["paired_event_block_intervals"],
        "capacity_diagnostics": manifest["capacity_diagnostics"],
        "comparison_scope": manifest["comparison_scope"],
        "context_lineage": manifest["context_lineage"],
        "label_invariance": manifest["label_invariance"],
        "outer_prediction_identities": manifest["outer_prediction_identities"],
        "promotion_decision": manifest["promotion_decision"],
        "adaptive_signal_for_family_10": manifest["adaptive_signal_for_family_10"],
        "development_safe_population": manifest["development_safe_population"],
        "preregistration_commit": manifest["preregistration_commit"],
    }


def _verify_family_10_run(campaign_root: Path, *, recompute_all: bool) -> dict[str, Any]:
    from .families.outcome_decomposition import (
        COMPONENT_IDS,
        VARIANT_IDS,
        build_preregistered_profile,
        validate_preregistered_profile,
    )
    from .outcome_decomposition import combine_law_of_total_probability

    campaign_root = Path(campaign_root)
    experiment_id = CAMPAIGN_FAMILY_IDS[9]
    manifest = read_json(campaign_root / "runs/family-10-outcome-decomposition/manifest.json")
    if manifest["experiment_id"] != experiment_id:
        raise ValueError("family 10 manifest experiment ID differs")
    profile_path = campaign_root / manifest["profile_path"]
    profile = read_json(profile_path)
    validated = validate_preregistered_profile(profile)
    if (
        profile != build_preregistered_profile()
        or canonical_sha256(profile) != manifest["profile_sha256"]
        or file_sha256(profile_path) != manifest["profile_file_sha256"]
        or not manifest["preregistration_commit"]
        or validated["variant_ids"] != list(VARIANT_IDS)
    ):
        raise ValueError("family 10 preregistration identity differs")
    local_artifact = campaign_root / manifest["artifact_path"]
    artifact_root = local_artifact if local_artifact.is_dir() else FIXED_FAMILY_10_ARTIFACT
    inventory = tree_inventory(artifact_root)
    if (
        inventory.tree_sha256 != manifest["artifact_tree_sha256"]
        or inventory.file_count != manifest["artifact_file_count"]
        or inventory.total_bytes != manifest["artifact_total_bytes"]
    ):
        raise ValueError("family 10 artifact tree inventory differs")
    attempts = read_jsonl(campaign_root / manifest["attempts_path"])
    launched = [row for row in attempts if row.get("state") == "launched"]
    exited = [row for row in attempts if row.get("state") == "exited"]
    expected_fits = [
        (str(year), component)
        for year in (2022, 2023, 2024, 2025)
        for component in COMPONENT_IDS
    ]
    if (
        len(launched) != 12
        or len(exited) != 12
        or [(row["fold"], row["component_id"]) for row in launched] != expected_fits
        or [(row["fold"], row["component_id"]) for row in exited] != expected_fits
        or any(row.get("retry") is not False for row in attempts)
        or any(row.get("exit_code") != 0 for row in exited)
    ):
        raise ValueError("family 10 serialized component attempt ledger differs")
    acquired = read_json(artifact_root / "production-lease-acquired.json")
    released = read_json(artifact_root / "production-lease-released.json")
    safety = read_json(artifact_root / "safety.json")
    if (
        acquired["lease_id"] != released["lease_id"]
        or acquired["pid"] != released["pid"]
        or safety["production_lease_count"] != 1
        or safety["production_process_count"] != 1
        or safety["component_fit_launch_count"] != 12
        or safety["retry_count"] != 0
        or safety["serialized"] is not True
        or safety["retired_label_reads"] != 0
        or safety["database_access"] != {"used": False, "sql": None, "urls": []}
    ):
        raise ValueError("family 10 safety evidence differs")
    gate = AccessLedger(campaign_root).gate_status()
    if gate["state"] != "closed" or gate["protected_access_count"] != 0:
        raise ValueError("family 10 requires the gate closed with zero access")
    safe_population = {
        "asserted_before_target_decode": True,
        "development_safe_id_count": 3_089,
        "development_max_date": "2025-12-13",
        "retired_id_count": 178,
    }
    if manifest["development_safe_population"] != safe_population:
        raise ValueError("family 10 safe population evidence differs")
    result = read_json(artifact_root / "result.json")
    result_keys = (
        "status",
        "terminal_failure",
        "variant_results",
        "component_fit_lineage_and_support",
        "component_train_outer_gaps",
        "component_prediction_identities",
        "combined_prediction_identities",
        "control_metrics",
        "paired_event_block_intervals",
        "promotion_decision",
        "development_final_incumbent_identity",
        "development_safe_population",
        "comparison_scope",
        "gate_access_count",
    )
    for key in result_keys:
        _assert_equal(result[key], manifest[key], f"family 10 {key}")
    lineage = manifest["component_fit_lineage_and_support"]
    row_counts = {"2022": 282, "2023": 251, "2024": 293, "2025": 282}
    for year, expected_count in row_counts.items():
        if set(lineage[year]) != set(COMPONENT_IDS):
            raise ValueError("family 10 component lineage menu differs")
        for component_id in COMPONENT_IDS:
            evidence = lineage[year][component_id]
            identity = evidence["outer_prediction_identity"]
            prediction_path = artifact_root / identity["path"]
            records = read_jsonl(prediction_path)
            model = evidence["model_identity"]
            model_path = prediction_path.parent / model["path"]
            if (
                identity["row_count"] != expected_count
                or len(records) != expected_count
                or file_sha256(prediction_path) != identity["sha256"]
                or file_sha256(model_path) != model["sha256"]
                or evidence["fit_scope"] != "prior-only"
                or evidence["outer_label_fit_count"] != 0
                or evidence["fallback_used"] is not False
                or evidence["support"] < 120
                or any(row.get("outer_label_reads") != 0 for row in records)
            ):
                raise ValueError("family 10 component prediction or support evidence differs")

    status = manifest["exit_state"]
    recomputed_metrics: dict[str, dict[str, Any]] = {}
    recomputed_intervals: dict[str, Any] = {}
    variant_rows: dict[str, list[dict[str, Any]]] = {}
    if status == "complete":
        if manifest["terminal_failure"] is not None:
            raise ValueError("complete family 10 result contains terminal failure evidence")
        if [row["variant_id"] for row in manifest["variant_results"]] != list(VARIANT_IDS):
            raise ValueError("family 10 result order differs from preregistration")
        profile_hashes = {row["id"]: row["profile_sha256"] for row in profile["variants"]}
        expected_identity: list[tuple[str, str]] | None = None
        for variant_result in manifest["variant_results"]:
            variant_id = variant_result["variant_id"]
            identity = manifest["combined_prediction_identities"][variant_id]
            if (
                variant_result["prediction_identity"] != identity
                or variant_result["profile_sha256"] != profile_hashes[variant_id]
                or identity["row_count"] != 1_108
                or identity["boundary"] != "Original"
                or identity["outer_years"] != [2022, 2023, 2024, 2025]
            ):
                raise ValueError("family 10 combined prediction identity differs")
            path = artifact_root / identity["path"]
            rows = read_jsonl(path)
            row_identity = [(str(row["fight_id"]), str(row["fold"])) for row in rows]
            if (
                len(rows) != 1_108
                or len(set(row_identity)) != 1_108
                or file_sha256(path) != identity["sha256"]
                or any(row.get("candidate_id") != variant_id for row in rows)
                or any(row.get("boundary") != "Original" for row in rows)
                or any(row.get("fit_scope") != "prior-only" for row in rows)
                or any(row.get("outer_label_reads") != 0 for row in rows)
            ):
                raise ValueError("family 10 combined prediction rows differ")
            if expected_identity is None:
                expected_identity = row_identity
            elif row_identity != expected_identity:
                raise ValueError("family 10 variant label/ID/fold identity mismatch")
            if variant_id in {
                "three-component",
                "shrinkage-gated-three-component",
                "constant-prior-fallback",
            }:
                for row in rows:
                    components = row["component_probabilities"]
                    expected_probability = combine_law_of_total_probability(
                        components["decision"],
                        components["decision-win"],
                        components["finish-win"],
                    )
                    if abs(row["probability"] - min(0.98, max(0.02, expected_probability))) > 1e-12:
                        raise ValueError("family 10 law-of-total-probability result differs")
            metrics = reduce_predictions(rows).as_dict()
            _assert_equal(metrics, variant_result["metrics"], f"family 10 {variant_id} metrics")
            recomputed_metrics[variant_id] = metrics
            variant_rows[variant_id] = rows
        control_id = "direct-incumbent-control"
        _assert_equal(recomputed_metrics[control_id], manifest["control_metrics"], "family 10 control metrics")
        for variant_result in manifest["variant_results"]:
            variant_id = variant_result["variant_id"]
            if variant_id == control_id:
                if variant_result["paired_event_block_intervals"] is not None:
                    raise ValueError("family 10 control unexpectedly has a paired interval")
                continue
            interval = event_block_bootstrap_delta(
                variant_rows[variant_id],
                variant_rows[control_id],
                iterations=2_000,
                seed=20260815,
            )
            _assert_equal(
                interval,
                variant_result["paired_event_block_intervals"],
                f"family 10 {variant_id} interval",
            )
            recomputed_intervals[variant_id] = interval
        _assert_equal(recomputed_intervals, manifest["paired_event_block_intervals"], "family 10 intervals")
        best_id = min(
            VARIANT_IDS[1:],
            key=lambda value: (recomputed_metrics[value]["log_loss"], VARIANT_IDS.index(value)),
        )
        best_metrics = recomputed_metrics[best_id]
        best_interval = recomputed_intervals[best_id]
        promoted = (
            best_metrics["log_loss"] < recomputed_metrics[control_id]["log_loss"]
            and best_interval["log_loss_delta"]["upper"] < 0.0
            and best_metrics["brier"] <= recomputed_metrics[control_id]["brier"]
            and best_metrics["accuracy"] >= recomputed_metrics[control_id]["accuracy"] - 0.005
        )
        incumbent_after = best_id if promoted else "family-01-weighted-v8-control"
        expected_promotion = {
            "action": "promote-family-10" if promoted else "retain-family-01-weighted-v8-control",
            "incumbent_before": "family-01-weighted-v8-control",
            "incumbent_after": incumbent_after,
            "promoted": promoted,
            "selected_decomposition_variant": best_id,
            "rule": (
                "full-1,108-row log loss improvement with paired 95% upper bound below zero, "
                "non-worse Brier, and accuracy no more than 0.5 percentage points lower"
            ),
        }
        _assert_equal(expected_promotion, manifest["promotion_decision"], "family 10 promotion")
        chosen_id = best_id if promoted else control_id
        expected_final = {
            "incumbent_id": incumbent_after,
            "candidate_prediction_identity": manifest["combined_prediction_identities"][chosen_id],
            "development_metrics": recomputed_metrics[chosen_id],
            "sealed": False,
            "gate_access_count": 0,
        }
        _assert_equal(
            expected_final,
            manifest["development_final_incumbent_identity"],
            "family 10 development final incumbent",
        )
        if manifest["comparison_scope"] != {
            "outer_years": [2022, 2023, 2024, 2025],
            "outer_row_count": 1_108,
            "family_1_comparable": True,
            "family_9_predictions_used": False,
            "development_only": True,
        }:
            raise ValueError("family 10 comparison scope differs")
    elif status == "failed":
        failure = manifest["terminal_failure"]
        error_path = artifact_root / failure["stderr_path"]
        if file_sha256(error_path) != failure["stderr_sha256"] or failure["retry"] is not False:
            raise ValueError("family 10 terminal failure evidence differs")
    else:
        raise ValueError("unsupported family 10 exit state")
    return {
        "experiment_id": experiment_id,
        "status": status,
        "artifact_tree_sha256": inventory.tree_sha256,
        "artifact_file_count": inventory.file_count,
        "component_fit_count": len(launched),
        "component_prediction_count": sum(
            identity["row_count"]
            for identities in manifest["component_prediction_identities"].values()
            for identity in identities
        ),
        "combined_prediction_count": sum(
            identity["row_count"] for identity in manifest["combined_prediction_identities"].values()
        ),
        "retry_count": 0,
        "terminal_failure": manifest["terminal_failure"],
        "metrics": recomputed_metrics,
        "paired_event_block_intervals": recomputed_intervals,
        "promotion_decision": manifest["promotion_decision"],
        "development_final_incumbent_identity": manifest["development_final_incumbent_identity"],
        "development_safe_population": safe_population,
        "gate_access_count": 0,
        "preregistration_commit": manifest["preregistration_commit"],
    }


def verify_feature_lineage(
    campaign_root: Path,
    experiment_id: str,
    *,
    strict: bool,
) -> dict[str, Any]:
    """Verify the preregistered lineage result, including an honest pre-construction failure."""

    experiment_id = _canonical_family_id(experiment_id)
    if experiment_id == CAMPAIGN_FAMILY_IDS[6]:
        verified = _verify_family_7_run(Path(campaign_root), recompute_all=strict)
        manifest = read_json(Path(campaign_root) / "runs/family-07-matchup-geometry/manifest.json")
        if (
            manifest["data_path"] is not None
            or manifest["data_sha256"] is not None
            or manifest["outer_original_prediction_identities"]
            or manifest["outer_swapped_prediction_identities"]
        ):
            raise ValueError("family 7 pre-construction failure unexpectedly materialized data")
        return {
            "experiment_id": experiment_id,
            "status": "failed-pre-construction",
            "profile_count": verified["profile_count"],
            "lineage_materialized": False,
            "failure_evidence_verified": True,
            "construction_started": False,
            "development_safe_id_count": verified["development_safe_population"]["development_safe_id_count"],
            "retired_id_count": verified["development_safe_population"]["retired_id_count"],
            "outer_label_selection_count": 0,
            "gate_access_count": verified["gate_access_count"],
        }
    if experiment_id != CAMPAIGN_FAMILY_IDS[5]:
        raise ValueError("feature-lineage verifier owns only families 6 and 7")
    verified = _verify_family_6_run(Path(campaign_root), recompute_all=strict)
    manifest = read_json(Path(campaign_root) / "runs/family-06-fighter-states/manifest.json")
    if verified["status"] == "failed":
        if (
            manifest["data_path"] is not None
            or manifest["data_sha256"] is not None
            or manifest["outer_prediction_identities"]
        ):
            raise ValueError("pre-construction failure unexpectedly materialized feature data")
        return {
            "experiment_id": experiment_id,
            "status": "failed-pre-construction",
            "profile_count": verified["profile_count"],
            "lineage_materialized": False,
            "failure_evidence_verified": True,
            "construction_started": False,
            "outer_label_selection_count": 0,
            "gate_access_count": verified["gate_access_count"],
        }
    raise ValueError("unsupported family 6 lineage result")


def _contains_forbidden_database_reference(searchable: bytes) -> bool:
    """Detect database references without treating metric digits as a port."""

    compact = b"".join(searchable.lower().split())
    return any(
        token in compact
        for token in (
            b"clankerfights",
            b"postgresql://",
            b"postgres://",
            b"localhost:5432",
            b"127.0.0.1:5432",
            b'"port":5432',
            b"'port':5432",
        )
    )


def audit_campaign_safety(
    campaign_root: Path,
    *,
    through: str,
    require_gate_closed: bool,
) -> dict[str, Any]:
    """Audit the sole lease, retry count, database manifest, and gate state."""

    campaign_root = Path(campaign_root)
    through = _canonical_family_id(through)
    if through not in CAMPAIGN_FAMILY_IDS[5:10]:
        raise ValueError("safety audit is bounded through families 6 through 10")
    if through == CAMPAIGN_FAMILY_IDS[9]:
        verified = _verify_family_10_run(campaign_root, recompute_all=True)
        manifest = read_json(campaign_root / "runs/family-10-outcome-decomposition/manifest.json")
        fixed_artifact = FIXED_FAMILY_10_ARTIFACT
    elif through == CAMPAIGN_FAMILY_IDS[8]:
        verified = _verify_family_9_run(campaign_root, recompute_all=True)
        manifest = read_json(campaign_root / "runs/family-09-capacity-foundation/manifest.json")
        fixed_artifact = FIXED_FAMILY_9_ARTIFACT
    elif through == CAMPAIGN_FAMILY_IDS[7]:
        verified = _verify_family_8_run(campaign_root, recompute_all=True)
        manifest = read_json(campaign_root / "runs/family-08-catboost-specialist/manifest.json")
        fixed_artifact = FIXED_FAMILY_8_ARTIFACT
    elif through == CAMPAIGN_FAMILY_IDS[6]:
        verified = _verify_family_7_run(campaign_root, recompute_all=True)
        manifest = read_json(campaign_root / "runs/family-07-matchup-geometry/manifest.json")
        fixed_artifact = FIXED_FAMILY_7_ARTIFACT
    else:
        verified = _verify_family_6_run(campaign_root, recompute_all=True)
        manifest = read_json(campaign_root / "runs/family-06-fighter-states/manifest.json")
        fixed_artifact = FIXED_FAMILY_6_ARTIFACT
    artifact_root = campaign_root / manifest["artifact_path"]
    if not artifact_root.is_dir():
        artifact_root = fixed_artifact
    safety = read_json(artifact_root / "safety.json")
    searchable = b"\n".join(
        path.read_bytes().lower()
        for path in sorted(artifact_root.rglob("*"))
        if path.is_file() and path.suffix.lower() in {".json", ".jsonl", ".txt", ".log"}
    )
    if _contains_forbidden_database_reference(searchable):
        raise ValueError("family safety evidence contains a forbidden database token")
    gate = AccessLedger(campaign_root).gate_status()
    if require_gate_closed and (gate["state"] != "closed" or gate["protected_access_count"] != 0):
        raise ValueError("campaign gate is not closed with zero access")
    common = {
        "through": through,
        "status": verified["status"],
        "retry_count": safety["retry_count"],
        "serialized": safety["serialized"],
        "database_access": safety["database_access"],
        "forbidden_database_token_count": 0,
        "gate_state": gate["state"],
        "gate_access_count": gate["protected_access_count"],
    }
    if through == CAMPAIGN_FAMILY_IDS[9]:
        return {
            **common,
            "gpu_lease_count": safety["gpu_fit_count"],
            "production_lease_count": safety["production_lease_count"],
            "production_attempt_count": safety["production_process_count"],
            "component_fit_launch_count": safety["component_fit_launch_count"],
        }
    return {
        **common,
        "gpu_lease_count": safety["gpu_lease_count"],
        "production_attempt_count": safety["production_attempt_count"],
    }


def replay_campaign_decisions(campaign_root: Path, *, through: str) -> dict[str, Any]:
    campaign_root = Path(campaign_root)
    through = _canonical_family_id(through)
    if through not in CAMPAIGN_FAMILY_IDS[:10]:
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
                "adaptive_signal_for_family_03",
                verified.get(
                    "adaptive_signal_for_family_04",
                    verified.get(
                        "adaptive_signal_for_family_05",
                        verified.get(
                            "adaptive_signal_for_family_06",
                            verified.get(
                                "adaptive_signal_for_family_07",
                                verified.get(
                                    "adaptive_signal_for_family_08",
                                    verified.get(
                                        "adaptive_signal_for_family_09",
                                        verified.get("adaptive_signal_for_family_10"),
                                    ),
                                ),
                            ),
                        ),
                    ),
                ),
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
