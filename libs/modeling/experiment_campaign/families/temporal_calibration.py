"""Family 3 temporal calibration source-lineage audit and execution."""

from __future__ import annotations

from datetime import date
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

from ..calibration import CALIBRATION_VARIANT_IDS, fit_temporal_calibrator
from ..hashing import canonical_sha256, file_sha256, read_json, tree_inventory
from ..metrics import event_block_bootstrap_delta, reduce_predictions
from ..protocol import AccessLedger


EXPERIMENT_ID = "family-03-temporal-calibration"
FIXED_FAMILY_1_ARTIFACT = Path(
    r"C:\Users\danhm\mma-ai\worktrees\top10-20260815"
    r"\experiments\top10_20260815\artifacts\02-family-01-weighted-v8-control"
)
FIXED_FAMILY_2_ARTIFACT = Path(
    r"C:\Users\danhm\mma-ai\worktrees\top10-20260815"
    r"\experiments\top10_20260815\artifacts\03-family-02-horizon-recency"
)


class SourceLineageError(ValueError):
    """A registered prediction source is inadmissible for calibration."""

    def __init__(self, message: str, audit: Mapping[str, Any]) -> None:
        super().__init__(message)
        self.audit = dict(audit)


def audit_registered_rows(
    inner_rows: Sequence[Mapping[str, Any]],
    outer_rows: Sequence[Mapping[str, Any]],
    *,
    outer_year: int,
) -> dict[str, Any]:
    fit_ids = [str(row["fight_id"]) for row in inner_rows]
    calibration_event_ids = {str(row["event_id"]) for row in inner_rows}
    model_fit_event_ids = {
        str(event_id) for row in inner_rows for event_id in row.get("fit_event_ids", [])
    }
    outer_ids = {str(row["fight_id"]) for row in outer_rows}
    outer_event_ids = {str(row["event_id"]) for row in outer_rows}
    same_fit_rows = [
        row for row in inner_rows if str(row["event_id"]) in {str(value) for value in row.get("fit_event_ids", [])}
    ]
    audit = {
        "outer_year": outer_year,
        "calibration_fit_row_count": len(inner_rows),
        "calibration_fit_id_count": len(set(fit_ids)),
        "calibration_fit_event_count": len(calibration_event_ids),
        "model_fit_event_count": len(model_fit_event_ids),
        "outer_row_count": len(outer_rows),
        "outer_event_count": len(outer_event_ids),
        "calibration_model_fit_overlap_count": len(same_fit_rows),
        "same_fit_row_count": len(same_fit_rows),
        "calibration_outer_id_overlap_count": len(set(fit_ids).intersection(outer_ids)),
        "calibration_outer_event_overlap_count": len(calibration_event_ids.intersection(outer_event_ids)),
        "variant_fit_count": 0,
        "variant_score_count": 0,
    }

    def reject(message: str) -> None:
        audit["status"] = "ineligible"
        audit["reason"] = message
        raise SourceLineageError(message, audit)

    if not inner_rows or not outer_rows:
        reject("registered calibration or outer history is empty")
    boundaries = {row.get("boundary") for row in inner_rows}
    if len(boundaries) != 1 or not boundaries.issubset({"InnerSelection", "Original"}):
        reject("calibration history has an unsupported or shuffled fold boundary")
    if any(row.get("boundary") != "Original" for row in outer_rows):
        reject("outer history must contain Original probabilities")
    if len(fit_ids) != len(set(fit_ids)):
        reject("calibration history contains duplicate IDs")
    if set(fit_ids).intersection(outer_ids):
        reject("calibration fit IDs overlap outer IDs")
    if same_fit_rows:
        reject("calibration event IDs overlap base model-fit event IDs")
    if calibration_event_ids.intersection(outer_event_ids):
        reject("calibration event IDs overlap outer event IDs")
    probabilities = [float(row["probability"]) for row in inner_rows]
    if any(not math.isfinite(value) or not 0.0 <= value <= 1.0 for value in probabilities):
        reject("calibration probabilities are outside range [0, 1]")
    if {int(row["y_true"]) for row in inner_rows} != {0, 1}:
        reject("calibration history must contain both classes")
    fit_dates = [date.fromisoformat(str(row["event_date"])) for row in inner_rows]
    if fit_dates != sorted(fit_dates):
        reject("calibration history is shuffled rather than chronological")
    outer_dates = [date.fromisoformat(str(row["event_date"])) for row in outer_rows]
    if any(value >= min(outer_dates) for value in fit_dates):
        reject("calibration history contains future IDs")
    if any(value.year != outer_year for value in outer_dates):
        reject("outer history year does not match its declared fold")
    audit["status"] = "eligible"
    audit["reason"] = None
    return audit


def _positive_log_loss(labels: Sequence[int], probabilities: Sequence[float]) -> float:
    epsilon = 1e-15
    losses = []
    for label, probability in zip(labels, probabilities, strict=True):
        value = min(max(float(probability), epsilon), 1.0 - epsilon)
        losses.append(-(int(label) * math.log(value) + (1 - int(label)) * math.log(1.0 - value)))
    return sum(losses) / len(losses)


def _chronological_event_split(
    rows: Sequence[Mapping[str, Any]], fraction: float = 0.7
) -> tuple[list[Mapping[str, Any]], list[Mapping[str, Any]]]:
    events: list[str] = []
    for row in rows:
        event_id = str(row["event_id"])
        if not events or events[-1] != event_id:
            events.append(event_id)
    if len(events) < 2:
        raise SourceLineageError(
            "calibration history has fewer than two chronological event blocks",
            {"status": "ineligible", "event_count": len(events)},
        )
    cut = min(max(int(len(events) * fraction), 1), len(events) - 1)
    fit_events = set(events[:cut])
    fit_rows = [row for row in rows if str(row["event_id"]) in fit_events]
    score_rows = [row for row in rows if str(row["event_id"]) not in fit_events]
    return fit_rows, score_rows


def _fit_variant(
    variant_id: str,
    config: Mapping[str, Any],
    fit_rows: Sequence[Mapping[str, Any]],
    score_rows: Sequence[Mapping[str, Any]],
):
    return fit_temporal_calibrator(
        variant_id=variant_id,
        config=config,
        probabilities=[float(row["probability"]) for row in fit_rows],
        labels=[int(row["y_true"]) for row in fit_rows],
        fit_ids=[str(row["fight_id"]) for row in fit_rows],
        fit_dates=[str(row["event_date"]) for row in fit_rows],
        model_fit_ids=[],
        outer_ids=[str(row["fight_id"]) for row in score_rows],
        outer_min_date=min(str(row["event_date"]) for row in score_rows),
    )


def select_and_calibrate_outer(
    history_rows: Sequence[Mapping[str, Any]],
    outer_rows: Sequence[Mapping[str, Any]],
    *,
    outer_year: int,
    variant_configs: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    if tuple(variant_configs) != CALIBRATION_VARIANT_IDS:
        raise ValueError("variant configuration order differs from the frozen menu")
    if not history_rows:
        predictions = [
            {
                **row,
                "original_probability": row["probability"],
                "selected_calibration_variant": "identity",
                "calibration_fit_row_count": 0,
                "calibration_fit_max_date": None,
            }
            for row in outer_rows
        ]
        return {
            "selection": {
                "variant_id": "identity",
                "selection_basis": "identity-only-no-fit",
                "fit_row_count": 0,
                "score_row_count": 0,
                "selection_max_date": None,
                "variant_scores": {},
            },
            "predictions": predictions,
        }
    audit = audit_registered_rows(history_rows, outer_rows, outer_year=outer_year)
    fit_rows, score_rows = _chronological_event_split(history_rows)
    scores: dict[str, float] = {}
    selection_fits: dict[str, Any] = {}
    for variant_id in CALIBRATION_VARIANT_IDS:
        fitted = _fit_variant(variant_id, variant_configs[variant_id], fit_rows, score_rows)
        calibrated = fitted.transform([float(row["probability"]) for row in score_rows])
        scores[variant_id] = _positive_log_loss(
            [int(row["y_true"]) for row in score_rows], calibrated
        )
        selection_fits[variant_id] = fitted.fit_summary
    selected = min(CALIBRATION_VARIANT_IDS, key=lambda value: (scores[value], CALIBRATION_VARIANT_IDS.index(value)))
    final_fit = _fit_variant(selected, variant_configs[selected], history_rows, outer_rows)
    calibrated_outer = final_fit.transform([float(row["probability"]) for row in outer_rows])
    fit_max_date = max(str(row["event_date"]) for row in history_rows)
    predictions = [
        {
            **row,
            "original_probability": row["probability"],
            "probability": calibrated,
            "selected_calibration_variant": selected,
            "calibration_fit_row_count": len(history_rows),
            "calibration_fit_max_date": fit_max_date,
        }
        for row, calibrated in zip(outer_rows, calibrated_outer, strict=True)
    ]
    return {
        "selection": {
            "variant_id": selected,
            "selection_basis": "chronological-event-block-tail",
            "fit_row_count": len(history_rows),
            "selection_fit_row_count": len(fit_rows),
            "score_row_count": len(score_rows),
            "selection_max_date": max(str(row["event_date"]) for row in score_rows),
            "variant_scores": scores,
            "variant_fit_summaries": selection_fits,
            "final_fit_summary": final_fit.fit_summary,
            "lineage_audit": audit,
        },
        "predictions": predictions,
    }


def promotion_decision(
    metrics: Mapping[str, Any],
    intervals: Mapping[str, Any],
) -> dict[str, Any]:
    promote = (
        float(metrics["log_loss_delta"]) < 0.0
        and float(intervals["log_loss_delta"]["upper"]) < 0.0
    )
    return {
        "action": "promote-temporal-calibration" if promote else "retain-family-01-weighted-v8-control",
        "incumbent_before": "family-01-weighted-v8-control",
        "incumbent_after": (
            EXPERIMENT_ID if promote else "family-01-weighted-v8-control"
        ),
        "promoted": promote,
        "rule": "pooled log-loss delta and paired event-block interval upper bound must both be below zero",
    }


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = "".join(
        json.dumps(dict(row), sort_keys=True, separators=(",", ":")) + "\n" for row in rows
    )
    path.write_text(payload, encoding="utf-8")


def _variant_configs(profile: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    variants = profile["calibration_variants"]
    configs = {str(item["id"]): item for item in variants}
    if tuple(configs) != CALIBRATION_VARIANT_IDS:
        raise ValueError("resolved profile differs from the frozen calibration menu")
    return configs


def _append_registry_record(campaign_root: Path, payload: Mapping[str, Any]) -> dict[str, Any]:
    registry_path = campaign_root / "registry.jsonl"
    head_path = campaign_root / "registry-head.json"
    registry_bytes = registry_path.read_bytes()
    records = [json.loads(line) for line in registry_bytes.splitlines()]
    if any(row["payload"]["experiment_id"] == EXPERIMENT_ID for row in records):
        raise ValueError("family 3 already exists in the registry")
    head = read_json(head_path)
    if hashlib.sha256(registry_bytes).hexdigest().upper() != head["registry_prefix_sha256"]:
        raise ValueError("registry head does not match the immutable prefix")
    record = {
        "payload": dict(payload),
        "prefix_sha256_before": head["registry_prefix_sha256"],
        "previous_record_sha256": head["last_record_sha256"],
        "sequence": head["record_count"],
    }
    record["record_sha256"] = canonical_sha256(record)
    appended = json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n"
    after = registry_bytes + appended.encode("utf-8")
    prefix_after = hashlib.sha256(after).hexdigest().upper()
    registry_path.write_bytes(after)
    _write_json(
        head_path,
        {
            "last_record_sha256": record["record_sha256"],
            "record_count": len(records) + 1,
            "registry_bytes": len(after),
            "registry_prefix_sha256": prefix_after,
        },
    )
    return {
        "registry_prefix_sha256_before": record["prefix_sha256_before"],
        "registry_prefix_sha256_after": prefix_after,
        "record_sha256": record["record_sha256"],
    }


def materialize_family_03(
    campaign_root: Path,
    *,
    source_revision: str,
    family_1_artifact: Path = FIXED_FAMILY_1_ARTIFACT,
    family_2_artifact: Path = FIXED_FAMILY_2_ARTIFACT,
) -> dict[str, Any]:
    campaign_root = Path(campaign_root)
    artifact_root = campaign_root / "artifacts/04-family-03-temporal-calibration"
    if artifact_root.exists():
        raise ValueError("family 3 artifact destination already exists")
    profile_path = campaign_root / "profiles/family-03-temporal-calibration.json"
    profile = read_json(profile_path)
    configs = _variant_configs(profile)
    gate = AccessLedger(campaign_root).gate_status()
    if gate["state"] != "closed" or gate["protected_access_count"] != 0:
        raise ValueError("family 3 requires the gate closed with zero access")

    negative_control = []
    for year in (2022, 2023, 2024, 2025):
        inner = _read_jsonl(family_2_artifact / f"fold-{year}/selected-inner-predictions.jsonl")
        outer = _read_jsonl(family_2_artifact / f"fold-{year}/outer-predictions.jsonl")
        try:
            audit_registered_rows(inner, outer, outer_year=year)
        except SourceLineageError as exc:
            negative_control.append(exc.audit)
        else:
            raise ValueError("family 2 selected-inner negative control unexpectedly passed")

    all_predictions: list[dict[str, Any]] = []
    incumbent_predictions: list[dict[str, Any]] = []
    selections = []
    fold_predictions = []
    attempts = []
    for year in (2022, 2023, 2024, 2025):
        history = [
            row
            for source_year in range(2022, year)
            for row in _read_jsonl(family_1_artifact / f"fold-{source_year}/outer-predictions.jsonl")
        ]
        outer = _read_jsonl(family_2_artifact / f"fold-{year}/outer-predictions.jsonl")
        incumbent_predictions.extend(outer)
        calibrated = select_and_calibrate_outer(
            history,
            outer,
            outer_year=year,
            variant_configs=configs,
        )
        selection = {"outer_year": year, **calibrated["selection"]}
        selections.append(selection)
        predictions = calibrated["predictions"]
        all_predictions.extend(predictions)
        fold_root = artifact_root / f"fold-{year}"
        prediction_path = fold_root / "outer-predictions.jsonl"
        selection_path = fold_root / "selection.json"
        _write_jsonl(prediction_path, predictions)
        _write_json(selection_path, selection)
        fold_predictions.append(
            {
                "year": year,
                "path": f"fold-{year}/outer-predictions.jsonl",
                "row_count": len(predictions),
                "sha256": file_sha256(prediction_path),
                "variant_id": selection["variant_id"],
                "calibration_fit_row_count": selection["fit_row_count"],
            }
        )
        if year == 2022:
            attempts.append(
                {
                    "fold": year,
                    "state": "identity-only-no-fit",
                    "variant_id": "identity",
                    "score": None,
                }
            )
        else:
            attempts.extend(
                {
                    "fold": year,
                    "state": "scored-prior-oof",
                    "variant_id": variant_id,
                    "score": selection["variant_scores"][variant_id],
                }
                for variant_id in CALIBRATION_VARIANT_IDS
            )

    candidate_metrics = reduce_predictions(all_predictions).as_dict()
    incumbent_metrics = reduce_predictions(incumbent_predictions).as_dict()
    intervals = event_block_bootstrap_delta(
        all_predictions,
        incumbent_predictions,
        iterations=int(profile["bootstrap"]["iterations"]),
        seed=int(profile["bootstrap"]["seed"]),
    )
    metric_deltas = {
        name: candidate_metrics[name] - incumbent_metrics[name]
        for name in ("log_loss", "brier", "calibration_intercept", "calibration_slope", "ece", "accuracy")
    }
    decision = promotion_decision(metric_deltas, intervals)
    adaptive_signal = {
        "selected_variants": [selection["variant_id"] for selection in selections],
        "fold_log_loss": {
            str(year): candidate_metrics["fold_metrics"][str(year)]["log_loss"]
            for year in (2022, 2023, 2024, 2025)
        },
        "pooled_log_loss_delta": metric_deltas["log_loss"],
        "pooled_ece_delta": metric_deltas["ece"],
    }
    lineage_audit = {
        "negative_control": negative_control,
        "accepted_fold_lineage": [selection.get("lineage_audit") for selection in selections],
        "base_model_retrain_count": 0,
        "gate_access_count": gate["protected_access_count"],
    }
    _write_json(artifact_root / "lineage-audit.json", lineage_audit)
    result = {
        "experiment_id": EXPERIMENT_ID,
        "status": "complete",
        "metrics": candidate_metrics,
        "incumbent_metrics": incumbent_metrics,
        "metric_deltas": metric_deltas,
        "paired_event_block_intervals": intervals,
        "promotion_decision": decision,
        "selected_variants": [selection["variant_id"] for selection in selections],
        "adaptive_signal_for_family_04": adaptive_signal,
        "gate_access_count": gate["protected_access_count"],
        "base_model_retrain_count": 0,
    }
    _write_json(artifact_root / "result.json", result)
    inventory = tree_inventory(artifact_root)

    run_root = campaign_root / f"runs/{EXPERIMENT_ID}"
    _write_jsonl(run_root / "attempts.jsonl", attempts)
    (run_root / "decision.md").write_text(
        f"# Family 3 decision\n\n{decision['action']}: {decision['rule']}.\n",
        encoding="utf-8",
    )
    manifest = {
        **result,
        "kind": "family",
        "exit_state": "complete",
        "artifact_path": "artifacts/04-family-03-temporal-calibration",
        "artifact_tree_sha256": inventory.tree_sha256,
        "artifact_file_count": inventory.file_count,
        "profile_path": "profiles/family-03-temporal-calibration.json",
        "profile_sha256": canonical_sha256(profile),
        "preregistration_path": f"runs/{EXPERIMENT_ID}/preregistration.json",
        "lineage_preregistration_path": f"runs/{EXPERIMENT_ID}/lineage-preregistration.json",
        "attempts_path": f"runs/{EXPERIMENT_ID}/attempts.jsonl",
        "fold_predictions": fold_predictions,
        "selections": selections,
        "lineage_audit_path": "lineage-audit.json",
        "source_revision": source_revision,
        "terminal_failure": None,
    }
    manifest_path = run_root / "manifest.json"
    _write_json(manifest_path, manifest)
    registry = _append_registry_record(
        campaign_root,
        {
            "artifact_path": manifest["artifact_path"],
            "artifact_tree_sha256": inventory.tree_sha256,
            "experiment_id": EXPERIMENT_ID,
            "kind": "family",
            "manifest_path": f"runs/{EXPERIMENT_ID}/manifest.json",
            "manifest_sha256": file_sha256(manifest_path),
            "profile_path": manifest["profile_path"],
            "profile_sha256": manifest["profile_sha256"],
            "status": "complete",
        },
    )
    return {**result, **registry, "artifact_tree_sha256": inventory.tree_sha256}
