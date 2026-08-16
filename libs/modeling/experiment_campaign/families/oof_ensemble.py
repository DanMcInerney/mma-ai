"""Family 4 chronological-OOF recipe selection and materialization."""

from __future__ import annotations

from datetime import date
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

from ..hashing import canonical_sha256, file_sha256, read_json, tree_inventory, write_canonical_json
from ..metrics import event_block_bootstrap_delta, reduce_predictions
from ..protocol import AccessLedger
from ..ensemble import (
    ENSEMBLE_VARIANT_IDS,
    ConstituentError,
    build_ensemble_predictions,
    fit_regularized_nonnegative_oof,
    validate_constituents,
)


EXPERIMENT_ID = "family-04-chronological-oof-ensemble"
FIXED_ARTIFACT_BASE = Path(
    r"C:\Users\danhm\mma-ai\worktrees\top10-20260815"
    r"\experiments\top10_20260815\artifacts"
)
RUN_PATH = "runs/family-04-oof-ensemble"
ARTIFACT_PATH = "artifacts/05-family-04-oof-ensemble"


class OOFLineageError(ValueError):
    """Chronological OOF fit lineage is inadmissible."""

    def __init__(self, message: str, audit: Mapping[str, Any]):
        super().__init__(message)
        self.audit = dict(audit)


def _lineage_failure(message: str, *, row_count: int) -> OOFLineageError:
    return OOFLineageError(
        message,
        {
            "status": "ineligible",
            "candidate_fit_row_count": row_count,
            "variant_fit_count": 0,
            "variant_score_count": 0,
        },
    )


def _aligned_history(
    constituents: Mapping[str, Sequence[Mapping[str, Any]]],
    *,
    constituent_ids: Sequence[str],
    outer_min_date: date,
) -> list[list[Mapping[str, Any]]]:
    if set(constituents) != set(constituent_ids):
        raise _lineage_failure("exact registered constituent history is required", row_count=0)
    ordered = [list(constituents[name]) for name in constituent_ids]
    row_count = len(ordered[0]) if ordered else 0
    if row_count == 0 or any(len(rows) != row_count for rows in ordered):
        raise _lineage_failure("constituent OOF histories must be non-empty and aligned", row_count=row_count)
    for rows in ordered:
        for row in rows:
            if row.get("boundary") == "FULL":
                raise _lineage_failure("FULL OOF nodes are forbidden", row_count=row_count)
            if row.get("boundary") != "Original":
                raise _lineage_failure("OOF history must derive from Original predictions", row_count=row_count)
            event_date = date.fromisoformat(str(row["event_date"]))
            if event_date >= outer_min_date:
                raise _lineage_failure("OOF history must be strictly prior to the outer fold", row_count=row_count)
            context = row.get("context_max_date")
            if context is None or date.fromisoformat(str(context)) >= event_date:
                raise _lineage_failure("OOF history contains contaminated context", row_count=row_count)
    for row_index, reference in enumerate(ordered[0]):
        identity = (reference.get("fight_id"), reference.get("event_id"), reference.get("fold"))
        label = reference.get("y_true")
        for rows in ordered[1:]:
            candidate = rows[row_index]
            candidate_identity = (
                candidate.get("fight_id"),
                candidate.get("event_id"),
                candidate.get("fold"),
            )
            if candidate_identity != identity:
                raise _lineage_failure("OOF fight/event/fold IDs do not align", row_count=row_count)
            if candidate.get("y_true") != label:
                raise _lineage_failure("OOF label mismatch", row_count=row_count)
    return ordered


def _log_loss(rows: Sequence[Mapping[str, Any]]) -> float:
    losses = []
    for row in rows:
        probability = min(max(float(row["probability"]), 1e-12), 1.0 - 1e-12)
        label = int(row["y_true"])
        losses.append(-(label * math.log(probability) + (1 - label) * math.log(1 - probability)))
    return sum(losses) / len(losses)


def _select_best_constituent(
    constituents: Mapping[str, Sequence[Mapping[str, Any]]],
    constituent_ids: Sequence[str],
) -> tuple[str, dict[str, float]]:
    scores = {name: _log_loss(constituents[name]) for name in constituent_ids}
    selected = min(constituent_ids, key=lambda name: (scores[name], constituent_ids.index(name)))
    return selected, scores


def select_recipe_for_outer(
    historical_constituents: Mapping[str, Sequence[Mapping[str, Any]]],
    outer_constituents: Mapping[str, Sequence[Mapping[str, Any]]],
    *,
    outer_year: int,
    profile: Mapping[str, Any],
) -> dict[str, Any]:
    constituent_ids = tuple(profile["constituent_ids"])
    if tuple(profile["selection_tie_break"]) != ENSEMBLE_VARIANT_IDS:
        raise ConstituentError("selection tie break differs from the exact five-recipe menu")
    current = str(profile["current_constituent_id"])
    if current not in constituent_ids:
        raise ConstituentError("current ensemble constituent is not registered")
    outer_rows = list(outer_constituents[current])
    if not outer_rows:
        raise ConstituentError("outer constituent set is empty")
    outer_min = min(date.fromisoformat(str(row["event_date"])) for row in outer_rows)
    outer_max = max(date.fromisoformat(str(row["event_date"])) for row in outer_rows)
    validate_constituents(
        outer_constituents,
        expected_constituent_ids=constituent_ids,
        outer_fold=str(outer_year),
        outer_max_date=outer_max.isoformat(),
    )
    if not historical_constituents:
        return {
            "selection": {
                "outer_year": outer_year,
                "selected_recipe_id": "current-autogluon-tune-ensemble",
                "selected_constituent_id": current,
                "selection_basis": "earliest-fold-current-ensemble-no-fit",
                "fit_role": "inner-chronological-oof",
                "fit_row_count": 0,
                "fit_folds": [],
                "fit_max_date": None,
                "recipe_scores": {},
                "constituent_scores": {},
                "weights": {name: 1.0 if name == current else 0.0 for name in constituent_ids},
            },
            "predictions": build_ensemble_predictions(
                outer_constituents,
                recipe_id="current-autogluon-tune-ensemble",
                selected_constituent_id=current,
            ),
        }

    history = _aligned_history(
        historical_constituents,
        constituent_ids=constituent_ids,
        outer_min_date=outer_min,
    )
    best_constituent, constituent_scores = _select_best_constituent(
        historical_constituents, constituent_ids
    )
    weights = fit_regularized_nonnegative_oof(
        historical_constituents,
        fit_role="inner-chronological-oof",
        shrinkage=float(profile["regularization_shrinkage"]),
        foundation_constituent_ids=tuple(profile["foundation_constituent_ids"]),
        foundation_aggregate_cap=float(profile["foundation_aggregate_cap"]),
        solver=profile["solver"],
    )
    history_predictions = {
        "best-single": build_ensemble_predictions(
            historical_constituents,
            recipe_id="best-single",
            selected_constituent_id=best_constituent,
        ),
        "current-autogluon-tune-ensemble": build_ensemble_predictions(
            historical_constituents,
            recipe_id="current-autogluon-tune-ensemble",
            selected_constituent_id=current,
        ),
        "median-probability-blend": build_ensemble_predictions(
            historical_constituents, recipe_id="median-probability-blend"
        ),
        "rank-probability-blend": build_ensemble_predictions(
            historical_constituents, recipe_id="rank-probability-blend"
        ),
        "regularized-nonnegative-oof-blend": build_ensemble_predictions(
            historical_constituents,
            recipe_id="regularized-nonnegative-oof-blend",
            weights=weights,
        ),
    }
    scores = {recipe_id: _log_loss(rows) for recipe_id, rows in history_predictions.items()}
    selected_recipe = min(
        ENSEMBLE_VARIANT_IDS,
        key=lambda recipe_id: (scores[recipe_id], ENSEMBLE_VARIANT_IDS.index(recipe_id)),
    )
    selected_constituent = (
        best_constituent if selected_recipe == "best-single" else current
    )
    predictions = build_ensemble_predictions(
        outer_constituents,
        recipe_id=selected_recipe,
        selected_constituent_id=selected_constituent,
        weights=weights if selected_recipe == "regularized-nonnegative-oof-blend" else None,
    )
    fit_rows = history[0]
    return {
        "selection": {
            "outer_year": outer_year,
            "selected_recipe_id": selected_recipe,
            "selected_constituent_id": selected_constituent,
            "selection_basis": "prior-chronological-oof-log-loss",
            "fit_role": "inner-chronological-oof",
            "fit_row_count": len(fit_rows),
            "fit_folds": sorted({str(row["fold"]) for row in fit_rows}),
            "fit_max_date": max(str(row["event_date"]) for row in fit_rows),
            "recipe_scores": scores,
            "constituent_scores": constituent_scores,
            "weights": weights,
        },
        "predictions": predictions,
    }


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_bytes().splitlines()]


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(
        b"".join(
            json.dumps(dict(row), sort_keys=True, separators=(",", ":")).encode("utf-8") + b"\n"
            for row in rows
        )
    )


def _append_registry(campaign_root: Path, payload: Mapping[str, Any]) -> dict[str, Any]:
    registry_path = campaign_root / "registry.jsonl"
    head_path = campaign_root / "registry-head.json"
    before = registry_path.read_bytes()
    records = [json.loads(line) for line in before.splitlines()]
    if any(record["payload"]["experiment_id"] == EXPERIMENT_ID for record in records):
        raise ValueError("family 4 already exists in the registry")
    head = read_json(head_path)
    prefix_before = hashlib.sha256(before).hexdigest().upper()
    if prefix_before != head["registry_prefix_sha256"]:
        raise ValueError("registry head does not match the immutable prefix")
    record = {
        "payload": dict(payload),
        "prefix_sha256_before": prefix_before,
        "previous_record_sha256": head["last_record_sha256"],
        "sequence": head["record_count"],
    }
    record["record_sha256"] = canonical_sha256(record)
    after = before + json.dumps(record, sort_keys=True, separators=(",", ":")).encode("utf-8") + b"\n"
    prefix_after = hashlib.sha256(after).hexdigest().upper()
    registry_path.write_bytes(after)
    write_canonical_json(
        head_path,
        {
            "last_record_sha256": record["record_sha256"],
            "record_count": len(records) + 1,
            "registry_bytes": len(after),
            "registry_prefix_sha256": prefix_after,
        },
    )
    return {
        "record_sha256": record["record_sha256"],
        "registry_prefix_sha256_before": prefix_before,
        "registry_prefix_sha256_after": prefix_after,
    }


def promotion_decision(
    candidate_metrics: Mapping[str, Any],
    intervals: Mapping[str, Any],
) -> dict[str, Any]:
    promote = (
        float(candidate_metrics["log_loss_delta"]) < 0.0
        and float(intervals["log_loss_delta"]["upper"]) < 0.0
    )
    return {
        "action": "promote-family-04" if promote else "retain-family-01-weighted-v8-control",
        "incumbent_before": "family-01-weighted-v8-control",
        "incumbent_after": EXPERIMENT_ID if promote else "family-01-weighted-v8-control",
        "promoted": promote,
        "rule": "pooled log-loss delta and paired event-block interval upper bound must both be below zero",
    }


def _metric_gaps(candidate: Mapping[str, Any], incumbent: Mapping[str, Any]) -> tuple[dict, dict]:
    calibration = {
        name: float(candidate[name]) - float(incumbent[name])
        for name in ("calibration_intercept", "calibration_slope", "ece")
    }
    subgroup = {
        group: {
            metric: float(candidate["subgroup_metrics"][group][metric])
            - float(incumbent["subgroup_metrics"][group][metric])
            for metric in ("log_loss", "brier", "accuracy")
        }
        for group in sorted(candidate["subgroup_metrics"])
    }
    return calibration, subgroup


def materialize_family_04(
    campaign_root: Path,
    *,
    source_revision: str,
    preregistration_commit: str,
    artifact_base: Path = FIXED_ARTIFACT_BASE,
) -> dict[str, Any]:
    campaign_root = Path(campaign_root)
    artifact_root = campaign_root / ARTIFACT_PATH
    run_root = campaign_root / RUN_PATH
    manifest_path = run_root / "manifest.json"
    if artifact_root.exists() or manifest_path.exists():
        raise ValueError("family 4 destination already exists; retries are forbidden")
    gate = AccessLedger(campaign_root).gate_status()
    if gate["state"] != "closed" or gate["protected_access_count"] != 0:
        raise ValueError("family 4 requires the gate closed with zero access")
    profile_path = campaign_root / "profiles/family-04-oof-ensemble.json"
    preregistration_path = run_root / "preregistration.json"
    profile = read_json(profile_path)
    preregistration = read_json(preregistration_path)
    if (
        tuple(profile["recipe_menu"]) != ENSEMBLE_VARIANT_IDS
        or tuple(preregistration["preregistered_recipe_ids"]) != ENSEMBLE_VARIANT_IDS
        or preregistration["scoring_state"] != "not-started"
        or preregistration["profile_sha256"] != file_sha256(profile_path)
    ):
        raise ValueError("family 4 menu and optimizer were not preregistered")
    registry_before = (campaign_root / "registry.jsonl").read_bytes()
    if hashlib.sha256(registry_before).hexdigest().upper() != preregistration["registry_prefix_sha256_before"]:
        raise ValueError("family 4 registry prefix changed after preregistration")

    sources = {
        item["id"]: artifact_base / Path(item["artifact_path"]).name
        for item in profile["constituents"]
    }
    history = {name: [] for name in profile["constituent_ids"]}
    candidate_predictions: list[dict[str, Any]] = []
    incumbent_predictions: list[dict[str, Any]] = []
    selections = []
    fold_entries = []
    source_lineage = []
    attempts = []
    for year in profile["outer_years"]:
        outer = {
            name: _read_jsonl(sources[name] / f"fold-{year}/outer-predictions.jsonl")
            for name in profile["constituent_ids"]
        }
        lineage_entry = {
            "year": year,
            "constituents": {
                name: {
                    "source_path": f"{sources[name].name}/fold-{year}/outer-predictions.jsonl",
                    "sha256": file_sha256(sources[name] / f"fold-{year}/outer-predictions.jsonl"),
                    "row_count": len(outer[name]),
                }
                for name in profile["constituent_ids"]
            },
        }
        selected = select_recipe_for_outer(
            history if any(history.values()) else {},
            outer,
            outer_year=year,
            profile=profile,
        )
        selection = selected["selection"]
        predictions = selected["predictions"]
        selections.append(selection)
        candidate_predictions.extend(predictions)
        incumbent_predictions.extend(outer[profile["current_constituent_id"]])
        fold_root = artifact_root / f"fold-{year}"
        prediction_path = fold_root / "outer-predictions.jsonl"
        selection_path = fold_root / "selection.json"
        _write_jsonl(prediction_path, predictions)
        write_canonical_json(selection_path, selection)
        fold_entries.append(
            {
                "year": year,
                "path": f"fold-{year}/outer-predictions.jsonl",
                "row_count": len(predictions),
                "sha256": file_sha256(prediction_path),
                "selected_recipe_id": selection["selected_recipe_id"],
                "selection_path": f"fold-{year}/selection.json",
                "selection_sha256": canonical_sha256(selection),
            }
        )
        lineage_entry["fit_row_count"] = selection["fit_row_count"]
        lineage_entry["fit_folds"] = selection["fit_folds"]
        lineage_entry["fit_max_date"] = selection["fit_max_date"]
        lineage_entry["weights"] = selection["weights"]
        source_lineage.append(lineage_entry)
        attempts.extend(
            {
                "fold": year,
                "recipe_id": recipe_id,
                "state": "not-scored-earliest" if year == profile["outer_years"][0] else "scored-inner-oof",
                "score": selection["recipe_scores"].get(recipe_id),
            }
            for recipe_id in ENSEMBLE_VARIANT_IDS
        )
        for name in profile["constituent_ids"]:
            history[name].extend(outer[name])

    metrics = reduce_predictions(candidate_predictions).as_dict()
    incumbent_metrics = reduce_predictions(incumbent_predictions).as_dict()
    intervals = event_block_bootstrap_delta(
        candidate_predictions,
        incumbent_predictions,
        iterations=int(profile["bootstrap"]["iterations"]),
        seed=int(profile["bootstrap"]["seed"]),
    )
    metric_deltas = {
        name: float(metrics[name]) - float(incumbent_metrics[name])
        for name in ("log_loss", "brier", "accuracy")
    }
    calibration_gaps, subgroup_gaps = _metric_gaps(metrics, incumbent_metrics)
    decision = promotion_decision(
        {"log_loss_delta": metric_deltas["log_loss"]}, intervals
    )
    adaptive_signal = {
        "selected_recipes": [selection["selected_recipe_id"] for selection in selections],
        "foundation_weights": [
            sum(selection["weights"][name] for name in profile["foundation_constituent_ids"])
            for selection in selections
        ],
        "pooled_log_loss_delta": metric_deltas["log_loss"],
        "pooled_ece_delta": calibration_gaps["ece"],
    }
    lineage = {
        "constituent_ids": profile["constituent_ids"],
        "foundation_constituent_ids": profile["foundation_constituent_ids"],
        "foundation_aggregate_cap": profile["foundation_aggregate_cap"],
        "folds": source_lineage,
        "outer_label_fit_count": 0,
        "full_prediction_node_count": 0,
        "gate_access_count": gate["protected_access_count"],
    }
    write_canonical_json(artifact_root / "constituent-lineage.json", lineage)
    result = {
        "experiment_id": EXPERIMENT_ID,
        "status": "complete",
        "metrics": metrics,
        "incumbent_metrics": incumbent_metrics,
        "metric_deltas": metric_deltas,
        "calibration_gaps": calibration_gaps,
        "subgroup_gaps": subgroup_gaps,
        "paired_event_block_intervals": intervals,
        "promotion_decision": decision,
        "selected_recipes": adaptive_signal["selected_recipes"],
        "adaptive_signal_for_family_05": adaptive_signal,
        "gate_access_count": gate["protected_access_count"],
    }
    write_canonical_json(artifact_root / "result.json", result)
    inventory = tree_inventory(artifact_root)
    _write_jsonl(run_root / "attempts.jsonl", attempts)
    (run_root / "decision.md").write_bytes(
        f"# Family 4 decision\n\n{decision['action']}: {decision['rule']}.\n".encode("utf-8")
    )
    manifest = {
        **result,
        "kind": "family",
        "exit_state": "complete",
        "artifact_path": ARTIFACT_PATH,
        "artifact_tree_sha256": inventory.tree_sha256,
        "artifact_file_count": inventory.file_count,
        "profile_path": "profiles/family-04-oof-ensemble.json",
        "profile_sha256": canonical_sha256(profile),
        "profile_file_sha256": file_sha256(profile_path),
        "preregistration_path": f"{RUN_PATH}/preregistration.json",
        "preregistration_commit": preregistration_commit,
        "attempts_path": f"{RUN_PATH}/attempts.jsonl",
        "constituent_lineage_path": "constituent-lineage.json",
        "fold_predictions": fold_entries,
        "selections": selections,
        "source_revision": source_revision,
        "terminal_failure": None,
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
            "status": "complete",
        },
    )
    return {**result, **registry, "artifact_tree_sha256": inventory.tree_sha256}
