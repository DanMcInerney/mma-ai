"""Family 4 chronological-OOF recipe selection and materialization."""

from __future__ import annotations

from copy import deepcopy
from datetime import date
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

from ..ensemble import (
    ENSEMBLE_VARIANT_IDS,
    ConstituentError,
    build_ensemble_predictions,
    fit_regularized_nonnegative_oof,
    validate_constituents,
)


EXPERIMENT_ID = "family-04-chronological-oof-ensemble"


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
