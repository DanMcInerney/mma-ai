"""Deterministic chronological-OOF probability ensemble primitives."""

from __future__ import annotations

from copy import deepcopy
from datetime import date
import math
from statistics import median
from typing import Any, Mapping, Sequence


ENSEMBLE_VARIANT_IDS = (
    "best-single",
    "current-autogluon-tune-ensemble",
    "median-probability-blend",
    "rank-probability-blend",
    "regularized-nonnegative-oof-blend",
)


class ConstituentError(ValueError):
    """A registered prediction constituent is inadmissible."""


class WeightError(ValueError):
    """Blend weights violate the preregistered feasible region."""


class SolverStateError(ValueError):
    """Optimizer state is not the deterministic preregistered state."""


def _identity(row: Mapping[str, Any]) -> tuple[str, str, str]:
    return (str(row.get("fight_id")), str(row.get("event_id")), str(row.get("fold")))


def _aligned_rows(
    constituents: Mapping[str, Sequence[Mapping[str, Any]]],
    *,
    expected_constituent_ids: Sequence[str] | None = None,
) -> tuple[tuple[str, ...], list[list[Mapping[str, Any]]]]:
    constituent_ids = tuple(expected_constituent_ids or constituents)
    if set(constituents) != set(constituent_ids) or not constituent_ids:
        raise ConstituentError("exact registered constituent set is required")
    ordered = [list(constituents[constituent_id]) for constituent_id in constituent_ids]
    if not ordered[0] or any(len(rows) != len(ordered[0]) for rows in ordered[1:]):
        raise ConstituentError("constituents must have the same non-empty row count")
    for row_index, reference in enumerate(ordered[0]):
        expected_identity = _identity(reference)
        expected_label = reference.get("y_true")
        for rows in ordered[1:]:
            candidate = rows[row_index]
            if _identity(candidate) != expected_identity:
                raise ConstituentError("constituent fight/event/fold IDs do not align")
            if candidate.get("y_true") != expected_label:
                raise ConstituentError("constituent label mismatch")
    return constituent_ids, ordered


def validate_constituents(
    constituents: Mapping[str, Sequence[Mapping[str, Any]]],
    *,
    expected_constituent_ids: Sequence[str],
    outer_fold: str,
    outer_max_date: str,
) -> None:
    constituent_ids, ordered = _aligned_rows(
        constituents, expected_constituent_ids=expected_constituent_ids
    )
    del constituent_ids
    maximum = date.fromisoformat(outer_max_date)
    for rows in ordered:
        for row in rows:
            boundary = row.get("boundary")
            if boundary == "FULL":
                raise ConstituentError("FULL prediction nodes are forbidden")
            if boundary != "Original":
                raise ConstituentError("constituent boundary must be Original")
            event_date = date.fromisoformat(str(row["event_date"]))
            if event_date > maximum:
                raise ConstituentError("future constituent rows are forbidden")
            if str(row.get("fold")) != outer_fold:
                raise ConstituentError("constituent fight/event/fold IDs do not align")
            context = row.get("context_max_date")
            if context is None or date.fromisoformat(str(context)) >= event_date:
                raise ConstituentError("constituent context is contaminated")
            probability = float(row["probability"])
            if not math.isfinite(probability) or not 0.0 <= probability <= 1.0:
                raise ConstituentError("constituent probability is outside [0, 1]")


def validate_weights(
    weights: Mapping[str, float],
    *,
    expected_constituent_ids: Sequence[str],
    foundation_constituent_ids: Sequence[str],
    foundation_aggregate_cap: float,
) -> None:
    if set(weights) != set(expected_constituent_ids):
        raise WeightError("weights must cover the exact registered constituent set")
    values = [float(weights[constituent_id]) for constituent_id in expected_constituent_ids]
    if any(not math.isfinite(value) or value < 0.0 for value in values):
        raise WeightError("negative or non-finite weights are forbidden")
    if not math.isclose(sum(values), 1.0, rel_tol=0.0, abs_tol=1e-10):
        raise WeightError("weights must have unit sum")
    if not 0.0 <= foundation_aggregate_cap <= 1.0:
        raise WeightError("foundation aggregate cap must be in [0, 1]")
    foundation_weight = sum(float(weights[name]) for name in foundation_constituent_ids)
    if foundation_weight > foundation_aggregate_cap + 1e-10:
        raise WeightError("foundation aggregate weight exceeds its registered cap")


def _average_ranks(values: Sequence[float]) -> list[float]:
    order = sorted(range(len(values)), key=lambda index: (values[index], index))
    ranks = [0.0] * len(values)
    start = 0
    while start < len(order):
        end = start + 1
        while end < len(order) and values[order[end]] == values[order[start]]:
            end += 1
        average = (start + end - 1) / 2.0
        normalized = 0.5 if len(values) == 1 else average / (len(values) - 1)
        for position in range(start, end):
            ranks[order[position]] = normalized
        start = end
    return ranks


def build_ensemble_predictions(
    constituents: Mapping[str, Sequence[Mapping[str, Any]]],
    *,
    recipe_id: str,
    selected_constituent_id: str | None = None,
    weights: Mapping[str, float] | None = None,
) -> list[dict[str, Any]]:
    if recipe_id not in ENSEMBLE_VARIANT_IDS:
        raise ConstituentError(f"unregistered ensemble recipe: {recipe_id}")
    constituent_ids, ordered = _aligned_rows(constituents)
    if recipe_id in ENSEMBLE_VARIANT_IDS[:2]:
        if selected_constituent_id not in constituents:
            raise ConstituentError("direct recipe requires a registered selected constituent")
        return deepcopy(list(constituents[selected_constituent_id]))

    probabilities = [
        [float(row["probability"]) for row in rows]
        for rows in ordered
    ]
    if recipe_id == "median-probability-blend":
        blended = [median(values) for values in zip(*probabilities, strict=True)]
    elif recipe_id == "rank-probability-blend":
        ranked = [_average_ranks(values) for values in probabilities]
        blended = [sum(values) / len(ranked) for values in zip(*ranked, strict=True)]
    else:
        if weights is None:
            raise WeightError("regularized blend requires fitted weights")
        validate_weights(
            weights,
            expected_constituent_ids=constituent_ids,
            foundation_constituent_ids=(),
            foundation_aggregate_cap=1.0,
        )
        blended = [
            sum(float(weights[name]) * probabilities[index][row_index]
                for index, name in enumerate(constituent_ids))
            for row_index in range(len(ordered[0]))
        ]
    result = deepcopy(ordered[0])
    for row, probability in zip(result, blended, strict=True):
        row["probability"] = probability
    return result


def _project_simplex(values: Sequence[float]) -> list[float]:
    ordered = sorted((float(value) for value in values), reverse=True)
    cumulative = 0.0
    threshold = 0.0
    for index, value in enumerate(ordered, start=1):
        cumulative += value
        candidate = (cumulative - 1.0) / index
        if value - candidate > 0.0:
            threshold = candidate
    projected = [max(float(value) - threshold, 0.0) for value in values]
    total = sum(projected)
    return [value / total for value in projected]


def _apply_foundation_cap(
    values: Sequence[float],
    *,
    constituent_ids: Sequence[str],
    foundation_constituent_ids: Sequence[str],
    cap: float,
) -> list[float]:
    foundation = [index for index, name in enumerate(constituent_ids) if name in foundation_constituent_ids]
    other = [index for index in range(len(values)) if index not in foundation]
    foundation_total = sum(values[index] for index in foundation)
    if foundation_total <= cap or not foundation:
        return list(values)
    if not other and cap < 1.0:
        raise WeightError("foundation cap leaves no admissible non-foundation constituent")
    result = list(values)
    for index in foundation:
        result[index] = values[index] * cap / foundation_total
    other_total = sum(values[index] for index in other)
    if other_total == 0.0:
        for index in other:
            result[index] = (1.0 - cap) / len(other)
    else:
        for index in other:
            result[index] = values[index] * (1.0 - cap) / other_total
    return result


def fit_regularized_nonnegative_oof(
    constituents: Mapping[str, Sequence[Mapping[str, Any]]],
    *,
    fit_role: str,
    shrinkage: float,
    foundation_constituent_ids: Sequence[str],
    foundation_aggregate_cap: float,
    solver: Mapping[str, Any],
) -> dict[str, float]:
    if fit_role != "inner-chronological-oof":
        raise ConstituentError("outer-label fitting is forbidden")
    expected_solver = {
        "algorithm",
        "learning_rate",
        "max_iterations",
        "tolerance",
        "seed",
    }
    if (
        set(solver) != expected_solver
        or solver.get("algorithm") != "projected-gradient-log-loss"
        or solver.get("seed") is not None
        or float(solver.get("learning_rate", 0.0)) <= 0.0
        or int(solver.get("max_iterations", 0)) <= 0
        or float(solver.get("tolerance", 0.0)) <= 0.0
    ):
        raise SolverStateError("solver must use the exact deterministic registered state")
    if not 0.0 <= shrinkage <= 1.0:
        raise WeightError("shrinkage must be in [0, 1]")
    constituent_ids, ordered = _aligned_rows(constituents)
    labels = [int(row["y_true"]) for row in ordered[0]]
    if any(label not in (0, 1) for label in labels):
        raise ConstituentError("OOF labels must be binary")
    probabilities = [[float(row["probability"]) for row in rows] for rows in ordered]
    count = len(constituent_ids)
    weights = _apply_foundation_cap(
        [1.0 / count] * count,
        constituent_ids=constituent_ids,
        foundation_constituent_ids=foundation_constituent_ids,
        cap=foundation_aggregate_cap,
    )
    learning_rate = float(solver["learning_rate"])
    tolerance = float(solver["tolerance"])
    uniform = 1.0 / count
    for _ in range(int(solver["max_iterations"])):
        gradients = [2.0 * shrinkage * (weight - uniform) for weight in weights]
        for row_index, label in enumerate(labels):
            prediction = sum(
                weights[index] * probabilities[index][row_index] for index in range(count)
            )
            clipped = min(max(prediction, 1e-12), 1.0 - 1e-12)
            derivative = (clipped - label) / (clipped * (1.0 - clipped) * len(labels))
            for index in range(count):
                gradients[index] += derivative * probabilities[index][row_index]
        candidate = _project_simplex(
            [weights[index] - learning_rate * gradients[index] for index in range(count)]
        )
        candidate = _apply_foundation_cap(
            candidate,
            constituent_ids=constituent_ids,
            foundation_constituent_ids=foundation_constituent_ids,
            cap=foundation_aggregate_cap,
        )
        if max(abs(candidate[index] - weights[index]) for index in range(count)) <= tolerance:
            weights = candidate
            break
        weights = candidate
    result = {name: weights[index] for index, name in enumerate(constituent_ids)}
    # Pin the tiny floating remainder to the final non-foundation component.
    final_name = next(
        (name for name in reversed(constituent_ids) if name not in foundation_constituent_ids),
        constituent_ids[-1],
    )
    result[final_name] += 1.0 - sum(result.values())
    validate_weights(
        result,
        expected_constituent_ids=constituent_ids,
        foundation_constituent_ids=foundation_constituent_ids,
        foundation_aggregate_cap=foundation_aggregate_cap,
    )
    return result
