"""Leakage-resistant component combination for hierarchical fight outcomes."""

from __future__ import annotations

from datetime import date, timedelta
import math
from typing import Any, Iterable, Mapping


class OutcomeDecompositionError(ValueError):
    """Raised when component evidence cannot support an honest decomposition."""


def _probability(value: Any, noun: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise OutcomeDecompositionError(f"{noun} probability is not numeric") from exc
    if not math.isfinite(result) or not 0.0 <= result <= 1.0:
        raise OutcomeDecompositionError(f"{noun} probability must be finite and within [0, 1]")
    return result


def combine_law_of_total_probability(
    decision_probability: float,
    decision_win_probability: float,
    finish_win_probability: float,
) -> float:
    """Return P(win) from mutually exclusive decision and finish components."""

    decision = _probability(decision_probability, "decision")
    decision_win = _probability(decision_win_probability, "decision-win")
    finish_win = _probability(finish_win_probability, "finish-win")
    return decision * decision_win + (1.0 - decision) * finish_win


def _component_identity(records: Iterable[Mapping[str, Any]]) -> list[tuple[str, str]]:
    identity: list[tuple[str, str]] = []
    for record in records:
        if record.get("fit_scope") != "prior-only":
            raise OutcomeDecompositionError("every component must be prior-only OOF")
        if int(record.get("outer_label_reads", -1)) != 0:
            raise OutcomeDecompositionError("component prediction read an outer label")
        _probability(record.get("probability"), "component")
        event_date = date.fromisoformat(str(record["event_date"])[:10])
        fit_max_date = date.fromisoformat(str(record["fit_max_date"])[:10])
        embargo_days = int(record.get("embargo_days", 0))
        if fit_max_date > event_date - timedelta(days=embargo_days):
            raise OutcomeDecompositionError("component fit violates the outer embargo")
        identity.append((str(record["fight_id"]), str(record["fold"])))
    if len(identity) != len(set(identity)):
        raise OutcomeDecompositionError("component identity contains duplicates")
    return identity


def validate_component_alignment(
    decision_records: Iterable[Mapping[str, Any]],
    decision_win_records: Iterable[Mapping[str, Any]],
    finish_win_records: Iterable[Mapping[str, Any]],
) -> list[str]:
    """Require byte-order-equivalent fight/fold identities across OOF components."""

    decision = _component_identity(decision_records)
    decision_win = _component_identity(decision_win_records)
    finish_win = _component_identity(finish_win_records)
    if decision != decision_win or decision != finish_win:
        raise OutcomeDecompositionError("component label/ID/fold identity mismatch")
    return [fight_id for fight_id, _ in decision]


def validate_gate_lineage(gate: Mapping[str, Any]) -> None:
    """Reject any gate whose construction or inference read outer labels."""

    if int(gate.get("outer_label_reads", -1)) != 0:
        raise OutcomeDecompositionError("learned gate read an outer label")
    if gate.get("kind") not in {"fixed", "chronological-oof"}:
        raise OutcomeDecompositionError("gate kind is not preregistered as fixed or chronological OOF")


def validate_fallback(
    fallback: Mapping[str, Any] | None,
    *,
    support: int,
    minimum_support: int,
) -> float | None:
    """Resolve a preregistered prior only when specialist support is insufficient."""

    if support >= minimum_support:
        return None
    if fallback is None or fallback.get("registered") is not True:
        raise OutcomeDecompositionError("sparse support requires a registered fallback")
    if fallback.get("id") != "constant-prior":
        raise OutcomeDecompositionError("fallback identity is not registered")
    return _probability(fallback.get("prior"), "fallback")


def shrink_probability(probability: float, prior: float, support: int, strength: float) -> float:
    """Shrink a specialist probability toward its chronological training prior."""

    value = _probability(probability, "specialist")
    base = _probability(prior, "prior")
    if support < 0 or strength < 0:
        raise OutcomeDecompositionError("support and shrinkage strength must be non-negative")
    weight = support / (support + strength) if support + strength else 0.0
    return weight * value + (1.0 - weight) * base


def blend_probability(first: float, second: float, first_weight: float) -> float:
    """Blend two probabilities with a fixed preregistered weight."""

    a = _probability(first, "first")
    b = _probability(second, "second")
    weight = _probability(first_weight, "blend-weight")
    return weight * a + (1.0 - weight) * b
