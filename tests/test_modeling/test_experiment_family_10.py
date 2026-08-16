"""Family 10 outcome-decomposition contracts."""

from __future__ import annotations

from copy import deepcopy

import pytest

from libs.modeling.experiment_campaign.outcome_decomposition import (
    OutcomeDecompositionError,
    build_combined_records,
    combine_law_of_total_probability,
    validate_component_alignment,
    validate_fallback,
    validate_gate_lineage,
)
from libs.modeling.experiment_campaign.families.outcome_decomposition import (
    build_preregistered_profile,
    validate_preregistered_profile,
    write_preregistration,
)
from libs.modeling.experiment_campaign.hashing import file_sha256
from libs.modeling.experiment_campaign.protocol import initialize_gate


def _component(fight_id: str, probability: float, *, fold: str = "2025") -> dict:
    return {
        "fight_id": fight_id,
        "fold": fold,
        "probability": probability,
        "fit_scope": "prior-only",
        "fit_max_date": "2024-12-14",
        "event_date": "2025-01-11",
        "embargo_days": 7,
        "outer_label_reads": 0,
    }


def test_law_of_total_probability_is_exact() -> None:
    assert combine_law_of_total_probability(0.25, 0.8, 0.2) == pytest.approx(0.35)


def test_combined_records_preserve_outer_identity_and_exact_formula() -> None:
    template = [{**_component("1", 0.5), "y_true": 1, "event_id": "9", "outcome_type": "finish"}]
    rows = build_combined_records(
        template,
        [_component("1", 0.25)],
        [_component("1", 0.8)],
        [_component("1", 0.2)],
        variant_id="three-component",
        clipping=(0.0, 1.0),
    )
    assert rows[0]["probability"] == pytest.approx(0.35)
    assert rows[0]["fight_id"] == "1" and rows[0]["y_true"] == 1
    assert rows[0]["component_probabilities"] == {
        "decision": 0.25,
        "decision-win": 0.8,
        "finish-win": 0.2,
    }


@pytest.mark.parametrize("values", [(-0.1, 0.5, 0.5), (0.5, 1.1, 0.5), (0.5, 0.5, float("nan"))])
def test_formula_rejects_invalid_probabilities(values: tuple[float, float, float]) -> None:
    with pytest.raises(OutcomeDecompositionError, match="probability"):
        combine_law_of_total_probability(*values)


def test_component_alignment_rejects_non_oof_and_identity_mismatch() -> None:
    decision = [_component("1", 0.6)]
    decision_win = [_component("1", 0.7)]
    finish_win = [_component("1", 0.4)]
    assert validate_component_alignment(decision, decision_win, finish_win) == ["1"]

    contaminated = deepcopy(decision)
    contaminated[0]["fit_scope"] = "same-row"
    with pytest.raises(OutcomeDecompositionError, match="prior-only"):
        validate_component_alignment(contaminated, decision_win, finish_win)

    wrong_fold = deepcopy(finish_win)
    wrong_fold[0]["fold"] = "2024"
    with pytest.raises(OutcomeDecompositionError, match="identity"):
        validate_component_alignment(decision, decision_win, wrong_fold)

    wrong_id = [_component("2", 0.4)]
    with pytest.raises(OutcomeDecompositionError, match="identity"):
        validate_component_alignment(decision, decision_win, wrong_id)


def test_component_alignment_rejects_chronology_and_outer_label_reads() -> None:
    bad_date = [_component("1", 0.6)]
    bad_date[0]["fit_max_date"] = "2025-01-10"
    with pytest.raises(OutcomeDecompositionError, match="embargo"):
        validate_component_alignment(bad_date, [_component("1", 0.7)], [_component("1", 0.4)])

    bad_gate = [_component("1", 0.6)]
    bad_gate[0]["outer_label_reads"] = 1
    with pytest.raises(OutcomeDecompositionError, match="outer label"):
        validate_component_alignment(bad_gate, [_component("1", 0.7)], [_component("1", 0.4)])


def test_learned_gate_and_sparse_fallback_are_explicit() -> None:
    validate_gate_lineage({"kind": "fixed", "outer_label_reads": 0})
    with pytest.raises(OutcomeDecompositionError, match="outer label"):
        validate_gate_lineage({"kind": "learned", "outer_label_reads": 2})

    registered = {"id": "constant-prior", "registered": True, "prior": 0.5}
    assert validate_fallback(registered, support=0, minimum_support=40) == 0.5
    with pytest.raises(OutcomeDecompositionError, match="registered"):
        validate_fallback({**registered, "registered": False}, support=0, minimum_support=40)
    with pytest.raises(OutcomeDecompositionError, match="support"):
        validate_fallback(None, support=10, minimum_support=40)


def test_preregistered_menu_is_exact_and_bounded() -> None:
    profile = build_preregistered_profile()
    result = validate_preregistered_profile(profile)
    assert result["variant_count"] == 6
    assert result["variant_ids"] == [
        "direct-incumbent-control",
        "three-component",
        "shrinkage-gated-three-component",
        "decision-finish-specialist-mixture",
        "support-trimmed-specialist-mixture",
        "constant-prior-fallback",
    ]

    too_many = deepcopy(profile)
    too_many["variants"].append(deepcopy(too_many["variants"][-1]))
    with pytest.raises(OutcomeDecompositionError, match="six"):
        validate_preregistered_profile(too_many)

    missing_fallback = deepcopy(profile)
    missing_fallback["fallbacks"] = []
    with pytest.raises(OutcomeDecompositionError, match="fallback"):
        validate_preregistered_profile(missing_fallback)


def test_preregistration_is_durable_before_any_launch(tmp_path) -> None:
    campaign = tmp_path / "campaign"
    campaign.mkdir()
    initialize_gate(campaign, expected_family_ids=())
    (campaign / "registry.jsonl").write_bytes(b"fixed-nine-family-prefix\n")

    preregistration = write_preregistration(campaign, source_revision="before-fit")
    profile = campaign / "profiles/family-10-outcome-decomposition.json"
    assert preregistration["launch_state"] == "not-started"
    assert preregistration["variant_count"] == 6
    assert preregistration["profile_file_sha256"] == file_sha256(profile)
    assert preregistration["gate_required_state"] == "closed-zero-access"
    with pytest.raises(ValueError, match="destinations must all be absent"):
        write_preregistration(campaign, source_revision="retry")
