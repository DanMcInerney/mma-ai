"""Family 10 outcome-decomposition contracts."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pytest

from libs.modeling.experiment_campaign.__main__ import _parser
from libs.modeling.experiment_campaign import runner
from libs.modeling.experiment_campaign.outcome_decomposition import (
    OutcomeDecompositionError,
    build_combined_records,
    combine_law_of_total_probability,
    validate_component_alignment,
    validate_fallback,
    validate_gate_lineage,
)
from libs.modeling.experiment_campaign.families.outcome_decomposition import (
    VARIANT_IDS,
    _variant_records,
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


def test_combined_variant_mapping_follows_preregistered_order() -> None:
    control = [
        {
            **_component(str(index), 0.5),
            "y_true": index % 2,
            "event_id": str(index // 2),
            "outcome_type": "decision",
        }
        for index in range(1_108)
    ]
    components = {
        component_id: [
            _component(str(index), probability)
            for index in range(1_108)
        ]
        for component_id, probability in (
            ("decision", 0.5),
            ("decision-win", 0.6),
            ("finish-win", 0.4),
        )
    }
    evidence = {
        "2025": {
            component_id: {"prior": 0.5, "support": 1_000}
            for component_id in components
        }
    }

    variants = _variant_records(
        build_preregistered_profile(),
        control=control,
        components=components,
        fold_evidence=evidence,
    )

    assert tuple(variants) == VARIANT_IDS


def test_completion_cli_flags_are_plumbed() -> None:
    replay = _parser().parse_args([
        "replay-decisions",
        "--campaign",
        "campaign",
        "--through",
        "family-10-outcome-decomposition",
        "--require-development-only",
    ])
    assert replay.require_development_only is True

    validate = _parser().parse_args([
        "validate",
        "--campaign",
        "campaign",
        "--strict",
        "--expect-terminal-through",
        "10",
        "--require-unsealed",
        "--require-gate-closed",
    ])
    assert validate.require_unsealed is True


def test_actual_successor_result_recomputes_without_retry_or_gate_access() -> None:
    verified = runner.verify_family_run(
        Path("experiments/top10_20260815"),
        "family-10-outcome-decomposition",
        recompute_all=True,
    )
    assert verified["status"] == "complete"
    assert verified["component_fit_count"] == 12
    assert verified["component_prediction_count"] == 3_324
    assert verified["combined_prediction_count"] == 6_648
    assert verified["retry_count"] == 0
    assert verified["gate_access_count"] == 0


def test_family_10_verifier_rejects_artifact_tree_mismatch(monkeypatch) -> None:
    monkeypatch.setattr(
        runner,
        "tree_inventory",
        lambda _: SimpleNamespace(tree_sha256="BAD", file_count=49, total_bytes=0),
    )
    with pytest.raises(ValueError, match="artifact tree"):
        runner.verify_family_run(
            Path("experiments/top10_20260815"),
            "family-10-outcome-decomposition",
            recompute_all=True,
        )
