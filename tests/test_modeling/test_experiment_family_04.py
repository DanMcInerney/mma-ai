from __future__ import annotations

from copy import deepcopy

import pytest

from libs.modeling.experiment_campaign.ensemble import (
    ENSEMBLE_VARIANT_IDS,
    ConstituentError,
    SolverStateError,
    WeightError,
    build_ensemble_predictions,
    fit_regularized_nonnegative_oof,
    validate_constituents,
    validate_weights,
)


CONSTITUENT_IDS = (
    "weighted-v8-control",
    "horizon-recency",
    "temporal-calibrated",
)


def _rows(*, fold: str = "2024", boundary: str = "Original") -> dict[str, list[dict]]:
    probabilities = {
        "weighted-v8-control": (0.2, 0.8),
        "horizon-recency": (0.3, 0.7),
        "temporal-calibrated": (0.4, 0.6),
    }
    return {
        constituent_id: [
            {
                "boundary": boundary,
                "context_max_date": "2023-12-16",
                "event_date": "2024-01-13",
                "event_id": "event-1",
                "fight_id": f"fight-{index}",
                "fold": fold,
                "probability": probability,
                "y_true": index,
            }
            for index, probability in enumerate(values)
        ]
        for constituent_id, values in probabilities.items()
    }


def _solver() -> dict[str, object]:
    return {
        "algorithm": "projected-gradient-log-loss",
        "learning_rate": 0.05,
        "max_iterations": 4000,
        "tolerance": 1e-12,
        "seed": None,
    }


def test_exact_five_recipe_menu_is_frozen() -> None:
    assert ENSEMBLE_VARIANT_IDS == (
        "best-single",
        "current-autogluon-tune-ensemble",
        "median-probability-blend",
        "rank-probability-blend",
        "regularized-nonnegative-oof-blend",
    )


@pytest.mark.parametrize(
    ("field", "value", "match"),
    [
        ("fight_id", "other-fight", "fight/event/fold"),
        ("event_id", "other-event", "fight/event/fold"),
        ("fold", "2025", "fight/event/fold"),
        ("y_true", 1, "label"),
        ("boundary", "InnerSelection", "Original"),
        ("boundary", "FULL", "FULL"),
        ("context_max_date", "2024-01-13", "context"),
        ("event_date", "2025-01-01", "future"),
    ],
)
def test_constituent_alignment_fails_closed(field: str, value: object, match: str) -> None:
    rows = _rows()
    rows["horizon-recency"][0][field] = value
    with pytest.raises(ConstituentError, match=match):
        validate_constituents(
            rows,
            expected_constituent_ids=CONSTITUENT_IDS,
            outer_fold="2024",
            outer_max_date="2024-12-14",
        )


def test_missing_or_unregistered_constituents_fail_closed() -> None:
    rows = _rows()
    rows.pop("temporal-calibrated")
    with pytest.raises(ConstituentError, match="exact registered constituent"):
        validate_constituents(
            rows,
            expected_constituent_ids=CONSTITUENT_IDS,
            outer_fold="2024",
            outer_max_date="2024-12-14",
        )


@pytest.mark.parametrize(
    ("weights", "foundation_cap", "match"),
    [
        ({"weighted-v8-control": -0.1, "horizon-recency": 0.6, "temporal-calibrated": 0.5}, 1.0, "negative"),
        ({"weighted-v8-control": 0.2, "horizon-recency": 0.2, "temporal-calibrated": 0.2}, 1.0, "unit sum"),
        ({"weighted-v8-control": 0.6, "horizon-recency": 0.2, "temporal-calibrated": 0.2}, 0.5, "foundation"),
    ],
)
def test_weight_admission_rejects_invalid_and_excess_foundation_weight(
    weights: dict[str, float], foundation_cap: float, match: str
) -> None:
    with pytest.raises(WeightError, match=match):
        validate_weights(
            weights,
            expected_constituent_ids=CONSTITUENT_IDS,
            foundation_constituent_ids=("weighted-v8-control",),
            foundation_aggregate_cap=foundation_cap,
        )


def test_direct_recipes_reproduce_registered_rows_exactly() -> None:
    rows = _rows()
    best_single = build_ensemble_predictions(
        rows,
        recipe_id="best-single",
        selected_constituent_id="horizon-recency",
    )
    current = build_ensemble_predictions(
        rows,
        recipe_id="current-autogluon-tune-ensemble",
        selected_constituent_id="weighted-v8-control",
    )
    assert best_single == rows["horizon-recency"]
    assert current == rows["weighted-v8-control"]
    assert best_single is not rows["horizon-recency"]
    assert current is not rows["weighted-v8-control"]


def test_regularized_fit_is_deterministic_inner_only_and_respects_cap() -> None:
    rows = _rows()
    first = fit_regularized_nonnegative_oof(
        rows,
        fit_role="inner-chronological-oof",
        shrinkage=0.15,
        foundation_constituent_ids=("weighted-v8-control",),
        foundation_aggregate_cap=0.6,
        solver=_solver(),
    )
    second = fit_regularized_nonnegative_oof(
        deepcopy(rows),
        fit_role="inner-chronological-oof",
        shrinkage=0.15,
        foundation_constituent_ids=("weighted-v8-control",),
        foundation_aggregate_cap=0.6,
        solver=_solver(),
    )
    assert first == second
    assert all(weight >= 0.0 for weight in first.values())
    assert sum(first.values()) == pytest.approx(1.0, abs=1e-12)
    assert first["weighted-v8-control"] <= 0.6 + 1e-12


def test_outer_label_role_and_nondeterministic_solver_state_are_rejected() -> None:
    rows = _rows()
    with pytest.raises(ConstituentError, match="outer-label"):
        fit_regularized_nonnegative_oof(
            rows,
            fit_role="outer-labels",
            shrinkage=0.15,
            foundation_constituent_ids=("weighted-v8-control",),
            foundation_aggregate_cap=0.6,
            solver=_solver(),
        )
    with pytest.raises(SolverStateError, match="deterministic"):
        fit_regularized_nonnegative_oof(
            rows,
            fit_role="inner-chronological-oof",
            shrinkage=0.15,
            foundation_constituent_ids=("weighted-v8-control",),
            foundation_aggregate_cap=0.6,
            solver={**_solver(), "seed": 123},
        )
