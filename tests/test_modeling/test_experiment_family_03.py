from __future__ import annotations

from datetime import date

import pytest

from libs.modeling.experiment_campaign.calibration import (
    CALIBRATION_VARIANT_IDS,
    CalibrationError,
    fit_temporal_calibrator,
)
from libs.modeling.experiment_campaign.families.temporal_calibration import (
    SourceLineageError,
    audit_registered_rows,
    promotion_decision,
    select_and_calibrate_outer,
)


PROFILE = {
    "identity": {"clipping_epsilon": None},
    "sigmoid-platt": {
        "clipping_epsilon": 1e-6,
        "inverse_regularization_strength": 0.25,
        "max_iterations": 1000,
        "seed": 20260815,
    },
    "temperature": {
        "clipping_epsilon": 1e-6,
        "minimum_temperature": 0.5,
        "maximum_temperature": 3.0,
        "grid_step": 0.025,
        "regularization_strength": 0.01,
    },
    "conservative-isotonic": {
        "clipping_epsilon": 1e-6,
        "minimum_support": 8,
        "minimum_class_support": 3,
        "minimum_unique_probabilities": 6,
        "shrinkage_weight": 0.5,
        "fallback": "identity",
    },
}


def _valid_history() -> dict[str, object]:
    return {
        "probabilities": [0.1, 0.2, 0.35, 0.45, 0.55, 0.65, 0.8, 0.9],
        "labels": [0, 0, 0, 1, 0, 1, 1, 1],
        "fit_ids": [f"fit-{index}" for index in range(8)],
        "fit_dates": [date(2021, 1, index + 1) for index in range(8)],
        "model_fit_ids": ["model-a", "model-b"],
        "outer_ids": ["outer-a", "outer-b"],
        "outer_min_date": date(2022, 1, 1),
    }


def _fit(variant_id: str = "identity", **updates: object):
    history = _valid_history()
    history.update(updates)
    return fit_temporal_calibrator(
        variant_id=variant_id,
        config=PROFILE[variant_id],
        **history,
    )


def test_exact_variant_menu_and_identity_is_bit_exact_noop() -> None:
    assert CALIBRATION_VARIANT_IDS == (
        "identity",
        "sigmoid-platt",
        "temperature",
        "conservative-isotonic",
    )
    fitted = _fit("identity")
    probabilities = [0.0, 0.125, 0.5, 0.875, 1.0]
    assert fitted.transform(probabilities) == probabilities
    assert fitted.fit_summary["fallback"] is None


@pytest.mark.parametrize(
    ("updates", "match"),
    [
        ({"probabilities": [], "labels": [], "fit_ids": [], "fit_dates": []}, "empty"),
        ({"labels": [0] * 8}, "both classes"),
        ({"fit_ids": ["duplicate"] * 8}, "duplicate"),
        ({"fit_ids": ["model-a", *[f"fit-{i}" for i in range(1, 8)]]}, "model-fit"),
        ({"fit_ids": ["outer-a", *[f"fit-{i}" for i in range(1, 8)]]}, "outer"),
        ({"fit_dates": [date(2021, 1, 2), date(2021, 1, 1), *[date(2021, 1, i) for i in range(3, 9)]]}, "chronological"),
        ({"fit_dates": [*[date(2021, 1, i) for i in range(1, 8)], date(2022, 1, 1)]}, "future"),
        ({"probabilities": [-0.01, 0.2, 0.35, 0.45, 0.55, 0.65, 0.8, 0.9]}, "range"),
        ({"probabilities": [0.1, 0.2, 0.35, 0.45, 0.55, 0.65, 0.8, 1.01]}, "range"),
    ],
)
def test_all_calibrators_reject_forbidden_temporal_histories(
    updates: dict[str, object], match: str
) -> None:
    for variant_id in CALIBRATION_VARIANT_IDS:
        with pytest.raises(CalibrationError, match=match):
            _fit(variant_id, **updates)


def test_non_identity_variants_are_deterministic_and_bounded() -> None:
    evaluation = [0.0, 0.15, 0.5, 0.85, 1.0]
    for variant_id in CALIBRATION_VARIANT_IDS[1:]:
        first = _fit(variant_id).transform(evaluation)
        second = _fit(variant_id).transform(evaluation)
        assert first == second
        assert len(first) == len(evaluation)
        assert all(0.0 <= value <= 1.0 for value in first)


def test_conservative_isotonic_obeys_declared_support_fallback() -> None:
    config = {**PROFILE["conservative-isotonic"], "minimum_support": 9}
    fitted = fit_temporal_calibrator(
        variant_id="conservative-isotonic",
        config=config,
        **_valid_history(),
    )
    probabilities = [0.0, 0.25, 0.5, 0.75, 1.0]
    assert fitted.transform(probabilities) == probabilities
    assert fitted.fit_summary == {
        "variant_id": "conservative-isotonic",
        "fit_row_count": 8,
        "fit_id_count": 8,
        "fallback": "identity",
        "fallback_reason": "minimum_support",
    }


def test_registered_source_audit_rejects_same_fit_probabilities_before_score() -> None:
    inner_rows = [
        {
            "boundary": "InnerSelection",
            "event_date": "2021-01-01",
            "event_id": "inner-a",
            "fight_id": "fight-a",
            "fit_event_ids": ["model-a", "inner-a"],
            "probability": 0.25,
            "y_true": 0,
        },
        {
            "boundary": "InnerSelection",
            "event_date": "2021-02-01",
            "event_id": "inner-b",
            "fight_id": "fight-b",
            "fit_event_ids": ["model-a", "inner-b"],
            "probability": 0.75,
            "y_true": 1,
        },
    ]
    outer_rows = [
        {
            "boundary": "Original",
            "event_date": "2022-01-01",
            "event_id": "outer-a",
            "fight_id": "fight-c",
            "probability": 0.6,
            "y_true": 1,
        }
    ]

    with pytest.raises(SourceLineageError, match="model-fit") as error:
        audit_registered_rows(inner_rows, outer_rows, outer_year=2022)
    assert error.value.audit["variant_fit_count"] == 0
    assert error.value.audit["variant_score_count"] == 0
    assert error.value.audit["calibration_model_fit_overlap_count"] == 2


def test_registered_source_audit_accepts_prior_disjoint_two_class_oof_rows() -> None:
    inner_rows = [
        {
            "boundary": "InnerSelection",
            "event_date": "2021-01-01",
            "event_id": "inner-a",
            "fight_id": "fight-a",
            "fit_event_ids": ["model-a"],
            "probability": 0.25,
            "y_true": 0,
        },
        {
            "boundary": "InnerSelection",
            "event_date": "2021-02-01",
            "event_id": "inner-b",
            "fight_id": "fight-b",
            "fit_event_ids": ["model-a"],
            "probability": 0.75,
            "y_true": 1,
        },
    ]
    outer_rows = [
        {
            "boundary": "Original",
            "event_date": "2022-01-01",
            "event_id": "outer-a",
            "fight_id": "fight-c",
            "probability": 0.6,
            "y_true": 1,
        }
    ]

    audit = audit_registered_rows(inner_rows, outer_rows, outer_year=2022)
    assert audit["status"] == "eligible"
    assert audit["calibration_fit_event_count"] == 2
    assert audit["variant_fit_count"] == 0
    assert audit["variant_score_count"] == 0


def test_registered_source_audit_accepts_original_oof_with_cross_row_training() -> None:
    inner_rows = [
        {
            "boundary": "Original",
            "event_date": "2021-01-01",
            "event_id": "prior-a",
            "fight_id": "fight-a",
            "fit_event_ids": ["model-a"],
            "probability": 0.25,
            "y_true": 0,
        },
        {
            "boundary": "Original",
            "event_date": "2021-02-01",
            "event_id": "prior-b",
            "fight_id": "fight-b",
            "fit_event_ids": ["model-a", "prior-a"],
            "probability": 0.75,
            "y_true": 1,
        },
    ]
    outer_rows = [
        {
            "boundary": "Original",
            "event_date": "2022-01-01",
            "event_id": "outer-a",
            "fight_id": "fight-c",
            "probability": 0.6,
            "y_true": 1,
        }
    ]

    audit = audit_registered_rows(inner_rows, outer_rows, outer_year=2022)
    assert audit["status"] == "eligible"
    assert audit["same_fit_row_count"] == 0


def test_earliest_outer_fold_is_deliberately_identity_only_without_fit() -> None:
    outer = [
        {
            "boundary": "Original",
            "event_date": "2022-01-01",
            "event_id": "outer-a",
            "fight_id": "fight-a",
            "probability": 0.23456789,
            "y_true": 0,
        }
    ]
    result = select_and_calibrate_outer([], outer, outer_year=2022, variant_configs=PROFILE)
    assert result["selection"]["variant_id"] == "identity"
    assert result["selection"]["selection_basis"] == "identity-only-no-fit"
    assert result["selection"]["fit_row_count"] == 0
    assert result["predictions"][0]["probability"] == 0.23456789


def test_outer_labels_cannot_change_chronological_variant_selection_or_probabilities() -> None:
    history = []
    for index in range(20):
        history.append(
            {
                "boundary": "Original",
                "event_date": f"2022-01-{index + 1:02d}",
                "event_id": f"history-event-{index}",
                "fight_id": f"history-fight-{index}",
                "fit_event_ids": ["older-model-event"],
                "probability": 0.1 + (0.8 * index / 19),
                "y_true": int(index >= 10),
            }
        )
    outer = [
        {
            "boundary": "Original",
            "event_date": "2023-01-01",
            "event_id": "outer-event",
            "fight_id": "outer-fight",
            "probability": 0.61,
            "y_true": 0,
        }
    ]
    changed_labels = [{**outer[0], "y_true": 1}]

    first = select_and_calibrate_outer(history, outer, outer_year=2023, variant_configs=PROFILE)
    second = select_and_calibrate_outer(
        history, changed_labels, outer_year=2023, variant_configs=PROFILE
    )
    assert first["selection"]["variant_scores"].keys() == set(CALIBRATION_VARIANT_IDS)
    assert first["selection"]["variant_id"] == second["selection"]["variant_id"]
    assert first["predictions"][0]["probability"] == second["predictions"][0]["probability"]
    assert first["selection"]["selection_max_date"] < outer[0]["event_date"]


def test_promotion_consumes_the_recorded_metric_delta_shape() -> None:
    decision = promotion_decision(
        {"log_loss": -0.01},
        {"log_loss_delta": {"upper": -0.001}},
    )
    assert decision["promoted"] is True
