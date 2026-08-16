from __future__ import annotations

from datetime import date

import pytest

from libs.modeling.experiment_campaign.calibration import (
    CALIBRATION_VARIANT_IDS,
    CalibrationError,
    fit_temporal_calibrator,
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
