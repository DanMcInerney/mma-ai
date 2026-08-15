import math

import numpy as np
import pytest

from libs.modeling.experiment_campaign.metrics import (
    MetricBoundaryError,
    event_block_bootstrap_delta,
    metric_gap,
    reduce_predictions,
)


def _records(probabilities):
    labels = [1, 0, 1, 1]
    return [
        {
            "fight_id": f"f{i}",
            "event_id": f"e{i // 2}",
            "event_date": f"202{2 + i // 2}-01-01",
            "y_true": labels[i],
            "probability": probability,
            "boundary": "Original",
            "fit_scope": "prior-only",
            "fold": f"fold-{i // 2}",
            "weight_class": "light" if i < 2 else "heavy",
            "experience": "veteran",
            "outcome_type": "decision" if i % 2 else "finish",
        }
        for i, probability in enumerate(probabilities)
    ]


def test_metric_reduction_uses_positive_log_loss_and_threshold_point_five():
    probabilities = np.array([0.8, 0.3, 0.6, 0.4])
    result = reduce_predictions(_records(probabilities), reliability_bins=2)
    expected_loss = -np.mean(np.log([0.8, 0.7, 0.6, 0.4]))
    expected_brier = np.mean((probabilities - np.array([1, 0, 1, 1])) ** 2)
    assert result.correct_count == 3
    assert result.row_count == 4
    assert result.accuracy == 0.75
    assert result.log_loss == pytest.approx(expected_loss)
    assert result.log_loss > 0
    assert result.brier == pytest.approx(expected_brier)
    assert math.isfinite(result.calibration_intercept)
    assert math.isfinite(result.calibration_slope)
    assert result.ece == pytest.approx(0.225)
    assert set(result.fold_metrics) == {"fold-0", "fold-1"}
    assert set(result.subgroup_metrics) >= {
        "weight_class=light",
        "weight_class=heavy",
        "experience=veteran",
        "outcome_type=decision",
        "confidence=high",
        "confidence=low",
    }


def test_wrong_probabilities_and_boundary_mislabeling_fail():
    bad = _records([0.8, 0.3, 1.2, 0.4])
    with pytest.raises(ValueError, match="probabilities"):
        reduce_predictions(bad)
    full = _records([0.8, 0.3, 0.6, 0.4])
    full[0]["boundary"] = "FULL"
    with pytest.raises(MetricBoundaryError, match="Original"):
        reduce_predictions(full)
    context = _records([0.8, 0.3, 0.6, 0.4])
    context[0]["fit_scope"] = "same-row-context"
    with pytest.raises(MetricBoundaryError, match="prior-only"):
        reduce_predictions(context)


def test_metric_gap_and_event_block_bootstrap_are_paired_and_deterministic():
    baseline = _records([0.55, 0.45, 0.55, 0.45])
    candidate = _records([0.8, 0.2, 0.7, 0.6])
    interval_one = event_block_bootstrap_delta(candidate, baseline, iterations=200, seed=17)
    interval_two = event_block_bootstrap_delta(candidate, baseline, iterations=200, seed=17)
    assert interval_one == interval_two
    assert interval_one["log_loss_delta"]["estimate"] < 0
    assert interval_one["brier_delta"]["estimate"] < 0
    assert interval_one["accuracy_delta"]["estimate"] > 0

    train = reduce_predictions(_records([0.99, 0.01, 0.99, 0.99]))
    outer = reduce_predictions(_records([0.8, 0.3, 0.6, 0.4]))
    gaps = metric_gap(train, outer)
    assert gaps["accuracy"] > 0
    assert gaps["log_loss"] < 0


def test_paired_bootstrap_rejects_misaligned_fights_and_events():
    baseline = _records([0.55, 0.45, 0.55, 0.45])
    candidate = _records([0.8, 0.2, 0.7, 0.6])
    candidate[0]["fight_id"] = "wrong"
    with pytest.raises(ValueError, match="aligned"):
        event_block_bootstrap_delta(candidate, baseline, iterations=20, seed=1)
