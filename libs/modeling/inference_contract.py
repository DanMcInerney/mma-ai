"""Fail-closed compatibility checks for saved prediction artifacts."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from libs.modeling.train import NormalizationManager
from libs.modeling.training_profiles import (
    WIN_V8_HYBRID_WORKING_PROFILE,
    WIN_V8_HYBRID_WORKING_PROFILE_NAME,
)


EXPECTED_V8_FEATURE_SHA256 = (
    "13E545D762A3F1BE4D023D82B8E65D77E41589031051F1F6796D742F25223022"
)
EXPECTED_V8_DECAY_RATE = 0.15
FORBIDDEN_PREDICTION_COLUMNS = frozenset(
    {"sample_weight", "y_true", "event_date", "fight_date", "fight_id"}
)


def ordered_feature_sha256(features: Iterable[str]) -> str:
    payload = json.dumps(
        list(features),
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest().upper()


def inference_features_to_scale(features: Iterable[str]) -> list[str]:
    """Use the training normalizer's exclusions for prediction-time scaling."""
    return [
        str(feature)
        for feature in features
        if not NormalizationManager._should_exclude_col(str(feature))
    ]


def _validate_saved_recency_weights(
    model_path: Path,
    training_features: list[str],
) -> dict[str, object]:
    """Prove the saved training rows carry the weighted-v8 decay schedule."""
    training_data_path = model_path / "training_data.csv"
    saved_training_path = model_path / "utils" / "data" / "X.pkl"
    saved_evaluation_path = model_path / "utils" / "data" / "X_val.pkl"
    try:
        event_dates = pd.to_datetime(
            pd.read_csv(training_data_path, usecols=["event_date"])["event_date"],
            errors="raise",
        )
        saved_training = pd.read_pickle(saved_training_path)
        saved_evaluation = pd.read_pickle(saved_evaluation_path)
    except (FileNotFoundError, KeyError, ValueError) as exc:
        raise ValueError("saved model lacks weighted-v8 training evidence") from exc

    saved_features = [str(column) for column in saved_training.columns if column != "sample_weight"]
    saved_evaluation_features = [
        str(column) for column in saved_evaluation.columns if column != "sample_weight"
    ]
    if (
        saved_features != training_features
        or saved_evaluation_features != training_features
        or "sample_weight" not in saved_training
        or "sample_weight" not in saved_evaluation
    ):
        raise ValueError("saved training matrix does not match weighted-v8 features and weights")
    forbidden = sorted(FORBIDDEN_PREDICTION_COLUMNS.intersection(training_features))
    if forbidden:
        raise ValueError(f"weight/date/label columns entered prediction features: {forbidden}")
    if event_dates.empty or event_dates.isna().any():
        raise ValueError("saved training event dates are incomplete")

    try:
        training_indices = saved_training.index.to_numpy(dtype=int)
    except (OverflowError, TypeError, ValueError) as exc:
        raise ValueError("saved training rows do not resolve into training event dates") from exc
    if (
        not len(training_indices)
        or len(set(training_indices.tolist())) != len(training_indices)
        or training_indices.min() < 0
        or training_indices.max() >= len(event_dates)
    ):
        raise ValueError("saved training rows do not resolve uniquely into training event dates")

    years_ago = (event_dates.max() - event_dates).dt.days.to_numpy(dtype=float) / 365.25
    all_expected = np.exp(-EXPECTED_V8_DECAY_RATE * years_ago)
    all_expected *= len(all_expected) / all_expected.sum()
    expected = all_expected[training_indices]
    actual = saved_training["sample_weight"].to_numpy(dtype=float)
    if (
        len(actual) != len(expected)
        or not np.all(np.isfinite(actual))
        or not np.allclose(actual, expected, rtol=0.0, atol=1e-12)
    ):
        raise ValueError("saved sample weights do not match weighted-v8 decay 0.15")
    evaluation_weights = saved_evaluation["sample_weight"].to_numpy(dtype=float)
    if (
        not len(evaluation_weights)
        or not np.all(np.isfinite(evaluation_weights))
        or not np.allclose(evaluation_weights, 1.0, rtol=0.0, atol=0.0)
    ):
        raise ValueError("saved validation evaluation weights are not unit contribution")

    return {
        "training_weight_rows": len(actual),
        "training_weight_min": float(actual.min()),
        "training_weight_max": float(actual.max()),
        "training_weight_max_abs_error": float(np.max(np.abs(actual - expected))),
        "evaluation_weight_rows": len(evaluation_weights),
        "evaluation_weights_unit": True,
        "prediction_forbidden_columns": forbidden,
    }


def validate_weighted_v8_inference_contract(model_path: Path, scaler: object) -> dict[str, object]:
    """Validate the accepted weighted-v8 feature and saved-scaler contract."""
    model_path = Path(model_path)
    saved_features = [
        line.strip()
        for line in (model_path / "feats.txt").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    training_features = list(WIN_V8_HYBRID_WORKING_PROFILE["features"])
    feature_sha256 = ordered_feature_sha256(saved_features)

    if saved_features != training_features or feature_sha256 != EXPECTED_V8_FEATURE_SHA256:
        raise ValueError(
            "saved model features do not match the ordered weighted-v8 training contract"
        )

    expected_scaled_features = inference_features_to_scale(training_features)
    saved_scaled_features = [str(value) for value in getattr(scaler, "feature_names_in_", [])]
    if saved_scaled_features != expected_scaled_features:
        raise ValueError("saved scaler feature order does not match weighted-v8 training")
    if int(getattr(scaler, "n_features_in_", -1)) != len(expected_scaled_features):
        raise ValueError("saved scaler feature count does not match weighted-v8 training")

    if (
        WIN_V8_HYBRID_WORKING_PROFILE["use_recency_weights"] is not True
        or float(WIN_V8_HYBRID_WORKING_PROFILE["decay_rate"]) != EXPECTED_V8_DECAY_RATE
    ):
        raise ValueError("weighted-v8 source profile recency contract changed")
    saved_weight_evidence = _validate_saved_recency_weights(model_path, training_features)

    return {
        "profile": WIN_V8_HYBRID_WORKING_PROFILE_NAME,
        "feature_count": len(saved_features),
        "feature_sha256": feature_sha256,
        "normalization": WIN_V8_HYBRID_WORKING_PROFILE["normalize"],
        "scaled_feature_count": len(expected_scaled_features),
        "recency_weights": WIN_V8_HYBRID_WORKING_PROFILE["use_recency_weights"],
        "decay_rate": WIN_V8_HYBRID_WORKING_PROFILE["decay_rate"],
        **saved_weight_evidence,
    }
