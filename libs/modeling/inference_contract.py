"""Fail-closed compatibility checks for saved prediction artifacts."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Iterable

from libs.modeling.train import NormalizationManager
from libs.modeling.training_profiles import (
    WIN_V8_HYBRID_WORKING_PROFILE,
    WIN_V8_HYBRID_WORKING_PROFILE_NAME,
)


EXPECTED_V8_FEATURE_SHA256 = (
    "13E545D762A3F1BE4D023D82B8E65D77E41589031051F1F6796D742F25223022"
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

    return {
        "profile": WIN_V8_HYBRID_WORKING_PROFILE_NAME,
        "feature_count": len(saved_features),
        "feature_sha256": feature_sha256,
        "normalization": WIN_V8_HYBRID_WORKING_PROFILE["normalize"],
        "scaled_feature_count": len(expected_scaled_features),
        "recency_weights": WIN_V8_HYBRID_WORKING_PROFILE["use_recency_weights"],
        "decay_rate": WIN_V8_HYBRID_WORKING_PROFILE["decay_rate"],
    }
