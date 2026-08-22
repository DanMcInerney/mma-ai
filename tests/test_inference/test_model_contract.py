from __future__ import annotations

from pathlib import Path

import pytest

from libs.modeling.inference_contract import (
    EXPECTED_V8_FEATURE_SHA256,
    validate_weighted_v8_inference_contract,
)
from libs.modeling.training_profiles import WIN_V8_HYBRID_WORKING_PROFILE


class SavedScaler:
    def __init__(self, feature_names: list[str]) -> None:
        self.feature_names_in_ = feature_names
        self.n_features_in_ = len(feature_names)


def _write_features(path: Path, features: list[str]) -> None:
    path.write_text("\n".join(features) + "\n", encoding="utf-8")


def test_weighted_v8_prediction_contract_matches_training_profile(tmp_path: Path) -> None:
    features = list(WIN_V8_HYBRID_WORKING_PROFILE["features"])
    model_path = tmp_path / "ag-20260815_090928-win-hybrid"
    model_path.mkdir()
    _write_features(model_path / "feats.txt", features)
    scaled_features = [name for name in features if name != "weightclass_encoded"]

    result = validate_weighted_v8_inference_contract(
        model_path,
        SavedScaler(scaled_features),
    )

    assert result == {
        "profile": "v8-hybrid-weighted",
        "feature_count": 40,
        "feature_sha256": EXPECTED_V8_FEATURE_SHA256,
        "normalization": "robust",
        "scaled_feature_count": 39,
        "recency_weights": True,
        "decay_rate": 0.15,
    }


def test_weighted_v8_prediction_contract_rejects_saved_scaler_drift(tmp_path: Path) -> None:
    features = list(WIN_V8_HYBRID_WORKING_PROFILE["features"])
    model_path = tmp_path / "ag-20260815_090928-win-hybrid"
    model_path.mkdir()
    _write_features(model_path / "feats.txt", features)

    with pytest.raises(ValueError, match="saved scaler feature order"):
        validate_weighted_v8_inference_contract(
            model_path,
            SavedScaler(list(reversed(features[1:]))),
        )
