from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from libs.modeling.inference_contract import (
    EXPECTED_V8_FEATURE_SHA256,
    FORBIDDEN_PREDICTION_COLUMNS,
    validate_weighted_v8_inference_contract,
)
from libs.modeling.training_profiles import WIN_V8_HYBRID_WORKING_PROFILE


class SavedScaler:
    def __init__(self, feature_names: list[str]) -> None:
        self.feature_names_in_ = feature_names
        self.n_features_in_ = len(feature_names)


def _write_features(path: Path, features: list[str]) -> None:
    path.write_text("\n".join(features) + "\n", encoding="utf-8")


def _write_weight_evidence(path: Path, features: list[str]) -> np.ndarray:
    dates = pd.to_datetime(["2020-01-01", "2021-01-01", "2022-01-01"])
    pd.DataFrame({"event_date": dates}).to_csv(path / "training_data.csv", index=False)
    years_ago = (dates.max() - dates).days.to_numpy(dtype=float) / 365.25
    weights = np.exp(-0.15 * years_ago)
    weights *= len(weights) / weights.sum()
    saved = pd.DataFrame({feature: np.zeros(len(dates)) for feature in features})
    saved["sample_weight"] = weights
    saved_path = path / "utils" / "data"
    saved_path.mkdir(parents=True)
    saved.to_pickle(saved_path / "X.pkl")
    evaluation = pd.DataFrame({feature: np.zeros(2) for feature in features})
    evaluation["sample_weight"] = 1.0
    evaluation.to_pickle(saved_path / "X_val.pkl")
    return weights


def test_weighted_v8_prediction_contract_matches_training_profile(tmp_path: Path) -> None:
    features = list(WIN_V8_HYBRID_WORKING_PROFILE["features"])
    model_path = tmp_path / "ag-20260815_090928-win-hybrid"
    model_path.mkdir()
    _write_features(model_path / "feats.txt", features)
    weights = _write_weight_evidence(model_path, features)
    scaled_features = [name for name in features if name != "weightclass_encoded"]

    result = validate_weighted_v8_inference_contract(
        model_path,
        SavedScaler(scaled_features),
    )

    assert {key: result[key] for key in (
        "profile",
        "feature_count",
        "feature_sha256",
        "normalization",
        "scaled_feature_count",
        "recency_weights",
        "decay_rate",
    )} == {
        "profile": "v8-hybrid-weighted",
        "feature_count": 40,
        "feature_sha256": EXPECTED_V8_FEATURE_SHA256,
        "normalization": "robust",
        "scaled_feature_count": 39,
        "recency_weights": True,
        "decay_rate": 0.15,
    }
    assert result["training_weight_rows"] == 3
    assert result["training_weight_min"] == pytest.approx(float(weights.min()))
    assert result["training_weight_max"] == pytest.approx(float(weights.max()))
    assert result["training_weight_max_abs_error"] == 0.0
    assert result["evaluation_weight_rows"] == 2
    assert result["evaluation_weights_unit"] is True
    assert result["prediction_forbidden_columns"] == []
    assert FORBIDDEN_PREDICTION_COLUMNS.isdisjoint(features)


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


def test_weighted_v8_prediction_contract_rejects_saved_recency_weight_drift(
    tmp_path: Path,
) -> None:
    features = list(WIN_V8_HYBRID_WORKING_PROFILE["features"])
    model_path = tmp_path / "ag-20260815_090928-win-hybrid"
    model_path.mkdir()
    _write_features(model_path / "feats.txt", features)
    _write_weight_evidence(model_path, features)
    saved_path = model_path / "utils" / "data" / "X.pkl"
    saved = pd.read_pickle(saved_path)
    saved.loc[0, "sample_weight"] += 0.01
    saved.to_pickle(saved_path)
    scaled_features = [name for name in features if name != "weightclass_encoded"]

    with pytest.raises(ValueError, match="saved sample weights"):
        validate_weighted_v8_inference_contract(
            model_path,
            SavedScaler(scaled_features),
        )


def test_weighted_v8_prediction_contract_rejects_nonunit_validation_weights(
    tmp_path: Path,
) -> None:
    features = list(WIN_V8_HYBRID_WORKING_PROFILE["features"])
    model_path = tmp_path / "ag-20260815_090928-win-hybrid"
    model_path.mkdir()
    _write_features(model_path / "feats.txt", features)
    _write_weight_evidence(model_path, features)
    saved_path = model_path / "utils" / "data" / "X_val.pkl"
    saved = pd.read_pickle(saved_path)
    saved.loc[0, "sample_weight"] = 0.9
    saved.to_pickle(saved_path)
    scaled_features = [name for name in features if name != "weightclass_encoded"]

    with pytest.raises(ValueError, match="validation evaluation weights"):
        validate_weighted_v8_inference_contract(
            model_path,
            SavedScaler(scaled_features),
        )
