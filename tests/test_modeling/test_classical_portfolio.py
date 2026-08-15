import json

import pandas as pd
import pytest

from libs.modeling import train


EXPECTED_CLASSICAL_KEYS = {"CAT", "GBM", "XT", "RF", "FASTAI", "REALMLP", "XGB"}
FORBIDDEN_CONTEXT_KEYS = {
    "TABICL",
    "TABDPT",
    "TABDPT-TURBO",
    "TABPFN",
    "TABPFNMIX",
    "MITRA",
    "NORI",
}


def _classical_config(**overrides):
    values = {
        "model_type": "win",
        "preset": "classical",
        "time_limit": 3000,
        "split_strategy": "timeseries_split",
        "refit_full": True,
    }
    values.update(overrides)
    return train.TrainingConfig(**values)


def test_classical_portfolio_is_exact_and_context_free():
    config = _classical_config()
    fit_kwargs = train.build_training_fit_kwargs(
        config,
        train_data=pd.DataFrame({"feature": [0, 1], "y_true": [0, 1]}),
        tuning_data=pd.DataFrame({"feature": [2], "y_true": [1]}),
    )

    assert "presets" not in fit_kwargs
    assert set(fit_kwargs["hyperparameters"]) == EXPECTED_CLASSICAL_KEYS
    assert len(fit_kwargs["hyperparameters"]["GBM"]) == 2
    assert sum(
        len(configs) if isinstance(configs, list) else 1
        for configs in fit_kwargs["hyperparameters"].values()
    ) == 8

    serialized = json.dumps(fit_kwargs["hyperparameters"], sort_keys=True).upper()
    assert "AG.MIN_ROWS" not in serialized
    assert not any(key in serialized for key in FORBIDDEN_CONTEXT_KEYS)

    prep_gbm, standard_gbm = fit_kwargs["hyperparameters"]["GBM"]
    assert prep_gbm["ag_args"]["name_prefix"] == "Prep"
    assert "ag.model_specific_feature_generator_kwargs" in prep_gbm
    assert standard_gbm["ag_args"]["name_suffix"] == "_r8"


@pytest.mark.parametrize("preset", train.STOCK_PRESETS)
def test_stock_preset_fit_contract_is_unchanged(preset):
    config = train.TrainingConfig(model_type="win", preset=preset)
    fit_kwargs = train.build_training_fit_kwargs(
        config,
        train_data=pd.DataFrame({"feature": [0], "y_true": [0]}),
    )

    assert fit_kwargs["presets"] == preset
    assert "hyperparameters" not in fit_kwargs


def test_unknown_preset_and_stock_allow_list_remain_rejected():
    with pytest.raises(ValueError, match="preset must be"):
        train.TrainingConfig(model_type="win", preset="unknown")

    with pytest.raises(ValueError, match="included_model_types is unsupported"):
        train.TrainingConfig(
            model_type="win",
            preset="extreme",
            included_model_types=["CAT"],
        )


def test_classical_path_keeps_fixed_training_defaults(monkeypatch):
    captured = {}

    class StubTrainer:
        def __init__(self, config):
            captured["config"] = config

        def train(self):
            return "predictor"

    monkeypatch.setattr(train, "ModelTrainer", StubTrainer)
    result = train.main(
        model_type="win",
        time_limit=3000,
        preset="classical",
        split_strategy="timeseries_split",
        refit_full=True,
    )

    config = captured["config"]
    assert result == "predictor"
    assert config.features == train.vSeven_testing2
    assert len(config.features) == 40
    assert config.test_size is None
    assert config.include_split_dec is True
    assert config.normalize == "robust"
    assert config.use_recency_weights is True
    assert config.decay_rate == 0.15
    assert config.refit_full is True


def test_classical_model_directory_keeps_preset_suffix(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    model_dir = train.FileManager.create_model_directory("win", "classical")
    assert model_dir.startswith("AutogluonModels")
    assert model_dir.endswith("-win-classical")
