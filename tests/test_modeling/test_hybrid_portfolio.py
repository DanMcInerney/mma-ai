import copy
import json

import pandas as pd
from autogluon.core.metrics import get_metric
from autogluon.tabular.trainer.model_presets.presets import get_preset_models

from libs.modeling import train


EXPECTED_HYBRID_KEYS = {
    "CAT",
    "GBM",
    "XT",
    "RF",
    "FASTAI",
    "REALMLP",
    "XGB",
    "MITRA",
    "TABICL",
}
EXPECTED_HYBRID_MODELS = {
    "CatBoost",
    "PrepLightGBM",
    "LightGBM_r8",
    "ExtraTreesGini",
    "RandomForestGini",
    "NeuralNetFastAI",
    "RealMLP_r9",
    "XGBoost",
    "Mitra",
    "TabICL",
}
FORBIDDEN_TOKENS = ("NORI", "TABDPT", "TABPFN", "REALTABPFN")


def _hybrid_config(**overrides):
    values = {
        "model_type": "win",
        "preset": "hybrid",
        "time_limit": 3000,
        "split_strategy": "timeseries_split",
        "refit_full": True,
    }
    values.update(overrides)
    return train.TrainingConfig(**values)


def _all_mapping_keys(value):
    if isinstance(value, dict):
        for key, item in value.items():
            yield str(key)
            yield from _all_mapping_keys(item)
    elif isinstance(value, list):
        for item in value:
            yield from _all_mapping_keys(item)


def test_hybrid_portfolio_is_exact_and_has_no_forbidden_gate():
    fit_kwargs = train.build_training_fit_kwargs(
        _hybrid_config(),
        train_data=pd.DataFrame({"feature": [0, 1], "y_true": [0, 1]}),
        tuning_data=pd.DataFrame({"feature": [2], "y_true": [1]}),
    )

    hyperparameters = fit_kwargs["hyperparameters"]
    assert "presets" not in fit_kwargs
    assert set(hyperparameters) == EXPECTED_HYBRID_KEYS
    assert sum(len(configs) for configs in hyperparameters.values()) == 10
    assert hyperparameters["MITRA"] == [{}]
    assert hyperparameters["TABICL"] == [{}]
    assert hyperparameters["CAT"] == train.CLASSICAL_HYPERPARAMETERS["CAT"]
    assert hyperparameters["GBM"] == train.CLASSICAL_HYPERPARAMETERS["GBM"]

    serialized = json.dumps(hyperparameters, sort_keys=True).upper()
    assert not any(token in serialized for token in FORBIDDEN_TOKENS)
    assert not any("ROW" in key.upper() for key in _all_mapping_keys(hyperparameters))


def test_hybrid_portfolio_resolves_exact_models(tmp_path):
    fit_kwargs = train.build_training_fit_kwargs(
        _hybrid_config(),
        train_data=pd.DataFrame({"feature": [0, 1], "y_true": [0, 1]}),
    )
    resolved, _ = get_preset_models(
        path=str(tmp_path),
        problem_type="binary",
        eval_metric=get_metric("log_loss"),
        hyperparameters=copy.deepcopy(fit_kwargs["hyperparameters"]),
    )

    assert {model.name for model in resolved} == EXPECTED_HYBRID_MODELS
    assert {type(model).__name__ for model in resolved if model.name == "Mitra"} == {
        "MitraModel"
    }
    assert {type(model).__name__ for model in resolved if model.name == "TabICL"} == {
        "TabICLModel"
    }


def test_hybrid_fit_kwargs_are_isolated_from_classical():
    hybrid = train.build_training_fit_kwargs(
        _hybrid_config(),
        train_data=pd.DataFrame({"feature": [0], "y_true": [0]}),
    )["hyperparameters"]
    hybrid["CAT"][0]["test_mutation"] = True

    classical = train.build_training_fit_kwargs(
        train.TrainingConfig(model_type="win", preset="classical"),
        train_data=pd.DataFrame({"feature": [0], "y_true": [0]}),
    )["hyperparameters"]
    assert "test_mutation" not in classical["CAT"][0]
    assert "MITRA" not in classical
    assert "TABICL" not in classical


def test_hybrid_path_keeps_fixed_training_defaults(monkeypatch):
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
        preset="hybrid",
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


def test_hybrid_model_directory_keeps_preset_suffix(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    model_dir = train.FileManager.create_model_directory("win", "hybrid")
    assert model_dir.startswith("AutogluonModels")
    assert model_dir.endswith("-win-hybrid")
