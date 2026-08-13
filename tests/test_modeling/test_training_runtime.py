from __future__ import annotations

import sys
import tomllib
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pandas as pd
import pytest
from packaging.requirements import Requirement

from libs.modeling import train, walk_forward


ROOT = Path(__file__).resolve().parents[2]
STOCK_PRESETS = ("noncommercial", "extreme")


class FitStopped(Exception):
    """Stop a mocked production fit after its public call contract is observed."""


def test_project_pins_autogluon_and_stable_cuda_torch() -> None:
    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    dependencies = project["project"]["dependencies"]

    assert "autogluon.tabular[tabarena]==1.6.1" in dependencies
    assert "torch==2.10.0" in dependencies
    assert project["tool"]["uv"]["sources"]["torch"]["index"] == "pytorch-cu130"
    index = next(item for item in project["tool"]["uv"]["index"] if item["name"] == "pytorch-cu130")
    assert index == {
        "name": "pytorch-cu130",
        "url": "https://download.pytorch.org/whl/cu130",
        "explicit": True,
    }


def test_lock_contains_exact_autogluon_and_foundation_distributions() -> None:
    lock = tomllib.loads((ROOT / "uv.lock").read_text(encoding="utf-8"))
    packages = {(package["name"], package["version"]) for package in lock["package"]}

    for name in ("autogluon-common", "autogluon-core", "autogluon-features", "autogluon-tabular"):
        assert (name, "1.6.1") in packages
    assert ("torch", "2.10.0+cu130") in packages
    for name in ("tabpfn", "tabicl", "tabdpt", "pytabkit", "synthefy-nori"):
        assert any(package_name == name for package_name, _ in packages)


def test_project_uses_only_autogluons_xgboost_cpu_distribution() -> None:
    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    direct_distributions = {
        Requirement(dependency).name for dependency in project["project"]["dependencies"]
    }
    lock = tomllib.loads((ROOT / "uv.lock").read_text(encoding="utf-8"))
    locked_distributions = {
        (package["name"], package["version"]) for package in lock["package"]
    }

    assert "xgboost" not in direct_distributions
    assert not any(name == "xgboost" for name, _ in locked_distributions)
    assert ("xgboost-cpu", "3.3.0") in locked_distributions


def test_training_config_defaults_to_extreme_and_accepts_stock_aliases() -> None:
    assert train.TrainingConfig(model_type="win").preset == "extreme"
    for preset in STOCK_PRESETS:
        assert train.TrainingConfig(model_type="win", preset=preset).preset == preset

    with pytest.raises(ValueError, match="noncommercial.*extreme"):
        train.TrainingConfig(model_type="win", preset="best")


def test_training_config_rejects_model_allow_list_for_stock_presets() -> None:
    with pytest.raises(ValueError, match="included_model_types.*stock presets"):
        train.TrainingConfig(model_type="win", included_model_types=["GBM"])


@pytest.mark.parametrize("preset", STOCK_PRESETS)
@pytest.mark.parametrize("with_tuning_data", (False, True))
def test_shared_fit_kwargs_forward_stock_presets_without_portfolio_overrides(
    preset: str,
    with_tuning_data: bool,
) -> None:
    train_data = pd.DataFrame({"feature": [0, 1], "y_true": [0, 1]})
    tuning_data = train_data.copy() if with_tuning_data else None

    kwargs = train.build_training_fit_kwargs(
        train.TrainingConfig(model_type="win", preset=preset, time_limit=17),
        train_data=train_data,
        tuning_data=tuning_data,
    )

    assert kwargs["presets"] == preset
    assert kwargs["time_limit"] == 17
    assert kwargs["num_gpus"] == 1
    assert kwargs["raise_on_model_failure"] is True
    assert kwargs["train_data"] is train_data
    if with_tuning_data:
        assert kwargs["tuning_data"] is tuning_data
        assert kwargs["use_bag_holdout"] is True
    else:
        assert "tuning_data" not in kwargs
        assert "use_bag_holdout" not in kwargs

    forbidden = {
        "included_model_types",
        "excluded_model_types",
        "hyperparameters",
        "num_bag_folds",
        "num_stack_levels",
        "auto_stack",
        "dynamic_stacking",
        "ag_args_fit",
        "ag_args_ensemble",
    }
    assert forbidden.isdisjoint(kwargs)


def test_standard_fit_uses_shared_kwargs_seam(monkeypatch: pytest.MonkeyPatch) -> None:
    config = train.TrainingConfig(model_type="win", time_limit=17)
    trainer = train.ModelTrainer(config)
    predictor = Mock()
    predictor.fit.side_effect = FitStopped
    shared_kwargs = {"shared_contract": object()}
    builder = Mock(return_value=shared_kwargs)
    monkeypatch.setattr(train, "build_training_fit_kwargs", builder)
    X_train = pd.DataFrame({"feature": [0, 1]})
    y_train = pd.Series([0, 1], name="y_true")

    with pytest.raises(FitStopped):
        trainer._train_model(predictor, X_train, y_train, pd.DataFrame(), pd.Series(dtype=int))

    fit_train_data = builder.call_args.kwargs["train_data"]
    builder.assert_called_once_with(config, train_data=fit_train_data)
    pd.testing.assert_frame_equal(fit_train_data, pd.concat([X_train, y_train], axis=1))
    predictor.fit.assert_called_once_with(**shared_kwargs)


def test_timeseries_fit_uses_shared_kwargs_seam(monkeypatch: pytest.MonkeyPatch) -> None:
    config = train.TrainingConfig(model_type="win", split_strategy="timeseries_split")
    trainer = train.TimeseriesSplitTrainer(config, "unused-model-dir", pd.DataFrame())
    train_data = pd.DataFrame({"feature": [0], "y_true": [0]})
    tune_data = pd.DataFrame({"feature": [1], "y_true": [1]})
    trainer.split_data = Mock(return_value=(
        train_data,
        tune_data,
        pd.DatetimeIndex(["2024-01-01"]),
        pd.DatetimeIndex(["2025-01-01"]),
        pd.Timestamp("2025-01-01"),
        None,
    ))
    trainer._prepare_training_data = Mock(return_value=(train_data, tune_data))
    predictor = Mock()
    predictor.fit.side_effect = FitStopped
    monkeypatch.setattr(train, "training_runtime_preflight", Mock())
    monkeypatch.setattr(train, "TabularPredictor", Mock(return_value=predictor))
    shared_kwargs = {"shared_contract": object()}
    builder = Mock(return_value=shared_kwargs)
    monkeypatch.setattr(train, "build_training_fit_kwargs", builder)

    with pytest.raises(FitStopped):
        trainer.train(pd.DataFrame(), pd.Series(dtype=int))

    builder.assert_called_once_with(config, train_data=train_data, tuning_data=tune_data)
    predictor.fit.assert_called_once_with(**shared_kwargs)


def test_internal_walkforward_fit_uses_shared_kwargs_seam(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    config = train.TrainingConfig(model_type="win", split_strategy="walkforward", normalize="none")
    trainer = train.WalkForwardTrainer(config)
    trainer.model_dir = str(tmp_path)
    trainer.full_df = pd.DataFrame(index=[0, 1, 2, 3])
    trainer.holdout_data = pd.DataFrame()
    trainer.selected_features = ["feature"]
    X_full = pd.DataFrame({"feature": [0, 1, 2, 3]})
    y_full = pd.Series([0, 1, 0, 1], name="y_true")
    fold = {
        "train_indices": pd.Index([0, 1]),
        "val_indices": pd.Index([2, 3]),
        "train_range": "2023-2024",
        "val_range": "2025",
    }
    predictor = Mock()
    predictor.fit.side_effect = FitStopped
    monkeypatch.setattr(train, "TabularPredictor", Mock(return_value=predictor))
    shared_kwargs = {"shared_contract": object()}
    builder = Mock(return_value=shared_kwargs)
    monkeypatch.setattr(train, "build_training_fit_kwargs", builder)

    with pytest.raises(FitStopped):
        trainer._train_window_model(fold, X_full, y_full, window_idx=0)

    fit_train_data = builder.call_args.kwargs["train_data"]
    fit_tuning_data = builder.call_args.kwargs["tuning_data"]
    builder.assert_called_once_with(config, train_data=fit_train_data, tuning_data=fit_tuning_data)
    assert fit_train_data["y_true"].tolist() == [0, 1]
    assert fit_tuning_data["y_true"].tolist() == [0, 1]
    predictor.fit.assert_called_once_with(**shared_kwargs)


def test_standalone_walkforward_fit_uses_shared_kwargs_seam(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    config = train.TrainingConfig(model_type="win", split_strategy="walkforward", normalize="none")
    validator = walk_forward.WalkForwardValidator(
        walk_forward.WalkForwardConfig(base_config=config, output_dir=str(tmp_path), save_fold_models=True)
    )
    X_full = pd.DataFrame({"feature": [0, 1, 2, 3]})
    y_full = pd.Series([0, 1, 0, 1], name="y_true")
    fold = {
        "train_idx": pd.Index([0, 1]),
        "test_idx": pd.Index([2, 3]),
        "train_range": "2023-2024",
        "test_range": "2025",
    }
    predictor = Mock()
    predictor.fit.side_effect = FitStopped
    monkeypatch.setattr(walk_forward, "TabularPredictor", Mock(return_value=predictor))
    shared_kwargs = {"shared_contract": object()}
    builder = Mock(return_value=shared_kwargs)
    monkeypatch.setattr(walk_forward, "build_training_fit_kwargs", builder)

    with pytest.raises(FitStopped):
        validator._process_fold(fold, X_full, y_full, fold_idx=0)

    fit_train_data = builder.call_args.kwargs["train_data"]
    builder.assert_called_once_with(config, train_data=fit_train_data)
    assert fit_train_data["y_true"].tolist() == [0, 1]
    predictor.fit.assert_called_once_with(**shared_kwargs)


def test_refit_fit_uses_shared_kwargs_seam(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    config = train.TrainingConfig(model_type="win", split_strategy="timeseries_split", normalize="none")
    trainer = train.RefitAllTrainer(
        config,
        str(tmp_path / "model"),
        pd.DataFrame(),
        pd.DataFrame(),
        ["feature"],
    )
    full_df = pd.DataFrame({"event_date": pd.to_datetime(["2024-01-01", "2025-01-01"])})
    X_full = pd.DataFrame({"feature": [0, 1]})
    y_full = pd.Series([0, 1], name="y_true")
    train_data = pd.DataFrame({"feature": [0], "y_true": [0]})
    tune_data = pd.DataFrame({"feature": [1], "y_true": [1]})
    trainer.prepare_refit_data = Mock(return_value=(full_df, X_full, y_full))
    split_trainer = Mock()
    split_trainer.split_data.return_value = (
        train_data,
        tune_data,
        pd.DatetimeIndex(["2024-01-01"]),
        pd.DatetimeIndex(["2025-01-01"]),
        pd.Timestamp("2025-01-01"),
        None,
    )
    split_trainer._prepare_training_data.return_value = (train_data, tune_data)
    monkeypatch.setattr(train, "training_runtime_preflight", Mock())
    monkeypatch.setattr(train, "TimeseriesSplitTrainer", Mock(return_value=split_trainer))
    predictor = Mock()
    predictor.fit.side_effect = FitStopped
    monkeypatch.setattr(train, "TabularPredictor", Mock(return_value=predictor))
    shared_kwargs = {"shared_contract": object()}
    builder = Mock(return_value=shared_kwargs)
    monkeypatch.setattr(train, "build_training_fit_kwargs", builder)

    with pytest.raises(FitStopped):
        trainer.train()

    builder.assert_called_once_with(config, train_data=train_data, tuning_data=tune_data)
    predictor.fit.assert_called_once_with(**shared_kwargs)


def test_main_and_cli_default_and_forward_presets(monkeypatch: pytest.MonkeyPatch) -> None:
    captured = []

    class FakeTrainer:
        def __init__(self, config: train.TrainingConfig):
            captured.append(config)

        def train(self):
            return "predictor"

    monkeypatch.setattr(train, "ModelTrainer", FakeTrainer)
    assert train.main(refit_full=False) == "predictor"
    assert captured[-1].preset == "extreme"

    monkeypatch.setattr(sys, "argv", ["mma-train", "--no-refit-full"])
    assert train.cli() == "predictor"
    assert captured[-1].preset == "extreme"

    monkeypatch.setattr(sys, "argv", ["mma-train", "--preset", "noncommercial", "--no-refit-full"])
    assert train.cli() == "predictor"
    assert captured[-1].preset == "noncommercial"


class FakeTensor:
    def __init__(self, value: float):
        self.value = value

    def __mul__(self, other: float) -> "FakeTensor":
        return FakeTensor(self.value * other)

    def item(self) -> float:
        return self.value


def fake_torch(
    *,
    available: bool = True,
    count: int = 1,
    name: str = "NVIDIA Test GPU",
    tensor_error: bool = False,
):
    cuda = SimpleNamespace(
        is_available=Mock(return_value=available),
        device_count=Mock(return_value=count),
        get_device_name=Mock(return_value=name),
        get_device_capability=Mock(return_value=(12, 0)),
    )
    def tensor(values, device):
        if tensor_error:
            raise RuntimeError("kernel launch failed")
        return FakeTensor(values[0])

    return SimpleNamespace(
        __version__="2.10.0+cu130",
        version=SimpleNamespace(cuda="13.0"),
        cuda=cuda,
        tensor=tensor,
    )


def install_runtime_mocks(
    monkeypatch: pytest.MonkeyPatch,
    *,
    torch_module=None,
    missing_module: str | None = None,
    autogluon_version: str = "1.6.1",
) -> None:
    versions = {
        "autogluon-common": autogluon_version,
        "autogluon-core": autogluon_version,
        "autogluon-features": autogluon_version,
        "autogluon-tabular": autogluon_version,
    }
    monkeypatch.setattr(train.importlib_metadata, "version", lambda name: versions[name])

    torch_module = torch_module or fake_torch()

    def import_module(name: str):
        if name == missing_module:
            raise ImportError(f"missing {name}")
        if name == "torch":
            return torch_module
        return SimpleNamespace()

    monkeypatch.setattr(train.importlib, "import_module", import_module)


def test_runtime_preflight_reports_the_verified_cuda_operation(monkeypatch: pytest.MonkeyPatch) -> None:
    install_runtime_mocks(monkeypatch)

    report = train.training_runtime_preflight()

    assert report == {
        "autogluon_version": "1.6.1",
        "torch_version": "2.10.0+cu130",
        "torch_cuda_version": "13.0",
        "cuda_available": True,
        "device_count": 1,
        "device_name": "NVIDIA Test GPU",
        "compute_capability": "12.0",
        "tensor_result": 6.0,
    }


@pytest.mark.parametrize(
    ("runtime_change", "message"),
    [
        ({"autogluon_version": "1.6.0"}, "AutoGluon.*1.6.1"),
        ({"missing_module": "tabpfn"}, "tabpfn"),
        ({"torch_module": fake_torch(available=False)}, "CUDA is unavailable"),
        ({"torch_module": fake_torch(count=0)}, "exactly one.*0"),
        ({"torch_module": fake_torch(count=2)}, "exactly one.*2"),
        ({"torch_module": fake_torch(name="AMD Test GPU")}, "NVIDIA"),
        ({"torch_module": fake_torch(tensor_error=True)}, "CUDA tensor operation failed"),
    ],
)
def test_runtime_preflight_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
    runtime_change: dict,
    message: str,
) -> None:
    install_runtime_mocks(monkeypatch, **runtime_change)

    with pytest.raises(RuntimeError, match=message):
        train.training_runtime_preflight()


def test_model_trainer_preflights_before_model_directory_creation(monkeypatch: pytest.MonkeyPatch) -> None:
    preflight = Mock(side_effect=RuntimeError("preflight stopped training"))
    create_directory = Mock()
    monkeypatch.setattr(train, "training_runtime_preflight", preflight)
    monkeypatch.setattr(train.FileManager, "create_model_directory", create_directory)

    trainer = train.ModelTrainer(train.TrainingConfig(model_type="win"))
    with pytest.raises(RuntimeError, match="preflight stopped training"):
        trainer.train()

    preflight.assert_called_once_with()
    create_directory.assert_not_called()


def test_timeseries_trainer_preflights_before_predictor_fit(monkeypatch: pytest.MonkeyPatch) -> None:
    preflight = Mock(side_effect=RuntimeError("preflight stopped training"))
    predictor = Mock()
    monkeypatch.setattr(train, "training_runtime_preflight", preflight)
    predictor_constructor = Mock(return_value=predictor)
    monkeypatch.setattr(train, "TabularPredictor", predictor_constructor)
    trainer = train.TimeseriesSplitTrainer(
        train.TrainingConfig(model_type="win", split_strategy="timeseries_split"),
        "unused-model-dir",
        pd.DataFrame(),
    )

    with pytest.raises(RuntimeError, match="preflight stopped training"):
        trainer.train(pd.DataFrame(), pd.Series(dtype=int))

    preflight.assert_called_once_with()
    predictor_constructor.assert_not_called()
    predictor.fit.assert_not_called()


def test_internal_walkforward_preflights_before_model_directory_creation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    preflight = Mock(side_effect=RuntimeError("preflight stopped training"))
    create_directory = Mock()
    predictor_constructor = Mock()
    monkeypatch.setattr(train, "training_runtime_preflight", preflight)
    monkeypatch.setattr(train.FileManager, "create_model_directory", create_directory)
    monkeypatch.setattr(train, "TabularPredictor", predictor_constructor)
    trainer = train.WalkForwardTrainer(
        train.TrainingConfig(model_type="win", split_strategy="walkforward")
    )

    with pytest.raises(RuntimeError, match="preflight stopped training"):
        trainer.train()

    preflight.assert_called_once_with()
    create_directory.assert_not_called()
    predictor_constructor.assert_not_called()


def test_standalone_walkforward_preflights_before_output_directory_creation(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    preflight = Mock(side_effect=RuntimeError("preflight stopped training"))
    make_directory = Mock()
    predictor_constructor = Mock()
    monkeypatch.setattr(walk_forward, "training_runtime_preflight", preflight)
    monkeypatch.setattr(walk_forward.os, "makedirs", make_directory)
    monkeypatch.setattr(walk_forward, "TabularPredictor", predictor_constructor)
    validator = walk_forward.WalkForwardValidator(
        walk_forward.WalkForwardConfig(
            base_config=train.TrainingConfig(model_type="win", split_strategy="walkforward"),
            output_dir=str(tmp_path),
        )
    )

    with pytest.raises(RuntimeError, match="preflight stopped training"):
        validator.run()

    preflight.assert_called_once_with()
    make_directory.assert_not_called()
    predictor_constructor.assert_not_called()


def test_refit_trainer_preflights_before_output_directory_creation(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    preflight = Mock(side_effect=RuntimeError("preflight stopped training"))
    make_directory = Mock()
    predictor_constructor = Mock()
    monkeypatch.setattr(train, "training_runtime_preflight", preflight)
    monkeypatch.setattr(train.os, "makedirs", make_directory)
    monkeypatch.setattr(train, "TabularPredictor", predictor_constructor)
    trainer = train.RefitAllTrainer(
        train.TrainingConfig(model_type="win", split_strategy="timeseries_split"),
        str(tmp_path / "model"),
        pd.DataFrame(),
        pd.DataFrame(),
        [],
    )

    with pytest.raises(RuntimeError, match="preflight stopped training"):
        trainer.train()

    preflight.assert_called_once_with()
    make_directory.assert_not_called()
    predictor_constructor.assert_not_called()
