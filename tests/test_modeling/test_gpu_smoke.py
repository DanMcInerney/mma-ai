"""One bounded, GPU-required smoke for AutoGluon's stock extreme preset."""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from autogluon.tabular import TabularPredictor

from libs.modeling.train import training_runtime_preflight


@pytest.mark.gpu_smoke
@pytest.mark.timeout(600)
def test_extreme_gpu_smoke_produces_a_prediction(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("TABPFN_TOKEN", raising=False)
    training_runtime_preflight()
    rng = np.random.default_rng(161)
    features = rng.normal(size=(1200, 12))
    target = (features[:, 0] + 0.5 * features[:, 1] - features[:, 2] > 0).astype(int)
    data = pd.DataFrame(features, columns=[f"feature_{index}" for index in range(features.shape[1])])
    data["target"] = target

    repository_root = Path(__file__).resolve().parents[2]
    with tempfile.TemporaryDirectory(prefix="mma-ai-ag161-smoke-") as output:
        output_path = Path(output).resolve()
        assert repository_root not in output_path.parents

        predictor = TabularPredictor(
            label="target",
            problem_type="binary",
            eval_metric="log_loss",
            path=str(output_path / "model"),
        )
        predictor.fit(
            train_data=data.iloc[:1000],
            presets="extreme",
            time_limit=300,
            num_gpus=1,
            raise_on_model_failure=True,
        )
        prediction = predictor.predict(data.iloc[[1000]].drop(columns="target"))

        assert len(prediction) == 1
        assert prediction.iloc[0] in (0, 1)
