from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from libs.modeling.split_refit_experiment.runner import (
    EXPECTED_BASE_MODELS,
    EvaluationError,
    assert_production_grammar,
    compute_recency_weights,
    freeze_selection,
    predict_without_labels,
    resolve_candidate_names,
)


class FakePredictor:
    model_best = "WeightedEnsemble_L2"
    class_labels = [0, 1]

    def model_names(self):
        return [*EXPECTED_BASE_MODELS, self.model_best]

    def info(self):
        return {
            "model_info": {
                name: {
                    "name": name,
                    "model_type": "Fake",
                    "stack_level": 1 if name == self.model_best else 0,
                    "children_info": (
                        {"S1F1": {"model_weights": {"Mitra": 0.75, "TabICL": 0.25}}}
                        if name == self.model_best
                        else {}
                    ),
                }
                for name in self.model_names()
            }
        }

    def predict_proba(self, frame: pd.DataFrame):
        probability = np.asarray(frame["feature"], dtype=float)
        return pd.DataFrame({0: 1.0 - probability, 1: probability})


def test_expected_portfolio_and_sample_weights_are_exact():
    assert EXPECTED_BASE_MODELS == (
        "RandomForestGini",
        "CatBoost",
        "TabICL",
        "ExtraTreesGini",
        "Mitra",
        "NeuralNetFastAI",
        "XGBoost",
        "PrepLightGBM",
        "LightGBM_r8",
        "RealMLP_r9",
    )
    assert resolve_candidate_names(FakePredictor()) == EXPECTED_BASE_MODELS

    dates = pd.Series(pd.to_datetime(["2020-01-01", "2021-01-01", "2022-01-01"]))
    weights, provenance = compute_recency_weights(dates, decay_rate=0.15)
    assert weights[-1] > weights[0]
    assert sum(weights) == pytest.approx(3.0)
    assert provenance == {
        "kind": "exponential-recency",
        "annual_decay": 0.15,
        "normalization": "mean-one",
        "reference_date": "2022-01-01",
        "row_count": 3,
        "sum": pytest.approx(3.0),
        "minimum": pytest.approx(min(weights)),
        "maximum": pytest.approx(max(weights)),
    }


def test_selection_freezes_graph_weights_and_rejects_full_or_context_nodes(tmp_path: Path):
    model_root = tmp_path / "model"
    model_root.mkdir()
    (model_root / "predictor.pkl").write_bytes(b"fixed model")
    (model_root / "scaler.pkl").write_bytes(b"fixed scaler")
    record = freeze_selection(
        FakePredictor(),
        model_root=model_root,
        scaler_path=model_root / "scaler.pkl",
        fixed_identities={"source_sha256": "A" * 64},
    )
    assert record["selected_node"] == "WeightedEnsemble_L2"
    assert record["ensemble_weights"] == {"Mitra": 0.75, "TabICL": 0.25}
    assert record["classes"] == [0, 1]
    assert record["base_models"] == list(EXPECTED_BASE_MODELS)
    assert record["model_tree"]["file_count"] == 2
    assert len(record["model_tree"]["sha256"]) == 64

    bad = FakePredictor()
    bad.model_best = "Mitra_FULL"
    with pytest.raises(EvaluationError, match="FULL/context"):
        freeze_selection(
            bad,
            model_root=model_root,
            scaler_path=model_root / "scaler.pkl",
            fixed_identities={},
        )


def test_predictions_are_label_invariant_and_keep_manifest_order():
    predictor = FakePredictor()
    rows = [
        {"fight_id": "f2", "event_id": "e1", "event_date": "2025-01-01"},
        {"fight_id": "f1", "event_id": "e1", "event_date": "2025-01-01"},
    ]
    features = pd.DataFrame(
        {"fight_id": ["f1", "f2"], "feature": [0.2, 0.8]}
    ).set_index("fight_id")
    first = predict_without_labels(predictor, rows, features)
    features_with_hostile_labels = features.assign(y_true=[1, 0], future_label=[0, 1])
    second = predict_without_labels(predictor, rows, features_with_hostile_labels)
    assert first == second
    assert [row["fight_id"] for row in first] == ["f2", "f1"]
    assert [row["probability"] for row in first] == [0.8, 0.2]
    assert all("y_true" not in row for row in first)


def test_production_grammar_requires_one_fit_pair_then_one_access():
    attempts = [
        {"attempt_id": "evaluation-fit-attempt-1", "state": "launched"},
        {"attempt_id": "evaluation-fit-attempt-1", "state": "exited", "exit_code": 0},
    ]
    access = [{"access_id": "retrospective-test-access-1", "state": "opened"}]
    assert_production_grammar(attempts, access, selection_frozen=True, require_access=True)
    with pytest.raises(EvaluationError, match="exactly one production fit"):
        assert_production_grammar(attempts * 2, access, selection_frozen=True, require_access=True)
    with pytest.raises(EvaluationError, match="before selection"):
        assert_production_grammar(attempts, access, selection_frozen=False, require_access=True)
    with pytest.raises(EvaluationError, match="exactly one test access"):
        assert_production_grammar(attempts, access * 2, selection_frozen=True, require_access=True)

