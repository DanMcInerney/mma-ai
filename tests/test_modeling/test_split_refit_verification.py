from __future__ import annotations

import copy

import pytest

from libs.modeling.experiment_campaign.metrics import reduce_predictions
from libs.modeling.split_refit_experiment.verification import (
    EvaluationVerificationError,
    RefitVerificationError,
    has_database_token,
    validate_evaluation_documents,
    validate_refit_documents,
    validate_loaded_predictor,
)


def _documents():
    predictions = [
        {
            "fight_id": "1",
            "event_id": "10",
            "event_date": "2025-01-01",
            "probability": 0.8,
            "y_true": 1,
            "boundary": "Original",
            "fit_scope": "prior-only",
            "fold": "retrospective-test",
        },
        {
            "fight_id": "2",
            "event_id": "10",
            "event_date": "2025-01-01",
            "probability": 0.3,
            "y_true": 0,
            "boundary": "Original",
            "fit_scope": "prior-only",
            "fold": "retrospective-test",
        },
    ]
    selection = {
        "selected_node": "WeightedEnsemble_L2",
        "base_models": ["Mitra", "TabICL"],
        "model_tree": {"sha256": "A" * 64},
        "source_revision": "B" * 40,
        "profile_sha256": "C" * 64,
        "data_sha256": "D" * 64,
        "partition_sha256": "E" * 64,
        "scaler_sha256": "F" * 64,
        "classes": [0, 1],
        "ensemble_weights": {"Mitra": 0.75, "TabICL": 0.25},
    }
    attempts = [
        {"attempt_id": "evaluation-fit-attempt-1", "state": "launched"},
        {"attempt_id": "evaluation-fit-attempt-1", "state": "exited", "exit_code": 0},
    ]
    access = [
        {
            "access_id": "retrospective-test-access-1",
            "state": "opened",
            "selection_sha256": "1" * 64,
            "selection_commit": "2" * 40,
            "row_count": 2,
            "fight_ids": ["1", "2"],
        }
    ]
    metrics = reduce_predictions(predictions).as_dict()
    result = {
        "row_count": 2,
        "prediction_sha256": "ignored-by-pure-validator",
        "metrics": metrics,
        "historical_boundaries": {
            "retrospective_test": 2,
            "accepted_tuning": 460,
            "nested_outer": 1108,
            "pooled": False,
        },
        "post_test_adaptation": False,
    }
    return selection, attempts, access, predictions, result


def test_document_verifier_recomputes_metrics_and_enforces_boundaries():
    selection, attempts, access, predictions, result = _documents()
    verified = validate_evaluation_documents(
        selection=selection,
        attempts=attempts,
        access=access,
        predictions=predictions,
        result=result,
        expected_count=2,
    )
    assert verified["metrics"]["correct_count"] == 2
    assert verified["metrics"]["log_loss"] > 0


@pytest.mark.parametrize("mutation", ["probability", "label", "access-order", "retry", "denominator", "full-node"])
def test_document_verifier_rejects_hostile_mutations(mutation: str):
    selection, attempts, access, predictions, result = _documents()
    if mutation == "probability":
        predictions[0]["probability"] = 0.1
    elif mutation == "label":
        predictions[0]["y_true"] = 0
    elif mutation == "access-order":
        access[0]["fight_ids"].reverse()
    elif mutation == "retry":
        attempts.extend(copy.deepcopy(attempts))
    elif mutation == "denominator":
        result["historical_boundaries"]["pooled"] = True
    else:
        selection["selected_node"] = "Mitra_FULL"
    with pytest.raises(EvaluationVerificationError):
        validate_evaluation_documents(
            selection=selection,
            attempts=attempts,
            access=access,
            predictions=predictions,
            result=result,
            expected_count=2,
        )


def test_database_audit_distinguishes_metrics_from_connection_tokens():
    assert not has_database_token('{"metric":0.6261685244094812}')
    assert has_database_token("postgresql://user@localhost:5432/clankerfights")
    assert has_database_token("host=localhost port=5432 dbname=clankerfights")


def test_predictor_load_smoke_requires_frozen_graph_and_selected_node():
    class Predictor:
        model_best = "WeightedEnsemble_L2"

        def model_names(self):
            return [
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
                "WeightedEnsemble_L2",
            ]

    selection = {
        "selected_node": "WeightedEnsemble_L2",
        "base_models": Predictor().model_names()[:-1],
    }
    validate_loaded_predictor(Predictor(), selection)
    Predictor.model_best = "Mitra"
    with pytest.raises(EvaluationVerificationError, match="selected node"):
        validate_loaded_predictor(Predictor(), selection)


def _refit_documents():
    attempts = [
        {"attempt_id": "full-data-refit-attempt-1", "state": "launched"},
        {
            "attempt_id": "full-data-refit-attempt-1",
            "state": "exited",
            "exit_code": 0,
        },
    ]
    lineage = {
        "Mitra": {
            "boundary": "Original",
            "origin": "internal-selection-fit",
            "fit_rows": 2807,
            "metric_claim": "none",
            "context_contaminated": True,
        },
        "Mitra_FULL": {
            "boundary": "FULL",
            "origin": "fresh-full-fit",
            "fit_rows": 3267,
            "metric_claim": "none",
            "context_contaminated": True,
        },
        "RealMLP_r9_FULL": {
            "boundary": "FULL",
            "origin": "original-clone",
            "fit_rows": 2807,
            "metric_claim": "none",
            "context_contaminated": True,
        },
    }
    result = {
        "state": "complete",
        "profile_name": "v8-hybrid-weighted",
        "source_rows": 3267,
        "feature_count": 40,
        "fit_invocation_count": 1,
        "validation_claims": [],
        "database_access": False,
        "lineage": lineage,
    }
    return attempts, result


def test_refit_documents_admit_full_data_without_validation_claim():
    attempts, result = _refit_documents()
    verified = validate_refit_documents(attempts=attempts, result=result)
    assert verified["source_rows"] == 3267
    assert verified["validation_claims"] == []


@pytest.mark.parametrize(
    "mutation", ["rows", "retry", "validation", "clone", "context", "database"]
)
def test_refit_documents_reject_wrong_boundary_claims(mutation: str):
    attempts, result = _refit_documents()
    if mutation == "rows":
        result["source_rows"] = 3266
    elif mutation == "retry":
        attempts += copy.deepcopy(attempts)
    elif mutation == "validation":
        result["validation_claims"] = [{"node": "Mitra_FULL", "kind": "validation"}]
    elif mutation == "clone":
        result["lineage"]["RealMLP_r9_FULL"]["fit_rows"] = 3267
    elif mutation == "context":
        result["lineage"]["Mitra_FULL"]["context_contaminated"] = False
    else:
        result["database_access"] = True
    with pytest.raises(RefitVerificationError):
        validate_refit_documents(attempts=attempts, result=result)
