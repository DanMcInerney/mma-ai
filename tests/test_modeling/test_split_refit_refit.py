from __future__ import annotations

import copy
import hashlib

import pytest

from libs.modeling.split_refit_experiment.__main__ import _parser
from libs.modeling.split_refit_experiment.refit import (
    RefitError,
    _registered_file_sha256,
    assert_single_refit_attempt,
    build_refit_invocation,
    classify_saved_lineage,
    validate_full_population,
    validate_named_profile,
    validate_saved_population,
)


@pytest.mark.parametrize(
    "command",
    [
        "preflight-refit",
        "fit-refit",
        "refit-child",
        "recover-refit-evidence",
        "verify-refit-attempt",
        "verify-refit",
    ],
)
def test_refit_cli_commands_are_explicit_named_entrypoints(command: str):
    arguments = [command, "--campaign", "campaign"]
    if command in {"preflight-refit", "fit-refit", "refit-child"}:
        arguments.extend(["--source-csv", "sealed.csv"])
    if command in {"preflight-refit", "verify-refit-attempt"}:
        arguments.append("--strict")
    if command == "verify-refit":
        arguments.append("--recompute-lineage")
    assert _parser().parse_args(arguments).command == command


def test_registered_identity_canonicalizes_tracked_crlf_json(tmp_path):
    artifact = tmp_path / "selection.json"
    artifact.write_bytes(b'{"a":1}\r\n')
    expected = hashlib.sha256(b'{"a":1}\n').hexdigest().upper()
    assert _registered_file_sha256(artifact) == expected


def test_saved_population_requires_exact_membership_but_records_native_order():
    expected = [str(value) for value in range(3267)]
    actual = expected.copy()
    actual[169], actual[170] = actual[170], actual[169]
    identity = validate_saved_population(actual, expected)
    assert identity["membership_equal"] is True
    assert identity["order_equal"] is False
    assert identity["expected_order_sha256"] != identity["saved_order_sha256"]
    with pytest.raises(RefitError, match="membership"):
        validate_saved_population(actual[:-1] + ["missing"], expected)


def _profile() -> dict[str, object]:
    return {
        "model_type": "win",
        "preset": "hybrid",
        "time_limit": 3000,
        "test_size": None,
        "val_date": None,
        "features": [f"feature-{index}" for index in range(40)],
        "included_strings": None,
        "excluded_strings": None,
        "required_strings": None,
        "start_date": "2014-01-01",
        "num_fights": 2,
        "include_split_dec": True,
        "normalize": "robust",
        "use_recency_weights": True,
        "decay_rate": 0.15,
        "calculate_importance": True,
        "included_model_types": None,
        "split_strategy": "timeseries_split",
        "walkforward_n_windows": 4,
        "walkforward_initial_year": 2021,
        "timeseries_split": None,
        "refit_all": False,
        "refit_full": True,
    }


def test_named_profile_and_invocation_are_exact_and_single_call():
    profile = _profile()
    expected = copy.deepcopy(profile)
    validate_named_profile(profile, expected_profile=expected, expected_feature_count=40)
    invocation = build_refit_invocation(
        source_csv="C:/sealed/training_data.csv",
        model_root="C:/unique/model",
        source_rows=3267,
        profile=profile,
    )
    assert invocation["call"] == 'libs.modeling.training_profiles.train_profile("v8-hybrid-weighted")'
    assert invocation["call_ordinal"] == 1
    assert invocation["source_rows"] == 3267
    assert invocation["profile_delta_from_named"] == {}


@pytest.mark.parametrize("mutation", ["profile", "features", "rows", "path"])
def test_refit_contract_rejects_profile_data_and_destination_drift(mutation: str):
    profile = _profile()
    expected = copy.deepcopy(profile)
    if mutation == "profile":
        profile["decay_rate"] = 0.0
        with pytest.raises(RefitError, match="profile"):
            validate_named_profile(profile, expected_profile=expected, expected_feature_count=40)
    elif mutation == "features":
        profile["features"] = profile["features"][:-1]
        with pytest.raises(RefitError, match="feature"):
            validate_named_profile(profile, expected_profile=expected, expected_feature_count=40)
    elif mutation == "rows":
        with pytest.raises(RefitError, match="3,267"):
            validate_full_population(list(range(3266)))
    else:
        with pytest.raises(RefitError, match="unique"):
            build_refit_invocation(
                source_csv="C:/sealed/training_data.csv",
                model_root="C:/existing/model",
                source_rows=3267,
                profile=profile,
                destination_exists=True,
            )


def test_attempt_grammar_refuses_retry_or_incomplete_marker_pair():
    valid = [
        {"attempt_id": "full-data-refit-attempt-1", "state": "launched"},
        {"attempt_id": "full-data-refit-attempt-1", "state": "exited", "exit_code": 0},
    ]
    assert_single_refit_attempt(valid, require_success=True)
    with pytest.raises(RefitError, match="exactly one"):
        assert_single_refit_attempt(valid + copy.deepcopy(valid), require_success=True)
    with pytest.raises(RefitError, match="marker"):
        assert_single_refit_attempt(valid[:1], require_success=True)


def test_lineage_uses_fit_metadata_and_prediction_identity_not_suffix():
    info = {
        "Mitra": {"model_type": "MitraModel", "num_samples": 2807},
        "RealMLP_r9": {"model_type": "RealMLPModel", "num_samples": 2807},
        "WeightedEnsemble_L2": {
            "model_type": "WeightedEnsembleModel",
            "num_samples": 460,
            "weights": {"Mitra": 1.0},
        },
        "Mitra_FULL": {"model_type": "MitraModel", "num_samples": 3267},
        "RealMLP_r9_FULL": {"model_type": "RealMLPModel", "num_samples": 2807},
        "WeightedEnsemble_L2_FULL": {
            "model_type": "WeightedEnsembleModel",
            "num_samples": 460,
            "weights": {"Mitra_FULL": 1.0},
        },
    }
    prediction_hashes = {
        "Mitra": "A" * 64,
        "RealMLP_r9": "B" * 64,
        "WeightedEnsemble_L2": "A" * 64,
        "Mitra_FULL": "C" * 64,
        "RealMLP_r9_FULL": "B" * 64,
        "WeightedEnsemble_L2_FULL": "C" * 64,
    }
    lineage = classify_saved_lineage(info, prediction_hashes, total_rows=3267)
    assert lineage["Mitra_FULL"]["origin"] == "fresh-full-fit"
    assert lineage["Mitra_FULL"]["fit_rows"] == 3267
    assert lineage["RealMLP_r9_FULL"]["origin"] == "original-clone"
    assert lineage["RealMLP_r9_FULL"]["fit_rows"] == 2807
    assert lineage["WeightedEnsemble_L2_FULL"]["origin"] == "cloned-ensemble-wrapper"
    assert lineage["WeightedEnsemble_L2_FULL"]["effective_fit_rows"] == 3267
    assert all(node["metric_claim"] == "none" for node in lineage.values())


def test_suffix_only_full_claim_and_unreported_context_are_rejected():
    info = {
        "Model": {"model_type": "Base", "num_samples": 2807},
        "Model_FULL": {"model_type": "Base", "num_samples": 2807},
    }
    with pytest.raises(RefitError, match="prediction identity"):
        classify_saved_lineage(info, {"Model": "A" * 64}, total_rows=3267)
    with pytest.raises(RefitError, match="unsupported FULL"):
        classify_saved_lineage(
            info,
            {"Model": "A" * 64, "Model_FULL": "B" * 64},
            total_rows=3267,
        )
