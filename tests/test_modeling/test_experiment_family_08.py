from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import pytest

from libs.modeling.experiment_campaign.families.catboost_specialist import (
    CATBOOST_PARAMETER_KEYS,
    PROFILE_IDS,
    CatBoostSpecialistError,
    build_preregistered_profile,
    validate_fold_fit_evidence,
    validate_preregistered_profile,
    write_preregistration,
)
from libs.modeling.experiment_campaign.hashing import canonical_sha256, file_sha256
from libs.modeling.experiment_campaign.protocol import initialize_gate


def test_exact_eight_materialized_catboost_profiles_compare_shared_raw_and_native() -> None:
    profile = build_preregistered_profile()
    validated = validate_preregistered_profile(profile)
    profiles = profile["profiles"]

    assert tuple(item["id"] for item in profiles) == PROFILE_IDS
    assert validated["profile_count"] == 8
    assert [item["representation_id"] for item in profiles].count("shared-normalized") == 2
    assert [item["representation_id"] for item in profiles].count("raw-count-exposure") == 2
    assert [item["representation_id"] for item in profiles].count("native-categorical") == 4
    assert {item["capacity"] for item in profiles} == {"low", "medium"}
    assert all(set(item["catboost_parameters"]) == set(CATBOOST_PARAMETER_KEYS) for item in profiles)
    assert all(item["catboost_parameters"]["loss_function"] == "Logloss" for item in profiles)
    assert all(item["catboost_parameters"]["eval_metric"] == "Logloss" for item in profiles)
    assert all(item["catboost_parameters"]["task_type"] == "GPU" for item in profiles)
    assert all(item["catboost_parameters"]["random_seed"] == 20260815 for item in profiles)
    assert all(item["refit_full"] is False for item in profiles)
    assert all(item["profile_sha256"] == canonical_sha256({
        key: value for key, value in item.items() if key != "profile_sha256"
    }) for item in profiles)
    native = [item for item in profiles if item["representation_id"] == "native-categorical"]
    assert all(item["catboost_parameters"]["boosting_type"] == "Ordered" for item in native)
    assert all(item["categorical_feature_names"] for item in native)


def test_shared_control_is_the_exact_frozen_normalized_input() -> None:
    shared = json.loads(
        Path("experiments/top10_20260815/profiles/family-01-weighted-v8-control.json").read_text(
            encoding="utf-8"
        )
    )
    representation = build_preregistered_profile()["representations"]["shared-normalized"]
    assert representation["numeric_feature_names"] == shared["features"]
    assert representation["normalization"] == {
        "fit_scope": "outer-train-only",
        "method": "robust",
        "missing_numeric": "outer-training-median",
    }
    assert representation["categorical_feature_names"] == []


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda p: p["profiles"].append(deepcopy(p["profiles"][-1])), "maximum eight"),
        (
            lambda p: p["profiles"][0]["catboost_parameters"].pop("random_strength"),
            "materialized CatBoost parameters",
        ),
        (
            lambda p: p["profiles"][4]["catboost_parameters"].__setitem__(
                "boosting_type", "Plain"
            ),
            "ordered boosting",
        ),
        (
            lambda p: p["profiles"][0].__setitem__("refit_full", True),
            "refit_full",
        ),
        (
            lambda p: p["representations"]["shared-normalized"]["normalization"].__setitem__(
                "fit_scope", "global"
            ),
            "global preprocessing",
        ),
        (
            lambda p: p["representations"]["raw-count-exposure"]["rate_features"][0].pop(
                "exposure"
            ),
            "missing exposure",
        ),
        (
            lambda p: p["representations"]["native-categorical"][
                "categorical_feature_names"
            ].append("fighter1_id"),
            "identity, target, post-fight, future, or odds",
        ),
        (
            lambda p: p["representations"]["native-categorical"][
                "categorical_feature_names"
            ].append("method"),
            "identity, target, post-fight, future, or odds",
        ),
        (
            lambda p: p["representations"]["raw-count-exposure"][
                "numeric_feature_names"
            ].append("closing_odds_diff"),
            "identity, target, post-fight, future, or odds",
        ),
    ],
)
def test_profile_rejects_leaky_or_unmaterialized_specialists(mutate, message: str) -> None:
    profile = build_preregistered_profile()
    mutate(profile)
    with pytest.raises(CatBoostSpecialistError, match=message):
        validate_preregistered_profile(profile)


def test_fold_fit_evidence_requires_identical_ids_and_fold_local_statistics() -> None:
    profile = build_preregistered_profile()
    fold_ids = ["outer-2022", "outer-2023", "outer-2024", "outer-2025"]
    evidence = [
        {
            "profile_id": profile_id,
            "fold_ids": fold_ids,
            "numeric_fit_scope": "fold-training-only",
            "categorical_statistics_scope": "fold-training-only",
            "missing_value_scope": "fold-training-only",
            "outer_label_selection_count": 0,
        }
        for profile_id in PROFILE_IDS
    ]
    result = validate_fold_fit_evidence(evidence, profile=profile)
    assert result == {"fold_ids": fold_ids, "profile_count": 8}

    wrong = deepcopy(evidence)
    wrong[-1]["fold_ids"] = fold_ids[:-1]
    with pytest.raises(CatBoostSpecialistError, match="identical fold IDs"):
        validate_fold_fit_evidence(wrong, profile=profile)
    global_stats = deepcopy(evidence)
    global_stats[4]["categorical_statistics_scope"] = "global"
    with pytest.raises(CatBoostSpecialistError, match="fold-local"):
        validate_fold_fit_evidence(global_stats, profile=profile)


def test_preregistration_freezes_the_exact_menu_before_any_launch(tmp_path: Path) -> None:
    campaign = tmp_path / "experiments" / "top10_20260815"
    campaign.mkdir(parents=True)
    initialize_gate(campaign, expected_family_ids=())
    (campaign / "registry.jsonl").write_bytes(b"fixed-prefix\n")

    preregistration = write_preregistration(campaign, source_revision="before-score")
    profile_path = campaign / "profiles/family-08-catboost-specialist.json"
    preregistration_path = campaign / "runs/family-08-catboost-specialist/preregistration.json"
    profile = build_preregistered_profile()
    assert profile_path.is_file() and preregistration_path.is_file()
    assert preregistration["scoring_state"] == "not-started"
    assert preregistration["preregistered_profile_ids"] == list(PROFILE_IDS)
    assert preregistration["profile_sha256"] == canonical_sha256(profile)
    assert preregistration["profile_file_sha256"] == file_sha256(profile_path)
    assert preregistration["invocation"] == {
        "gpu_lease_count": 1,
        "retry_count": 0,
        "serialized": True,
    }
    assert preregistration["database_access"] == {"used": False, "sql": None, "urls": []}
    with pytest.raises(ValueError, match="destinations must all be absent"):
        write_preregistration(campaign, source_revision="retry")
