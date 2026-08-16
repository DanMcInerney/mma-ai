from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path
import shutil

import pytest

from libs.modeling.experiment_campaign.families.catboost_specialist import (
    CATBOOST_PARAMETER_KEYS,
    PROFILE_IDS,
    CatBoostSpecialistError,
    build_preregistered_profile,
    materialize_family_08,
    validate_fold_fit_evidence,
    validate_preregistered_profile,
    write_preregistration,
)
from libs.modeling.experiment_campaign.hashing import (
    canonical_sha256,
    file_sha256,
    write_canonical_json,
)
from libs.modeling.experiment_campaign.protocol import initialize_gate
from libs.modeling.experiment_campaign.runner import (
    audit_campaign_safety,
    replay_campaign_decisions,
    validate_terminal_campaign,
    verify_family_run,
)


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


def test_actual_campaign_has_prelaunch_eight_profile_preregistration() -> None:
    campaign = Path("experiments/top10_20260815")
    profile_path = campaign / "profiles/family-08-catboost-specialist.json"
    preregistration_path = campaign / "runs/family-08-catboost-specialist/preregistration.json"
    profile = json.loads(profile_path.read_text(encoding="utf-8"))
    preregistration = json.loads(preregistration_path.read_text(encoding="utf-8"))
    validated = validate_preregistered_profile(profile)
    assert profile == build_preregistered_profile()
    assert tuple(validated["profile_ids"]) == PROFILE_IDS
    assert preregistration["scoring_state"] == "not-started"
    assert preregistration["profile_sha256"] == canonical_sha256(profile)
    assert preregistration["profile_file_sha256"] == file_sha256(profile_path)
    assert preregistration["ordered_profile_hashes"] == validated["profile_hashes"]
    assert preregistration["representation_hashes"] == validated["representation_hashes"]


def test_one_shot_materializer_records_a_recomputable_predecode_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("DATABASE_URL", raising=False)
    monkeypatch.delenv("ODDS_DATABASE_URL", raising=False)
    campaign = tmp_path / "experiments/top10_20260815"
    shutil.copytree(
        Path("experiments/top10_20260815"),
        campaign,
        ignore=shutil.ignore_patterns("artifacts"),
    )
    run_root = campaign / "runs/family-08-catboost-specialist"
    for name in ("attempts.jsonl", "decision.md", "manifest.json"):
        (run_root / name).unlink(missing_ok=True)
    registry_path = campaign / "registry.jsonl"
    lines = registry_path.read_bytes().splitlines(keepends=True)
    assert json.loads(lines[-1])["payload"]["experiment_id"] == (
        "family-08-catboost-native-specialist"
    )
    prefix = b"".join(lines[:-1])
    previous = json.loads(lines[-2])
    registry_path.write_bytes(prefix)
    write_canonical_json(
        campaign / "registry-head.json",
        {
            "last_record_sha256": previous["record_sha256"],
            "record_count": len(lines) - 1,
            "registry_bytes": len(prefix),
            "registry_prefix_sha256": hashlib.sha256(prefix).hexdigest().upper(),
        },
    )
    result = materialize_family_08(
        campaign,
        source_revision="committed-scorer",
        preregistration_commit="committed-preregistration",
    )
    assert result["status"] == "failed"
    assert result["terminal_failure"]["row_decode_started"] is False
    assert result["terminal_failure"]["target_decode_started"] is False
    assert result["terminal_failure"]["fit_started"] is False
    assert result["outer_prediction_identities"] == []
    verified = verify_family_run(campaign, "family-08-catboost-specialist", recompute_all=True)
    assert verified["status"] == "failed"
    assert verified["profile_count"] == 8
    assert verified["attempt_count"] == 1
    assert verified["retry_count"] == 0


def test_actual_campaign_records_one_terminal_dependency_failure_before_decode() -> None:
    campaign = Path("experiments/top10_20260815")
    manifest = json.loads(
        (campaign / "runs/family-08-catboost-specialist/manifest.json").read_text(
            encoding="utf-8"
        )
    )
    failure = manifest["terminal_failure"]
    assert manifest["experiment_id"] == "family-08-catboost-native-specialist"
    assert manifest["exit_state"] == "failed"
    assert failure["stage"] == "pre-construction-dependency-resolution"
    assert failure["row_decode_started"] is False
    assert failure["target_decode_started"] is False
    assert failure["fit_started"] is False
    assert failure["outer_labels_scored"] is False
    assert failure["retry_performed"] is False
    assert manifest["outer_prediction_identities"] == []
    assert manifest["metrics"] is None
    assert manifest["paired_event_block_intervals"] is None
    assert manifest["representation_comparison"]["status"] == "unavailable"
    assert manifest["development_safe_population"] == {
        "asserted_before_row_or_target_decode": True,
        "development_safe_id_count": 3_089,
        "development_max_date": "2025-12-13",
        "retired_id_count": 178,
    }


def test_family_08_terminal_result_recomputes_run_replay_safety_and_prefix() -> None:
    campaign = Path("experiments/top10_20260815")
    verified = verify_family_run(campaign, "family-08-catboost-specialist", recompute_all=True)
    assert verified["status"] == "failed"
    assert verified["profile_count"] == 8
    assert verified["attempt_count"] == 1
    assert verified["retry_count"] == 0
    assert verified["outer_prediction_identities"] == []
    assert verified["representation_comparison"]["status"] == "unavailable"
    replayed = replay_campaign_decisions(campaign, through="family-08-catboost-specialist")
    assert len(replayed["decisions"]) == 8
    assert replayed["decisions"][-1]["status"] == "failed"
    assert replayed["incumbent_after"] == "family-01-weighted-v8-control"
    safety = audit_campaign_safety(
        campaign,
        through="family-08-catboost-specialist",
        require_gate_closed=True,
    )
    assert safety["gpu_lease_count"] == 1
    assert safety["production_attempt_count"] == 1
    assert safety["retry_count"] == 0
    assert safety["database_access"] == {"used": False, "sql": None, "urls": []}
    terminal = validate_terminal_campaign(
        campaign,
        expect_terminal_through=8,
        require_gate_closed=True,
    )
    assert len(terminal["family_ids"]) == 8
    assert terminal["protected_gate_access_count"] == 0
