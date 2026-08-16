from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path

import pytest

from libs.modeling.experiment_campaign.fighter_states import (
    FighterStateError,
    build_fighter_state_rows,
    select_state_profile,
    validate_preregistered_profiles,
)
from libs.modeling.experiment_campaign.families.fighter_states import (
    PROFILE_IDS,
    build_preregistered_profile,
)
from libs.modeling.experiment_campaign.hashing import canonical_sha256, file_sha256
from libs.modeling.experiment_campaign.runner import (
    audit_campaign_safety,
    validate_terminal_campaign,
    verify_family_run,
    verify_feature_lineage,
)
from libs.modeling.experiment_campaign.feature_lineage import validate_feature_lineage_rows


def _profile() -> dict:
    return {
        "source_sha256": "A" * 64,
        "cutoff": "2025-12-31",
        "registered_priors": {
            "sparse-beta-1-4": {"alpha": 1.0, "beta": 4.0},
        },
        "normalization": {"fit_scope": "outer-train-only"},
        "profiles": [
            {
                "id": "recent-one-two",
                "feature_names": ["recent_win_rate"],
                "formula_versions": ["recent-last-two-v1"],
            }
        ],
    }


def _fights() -> list[dict]:
    return [
        {
            "fight_id": "1",
            "event_id": "10",
            "event_date": "2024-01-01",
            "fighter1_id": "a",
            "fighter2_id": "b",
            "fighter1_name": "A",
            "fighter2_name": "B",
            "method": "KO/TKO",
            "y_true": 1,
            "sig_str_land_state": 2.0,
            "sig_str_land_state_diff": 0.5,
            "head_land_state": 1.0,
            "head_land_state_diff": 0.2,
        },
        {
            "fight_id": "2",
            "event_id": "20",
            "event_date": "2024-06-01",
            "fighter1_id": "a",
            "fighter2_id": "c",
            "fighter1_name": "A",
            "fighter2_name": "C",
            "method": "Decision - Unanimous",
            "y_true": 0,
            "sig_str_land_state": 2.2,
            "sig_str_land_state_diff": 0.4,
            "head_land_state": 1.1,
            "head_land_state_diff": 0.2,
        },
    ]


def test_profile_menu_is_frozen_before_score_and_bounded_to_eight() -> None:
    profile = _profile()
    validated = validate_preregistered_profiles(profile)
    assert validated["profile_count"] == 1
    too_many = deepcopy(profile)
    too_many["profiles"] = too_many["profiles"] * 9
    for index, item in enumerate(too_many["profiles"]):
        item["id"] = f"profile-{index}"
    with pytest.raises(FighterStateError, match="maximum eight"):
        validate_preregistered_profiles(too_many)
    global_fit = deepcopy(profile)
    global_fit["normalization"]["fit_scope"] = "global"
    with pytest.raises(FighterStateError, match="global-fit"):
        validate_preregistered_profiles(global_fit)


def test_fighter_states_use_only_strictly_prior_events() -> None:
    rows = build_fighter_state_rows(
        _fights(),
        profile=_profile(),
        artifact_sha256="B" * 64,
    )
    target = [
        row
        for row in rows
        if row["fight_id"] == "2"
        and row["fighter_id"] == "a"
        and row["feature_name"] == "recent_win_rate"
    ][0]
    assert target["source_row_ids"] == ["1"]
    assert target["source_event_ids"] == ["10"]
    assert target["source_dates"] == ["2024-01-01"]
    assert target["value"] == 1.0
    assert target["effective_support"] == 1.0
    assert target["uncertainty"] > 0.0


def test_same_event_history_is_not_visible() -> None:
    fights = _fights()
    fights[1]["event_id"] = "10"
    fights[1]["event_date"] = "2024-01-01"
    rows = build_fighter_state_rows(
        fights,
        profile=_profile(),
        artifact_sha256="B" * 64,
    )
    target = [
        row
        for row in rows
        if row["fight_id"] == "2"
        and row["fighter_id"] == "a"
        and row["feature_name"] == "recent_win_rate"
    ][0]
    assert target["source_row_ids"] == []
    assert target["effective_support"] == 0.0


def test_exact_eight_profile_menu_freezes_all_state_formulas() -> None:
    profile = build_preregistered_profile()
    validated = validate_preregistered_profiles(profile)
    assert tuple(validated["profile_ids"]) == PROFILE_IDS
    assert validated["profile_count"] == 8
    assert profile["source_sha256"] == "157649B780965ECC585F18B3030199CDC0F4FE3013958FFA4095FCF665FDB1EA"
    assert profile["construction"]["same_date_policy"] == "exclude-entire-date"
    assert profile["construction"]["decay_18m_days"] == 548
    assert profile["construction"]["decay_half_life_days"] == 180
    assert profile["construction"]["inactivity_cap_days"] == 730
    assert profile["construction"]["age_interaction_center_years"] == 30.0
    assert set(profile["feature_definitions"]) == set(
        profile["profiles"][-1]["feature_names"]
    )
    assert all(len(item["ordered_feature_sha256"]) == 64 for item in profile["profiles"])


def test_full_state_portfolio_has_count_aware_valid_lineage() -> None:
    profile = build_preregistered_profile()
    rows = build_fighter_state_rows(
        _fights(),
        profile=profile,
        artifact_sha256="B" * 64,
    )
    summary = validate_feature_lineage_rows(
        rows,
        registered_prior_ids=set(profile["registered_priors"]),
    )
    assert summary["row_count"] == 2 * len(_fights()) * len(profile["feature_definitions"])
    assert summary["feature_count"] == len(profile["feature_definitions"])
    target = {
        row["feature_name"]: row
        for row in rows
        if row["fight_id"] == "2" and row["fighter_id"] == "a"
    }
    assert target["career_win_rate"]["effective_support"] == 1.0
    assert target["sparse_ko_posterior"]["denominator"] > 1.0
    assert target["sparse_ko_posterior"]["uncertainty"] > 0.0
    assert target["high_count_striking_state"]["source_row_ids"] == ["1"]
    assert target["inactivity_days"]["value"] == 152.0
    assert target["age_win_interaction"]["source_dates"] == ["2024-01-01"]


def test_actual_campaign_has_pre_score_eight_profile_preregistration() -> None:
    campaign = Path("experiments/top10_20260815")
    profile_path = campaign / "profiles/family-06-fighter-states.json"
    preregistration_path = campaign / "runs/family-06-fighter-states/preregistration.json"
    profile = json.loads(profile_path.read_text(encoding="utf-8"))
    preregistration = json.loads(preregistration_path.read_text(encoding="utf-8"))
    assert profile == build_preregistered_profile()
    assert tuple(item["id"] for item in profile["profiles"]) == PROFILE_IDS
    assert preregistration["scoring_state"] == "not-started"
    assert tuple(preregistration["preregistered_profile_ids"]) == PROFILE_IDS
    assert preregistration["profile_sha256"] == canonical_sha256(profile)
    assert preregistration["profile_file_sha256"] == file_sha256(profile_path)
    registry_lines = (campaign / "registry.jsonl").read_text(encoding="utf-8").splitlines()
    family_record = json.loads(registry_lines[-1])
    assert preregistration["registry_prefix_sha256_before"] == family_record[
        "prefix_sha256_before"
    ]
    assert hashlib.sha256((campaign / "registry.jsonl").read_bytes()).hexdigest().upper() != preregistration[
        "registry_prefix_sha256_before"
    ]
    assert preregistration["database_access"] == {"used": False, "sql": None, "urls": []}
    assert preregistration["invocation"] == {
        "gpu_lease_count": 1,
        "retry_count": 0,
        "serialized": True,
    }


def test_profile_selection_is_strictly_inner_and_prior() -> None:
    profile = build_preregistered_profile()
    evidence = [
        {
            "outer_year": 2025,
            "validation_year": year,
            "profile_id": profile_id,
            "role": "inner-chronological",
            "validation_log_loss": 0.6 + index / 100.0,
        }
        for year in (2022, 2023, 2024)
        for index, profile_id in enumerate(PROFILE_IDS)
    ]
    selected = select_state_profile(evidence, profile=profile, outer_year=2025)
    assert selected["selected_profile_id"] == PROFILE_IDS[0]
    assert selected["selection_years"] == [2022, 2023, 2024]
    assert selected["outer_label_selection_count"] == 0
    future = [dict(row) for row in evidence]
    future[0]["validation_year"] = 2025
    with pytest.raises(FighterStateError, match="outer or future"):
        select_state_profile(future, profile=profile, outer_year=2025)


def test_actual_campaign_has_terminal_family_06_result() -> None:
    campaign = Path("experiments/top10_20260815")
    manifest = json.loads(
        (campaign / "runs/family-06-fighter-states/manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["experiment_id"] == "family-06-multiscale-count-aware-state"
    assert manifest["exit_state"] in {"complete", "failed"}
    if manifest["exit_state"] == "complete":
        assert len(manifest["folds"]) == 4
    else:
        assert manifest["terminal_failure"]["attempt_ordinal"] == 1
        assert manifest["terminal_failure"]["stage"] == "pre-construction-symbol-resolution"
        assert manifest["terminal_failure"]["construction_started"] is False
        assert manifest["terminal_failure"]["fit_started"] is False
        assert manifest["terminal_failure"]["retry_performed"] is False
        assert manifest["outer_prediction_identities"] == []
    assert manifest["outer_label_selection_count"] == 0
    assert manifest["gate_access_count"] == 0
    assert manifest["invocation"]["gpu_lease_count"] == 1
    assert manifest["invocation"]["retry_count"] == 0


def test_failed_result_replays_lineage_safety_and_terminal_prefix() -> None:
    campaign = Path("experiments/top10_20260815")
    lineage = verify_feature_lineage(
        campaign,
        "family-06-fighter-states",
        strict=True,
    )
    assert lineage["status"] == "failed-pre-construction"
    assert lineage["lineage_materialized"] is False
    assert lineage["failure_evidence_verified"] is True
    verified = verify_family_run(
        campaign,
        "family-06-fighter-states",
        recompute_all=True,
    )
    assert verified["status"] == "failed"
    assert verified["attempt_count"] == 1
    assert verified["retry_count"] == 0
    safety = audit_campaign_safety(
        campaign,
        through="family-06-fighter-states",
        require_gate_closed=True,
    )
    assert safety["gpu_lease_count"] == 1
    assert safety["retry_count"] == 0
    assert safety["database_access"] == {"used": False, "sql": None, "urls": []}
    terminal = validate_terminal_campaign(
        campaign,
        expect_terminal_through=6,
        require_gate_closed=True,
    )
    assert len(terminal["family_ids"]) == 6
    assert terminal["protected_gate_access_count"] == 0
