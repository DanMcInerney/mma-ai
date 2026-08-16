from __future__ import annotations

from copy import deepcopy

import pytest

from libs.modeling.experiment_campaign.fighter_states import (
    FighterStateError,
    build_fighter_state_rows,
    validate_preregistered_profiles,
)


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
