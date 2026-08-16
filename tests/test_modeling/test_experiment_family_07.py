from __future__ import annotations

from copy import deepcopy

import pytest

from libs.modeling.experiment_campaign.families.matchup_geometry import (
    PROFILE_IDS,
    build_preregistered_profile,
)
from libs.modeling.experiment_campaign.matchup_geometry import (
    MatchupGeometryError,
    build_directional_interactions,
    select_matchup_profile,
    swap_matchup_row,
    validate_prediction_geometry,
    validate_preregistered_matchup_profiles,
)


def _row() -> dict:
    return {
        "fight_id": "fight-1",
        "event_id": "event-1",
        "event_date": "2025-05-03",
        "fighter1_id": "a",
        "fighter2_id": "b",
        "fighter1_name": "Alpha",
        "fighter2_name": "Beta",
        "fighter1_url": "https://example.test/a",
        "fighter2_url": "https://example.test/b",
        "fighter1_label": 1,
        "fighter2_label": 0,
        "fighter1_odds": -130.0,
        "fighter2_odds": 110.0,
        "y_true": 1,
        "fighter1_features": {"offense": 0.8, "defense": 0.3},
        "fighter2_features": {"offense": 0.4, "defense": 0.7},
        "fighter1_lineage": {"offense": ["prior-a"]},
        "fighter2_lineage": {"offense": ["prior-b"]},
        "offense_minus_defense": 0.1,
        "fighter1_market_probability": 0.55,
        "fighter2_market_probability": 0.45,
    }


def test_complete_role_swap_reverses_every_registered_orientation() -> None:
    original = _row()
    swapped = swap_matchup_row(original)
    assert swapped["fight_id"] == original["fight_id"]
    for suffix in ("id", "name", "url", "label", "odds", "features", "lineage", "market_probability"):
        assert swapped[f"fighter1_{suffix}"] == original[f"fighter2_{suffix}"]
        assert swapped[f"fighter2_{suffix}"] == original[f"fighter1_{suffix}"]
    assert swapped["y_true"] == 0
    assert swapped["offense_minus_defense"] == pytest.approx(-0.1)
    assert swap_matchup_row(swapped) == original


@pytest.mark.parametrize(
    "missing",
    [
        "fighter2_id",
        "fighter1_name",
        "fighter2_url",
        "fighter1_label",
        "fighter2_odds",
        "fighter1_features",
        "fighter2_lineage",
        "y_true",
    ],
)
def test_incomplete_role_swap_is_rejected(missing: str) -> None:
    row = _row()
    row.pop(missing)
    with pytest.raises(MatchupGeometryError, match="complete role swap"):
        swap_matchup_row(row)


def test_unpaired_source_or_lineage_keys_are_rejected() -> None:
    row = _row()
    row["fighter2_features"].pop("defense")
    with pytest.raises(MatchupGeometryError, match="paired feature keys"):
        swap_matchup_row(row)
    row = _row()
    row["fighter2_lineage"]["grappling"] = ["prior-b"]
    with pytest.raises(MatchupGeometryError, match="paired lineage keys"):
        swap_matchup_row(row)


def test_registered_directional_interactions_are_antisymmetric_and_supported() -> None:
    declarations = [
        {
            "name": "striking_attack_vs_defense",
            "left": "striking_attack",
            "right": "striking_defense",
            "formula": "cross-difference",
            "minimum_support": 2.0,
            "swap_rule": "negate",
        }
    ]
    first = {
        "striking_attack": 0.8,
        "striking_defense": 0.3,
        "effective_support": 4.0,
    }
    second = {
        "striking_attack": 0.4,
        "striking_defense": 0.7,
        "effective_support": 3.0,
    }
    original = build_directional_interactions(first, second, declarations)
    swapped = build_directional_interactions(second, first, declarations)
    assert original["striking_attack_vs_defense"] == pytest.approx(0.8)
    assert swapped["striking_attack_vs_defense"] == pytest.approx(-0.8)
    sparse = dict(second, effective_support=1.0)
    with pytest.raises(MatchupGeometryError, match="support gate"):
        build_directional_interactions(first, sparse, declarations)
    wrong = deepcopy(declarations)
    wrong[0]["swap_rule"] = "identity"
    with pytest.raises(MatchupGeometryError, match="antisymmetric"):
        build_directional_interactions(first, second, wrong)


def test_prediction_geometry_requires_complementarity_and_no_identity_leakage() -> None:
    rows = [
        {
            "fight_id": "fight-1",
            "original_prediction_id": "fight-1:original",
            "swapped_prediction_id": "fight-1:swapped",
            "original_probability": 0.72,
            "swapped_probability": 0.28,
            "averaged_probability": 0.72,
            "invariance_residual": 0.0,
        }
    ]
    result = validate_prediction_geometry(rows, tolerance=1e-12)
    assert result["maximum_invariance_residual"] == 0.0
    wrong = deepcopy(rows)
    wrong[0]["swapped_probability"] = 0.31
    with pytest.raises(MatchupGeometryError, match="complementary"):
        validate_prediction_geometry(wrong, tolerance=1e-12)
    leaked = deepcopy(rows)
    leaked[0]["model_feature_names"] = ["fighter1_id", "fighter2_odds"]
    with pytest.raises(MatchupGeometryError, match="identity, URL, label, or odds leakage"):
        validate_prediction_geometry(leaked, tolerance=1e-12)


def test_exact_eight_profiles_freeze_pairs_support_geometry_and_inner_selection() -> None:
    profile = build_preregistered_profile()
    validated = validate_preregistered_matchup_profiles(profile)
    assert tuple(validated["profile_ids"]) == PROFILE_IDS
    assert validated["profile_count"] == 8
    assert profile["normalization"]["fit_scope"] == "outer-train-only"
    assert profile["selection"]["fit_scope"] == "prior-inner-only"
    assert profile["selection"]["outer_label_selection_count"] == 0
    assert profile["geometry"]["probability_sum"] == 1.0
    assert profile["geometry"]["tolerance"] == 1e-12
    assert all(item["ordered_interaction_sha256"] for item in profile["profiles"])

    evidence = [
        {
            "outer_year": 2025,
            "validation_year": year,
            "profile_id": profile_id,
            "role": "inner-chronological",
            "validation_log_loss": 0.60 + index / 100.0,
        }
        for year in (2022, 2023, 2024)
        for index, profile_id in enumerate(PROFILE_IDS)
    ]
    selected = select_matchup_profile(evidence, profile=profile, outer_year=2025)
    assert selected["selected_profile_id"] == PROFILE_IDS[0]
    assert selected["selection_years"] == [2022, 2023, 2024]
    future = deepcopy(evidence)
    future[0]["validation_year"] = 2025
    with pytest.raises(MatchupGeometryError, match="outer or future"):
        select_matchup_profile(future, profile=profile, outer_year=2025)


def test_global_fit_normalization_and_more_than_eight_profiles_are_rejected() -> None:
    profile = build_preregistered_profile()
    profile["normalization"]["fit_scope"] = "global"
    with pytest.raises(MatchupGeometryError, match="global-fit"):
        validate_preregistered_matchup_profiles(profile)
    profile = build_preregistered_profile()
    profile["profiles"].append(deepcopy(profile["profiles"][-1]))
    profile["profiles"][-1]["id"] = "ninth-profile"
    with pytest.raises(MatchupGeometryError, match="maximum eight"):
        validate_preregistered_matchup_profiles(profile)
