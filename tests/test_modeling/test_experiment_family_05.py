from __future__ import annotations

import hashlib

import pytest

from libs.modeling.experiment_campaign.semantic_portfolio import (
    MEASUREMENT_GROUP_IDS,
    SemanticPortfolioError,
    select_stable_features,
    validate_preregistered_profile,
)


SOURCE_SHA256 = "A" * 64
HEADER = ("feature_a", "feature_b", "event_date", "target")
HEADER_SHA256 = hashlib.sha256("\n".join(HEADER).encode()).hexdigest().upper()


def _profile() -> dict:
    candidates = [
        {
            "name": "feature_a",
            "semantic_id": "pace-a",
            "measurement_group": "global-striking-pace-efficiency",
            "source_file_sha256": SOURCE_SHA256,
            "source_header_sha256": HEADER_SHA256,
            "formula": "identity(feature_a)",
            "available_by": "2021-12-31",
            "domain_redundancy_rank": 1,
        },
        {
            "name": "feature_b",
            "semantic_id": "pace-b",
            "measurement_group": "global-striking-pace-efficiency",
            "source_file_sha256": SOURCE_SHA256,
            "source_header_sha256": HEADER_SHA256,
            "formula": "identity(feature_b)",
            "available_by": "2021-12-31",
            "domain_redundancy_rank": 2,
        },
    ]
    ordered_hash = hashlib.sha256(
        b'["feature_a","feature_b"]'
    ).hexdigest().upper()
    return {
        "experiment_id": "family-05-stable-semantic-portfolio",
        "frozen_source": {
            "path": "frozen/training_data.csv",
            "sha256": SOURCE_SHA256,
            "ordered_header_sha256": HEADER_SHA256,
            "cutoff": "2025-12-31",
        },
        "measurement_group_ids": list(MEASUREMENT_GROUP_IDS),
        "candidate_features": candidates,
        "measurement_profiles": [
            {
                "id": "v8-control",
                "included_groups": list(MEASUREMENT_GROUP_IDS),
                "ordered_features": ["feature_a", "feature_b"],
                "ordered_feature_sha256": ordered_hash,
            }
        ],
        "selection": {
            "evidence_role": "inner-chronological",
            "stability_threshold": 0.75,
            "drop_column_min_improvement": 0.0,
            "minimum_fold_support": 2,
            "domain_redundancy_cap": 1,
            "tie_break": "profile-order-then-feature-order",
            "combined_row_importance_role": "non-selection",
        },
    }


def test_preregistration_requires_exact_lineage_groups_and_maximum_eight_profiles() -> None:
    profile = _profile()
    validated = validate_preregistered_profile(profile, source_header=HEADER)
    assert validated["candidate_count"] == 2
    assert validated["profile_count"] == 1

    duplicate_semantic = _profile()
    duplicate_semantic["candidate_features"][1]["semantic_id"] = "pace-a"
    with pytest.raises(SemanticPortfolioError, match="duplicate semantic"):
        validate_preregistered_profile(duplicate_semantic, source_header=HEADER)

    missing_lineage = _profile()
    missing_lineage["candidate_features"][0]["name"] = "not_in_header"
    with pytest.raises(SemanticPortfolioError, match="source header"):
        validate_preregistered_profile(missing_lineage, source_header=HEADER)

    too_many = _profile()
    too_many["measurement_profiles"] *= 9
    with pytest.raises(SemanticPortfolioError, match="maximum eight"):
        validate_preregistered_profile(too_many, source_header=HEADER)


def test_stability_selection_is_inner_only_directional_and_redundancy_capped() -> None:
    profile = _profile()
    evidence = [
        {"feature": "feature_a", "fold": 2019, "direction": 1, "drop_column_delta": 0.02, "role": "inner-chronological"},
        {"feature": "feature_a", "fold": 2020, "direction": 1, "drop_column_delta": 0.03, "role": "inner-chronological"},
        {"feature": "feature_b", "fold": 2019, "direction": 1, "drop_column_delta": 0.01, "role": "inner-chronological"},
        {"feature": "feature_b", "fold": 2020, "direction": 1, "drop_column_delta": 0.01, "role": "inner-chronological"},
    ]
    selected = select_stable_features(evidence, profile=profile, outer_year=2022)
    assert selected["selected_features"] == ["feature_a"]
    assert selected["fit_folds"] == [2019, 2020]
    assert selected["outer_label_selection_count"] == 0
    assert selected["combined_row_importance_used"] is False

    future = evidence + [
        {"feature": "feature_a", "fold": 2022, "direction": 1, "drop_column_delta": 1.0, "role": "outer"}
    ]
    with pytest.raises(SemanticPortfolioError, match="outer or future"):
        select_stable_features(future, profile=profile, outer_year=2022)

    unstable = [dict(row) for row in evidence]
    unstable[1]["direction"] = -1
    result = select_stable_features(unstable, profile=profile, outer_year=2022)
    assert result["selected_features"] == ["feature_b"]
