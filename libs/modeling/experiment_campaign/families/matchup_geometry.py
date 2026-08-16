"""Family 7 matchup-interaction and fighter-swap preregistration."""

from __future__ import annotations

from typing import Any

from ..hashing import canonical_sha256
from ..matchup_geometry import validate_preregistered_matchup_profiles


EXPERIMENT_ID = "family-07-matchup-swap-geometry"
RUN_ALIAS = "family-07-matchup-geometry"
FROZEN_SPEC_SHA256 = "93FB5CC31AD810B1867FFC8A250DD257AAF74732998D103D56AB8D3A2D309A23"
FROZEN_SOURCE_SHA256 = "157649B780965ECC585F18B3030199CDC0F4FE3013958FFA4095FCF665FDB1EA"

PROFILE_IDS = (
    "retained-incumbent-control",
    "striking-offense-versus-defense",
    "grappling-weakness-versus-exploitation",
    "opponent-style-matchmaking-tendency",
    "damage-versus-durability",
    "all-directional-interactions",
    "fighter-swap-averaged-prediction",
    "hard-complementary-antisymmetric-geometry",
)

INTERACTION_DEFINITIONS = (
    {
        "name": "striking_offense_vs_defense",
        "left": "high_count_striking_state",
        "right": "striking_defense_state",
        "formula": "cross-difference",
        "minimum_support": 2.0,
        "fallback": 0.0,
        "swap_rule": "negate",
        "lineage": "both-fighters-prior-only",
    },
    {
        "name": "grappling_exploitation_vs_weakness",
        "left": "grappling_exploitation_state",
        "right": "grappling_weakness_state",
        "formula": "cross-difference",
        "minimum_support": 2.0,
        "fallback": 0.0,
        "swap_rule": "negate",
        "lineage": "both-fighters-prior-only",
    },
    {
        "name": "style_matchmaking_tendency",
        "left": "opponent_style_success_state",
        "right": "style_susceptibility_state",
        "formula": "cross-difference",
        "minimum_support": 3.0,
        "fallback": 0.0,
        "swap_rule": "negate",
        "lineage": "both-fighters-prior-only",
    },
    {
        "name": "damage_vs_durability",
        "left": "damage_output_state",
        "right": "durability_weakness_state",
        "formula": "cross-difference",
        "minimum_support": 2.0,
        "fallback": 0.0,
        "swap_rule": "negate",
        "lineage": "both-fighters-prior-only",
    },
)


def build_preregistered_profile() -> dict[str, Any]:
    interaction_names = [item["name"] for item in INTERACTION_DEFINITIONS]
    groups = (
        (),
        (interaction_names[0],),
        (interaction_names[1],),
        (interaction_names[2],),
        (interaction_names[3],),
        tuple(interaction_names),
        tuple(interaction_names),
        tuple(interaction_names),
    )
    geometries = (
        "original-only-control",
        "original-only",
        "original-only",
        "original-only",
        "original-only",
        "original-only",
        "original-swapped-average",
        "hard-complementary-antisymmetric",
    )
    profiles = []
    for profile_id, names, geometry in zip(PROFILE_IDS, groups, geometries, strict=True):
        names = list(names)
        profiles.append(
            {
                "id": profile_id,
                "interaction_names": names,
                "prediction_geometry": geometry,
                "ordered_interaction_sha256": canonical_sha256(
                    {"interaction_names": names, "prediction_geometry": geometry}
                ),
            }
        )
    profile = {
        "experiment_id": EXPERIMENT_ID,
        "family_number": 7,
        "frozen_spec_sha256": FROZEN_SPEC_SHA256,
        "frozen_source": {
            "path": "artifacts/01-campaign-harness/frozen/training_data.csv",
            "sha256": FROZEN_SOURCE_SHA256,
            "cutoff": "2025-12-31",
            "development_safe_id_count": 3_089,
            "retired_id_count": 178,
            "development_max_date": "2025-12-13",
        },
        "dependency": {
            "experiment_id": "family-06-multiscale-count-aware-state",
            "run_alias": "family-06-fighter-states",
            "required_data_path": "data/experiments/top10_20260815/family-06-fighter-states/matched-state-table.csv",
            "fallback": "terminal-failure-before-row-decode",
        },
        "role_swap": {
            "paired_metadata": ["id", "name", "url", "label", "odds", "market_probability"],
            "paired_payloads": ["features", "lineage"],
            "target_rule": "binary-complement",
            "antisymmetric_rule": "negate",
        },
        "interaction_definitions": [dict(item) for item in INTERACTION_DEFINITIONS],
        "normalization": {"fit_scope": "outer-train-only", "method": "training-median-standard-scale"},
        "profiles": profiles,
        "outer_years": [2022, 2023, 2024, 2025],
        "inner_validation_year_count": 3,
        "selection": {
            "fit_scope": "prior-inner-only",
            "score": "mean-inner-log-loss",
            "tie_break": list(PROFILE_IDS),
            "outer_label_selection_count": 0,
        },
        "geometry": {
            "probability_sum": 1.0,
            "tolerance": 1e-12,
            "store_original_and_swapped": True,
            "store_invariance_residual": True,
        },
        "database_access": {"used": False, "sql": None, "urls": []},
        "invocation": {"gpu_lease_count": 1, "retry_count": 0, "serialized": True},
    }
    validate_preregistered_matchup_profiles(profile)
    return profile
