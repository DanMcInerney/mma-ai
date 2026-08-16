"""Family 6 multi-timescale fighter-state preregistration constants."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

from ..fighter_states import validate_preregistered_profiles
from ..hashing import canonical_sha256, file_sha256, write_canonical_json
from ..protocol import AccessLedger
from .semantic_portfolio import V8_FEATURES


EXPERIMENT_ID = "family-06-multiscale-count-aware-state"
RUN_ALIAS = "family-06-fighter-states"
FROZEN_SPEC_SHA256 = "93FB5CC31AD810B1867FFC8A250DD257AAF74732998D103D56AB8D3A2D309A23"
FROZEN_SOURCE_SHA256 = "157649B780965ECC585F18B3030199CDC0F4FE3013958FFA4095FCF665FDB1EA"

STATE_FEATURES = (
    "recent_win_rate",
    "recent_ko_rate",
    "recent_submission_rate",
    "decay18_win_rate",
    "decay18_ko_rate",
    "decay18_submission_rate",
    "career_win_rate",
    "career_ko_rate",
    "career_submission_rate",
    "win_trend_short_medium",
    "win_trend_medium_career",
    "sparse_ko_posterior",
    "sparse_submission_posterior",
    "high_count_striking_state",
    "high_count_head_state",
    "inactivity_days",
    "age_win_interaction",
    "state_effective_support",
    "state_win_uncertainty",
)

PROFILE_IDS = (
    "v8-retained-incumbent-control",
    "recent-one-two-fight-state",
    "18-month-decayed-state",
    "career-state",
    "short-medium-career-trend-stack",
    "sparse-event-hierarchical-shrinkage",
    "robust-high-count-striking-state",
    "bounded-all-state-portfolio",
)


def _definition(name: str) -> dict[str, Any]:
    if name.startswith("recent_"):
        version, scale = "recent-last-two-v1", "last-two-prior-fights"
    elif name.startswith("decay18_"):
        version, scale = "decay-18m-half-life-180d-v1", "prior-548-days"
    elif name.startswith("career_"):
        version, scale = "career-count-aware-v1", "all-prior-fights"
    elif name.startswith("sparse_"):
        version, scale = "beta-binomial-hierarchical-v1", "prior-548-days"
    elif name.startswith("high_count_"):
        version, scale = "winsorized-decay-adjusted-performance-v1", "all-prior-snapshots"
    elif name.startswith("win_trend_"):
        version, scale = "bounded-rate-difference-v1", "cross-timescale-prior"
    elif name == "inactivity_days":
        version, scale = "capped-prior-gap-v1", "most-recent-prior-fight"
    elif name == "age_win_interaction":
        version, scale = "prior-age-x-career-win-v1", "all-prior-fights"
    elif name == "state_effective_support":
        version, scale = "log1p-career-exposure-v1", "all-prior-fights"
    else:
        version, scale = "career-beta-uncertainty-v1", "all-prior-fights"
    return {
        "formula_version": version,
        "timescale": scale,
        "exposure": "explicit-prior-fight-effective-count",
        "prior_id": "sparse-beta-1-4",
    }


def build_preregistered_profile() -> dict[str, Any]:
    """Return the exact maximum-eight menu without decoding a source row."""

    groups = (
        (),
        ("recent_win_rate", "recent_ko_rate", "recent_submission_rate", "inactivity_days", "age_win_interaction"),
        ("decay18_win_rate", "decay18_ko_rate", "decay18_submission_rate", "inactivity_days"),
        ("career_win_rate", "career_ko_rate", "career_submission_rate", "state_effective_support"),
        (
            "recent_win_rate",
            "decay18_win_rate",
            "career_win_rate",
            "win_trend_short_medium",
            "win_trend_medium_career",
        ),
        ("sparse_ko_posterior", "sparse_submission_posterior", "state_effective_support", "state_win_uncertainty"),
        ("high_count_striking_state", "high_count_head_state", "state_effective_support"),
        STATE_FEATURES,
    )
    definitions = {name: _definition(name) for name in STATE_FEATURES}
    profiles = []
    for profile_id, names in zip(PROFILE_IDS, groups, strict=True):
        names = list(names)
        profiles.append(
            {
                "id": profile_id,
                "base_feature_names": list(V8_FEATURES),
                "base_feature_sha256": canonical_sha256(list(V8_FEATURES)),
                "feature_names": names,
                "formula_versions": [definitions[name]["formula_version"] for name in names],
                "ordered_feature_sha256": canonical_sha256([*V8_FEATURES, *names]),
            }
        )
    profile = {
        "experiment_id": EXPERIMENT_ID,
        "family_number": 6,
        "frozen_spec_sha256": FROZEN_SPEC_SHA256,
        "source_sha256": FROZEN_SOURCE_SHA256,
        "source_path": "artifacts/01-campaign-harness/frozen/training_data.csv",
        "cutoff": "2025-12-31",
        "registered_priors": {
            "sparse-beta-1-4": {
                "family": "beta-binomial",
                "alpha": 1.0,
                "beta": 4.0,
                "fallback_mean": 0.2,
            }
        },
        "normalization": {"fit_scope": "outer-train-only", "method": "training-median-standard-scale"},
        "construction": {
            "same_date_policy": "exclude-entire-date",
            "recent_fight_count": 2,
            "decay_18m_days": 548,
            "decay_half_life_days": 180,
            "high_count_half_life_days": 365,
            "robust_winsor_limits": [0.1, 0.9],
            "inactivity_cap_days": 730,
            "age_interaction_center_years": 30.0,
        },
        "feature_definitions": definitions,
        "profiles": profiles,
        "outer_years": [2022, 2023, 2024, 2025],
        "inner_validation_year_count": 3,
        "selection": {
            "fit_scope": "prior-inner-only",
            "score": "mean-inner-log-loss",
            "tie_break": list(PROFILE_IDS),
            "outer_label_selection_count": 0,
        },
        "model": {
            "type": "logistic-regression",
            "imputation": "training-median",
            "scaling": "standard",
            "penalty": "l2",
            "C": 1.0,
            "solver": "lbfgs",
            "max_iter": 2000,
            "random_state": 20260815,
        },
        "bootstrap": {"iterations": 2000, "seed": 20260815},
        "database_access": {"used": False, "sql": None, "urls": []},
        "invocation": {"gpu_lease_count": 1, "retry_count": 0, "serialized": True},
    }
    validate_preregistered_profiles(profile)
    return profile


def write_preregistration(campaign_root: Path, *, source_revision: str) -> dict[str, Any]:
    """Persist the frozen menu while every construction and score path is absent."""

    campaign_root = Path(campaign_root)
    profile_path = campaign_root / "profiles/family-06-fighter-states.json"
    preregistration_path = campaign_root / "runs/family-06-fighter-states/preregistration.json"
    artifact_root = campaign_root / "artifacts/07-family-06-fighter-states"
    data_root = campaign_root.parents[1] / "data/experiments/top10_20260815/family-06-fighter-states"
    if profile_path.exists() or preregistration_path.exists() or artifact_root.exists() or data_root.exists():
        raise ValueError("family 6 preregistration destinations must all be absent")
    gate = AccessLedger(campaign_root).gate_status()
    if gate["state"] != "closed" or gate["protected_access_count"] != 0:
        raise ValueError("family 6 preregistration requires the gate closed with zero access")
    profile = build_preregistered_profile()
    write_canonical_json(profile_path, profile)
    preregistration = {
        "experiment_id": EXPERIMENT_ID,
        "family_number": 6,
        "source_revision": source_revision,
        "frozen_spec_sha256": FROZEN_SPEC_SHA256,
        "profile_path": "profiles/family-06-fighter-states.json",
        "profile_sha256": canonical_sha256(profile),
        "profile_file_sha256": file_sha256(profile_path),
        "preregistered_profile_ids": list(PROFILE_IDS),
        "ordered_profile_hashes": {
            item["id"]: item["ordered_feature_sha256"] for item in profile["profiles"]
        },
        "source_file_sha256": profile["source_sha256"],
        "registry_prefix_sha256_before": hashlib.sha256(
            (campaign_root / "registry.jsonl").read_bytes()
        ).hexdigest().upper(),
        "scoring_state": "not-started",
        "selection": profile["selection"],
        "database_access": profile["database_access"],
        "invocation": profile["invocation"],
        "gate_required_state": "closed-zero-access",
        "terminal_failure_rule": "Any lineage, chronology, source, menu, safety, or destination mismatch terminates without retry.",
    }
    write_canonical_json(preregistration_path, preregistration)
    return preregistration
