"""Family 5 stable semantic portfolio preregistration and materialization."""

from __future__ import annotations

import csv
import hashlib
from pathlib import Path
from typing import Any

from ..hashing import canonical_sha256, file_sha256, write_canonical_json
from ..protocol import AccessLedger
from ..semantic_portfolio import MEASUREMENT_GROUP_IDS, validate_preregistered_profile


EXPERIMENT_ID = "family-05-stable-semantic-portfolio"
RUN_PATH = "runs/family-05-semantic-portfolio"
ARTIFACT_PATH = "artifacts/06-family-05-semantic-portfolio"
DATA_PATH = "../../data/experiments/top10_20260815/family-05-semantic-portfolio"
FROZEN_SPEC_SHA256 = "93FB5CC31AD810B1867FFC8A250DD257AAF74732998D103D56AB8D3A2D309A23"
FROZEN_SOURCE_SHA256 = "157649B780965ECC585F18B3030199CDC0F4FE3013958FFA4095FCF665FDB1EA"
V8_ORDERED_FEATURE_SHA256 = "13E545D762A3F1BE4D023D82B8E65D77E41589031051F1F6796D742F25223022"
FROZEN_SOURCE = Path(
    r"C:\Users\danhm\mma-ai\worktrees\top10-20260815"
    r"\experiments\top10_20260815\artifacts\01-campaign-harness"
    r"\frozen\training_data.csv"
)

V8_FEATURES = (
    "age_dec_avg_diff",
    "age_ratio_diff",
    "reach_ratio_dec_avg_diff",
    "days_since_last_fight_dec_avg_diff",
    "sig_str_land_per_min_dec_adjperf_dec_avg_diff",
    "weightclass_encoded",
    "sig_str_land_ratio_dec_adjperf_dec_avg_diff",
    "clinch_def_dec_adjperf_dec_avg_diff",
    "sig_str_acc_dec_adjperf_dec_avg_diff",
    "head_land_ratio_dec_adjperf_dec_avg_diff",
    "head_def_dec_adjperf_dec_avg_diff",
    "head_acc_dec_adjperf_dec_avg_diff",
    "body_def_dec_adjperf_dec_avg_diff",
    "body_acc_dec_adjperf_dec_avg_diff",
    "leg_land_per_min_dec_adjperf_dec_avg_diff",
    "leg_acc_dec_adjperf_dec_avg_diff",
    "distance_land_ratio_dec_adjperf_dec_avg_diff",
    "sig_str_land_pressure_dec_adjperf_dec_avg_diff",
    "distance_acc_dec_adjperf_dec_avg_diff",
    "distance_land_per_min_dec_adjperf_dec_avg_diff",
    "body_land_ratio_dec_adjperf_dec_avg_diff",
    "clinch_land_ratio_dec_adjperf_dec_avg_diff",
    "clinch_acc_dec_adjperf_dec_avg_diff",
    "ground_acc_dec_adjperf_dec_avg_diff",
    "ground_land_per_ctrl_dec_avg_diff",
    "ko_per_sig_str_land_dec_adjperf_dec_avg_diff",
    "ko_ratio_dec_adjperf_dec_avg_diff",
    "sub_att_ratio_dec_adjperf_dec_avg_diff",
    "sub_att_dec_avg_diff",
    "sub_def_dec_adjperf_dec_avg_diff",
    "win_dec_adjperf_dec_avg_diff",
    "rev_per_ctrlopp_dec_adjperf_dec_avg_diff",
    "td_acc_dec_adjperf_dec_avg_diff",
    "td_def_dec_adjperf_dec_avg_diff",
    "ctrl_ratio_dec_adjperf_dec_avg_diff",
    "ctrl_per_min_opp_dec_avg_diff",
    "td_att_opp_dec_avg_diff",
    "td_att_rd1_opp_dec_avg_diff",
    "td_land_per_ctrl_dec_adjperf_dec_avg_diff",
    "td_per_sig_str_att_dec_adjperf_dec_avg_diff",
)

FEATURE_GROUPS = {
    "age_dec_avg_diff": "demographics-experience",
    "age_ratio_diff": "demographics-experience",
    "reach_ratio_dec_avg_diff": "demographics-experience",
    "days_since_last_fight_dec_avg_diff": "demographics-experience",
    "weightclass_encoded": "demographics-experience",
    "sig_str_land_per_min_dec_adjperf_dec_avg_diff": "global-striking-pace-efficiency",
    "sig_str_land_ratio_dec_adjperf_dec_avg_diff": "global-striking-pace-efficiency",
    "sig_str_acc_dec_adjperf_dec_avg_diff": "global-striking-pace-efficiency",
    "head_land_ratio_dec_adjperf_dec_avg_diff": "head-body-leg-targeting",
    "head_def_dec_adjperf_dec_avg_diff": "head-body-leg-targeting",
    "head_acc_dec_adjperf_dec_avg_diff": "head-body-leg-targeting",
    "body_def_dec_adjperf_dec_avg_diff": "head-body-leg-targeting",
    "body_acc_dec_adjperf_dec_avg_diff": "head-body-leg-targeting",
    "leg_land_per_min_dec_adjperf_dec_avg_diff": "head-body-leg-targeting",
    "leg_acc_dec_adjperf_dec_avg_diff": "head-body-leg-targeting",
    "body_land_ratio_dec_adjperf_dec_avg_diff": "head-body-leg-targeting",
    "distance_land_ratio_dec_adjperf_dec_avg_diff": "range-clinch-ground-position",
    "sig_str_land_pressure_dec_adjperf_dec_avg_diff": "range-clinch-ground-position",
    "distance_acc_dec_adjperf_dec_avg_diff": "range-clinch-ground-position",
    "distance_land_per_min_dec_adjperf_dec_avg_diff": "range-clinch-ground-position",
    "clinch_land_ratio_dec_adjperf_dec_avg_diff": "range-clinch-ground-position",
    "clinch_acc_dec_adjperf_dec_avg_diff": "range-clinch-ground-position",
    "ground_acc_dec_adjperf_dec_avg_diff": "range-clinch-ground-position",
    "ground_land_per_ctrl_dec_avg_diff": "range-clinch-ground-position",
    "sub_att_ratio_dec_adjperf_dec_avg_diff": "takedown-control-submission",
    "sub_att_dec_avg_diff": "takedown-control-submission",
    "sub_def_dec_adjperf_dec_avg_diff": "takedown-control-submission",
    "rev_per_ctrlopp_dec_adjperf_dec_avg_diff": "takedown-control-submission",
    "td_acc_dec_adjperf_dec_avg_diff": "takedown-control-submission",
    "td_def_dec_adjperf_dec_avg_diff": "takedown-control-submission",
    "ctrl_ratio_dec_adjperf_dec_avg_diff": "takedown-control-submission",
    "ctrl_per_min_opp_dec_avg_diff": "takedown-control-submission",
    "td_att_opp_dec_avg_diff": "takedown-control-submission",
    "td_att_rd1_opp_dec_avg_diff": "takedown-control-submission",
    "td_land_per_ctrl_dec_adjperf_dec_avg_diff": "takedown-control-submission",
    "td_per_sig_str_att_dec_adjperf_dec_avg_diff": "takedown-control-submission",
    "ko_per_sig_str_land_dec_adjperf_dec_avg_diff": "damage-finish",
    "ko_ratio_dec_adjperf_dec_avg_diff": "damage-finish",
    "clinch_def_dec_adjperf_dec_avg_diff": "opponent-style-strength-of-schedule",
    "win_dec_adjperf_dec_avg_diff": "opponent-style-strength-of-schedule",
}


def _source_header(source_path: Path) -> tuple[str, ...]:
    with Path(source_path).open(encoding="utf-8", newline="") as source:
        return tuple(next(csv.reader(source)))


def _header_sha256(header: tuple[str, ...]) -> str:
    return hashlib.sha256("\n".join(header).encode()).hexdigest().upper()


def build_preregistered_profile(source_path: Path = FROZEN_SOURCE) -> dict[str, Any]:
    """Build the exact eight-profile menu without reading any score or label row."""

    header = _source_header(source_path)
    if canonical_sha256(list(V8_FEATURES)) != V8_ORDERED_FEATURE_SHA256:
        raise ValueError("v8 feature anchor differs from the immutable ordered list")
    if set(FEATURE_GROUPS) != set(V8_FEATURES):
        raise ValueError("every v8 candidate must have exactly one authored semantic group")
    header_sha256 = _header_sha256(header)
    candidates = [
        {
            "name": feature,
            "semantic_id": feature,
            "measurement_group": FEATURE_GROUPS[feature],
            "source_file_sha256": FROZEN_SOURCE_SHA256,
            "source_header_sha256": header_sha256,
            "formula": f"identity({feature})",
            "available_by": "2014-01-01",
            "domain_redundancy_rank": sum(
                1
                for prior in V8_FEATURES[: index + 1]
                if FEATURE_GROUPS[prior] == FEATURE_GROUPS[feature]
            ),
        }
        for index, feature in enumerate(V8_FEATURES)
    ]
    profiles = []
    for profile_id, groups in (
        ("v8-control", MEASUREMENT_GROUP_IDS),
        *((group, (group,)) for group in MEASUREMENT_GROUP_IDS),
    ):
        features = [feature for feature in V8_FEATURES if FEATURE_GROUPS[feature] in groups]
        profiles.append(
            {
                "id": profile_id,
                "included_groups": list(groups),
                "ordered_features": features,
                "ordered_feature_sha256": canonical_sha256(features),
            }
        )
    profile = {
        "experiment_id": EXPERIMENT_ID,
        "family_number": 5,
        "frozen_spec_sha256": FROZEN_SPEC_SHA256,
        "frozen_source": {
            "path": "artifacts/01-campaign-harness/frozen/training_data.csv",
            "absolute_path": str(Path(source_path)),
            "sha256": FROZEN_SOURCE_SHA256,
            "ordered_header_sha256": header_sha256,
            "cutoff": "2025-12-31",
        },
        "v8_ordered_features": list(V8_FEATURES),
        "v8_ordered_feature_sha256": V8_ORDERED_FEATURE_SHA256,
        "measurement_group_ids": list(MEASUREMENT_GROUP_IDS),
        "candidate_features": candidates,
        "measurement_profiles": profiles,
        "outer_years": [2022, 2023, 2024, 2025],
        "inner_validation_year_count": 3,
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
        "selection": {
            "evidence_role": "inner-chronological",
            "profile_score": "mean-inner-log-loss",
            "profile_tie_break": [item["id"] for item in profiles],
            "stability_threshold": 2 / 3,
            "drop_column_min_improvement": 0.0,
            "minimum_fold_support": 3,
            "domain_redundancy_cap": 3,
            "tie_break": "profile-order-then-feature-order",
            "combined_row_importance_role": "non-selection",
        },
        "bootstrap": {"iterations": 2000, "seed": 20260815},
        "promotion_rule": "pooled log-loss delta and paired event-block interval upper bound must both be below zero",
        "incumbent_id": "family-01-weighted-v8-control",
        "historical_authored_documents_role": "group-membership-proposal-only",
        "outer_label_roles": ["final-metrics-only"],
        "gate_required_state": "closed-zero-access",
    }
    validate_preregistered_profile(profile, source_header=header)
    return profile


def write_preregistration(campaign_root: Path) -> dict[str, Any]:
    """Persist the frozen menu and not-started commitment before any score."""

    campaign_root = Path(campaign_root)
    gate = AccessLedger(campaign_root).gate_status()
    if gate["state"] != "closed" or gate["protected_access_count"] != 0:
        raise ValueError("family 5 preregistration requires a closed zero-access gate")
    profile_path = campaign_root / "profiles/family-05-semantic-portfolio.json"
    preregistration_path = campaign_root / RUN_PATH / "preregistration.json"
    if profile_path.exists() or preregistration_path.exists():
        raise ValueError("family 5 preregistration destination already exists")
    profile = build_preregistered_profile()
    write_canonical_json(profile_path, profile)
    registry_bytes = (campaign_root / "registry.jsonl").read_bytes()
    profiles = profile["measurement_profiles"]
    preregistration = {
        "experiment_id": EXPERIMENT_ID,
        "family_number": 5,
        "frozen_spec_sha256": FROZEN_SPEC_SHA256,
        "profile_path": "profiles/family-05-semantic-portfolio.json",
        "profile_file_sha256": file_sha256(profile_path),
        "profile_sha256": canonical_sha256(profile),
        "registry_prefix_sha256_before": hashlib.sha256(registry_bytes).hexdigest().upper(),
        "scoring_state": "not-started",
        "preregistered_profile_ids": [item["id"] for item in profiles],
        "ordered_profile_hashes": {
            item["id"]: item["ordered_feature_sha256"] for item in profiles
        },
        "source_file_sha256": FROZEN_SOURCE_SHA256,
        "source_header_sha256": profile["frozen_source"]["ordered_header_sha256"],
        "selection": profile["selection"],
        "outer_label_roles": profile["outer_label_roles"],
        "gate_required_state": profile["gate_required_state"],
        "terminal_failure_rule": "Any lineage, chronology, menu, source, gate, or destination mismatch terminates without retry.",
    }
    write_canonical_json(preregistration_path, preregistration)
    return preregistration
