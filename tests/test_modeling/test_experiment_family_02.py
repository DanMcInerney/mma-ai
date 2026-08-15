import hashlib
import json
import math
from datetime import date
from pathlib import Path

import pytest

from libs.modeling.experiment_campaign.hashing import canonical_sha256, file_sha256
from libs.modeling.experiment_campaign.protocol import AccessLedger


CAMPAIGN = Path("experiments/top10_20260815")
EXPERIMENT_ID = "family-02-horizon-recency"
EXPECTED_MENU = [
    "expanding-decay-0",
    "expanding-decay-0.05",
    "rolling-8y-decay-0.10",
    "rolling-6y-decay-0.15",
    "rolling-4y-decay-0.25",
    "expanding-piecewise-event-count",
    "rolling-8y-decay-0",
    "expanding-decay-0.15",
]
FAMILY_1_PREFIX = "B4F6FEE4AE5C2EDE6055684AC26D8A6426D02C8DB0920BB482B09750587C4279"


def _read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def test_exact_joint_menu_and_materialized_profile_cover_frozen_regimes():
    from libs.modeling.experiment_campaign.families.horizon_recency import (
        JOINT_VARIANTS,
        materialized_profile,
        validate_family_profile,
    )

    profile = materialized_profile()
    assert [variant["id"] for variant in JOINT_VARIANTS] == EXPECTED_MENU
    assert [variant["id"] for variant in profile["joint_variants"]] == EXPECTED_MENU
    assert len(profile["joint_variants"]) == len(set(EXPECTED_MENU)) == 8
    assert {variant["horizon"]["years"] for variant in JOINT_VARIANTS} == {
        None, 4, 6, 8
    }
    assert {variant["decay_rate"] for variant in JOINT_VARIANTS} == {
        0.0, 0.05, 0.10, 0.15, 0.25
    }
    assert {variant["weight_scheme"] for variant in JOINT_VARIANTS} == {
        "exponential-date", "piecewise-event-count"
    }
    assert profile["per_fit_time_cap_seconds"] > 0
    assert profile["base_training_profile"]["refit_full"] is False
    assert profile["base_training_profile"]["calculate_importance"] is False
    assert validate_family_profile(profile) == canonical_sha256(profile)


def test_date_and_event_weights_recompute_from_prior_only_metadata():
    from libs.modeling.experiment_campaign.families.horizon_recency import (
        compute_training_weights,
    )

    prior_rows = [
        {"event_id": "old", "event_date": "2019-12-31"},
        {"event_id": "recent-a", "event_date": "2023-12-31"},
        {"event_id": "recent-b", "event_date": "2023-12-31"},
        {"event_id": "newest", "event_date": "2024-12-31"},
    ]
    exponential = compute_training_weights(
        prior_rows,
        next(variant for variant in _variants() if variant["id"] == "rolling-4y-decay-0.25"),
        as_of_date=date(2025, 1, 1),
    )
    assert exponential[0] == 0.0
    assert exponential[1] == pytest.approx(math.exp(-0.25 * 367 / 365.25))
    assert exponential[2] == exponential[1]
    assert exponential[3] == pytest.approx(math.exp(-0.25 * 1 / 365.25))

    event_rows = [
        {"event_id": f"event-{index:03d}", "event_date": f"2024-01-{index + 1:02d}"}
        for index in range(30)
    ]
    piecewise = compute_training_weights(
        event_rows,
        next(variant for variant in _variants() if variant["id"] == "expanding-piecewise-event-count"),
        as_of_date=date(2025, 1, 1),
    )
    assert piecewise[:5] == [0.75] * 5
    assert piecewise[5:] == [1.0] * 25

    with pytest.raises(ValueError, match="prior-only"):
        compute_training_weights(
            prior_rows + [{"event_id": "outer", "event_date": "2025-01-01"}],
            _variants()[0],
            as_of_date=date(2025, 1, 1),
        )


def _variants():
    from libs.modeling.experiment_campaign.families.horizon_recency import JOINT_VARIANTS

    return JOINT_VARIANTS


def test_inner_selector_rejects_outer_or_gate_evidence_and_uses_frozen_tie_break():
    from libs.modeling.experiment_campaign.families.horizon_recency import select_joint_variant

    scores = [
        {
            "variant_id": variant_id,
            "partition": "inner-validation",
            "selection_max_date": "2024-12-20",
            "log_loss": 0.61,
        }
        for variant_id in EXPECTED_MENU
    ]
    selected = select_joint_variant(
        scores,
        outer_min_date=date(2025, 1, 1),
        embargo_days=7,
    )
    assert selected["variant_id"] == EXPECTED_MENU[0]
    assert selected["selection_basis"] == "chronological-inner-log-loss"

    with pytest.raises(ValueError, match="inner-validation"):
        select_joint_variant(
            [{**scores[0], "partition": "outer"}],
            outer_min_date=date(2025, 1, 1),
            embargo_days=7,
        )
    with pytest.raises(ValueError, match="embargo"):
        select_joint_variant(
            [{**scores[0], "selection_max_date": "2024-12-26"}],
            outer_min_date=date(2025, 1, 1),
            embargo_days=7,
        )


def test_actual_preregistration_precedes_launch_and_preserves_family_1_prefix():
    profile_path = CAMPAIGN / "profiles" / f"{EXPERIMENT_ID}.json"
    prereg_path = CAMPAIGN / "runs" / EXPERIMENT_ID / "preregistration.json"
    attempts_path = CAMPAIGN / "runs" / EXPERIMENT_ID / "attempts.jsonl"

    assert profile_path.is_file()
    assert prereg_path.is_file()
    assert attempts_path.is_file()
    profile = _read_json(profile_path)
    prereg = _read_json(prereg_path)
    assert [variant["id"] for variant in profile["joint_variants"]] == EXPECTED_MENU
    assert prereg["profile_sha256"] == canonical_sha256(profile)
    assert prereg["variant_menu"] == EXPECTED_MENU
    assert prereg["registry_prefix_sha256_before"] == FAMILY_1_PREFIX
    assert prereg["outer_years"] == [2022, 2023, 2024, 2025]
    assert prereg["selection_evidence"] == "chronological-inner-only"
    assert prereg["gate_state_required"] == "closed"
    launched = [
        json.loads(line)
        for line in attempts_path.read_text(encoding="utf-8").splitlines()
        if json.loads(line)["state"] == "launched"
    ]
    assert all(record["preregistration_commit"] for record in launched)
    assert len({record["attempt_id"] for record in launched}) == len(launched)
    assert {record["variant_id"] for record in launched} <= set(EXPECTED_MENU)


def test_actual_terminal_result_has_four_outer_sets_or_explicit_failure():
    from libs.modeling.experiment_campaign.runner import verify_family_run

    manifest_path = CAMPAIGN / "runs" / EXPERIMENT_ID / "manifest.json"
    if not manifest_path.exists():
        pytest.skip("pre-fit preregistration slice has not produced a terminal result")
    result = verify_family_run(CAMPAIGN, EXPERIMENT_ID, recompute_all=True)
    assert result["status"] in {"complete", "failed", "cancelled", "limited"}
    assert result["gate_access_count"] == 0
    if result["status"] == "complete":
        assert result["outer_years"] == [2022, 2023, 2024, 2025]
        assert result["fold_prediction_count"] == 4
        assert len(result["selected_variants"]) == 4
        assert result["metrics"]["row_count"] == 1108
        assert set(result["paired_event_block_intervals"]) == {
            "log_loss_delta", "brier_delta", "accuracy_delta"
        }
        assert set(result["drift_summary"]) == {"fold_log_loss", "year_over_year_log_loss_delta"}
    else:
        assert result["terminal_failure"]["attempt_id"]
        assert result["terminal_failure"]["exit_state"] in {"failed", "cancelled", "limited"}


def test_registry_prefix_gate_and_prior_manifest_remain_unchanged():
    prereg = _read_json(CAMPAIGN / "runs" / EXPERIMENT_ID / "preregistration.json")
    registry = (CAMPAIGN / "registry.jsonl").read_bytes()
    family_1_prefix = registry.splitlines(keepends=True)[:2]
    assert hashlib.sha256(b"".join(family_1_prefix)).hexdigest().upper() == FAMILY_1_PREFIX
    assert file_sha256(CAMPAIGN / "runs/family-01-weighted-v8-control/manifest.json") == prereg["family_1_manifest_file_sha256"]
    assert AccessLedger(CAMPAIGN).gate_status()["state"] == "closed"
    assert AccessLedger(CAMPAIGN).gate_status()["protected_access_count"] == 0
