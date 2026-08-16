import hashlib
import json
from pathlib import Path

import pytest

from libs.modeling.experiment_campaign.hashing import canonical_sha256, file_sha256
from libs.modeling.experiment_campaign.protocol import AccessLedger
from libs.modeling.experiment_campaign.registry import validate_resolved_profile


CAMPAIGN = Path("experiments/top10_20260815")
EXPERIMENT_ID = "family-01-weighted-v8-control"
EXPECTED_FEATURE_SHA256 = "13E545D762A3F1BE4D023D82B8E65D77E41589031051F1F6796D742F25223022"
EXPERIMENT_ZERO_PREFIX = "D3F2BC6807F707C0A4696091E64DB92E773BD26F7266A1E7B718BFDC5AE891FB"


def _read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def test_materialized_profile_and_preregistration_are_frozen_before_launch():
    profile_path = CAMPAIGN / "profiles" / f"{EXPERIMENT_ID}.json"
    prereg_path = CAMPAIGN / "runs" / EXPERIMENT_ID / "preregistration.json"
    attempts_path = CAMPAIGN / "runs" / EXPERIMENT_ID / "attempts.jsonl"

    assert profile_path.is_file()
    assert prereg_path.is_file()
    assert attempts_path.is_file()
    profile = _read_json(profile_path)
    prereg = _read_json(prereg_path)
    assert validate_resolved_profile(profile) == prereg["profile_sha256"]
    assert canonical_sha256(profile["features"]) == EXPECTED_FEATURE_SHA256
    assert profile["preset"] == "hybrid"
    assert profile["use_recency_weights"] is True
    assert profile["decay_rate"] == 0.15
    assert profile["refit_full"] is False
    assert profile["calculate_importance"] is False
    assert prereg["variant_bound"] == 1
    assert prereg["outer_years"] == [2022, 2023, 2024, 2025]
    assert prereg["selection_boundary"] == "Original"
    assert prereg["same_row_foundation_context_admissible"] is False
    assert prereg["gate_state_required"] == "closed"
    assert prereg["registry_prefix_sha256_before"] == EXPERIMENT_ZERO_PREFIX
    assert prereg["source_artifact_mode"] == "fixed-read-only-campaign-artifact"


def test_chronology_validator_rejects_same_event_and_future_context():
    from libs.modeling.experiment_campaign.families.weighted_v8 import validate_prediction_chronology

    valid = [{
        "fight_id": "outer",
        "event_id": "outer-event",
        "event_date": "2025-01-11",
        "fit_max_date": "2025-01-04",
        "context_max_date": "2025-01-04",
        "fit_event_ids": ["prior-event"],
        "embargo_days": 7,
    }]
    validate_prediction_chronology(valid)
    same_event = [{**valid[0], "fit_event_ids": ["outer-event"]}]
    with pytest.raises(ValueError, match="same-event"):
        validate_prediction_chronology(same_event)
    future = [{**valid[0], "context_max_date": "2025-01-05"}]
    with pytest.raises(ValueError, match="embargo"):
        validate_prediction_chronology(future)


def test_actual_terminal_result_has_complete_folds_or_one_preserved_failure():
    from libs.modeling.experiment_campaign.runner import verify_family_run

    manifest_path = CAMPAIGN / "runs" / EXPERIMENT_ID / "manifest.json"
    if not manifest_path.exists():
        pytest.skip("pre-fit preregistration slice has not produced a terminal result")
    result = verify_family_run(CAMPAIGN, EXPERIMENT_ID, recompute_all=True)
    assert result["status"] in {"complete", "failed"}
    assert result["gate_access_count"] == 0
    if result["status"] == "complete":
        assert result["outer_years"] == [2022, 2023, 2024, 2025]
        assert result["fold_prediction_count"] == 4
        assert result["metrics"]["row_count"] > 0
        assert result["metrics"]["log_loss"] > 0
        assert set(result["metrics"]["fold_metrics"]) == {
            "2022", "2023", "2024", "2025"
        }
        assert set(result["paired_event_block_intervals"]) == {
            "log_loss_delta", "brier_delta", "accuracy_delta"
        }
    else:
        assert result["terminal_failure"]["failed_fold"] in {2022, 2023, 2024, 2025}
        assert result["terminal_failure"]["exit_code"] != 0
        assert result["terminal_failure"]["stderr_sha256"]


def test_registry_prefix_and_gate_are_preserved_until_terminal_append():
    before = (CAMPAIGN / "runs" / EXPERIMENT_ID / "preregistration.json")
    prereg = _read_json(before)
    assert prereg["registry_prefix_sha256_before"] == EXPERIMENT_ZERO_PREFIX
    registry = (CAMPAIGN / "registry.jsonl").read_bytes()
    assert file_sha256(CAMPAIGN / "baseline" / "manifest.json") == prereg["baseline_manifest_file_sha256"]
    first_record = registry.splitlines(keepends=True)[0]
    assert hashlib.sha256(first_record).hexdigest().upper() == EXPERIMENT_ZERO_PREFIX
    assert canonical_sha256(_read_json(CAMPAIGN / "baseline" / "fold-manifest.json")) == prereg["fold_manifest_sha256"]
    gate = AccessLedger(CAMPAIGN).gate_status()
    assert gate["state"] == "closed"
    assert gate["protected_access_count"] == 0
