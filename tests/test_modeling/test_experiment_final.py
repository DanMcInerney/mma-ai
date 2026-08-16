from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pytest

from libs.modeling.experiment_campaign.gate import (
    FinalProtocolError,
    record_prospective_prediction,
    validate_candidate_seal,
    validate_compromise_record,
)
from libs.modeling.experiment_campaign.report import validate_report_language


PREDICTION_SHA = "6536FEEF899FEF40E0FC7979ECE96B7653EEEB603493120D7C89D8176419CF14"
REGISTRY_SHA = "A1DA8BB50D1E1685061222CCFF73F83B38E00EFB65CE1CA97B4D4E751B08A6DB"


def _seal() -> dict:
    return {
        "kind": "development-candidate-seal",
        "sealed_at": datetime(2026, 8, 16, tzinfo=timezone.utc).isoformat(),
        "registry_prefix_sha256": REGISTRY_SHA,
        "candidate": {
            "experiment_id": "family-01-weighted-v8-control",
            "boundary": "Original",
            "row_count": 1108,
            "outer_years": [2022, 2023, 2024, 2025],
            "prediction_sha256": PREDICTION_SHA,
        },
        "gate": {
            "gate_id": "historically_exposed_campaign_gate",
            "status": "retired-compromised-unscored",
            "software_access_count": 0,
            "metric": None,
        },
    }


def test_development_candidate_seal_is_exact_and_contains_no_gate_metric():
    assert validate_candidate_seal(_seal())["candidate"]["prediction_sha256"] == PREDICTION_SHA
    bad = _seal()
    bad["candidate"]["prediction_sha256"] = "0" * 64
    with pytest.raises(FinalProtocolError, match="prediction identity"):
        validate_candidate_seal(bad)
    bad = _seal()
    bad["gate"]["metric"] = {"accuracy": 1.0}
    with pytest.raises(FinalProtocolError, match="must not have a metric"):
        validate_candidate_seal(bad)


def test_compromise_record_preserves_zero_ledger_access_without_untouched_claim():
    incident = {
        "incident_id": "family-05-preflight-2026-label-decode",
        "gate_id": "historically_exposed_campaign_gate",
        "population": {
            "row_count": 178,
            "date_range": ["2026-01-01", "2026-08-08"],
        },
        "classification": "protocol-compromise-retired-unscored",
        "software_access_count_before": 0,
        "software_access_count_after": 0,
        "prediction_identity": None,
        "metric": None,
        "facts": [
            "A failed Family-5 preflight decoded target values before aborting.",
            "No prediction, score, persisted label, printed label, or selection use occurred.",
        ],
    }
    assert validate_compromise_record(incident)["population"]["row_count"] == 178
    bad = json.loads(json.dumps(incident))
    bad["facts"].append("The 2026 holdout remains untouched.")
    with pytest.raises(FinalProtocolError, match="forbidden 2026 boundary claim"):
        validate_compromise_record(bad)


def test_prospective_seam_rejects_past_outcomes_and_overwrite(tmp_path: Path):
    record = {
        "fight_id": "future-fixture-1",
        "event_id": "future-event-1",
        "event_date": "2026-08-09",
        "fighter_1_id": "alpha",
        "fighter_2_id": "beta",
        "probability_fighter_1": 0.62,
        "candidate_prediction_sha256": PREDICTION_SHA,
        "source_revision": "abc123",
        "data_identity": "D" * 64,
    }
    first = record_prospective_prediction(tmp_path, record)
    assert first["record_sha256"]
    with pytest.raises(FinalProtocolError, match="already exists"):
        record_prospective_prediction(tmp_path, record)
    outcome_known = {**record, "fight_id": "future-fixture-2", "winner": "alpha"}
    with pytest.raises(FinalProtocolError, match="outcome"):
        record_prospective_prediction(tmp_path, outcome_known)
    past = {**record, "fight_id": "future-fixture-3", "event_date": "2026-08-08"}
    with pytest.raises(FinalProtocolError, match="after 2026-08-08"):
        record_prospective_prediction(tmp_path, past)


def test_report_language_distinguishes_negative_and_inconclusive():
    report = {
        "gate": {
            "status": "compromised-retired-unscored",
            "metric": None,
            "software_access_count": 0,
        },
        "experiments": [
            {"family_number": 2, "classification": "negative"},
            {
                "family_number": 8,
                "classification": "inconclusive",
                "summary": "CatBoost was not evaluated because its dependency was unavailable.",
            },
        ],
        "recommendation": (
            "Keep the development incumbent as a research candidate; do not claim a production "
            "validation result until outcome-unknown post-2026-08-08 prospective fights mature."
        ),
    }
    validate_report_language(report)
    report["experiments"][1]["classification"] = "negative"
    with pytest.raises(FinalProtocolError, match="CatBoost"):
        validate_report_language(report)
