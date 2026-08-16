from __future__ import annotations

import copy
from pathlib import Path

import pytest

from libs.modeling.split_refit_experiment.report import (
    ReportError,
    build_report,
    render_report_markdown,
    validate_report_documents,
)


CAMPAIGN = Path("experiments/split_refit_20260816")


def test_report_keeps_all_historical_denominators_separate_and_exact():
    report = build_report(CAMPAIGN)
    results = report["historical_evaluations"]
    assert [(row["correct_count"], row["row_count"]) for row in results] == [
        (309, 460),
        (726, 1108),
        (202, 307),
    ]
    assert [row["accuracy"] for row in results] == [
        0.6717391304347826,
        0.6552346570397112,
        0.6579804560260586,
    ]
    assert [row["positive_log_loss"] for row in results] == [
        0.6131854280928061,
        0.6195954814877112,
        0.6261685244094812,
    ]
    assert report["evidence_boundary"]["pooled"] is False
    assert report["evidence_boundary"]["historically_exposed"] is True
    assert report["evidence_boundary"]["untouched_or_external_claim"] is False
    assert report["decision"]["recommendation"] == "retain-rollback"


def test_report_preserves_full_refit_lineage_and_no_validation_claim():
    report = build_report(CAMPAIGN)
    refit = report["full_data_refit"]
    assert refit["source_rows"] == 3267
    assert refit["node_count"] == 22
    assert refit["fresh_full_base_count"] == 9
    assert refit["realmlp_full"] == {
        "fit_rows": 2807,
        "origin": "original-clone",
    }
    assert refit["full_ensemble"] == {
        "dependencies": ["Mitra_FULL", "XGBoost_FULL"],
        "effective_fit_rows": 3267,
        "fit_rows": 460,
        "weights": {"Mitra_FULL": 0.96, "XGBoost_FULL": 0.04},
    }
    assert refit["validation_claims"] == []
    assert refit["process_exit_code"] == 1
    assert refit["post_fit_evidence_recovery"]["retry_count"] == 0
    assert refit["post_fit_evidence_recovery"]["model_mutation"] is False


def test_report_validation_rejects_denominator_pooling_and_full_score_claims():
    report = build_report(CAMPAIGN)
    validate_report_documents(report)
    pooled = copy.deepcopy(report)
    pooled["evidence_boundary"]["pooled"] = True
    with pytest.raises(ReportError, match="pooled"):
        validate_report_documents(pooled)
    wrong = copy.deepcopy(report)
    wrong["historical_evaluations"][2]["row_count"] = 460
    with pytest.raises(ReportError, match="denominator"):
        validate_report_documents(wrong)
    full_claim = copy.deepcopy(report)
    full_claim["full_data_refit"]["validation_claims"] = [{"accuracy": 1.0}]
    with pytest.raises(ReportError, match="validation"):
        validate_report_documents(full_claim)


def test_markdown_uses_precise_retrospective_and_refit_language():
    text = render_report_markdown(build_report(CAMPAIGN))
    assert "309 / 460" in text
    assert "726 / 1,108" in text
    assert "202 / 307" in text
    assert "historical and selection-exposed" in text
    assert "not pooled" in text
    assert "does not establish untouched, external, or prospective performance" in text
    assert "No validation metric is claimed for the full-data refit" in text
    assert "RealMLP_r9_FULL is an Original clone fitted on 2,807 rows" in text

