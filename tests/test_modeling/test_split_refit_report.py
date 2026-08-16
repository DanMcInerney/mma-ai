from __future__ import annotations

import copy
import hashlib
import json
import shutil
from pathlib import Path

import pytest

from libs.modeling.split_refit_experiment.report import (
    ReportError,
    build_report,
    render_report_markdown,
    validate_report_documents,
    write_final_report,
)
from libs.modeling.split_refit_experiment.verification import (
    EvaluationVerificationError,
    validate_final_campaign,
    verify_report,
)
from libs.modeling.split_refit_experiment.__main__ import _parser


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

    unsafe = copy.deepcopy(report)
    unsafe["rollback"]["verification_commands"][0] = (
        "git -C C:/Users/danhm/mma-ai/mma-ai switch codex/weighted-v8-67-baseline"
    )
    with pytest.raises(ReportError, match="dirty main checkout"):
        validate_report_documents(unsafe)

    false_cut = copy.deepcopy(report)
    false_cut["branches"]["evaluation"]["direct_cut_from_rollback"] = True
    with pytest.raises(ReportError, match="direct-cut"):
        validate_report_documents(false_cut)


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
    assert "were not directly cut from the rollback revision" in text
    assert "does not switch production" in text
    assert "starts a new training run" in text


def test_rollback_instructions_select_only_the_immutable_worktree_and_name_profile_seam():
    rollback = build_report(CAMPAIGN)["rollback"]
    commands = "\n".join(
        [
            *rollback["selection_commands"],
            *rollback["verification_commands"],
            rollback["profile_verification_command"],
            rollback["training_invocation"],
        ]
    )
    assert "C:/Users/danhm/mma-ai/worktrees/weighted-v8-67-baseline" in commands
    assert "C:/Users/danhm/mma-ai/mma-ai" not in commands
    assert "rev-parse HEAD^{tree}" in commands
    assert "get_training_profile('v8-hybrid-weighted')" in commands
    assert "train_profile('v8-hybrid-weighted')" in commands
    assert rollback["verification_changes_production"] is False
    assert rollback["training_invocation_starts_new_fit"] is True


def _copy_report_inputs(destination: Path) -> Path:
    for relative in (
        "rollback-manifest.json",
        "partitions/manifest.json",
        "profiles/evaluation.json",
        "runs/80-10-10-evaluation/evaluation.json",
        "runs/80-10-10-evaluation/selection.json",
        "runs/80-10-10-evaluation/attempts.jsonl",
        "runs/80-10-10-evaluation/test-access.jsonl",
        "runs/80-10-10-evaluation/test-predictions.jsonl",
        "runs/full-data-refit/refit-lineage-correction.json",
        "runs/full-data-refit/refit.json",
        "runs/full-data-refit/fit-failure.json",
        "runs/full-data-refit/attempts.jsonl",
        "registry.jsonl",
        "registry-head.json",
    ):
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(CAMPAIGN / relative, target)
    registry_path = destination / "registry.jsonl"
    lines = [line.replace(b"\r\n", b"\n") for line in registry_path.read_bytes().splitlines(keepends=True)[:7]]
    prefix = b"".join(lines)
    registry_path.write_bytes(prefix)
    records = [json.loads(line) for line in lines]
    (destination / "registry-head.json").write_text(
        json.dumps(
            {
                "last_record_sha256": records[-1]["record_sha256"],
                "record_count": 7,
                "registry_bytes": len(prefix),
                "registry_prefix_sha256": hashlib.sha256(prefix).hexdigest().upper(),
            },
            separators=(",", ":"),
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return destination


def test_final_report_write_is_append_once_and_does_not_open_predictions_or_models(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    import libs.modeling.split_refit_experiment.report as report_module

    campaign = _copy_report_inputs(tmp_path / "campaign")
    opened: list[str] = []
    original = report_module._read_json

    def recording_read(path: Path):
        opened.append(Path(path).as_posix())
        return original(path)

    monkeypatch.setattr(report_module, "_read_json", recording_read)
    result = write_final_report(campaign)
    assert result["registry_record_id"] == "final-evidence-report"
    assert (campaign / "report.json").is_file()
    assert (campaign / "report.md").is_file()
    assert (campaign / "final-manifest.json").is_file()
    assert not any("prediction" in path or "model" in path or "access" in path for path in opened)
    with pytest.raises(ReportError, match="already exists"):
        write_final_report(campaign)


def test_strict_report_and_campaign_replay_recompute_registered_predictions(tmp_path: Path):
    campaign = _copy_report_inputs(tmp_path / "campaign")
    write_final_report(campaign)
    report = verify_report(campaign, strict=True)
    assert report["status"] == "PASS"
    assert report["prediction_replay"]["correct_count"] == 202
    assert report["prediction_replay"]["row_count"] == 307
    campaign_result = validate_final_campaign(campaign, strict=True)
    assert campaign_result["status"] == "PASS"
    assert campaign_result["registry"]["record_count"] == 8

    document = (campaign / "report.json").read_text(encoding="utf-8")
    (campaign / "report.json").write_text(document.replace("202,", "201,", 1), encoding="utf-8")
    with pytest.raises(EvaluationVerificationError):
        verify_report(campaign, strict=True)


def test_cli_exposes_append_once_report_and_strict_final_verifiers():
    parser = _parser()
    assert parser.parse_args(["write-report", "--campaign", "x"]).command == "write-report"
    report = parser.parse_args(["verify-report", "--campaign", "x", "--strict"])
    assert report.command == "verify-report" and report.strict is True
    branches = parser.parse_args(["verify-branches", "--campaign", "x", "--strict"])
    assert branches.command == "verify-branches" and branches.strict is True
    handoffs = parser.parse_args(
        ["verify-artifact-handoffs", "--campaign", "x", "--strict"]
    )
    assert handoffs.command == "verify-artifact-handoffs" and handoffs.strict is True
    final = parser.parse_args(["validate", "--campaign", "x", "--strict"])
    assert final.through == "final"
