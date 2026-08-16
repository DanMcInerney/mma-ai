"""Evidence-only final report for the chronological split/refit campaign."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from .protocol import canonical_json_bytes, canonical_sha256, file_sha256


ROLLBACK_REVISION = "545441975b86caf0abb6136e099e44e6b93caf22"
EVALUATION_REVISION = "7217012abcee3c22937dd378c0a904033564018d"
FULL_REFIT_REVISION = "70559ac40300c62067f23b335050dda3e4931ce6"
EXPECTED_DENOMINATORS = (460, 1108, 307)


class ReportError(ValueError):
    pass


def _read_json(path: Path) -> dict[str, Any]:
    try:
        raw = path.read_bytes()
        value = json.loads(raw)
    except (FileNotFoundError, json.JSONDecodeError) as exc:
        raise ReportError(f"missing or invalid report input: {path}") from exc
    if not isinstance(value, dict):
        raise ReportError(f"report input is not an object: {path}")
    canonical = canonical_json_bytes(value)
    if raw not in (canonical, canonical + b"\n", canonical + b"\r\n"):
        raise ReportError(f"report input is not canonical JSON: {path}")
    return value


def _evaluation_row(
    *,
    name: str,
    boundary: str,
    metrics: Mapping[str, Any],
    exposed_degree: str,
) -> dict[str, Any]:
    return {
        "name": name,
        "boundary": boundary,
        "evidence_status": "historical-retrospective",
        "selection_exposure": exposed_degree,
        "correct_count": metrics["correct_count"],
        "row_count": metrics["row_count"],
        "accuracy": metrics["accuracy"],
        "positive_log_loss": metrics.get("positive_log_loss", metrics.get("log_loss")),
        **{
            key: metrics[key]
            for key in (
                "brier",
                "ece",
                "calibration_intercept",
                "calibration_slope",
            )
            if key in metrics
        },
    }


def build_report(campaign_root: Path) -> dict[str, Any]:
    """Build claims only from small tracked manifests; never load models or source data."""
    root = Path(campaign_root)
    rollback = _read_json(root / "rollback-manifest.json")
    partitions = _read_json(root / "partitions/manifest.json")
    evaluation = _read_json(root / "runs/80-10-10-evaluation/evaluation.json")
    selection = _read_json(root / "runs/80-10-10-evaluation/selection.json")
    refit = _read_json(root / "runs/full-data-refit/refit-lineage-correction.json")
    failure = _read_json(root / "runs/full-data-refit/fit-failure.json")
    registry_head = _read_json(root / "registry-head.json")

    accepted = rollback["evaluation"]["accepted_direct_validation"]
    nested = rollback["evaluation"]["nested_historical"]["metrics"]
    direct = evaluation["metrics"]
    lineage = refit["lineage"]
    realmlp = lineage["RealMLP_r9_FULL"]
    full_ensemble = lineage["WeightedEnsemble_L2_FULL"]
    graph_by_name = {node["name"]: node for node in refit["model_graph"]}
    ensemble_graph = graph_by_name["WeightedEnsemble_L2_FULL"]

    report = {
        "schema_version": 1,
        "campaign": "split_refit_20260816",
        "evidence_boundary": {
            "historically_exposed": True,
            "selection_exposed_to_different_degrees": True,
            "pooled": False,
            "untouched_or_external_claim": False,
            "prospective_claim": False,
            "next_untouched_boundary": "outcome-unknown fights after 2026-08-08 predicted before results",
        },
        "historical_evaluations": [
            _evaluation_row(
                name="accepted-direct-tuning",
                boundary=accepted["boundary"],
                metrics=accepted,
                exposed_degree="accepted tuning/selection evidence",
            ),
            _evaluation_row(
                name="nested-whole-event-2022-2025",
                boundary=rollback["evaluation"]["nested_historical"]["boundary"],
                metrics=nested,
                exposed_degree="repeated historical model-development evidence",
            ),
            _evaluation_row(
                name="one-shot-chronological-test-2024-2025",
                boundary="retrospective whole-event chronological test",
                metrics=direct,
                exposed_degree="one-shot after frozen selection, but outcomes historically exposed",
            ),
        ],
        "one_shot_uncertainty": evaluation["event_block_intervals"],
        "evaluation_selection": {
            "selected_node": selection["selected_node"],
            "dependencies": selection["ensemble_dependencies"],
            "weights": selection["ensemble_weights"],
            "model_tree_sha256": evaluation["model_tree_sha256"],
            "prediction_sha256": evaluation["prediction_sha256"],
            "post_test_adaptation": evaluation["post_test_adaptation"],
        },
        "split": {
            "eligible_rows": partitions["eligible_row_count"],
            "development_rows": partitions["development_row_count"],
            "train_rows": partitions["partitions"]["train"]["row_count"],
            "validation_rows": partitions["partitions"]["validation"]["row_count"],
            "test_rows": partitions["partitions"]["test"]["row_count"],
            "retired_rows": partitions["retired_row_count"],
        },
        "full_data_refit": {
            "profile_name": refit["profile_name"],
            "source_rows": refit["source_rows"],
            "feature_count": refit["feature_count"],
            "selected_node": refit["selected_node"],
            "node_count": len(lineage),
            "fresh_full_base_count": sum(
                node["origin"] == "fresh-full-fit" and node["fit_rows"] == 3267
                for node in lineage.values()
            ),
            "realmlp_full": {
                "origin": realmlp["origin"],
                "fit_rows": realmlp["fit_rows"],
            },
            "full_ensemble": {
                "dependencies": ensemble_graph["dependencies"],
                "weights": ensemble_graph["weights"],
                "fit_rows": full_ensemble["fit_rows"],
                "effective_fit_rows": full_ensemble["effective_fit_rows"],
            },
            "validation_claims": refit["validation_claims"],
            "complete_tree_sha256": refit["complete_tree"]["sha256"],
            "native_tree_sha256": refit["native_tree"]["sha256"],
            "process_exit_code": failure["attempt"]["exit_code"],
            "post_fit_evidence_recovery": refit["post_fit_evidence_recovery"],
        },
        "branches": {
            "rollback": {
                "name": "codex/weighted-v8-67-baseline",
                "revision": ROLLBACK_REVISION,
            },
            "evaluation": {
                "name": "codex/exp-80-10-10-v8-20260816",
                "revision": EVALUATION_REVISION,
                "merge_base": ROLLBACK_REVISION,
            },
            "full_refit": {
                "name": "codex/exp-full-refit-v8-20260816",
                "revision": FULL_REFIT_REVISION,
                "merge_base": ROLLBACK_REVISION,
            },
        },
        "rollback": {
            "named_profile": "v8-hybrid-weighted",
            "named_seam": 'libs.modeling.training_profiles.train_profile("v8-hybrid-weighted")',
            "commands": [
                "git -C C:/Users/danhm/mma-ai/mma-ai rev-parse codex/weighted-v8-67-baseline",
                "git -C C:/Users/danhm/mma-ai/worktrees/weighted-v8-67-baseline rev-parse HEAD",
                "git -C C:/Users/danhm/mma-ai/worktrees/weighted-v8-67-baseline status --short",
            ],
            "expected_revision": ROLLBACK_REVISION,
        },
        "decision": {
            "recommendation": "retain-rollback",
            "current_reasons": [
                "all three evaluation results are historical and selection-exposed to different degrees",
                "the 307-row one-shot estimate is consistent with the nested estimate and does not exceed the accepted tuning estimate",
                "the one-shot event-block interval overlaps both earlier estimates",
                "the full-data refit has no admissible validation metric",
            ],
            "retain_if_any": [
                "only historical or outcome-exposed evidence is available",
                "the candidate has no admissible external/prospective evaluation",
                "the candidate requires a test-guided profile, feature, weight, calibration, or threshold change",
                "rollback, source, feature, profile, or model identities fail replay",
            ],
            "replace_if_all": [
                "a successor decision rule is preregistered before outcome access",
                "evaluation uses outcome-unknown whole-event fights after 2026-08-08",
                "the successor is evaluated once without post-result adaptation",
                "the preregistered accuracy and positive-log-loss superiority/noninferiority criteria both pass",
                "all source/profile/feature/model/prediction identities and rollback preservation replay",
            ],
            "full_refit_disposition": "preserve as a loadable non-validated deployment candidate; do not promote from FULL evidence",
        },
        "source_identities": {
            "source_sha256": rollback["source_identities"]["frozen_source_csv"]["sha256"],
            "feature_sha256": rollback["reproduction"]["features"]["ordered_sha256"],
            "profile_sha256": rollback["reproduction"]["profile"]["canonical_sha256"],
            "rollback_model_tree_sha256": rollback["model_identity"]["complete_tree_sha256"],
        },
        "registry_input": {
            "record_count": registry_head["record_count"],
            "prefix_sha256": registry_head["registry_prefix_sha256"],
            "last_record_sha256": registry_head["last_record_sha256"],
        },
    }
    validate_report_documents(report)
    return report


def validate_report_documents(report: Mapping[str, Any]) -> None:
    historical = report.get("historical_evaluations")
    if not isinstance(historical, list) or len(historical) != 3:
        raise ReportError("exactly three separate historical evaluations are required")
    denominators = tuple(row.get("row_count") for row in historical)
    numerators = tuple(row.get("correct_count") for row in historical)
    if denominators != EXPECTED_DENOMINATORS or numerators != (309, 726, 202):
        raise ReportError("historical denominator or correct-count boundary changed")
    if any(row.get("evidence_status") != "historical-retrospective" for row in historical):
        raise ReportError("historical retrospective label is missing")
    boundary = report.get("evidence_boundary", {})
    if boundary.get("pooled") is not False:
        raise ReportError("historical results must not be pooled")
    if boundary.get("untouched_or_external_claim") is not False or boundary.get("prospective_claim") is not False:
        raise ReportError("report makes an untouched, external, or prospective claim")
    refit = report.get("full_data_refit", {})
    if refit.get("validation_claims") != []:
        raise ReportError("full-data refit makes a validation claim")
    if (
        refit.get("source_rows") != 3267
        or refit.get("node_count") != 22
        or refit.get("fresh_full_base_count") != 9
        or refit.get("realmlp_full") != {"origin": "original-clone", "fit_rows": 2807}
    ):
        raise ReportError("full-data refit lineage boundary changed")
    ensemble = refit.get("full_ensemble", {})
    if (
        ensemble.get("dependencies") != ["Mitra_FULL", "XGBoost_FULL"]
        or ensemble.get("fit_rows") != 460
        or ensemble.get("effective_fit_rows") != 3267
    ):
        raise ReportError("FULL ensemble dependency boundary changed")
    recovery = refit.get("post_fit_evidence_recovery", {})
    if (
        refit.get("process_exit_code") != 1
        or recovery.get("training_completed") is not True
        or recovery.get("refit_full_completed") is not True
        or recovery.get("failure_preserved") is not True
        or recovery.get("retry_count") != 0
        or recovery.get("model_mutation") is not False
    ):
        raise ReportError("post-fit failure/recovery order changed")
    decision = report.get("decision", {})
    if (
        decision.get("recommendation") != "retain-rollback"
        or not decision.get("retain_if_any")
        or not decision.get("replace_if_all")
    ):
        raise ReportError("retain/replace decision predicates are incomplete")


def render_report_markdown(report: Mapping[str, Any]) -> str:
    validate_report_documents(report)
    accepted, nested, direct = report["historical_evaluations"]
    refit = report["full_data_refit"]
    intervals = report["one_shot_uncertainty"]
    lines = [
        "# Chronological split and full-refit evidence report",
        "",
        "## Decision",
        "",
        "Retain the immutable weighted-v8 rollback as the production fallback. Preserve the fresh full-data refit as a loadable, non-validated deployment candidate; do not promote it from FULL evidence.",
        "",
        "## Historical evaluation evidence",
        "",
        "All three results are historical and selection-exposed to different degrees. Their denominators represent different protocols and are not pooled or treated as directly equivalent. This campaign does not establish untouched, external, or prospective performance.",
        "",
        "| protocol | correct / rows | accuracy | positive log loss |",
        "| --- | ---: | ---: | ---: |",
        f"| accepted direct tuning | {accepted['correct_count']} / {accepted['row_count']:,} | {accepted['accuracy']:.6f} | {accepted['positive_log_loss']:.6f} |",
        f"| nested whole-event 2022–2025 | {nested['correct_count']} / {nested['row_count']:,} | {nested['accuracy']:.6f} | {nested['positive_log_loss']:.6f} |",
        f"| one-shot chronological test | {direct['correct_count']} / {direct['row_count']:,} | {direct['accuracy']:.6f} | {direct['positive_log_loss']:.6f} |",
        "",
        f"The one-shot event-block 95% interval is [{intervals['accuracy']['lower']:.6f}, {intervals['accuracy']['upper']:.6f}] for accuracy and [{intervals['log_loss']['lower']:.6f}, {intervals['log_loss']['upper']:.6f}] for positive log loss.",
        "",
        "## Full-data refit boundary",
        "",
        f"The saved predictor loads with {refit['node_count']} nodes and selects `{refit['selected_node']}`. Nine FULL base nodes are fresh 3,267-row fits. RealMLP_r9_FULL is an Original clone fitted on 2,807 rows. The FULL ensemble wrapper retains 460-row metadata but has effective 3,267-row lineage through `Mitra_FULL` and `XGBoost_FULL` (weights 0.96/0.04).",
        "",
        "No validation metric is claimed for the full-data refit. The only production process exited 1 after training, `refit_full`, and permutation importance completed; the order-assertion failure is preserved, read-only recovery mutated no model, and retry count is zero.",
        "",
        "## Branches and rollback",
        "",
        f"- rollback: `codex/weighted-v8-67-baseline` at `{ROLLBACK_REVISION}`",
        f"- evaluation: `codex/exp-80-10-10-v8-20260816` at `{EVALUATION_REVISION}`",
        f"- full refit: `codex/exp-full-refit-v8-20260816` at `{FULL_REFIT_REVISION}`",
        "",
        "Verify the existing rollback worktree without touching the original dirty checkout:",
        "",
        "```powershell",
        *report["rollback"]["commands"],
        "```",
        "",
        "The first two commands must both print the fixed rollback revision above; the final command must print nothing.",
        "",
        "## Replacement predicate",
        "",
        "Keep the rollback if any retain predicate holds. Replace it only when every preregistered replace predicate passes on future outcome-unknown whole-event evidence, with no post-result adaptation and full identity replay.",
        "",
    ]
    return "\n".join(lines)


def report_manifest(campaign_root: Path, report: Mapping[str, Any], markdown: str) -> dict[str, Any]:
    root = Path(campaign_root)
    return {
        "schema_version": 1,
        "report_json_sha256": canonical_sha256(report),
        "report_markdown_sha256": canonical_sha256(markdown),
        "report_markdown_bytes_sha256": file_sha256(root / "report.md"),
        "registry_before_final_record": report["registry_input"],
        "branch_revisions": {
            "rollback": ROLLBACK_REVISION,
            "evaluation": EVALUATION_REVISION,
            "full_refit": FULL_REFIT_REVISION,
        },
    }

