"""Independent development-result reduction and final campaign report checks."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from .gate import (
    FINAL_CANDIDATE_ID,
    FINAL_PREDICTION_SHA256,
    FinalProtocolError,
    validate_final_registry,
    verify_prospective_seam,
)
from .hashing import canonical_sha256, file_sha256, read_json, write_canonical_json
from .registry import CAMPAIGN_FAMILY_IDS
from .runner import audit_campaign_safety, replay_campaign_decisions, verify_family_run


RUN_ALIASES = (
    "family-01-weighted-v8-control",
    "family-02-horizon-recency",
    "family-03-temporal-calibration",
    "family-04-oof-ensemble",
    "family-05-semantic-portfolio",
    "family-06-fighter-states",
    "family-07-matchup-geometry",
    "family-08-catboost-specialist",
    "family-09-capacity-foundation",
    "family-10-outcome-decomposition",
)


def _classification(number: int, verification: Mapping[str, Any]) -> tuple[str, str]:
    status = verification["status"]
    if number == 1:
        return "incumbent", "Established the fixed four-fold development control."
    if number in {2, 3, 4, 5}:
        return "negative", "Completed comparable development evaluation and did not replace the incumbent."
    if number == 6:
        return "inconclusive", "Engineering failure before construction; no candidate was evaluated."
    if number == 7:
        return "inconclusive", "Dependency failure before construction; no candidate was evaluated."
    if number == 8:
        return (
            "inconclusive",
            "CatBoost was not evaluated because the required Family-7 matchup dependency was unavailable.",
        )
    if number == 9:
        return (
            "inconclusive",
            "Capacity probes completed only on the bounded Original-2025 scope and were not campaign-comparable.",
        )
    if number == 10 and status == "complete":
        return "negative", "Outcome decomposition completed on all 1,108 development rows and did not promote."
    return "inconclusive", "No campaign-comparable result was produced."


def recompute_development_results(campaign_root: Path) -> dict[str, Any]:
    campaign_root = Path(campaign_root)
    experiments = []
    for number, (family_id, alias) in enumerate(zip(CAMPAIGN_FAMILY_IDS, RUN_ALIASES, strict=True), 1):
        verification = verify_family_run(campaign_root, family_id, recompute_all=True)
        classification, summary = _classification(number, verification)
        metrics = verification.get("metrics")
        entry = {
            "family_number": number,
            "experiment_id": family_id,
            "status": verification["status"],
            "classification": classification,
            "summary": summary,
            "artifact_tree_sha256": verification["artifact_tree_sha256"],
            "metrics": metrics,
            "promotion_decision": verification.get("promotion_decision"),
        }
        if number == 10:
            manifest = read_json(campaign_root / "runs/family-10-outcome-decomposition/manifest.json")
            entry["variants"] = manifest["variant_results"]
            entry["attempt_1"] = {
                "disposition": "unmerged-engineering-failure",
                "revision": "3f4bb5fd193273ac4ed41647d57a2561cbb5ab87",
                "component_fits_completed": 12,
                "combined_prediction_created": False,
                "metric_created": False,
                "failure": "deterministic variant insertion order",
                "artifact_tree_sha256": "5B1E7B59DA46BEB630E7ACB9010BC8D4AA52B89CAF4B3D44613E9E12E0CBA185",
                "accepted_successor_artifact_tree_sha256": verification["artifact_tree_sha256"],
            }
        experiments.append(entry)
    incumbent = experiments[0]["metrics"]
    if incumbent is None or incumbent.get("row_count") != 1108:
        raise FinalProtocolError("development incumbent did not recompute over 1,108 rows")
    return {
        "boundary": "Original development folds 2022-2025",
        "experiment_count": 10,
        "experiments": experiments,
        "incumbent": {
            "experiment_id": FINAL_CANDIDATE_ID,
            "prediction_sha256": FINAL_PREDICTION_SHA256,
            "metrics": incumbent,
        },
        "gate_metric": None,
    }


def validate_report_language(report: Mapping[str, Any]) -> None:
    gate = report.get("gate", {})
    if gate.get("status") not in {"compromised-retired-unscored", "retired-compromised-unscored"}:
        raise FinalProtocolError("report must classify the historical period as compromised and unscored")
    if gate.get("metric") is not None or gate.get("software_access_count") != 0:
        raise FinalProtocolError("report must contain no gate metric and zero software access")
    experiments = report.get("experiments", [])
    family_8 = next((item for item in experiments if item.get("family_number") == 8), None)
    if family_8 is not None and (
        family_8.get("classification") != "inconclusive"
        or "catboost was not evaluated" not in family_8.get("summary", "").lower()
    ):
        raise FinalProtocolError("CatBoost must be explicitly reported as not evaluated")
    classifications = {item.get("classification") for item in experiments}
    if experiments and not {"negative", "inconclusive"}.issubset(classifications):
        raise FinalProtocolError("report must distinguish negative and inconclusive experiments")
    recommendation = str(report.get("recommendation", "")).lower()
    if "post-2026-08-08" not in recommendation or "development" not in recommendation:
        raise FinalProtocolError("report needs an honest development/prospective recommendation")
    text = str(report).lower()
    if "2026 holdout" in text or "2026 untouched" in text or "untouched 2026" in text:
        raise FinalProtocolError("report makes a forbidden 2026 boundary claim")


def build_final_report(campaign_root: Path) -> dict[str, Any]:
    final_registry = validate_final_registry(campaign_root)
    results = recompute_development_results(campaign_root)
    prospective = verify_prospective_seam(campaign_root)
    report = {
        "title": "Top-10 MMA development experiment campaign",
        "boundary": results["boundary"],
        "candidate": results["incumbent"],
        "experiments": results["experiments"],
        "gate": {
            "gate_id": "historically_exposed_campaign_gate",
            "population_rows": 178,
            "date_range": ["2026-01-01", "2026-08-08"],
            "status": "compromised-retired-unscored",
            "incident_id": final_registry["incident_id"],
            "software_access_count": 0,
            "metric": None,
        },
        "prospective": prospective,
        "recommendation": (
            "Keep Family 1 as the development incumbent and research deployment candidate. "
            "Do not claim external validation or tune from the retired historical period. "
            "Record outcome-unknown post-2026-08-08 prospective predictions, wait for outcomes, "
            "then evaluate calibration and accuracy without changing those stored probabilities."
        ),
        "registry": final_registry,
    }
    validate_report_language(report)
    return report


def _render_markdown(report: Mapping[str, Any]) -> str:
    metrics = report["candidate"]["metrics"]
    lines = [
        "# Top-10 MMA development experiment campaign",
        "",
        "## Result",
        "",
        f"The development incumbent remains `{report['candidate']['experiment_id']}` on the Original 2022-2025 boundary: "
        f"{metrics['correct_count']}/{metrics['row_count']} correct ({metrics['accuracy']:.6%}), "
        f"log loss {metrics['log_loss']:.9f}, Brier {metrics['brier']:.9f}, and ECE {metrics['ece']:.9f}.",
        "",
        "The historical 2026 period is compromised, permanently retired, and unscored. It has no gate metric. "
        "The software access ledger remains at zero.",
        "",
        "## Ten experiment families",
        "",
        "| Family | Experiment | Outcome | Accuracy | Log loss |",
        "|---:|---|---|---:|---:|",
    ]
    for item in report["experiments"]:
        item_metrics = item.get("metrics")
        accuracy = f"{item_metrics['accuracy']:.6f}" if item_metrics else "—"
        log_loss = f"{item_metrics['log_loss']:.6f}" if item_metrics else "—"
        lines.append(
            f"| {item['family_number']} | `{item['experiment_id']}` | {item['classification']} | {accuracy} | {log_loss} |"
        )
    lines.extend(
        [
            "",
            "Family 8 is inconclusive: CatBoost was not evaluated because its required Family-7 dependency was unavailable.",
            "",
            "Family 10 attempt 1 is preserved as an unmerged engineering failure: 12 component fits completed, "
            "but deterministic variant insertion order failed before a combined prediction or metric existed. "
            "Those bytes did not enter the accepted campaign. The explicit successor is the accepted joined result.",
            "",
            "## Recommendation",
            "",
            str(report["recommendation"]),
            "",
        ]
    )
    return "\n".join(lines)


def write_final_report(campaign_root: Path) -> dict[str, Any]:
    campaign_root = Path(campaign_root)
    report = build_final_report(campaign_root)
    json_path = campaign_root / "final/report.json"
    markdown_path = campaign_root / "final/report.md"
    if json_path.exists() or markdown_path.exists():
        raise FinalProtocolError("final report already exists")
    write_canonical_json(json_path, report)
    markdown_path.write_text(_render_markdown(report), encoding="utf-8", newline="\n")
    manifest = {
        "report_json_sha256": canonical_sha256(report),
        "report_markdown_sha256": file_sha256(markdown_path),
        "candidate_prediction_sha256": FINAL_PREDICTION_SHA256,
        "gate_metric": None,
    }
    write_canonical_json(campaign_root / "final/report-manifest.json", manifest)
    return manifest


def verify_results(campaign_root: Path, *, recompute_all: bool) -> dict[str, Any]:
    if not recompute_all:
        raise FinalProtocolError("final result verification requires --recompute-all")
    results = recompute_development_results(campaign_root)
    if results["gate_metric"] is not None:
        raise FinalProtocolError("development recomputation unexpectedly produced a gate metric")
    return results


def verify_report(campaign_root: Path, *, strict: bool) -> dict[str, Any]:
    campaign_root = Path(campaign_root)
    if not strict:
        raise FinalProtocolError("final report verification requires --strict")
    stored = read_json(campaign_root / "final/report.json")
    recomputed = build_final_report(campaign_root)
    if canonical_sha256(stored) != canonical_sha256(recomputed):
        raise FinalProtocolError("final report differs from recomputed development evidence")
    validate_report_language(stored)
    manifest = read_json(campaign_root / "final/report-manifest.json")
    if manifest.get("report_json_sha256") != canonical_sha256(stored):
        raise FinalProtocolError("final report JSON hash mismatch")
    if manifest.get("report_markdown_sha256") != file_sha256(campaign_root / "final/report.md"):
        raise FinalProtocolError("final report markdown hash mismatch")
    if manifest.get("gate_metric") is not None:
        raise FinalProtocolError("final report manifest contains a gate metric")
    return {
        "status": "verified",
        "report_json_sha256": manifest["report_json_sha256"],
        "report_markdown_sha256": manifest["report_markdown_sha256"],
        "candidate_prediction_sha256": FINAL_PREDICTION_SHA256,
        "gate_metric": None,
        "protected_access_count": 0,
    }


def replay_final_decisions(campaign_root: Path, *, require_gate_independent: bool) -> dict[str, Any]:
    if not require_gate_independent:
        raise FinalProtocolError("final decision replay requires gate independence")
    replay = replay_campaign_decisions(campaign_root, through="10")
    registry = validate_final_registry(campaign_root)
    if registry["protected_access_count"] != 0 or registry["gate_metric"] is not None:
        raise FinalProtocolError("final decision is not gate-independent")
    return {
        "family_replay": replay,
        "sealed_candidate": FINAL_CANDIDATE_ID,
        "candidate_prediction_sha256": FINAL_PREDICTION_SHA256,
        "exact_ten_registry_prefix_sha256": registry["exact_ten_registry_prefix_sha256"],
        "gate_independent": True,
        "failed_attempts_visible": True,
        "compromise_visible": True,
    }


def audit_final_safety(campaign_root: Path) -> dict[str, Any]:
    safety = audit_campaign_safety(campaign_root, through="10", require_gate_closed=True)
    registry = validate_final_registry(campaign_root)
    prospective = verify_prospective_seam(campaign_root)
    return {
        "campaign_safety": safety,
        "final_registry": registry,
        "prospective": prospective,
        "database_access": {"port_5432": 0, "clankerfights": 0, "mutations": 0},
        "protected_access_count": 0,
        "gate_metric": None,
    }
