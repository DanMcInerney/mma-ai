"""Nested joint horizon and recency experiment over development folds only."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from datetime import date
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from ..hashing import canonical_sha256, file_sha256, read_json, write_canonical_json
from ..protocol import AccessLedger
from ..registry import (
    CAMPAIGN_FAMILY_IDS,
    FAMILY_2_VARIANT_IDS,
    validate_resolved_profile,
)


EXPERIMENT_ID = CAMPAIGN_FAMILY_IDS[1]
OUTER_YEARS = (2022, 2023, 2024, 2025)
SEEDS = {"python": 20260815, "numpy": 20260815, "torch": 20260815, "bootstrap": 20260815}
EXPECTED_REGISTRY_PREFIX = "B4F6FEE4AE5C2EDE6055684AC26D8A6426D02C8DB0920BB482B09750587C4279"
FIXED_SOURCE_ARTIFACT = Path(
    r"C:\Users\danhm\mma-ai\worktrees\top10-20260815"
    r"\experiments\top10_20260815\artifacts\01-campaign-harness"
)
FIXED_INCUMBENT_ARTIFACT = Path(
    r"C:\Users\danhm\mma-ai\worktrees\top10-20260815"
    r"\experiments\top10_20260815\artifacts\02-family-01-weighted-v8-control"
)


JOINT_VARIANTS: list[dict[str, Any]] = [
    {
        "id": "expanding-decay-0",
        "horizon": {"kind": "expanding", "years": None, "start_date": "2014-01-01"},
        "weight_scheme": "exponential-date",
        "decay_rate": 0.0,
        "formula": "I[event_date < as_of_date]",
        "half_life_years": None,
    },
    {
        "id": "expanding-decay-0.05",
        "horizon": {"kind": "expanding", "years": None, "start_date": "2014-01-01"},
        "weight_scheme": "exponential-date",
        "decay_rate": 0.05,
        "formula": "exp(-0.05 * age_days / 365.25) * I[event_date < as_of_date]",
        "half_life_years": math.log(2) / 0.05,
    },
    {
        "id": "rolling-8y-decay-0.10",
        "horizon": {"kind": "rolling-calendar-years", "years": 8, "start_date": None},
        "weight_scheme": "exponential-date",
        "decay_rate": 0.10,
        "formula": "exp(-0.10 * age_days / 365.25) * I[event_date >= as_of_date - 8 calendar years]",
        "half_life_years": math.log(2) / 0.10,
    },
    {
        "id": "rolling-6y-decay-0.15",
        "horizon": {"kind": "rolling-calendar-years", "years": 6, "start_date": None},
        "weight_scheme": "exponential-date",
        "decay_rate": 0.15,
        "formula": "exp(-0.15 * age_days / 365.25) * I[event_date >= as_of_date - 6 calendar years]",
        "half_life_years": math.log(2) / 0.15,
    },
    {
        "id": "rolling-4y-decay-0.25",
        "horizon": {"kind": "rolling-calendar-years", "years": 4, "start_date": None},
        "weight_scheme": "exponential-date",
        "decay_rate": 0.25,
        "formula": "exp(-0.25 * age_days / 365.25) * I[event_date >= as_of_date - 4 calendar years]",
        "half_life_years": math.log(2) / 0.25,
    },
    {
        "id": "expanding-piecewise-event-count",
        "horizon": {"kind": "expanding", "years": None, "start_date": "2014-01-01"},
        "weight_scheme": "piecewise-event-count",
        "decay_rate": 0.0,
        "formula": "event_rank<=25:1; <=75:0.75; <=150:0.5; older:0.25",
        "event_rank_cutoffs": [25, 75, 150],
        "event_rank_weights": [1.0, 0.75, 0.5, 0.25],
        "event_count_interpretation": "rank distinct prior events newest-first; all fights on one event share one rank",
        "half_life_years": None,
    },
    {
        "id": "rolling-8y-decay-0",
        "horizon": {"kind": "rolling-calendar-years", "years": 8, "start_date": None},
        "weight_scheme": "exponential-date",
        "decay_rate": 0.0,
        "formula": "I[event_date >= as_of_date - 8 calendar years]",
        "half_life_years": None,
    },
    {
        "id": "expanding-decay-0.15",
        "horizon": {"kind": "expanding", "years": None, "start_date": "2014-01-01"},
        "weight_scheme": "exponential-date",
        "decay_rate": 0.15,
        "formula": "exp(-0.15 * age_days / 365.25) * I[event_date < as_of_date]",
        "half_life_years": math.log(2) / 0.15,
    },
]


def materialized_profile() -> dict[str, Any]:
    from libs.modeling.training_profiles import WIN_V8_HYBRID_WORKING_PROFILE

    base = dict(WIN_V8_HYBRID_WORKING_PROFILE)
    base["features"] = list(base["features"])
    base["time_limit"] = 420
    base["use_recency_weights"] = False
    base["decay_rate"] = 0.0
    base["refit_full"] = False
    base["calculate_importance"] = False
    profile = {
        "experiment_id": EXPERIMENT_ID,
        "family_number": 2,
        "base_training_profile": base,
        "joint_variants": JOINT_VARIANTS,
        "outer_years": list(OUTER_YEARS),
        "embargo_days": 7,
        "selection_metric": "positive-log-loss",
        "selection_tie_break": list(FAMILY_2_VARIANT_IDS),
        "selection_evidence": "chronological-inner-only",
        "per_fit_time_cap_seconds": 480,
        "family_deadline_seconds": 14400,
        "early_stop_rule": (
            "Launch every preregistered variant once per fold in frozen order; after the first "
            "failed, timed-out, or cancelled fit, launch no further fit and preserve terminal evidence."
        ),
        "seeds": SEEDS,
        "bootstrap": {"iterations": 2000, "seed": SEEDS["bootstrap"], "block": "event_id"},
        "promotion_rule": (
            "Promote only when all four outer folds complete and pooled candidate log loss is below "
            "the aligned family-1 incumbent with a paired event-block log-loss interval upper bound "
            "below zero; otherwise retain family 1."
        ),
        "adaptive_emphasis": (
            "Family 1's 2025 degradation and fold-varying ensemble weights prioritize drift reporting; "
            "the frozen menu order and membership remain unchanged."
        ),
    }
    validate_family_profile(profile)
    return profile


def validate_family_profile(profile: Mapping[str, Any]) -> str:
    return validate_resolved_profile(profile)


def _subtract_calendar_years(value: date, years: int) -> date:
    try:
        return value.replace(year=value.year - years)
    except ValueError:
        return value.replace(year=value.year - years, day=28)


def compute_training_weights(
    rows: Iterable[Mapping[str, Any]],
    variant: Mapping[str, Any],
    *,
    as_of_date: date,
) -> list[float]:
    rows = list(rows)
    event_dates = [date.fromisoformat(str(row["event_date"])[:10]) for row in rows]
    if any(observed >= as_of_date for observed in event_dates):
        raise ValueError("training weights require prior-only event metadata")
    years = variant["horizon"]["years"]
    cutoff = _subtract_calendar_years(as_of_date, int(years)) if years is not None else None
    admitted = [cutoff is None or observed >= cutoff for observed in event_dates]
    if variant["weight_scheme"] == "exponential-date":
        rate = float(variant["decay_rate"])
        return [
            math.exp(-rate * (as_of_date - observed).days / 365.25) if keep else 0.0
            for observed, keep in zip(event_dates, admitted, strict=True)
        ]
    if variant["weight_scheme"] != "piecewise-event-count":
        raise ValueError("unknown weighting scheme")
    ordered_events = sorted(
        {str(row["event_id"]): observed for row, observed in zip(rows, event_dates, strict=True)}.items(),
        key=lambda item: (item[1], item[0]),
        reverse=True,
    )
    ranks = {event_id: rank for rank, (event_id, _) in enumerate(ordered_events, start=1)}
    cutoffs = variant["event_rank_cutoffs"]
    values = variant["event_rank_weights"]
    result = []
    for row, keep in zip(rows, admitted, strict=True):
        rank = ranks[str(row["event_id"])]
        bucket = next((index for index, boundary in enumerate(cutoffs) if rank <= boundary), len(cutoffs))
        result.append(float(values[bucket]) if keep else 0.0)
    return result


def select_joint_variant(
    scores: Sequence[Mapping[str, Any]],
    *,
    outer_min_date: date,
    embargo_days: int,
) -> dict[str, Any]:
    for score in scores:
        if score.get("partition") != "inner-validation":
            raise ValueError("joint selection accepts inner-validation evidence only")
        selection_max = date.fromisoformat(str(score["selection_max_date"])[:10])
        if (outer_min_date - selection_max).days < embargo_days:
            raise ValueError("inner selection violates the outer embargo")
        if score.get("variant_id") not in FAMILY_2_VARIANT_IDS:
            raise ValueError("inner evidence names an unregistered variant")
    if {score["variant_id"] for score in scores} != set(FAMILY_2_VARIANT_IDS):
        raise ValueError("inner selection requires one score for every frozen variant")
    order = {variant_id: index for index, variant_id in enumerate(FAMILY_2_VARIANT_IDS)}
    selected = min(scores, key=lambda score: (float(score["log_loss"]), order[score["variant_id"]]))
    return {
        "variant_id": selected["variant_id"],
        "inner_log_loss": float(selected["log_loss"]),
        "selection_max_date": selected["selection_max_date"],
        "selection_basis": "chronological-inner-log-loss",
    }


def _registry_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def preregister(campaign_root: Path) -> dict[str, Any]:
    campaign_root = Path(campaign_root)
    profile_path = campaign_root / "profiles" / f"{EXPERIMENT_ID}.json"
    run_root = campaign_root / "runs" / EXPERIMENT_ID
    prereg_path = run_root / "preregistration.json"
    attempts_path = run_root / "attempts.jsonl"
    for path in (profile_path, prereg_path, attempts_path):
        if path.exists():
            raise ValueError(f"refusing to overwrite preregistration artifact: {path}")
    if _registry_sha(campaign_root / "registry.jsonl") != EXPECTED_REGISTRY_PREFIX:
        raise ValueError("family-1 registry prefix is not the frozen dependency")
    gate = AccessLedger(campaign_root).gate_status()
    if gate["state"] != "closed" or gate["protected_access_count"] != 0:
        raise ValueError("preregistration requires the gate closed with zero access")
    if not FIXED_SOURCE_ARTIFACT.is_dir() or not FIXED_INCUMBENT_ARTIFACT.is_dir():
        raise ValueError("fixed read-only campaign artifacts are unavailable")

    profile = materialized_profile()
    profile_sha = write_canonical_json(profile_path, profile)
    family_1_manifest = campaign_root / "runs/family-01-weighted-v8-control/manifest.json"
    preregistration = {
        "experiment_id": EXPERIMENT_ID,
        "family_number": 2,
        "hypothesis": (
            "Jointly selecting horizon and prior-only recency weighting on chronological inner evidence "
            "will reduce the temporal drift observed in the family-1 development control."
        ),
        "variant_bound": 8,
        "variant_menu": list(FAMILY_2_VARIANT_IDS),
        "profile_path": profile_path.relative_to(campaign_root).as_posix(),
        "profile_sha256": profile_sha,
        "outer_years": list(OUTER_YEARS),
        "embargo_days": profile["embargo_days"],
        "selection_evidence": profile["selection_evidence"],
        "selection_metric": profile["selection_metric"],
        "selection_boundary": "Original",
        "same_row_or_outer_selection_admissible": False,
        "gate_state_required": "closed",
        "source_artifact_mode": "fixed-read-only-campaign-artifacts",
        "source_artifact_path": str(FIXED_SOURCE_ARTIFACT),
        "incumbent_artifact_path": str(FIXED_INCUMBENT_ARTIFACT),
        "incumbent_artifact_tree_sha256": "B2E83125540C7DACF6B1138C9E2C5DEB0DEE0C619217C472D6EA76D5B482BA09",
        "artifact_path": "artifacts/03-family-02-horizon-recency",
        "registry_prefix_sha256_before": EXPECTED_REGISTRY_PREFIX,
        "family_1_manifest_file_sha256": file_sha256(family_1_manifest),
        "seeds": SEEDS,
        "per_fit_time_cap_seconds": profile["per_fit_time_cap_seconds"],
        "family_deadline_seconds": profile["family_deadline_seconds"],
        "early_stop_rule": profile["early_stop_rule"],
        "invocation": (
            "uv run python -m libs.modeling.experiment_campaign.families.horizon_recency "
            "launch --campaign experiments/top10_20260815"
        ),
        "promotion_rule": profile["promotion_rule"],
        "adaptive_emphasis": profile["adaptive_emphasis"],
    }
    write_canonical_json(prereg_path, preregistration)
    attempts_path.parent.mkdir(parents=True, exist_ok=True)
    attempts_path.write_bytes(b"")
    (run_root / "decision.md").write_bytes(
        b"# Family 2 preregistration\n\n"
        b"Eight frozen joint horizon/recency variants; chronological inner selection only; "
        b"four Original outer folds; 2026 gate closed.\n"
    )
    return preregistration


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)
    pre = commands.add_parser("preregister")
    pre.add_argument("--campaign", type=Path, required=True)
    return parser


def main() -> int:
    args = _parser().parse_args()
    result = preregister(args.campaign)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
