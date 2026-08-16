"""Nested joint horizon and recency experiment over development folds only."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import sys
import time
from datetime import date
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

from ..hashing import (
    canonical_json_bytes,
    canonical_sha256,
    file_sha256,
    read_json,
    tree_inventory,
    write_canonical_json,
)
from ..metrics import event_block_bootstrap_delta, reduce_predictions
from ..protocol import AccessLedger
from ..registry import (
    CAMPAIGN_FAMILY_IDS,
    FAMILY_2_VARIANT_IDS,
    append_registry_record,
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


def _append_jsonl(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("ab") as handle:
        handle.write(canonical_json_bytes(dict(value)) + b"\n")
        handle.flush()
        os.fsync(handle.fileno())


def _write_jsonl(path: Path, values: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        for value in values:
            handle.write(canonical_json_bytes(dict(value)) + b"\n")


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines()]


def _variant(variant_id: str) -> dict[str, Any]:
    if variant_id not in FAMILY_2_VARIANT_IDS:
        raise ValueError("variant is outside the preregistered menu")
    return next(item for item in JOINT_VARIANTS if item["id"] == variant_id)


def _positive_probabilities(values: Any) -> np.ndarray:
    if hasattr(values, "columns"):
        series = values[1] if 1 in values.columns else values.iloc[:, -1]
        return np.asarray(series, dtype=float)
    return np.asarray(values, dtype=float)


def _prediction_rows(
    frame: pd.DataFrame,
    probabilities: np.ndarray,
    *,
    fold: str,
    fit_max_date: str,
    selection_max_date: str,
    fit_event_ids: list[str],
    experience: Mapping[str, str],
    selected_variant: str,
    boundary: str,
) -> list[dict[str, Any]]:
    rows = []
    for position, (_, row) in enumerate(frame.iterrows()):
        method = str(row["method"]).lower()
        rows.append({
            "fight_id": str(row["fight_id"]),
            "event_id": str(row["event_id"]),
            "event_date": row["event_date"].date().isoformat(),
            "y_true": int(row["y_true"]),
            "probability": float(probabilities[position]),
            "boundary": boundary,
            "fit_scope": "prior-only",
            "fold": fold,
            "weight_class": str(row["weightclass_encoded"]),
            "experience": experience[str(row["fight_id"])],
            "outcome_type": "decision" if "dec" in method else "finish",
            "fit_max_date": fit_max_date,
            "selection_max_date": selection_max_date,
            "context_max_date": selection_max_date,
            "fit_event_ids": fit_event_ids,
            "embargo_days": 7,
            "selected_variant": selected_variant,
        })
    return rows


def fit_variant(campaign_root: Path, artifact_root: Path, year: int, variant_id: str) -> None:
    from autogluon.tabular import TabularPredictor
    from sklearn.preprocessing import RobustScaler

    from libs.modeling.train import TrainingConfig, build_training_fit_kwargs, training_runtime_preflight

    from .weighted_v8 import (
        _experience_labels,
        _load_pre_gate_frame,
        _seed_runtime,
        validate_prediction_chronology,
    )

    if year not in OUTER_YEARS:
        raise ValueError("year is outside the frozen outer folds")
    variant = _variant(variant_id)
    _seed_runtime()
    runtime = training_runtime_preflight()
    profile = read_json(Path(campaign_root) / "profiles" / f"{EXPERIMENT_ID}.json")
    config = TrainingConfig(**profile["base_training_profile"])
    source_csv = FIXED_SOURCE_ARTIFACT / "models/accepted/training_data.csv"
    frame, roster = _load_pre_gate_frame(source_csv, config.features)
    frame["fight_id"] = frame["fight_id"].astype(str)
    frame["event_id"] = frame["event_id"].astype(str)
    fold_manifest = read_json(Path(campaign_root) / "baseline/fold-manifest.json")
    fold = next(item for item in fold_manifest["folds"] if item["test_year"] == year)
    inner_train = frame[frame["fight_id"].isin(set(fold["inner"]["train_fight_ids"]))].copy()
    inner_val = frame[frame["fight_id"].isin(set(fold["inner"]["validation_fight_ids"]))].copy()
    outer = frame[frame["fight_id"].isin(set(fold["outer"]["test_fight_ids"]))].copy()
    if min(len(inner_train), len(inner_val), len(outer)) == 0:
        raise ValueError("frozen fold membership is incomplete")
    outer_start = outer["event_date"].min()
    if inner_val["event_date"].max() > outer_start - pd.Timedelta(days=profile["embargo_days"]):
        raise ValueError("inner selection rows cross the outer embargo")

    raw_weights = compute_training_weights(
        inner_train[["event_id", "event_date"]].to_dict("records"),
        variant,
        as_of_date=inner_val["event_date"].min().date(),
    )
    admitted = np.asarray(raw_weights, dtype=float) > 0
    train = inner_train.loc[admitted].copy()
    weights = np.asarray(raw_weights, dtype=float)[admitted]
    weights = weights * len(weights) / weights.sum()
    features = config.features
    scale_columns = [name for name in features if name != "weightclass_encoded"]
    scaler = RobustScaler().fit(train[scale_columns])
    X_train = train[features].copy()
    X_val = inner_val[features].copy()
    X_outer = outer[features].copy()
    for candidate in (X_train, X_val, X_outer):
        candidate.loc[:, scale_columns] = scaler.transform(candidate[scale_columns])
    train_data = X_train.copy()
    train_data["y_true"] = train["y_true"].astype(int).to_numpy()
    train_data["sample_weight"] = weights
    tuning_data = X_val.copy()
    tuning_data["y_true"] = inner_val["y_true"].astype(int).to_numpy()
    tuning_data["sample_weight"] = 1.0

    variant_root = Path(artifact_root) / f"fold-{year}/variants/{variant_id}"
    predictor = TabularPredictor(
        label="y_true",
        eval_metric="log_loss",
        problem_type="binary",
        path=str(variant_root / "model"),
        verbosity=2,
        sample_weight="sample_weight",
        weight_evaluation=False,
    )
    predictor.fit(**build_training_fit_kwargs(config, train_data=train_data, tuning_data=tuning_data))
    inner_probabilities = _positive_probabilities(predictor.predict_proba(X_val))
    outer_probabilities = _positive_probabilities(predictor.predict_proba(X_outer))
    clipped = np.clip(inner_probabilities, 1e-15, 1 - 1e-15)
    labels = inner_val["y_true"].astype(int).to_numpy()
    inner_log_loss = float(-np.mean(labels * np.log(clipped) + (1 - labels) * np.log(1 - clipped)))
    fit_max = train["event_date"].max().date().isoformat()
    selection_max = inner_val["event_date"].max().date().isoformat()
    fit_events = sorted(set(train["event_id"].astype(str)) | set(inner_val["event_id"].astype(str)))
    experience = _experience_labels(roster.iloc[:3089])
    inner_rows = _prediction_rows(
        inner_val,
        inner_probabilities,
        fold=f"inner-{year}",
        fit_max_date=fit_max,
        selection_max_date=selection_max,
        fit_event_ids=fit_events,
        experience=experience,
        selected_variant=variant_id,
        boundary="InnerSelection",
    )
    outer_rows = _prediction_rows(
        outer,
        outer_probabilities,
        fold=str(year),
        fit_max_date=fit_max,
        selection_max_date=selection_max,
        fit_event_ids=fit_events,
        experience=experience,
        selected_variant=variant_id,
        boundary="Original",
    )
    validate_prediction_chronology(outer_rows)
    _write_jsonl(variant_root / "inner-predictions.jsonl", inner_rows)
    _write_jsonl(variant_root / "outer-predictions.jsonl", outer_rows)
    write_canonical_json(variant_root / "inner-evidence.json", {
        "variant_id": variant_id,
        "partition": "inner-validation",
        "selection_max_date": selection_max,
        "outer_min_date": outer_start.date().isoformat(),
        "log_loss": inner_log_loss,
        "row_count": len(inner_rows),
        "training_row_count": len(train),
        "zero_weight_row_count": int(len(inner_train) - len(train)),
        "as_of_date": inner_val["event_date"].min().date().isoformat(),
        "runtime": runtime,
        "best_model": predictor.model_best,
        "model_names": predictor.model_names(),
        "refit_full": False,
        "feature_importance_computed": False,
    })


def _promotion_decision(
    candidate_metrics: Mapping[str, Any],
    intervals: Mapping[str, Any],
    *,
    complete: bool,
) -> dict[str, Any]:
    improves = complete and candidate_metrics["log_loss"] < 0.6195954814877112
    interval_below_zero = complete and intervals["log_loss_delta"]["upper"] < 0
    promoted = bool(improves and interval_below_zero)
    return {
        "action": "promote-family-02" if promoted else "retain-family-01",
        "incumbent_before": "family-01-weighted-v8-control",
        "incumbent_after": EXPERIMENT_ID if promoted else "family-01-weighted-v8-control",
        "all_four_outer_folds_complete": complete,
        "candidate_log_loss_below_incumbent": bool(improves),
        "paired_log_loss_interval_upper_below_zero": bool(interval_below_zero),
    }


def _finalize(
    campaign_root: Path,
    artifact_root: Path,
    *,
    status: str,
    failure: Mapping[str, Any] | None,
) -> dict[str, Any]:
    campaign_root = Path(campaign_root)
    artifact_root = Path(artifact_root)
    predictions = []
    inner_predictions = []
    fold_entries = []
    selected_variants = []
    if status == "complete":
        for year in OUTER_YEARS:
            fold_root = artifact_root / f"fold-{year}"
            selection = read_json(fold_root / "selection.json")
            selected_variants.append(selection)
            outer_path = fold_root / "outer-predictions.jsonl"
            inner_path = fold_root / "selected-inner-predictions.jsonl"
            outer_rows = _read_jsonl(outer_path)
            predictions.extend(outer_rows)
            inner_predictions.extend(_read_jsonl(inner_path))
            fold_entries.append({
                "year": year,
                "variant_id": selection["variant_id"],
                "path": outer_path.relative_to(artifact_root).as_posix(),
                "row_count": len(outer_rows),
                "sha256": file_sha256(outer_path),
            })
        candidate_metrics = reduce_predictions(predictions).as_dict()
        incumbent = []
        for year in OUTER_YEARS:
            incumbent.extend(_read_jsonl(FIXED_INCUMBENT_ARTIFACT / f"fold-{year}/outer-predictions.jsonl"))
        incumbent_metrics = reduce_predictions(incumbent).as_dict()
        profile = read_json(campaign_root / f"profiles/{EXPERIMENT_ID}.json")
        intervals = event_block_bootstrap_delta(
            predictions,
            incumbent,
            iterations=profile["bootstrap"]["iterations"],
            seed=profile["bootstrap"]["seed"],
        )
        inner_metrics = {
            "status": "selection-context",
            "fold_log_loss": {
                str(year): selected_variants[index]["inner_log_loss"]
                for index, year in enumerate(OUTER_YEARS)
            },
            "row_count": len(inner_predictions),
        }
        inner_outer_gap = {
            "status": "inadmissible",
            "reason": (
                "inner labels select both AutoGluon models and the joint variant, so their same-row "
                "score is reported as selection context and cannot be laundered into a train gap"
            ),
        }
        fold_losses = candidate_metrics["fold_metrics"]
        drift = {
            "fold_log_loss": {year: fold_losses[str(year)]["log_loss"] for year in OUTER_YEARS},
            "year_over_year_log_loss_delta": {
                str(year): fold_losses[str(year)]["log_loss"] - fold_losses[str(year - 1)]["log_loss"]
                for year in OUTER_YEARS[1:]
            },
        }
        decision = _promotion_decision(candidate_metrics, intervals, complete=True)
    else:
        candidate_metrics = incumbent_metrics = intervals = inner_metrics = inner_outer_gap = drift = None
        decision = _promotion_decision({}, {}, complete=False)
    result_payload = {
        "experiment_id": EXPERIMENT_ID,
        "status": status,
        "metrics": candidate_metrics,
        "incumbent_metrics": incumbent_metrics,
        "paired_event_block_intervals": intervals,
        "drift_summary": drift,
        "selected_inner_metrics": inner_metrics,
        "inner_outer_gap": inner_outer_gap,
        "selected_variants": selected_variants,
        "fold_predictions": fold_entries,
        "promotion_decision": decision,
        "adaptive_signal_for_family_03": (
            "Use only this registered fold drift and selected-inner/outer calibration evidence; "
            "do not branch from gate or outer-label selector feedback."
        ),
        "terminal_failure": dict(failure) if failure else None,
    }
    write_canonical_json(artifact_root / "result.json", result_payload)
    inventory = tree_inventory(artifact_root)
    profile_path = campaign_root / f"profiles/{EXPERIMENT_ID}.json"
    manifest = {
        **result_payload,
        "kind": "family",
        "exit_state": status,
        "source_revision": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "profile_path": profile_path.relative_to(campaign_root).as_posix(),
        "profile_sha256": canonical_sha256(read_json(profile_path)),
        "preregistration_path": f"runs/{EXPERIMENT_ID}/preregistration.json",
        "attempts_path": f"runs/{EXPERIMENT_ID}/attempts.jsonl",
        "artifact_path": artifact_root.relative_to(campaign_root).as_posix(),
        "artifact_tree_sha256": inventory.tree_sha256,
        "artifact_file_count": inventory.file_count,
        "gate_state": "closed",
        "gate_access_count": 0,
    }
    manifest_path = campaign_root / f"runs/{EXPERIMENT_ID}/manifest.json"
    manifest_sha = write_canonical_json(manifest_path, manifest)
    append_registry_record(campaign_root, {
        "experiment_id": EXPERIMENT_ID,
        "kind": "family",
        "status": status,
        "profile_path": manifest["profile_path"],
        "profile_sha256": manifest["profile_sha256"],
        "manifest_path": manifest_path.relative_to(campaign_root).as_posix(),
        "manifest_sha256": manifest_sha,
        "artifact_path": manifest["artifact_path"],
        "artifact_tree_sha256": inventory.tree_sha256,
    })
    return manifest


def launch(campaign_root: Path, *, deadline_seconds: int | None = None) -> dict[str, Any]:
    from .weighted_v8 import _gpu_process_snapshot

    campaign_root = Path(campaign_root)
    prereg = read_json(campaign_root / f"runs/{EXPERIMENT_ID}/preregistration.json")
    profile = read_json(campaign_root / prereg["profile_path"])
    if canonical_sha256(profile) != prereg["profile_sha256"]:
        raise ValueError("profile differs from preregistration")
    if _registry_sha(campaign_root / "registry.jsonl") != prereg["registry_prefix_sha256_before"]:
        raise ValueError("registry changed before the authorized launch")
    if AccessLedger(campaign_root).gate_status()["protected_access_count"] != 0:
        raise ValueError("gate access occurred before launch")
    artifact_root = campaign_root / prereg["artifact_path"]
    if artifact_root.exists():
        raise ValueError("refusing to reuse the family artifact path")
    artifact_root.mkdir(parents=True)
    attempts_path = campaign_root / f"runs/{EXPERIMENT_ID}/attempts.jsonl"
    deadline_seconds = deadline_seconds or int(prereg["family_deadline_seconds"])
    start = time.monotonic()
    lock_path = artifact_root / "gpu.lock"
    descriptor = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    failure = None
    write_canonical_json(artifact_root / "gpu-lease-acquired.json", {
        "pid": os.getpid(), "acquired_unix_ns": time.time_ns(), "prelaunch_processes": _gpu_process_snapshot(),
    })
    try:
        for year in OUTER_YEARS:
            scores = []
            for variant_id in FAMILY_2_VARIANT_IDS:
                elapsed = time.monotonic() - start
                remaining = deadline_seconds - elapsed
                if remaining <= 60:
                    failure = {
                        "attempt_id": f"{year}-{variant_id}-attempt-1",
                        "fold": year,
                        "variant_id": variant_id,
                        "exit_state": "limited",
                        "exit_code": 124,
                        "reason": "family deadline exhausted before the next unique launch",
                    }
                    break
                variant_root = artifact_root / f"fold-{year}/variants/{variant_id}"
                variant_root.mkdir(parents=True)
                stdout_path = variant_root / "stdout.log"
                stderr_path = variant_root / "stderr.log"
                attempt_id = f"{year}-{variant_id}-attempt-1"
                command = [
                    sys.executable, "-m", "libs.modeling.experiment_campaign.families.horizon_recency",
                    "fit-variant", "--campaign", str(campaign_root), "--artifact-root", str(artifact_root),
                    "--year", str(year), "--variant", variant_id,
                ]
                _append_jsonl(attempts_path, {
                    "attempt_id": attempt_id,
                    "fold": year,
                    "variant_id": variant_id,
                    "state": "launched",
                    "launched_unix_ns": time.time_ns(),
                    "command": command,
                    "preregistration_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
                    "stdout_path": stdout_path.relative_to(artifact_root).as_posix(),
                    "stderr_path": stderr_path.relative_to(artifact_root).as_posix(),
                })
                cap = min(int(prereg["per_fit_time_cap_seconds"]), max(60, int(remaining)))
                exit_code = 124
                exit_state = "limited"
                with stdout_path.open("wb") as stdout, stderr_path.open("wb") as stderr:
                    try:
                        completed = subprocess.run(command, stdout=stdout, stderr=stderr, timeout=cap, check=False)
                        exit_code = completed.returncode
                        exit_state = "succeeded" if exit_code == 0 else "failed"
                    except subprocess.TimeoutExpired:
                        stderr.write(f"\nfit exceeded preregistered cap of {cap}s\n".encode())
                _append_jsonl(attempts_path, {
                    "attempt_id": attempt_id,
                    "fold": year,
                    "variant_id": variant_id,
                    "state": exit_state,
                    "exited_unix_ns": time.time_ns(),
                    "exit_code": exit_code,
                    "stdout_sha256": file_sha256(stdout_path),
                    "stderr_sha256": file_sha256(stderr_path),
                    "attempt_elapsed_seconds": time.monotonic() - start - elapsed,
                    "family_elapsed_seconds": time.monotonic() - start,
                })
                if exit_state != "succeeded":
                    failure = {
                        "attempt_id": attempt_id,
                        "fold": year,
                        "variant_id": variant_id,
                        "exit_state": exit_state,
                        "exit_code": exit_code,
                        "stdout_path": stdout_path.relative_to(artifact_root).as_posix(),
                        "stdout_sha256": file_sha256(stdout_path),
                        "stderr_path": stderr_path.relative_to(artifact_root).as_posix(),
                        "stderr_sha256": file_sha256(stderr_path),
                    }
                    break
                scores.append(read_json(variant_root / "inner-evidence.json"))
            if failure:
                break
            outer_start = date.fromisoformat(scores[0]["outer_min_date"])
            selected = select_joint_variant(scores, outer_min_date=outer_start, embargo_days=profile["embargo_days"])
            fold_root = artifact_root / f"fold-{year}"
            source_root = fold_root / "variants" / selected["variant_id"]
            (fold_root / "outer-predictions.jsonl").write_bytes((source_root / "outer-predictions.jsonl").read_bytes())
            (fold_root / "selected-inner-predictions.jsonl").write_bytes((source_root / "inner-predictions.jsonl").read_bytes())
            write_canonical_json(fold_root / "selection.json", {**selected, "inner_scores": scores})
    finally:
        os.close(descriptor)
        lock_path.unlink()
        write_canonical_json(artifact_root / "gpu-lease-released.json", {
            "pid": os.getpid(), "released_unix_ns": time.time_ns(), "elapsed_seconds": time.monotonic() - start,
        })
    status = failure["exit_state"] if failure else "complete"
    return _finalize(campaign_root, artifact_root, status=status, failure=failure)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)
    pre = commands.add_parser("preregister")
    pre.add_argument("--campaign", type=Path, required=True)
    launch_parser = commands.add_parser("launch")
    launch_parser.add_argument("--campaign", type=Path, required=True)
    launch_parser.add_argument("--deadline-seconds", type=int)
    fit = commands.add_parser("fit-variant")
    fit.add_argument("--campaign", type=Path, required=True)
    fit.add_argument("--artifact-root", type=Path, required=True)
    fit.add_argument("--year", type=int, required=True)
    fit.add_argument("--variant", required=True)
    return parser


def main() -> int:
    args = _parser().parse_args()
    if args.command == "preregister":
        result = preregister(args.campaign)
    elif args.command == "launch":
        result = launch(args.campaign, deadline_seconds=args.deadline_seconds)
    else:
        fit_variant(args.campaign, args.artifact_root, args.year, args.variant)
        result = {"status": "succeeded", "year": args.year, "variant_id": args.variant}
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
