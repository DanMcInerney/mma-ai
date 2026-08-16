"""One-shot nested weighted-v8 control over the frozen development folds."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pickle
import random
import subprocess
import sys
import time
from datetime import date, timedelta
from pathlib import Path
from typing import Any, Iterable, Mapping

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
from ..registry import append_registry_record, validate_registry, validate_resolved_profile


EXPERIMENT_ID = "family-01-weighted-v8-control"
OUTER_YEARS = (2022, 2023, 2024, 2025)
SEEDS = {"python": 20260815, "numpy": 20260815, "torch": 20260815, "bootstrap": 20260815}
FIXED_SOURCE_ARTIFACT = Path(
    r"C:\Users\danhm\mma-ai\worktrees\top10-20260815"
    r"\experiments\top10_20260815\artifacts\01-campaign-harness"
)
EXPECTED_REGISTRY_PREFIX = "D3F2BC6807F707C0A4696091E64DB92E773BD26F7266A1E7B718BFDC5AE891FB"
EXPECTED_FEATURE_SHA256 = "13E545D762A3F1BE4D023D82B8E65D77E41589031051F1F6796D742F25223022"


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


def _registry_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def materialized_profile() -> dict[str, Any]:
    from libs.modeling.training_profiles import WIN_V8_HYBRID_WORKING_PROFILE

    profile = dict(WIN_V8_HYBRID_WORKING_PROFILE)
    profile["features"] = list(profile["features"])
    profile["refit_full"] = False
    profile["calculate_importance"] = False
    validate_resolved_profile(profile)
    if canonical_sha256(profile["features"]) != EXPECTED_FEATURE_SHA256:
        raise ValueError("weighted-v8 ordered feature identity changed")
    return profile


def preregister(campaign_root: Path) -> dict[str, Any]:
    campaign_root = Path(campaign_root)
    profile_path = campaign_root / "profiles" / f"{EXPERIMENT_ID}.json"
    run_root = campaign_root / "runs" / EXPERIMENT_ID
    prereg_path = run_root / "preregistration.json"
    attempts_path = run_root / "attempts.jsonl"
    for path in (profile_path, prereg_path, attempts_path):
        if path.exists():
            raise ValueError(f"refusing to overwrite preregistration artifact: {path}")
    registry_path = campaign_root / "registry.jsonl"
    if _registry_sha(registry_path) != EXPECTED_REGISTRY_PREFIX:
        raise ValueError("experiment-zero registry prefix is not the frozen input")
    gate = AccessLedger(campaign_root).gate_status()
    if gate["state"] != "closed" or gate["protected_access_count"] != 0:
        raise ValueError("preregistration requires the gate closed with zero access")
    if not FIXED_SOURCE_ARTIFACT.is_dir():
        raise ValueError("fixed read-only campaign artifact is unavailable")

    profile = materialized_profile()
    profile_sha = write_canonical_json(profile_path, profile)
    fold_manifest = read_json(campaign_root / "baseline" / "fold-manifest.json")
    baseline_manifest_path = campaign_root / "baseline" / "manifest.json"
    preregistration = {
        "experiment_id": EXPERIMENT_ID,
        "family_number": 1,
        "hypothesis": (
            "The exact recency-weighted v8 hybrid, evaluated by nested whole-event temporal folds, "
            "will establish a less biased development denominator than its exposed 2025 tuning score."
        ),
        "variant_bound": 1,
        "variant_id": "weighted-v8-exact-control",
        "profile_path": profile_path.relative_to(campaign_root).as_posix(),
        "profile_sha256": profile_sha,
        "outer_years": list(OUTER_YEARS),
        "embargo_days": 7,
        "selection_boundary": "Original",
        "same_row_foundation_context_admissible": False,
        "full_metrics_admissible": False,
        "gate_state_required": "closed",
        "source_artifact_mode": "fixed-read-only-campaign-artifact",
        "source_artifact_path": str(FIXED_SOURCE_ARTIFACT),
        "source_artifact_tree_sha256": "40FB5DCC31B6B3D9B920F54ECB82D85B0FFA4B6814A7207B19B2336389E08F50",
        "artifact_path": "artifacts/02-family-01-weighted-v8-control",
        "registry_prefix_sha256_before": EXPECTED_REGISTRY_PREFIX,
        "baseline_manifest_file_sha256": file_sha256(baseline_manifest_path),
        "fold_manifest_sha256": canonical_sha256(fold_manifest),
        "seeds": SEEDS,
        "invocation": (
            "uv run python -m libs.modeling.experiment_campaign.families.weighted_v8 launch "
            "--campaign experiments/top10_20260815"
        ),
        "promotion_rule": (
            "Family 1 becomes the development control denominator after four valid outer folds. "
            "The frozen replacement predicate is not applied because experiment zero has no aligned "
            "four-fold predictions; the aligned 2025 comparison is diagnostic only."
        ),
        "no_recency_negative_control": {
            "training_launched": False,
            "correct_count": 307,
            "row_count": 460,
            "positive_log_loss": 0.617117449878959,
            "verification_sha256": "7CC2285DB55AE6BBCF1E9897D4C752DDA7004B70D7C7F39ABCB79CCE04B7D0C4",
        },
    }
    write_canonical_json(prereg_path, preregistration)
    attempts_path.write_bytes(b"")
    (run_root / "decision.md").write_bytes(
        b"# Family 1 preregistration\n\n"
        b"One exact weighted-v8 control variant; four frozen outer folds; "
        b"Original out-of-time predictions only; 2026 gate closed.\n"
    )
    return preregistration


def validate_prediction_chronology(records: Iterable[Mapping[str, Any]]) -> None:
    for record in records:
        event_date = date.fromisoformat(str(record["event_date"])[:10])
        embargo = int(record["embargo_days"])
        latest_allowed = event_date - timedelta(days=embargo)
        for noun in ("fit_max_date", "context_max_date", "selection_max_date"):
            if noun not in record:
                continue
            observed = date.fromisoformat(str(record[noun])[:10])
            if observed > latest_allowed:
                raise ValueError(f"{noun} violates the {embargo}-day embargo")
        if record["event_id"] in set(record.get("fit_event_ids", ())):
            raise ValueError("same-event fit/context crossing")


def _safe_roster(source_csv: Path) -> pd.DataFrame:
    columns = ["fight_id", "event_id", "event_date", "fighter1_id", "fighter2_id"]
    roster = pd.read_csv(source_csv, usecols=columns, dtype="string")
    roster["event_date"] = pd.to_datetime(roster["event_date"], errors="raise")
    if not roster["event_date"].is_monotonic_increasing:
        raise ValueError("fixed filtered source must be chronological before bounded label parsing")
    if roster["fight_id"].duplicated().any() or roster[columns].isna().any().any():
        raise ValueError("fixed roster identities must be complete and unique")
    return roster


def _experience_labels(roster: pd.DataFrame) -> dict[str, str]:
    counts: dict[str, int] = {}
    labels: dict[str, str] = {}
    for row in roster.itertuples(index=False):
        first = counts.get(str(row.fighter1_id), 0)
        second = counts.get(str(row.fighter2_id), 0)
        minimum = min(first, second)
        labels[str(row.fight_id)] = "novice" if minimum < 3 else "veteran" if minimum >= 5 else "mixed"
        counts[str(row.fighter1_id)] = first + 1
        counts[str(row.fighter2_id)] = second + 1
    return labels


def _load_pre_gate_frame(source_csv: Path, features: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    roster = _safe_roster(source_csv)
    pre_gate_count = int((roster["event_date"] < pd.Timestamp("2026-01-01")).sum())
    if pre_gate_count != 3089:
        raise ValueError(f"expected 3,089 pre-gate rows, found {pre_gate_count}")
    if not (roster.iloc[:pre_gate_count]["event_date"] < pd.Timestamp("2026-01-01")).all():
        raise ValueError("pre-gate roster is not a chronological prefix")
    columns = list(dict.fromkeys(
        features + ["fight_id", "event_id", "event_date", "fighter1_id", "fighter2_id", "method", "y_true"]
    ))
    frame = pd.read_csv(source_csv, usecols=columns, nrows=pre_gate_count)
    frame["event_date"] = pd.to_datetime(frame["event_date"], errors="raise")
    if (frame["event_date"] >= pd.Timestamp("2026-01-01")).any():
        raise ValueError("bounded parser crossed into the closed campaign gate")
    expected_ids = roster.iloc[:pre_gate_count]["fight_id"].astype(str).tolist()
    if frame["fight_id"].astype(str).tolist() != expected_ids:
        raise ValueError("bounded label rows do not align with the safe roster prefix")
    return frame, roster


def _seed_runtime() -> None:
    os.environ["PYTHONHASHSEED"] = str(SEEDS["python"])
    random.seed(SEEDS["python"])
    np.random.seed(SEEDS["numpy"])
    import torch

    torch.manual_seed(SEEDS["torch"])
    torch.cuda.manual_seed_all(SEEDS["torch"])


def fit_fold(campaign_root: Path, artifact_root: Path, year: int) -> None:
    from autogluon.tabular import TabularPredictor
    from sklearn.preprocessing import RobustScaler

    from libs.modeling.train import TrainingConfig, build_training_fit_kwargs, training_runtime_preflight

    if year not in OUTER_YEARS:
        raise ValueError("year is outside the frozen outer folds")
    _seed_runtime()
    runtime = training_runtime_preflight()
    profile = read_json(Path(campaign_root) / "profiles" / f"{EXPERIMENT_ID}.json")
    config = TrainingConfig(**profile)
    source_csv = FIXED_SOURCE_ARTIFACT / "models" / "accepted" / "training_data.csv"
    frame, roster = _load_pre_gate_frame(source_csv, profile["features"])
    frame["fight_id"] = frame["fight_id"].astype(str)
    frame["event_id"] = frame["event_id"].astype(str)
    fold_manifest = read_json(Path(campaign_root) / "baseline" / "fold-manifest.json")
    fold = next(item for item in fold_manifest["folds"] if item["test_year"] == year)
    inner_train_ids = set(fold["inner"]["train_fight_ids"])
    inner_val_ids = set(fold["inner"]["validation_fight_ids"])
    outer_ids = set(fold["outer"]["test_fight_ids"])
    inner_train = frame[frame["fight_id"].isin(inner_train_ids)].copy()
    inner_val = frame[frame["fight_id"].isin(inner_val_ids)].copy()
    outer = frame[frame["fight_id"].isin(outer_ids)].copy()
    if (len(inner_train), len(inner_val), len(outer)) != (
        len(inner_train_ids), len(inner_val_ids), len(outer_ids)
    ):
        raise ValueError("fold membership does not resolve exactly in the bounded source")
    outer_start = outer["event_date"].min()
    if inner_val["event_date"].max() > outer_start - pd.Timedelta(days=7):
        raise ValueError("inner selection rows cross the outer embargo")

    features = profile["features"]
    scale_columns = [name for name in features if name != "weightclass_encoded"]
    scaler = RobustScaler().fit(inner_train[scale_columns])
    X_train = inner_train[features].copy()
    X_val = inner_val[features].copy()
    X_outer = outer[features].copy()
    for candidate in (X_train, X_val, X_outer):
        candidate.loc[:, scale_columns] = scaler.transform(candidate[scale_columns])
    days_ago = (inner_train["event_date"].max() - inner_train["event_date"]).dt.days
    weights = np.exp(-profile["decay_rate"] * days_ago / 365.25)
    weights = weights * len(weights) / weights.sum()
    train_data = X_train.copy()
    train_data["y_true"] = inner_train["y_true"].astype(int).to_numpy()
    train_data["sample_weight"] = weights.to_numpy()
    tuning_data = X_val.copy()
    tuning_data["y_true"] = inner_val["y_true"].astype(int).to_numpy()
    tuning_data["sample_weight"] = 1.0

    fold_root = Path(artifact_root) / f"fold-{year}"
    model_root = fold_root / "model"
    predictor = TabularPredictor(
        label="y_true",
        eval_metric="log_loss",
        problem_type="binary",
        path=str(model_root),
        verbosity=2,
        sample_weight="sample_weight",
        weight_evaluation=False,
    )
    predictor.fit(**build_training_fit_kwargs(config, train_data=train_data, tuning_data=tuning_data))
    probabilities = predictor.predict_proba(X_outer)
    if hasattr(probabilities, "columns"):
        positive = probabilities[1] if 1 in probabilities.columns else probabilities.iloc[:, -1]
    else:
        positive = np.asarray(probabilities)

    experience = _experience_labels(roster.iloc[:3089])
    fit_events = sorted(set(inner_train["event_id"].astype(str)) | set(inner_val["event_id"].astype(str)))
    records = []
    for position, (_, row) in enumerate(outer.iterrows()):
        method = str(row["method"]).lower()
        records.append({
            "fight_id": str(row["fight_id"]),
            "event_id": str(row["event_id"]),
            "event_date": row["event_date"].date().isoformat(),
            "y_true": int(row["y_true"]),
            "probability": float(np.asarray(positive)[position]),
            "boundary": "Original",
            "fit_scope": "prior-only",
            "fold": str(year),
            "weight_class": str(row["weightclass_encoded"]),
            "experience": experience[str(row["fight_id"])],
            "outcome_type": "decision" if "dec" in method else "finish",
            "fit_max_date": inner_train["event_date"].max().date().isoformat(),
            "selection_max_date": inner_val["event_date"].max().date().isoformat(),
            "context_max_date": inner_val["event_date"].max().date().isoformat(),
            "fit_event_ids": fit_events,
            "embargo_days": 7,
        })
    validate_prediction_chronology(records)
    predictions_path = fold_root / "outer-predictions.jsonl"
    _write_jsonl(predictions_path, records)
    write_canonical_json(fold_root / "fit-evidence.json", {
        "year": year,
        "runtime": runtime,
        "seeds": SEEDS,
        "inner_train_rows": len(inner_train),
        "inner_validation_rows": len(inner_val),
        "outer_rows": len(outer),
        "inner_train_date_range": [inner_train["event_date"].min().date().isoformat(), inner_train["event_date"].max().date().isoformat()],
        "inner_validation_date_range": [inner_val["event_date"].min().date().isoformat(), inner_val["event_date"].max().date().isoformat()],
        "outer_date_range": [outer["event_date"].min().date().isoformat(), outer["event_date"].max().date().isoformat()],
        "best_model": predictor.model_best,
        "model_names": predictor.model_names(),
        "refit_full": False,
        "feature_importance_computed": False,
        "prediction_boundary": "Original",
        "same_row_scores_emitted": False,
    })


def _gpu_process_snapshot() -> dict[str, Any]:
    result = subprocess.run(
        ["nvidia-smi", "--query-compute-apps=pid,process_name,used_gpu_memory", "--format=csv,noheader"],
        capture_output=True,
        text=True,
        check=False,
    )
    lines = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    foreign_python = [line for line in lines if "python" in line.lower() and not line.startswith(f"{os.getpid()},")]
    if foreign_python:
        raise RuntimeError(f"another Python GPU process is active: {foreign_python}")
    return {"exit_code": result.returncode, "rows": lines, "foreign_python_rows": foreign_python}


def _baseline_2025(candidate_2025: list[dict[str, Any]]) -> list[dict[str, Any]]:
    path = FIXED_SOURCE_ARTIFACT / "models" / "accepted" / "utils" / "attr" / "Mitra" / "y_pred_proba_val.pkl"
    with path.open("rb") as handle:
        probabilities = np.asarray(pickle.load(handle), dtype=float)
    if len(probabilities) != 460 or len(candidate_2025) != 282:
        raise ValueError("historical validation prediction shape changed")
    baseline = []
    for candidate, probability in zip(candidate_2025, probabilities[:282], strict=True):
        baseline.append({**candidate, "probability": float(probability)})
    return baseline


def _finalize(
    campaign_root: Path,
    artifact_root: Path,
    *,
    status: str,
    failure: Mapping[str, Any] | None,
) -> dict[str, Any]:
    campaign_root = Path(campaign_root)
    artifact_root = Path(artifact_root)
    profile_path = campaign_root / "profiles" / f"{EXPERIMENT_ID}.json"
    profile_sha = canonical_sha256(read_json(profile_path))
    fold_entries: list[dict[str, Any]] = []
    metrics = None
    baseline_metrics = None
    intervals = None
    if status == "complete":
        predictions: list[dict[str, Any]] = []
        for year in OUTER_YEARS:
            path = artifact_root / f"fold-{year}" / "outer-predictions.jsonl"
            rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
            predictions.extend(rows)
            fold_entries.append({
                "year": year,
                "path": path.relative_to(artifact_root).as_posix(),
                "row_count": len(rows),
                "sha256": file_sha256(path),
            })
        validate_prediction_chronology(predictions)
        metrics = reduce_predictions(predictions).as_dict()
        candidate_2025 = [row for row in predictions if row["fold"] == "2025"]
        baseline = _baseline_2025(candidate_2025)
        baseline_path = artifact_root / "historical-experiment-zero-2025-predictions.jsonl"
        _write_jsonl(baseline_path, baseline)
        baseline_metrics = reduce_predictions(baseline).as_dict()
        intervals = event_block_bootstrap_delta(candidate_2025, baseline, iterations=2000, seed=SEEDS["bootstrap"])
    result_payload = {
        "experiment_id": EXPERIMENT_ID,
        "status": status,
        "metrics": metrics,
        "historical_baseline_2025_metrics": baseline_metrics,
        "paired_event_block_intervals": intervals,
        "fold_predictions": fold_entries,
        "train_outer_gap": {
            "status": "inadmissible",
            "reason": "same-row foundation context and tuning-selected ensemble scores cannot form an honest train gap",
        },
        "terminal_failure": dict(failure) if failure else None,
    }
    write_canonical_json(artifact_root / "result.json", result_payload)
    inventory = tree_inventory(artifact_root)
    decision = {
        "action": "establish-development-control" if status == "complete" else "retain-experiment-zero",
        "incumbent_before": "experiment-zero",
        "incumbent_after": EXPERIMENT_ID if status == "complete" else "experiment-zero",
        "frozen_replacement_predicate_applicable": False,
        "reason": (
            "experiment zero has no aligned four-fold development predictions; family 1 is the preregistered control denominator"
            if status == "complete"
            else "the only authorized control launch ended in a preserved terminal failure"
        ),
    }
    manifest = {
        "experiment_id": EXPERIMENT_ID,
        "kind": "family",
        "exit_state": status,
        "source_revision": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "dirty_state": "tracked result artifacts pending terminal commit",
        "profile_path": profile_path.relative_to(campaign_root).as_posix(),
        "profile_sha256": profile_sha,
        "preregistration_path": f"runs/{EXPERIMENT_ID}/preregistration.json",
        "attempts_path": f"runs/{EXPERIMENT_ID}/attempts.jsonl",
        "artifact_path": artifact_root.relative_to(campaign_root).as_posix(),
        "artifact_tree_sha256": inventory.tree_sha256,
        "artifact_file_count": inventory.file_count,
        "fold_predictions": fold_entries,
        "baseline_predictions_path": "historical-experiment-zero-2025-predictions.jsonl" if status == "complete" else None,
        "metrics": metrics,
        "historical_baseline_2025_metrics": baseline_metrics,
        "paired_event_block_intervals": intervals,
        "bootstrap": {"iterations": 2000, "seed": SEEDS["bootstrap"]},
        "train_outer_gap": result_payload["train_outer_gap"],
        "promotion_decision": decision,
        "terminal_failure": dict(failure) if failure else None,
        "gate_state": "closed",
        "gate_access_count": 0,
    }
    manifest_path = campaign_root / "runs" / EXPERIMENT_ID / "manifest.json"
    manifest_sha = write_canonical_json(manifest_path, manifest)
    append_registry_record(campaign_root, {
        "experiment_id": EXPERIMENT_ID,
        "kind": "family",
        "status": status,
        "profile_path": profile_path.relative_to(campaign_root).as_posix(),
        "profile_sha256": profile_sha,
        "manifest_path": manifest_path.relative_to(campaign_root).as_posix(),
        "manifest_sha256": manifest_sha,
        "artifact_path": artifact_root.relative_to(campaign_root).as_posix(),
        "artifact_tree_sha256": inventory.tree_sha256,
    })
    return manifest


def launch(campaign_root: Path, *, deadline_seconds: int) -> dict[str, Any]:
    campaign_root = Path(campaign_root)
    prereg = read_json(campaign_root / "runs" / EXPERIMENT_ID / "preregistration.json")
    if canonical_sha256(read_json(campaign_root / prereg["profile_path"])) != prereg["profile_sha256"]:
        raise ValueError("profile differs from preregistration")
    if _registry_sha(campaign_root / "registry.jsonl") != prereg["registry_prefix_sha256_before"]:
        raise ValueError("registry changed before the authorized launch")
    if AccessLedger(campaign_root).gate_status()["protected_access_count"] != 0:
        raise ValueError("gate access occurred before launch")
    artifact_root = campaign_root / prereg["artifact_path"]
    if artifact_root.exists():
        raise ValueError("refusing to reuse the family artifact path")
    artifact_root.mkdir(parents=True)
    lock_path = artifact_root / "gpu.lock"
    descriptor = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    start = time.monotonic()
    attempts_path = campaign_root / "runs" / EXPERIMENT_ID / "attempts.jsonl"
    failure: dict[str, Any] | None = None
    snapshot = _gpu_process_snapshot()
    write_canonical_json(artifact_root / "gpu-lease-acquired.json", {
        "pid": os.getpid(), "acquired_unix_ns": time.time_ns(), "prelaunch_processes": snapshot,
    })
    try:
        for year in OUTER_YEARS:
            remaining = deadline_seconds - (time.monotonic() - start)
            if remaining <= 60:
                failure = {
                    "failed_fold": year,
                    "exit_code": 124,
                    "reason": "family deadline exhausted before the next unique authorized fold launch",
                    "stderr_path": f"fold-{year}/stderr.log",
                    "stderr_sha256": "",
                }
                stderr = artifact_root / f"fold-{year}" / "stderr.log"
                stderr.parent.mkdir(parents=True, exist_ok=True)
                stderr.write_text(failure["reason"] + "\n", encoding="utf-8")
                failure["stderr_sha256"] = file_sha256(stderr)
                break
            fold_root = artifact_root / f"fold-{year}"
            fold_root.mkdir(parents=True)
            stdout_path = fold_root / "stdout.log"
            stderr_path = fold_root / "stderr.log"
            attempt_id = f"weighted-v8-{year}-attempt-1"
            command = [
                sys.executable, "-m", "libs.modeling.experiment_campaign.families.weighted_v8",
                "fit-fold", "--campaign", str(campaign_root), "--artifact-root", str(artifact_root), "--year", str(year),
            ]
            _append_jsonl(attempts_path, {
                "attempt_id": attempt_id,
                "variant_id": "weighted-v8-exact-control",
                "fold": year,
                "state": "launched",
                "launched_unix_ns": time.time_ns(),
                "command": command,
                "preregistration_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
                "stdout_path": stdout_path.relative_to(artifact_root).as_posix(),
                "stderr_path": stderr_path.relative_to(artifact_root).as_posix(),
            })
            timeout = min(3600, max(60, int(remaining)))
            exit_code = 124
            exit_reason = "timeout"
            with stdout_path.open("wb") as stdout, stderr_path.open("wb") as stderr:
                try:
                    completed = subprocess.run(command, stdout=stdout, stderr=stderr, timeout=timeout, check=False)
                    exit_code = completed.returncode
                    exit_reason = "process-exit"
                except subprocess.TimeoutExpired:
                    stderr.write(f"\nfit exceeded authorized timeout of {timeout}s\n".encode())
            _append_jsonl(attempts_path, {
                "attempt_id": attempt_id,
                "fold": year,
                "state": "exited",
                "exited_unix_ns": time.time_ns(),
                "exit_code": exit_code,
                "exit_reason": exit_reason,
                "stdout_sha256": file_sha256(stdout_path),
                "stderr_sha256": file_sha256(stderr_path),
                "elapsed_seconds": time.monotonic() - start,
            })
            if exit_code != 0:
                failure = {
                    "failed_fold": year,
                    "exit_code": exit_code,
                    "reason": exit_reason,
                    "stdout_path": stdout_path.relative_to(artifact_root).as_posix(),
                    "stdout_sha256": file_sha256(stdout_path),
                    "stderr_path": stderr_path.relative_to(artifact_root).as_posix(),
                    "stderr_sha256": file_sha256(stderr_path),
                }
                break
    finally:
        os.close(descriptor)
        lock_path.unlink()
        write_canonical_json(artifact_root / "gpu-lease-released.json", {
            "pid": os.getpid(), "released_unix_ns": time.time_ns(), "elapsed_seconds": time.monotonic() - start,
        })
    status = "failed" if failure else "complete"
    return _finalize(campaign_root, artifact_root, status=status, failure=failure)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    subs = parser.add_subparsers(dest="command", required=True)
    pre = subs.add_parser("preregister")
    pre.add_argument("--campaign", type=Path, required=True)
    launch_parser = subs.add_parser("launch")
    launch_parser.add_argument("--campaign", type=Path, required=True)
    launch_parser.add_argument("--deadline-seconds", type=int, default=8400)
    fold = subs.add_parser("fit-fold")
    fold.add_argument("--campaign", type=Path, required=True)
    fold.add_argument("--artifact-root", type=Path, required=True)
    fold.add_argument("--year", type=int, required=True)
    return parser


def main() -> int:
    args = _parser().parse_args()
    if args.command == "preregister":
        result = preregister(args.campaign)
    elif args.command == "launch":
        result = launch(args.campaign, deadline_seconds=args.deadline_seconds)
    else:
        fit_fold(args.campaign, args.artifact_root, args.year)
        result = {"status": "complete", "year": args.year}
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
