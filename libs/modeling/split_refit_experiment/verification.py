"""Strict replay of one-shot evaluation evidence."""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

from libs.modeling.experiment_campaign.metrics import reduce_predictions

from .protocol import canonical_json_bytes, canonical_sha256, file_sha256
from .runner import (
    BASELINE_REVISION,
    EXPECTED_BASE_MODELS,
    EXPECTED_PROFILE_SHA256,
    EXPECTED_REGISTRY_PREFIX,
    FROZEN_SOURCE_SHA256,
    RUN_ID,
    EvaluationError,
    _direct_event_intervals,
    _git,
    _original_checkout_identity,
    _paths,
    _read_json,
    assert_production_grammar,
    gpu_process_snapshot,
    read_jsonl,
    tree_identity,
)


class EvaluationVerificationError(ValueError):
    pass


class RefitVerificationError(ValueError):
    pass


def has_database_token(text: str) -> bool:
    lowered = text.lower()
    return "clankerfights" in lowered or bool(
        re.search(r"(?:localhost\s*:|\bport\s*=\s*)5432\b", lowered)
    )


def validate_loaded_predictor(predictor: Any, selection: Mapping[str, Any]) -> None:
    if str(predictor.model_best) != selection.get("selected_node"):
        raise EvaluationVerificationError("loaded predictor selected node changed")
    names = [str(name) for name in predictor.model_names()]
    base = [name for name in names if not name.startswith("WeightedEnsemble")]
    if base != selection.get("base_models") or tuple(base) != EXPECTED_BASE_MODELS:
        raise EvaluationVerificationError("loaded predictor graph changed")
    if any("_FULL" in name.upper() or "CONTEXT" in name.upper() for name in names):
        raise EvaluationVerificationError("loaded predictor contains FULL/context nodes")


def _same(actual: Any, expected: Any, noun: str) -> None:
    if canonical_sha256(actual) != canonical_sha256(expected):
        raise EvaluationVerificationError(f"{noun} does not recompute")


def validate_evaluation_documents(
    *,
    selection: Mapping[str, Any],
    attempts: Sequence[Mapping[str, Any]],
    access: Sequence[Mapping[str, Any]],
    predictions: Sequence[Mapping[str, Any]],
    result: Mapping[str, Any],
    expected_count: int,
) -> dict[str, Any]:
    try:
        assert_production_grammar(attempts, access, selection_frozen=bool(selection), require_access=True)
    except EvaluationError as exc:
        raise EvaluationVerificationError(str(exc)) from exc
    if attempts[-1].get("exit_code") != 0:
        raise EvaluationVerificationError("successful evaluation cannot cite a failed fit")
    if len(predictions) != expected_count or result.get("row_count") != expected_count:
        raise EvaluationVerificationError("evaluation row count changed")
    ids = [str(row.get("fight_id")) for row in predictions]
    if len(ids) != len(set(ids)) or access[0].get("fight_ids") != ids:
        raise EvaluationVerificationError("prediction/access order or identity changed")
    if access[0].get("row_count") != expected_count:
        raise EvaluationVerificationError("test access row count changed")
    selected = str(selection.get("selected_node", ""))
    graph_names = [selected, *[str(name) for name in selection.get("base_models", [])]]
    if any("_FULL" in name.upper() or "CONTEXT" in name.upper() for name in graph_names):
        raise EvaluationVerificationError("FULL/context selection is inadmissible")
    if selection.get("selection_uses_full_or_context_metrics") not in (None, False):
        raise EvaluationVerificationError("selection cites inadmissible metrics")
    for record in predictions:
        if record.get("boundary") != "Original" or record.get("fit_scope") != "prior-only":
            raise EvaluationVerificationError("prediction boundary changed")
    metrics = reduce_predictions(predictions).as_dict()
    if metrics["log_loss"] <= 0:
        raise EvaluationVerificationError("positive log loss must be reported")
    _same(metrics, result.get("metrics"), "direct metrics")
    boundaries = result.get("historical_boundaries", {})
    if (
        boundaries.get("retrospective_test") != expected_count
        or boundaries.get("accepted_tuning") != 460
        or boundaries.get("nested_outer") != 1108
        or boundaries.get("pooled") is not False
    ):
        raise EvaluationVerificationError("historical denominators are mixed")
    if result.get("post_test_adaptation") is not False:
        raise EvaluationVerificationError("post-test adaptation is forbidden")
    return {"row_count": expected_count, "metrics": metrics, "prediction_ids_sha256": canonical_sha256(ids)}


def validate_refit_documents(
    *, attempts: Sequence[Mapping[str, Any]], result: Mapping[str, Any]
) -> dict[str, Any]:
    from .refit import RefitError, assert_single_refit_attempt

    try:
        assert_single_refit_attempt(attempts, require_success=True)
    except RefitError as exc:
        raise RefitVerificationError(str(exc)) from exc
    if result.get("state") != "complete":
        raise RefitVerificationError("refit result is not complete")
    if result.get("profile_name") != "v8-hybrid-weighted":
        raise RefitVerificationError("refit did not use the exact named profile")
    if result.get("source_rows") != 3267 or result.get("feature_count") != 40:
        raise RefitVerificationError("refit population or feature count changed")
    if result.get("fit_invocation_count") != 1:
        raise RefitVerificationError("refit invocation count changed")
    if result.get("validation_claims") != []:
        raise RefitVerificationError("full/context result makes a validation claim")
    if result.get("database_access") is not False:
        raise RefitVerificationError("refit evidence reports database access")
    lineage = result.get("lineage")
    if not isinstance(lineage, Mapping) or not lineage:
        raise RefitVerificationError("refit lineage is missing")
    for name, node in lineage.items():
        if node.get("metric_claim") != "none" or node.get("context_contaminated") is not True:
            raise RefitVerificationError(f"unqualified context boundary: {name}")
        if node.get("boundary") not in {"Original", "FULL"}:
            raise RefitVerificationError(f"unknown saved-node boundary: {name}")
        if node.get("origin") == "fresh-full-fit" and node.get("fit_rows") != 3267:
            raise RefitVerificationError(f"fresh FULL fit row count changed: {name}")
        if node.get("origin") == "original-clone" and node.get("fit_rows") == 3267:
            raise RefitVerificationError(f"clone is falsely labeled full-row: {name}")
    return {
        "source_rows": 3267,
        "feature_count": 40,
        "validation_claims": [],
        "node_count": len(lineage),
    }


def _validate_registry(campaign_root: Path) -> dict[str, Any]:
    path = campaign_root / "registry.jsonl"
    raw_lines = path.read_bytes().splitlines(keepends=True)
    canonical_lines = [line.replace(b"\r\n", b"\n") for line in raw_lines]
    records = [json.loads(line) for line in canonical_lines]
    expected_ids = ["rollback-capsule", "split-materialization", "evaluation-selection", "evaluation-result"]
    if [row.get("record_id") for row in records] != expected_ids:
        raise EvaluationVerificationError("registry does not have exact evaluation prefix")
    prefix = b""
    previous = "0" * 64
    for sequence, (raw, record) in enumerate(zip(canonical_lines, records, strict=True)):
        core = {key: value for key, value in record.items() if key != "record_sha256"}
        if (
            record.get("sequence") != sequence
            or record.get("prefix_sha256_before") != hashlib.sha256(prefix).hexdigest().upper()
            or record.get("previous_record_sha256") != previous
            or record.get("record_sha256") != canonical_sha256(core)
        ):
            raise EvaluationVerificationError("registry chain changed")
        artifact = (campaign_root / record["payload"]["artifact_path"]).resolve()
        try:
            artifact.relative_to(campaign_root.resolve())
        except ValueError as exc:
            raise EvaluationVerificationError("registry artifact escapes campaign") from exc
        expected_hash = record["payload"]["artifact_sha256"]
        raw_artifact = artifact.read_bytes()
        hashes = {
            hashlib.sha256(raw_artifact).hexdigest().upper(),
            hashlib.sha256(raw_artifact.replace(b"\r\n", b"\n")).hexdigest().upper(),
        }
        if expected_hash not in hashes:
            raise EvaluationVerificationError("registry artifact hash changed")
        prefix += raw
        previous = record["record_sha256"]
    initial = hashlib.sha256(b"".join(canonical_lines[:2])).hexdigest().upper()
    if initial != EXPECTED_REGISTRY_PREFIX:
        raise EvaluationVerificationError("frozen registry prefix changed")
    head = _read_json(campaign_root / "registry-head.json")
    expected_head = {
        "record_count": 4,
        "registry_bytes": len(prefix),
        "registry_prefix_sha256": hashlib.sha256(prefix).hexdigest().upper(),
        "last_record_sha256": previous,
    }
    _same(head, expected_head, "registry head")
    return {"record_ids": expected_ids, "prefix_sha256": expected_head["registry_prefix_sha256"]}


def _git_scope(repo: Path) -> list[str]:
    completed = subprocess.run(
        ["git", "diff", "--name-only", f"{BASELINE_REVISION}..HEAD"],
        cwd=repo,
        capture_output=True,
        text=True,
        check=True,
    )
    paths = [line.strip().replace("\\", "/") for line in completed.stdout.splitlines() if line.strip()]
    allowed_files = {
        "libs/modeling/split_refit_experiment/__main__.py",
        "libs/modeling/split_refit_experiment/runner.py",
        "libs/modeling/split_refit_experiment/verification.py",
        "tests/test_modeling/test_split_refit_runner.py",
        "tests/test_modeling/test_split_refit_verification.py",
        "experiments/split_refit_20260816/registry.jsonl",
        "experiments/split_refit_20260816/registry-head.json",
    }
    allowed_prefix = "experiments/split_refit_20260816/runs/80-10-10-evaluation/"
    unexpected = [path for path in paths if path not in allowed_files and not path.startswith(allowed_prefix)]
    if unexpected:
        raise EvaluationVerificationError(f"result paths exceed ticket scope: {unexpected}")
    return paths


def verify_evaluation(campaign_root: Path, *, recompute_all: bool) -> dict[str, Any]:
    if not recompute_all:
        raise EvaluationVerificationError("strict evidence verification requires --recompute-all")
    campaign_root = Path(campaign_root).resolve()
    paths = _paths(campaign_root)
    preflight = _read_json(paths["preflight"])
    prereg = _read_json(paths["preregistration"])
    selection = _read_json(paths["selection"])
    attempts = read_jsonl(paths["attempts"])
    access = read_jsonl(paths["access"])
    predictions = read_jsonl(paths["predictions"])
    result = _read_json(paths["result"])
    if selection.get("base_models") != list(EXPECTED_BASE_MODELS):
        raise EvaluationVerificationError("selection did not resolve exact ten-model portfolio")
    verified = validate_evaluation_documents(
        selection=selection,
        attempts=attempts,
        access=access,
        predictions=predictions,
        result=result,
        expected_count=307,
    )
    if access[0].get("selection_sha256") != file_sha256(paths["selection"]):
        raise EvaluationVerificationError("test access selection hash changed")
    if access[0].get("label_decode_count") != 307:
        raise EvaluationVerificationError("test label access did not decode exactly 307 labels")
    if attempts[-1].get("exited_unix_ns", 0) >= access[0].get("opened_unix_ns", 0):
        raise EvaluationVerificationError("test access did not occur after fit exit")
    for noun in ("stdout", "stderr"):
        log_path = Path(attempts[-1][f"{noun}_path"])
        if file_sha256(log_path) != attempts[-1][f"{noun}_sha256"]:
            raise EvaluationVerificationError(f"fit {noun} log changed")
    repo = Path.cwd().resolve()
    selection_relative = paths["selection"].resolve().relative_to(repo).as_posix()
    selection_commit = subprocess.check_output(
        ["git", "log", "-1", "--format=%H", "--", selection_relative], cwd=repo, text=True
    ).strip()
    if not selection_commit or access[0].get("selection_commit") != selection_commit:
        raise EvaluationVerificationError("test access did not cite committed selection identity")
    if file_sha256(paths["predictions"]) != result.get("prediction_sha256"):
        raise EvaluationVerificationError("prediction bytes changed")
    intervals = _direct_event_intervals(predictions, iterations=2000, seed=20260816)
    _same(intervals, result.get("event_block_intervals"), "event-block intervals")
    if tree_identity(Path(selection["model_root"])) != selection.get("model_tree"):
        raise EvaluationVerificationError("post-test model bytes changed")
    from autogluon.tabular import TabularPredictor

    predictor = TabularPredictor.load(selection["model_root"])
    validate_loaded_predictor(predictor, selection)
    if tree_identity(Path(selection["model_root"])) != selection.get("model_tree"):
        raise EvaluationVerificationError("predictor load mutated frozen model bytes")
    if selection.get("profile_sha256") != EXPECTED_PROFILE_SHA256:
        raise EvaluationVerificationError("post-test profile identity changed")
    profile = _read_json(campaign_root / prereg["profile_path"])
    if canonical_sha256(profile) != EXPECTED_PROFILE_SHA256:
        raise EvaluationVerificationError("post-test profile bytes changed")
    if file_sha256(Path(prereg["source_csv_path"])) != FROZEN_SOURCE_SHA256:
        raise EvaluationVerificationError("sealed source changed")
    retired_ids = set(_read_json(campaign_root / "partitions/retired.json")["fight_ids"])
    if retired_ids.intersection(row["fight_id"] for row in predictions):
        raise EvaluationVerificationError("retired fight entered evaluation")
    if preflight.get("retired_label_reads") != 0 or preflight.get("database_access") is not False:
        raise EvaluationVerificationError("preflight safety state changed")
    if file_sha256(campaign_root / "rollback-manifest.json") != preflight["preservation"]["rollback_manifest_sha256"]:
        raise EvaluationVerificationError("rollback manifest changed")
    rollback_ref = _git("rev-parse", "codex/weighted-v8-67-baseline", cwd=repo)
    rollback_tree = _git("rev-parse", "codex/weighted-v8-67-baseline^{tree}", cwd=repo)
    if {"revision": rollback_ref, "tree": rollback_tree} != preflight.get("rollback"):
        raise EvaluationVerificationError("rollback ref/tree changed")
    if _original_checkout_identity() != preflight.get("original_checkout"):
        raise EvaluationVerificationError("original dirty checkout identity changed")
    if gpu_process_snapshot()["python_rows"]:
        raise EvaluationVerificationError("training process remained active after evaluation")
    registry = _validate_registry(campaign_root)
    changed = _git_scope(repo)
    diff = subprocess.run(
        ["git", "diff", "--check", f"{BASELINE_REVISION}..HEAD"], cwd=repo, capture_output=True, text=True
    )
    if diff.returncode:
        raise EvaluationVerificationError(f"whitespace verification failed: {diff.stdout}{diff.stderr}")
    compile_result = subprocess.run(
        [sys.executable, "-m", "compileall", "-q", "libs/modeling/split_refit_experiment"],
        cwd=repo,
        capture_output=True,
        text=True,
    )
    if compile_result.returncode:
        raise EvaluationVerificationError("evaluation package compile check failed")
    forbidden_hits = []
    for path in [paths["preflight"], paths["preregistration"], paths["attempts"], paths["access"], paths["selection"], paths["result"]]:
        text = path.read_text(encoding="utf-8")
        if has_database_token(text):
            forbidden_hits.append(str(path))
    if forbidden_hits:
        raise EvaluationVerificationError(f"database tokens found in evidence: {forbidden_hits}")
    return {
        "status": "PASS",
        **verified,
        "event_block_intervals": intervals,
        "selection_sha256": file_sha256(paths["selection"]),
        "selection_commit": selection_commit,
        "prediction_sha256": file_sha256(paths["predictions"]),
        "predictor_load": {"selected_node": predictor.model_best, "model_count": len(predictor.model_names())},
        "registry": registry,
        "changed_paths": changed,
        "preservation": {
            "source_sha256": FROZEN_SOURCE_SHA256,
            "rollback_revision": preflight["rollback"]["revision"],
            "rollback_tree": preflight["rollback"]["tree"],
            "retired_overlap": 0,
            "database_tokens": [],
        },
    }
