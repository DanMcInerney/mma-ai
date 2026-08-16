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


class BranchVerificationError(ValueError):
    pass


EXPECTED_BRANCH_REVISIONS = {
    "codex/weighted-v8-67-baseline": "545441975b86caf0abb6136e099e44e6b93caf22",
    "codex/exp-80-10-10-v8-20260816": "7217012abcee3c22937dd378c0a904033564018d",
    "codex/exp-full-refit-v8-20260816": "70559ac40300c62067f23b335050dda3e4931ce6",
}


def validate_branch_documents(
    revisions: Mapping[str, str],
    merge_bases: Mapping[str, str],
    worktrees: Mapping[str, str],
) -> None:
    if dict(revisions) != EXPECTED_BRANCH_REVISIONS:
        raise BranchVerificationError("branch target revisions changed")
    rollback = EXPECTED_BRANCH_REVISIONS["codex/weighted-v8-67-baseline"]
    experiments = (
        "codex/exp-80-10-10-v8-20260816",
        "codex/exp-full-refit-v8-20260816",
    )
    if set(merge_bases) != set(experiments) or any(
        merge_bases[name] != rollback for name in experiments
    ):
        raise BranchVerificationError("experiment merge base changed")
    if set(worktrees) != set(EXPECTED_BRANCH_REVISIONS):
        raise BranchVerificationError("branch worktree mapping is incomplete")
    normalized = [str(Path(worktrees[name]).resolve()).lower() for name in EXPECTED_BRANCH_REVISIONS]
    if len(normalized) != len(set(normalized)):
        raise BranchVerificationError("branch worktrees are not distinct")


def _artifact_worktree(model_root: str) -> str:
    normalized = str(model_root).replace("/", "\\")
    marker = "\\experiments\\split_refit_20260816\\"
    index = normalized.lower().find(marker.lower())
    if index < 0:
        raise BranchVerificationError("model artifact is not inside the campaign worktree")
    return normalized[:index]


def verify_branches(campaign_root: Path, *, repo: Path, strict: bool) -> dict[str, Any]:
    if not strict:
        raise BranchVerificationError("branch verification requires --strict")
    campaign_root = Path(campaign_root).resolve()
    repo = Path(repo).resolve()
    revisions = {
        name: _git("rev-parse", name, cwd=repo) for name in EXPECTED_BRANCH_REVISIONS
    }
    rollback_name = "codex/weighted-v8-67-baseline"
    merge_bases = {
        name: _git("merge-base", rollback_name, name, cwd=repo)
        for name in EXPECTED_BRANCH_REVISIONS
        if name != rollback_name
    }
    rollback = _read_json(campaign_root / "rollback-manifest.json")
    selection = _read_json(campaign_root / "runs/80-10-10-evaluation/selection.json")
    refit = _read_json(campaign_root / "runs/full-data-refit/refit-lineage-correction.json")
    worktrees = {
        rollback_name: rollback["rollback"]["worktree"],
        "codex/exp-80-10-10-v8-20260816": _artifact_worktree(selection["model_root"]),
        "codex/exp-full-refit-v8-20260816": _artifact_worktree(refit["model_root"]),
    }
    validate_branch_documents(revisions, merge_bases, worktrees)
    rollback_root = Path(worktrees[rollback_name])
    if _git("rev-parse", "HEAD", cwd=rollback_root) != EXPECTED_BRANCH_REVISIONS[rollback_name]:
        raise BranchVerificationError("rollback worktree HEAD changed")
    if _git("rev-parse", "HEAD^{tree}", cwd=rollback_root) != rollback["rollback"]["tree"]:
        raise BranchVerificationError("rollback worktree tree changed")
    if _git("status", "--porcelain", cwd=rollback_root):
        raise BranchVerificationError("rollback worktree is dirty")
    preserved_original = refit["preservation_before"]["original_checkout"]
    if _original_checkout_identity() != preserved_original:
        raise BranchVerificationError("original checkout manifest changed")
    return {
        "status": "PASS",
        "revisions": revisions,
        "merge_bases": merge_bases,
        "worktrees": worktrees,
        "rollback_tree": rollback["rollback"]["tree"],
        "original_checkout": preserved_original,
    }


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

    recovery = result.get("post_fit_evidence_recovery")
    try:
        assert_single_refit_attempt(attempts, require_success=recovery is None)
    except RefitError as exc:
        raise RefitVerificationError(str(exc)) from exc
    if recovery is not None and (
        attempts[-1].get("exit_code") != 1
        or recovery.get("training_completed") is not True
        or recovery.get("refit_full_completed") is not True
        or recovery.get("failure_preserved") is not True
        or recovery.get("retry_count") != 0
    ):
        raise RefitVerificationError("post-fit evidence recovery contract changed")
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
        "post_fit_evidence_recovery": recovery is not None,
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


def _validate_refit_registry(campaign_root: Path) -> dict[str, Any]:
    from .refit import EXPECTED_REGISTRY_PREFIX

    path = campaign_root / "registry.jsonl"
    lines = [line.replace(b"\r\n", b"\n") for line in path.read_bytes().splitlines(keepends=True)]
    records = [json.loads(line) for line in lines]
    expected_ids = [
        "rollback-capsule",
        "split-materialization",
        "evaluation-selection",
        "evaluation-result",
        "full-data-refit-failure",
        "full-data-refit-recovery",
        "full-data-refit-lineage-correction",
    ]
    if [record.get("record_id") for record in records] != expected_ids:
        raise RefitVerificationError("registry does not have the exact full-refit sequence")
    if hashlib.sha256(b"".join(lines[:4])).hexdigest().upper() != EXPECTED_REGISTRY_PREFIX:
        raise RefitVerificationError("post-evaluation registry prefix changed")
    prefix = b""
    previous = "0" * 64
    for sequence, (raw, record) in enumerate(zip(lines, records, strict=True)):
        core = {key: value for key, value in record.items() if key != "record_sha256"}
        if (
            record.get("sequence") != sequence
            or record.get("prefix_sha256_before") != hashlib.sha256(prefix).hexdigest().upper()
            or record.get("previous_record_sha256") != previous
            or record.get("record_sha256") != canonical_sha256(core)
        ):
            raise RefitVerificationError("full-refit registry chain changed")
        artifact = (campaign_root / record["payload"]["artifact_path"]).resolve()
        try:
            artifact.relative_to(campaign_root.resolve())
        except ValueError as exc:
            raise RefitVerificationError("full-refit registry artifact escapes campaign") from exc
        raw_artifact = artifact.read_bytes()
        artifact_hashes = {
            hashlib.sha256(raw_artifact).hexdigest().upper(),
            hashlib.sha256(raw_artifact.replace(b"\r\n", b"\n")).hexdigest().upper(),
        }
        if record["payload"]["artifact_sha256"] not in artifact_hashes:
            raise RefitVerificationError("full-refit registry artifact identity changed")
        prefix += raw
        previous = record["record_sha256"]
    head = _read_json(campaign_root / "registry-head.json")
    expected_head = {
        "record_count": len(records),
        "registry_bytes": len(prefix),
        "registry_prefix_sha256": hashlib.sha256(prefix).hexdigest().upper(),
        "last_record_sha256": previous,
    }
    _same(head, expected_head, "full-refit registry head")
    return {"record_ids": expected_ids, "prefix_sha256": expected_head["registry_prefix_sha256"]}


def _refit_git_scope(repo: Path) -> list[str]:
    from .refit import BASELINE_REVISION

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
        "libs/modeling/split_refit_experiment/refit.py",
        "libs/modeling/split_refit_experiment/verification.py",
        "tests/test_modeling/test_split_refit_refit.py",
        "tests/test_modeling/test_split_refit_verification.py",
        "experiments/split_refit_20260816/registry.jsonl",
        "experiments/split_refit_20260816/registry-head.json",
    }
    allowed_prefix = "experiments/split_refit_20260816/runs/full-data-refit/"
    unexpected = [path for path in paths if path not in allowed_files and not path.startswith(allowed_prefix)]
    if unexpected:
        raise RefitVerificationError(f"full-refit result paths exceed ticket scope: {unexpected}")
    return paths


def verify_refit(campaign_root: Path, *, recompute_lineage: bool) -> dict[str, Any]:
    if not recompute_lineage:
        raise RefitVerificationError("strict refit verification requires --recompute-lineage")
    from autogluon.tabular import TabularPredictor

    from .refit import (
        FROZEN_SOURCE_SHA256,
        PROFILE_NAME,
        _native_tree,
        _paths as refit_paths,
        _preservation_snapshot,
        _standard_model_info,
        classify_saved_lineage,
        prediction_identities,
    )

    campaign_root = Path(campaign_root).resolve()
    paths = refit_paths(campaign_root)
    preflight = _read_json(paths["preflight"])
    prereg = _read_json(paths["preregistration"])
    attempts = read_jsonl(paths["attempts"])
    result_path = paths["correction"] if paths["correction"].is_file() else paths["result"]
    result = _read_json(result_path)
    verified = validate_refit_documents(attempts=attempts, result=result)
    if result.get("profile_name") != PROFILE_NAME or result.get("profile") != prereg.get("profile"):
        raise RefitVerificationError("saved refit profile differs from preregistration")
    if result.get("profile_sha256") != prereg.get("profile_sha256"):
        raise RefitVerificationError("saved refit profile hash changed")
    if result.get("feature_sha256") != prereg.get("feature_sha256"):
        raise RefitVerificationError("saved refit feature hash changed")
    source = Path(prereg["source_csv_path"])
    if file_sha256(source) != FROZEN_SOURCE_SHA256:
        raise RefitVerificationError("sealed source CSV changed")
    model_root = Path(result["model_root"])
    complete_before = tree_identity(model_root)
    if complete_before != result.get("complete_tree"):
        raise RefitVerificationError("registered complete model tree changed")
    if _native_tree(model_root) != result.get("native_tree"):
        raise RefitVerificationError("registered native model tree changed")
    predictor = TabularPredictor.load(str(model_root))
    if str(predictor.model_best) != result.get("selected_node"):
        raise RefitVerificationError("loaded full-refit selected node changed")
    names = [str(name) for name in predictor.model_names()]
    if names != [node["name"] for node in result.get("model_graph", [])]:
        raise RefitVerificationError("loaded full-refit model graph changed")
    model_info = _standard_model_info(predictor)
    prediction_hashes, probe = prediction_identities(predictor, model_root)
    lineage = classify_saved_lineage(model_info, prediction_hashes, total_rows=3267)
    _same(prediction_hashes, result.get("prediction_hashes"), "saved-node prediction identities")
    _same(probe, result.get("prediction_probe"), "saved-node prediction probe")
    _same(lineage, result.get("lineage"), "saved-node fit lineage")
    if tree_identity(model_root) != complete_before:
        raise RefitVerificationError("predictor load/lineage recomputation mutated model bytes")
    if file_sha256(Path(result["scaler_path"])) != result.get("scaler_sha256"):
        raise RefitVerificationError("refit scaler identity changed")
    preservation_after = _preservation_snapshot(
        campaign_root,
        _read_json(campaign_root / "runs/80-10-10-evaluation/selection.json"),
    )
    _same(preservation_after, preflight.get("preservation"), "prior-artifact preservation")
    for noun in ("stdout", "stderr"):
        log = Path(attempts[-1][f"{noun}_path"])
        if file_sha256(log) != attempts[-1][f"{noun}_sha256"]:
            raise RefitVerificationError(f"refit {noun} log identity changed")
    forbidden_hits = []
    for path in (paths["preflight"], paths["preregistration"], paths["attempts"], paths["result"]):
        if has_database_token(path.read_text(encoding="utf-8")):
            forbidden_hits.append(str(path))
    if forbidden_hits:
        raise RefitVerificationError(f"database tokens found in refit evidence: {forbidden_hits}")
    if gpu_process_snapshot()["python_rows"]:
        raise RefitVerificationError("training process remained active after refit")
    registry = _validate_refit_registry(campaign_root)
    repo = Path.cwd().resolve()
    changed = _refit_git_scope(repo)
    diff = subprocess.run(
        ["git", "diff", "--check", f"{BASELINE_REVISION}..HEAD"],
        cwd=repo,
        capture_output=True,
        text=True,
    )
    if diff.returncode:
        raise RefitVerificationError(f"full-refit whitespace verification failed: {diff.stdout}{diff.stderr}")
    return {
        "status": "PASS",
        **verified,
        "selected_node": predictor.model_best,
        "classes": result["classes"],
        "model_count": len(names),
        "complete_tree_sha256": complete_before["sha256"],
        "native_tree_sha256": result["native_tree"]["sha256"],
        "scaler_sha256": result["scaler_sha256"],
        "lineage": lineage,
        "registry": registry,
        "changed_paths": changed,
        "preservation": preservation_after,
        "validation_claims": [],
        "database_tokens": [],
    }
