"""Append-only campaign registry with byte-level prefix commitments."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from .hashing import (
    canonical_json_bytes,
    canonical_sha256 as _canonical_sha256,
    read_json,
    write_canonical_json,
)


CAMPAIGN_FAMILY_IDS = (
    "family-01-weighted-v8-control",
    "family-02-horizon-recency",
    "family-03-temporal-calibration",
    "family-04-chronological-oof-ensemble",
    "family-05-stable-semantic-portfolio",
    "family-06-multiscale-count-aware-state",
    "family-07-matchup-swap-geometry",
    "family-08-catboost-native-specialist",
    "family-09-capacity-foundation-context",
    "family-10-outcome-decomposition",
)

FAMILY_2_VARIANT_IDS = (
    "expanding-decay-0",
    "expanding-decay-0.05",
    "rolling-8y-decay-0.10",
    "rolling-6y-decay-0.15",
    "rolling-4y-decay-0.25",
    "expanding-piecewise-event-count",
    "rolling-8y-decay-0",
    "expanding-decay-0.15",
)

RESOLVED_PROFILE_FIELDS = (
    "model_type",
    "preset",
    "time_limit",
    "test_size",
    "val_date",
    "features",
    "included_strings",
    "excluded_strings",
    "required_strings",
    "start_date",
    "num_fights",
    "include_split_dec",
    "normalize",
    "use_recency_weights",
    "decay_rate",
    "calculate_importance",
    "included_model_types",
    "split_strategy",
    "walkforward_n_windows",
    "walkforward_initial_year",
    "timeseries_split",
    "refit_all",
    "refit_full",
)

FAMILY_2_PROFILE_FIELDS = (
    "experiment_id",
    "family_number",
    "base_training_profile",
    "joint_variants",
    "outer_years",
    "embargo_days",
    "selection_metric",
    "selection_tie_break",
    "selection_evidence",
    "per_fit_time_cap_seconds",
    "family_deadline_seconds",
    "early_stop_rule",
    "seeds",
    "bootstrap",
    "promotion_rule",
    "adaptive_emphasis",
)

TERMINAL_STATES = {"complete", "failed", "cancelled", "limited", "superseded"}
ZERO_HASH = "0" * 64


class RegistryError(ValueError):
    pass


@dataclass(frozen=True)
class RegistryValidation:
    record_count: int
    family_ids: tuple[str, ...]
    registry_prefix_sha256: str


def canonical_sha256(value: Any) -> str:
    return _canonical_sha256(value)


def validate_resolved_profile(
    profile: Mapping[str, Any],
    *,
    required_fields: tuple[str, ...] | None = RESOLVED_PROFILE_FIELDS,
) -> str:
    if profile.get("experiment_id") == CAMPAIGN_FAMILY_IDS[1]:
        missing = sorted(set(FAMILY_2_PROFILE_FIELDS) - set(profile))
        extra = sorted(set(profile) - set(FAMILY_2_PROFILE_FIELDS))
        if missing or extra:
            raise RegistryError(
                "family 2 profile must be fully materialized; "
                f"missing={missing or []}, extra={extra or []}"
            )
        validate_resolved_profile(profile["base_training_profile"])
        variants = profile.get("joint_variants")
        if not isinstance(variants, list):
            raise RegistryError("family 2 joint_variants must be an ordered list")
        if tuple(variant.get("id") for variant in variants) != FAMILY_2_VARIANT_IDS:
            raise RegistryError("family 2 profile does not contain the frozen joint menu")
        if profile.get("outer_years") != [2022, 2023, 2024, 2025]:
            raise RegistryError("family 2 profile must use the four frozen outer folds")
        if profile.get("per_fit_time_cap_seconds", 0) <= 0:
            raise RegistryError("family 2 per-fit cap must be positive")
        return _canonical_sha256(profile)
    if required_fields is not None:
        missing = sorted(set(required_fields) - set(profile))
        extra = sorted(set(profile) - set(required_fields))
        if missing or extra:
            raise RegistryError(
                "profile must be fully materialized; "
                f"missing={missing or []}, extra={extra or []}"
            )
    if "extends" in profile:
        raise RegistryError("profile must be fully materialized, not inheritance-only")
    if not isinstance(profile.get("features"), list):
        raise RegistryError("fully materialized profile must contain an ordered features list")
    return _canonical_sha256(profile)


def initialize_registry(campaign_root: Path) -> None:
    campaign_root = Path(campaign_root)
    campaign_root.mkdir(parents=True, exist_ok=True)
    registry = campaign_root / "registry.jsonl"
    head = campaign_root / "registry-head.json"
    if registry.exists() or head.exists():
        raise RegistryError("registry destination already exists")
    registry.write_bytes(b"")
    _write_head(head, b"", record_count=0, last_record_sha256=ZERO_HASH)


def _write_head(path: Path, registry_bytes: bytes, *, record_count: int, last_record_sha256: str) -> None:
    write_canonical_json(
        path,
        {
            "record_count": record_count,
            "registry_bytes": len(registry_bytes),
            "registry_prefix_sha256": hashlib.sha256(registry_bytes).hexdigest().upper(),
            "last_record_sha256": last_record_sha256,
        },
    )


def _resolve(campaign_root: Path, relative: str) -> Path:
    candidate = (campaign_root / relative).resolve()
    try:
        candidate.relative_to(campaign_root.resolve())
    except ValueError as exc:
        raise RegistryError(f"path escapes campaign: {relative}") from exc
    return candidate


def _load_lines(registry_path: Path) -> list[tuple[bytes, dict[str, Any]]]:
    result = []
    for line_number, raw in enumerate(registry_path.read_bytes().splitlines(keepends=True), start=1):
        if not raw.endswith(b"\n"):
            raise RegistryError(f"registry line {line_number} is not newline terminated")
        try:
            record = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise RegistryError(f"registry line {line_number} is invalid JSON") from exc
        if raw != canonical_json_bytes(record) + b"\n":
            raise RegistryError(f"registry line {line_number} is not canonical")
        result.append((raw, record))
    return result


def _verify_profile_and_manifest(campaign_root: Path, payload: Mapping[str, Any]) -> None:
    for noun in ("profile", "manifest"):
        path_key = f"{noun}_path"
        hash_key = f"{noun}_sha256"
        if path_key not in payload or hash_key not in payload:
            raise RegistryError(f"record missing {noun} identity")
        path = _resolve(campaign_root, str(payload[path_key]))
        if not path.is_file():
            raise RegistryError(f"missing {noun}: {path}")
        value = read_json(path)
        actual = _canonical_sha256(value)
        if actual != payload[hash_key]:
            raise RegistryError(f"{noun} hash mismatch")
        if noun == "profile":
            validate_resolved_profile(value)
        elif value.get("experiment_id") != payload.get("experiment_id"):
            raise RegistryError("manifest experiment ID mismatch")


def _validate_payload_sequence(records: list[dict[str, Any]], payload: Mapping[str, Any]) -> None:
    experiment_id = payload.get("experiment_id")
    if any(record["payload"]["experiment_id"] == experiment_id for record in records):
        raise RegistryError(f"duplicate experiment ID: {experiment_id}")
    if payload.get("status") not in TERMINAL_STATES:
        raise RegistryError("registry record must have a terminal status")
    kind = payload.get("kind")
    if not records:
        if kind != "experiment-zero" or experiment_id != "experiment-zero":
            raise RegistryError("experiment zero must be the first and only admission record")
        return
    if kind != "family":
        raise RegistryError("experiment zero is the only non-family admission record")
    if experiment_id not in CAMPAIGN_FAMILY_IDS:
        raise RegistryError(f"family outside frozen ten: {experiment_id}")
    completed_families = [
        record["payload"]["experiment_id"]
        for record in records
        if record["payload"]["kind"] == "family"
    ]
    expected = CAMPAIGN_FAMILY_IDS[len(completed_families)]
    if experiment_id != expected:
        raise RegistryError(f"next family must be {expected}, got {experiment_id}")


def append_registry_record(campaign_root: Path, payload: Mapping[str, Any]) -> dict[str, Any]:
    campaign_root = Path(campaign_root)
    registry_path = campaign_root / "registry.jsonl"
    if not registry_path.is_file():
        raise RegistryError("registry is not initialized")
    validate_registry(campaign_root, strict=False)
    lines = _load_lines(registry_path)
    records = [record for _, record in lines]
    payload = dict(payload)
    _validate_payload_sequence(records, payload)
    _verify_profile_and_manifest(campaign_root, payload)
    reused = {
        record["payload"].get("artifact_path") for record in records
    }
    if payload.get("artifact_path") in reused:
        raise RegistryError(f"reused artifact path: {payload['artifact_path']}")
    before = registry_path.read_bytes()
    record_core = {
        "sequence": len(records),
        "prefix_sha256_before": hashlib.sha256(before).hexdigest().upper(),
        "previous_record_sha256": records[-1]["record_sha256"] if records else ZERO_HASH,
        "payload": payload,
    }
    record = {**record_core, "record_sha256": _canonical_sha256(record_core)}
    appended = before + canonical_json_bytes(record) + b"\n"
    registry_path.write_bytes(appended)
    _write_head(
        campaign_root / "registry-head.json",
        appended,
        record_count=len(records) + 1,
        last_record_sha256=record["record_sha256"],
    )
    return record


def validate_registry(campaign_root: Path, *, strict: bool) -> RegistryValidation:
    campaign_root = Path(campaign_root)
    registry_path = campaign_root / "registry.jsonl"
    head_path = campaign_root / "registry-head.json"
    if not registry_path.is_file() or not head_path.is_file():
        raise RegistryError("registry/head missing")
    registry_bytes = registry_path.read_bytes()
    lines = _load_lines(registry_path)
    previous_hash = ZERO_HASH
    prefix = b""
    family_ids: list[str] = []
    records: list[dict[str, Any]] = []
    artifact_paths: set[str] = set()
    for sequence, (raw, record) in enumerate(lines):
        if record.get("sequence") != sequence:
            raise RegistryError("registry sequence is reordered")
        if record.get("prefix_sha256_before") != hashlib.sha256(prefix).hexdigest().upper():
            raise RegistryError("registry prefix commitment mismatch")
        if record.get("previous_record_sha256") != previous_hash:
            raise RegistryError("registry chain mismatch")
        core = {key: value for key, value in record.items() if key != "record_sha256"}
        if record.get("record_sha256") != _canonical_sha256(core):
            raise RegistryError("registry record hash mismatch")
        payload = record.get("payload", {})
        _validate_payload_sequence(records, payload)
        _verify_profile_and_manifest(campaign_root, payload)
        artifact_path = payload.get("artifact_path")
        if artifact_path in artifact_paths:
            raise RegistryError(f"reused artifact path: {artifact_path}")
        artifact_paths.add(artifact_path)
        if payload.get("kind") == "family":
            family_ids.append(payload["experiment_id"])
        records.append(record)
        previous_hash = record["record_sha256"]
        prefix += raw

    head = read_json(head_path)
    expected_head = {
        "record_count": len(lines),
        "registry_bytes": len(registry_bytes),
        "registry_prefix_sha256": hashlib.sha256(registry_bytes).hexdigest().upper(),
        "last_record_sha256": previous_hash,
    }
    if head != expected_head:
        raise RegistryError("registry head does not match committed bytes")
    if strict:
        registered = {record["payload"]["experiment_id"] for record in records}
        for manifest_path in sorted((campaign_root / "runs").glob("*/manifest.json")):
            manifest = read_json(manifest_path)
            experiment_id = manifest.get("experiment_id")
            if experiment_id not in registered:
                raise RegistryError(f"unregistered run manifest: {experiment_id}")
    return RegistryValidation(
        record_count=len(lines),
        family_ids=tuple(family_ids),
        registry_prefix_sha256=expected_head["registry_prefix_sha256"],
    )
