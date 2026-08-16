"""Canonical append-only registry for the split/refit campaign."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from .protocol import canonical_json_bytes, canonical_sha256, file_sha256


ZERO_HASH = "0" * 64
THROUGH_IDS = {"rollback": ("rollback-capsule",), "split": ("rollback-capsule", "split-materialization")}


class RegistryError(ValueError):
    pass


@dataclass(frozen=True)
class RegistryValidation:
    record_count: int
    record_ids: tuple[str, ...]
    registry_prefix_sha256: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "record_count": self.record_count,
            "record_ids": self.record_ids,
            "registry_prefix_sha256": self.registry_prefix_sha256,
        }


def _write_json(path: Path, value: Any) -> None:
    path.write_bytes(canonical_json_bytes(value) + b"\n")


def _read_canonical_json(path: Path) -> Any:
    try:
        raw = path.read_bytes()
    except FileNotFoundError as exc:
        raise RegistryError(f"missing registry artifact: {path}") from exc
    try:
        value = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise RegistryError(f"invalid registry JSON: {path}") from exc
    canonical = canonical_json_bytes(value)
    if raw not in (canonical, canonical + b"\n", canonical + b"\r\n"):
        raise RegistryError(f"noncanonical registry JSON: {path}")
    return value


def _read_records(path: Path) -> list[tuple[bytes, dict[str, Any]]]:
    try:
        raw_registry = path.read_bytes()
    except FileNotFoundError as exc:
        raise RegistryError("registry is missing") from exc
    records = []
    for number, raw in enumerate(raw_registry.splitlines(keepends=True), start=1):
        if not raw.endswith(b"\n"):
            raise RegistryError(f"registry line {number} is not newline terminated")
        canonical_line = raw.replace(b"\r\n", b"\n")
        try:
            record = json.loads(canonical_line)
        except json.JSONDecodeError as exc:
            raise RegistryError(f"registry line {number} is invalid JSON") from exc
        if canonical_line != canonical_json_bytes(record) + b"\n":
            raise RegistryError(f"registry line {number} is noncanonical")
        records.append((canonical_line, record))
    return records


def _write_head(campaign_root: Path, registry_bytes: bytes, records: list[dict[str, Any]]) -> None:
    _write_json(
        campaign_root / "registry-head.json",
        {
            "record_count": len(records),
            "registry_bytes": len(registry_bytes),
            "registry_prefix_sha256": hashlib.sha256(registry_bytes).hexdigest().upper(),
            "last_record_sha256": records[-1]["record_sha256"] if records else ZERO_HASH,
        },
    )


def append_registry_record(
    campaign_root: Path, *, record_id: str, payload: Mapping[str, Any]
) -> dict[str, Any]:
    campaign_root = Path(campaign_root)
    registry_path = campaign_root / "registry.jsonl"
    if not registry_path.exists():
        registry_path.write_bytes(b"")
    lines = _read_records(registry_path)
    records = [record for _, record in lines]
    if any(record.get("record_id") == record_id for record in records):
        raise RegistryError(f"duplicate registry record ID: {record_id}")
    physical_before = registry_path.read_bytes()
    before = b"".join(raw for raw, _ in lines)
    core = {
        "sequence": len(records),
        "record_id": record_id,
        "prefix_sha256_before": hashlib.sha256(before).hexdigest().upper(),
        "previous_record_sha256": records[-1]["record_sha256"] if records else ZERO_HASH,
        "payload": dict(payload),
    }
    record = {**core, "record_sha256": canonical_sha256(core)}
    physical_appended = physical_before + canonical_json_bytes(record) + b"\n"
    canonical_appended = before + canonical_json_bytes(record) + b"\n"
    registry_path.write_bytes(physical_appended)
    _write_head(campaign_root, canonical_appended, [*records, record])
    return record


def initialize_split_registry(
    campaign_root: Path, *, split_manifest_sha256: str, profile_sha256: str
) -> None:
    campaign_root = Path(campaign_root)
    registry_path = campaign_root / "registry.jsonl"
    head_path = campaign_root / "registry-head.json"
    if registry_path.exists() or head_path.exists():
        raise RegistryError("append-only registry destination already exists")
    append_registry_record(
        campaign_root,
        record_id="rollback-capsule",
        payload={
            "kind": "rollback",
            "artifact_path": "rollback-manifest.json",
            "artifact_sha256": file_sha256(campaign_root / "rollback-manifest.json"),
        },
    )
    append_registry_record(
        campaign_root,
        record_id="split-materialization",
        payload={
            "kind": "split",
            "artifact_path": "partitions/manifest.json",
            "artifact_sha256": split_manifest_sha256,
            "profile_path": "profiles/evaluation.json",
            "profile_sha256": profile_sha256,
        },
    )


def _resolve(campaign_root: Path, relative: str) -> Path:
    candidate = (campaign_root / relative).resolve()
    try:
        candidate.relative_to(campaign_root.resolve())
    except ValueError as exc:
        raise RegistryError(f"registry path escapes campaign: {relative}") from exc
    return candidate


def _matches_registered_sha256(path: Path, expected: str) -> bool:
    raw = path.read_bytes()
    return expected in {
        hashlib.sha256(raw).hexdigest().upper(),
        hashlib.sha256(raw.replace(b"\r\n", b"\n")).hexdigest().upper(),
    }


def validate_registry(
    campaign_root: Path, *, strict: bool, through: str = "split"
) -> RegistryValidation:
    campaign_root = Path(campaign_root)
    expected_ids = THROUGH_IDS.get(through)
    if expected_ids is None:
        raise RegistryError(f"unsupported registry phase: {through}")
    lines = _read_records(campaign_root / "registry.jsonl")
    records: list[dict[str, Any]] = []
    prefix = b""
    previous_hash = ZERO_HASH
    ids: list[str] = []
    for sequence, (raw, record) in enumerate(lines):
        if record.get("sequence") != sequence:
            raise RegistryError("registry records are reordered")
        record_id = record.get("record_id")
        if not isinstance(record_id, str) or record_id in ids:
            raise RegistryError("duplicate or missing registry record ID")
        if record.get("prefix_sha256_before") != hashlib.sha256(prefix).hexdigest().upper():
            raise RegistryError("registry prefix commitment mismatch")
        if record.get("previous_record_sha256") != previous_hash:
            raise RegistryError("registry previous-record commitment mismatch")
        core = {key: value for key, value in record.items() if key != "record_sha256"}
        if record.get("record_sha256") != canonical_sha256(core):
            raise RegistryError("registry record hash mismatch")
        payload = record.get("payload")
        if not isinstance(payload, dict):
            raise RegistryError("registry payload is missing")
        artifact_path = payload.get("artifact_path")
        artifact_sha256 = payload.get("artifact_sha256")
        if not isinstance(artifact_path, str) or not isinstance(artifact_sha256, str):
            raise RegistryError("registry artifact identity is missing")
        artifact = _resolve(campaign_root, artifact_path)
        if not _matches_registered_sha256(artifact, artifact_sha256):
            raise RegistryError(f"registered artifact hash mismatch: {artifact_path}")
        if payload.get("kind") == "split":
            profile = _resolve(campaign_root, str(payload.get("profile_path")))
            if not _matches_registered_sha256(profile, str(payload.get("profile_sha256"))):
                raise RegistryError("registered profile hash mismatch")
        ids.append(record_id)
        records.append(record)
        previous_hash = record["record_sha256"]
        prefix += raw
    if tuple(ids) != expected_ids:
        raise RegistryError(f"registry through {through} must contain exactly {expected_ids}")
    expected_head = {
        "record_count": len(records),
        "registry_bytes": len(prefix),
        "registry_prefix_sha256": hashlib.sha256(prefix).hexdigest().upper(),
        "last_record_sha256": previous_hash,
    }
    if _read_canonical_json(campaign_root / "registry-head.json") != expected_head:
        raise RegistryError("registry head does not match append-only bytes")
    if strict:
        for record in records:
            artifact = _resolve(campaign_root, record["payload"]["artifact_path"])
            _read_canonical_json(artifact)
    return RegistryValidation(
        record_count=len(records),
        record_ids=tuple(ids),
        registry_prefix_sha256=expected_head["registry_prefix_sha256"],
    )
