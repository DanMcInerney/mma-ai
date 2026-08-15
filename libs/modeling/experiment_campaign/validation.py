"""Strict recomputation of tracked campaign identities."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .hashing import canonical_sha256, file_sha256, read_json, tree_inventory
from .protocol import AccessLedger
from .registry import RegistryError, validate_registry, validate_resolved_profile


@dataclass(frozen=True)
class CampaignValidation:
    record_count: int
    registry_prefix_sha256: str
    baseline_manifest_sha256: str
    artifact_tree_sha256: str
    gate_state: str
    protected_gate_access_count: int


def validate_campaign(campaign_root: Path, *, strict: bool) -> CampaignValidation:
    campaign_root = Path(campaign_root)
    registry = validate_registry(campaign_root, strict=strict)
    baseline_path = campaign_root / "baseline" / "manifest.json"
    baseline = read_json(baseline_path)
    if baseline.get("experiment_id") != "experiment-zero":
        raise RegistryError("baseline manifest is not experiment zero")
    baseline_sha = canonical_sha256(baseline)

    for profile in baseline["profiles"].values():
        path = campaign_root / profile["path"]
        value = read_json(path)
        actual = validate_resolved_profile(value)
        if actual != profile["sha256"]:
            raise RegistryError("baseline profile hash mismatch")
    if canonical_sha256(baseline["features"]["ordered_names"]) != baseline["features"]["ordered_sha256"]:
        raise RegistryError("baseline ordered feature hash mismatch")
    fold_path = campaign_root / baseline["fold_manifest"]["path"]
    if canonical_sha256(read_json(fold_path)) != baseline["fold_manifest"]["sha256"]:
        raise RegistryError("baseline fold manifest hash mismatch")

    artifact_path = campaign_root / baseline["artifact_inventory"]["root"]
    current_inventory = tree_inventory(artifact_path)
    if current_inventory.tree_sha256 != baseline["artifact_inventory"]["tree_sha256"]:
        raise RegistryError("baseline artifact tree hash mismatch")
    if current_inventory.file_count != baseline["artifact_inventory"]["file_count"]:
        raise RegistryError("baseline artifact file count mismatch")
    frozen_copy = campaign_root / baseline["frozen_csv"]["copy_path"]
    if file_sha256(frozen_copy) != baseline["frozen_csv"]["sha256"]:
        raise RegistryError("frozen CSV copy hash mismatch")

    gate = AccessLedger(campaign_root).gate_status()
    if gate["protected_access_count"] != baseline["gate"]["protected_access_count"]:
        raise RegistryError("baseline gate access count changed")
    if registry.record_count == 1 and gate["state"] != "closed":
        raise RegistryError("experiment-zero gate must remain closed")
    return CampaignValidation(
        record_count=registry.record_count,
        registry_prefix_sha256=registry.registry_prefix_sha256,
        baseline_manifest_sha256=baseline_sha,
        artifact_tree_sha256=current_inventory.tree_sha256,
        gate_state=gate["state"],
        protected_gate_access_count=gate["protected_access_count"],
    )
