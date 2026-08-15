import json
from pathlib import Path

import pytest

from libs.modeling.experiment_campaign.registry import (
    CAMPAIGN_FAMILY_IDS,
    RegistryError,
    append_registry_record,
    canonical_sha256,
    initialize_registry,
    validate_registry,
    validate_resolved_profile,
)


PROFILE_FIELDS = {
    "model_type": "win",
    "preset": "hybrid",
    "time_limit": 3000,
    "test_size": None,
    "val_date": None,
    "features": ["age_diff", "reach_diff"],
    "included_strings": None,
    "excluded_strings": None,
    "required_strings": None,
    "start_date": "2014-01-01",
    "num_fights": 2,
    "include_split_dec": True,
    "normalize": "robust",
    "use_recency_weights": True,
    "decay_rate": 0.15,
    "calculate_importance": False,
    "included_model_types": None,
    "split_strategy": "timeseries_split",
    "walkforward_n_windows": 4,
    "walkforward_initial_year": 2021,
    "timeseries_split": None,
    "refit_all": False,
    "refit_full": False,
}


def _write_json(path: Path, value: dict) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True), encoding="utf-8")
    return canonical_sha256(value)


def _record(campaign: Path, experiment_id: str, *, kind: str, status: str = "complete") -> dict:
    profile_path = campaign / "profiles" / f"{experiment_id}.json"
    manifest_path = campaign / "runs" / experiment_id / "manifest.json"
    profile_sha = _write_json(profile_path, PROFILE_FIELDS)
    manifest = {
        "experiment_id": experiment_id,
        "exit_state": status,
        "profile_sha256": profile_sha,
        "artifact_path": f"artifacts/{experiment_id}",
        "artifact_tree_sha256": "a" * 64,
    }
    manifest_sha = _write_json(manifest_path, manifest)
    return {
        "experiment_id": experiment_id,
        "kind": kind,
        "status": status,
        "profile_path": profile_path.relative_to(campaign).as_posix(),
        "profile_sha256": profile_sha,
        "manifest_path": manifest_path.relative_to(campaign).as_posix(),
        "manifest_sha256": manifest_sha,
        "artifact_path": manifest["artifact_path"],
        "artifact_tree_sha256": manifest["artifact_tree_sha256"],
    }


def test_registry_is_append_only_and_preserves_every_prefix(tmp_path):
    campaign = tmp_path / "campaign"
    initialize_registry(campaign)
    append_registry_record(campaign, _record(campaign, "experiment-zero", kind="experiment-zero"))
    first_bytes = (campaign / "registry.jsonl").read_bytes()

    append_registry_record(
        campaign,
        _record(campaign, CAMPAIGN_FAMILY_IDS[0], kind="family", status="failed"),
    )

    assert (campaign / "registry.jsonl").read_bytes().startswith(first_bytes)
    result = validate_registry(campaign, strict=True)
    assert result.record_count == 2
    assert result.family_ids == (CAMPAIGN_FAMILY_IDS[0],)
    assert len(result.registry_prefix_sha256) == 64


@pytest.mark.parametrize("mutation", ["truncate", "reorder", "alter"])
def test_registry_rejects_changed_prior_bytes(tmp_path, mutation):
    campaign = tmp_path / "campaign"
    initialize_registry(campaign)
    append_registry_record(campaign, _record(campaign, "experiment-zero", kind="experiment-zero"))
    append_registry_record(campaign, _record(campaign, CAMPAIGN_FAMILY_IDS[0], kind="family"))
    path = campaign / "registry.jsonl"
    lines = path.read_bytes().splitlines(keepends=True)
    if mutation == "truncate":
        path.write_bytes(lines[0])
    elif mutation == "reorder":
        path.write_bytes(lines[1] + lines[0])
    else:
        path.write_bytes(path.read_bytes().replace(b'"status":"complete"', b'"status":"failed"', 1))

    with pytest.raises(RegistryError):
        validate_registry(campaign, strict=True)


def test_registry_rejects_duplicate_out_of_order_family_and_reused_artifact(tmp_path):
    campaign = tmp_path / "campaign"
    initialize_registry(campaign)
    zero = _record(campaign, "experiment-zero", kind="experiment-zero")
    append_registry_record(campaign, zero)
    family_one = _record(campaign, CAMPAIGN_FAMILY_IDS[0], kind="family")
    append_registry_record(campaign, family_one)

    with pytest.raises(RegistryError, match="duplicate"):
        append_registry_record(campaign, family_one)
    with pytest.raises(RegistryError, match="next family"):
        append_registry_record(campaign, _record(campaign, CAMPAIGN_FAMILY_IDS[2], kind="family"))

    family_two = _record(campaign, CAMPAIGN_FAMILY_IDS[1], kind="family")
    family_two["artifact_path"] = family_one["artifact_path"]
    with pytest.raises(RegistryError, match="artifact path"):
        append_registry_record(campaign, family_two)


def test_registry_rejects_hash_mismatch_missing_failure_and_eleventh_family(tmp_path):
    campaign = tmp_path / "campaign"
    initialize_registry(campaign)
    zero = _record(campaign, "experiment-zero", kind="experiment-zero")
    zero["profile_sha256"] = "0" * 64
    with pytest.raises(RegistryError, match="profile hash"):
        append_registry_record(campaign, zero)

    zero = _record(campaign, "experiment-zero", kind="experiment-zero")
    append_registry_record(campaign, zero)
    orphan = campaign / "runs" / "failed-unregistered" / "manifest.json"
    _write_json(orphan, {"experiment_id": "failed-unregistered", "exit_state": "failed"})
    with pytest.raises(RegistryError, match="unregistered run manifest"):
        validate_registry(campaign, strict=True)

    outside = _record(campaign, "family-11-not-admitted", kind="family")
    with pytest.raises(RegistryError, match="frozen ten"):
        append_registry_record(campaign, outside)


def test_profile_must_be_fully_materialized_and_canonical():
    assert validate_resolved_profile(PROFILE_FIELDS) == canonical_sha256(PROFILE_FIELDS)
    inherited = {"extends": "v8-hybrid-weighted", "decay_rate": 0.25}
    with pytest.raises(RegistryError, match="fully materialized"):
        validate_resolved_profile(inherited)
