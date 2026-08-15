import csv
import json
from pathlib import Path

from libs.modeling.experiment_campaign.baseline import BaselineSources, bootstrap_experiment_zero
from libs.modeling.experiment_campaign.hashing import file_sha256, tree_inventory
from libs.modeling.experiment_campaign.registry import validate_registry
from libs.modeling.experiment_campaign.validation import validate_campaign


def _profile(*, use_recency_weights: bool) -> dict:
    return {
        "model_type": "win",
        "preset": "hybrid",
        "time_limit": 3000,
        "test_size": None,
        "val_date": None,
        "features": ["a", "b"],
        "included_strings": None,
        "excluded_strings": None,
        "required_strings": None,
        "start_date": "2014-01-01",
        "num_fights": 2,
        "include_split_dec": True,
        "normalize": "robust",
        "use_recency_weights": use_recency_weights,
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


def _write_training_csv(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["fight_id", "event_id", "event_date", "y_true"])
        writer.writeheader()
        for year in range(2019, 2027):
            writer.writerow(
                {
                    "fight_id": f"fight-{year}",
                    "event_id": f"event-{year}",
                    "event_date": f"{year}-02-01",
                    "y_true": 1,
                }
            )


def _sources(tmp_path: Path) -> BaselineSources:
    source = tmp_path / "source"
    frozen_csv = source / "data" / "training_data.csv"
    accepted = source / "models" / "accepted"
    no_recency = source / "models" / "no-recency"
    accepted_evidence = source / "evidence" / "accepted"
    no_recency_evidence = source / "evidence" / "no-recency"
    _write_training_csv(frozen_csv)
    _write_training_csv(accepted / "training_data.csv")
    (accepted / "model.bin").write_bytes(b"accepted-model")
    (no_recency / "model.bin").parent.mkdir(parents=True, exist_ok=True)
    (no_recency / "model.bin").write_bytes(b"no-recency-model")
    accepted_evidence.mkdir(parents=True)
    no_recency_evidence.mkdir(parents=True)
    (accepted_evidence / "direct-evaluation.json").write_text('{"opaque":true}', encoding="utf-8")
    (accepted_evidence / "final-verification.md").write_text("accepted", encoding="utf-8")
    (no_recency_evidence / "direct-evaluation.json").write_text('{"opaque":true}', encoding="utf-8")
    (no_recency_evidence / "final-verification.md").write_text("no-recency", encoding="utf-8")
    return BaselineSources(
        frozen_csv=frozen_csv,
        accepted_model=accepted,
        no_recency_model=no_recency,
        accepted_evidence=accepted_evidence,
        no_recency_evidence=no_recency_evidence,
    )


def test_experiment_zero_is_self_contained_hashed_and_gate_closed(tmp_path):
    campaign = tmp_path / "campaign"
    artifact_root = campaign / "artifacts" / "01-campaign-harness"
    sources = _sources(tmp_path)
    result = bootstrap_experiment_zero(
        campaign,
        artifact_root,
        sources=sources,
        source_revision="a" * 40,
        working_profile=_profile(use_recency_weights=True),
        no_recency_profile=_profile(use_recency_weights=False),
        expected_population={"total": 8, "pre_2025": 6, "from_2025": 2, "gate": 1},
        expected_source_hashes={"frozen_csv": file_sha256(sources.frozen_csv)},
        expected_model_identities={},
    )

    manifest = json.loads((campaign / "baseline" / "manifest.json").read_text(encoding="utf-8"))
    assert result.experiment_id == "experiment-zero"
    assert (campaign / ".gitattributes").read_bytes() == b"* -text\n"
    assert manifest["source_revision"] == "a" * 40
    assert manifest["population"]["total"] == 8
    assert manifest["fold_manifest"]["years"] == [2022, 2023, 2024, 2025]
    assert manifest["gate"]["state"] == "closed"
    assert manifest["gate"]["protected_access_count"] == 0
    assert (artifact_root / "frozen" / "training_data.csv").read_bytes() == sources.frozen_csv.read_bytes()
    assert (artifact_root / "models" / "accepted" / "model.bin").read_bytes() == b"accepted-model"
    assert (artifact_root / "evidence" / "accepted" / "direct-evaluation.json").read_bytes() == b'{"opaque":true}'
    source_inventory = tree_inventory(artifact_root)
    assert source_inventory.file_count > 0
    assert source_inventory.tree_sha256 == manifest["artifact_inventory"]["tree_sha256"]

    registry = validate_registry(campaign, strict=True)
    assert registry.record_count == 1
    assert registry.family_ids == ()
    assert result.registry_prefix_sha256 == registry.registry_prefix_sha256
    report = validate_campaign(campaign, strict=True)
    assert report.gate_state == "closed"
    assert report.protected_gate_access_count == 0


def test_bootstrap_refuses_preexisting_artifact_destination_and_source_hash_change(tmp_path):
    campaign = tmp_path / "campaign"
    artifact_root = campaign / "artifacts" / "01-campaign-harness"
    sources = _sources(tmp_path)
    artifact_root.mkdir(parents=True)
    try:
        bootstrap_experiment_zero(
            campaign,
            artifact_root,
            sources=sources,
            source_revision="a" * 40,
            working_profile=_profile(use_recency_weights=True),
            no_recency_profile=_profile(use_recency_weights=False),
            expected_population={"total": 8, "pre_2025": 6, "from_2025": 2, "gate": 1},
            expected_source_hashes={"frozen_csv": "0" * 64},
            expected_model_identities={},
        )
    except ValueError as exc:
        assert "destination" in str(exc) or "hash" in str(exc)
    else:
        raise AssertionError("bootstrap accepted a reused artifact destination")
