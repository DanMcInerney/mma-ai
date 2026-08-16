from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

import pytest

from libs.modeling.split_refit_experiment.protocol import (
    EXPECTED_COUNTS,
    EXPECTED_DATES,
    FROZEN_SOURCE_SHA256,
    ProtocolError,
    load_partition,
    read_source_metadata,
    validate_materialized_split,
    verify_split,
)
from libs.modeling.split_refit_experiment.registry import (
    RegistryError,
    validate_registry,
)


REPO_ROOT = Path(__file__).parents[2]
CAMPAIGN = REPO_ROOT / "experiments/split_refit_20260816"
SOURCE = Path(
    r"C:\Users\danhm\mma-ai\worktrees\top10-20260815\data\training_data.csv"
)


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, separators=(",", ":"), sort_keys=True
    ).encode("utf-8")


def _read_json(path: Path) -> object:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, value: object) -> None:
    path.write_bytes(_canonical_bytes(value) + b"\n")


def _copy_campaign(tmp_path: Path) -> Path:
    clone = tmp_path / "campaign"
    shutil.copytree(CAMPAIGN, clone)
    return clone


def _partition_path(campaign: Path, name: str) -> Path:
    return campaign / "partitions" / f"{name}.json"


def _refresh_partition_reference(campaign: Path, name: str) -> None:
    path = _partition_path(campaign, name)
    document = _read_json(path)
    digest = hashlib.sha256(_canonical_bytes(document)).hexdigest().upper()
    master_path = campaign / "partitions/manifest.json"
    master = _read_json(master_path)
    master["partitions"][name]["sha256"] = digest
    _write_json(master_path, master)


def test_real_materialization_is_exact_event_safe_and_metadata_only():
    result = verify_split(CAMPAIGN, source_csv=SOURCE, strict=True)

    assert result.source_sha256 == FROZEN_SOURCE_SHA256
    assert result.eligible_count == 3267
    assert result.development_count == 3089
    assert result.partition_counts == EXPECTED_COUNTS
    assert result.partition_dates == EXPECTED_DATES
    assert result.retired_count == 178
    assert result.retired_dates == ("2026-01-24", "2026-08-08")
    assert result.retired_label_reads == 0

    documents = {
        name: _read_json(_partition_path(CAMPAIGN, name))
        for name in (*EXPECTED_COUNTS, "retired")
    }
    development_ids = [
        fight_id
        for name in EXPECTED_COUNTS
        for fight_id in documents[name]["fight_ids"]
    ]
    assert len(development_ids) == len(set(development_ids)) == 3089
    assert set(development_ids).isdisjoint(documents["retired"]["fight_ids"])
    assert len(set(development_ids) | set(documents["retired"]["fight_ids"])) == 3267

    event_owner = {}
    for name in EXPECTED_COUNTS:
        for event_id in documents[name]["event_ids"]:
            assert event_owner.setdefault(event_id, name) == name

    profile = _read_json(CAMPAIGN / "profiles/evaluation.json")
    assert len(profile) == 23
    assert profile["calculate_importance"] is False
    assert profile["refit_full"] is False
    assert profile["timeseries_split"]["manifest_path"] == "partitions/manifest.json"
    assert len(profile["features"]) == 40


def test_source_metadata_reader_never_requests_the_label(monkeypatch):
    import libs.modeling.split_refit_experiment.protocol as protocol

    real_read_csv = protocol.pd.read_csv
    seen_usecols = []

    def guarded_read_csv(*args, **kwargs):
        usecols = tuple(kwargs.get("usecols", ()))
        seen_usecols.append(usecols)
        assert "y_true" not in usecols
        return real_read_csv(*args, **kwargs)

    monkeypatch.setattr(protocol.pd, "read_csv", guarded_read_csv)
    rows = read_source_metadata(SOURCE)

    assert len(rows) == 3267
    assert seen_usecols and all("y_true" not in columns for columns in seen_usecols)


def test_loader_rejects_retired_label_decode_before_calling_decoder():
    called = False

    def forbidden_decoder(_ids):
        nonlocal called
        called = True
        raise AssertionError("retired labels were touched")

    with pytest.raises(ProtocolError, match="retired.*label"):
        load_partition(
            CAMPAIGN,
            source_csv=SOURCE,
            partition="retired",
            label_decoder=forbidden_decoder,
        )
    assert called is False


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("split_event", "event.*crosses"),
        ("missing_fight", "exhaustive|count"),
        ("duplicate_fight", "duplicate"),
        ("reordered_fight", "order"),
        ("wrong_date", "date"),
        ("wrong_count", "count"),
        ("wrong_hash", "hash"),
        ("retired_overlap", "retired|overlap"),
    ],
)
def test_hostile_partition_fixtures_fail_before_data_return(
    tmp_path, mutation, message
):
    clone = _copy_campaign(tmp_path)
    train_path = _partition_path(clone, "train")
    validation_path = _partition_path(clone, "validation")
    retired_path = _partition_path(clone, "retired")
    train = _read_json(train_path)
    validation = _read_json(validation_path)
    retired = _read_json(retired_path)

    if mutation == "split_event":
        event_id = validation["event_ids"][0]
        validation_row = next(row for row in validation["rows"] if row["event_id"] == event_id)
        train["rows"].append(validation_row)
        train["fight_ids"].append(validation_row["fight_id"])
        train["row_count"] += 1
        _write_json(train_path, train)
        _refresh_partition_reference(clone, "train")
    elif mutation == "missing_fight":
        train["rows"].pop()
        train["fight_ids"].pop()
        train["row_count"] -= 1
        _write_json(train_path, train)
        _refresh_partition_reference(clone, "train")
    elif mutation == "duplicate_fight":
        train["rows"].append(train["rows"][-1])
        train["fight_ids"].append(train["fight_ids"][-1])
        train["row_count"] += 1
        _write_json(train_path, train)
        _refresh_partition_reference(clone, "train")
    elif mutation == "reordered_fight":
        train["rows"][0], train["rows"][1] = train["rows"][1], train["rows"][0]
        train["fight_ids"][0], train["fight_ids"][1] = (
            train["fight_ids"][1],
            train["fight_ids"][0],
        )
        _write_json(train_path, train)
        _refresh_partition_reference(clone, "train")
    elif mutation == "wrong_date":
        train["date_range"][1] = "2023-10-22"
        _write_json(train_path, train)
        _refresh_partition_reference(clone, "train")
    elif mutation == "wrong_count":
        train["row_count"] += 1
        _write_json(train_path, train)
        _refresh_partition_reference(clone, "train")
    elif mutation == "wrong_hash":
        master = _read_json(clone / "partitions/manifest.json")
        master["partitions"]["train"]["sha256"] = "0" * 64
        _write_json(clone / "partitions/manifest.json", master)
    elif mutation == "retired_overlap":
        retired["rows"][0] = train["rows"][0]
        retired["fight_ids"][0] = train["fight_ids"][0]
        _write_json(retired_path, retired)
        _refresh_partition_reference(clone, "retired")

    metadata = read_source_metadata(SOURCE)
    with pytest.raises(ProtocolError, match=message):
        validate_materialized_split(clone, metadata, source_sha256=FROZEN_SOURCE_SHA256)


def test_malformed_manifest_prevents_label_decoder(tmp_path):
    clone = _copy_campaign(tmp_path)
    train = _read_json(_partition_path(clone, "train"))
    train["fight_ids"] = train["fight_ids"][:-1]
    _write_json(_partition_path(clone, "train"), train)
    _refresh_partition_reference(clone, "train")
    called = False

    def decoder(_ids):
        nonlocal called
        called = True
        return []

    with pytest.raises(ProtocolError):
        load_partition(
            clone,
            source_csv=SOURCE,
            partition="train",
            label_decoder=decoder,
        )
    assert called is False


def test_registry_is_exact_canonical_append_only_chain():
    result = validate_registry(CAMPAIGN, strict=True, through="split")

    assert result.record_count == 2
    assert result.record_ids == ("rollback-capsule", "split-materialization")
    registry_bytes = (CAMPAIGN / "registry.jsonl").read_bytes()
    assert registry_bytes.endswith(b"\n")
    for line in registry_bytes.splitlines():
        assert line == _canonical_bytes(json.loads(line))


@pytest.mark.parametrize(
    "mutation",
    ["reorder", "duplicate_id", "changed_prior", "missing_head", "noncanonical"],
)
def test_registry_hostile_fixtures_fail(tmp_path, mutation):
    clone = _copy_campaign(tmp_path)
    registry_path = clone / "registry.jsonl"
    lines = registry_path.read_bytes().splitlines()
    records = [json.loads(line) for line in lines]

    if mutation == "reorder":
        registry_path.write_bytes(lines[1] + b"\n" + lines[0] + b"\n")
    elif mutation == "duplicate_id":
        records[1]["record_id"] = records[0]["record_id"]
        registry_path.write_bytes(
            b"".join(_canonical_bytes(record) + b"\n" for record in records)
        )
    elif mutation == "changed_prior":
        records[0]["payload"]["artifact_sha256"] = "0" * 64
        registry_path.write_bytes(
            b"".join(_canonical_bytes(record) + b"\n" for record in records)
        )
    elif mutation == "missing_head":
        (clone / "registry-head.json").unlink()
    elif mutation == "noncanonical":
        registry_path.write_text(
            "\n".join(json.dumps(record, indent=2) for record in records) + "\n",
            encoding="utf-8",
        )

    with pytest.raises(RegistryError):
        validate_registry(clone, strict=True, through="split")
