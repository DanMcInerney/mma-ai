"""Pinned metadata-only materialization of the retrospective event split."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence, TypeVar

import pandas as pd


FROZEN_SOURCE_SHA256 = "157649B780965ECC585F18B3030199CDC0F4FE3013958FFA4095FCF665FDB1EA"
FROZEN_FEATURE_SHA256 = "13E545D762A3F1BE4D023D82B8E65D77E41589031051F1F6796D742F25223022"
FROZEN_ROSTER_SHA256 = "C2282A441448F0ABF18FAE9B78792ACBA9275B67C206F1F93FD5373AB83370AD"
DEFAULT_SOURCE_CSV = Path(
    r"C:\Users\danhm\mma-ai\worktrees\top10-20260815\data\training_data.csv"
)
DEFAULT_ROSTER = Path(
    r"C:\Users\danhm\mma-ai\worktrees\top10-20260815\experiments"
    r"\top10_20260815\baseline\fold-manifest.json"
)
EXPECTED_COUNTS = {"train": 2473, "validation": 309, "test": 307}
EXPECTED_DATES = {
    "train": ("2014-01-15", "2023-10-21"),
    "validation": ("2023-11-04", "2024-11-16"),
    "test": ("2024-11-23", "2025-12-13"),
}
RETIRED_DATES = ("2026-01-24", "2026-08-08")
PARTITION_ORDER = ("train", "validation", "test", "retired")
ZERO_HASH = "0" * 64

T = TypeVar("T")


class ProtocolError(ValueError):
    pass


@dataclass(frozen=True)
class SplitVerification:
    source_sha256: str
    eligible_count: int
    development_count: int
    partition_counts: dict[str, int]
    partition_dates: dict[str, tuple[str, str]]
    partition_hashes: dict[str, str]
    retired_count: int
    retired_dates: tuple[str, str]
    retired_ids_sha256: str
    retired_label_reads: int
    manifest_sha256: str
    profile_sha256: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "source_sha256": self.source_sha256,
            "eligible_count": self.eligible_count,
            "development_count": self.development_count,
            "partition_counts": self.partition_counts,
            "partition_dates": self.partition_dates,
            "partition_hashes": self.partition_hashes,
            "retired_count": self.retired_count,
            "retired_dates": self.retired_dates,
            "retired_ids_sha256": self.retired_ids_sha256,
            "retired_label_reads": self.retired_label_reads,
            "manifest_sha256": self.manifest_sha256,
            "profile_sha256": self.profile_sha256,
        }


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, separators=(",", ":"), sort_keys=True
    ).encode("utf-8")


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest().upper()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def _write_json(path: Path, value: Any) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical_json_bytes(value) + b"\n")
    return canonical_sha256(value)


def _read_canonical_json(path: Path) -> Any:
    try:
        raw = path.read_bytes()
    except FileNotFoundError as exc:
        raise ProtocolError(f"missing artifact: {path}") from exc
    try:
        value = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ProtocolError(f"invalid JSON artifact: {path}") from exc
    canonical = canonical_json_bytes(value)
    if raw not in (canonical, canonical + b"\n", canonical + b"\r\n"):
        raise ProtocolError(f"noncanonical JSON artifact: {path}")
    return value


def _id_sort_key(value: Any) -> tuple[int, int | str]:
    text = str(value)
    return (0, int(text)) if text.isdigit() else (1, text)


def _row_sort_key(row: Mapping[str, str]) -> tuple[str, tuple[int, int | str], tuple[int, int | str]]:
    return (row["event_date"], _id_sort_key(row["event_id"]), _id_sort_key(row["fight_id"]))


def _load_roster(path: Path) -> dict[str, Any]:
    if file_sha256(path) != FROZEN_ROSTER_SHA256:
        raise ProtocolError("pre-existing metadata roster hash mismatch")
    roster = json.loads(path.read_text(encoding="utf-8"))
    if roster.get("label_columns_accessed") != []:
        raise ProtocolError("metadata roster records label access")
    if roster.get("safe_columns") != ["fight_id", "event_id", "event_date"]:
        raise ProtocolError("metadata roster safe-column contract changed")
    population = [str(value) for value in roster.get("population_fight_ids", [])]
    retired = roster.get("gate_roster", [])
    if len(population) != len(set(population)) or len(population) != 3267:
        raise ProtocolError("metadata roster population is not the exact 3,267 IDs")
    if len(retired) != 178:
        raise ProtocolError("metadata roster retired count changed")
    return roster


def read_source_metadata(
    source_csv: Path = DEFAULT_SOURCE_CSV,
    *,
    roster_path: Path = DEFAULT_ROSTER,
) -> list[dict[str, str]]:
    """Read only the three pre-existing safe columns and admit the pinned roster."""
    source_csv = Path(source_csv)
    if file_sha256(source_csv) != FROZEN_SOURCE_SHA256:
        raise ProtocolError("frozen source CSV hash mismatch")
    roster = _load_roster(Path(roster_path))
    frame = pd.read_csv(
        source_csv,
        usecols=["fight_id", "event_id", "event_date"],
        dtype={"fight_id": "string", "event_id": "string", "event_date": "string"},
    )
    if len(frame) != 7704:
        raise ProtocolError("frozen source raw row count changed")
    if frame["fight_id"].isna().any() or frame["fight_id"].duplicated().any():
        raise ProtocolError("source contains missing or duplicate fight IDs")
    source_by_id = {
        str(row.fight_id): {
            "fight_id": str(row.fight_id),
            "event_id": str(row.event_id),
            "event_date": str(row.event_date)[:10],
        }
        for row in frame.itertuples(index=False)
    }
    population_ids = [str(value) for value in roster["population_fight_ids"]]
    missing = [fight_id for fight_id in population_ids if fight_id not in source_by_id]
    if missing:
        raise ProtocolError(f"source is missing roster fight IDs: {missing[:3]}")
    rows = sorted((source_by_id[fight_id] for fight_id in population_ids), key=_row_sort_key)
    retired_by_id = {str(row["fight_id"]): row for row in roster["gate_roster"]}
    for fight_id, expected in retired_by_id.items():
        actual = source_by_id.get(fight_id)
        if actual != {key: str(expected[key]) for key in ("fight_id", "event_id", "event_date")}:
            raise ProtocolError(f"retired metadata drift for fight {fight_id}")
    return rows


def _partition_name(row: Mapping[str, str]) -> str:
    value = row["event_date"]
    for name, (first, last) in EXPECTED_DATES.items():
        if first <= value <= last:
            return name
    if RETIRED_DATES[0] <= value <= RETIRED_DATES[1]:
        return "retired"
    raise ProtocolError(f"eligible fight falls outside exact partition dates: {value}")


def _partition_document(name: str, rows: Sequence[dict[str, str]]) -> dict[str, Any]:
    fight_ids = [row["fight_id"] for row in rows]
    event_ids = list(dict.fromkeys(row["event_id"] for row in rows))
    date_range = [rows[0]["event_date"], rows[-1]["event_date"]]
    return {
        "schema_version": 1,
        "partition": name,
        "row_count": len(rows),
        "date_range": date_range,
        "event_ids": event_ids,
        "event_ids_sha256": canonical_sha256(event_ids),
        "fight_ids": fight_ids,
        "fight_ids_sha256": canonical_sha256(fight_ids),
        "labels_decoded": False,
        "rows": list(rows),
    }


def _evaluation_profile(rollback: Mapping[str, Any], manifest_sha256: str) -> dict[str, Any]:
    fields = dict(rollback["reproduction"]["profile"]["fields"])
    fields["features"] = list(fields["features"])
    fields["calculate_importance"] = False
    fields["refit_full"] = False
    fields["timeseries_split"] = {
        "manifest_path": "partitions/manifest.json",
        "manifest_sha256": manifest_sha256,
        "train": "train",
        "validation": "validation",
        "test": "test",
    }
    if len(fields) != 23:
        raise ProtocolError("evaluation profile is not the exact 23-field profile")
    if canonical_sha256(fields["features"]) != FROZEN_FEATURE_SHA256:
        raise ProtocolError("evaluation profile feature order changed")
    return fields


def materialize_split(
    campaign_root: Path,
    *,
    source_csv: Path = DEFAULT_SOURCE_CSV,
    roster_path: Path = DEFAULT_ROSTER,
) -> SplitVerification:
    campaign_root = Path(campaign_root)
    rows = read_source_metadata(source_csv, roster_path=roster_path)
    grouped = {name: [] for name in PARTITION_ORDER}
    for row in rows:
        grouped[_partition_name(row)].append(row)
    if {name: len(grouped[name]) for name in EXPECTED_COUNTS} != EXPECTED_COUNTS:
        raise ProtocolError("fresh partition counts differ from the fixed 2,473/309/307")
    if len(grouped["retired"]) != 178:
        raise ProtocolError("fresh retired metadata count differs from 178")

    partitions: dict[str, dict[str, Any]] = {}
    for name in PARTITION_ORDER:
        document = _partition_document(name, grouped[name])
        relative = f"partitions/{name}.json"
        digest = _write_json(campaign_root / relative, document)
        partitions[name] = {
            "path": relative,
            "sha256": digest,
            "row_count": document["row_count"],
            "date_range": document["date_range"],
            "fight_ids_sha256": document["fight_ids_sha256"],
            "event_ids_sha256": document["event_ids_sha256"],
        }

    development_ids = [
        row["fight_id"] for name in EXPECTED_COUNTS for row in grouped[name]
    ]
    retired_ids = [row["fight_id"] for row in grouped["retired"]]
    master = {
        "schema_version": 1,
        "source_csv_sha256": FROZEN_SOURCE_SHA256,
        "source_metadata_roster_sha256": FROZEN_ROSTER_SHA256,
        "eligible_row_count": len(rows),
        "development_row_count": len(development_ids),
        "development_fight_ids_sha256": canonical_sha256(development_ids),
        "retired_row_count": len(retired_ids),
        "retired_fight_ids_sha256": canonical_sha256(retired_ids),
        "retired_labels_decoded": False,
        "partitions": partitions,
    }
    manifest_sha256 = _write_json(campaign_root / "partitions/manifest.json", master)
    rollback = _read_canonical_json(campaign_root / "rollback-manifest.json")
    profile = _evaluation_profile(rollback, manifest_sha256)
    profile_sha256 = _write_json(campaign_root / "profiles/evaluation.json", profile)

    from .registry import initialize_split_registry

    initialize_split_registry(
        campaign_root,
        split_manifest_sha256=file_sha256(campaign_root / "partitions/manifest.json"),
        profile_sha256=file_sha256(campaign_root / "profiles/evaluation.json"),
    )
    return verify_split(campaign_root, source_csv=source_csv, strict=True)


def _validate_partition_document(name: str, document: Mapping[str, Any]) -> None:
    rows = document.get("rows")
    if not isinstance(rows, list) or not rows:
        raise ProtocolError(f"{name} partition rows are missing")
    if rows != sorted(rows, key=_row_sort_key):
        raise ProtocolError(f"{name} fight order is not canonical chronological order")
    fight_ids = [row.get("fight_id") for row in rows]
    if len(fight_ids) != len(set(fight_ids)):
        raise ProtocolError(f"duplicate fight ID in {name}")
    if document.get("fight_ids") != fight_ids:
        raise ProtocolError(f"{name} fight ID order differs from rows")
    if document.get("fight_ids_sha256") != canonical_sha256(fight_ids):
        raise ProtocolError(f"{name} fight ID hash mismatch")
    event_ids = list(dict.fromkeys(row.get("event_id") for row in rows))
    if document.get("event_ids") != event_ids:
        raise ProtocolError(f"{name} event IDs differ from rows")
    if document.get("event_ids_sha256") != canonical_sha256(event_ids):
        raise ProtocolError(f"{name} event ID hash mismatch")
    if document.get("row_count") != len(rows):
        raise ProtocolError(f"{name} row count mismatch")
    actual_dates = [rows[0]["event_date"], rows[-1]["event_date"]]
    if document.get("date_range") != actual_dates:
        raise ProtocolError(f"{name} date range mismatch")
    if document.get("labels_decoded") is not False:
        raise ProtocolError(f"{name} manifest claims label decoding")


def validate_materialized_split(
    campaign_root: Path,
    source_rows: Sequence[dict[str, str]],
    *,
    source_sha256: str,
) -> SplitVerification:
    campaign_root = Path(campaign_root)
    if source_sha256 != FROZEN_SOURCE_SHA256:
        raise ProtocolError("source hash mismatch")
    master = _read_canonical_json(campaign_root / "partitions/manifest.json")
    if master.get("source_csv_sha256") != source_sha256:
        raise ProtocolError("partition manifest source hash mismatch")
    documents: dict[str, Mapping[str, Any]] = {}
    for name in PARTITION_ORDER:
        reference = master.get("partitions", {}).get(name, {})
        expected_path = f"partitions/{name}.json"
        if reference.get("path") != expected_path:
            raise ProtocolError(f"{name} partition path changed")
        document = _read_canonical_json(campaign_root / expected_path)
        if canonical_sha256(document) != reference.get("sha256"):
            raise ProtocolError(f"{name} partition hash mismatch")
        documents[name] = document

    event_owner: dict[str, str] = {}
    for name, document in documents.items():
        for row in document["rows"]:
            event_id = row["event_id"]
            owner = event_owner.setdefault(event_id, name)
            if owner != name:
                raise ProtocolError(f"event {event_id} crosses {owner}/{name} boundary")

    for name, document in documents.items():
        fight_ids = [row.get("fight_id") for row in document["rows"]]
        if len(fight_ids) != len(set(fight_ids)):
            raise ProtocolError(f"duplicate fight ID in {name}")

    train_ids = set(documents["train"]["fight_ids"])
    validation_ids = set(documents["validation"]["fight_ids"])
    test_ids = set(documents["test"]["fight_ids"])
    retired_ids = set(documents["retired"]["fight_ids"])
    development_sets = (train_ids, validation_ids, test_ids)
    if any(left & right for index, left in enumerate(development_sets) for right in development_sets[index + 1 :]):
        raise ProtocolError("duplicate fight ID crosses development partitions")
    if retired_ids & set().union(*development_sets):
        raise ProtocolError("retired and development fight IDs overlap")

    counts = {name: documents[name].get("row_count") for name in EXPECTED_COUNTS}
    dates = {name: tuple(documents[name].get("date_range", ())) for name in EXPECTED_COUNTS}
    if counts != EXPECTED_COUNTS:
        raise ProtocolError("development partition count mismatch")
    if dates != EXPECTED_DATES:
        raise ProtocolError("development partition date mismatch")
    if documents["retired"].get("row_count") != 178:
        raise ProtocolError("retired row count mismatch")
    if tuple(documents["retired"].get("date_range", ())) != RETIRED_DATES:
        raise ProtocolError("retired date range mismatch")

    for name, document in documents.items():
        _validate_partition_document(name, document)
        reference = master["partitions"][name]
        for key in ("row_count", "date_range", "fight_ids_sha256", "event_ids_sha256"):
            if reference.get(key) != document.get(key):
                raise ProtocolError(f"{name} manifest {key} mismatch")

    materialized_rows = [
        row for name in PARTITION_ORDER for row in documents[name]["rows"]
    ]
    if len(materialized_rows) != 3267 or len({row["fight_id"] for row in materialized_rows}) != 3267:
        raise ProtocolError("materialized split is not exhaustive and disjoint")
    if materialized_rows != list(source_rows):
        raise ProtocolError("materialized fight metadata/order differs from frozen source")
    if master.get("eligible_row_count") != 3267 or master.get("development_row_count") != 3089:
        raise ProtocolError("master population count mismatch")
    if master.get("retired_labels_decoded") is not False:
        raise ProtocolError("master permits retired label access")

    profile = _read_canonical_json(campaign_root / "profiles/evaluation.json")
    if len(profile) != 23:
        raise ProtocolError("evaluation profile field count mismatch")
    if canonical_sha256(profile["features"]) != FROZEN_FEATURE_SHA256:
        raise ProtocolError("evaluation profile feature hash mismatch")
    manifest_sha256 = canonical_sha256(master)
    split_seam = profile.get("timeseries_split", {})
    if split_seam.get("manifest_sha256") != manifest_sha256:
        raise ProtocolError("evaluation profile partition hash mismatch")
    if profile.get("calculate_importance") is not False or profile.get("refit_full") is not False:
        raise ProtocolError("evaluation profile did not disable importance/refit")

    return SplitVerification(
        source_sha256=source_sha256,
        eligible_count=3267,
        development_count=3089,
        partition_counts=counts,
        partition_dates=dates,
        partition_hashes={name: master["partitions"][name]["sha256"] for name in PARTITION_ORDER},
        retired_count=178,
        retired_dates=RETIRED_DATES,
        retired_ids_sha256=documents["retired"]["fight_ids_sha256"],
        retired_label_reads=0,
        manifest_sha256=manifest_sha256,
        profile_sha256=canonical_sha256(profile),
    )


def verify_split(
    campaign_root: Path,
    *,
    source_csv: Path = DEFAULT_SOURCE_CSV,
    strict: bool = False,
) -> SplitVerification:
    rows = read_source_metadata(source_csv)
    result = validate_materialized_split(
        campaign_root, rows, source_sha256=file_sha256(source_csv)
    )
    if strict:
        from .registry import validate_registry

        validate_registry(campaign_root, strict=True, through="split")
    return result


def load_partition(
    campaign_root: Path,
    *,
    source_csv: Path,
    partition: str,
    label_decoder: Callable[[tuple[str, ...]], T] | None = None,
) -> list[dict[str, str]] | T:
    if partition not in PARTITION_ORDER:
        raise ProtocolError(f"unknown partition: {partition}")
    rows = read_source_metadata(source_csv)
    validate_materialized_split(
        campaign_root, rows, source_sha256=file_sha256(source_csv)
    )
    document = _read_canonical_json(Path(campaign_root) / f"partitions/{partition}.json")
    if partition == "retired" and label_decoder is not None:
        raise ProtocolError("retired partition label decoding is forbidden")
    selected = list(document["rows"])
    if label_decoder is None:
        return selected
    return label_decoder(tuple(document["fight_ids"]))
