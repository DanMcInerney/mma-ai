"""Frozen temporal-fold protocol and fail-closed gate state."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Callable, Iterable, NamedTuple, Sequence, TypeVar

from .hashing import canonical_json_bytes, read_json, write_canonical_json


GATE_START = date(2026, 1, 1)
GATE_END = date(2026, 8, 8)
DEVELOPMENT_YEARS = (2022, 2023, 2024, 2025)


class ProtocolError(ValueError):
    pass


class GateError(ProtocolError):
    pass


class TemporalFold(NamedTuple):
    test_year: int
    embargo_days: int
    train_ids: tuple[str, ...]
    test_ids: tuple[str, ...]
    train_event_ids: tuple[str, ...]
    test_event_ids: tuple[str, ...]
    train_dates: tuple[date, ...]
    test_dates: tuple[date, ...]
    inner_train_ids: tuple[str, ...]
    inner_validation_ids: tuple[str, ...]
    inner_train_event_ids: tuple[str, ...]
    inner_validation_event_ids: tuple[str, ...]
    inner_train_dates: tuple[date, ...]
    inner_validation_dates: tuple[date, ...]

    def as_dict(self) -> dict:
        return {
            "test_year": self.test_year,
            "embargo_days": self.embargo_days,
            "outer": {
                "train_fight_ids": list(self.train_ids),
                "test_fight_ids": list(self.test_ids),
                "train_event_ids": list(self.train_event_ids),
                "test_event_ids": list(self.test_event_ids),
                "train_date_range": _date_range(self.train_dates),
                "test_date_range": _date_range(self.test_dates),
            },
            "inner": {
                "train_fight_ids": list(self.inner_train_ids),
                "validation_fight_ids": list(self.inner_validation_ids),
                "train_event_ids": list(self.inner_train_event_ids),
                "validation_event_ids": list(self.inner_validation_event_ids),
                "train_date_range": _date_range(self.inner_train_dates),
                "validation_date_range": _date_range(self.inner_validation_dates),
            },
        }


@dataclass(frozen=True)
class LearnedStep:
    name: str
    fit_partition: str
    ordering: str
    prior_only: bool


def _as_date(value: date | datetime | str) -> date:
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    return date.fromisoformat(str(value)[:10])


def _date_range(values: Sequence[date]) -> list[str] | None:
    return [min(values).isoformat(), max(values).isoformat()] if values else None


def _ordered_rows(rows: Iterable[dict]) -> list[dict]:
    normalized = []
    seen_fights: set[str] = set()
    for row in rows:
        required = {"fight_id", "event_id", "event_date"}
        if not required.issubset(row):
            raise ProtocolError(f"row missing temporal identifiers: {sorted(required - set(row))}")
        fight_id = str(row["fight_id"])
        if fight_id in seen_fights:
            raise ProtocolError(f"duplicate fight ID: {fight_id}")
        seen_fights.add(fight_id)
        normalized.append(
            {
                "fight_id": fight_id,
                "event_id": str(row["event_id"]),
                "event_date": _as_date(row["event_date"]),
            }
        )
    return sorted(normalized, key=lambda row: (row["event_date"], row["event_id"], row["fight_id"]))


def _partition(rows: list[dict], ids: set[str]) -> tuple[tuple[str, ...], tuple[str, ...], tuple[date, ...]]:
    selected = [row for row in rows if row["fight_id"] in ids]
    return (
        tuple(row["fight_id"] for row in selected),
        tuple(dict.fromkeys(row["event_id"] for row in selected)),
        tuple(row["event_date"] for row in selected),
    )


def build_development_folds(
    rows: Iterable[dict],
    *,
    years: Sequence[int] = DEVELOPMENT_YEARS,
    embargo_days: int = 7,
) -> tuple[TemporalFold, ...]:
    if embargo_days < 0:
        raise ProtocolError("embargo must be non-negative")
    ordered = _ordered_rows(rows)
    folds: list[TemporalFold] = []
    for year in years:
        test_rows = [row for row in ordered if row["event_date"].year == year]
        if not test_rows:
            raise ProtocolError(f"no whole events available for outer year {year}")
        outer_cutoff = min(row["event_date"] for row in test_rows) - timedelta(days=embargo_days)
        train_rows = [row for row in ordered if row["event_date"] <= outer_cutoff]

        inner_year = year - 1
        inner_validation_rows = [row for row in train_rows if row["event_date"].year == inner_year]
        if not inner_validation_rows:
            raise ProtocolError(f"no inner chronological validation events for outer year {year}")
        inner_cutoff = min(row["event_date"] for row in inner_validation_rows) - timedelta(days=embargo_days)
        inner_train_rows = [row for row in train_rows if row["event_date"] <= inner_cutoff]

        train_ids = {row["fight_id"] for row in train_rows}
        test_ids = {row["fight_id"] for row in test_rows}
        inner_train_ids = {row["fight_id"] for row in inner_train_rows}
        inner_validation_ids = {row["fight_id"] for row in inner_validation_rows}
        outer_train = _partition(ordered, train_ids)
        outer_test = _partition(ordered, test_ids)
        inner_train = _partition(ordered, inner_train_ids)
        inner_validation = _partition(ordered, inner_validation_ids)
        fold = TemporalFold(
            test_year=year,
            embargo_days=embargo_days,
            train_ids=outer_train[0],
            test_ids=outer_test[0],
            train_event_ids=outer_train[1],
            test_event_ids=outer_test[1],
            train_dates=outer_train[2],
            test_dates=outer_test[2],
            inner_train_ids=inner_train[0],
            inner_validation_ids=inner_validation[0],
            inner_train_event_ids=inner_train[1],
            inner_validation_event_ids=inner_validation[1],
            inner_train_dates=inner_train[2],
            inner_validation_dates=inner_validation[2],
        )
        validate_fold(fold)
        folds.append(fold)
    return tuple(folds)


def validate_fold(fold: TemporalFold) -> None:
    if set(fold.train_ids) & set(fold.test_ids):
        raise ProtocolError("fight crosses outer boundary")
    if set(fold.train_event_ids) & set(fold.test_event_ids):
        raise ProtocolError("same event crosses outer boundary")
    if not fold.train_dates or not fold.test_dates:
        raise ProtocolError("outer fold is empty")
    if max(fold.train_dates) > min(fold.test_dates) - timedelta(days=fold.embargo_days):
        raise ProtocolError("outer fold violates embargo")
    if set(fold.inner_train_ids) & set(fold.inner_validation_ids):
        raise ProtocolError("fight crosses inner boundary")
    if set(fold.inner_train_event_ids) & set(fold.inner_validation_event_ids):
        raise ProtocolError("same event crosses inner boundary")
    if not fold.inner_train_dates or not fold.inner_validation_dates:
        raise ProtocolError("inner fold is empty")
    if max(fold.inner_train_dates) > min(fold.inner_validation_dates) - timedelta(days=fold.embargo_days):
        raise ProtocolError("inner fold violates embargo")
    if max(fold.inner_validation_dates) >= min(fold.test_dates):
        raise ProtocolError("inner validation reaches outer future")


def validate_learned_step(step: LearnedStep) -> None:
    if step.fit_partition != "inner":
        raise ProtocolError(f"{step.name} must fit inner-only, not {step.fit_partition}")
    if step.ordering != "chronological":
        raise ProtocolError(f"{step.name} cannot use shuffled calibration/selection")
    if not step.prior_only:
        raise ProtocolError(f"{step.name} violates prior-only as-of context")


def initialize_gate(campaign_root: Path, *, expected_family_ids: tuple[str, ...]) -> None:
    campaign_root = Path(campaign_root)
    gate_path = campaign_root / "gate-state.json"
    access_path = campaign_root / "access-log.jsonl"
    if gate_path.exists() or access_path.exists():
        raise GateError("gate destination already exists")
    access_path.parent.mkdir(parents=True, exist_ok=True)
    access_path.write_bytes(b"")
    write_canonical_json(
        gate_path,
        {
            "gate_id": "historically_exposed_campaign_gate",
            "state": "closed",
            "date_range": [GATE_START.isoformat(), GATE_END.isoformat()],
            "expected_family_ids": list(expected_family_ids),
            "candidate_id": None,
            "protected_access_count": 0,
            "opened_at": None,
        },
    )


def seal_candidate(campaign_root: Path, *, family_ids: Sequence[str], candidate_id: str) -> None:
    gate_path = Path(campaign_root) / "gate-state.json"
    state = read_json(gate_path)
    expected = tuple(state["expected_family_ids"])
    if len(family_ids) != 10 or tuple(family_ids) != expected:
        raise GateError("candidate seal requires exactly ten frozen terminal families")
    if state["state"] != "closed" or state["protected_access_count"] != 0:
        raise GateError("candidate cannot be sealed after gate access")
    state["state"] = "sealed"
    state["candidate_id"] = candidate_id
    write_canonical_json(gate_path, state)


T = TypeVar("T")


class AccessLedger:
    def __init__(self, campaign_root: Path):
        self.campaign_root = Path(campaign_root)
        self.gate_path = self.campaign_root / "gate-state.json"
        self.access_path = self.campaign_root / "access-log.jsonl"

    def gate_status(self) -> dict:
        state = read_json(self.gate_path)
        protected_records = 0
        if self.access_path.exists():
            for line in self.access_path.read_text(encoding="utf-8").splitlines():
                if line and json.loads(line).get("protected_gate_labels"):
                    protected_records += 1
        if protected_records != state["protected_access_count"]:
            raise GateError("gate access log/state mismatch")
        return state

    def record(
        self,
        *,
        purpose: str,
        columns: Sequence[str],
        min_date: date,
        max_date: date,
        protected_gate_labels: bool,
    ) -> None:
        if protected_gate_labels:
            raise GateError("protected gate labels require sealed one-open authorization")
        entry = {
            "purpose": purpose,
            "columns": list(columns),
            "date_range": [min_date.isoformat(), max_date.isoformat()],
            "protected_gate_labels": False,
        }
        with self.access_path.open("ab") as handle:
            handle.write(canonical_json_bytes(entry) + b"\n")

    def read_protected_gate(self, reader: Callable[[], T]) -> T:
        state = self.gate_status()
        if state["protected_access_count"] != 0 or state["state"] == "open":
            raise GateError("historically exposed campaign gate opens exactly once")
        if state["state"] != "sealed" or not state["candidate_id"]:
            raise GateError("final candidate must be sealed before gate access")
        entry = {
            "purpose": "final-descriptive-gate",
            "columns": ["y_true", "probability"],
            "date_range": [GATE_START.isoformat(), GATE_END.isoformat()],
            "protected_gate_labels": True,
            "candidate_id": state["candidate_id"],
        }
        with self.access_path.open("ab") as handle:
            handle.write(canonical_json_bytes(entry) + b"\n")
        state["state"] = "open"
        state["protected_access_count"] = 1
        state["opened_at"] = datetime.now(timezone.utc).isoformat()
        write_canonical_json(self.gate_path, state)
        # Authorization is durably consumed first: a crashing reader cannot hide a peek.
        return reader()

    def record_adaptation(self, description: str) -> None:
        state = self.gate_status()
        if state["state"] in {"sealed", "open"}:
            raise GateError(f"post-gate adaptation is forbidden: {description}")
