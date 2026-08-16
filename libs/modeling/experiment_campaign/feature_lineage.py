"""Fail-closed lineage checks for point-in-time campaign features."""

from __future__ import annotations

import csv
from datetime import date
from math import isfinite
from typing import Any, Iterable, Mapping, Sequence


DEVELOPMENT_FIGHT_COUNT = 3_089
RETIRED_FIGHT_COUNT = 178
POPULATION_FIGHT_COUNT = DEVELOPMENT_FIGHT_COUNT + RETIRED_FIGHT_COUNT


class FeatureLineageError(ValueError):
    """A feature row or its source partition is not causally admissible."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise FeatureLineageError(message)


def build_development_safe_ids(
    fold_manifest: Mapping[str, Any],
) -> tuple[tuple[str, ...], frozenset[str]]:
    """Subtract the retired roster and assert the exact partition."""

    population = tuple(str(value) for value in fold_manifest["population_fight_ids"])
    retired_roster = tuple(str(row["fight_id"]) for row in fold_manifest["gate_roster"])
    retired = frozenset(retired_roster)
    safe = tuple(fight_id for fight_id in population if fight_id not in retired)
    _require(
        len(population) == len(set(population)) == POPULATION_FIGHT_COUNT,
        "population identities are not the exact immutable population",
    )
    _require(
        len(retired_roster) == len(retired) == RETIRED_FIGHT_COUNT,
        "retired identities are not the exact immutable roster",
    )
    _require(
        len(safe) == len(set(safe)) == DEVELOPMENT_FIGHT_COUNT,
        "development-safe identities are not the exact 3,089-ID set",
    )
    _require(set(safe).isdisjoint(retired), "development and retired IDs overlap")
    _require(set(safe) | retired == set(population), "safe partition is incomplete")
    return safe, retired


def _decode_full_row(raw: bytes, indices: Sequence[int]) -> list[str]:
    parsed = next(csv.reader([raw.decode("utf-8")]))
    return [parsed[index] for index in indices]


def decode_development_rows(
    raw_rows: Iterable[bytes],
    *,
    safe_ids: Sequence[str],
    retired_ids: frozenset[str],
    indices: Sequence[int],
) -> list[list[str]]:
    """Inspect only the leading ID until retired rows have been discarded."""

    safe_ids = tuple(str(value) for value in safe_ids)
    safe = frozenset(safe_ids)
    retired = frozenset(str(value) for value in retired_ids)
    _require(
        len(safe_ids) == len(safe) == DEVELOPMENT_FIGHT_COUNT,
        "decoder requires the exact 3,089-ID development-safe set",
    )
    _require(
        len(retired) == RETIRED_FIGHT_COUNT and safe.isdisjoint(retired),
        "decoder requires the exact disjoint retired roster",
    )
    decoded: list[list[str]] = []
    for raw in raw_rows:
        fight_id = raw.split(b",", 1)[0].strip(b'"').decode("utf-8")
        if fight_id in retired:
            continue
        if fight_id in safe:
            decoded.append(_decode_full_row(raw, indices))
    _require(len(decoded) == DEVELOPMENT_FIGHT_COUNT, "safe source rows are incomplete")
    return decoded


LINEAGE_FIELDS = frozenset(
    {
        "event_id",
        "fight_id",
        "fighter_id",
        "opponent_id",
        "event_date",
        "cutoff",
        "feature_name",
        "value",
        "formula_version",
        "fit_scope",
        "numerator",
        "denominator",
        "effective_support",
        "uncertainty",
        "prior_id",
        "source_row_ids",
        "source_event_ids",
        "source_dates",
        "artifact_sha256",
    }
)


def validate_feature_lineage_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    registered_prior_ids: set[str] | frozenset[str],
) -> dict[str, Any]:
    """Validate complete causal lineage and count-aware rate evidence."""

    features: set[str] = set()
    minimum_support = float("inf")
    maximum_uncertainty = 0.0
    for row in rows:
        _require(LINEAGE_FIELDS <= set(row), "missing lineage fields")
        _require(row["fighter_id"] != row["opponent_id"], "identity collision")
        _require(row["fit_scope"] != "global", "global-fit normalization is forbidden")
        _require(row["fit_scope"] == "prior-only", "feature fit scope must be prior-only")
        _require(row["prior_id"] in registered_prior_ids, "unregistered prior")
        rate_feature = "rate" in str(row["feature_name"]) or "posterior" in str(row["feature_name"])
        if rate_feature:
            _require(float(row["denominator"]) > 0.0, "zero denominator rate")
            _require(float(row["numerator"]) >= 0.0, "negative rate numerator")
        support = float(row["effective_support"])
        uncertainty = float(row["uncertainty"])
        _require(support >= 0.0, "negative effective support")
        _require(isfinite(uncertainty) and uncertainty >= 0.0, "invalid uncertainty")
        target_date = date.fromisoformat(str(row["cutoff"]))
        source_dates = tuple(date.fromisoformat(str(value)) for value in row["source_dates"])
        _require(
            all(value < target_date for value in source_dates)
            and row["event_id"] not in set(row["source_event_ids"]),
            "same-event or future source rows are forbidden",
        )
        _require(
            len(row["source_row_ids"])
            == len(row["source_event_ids"])
            == len(row["source_dates"]),
            "source lineage arrays differ in length",
        )
        _require(len(str(row["artifact_sha256"])) == 64, "invalid artifact hash")
        features.add(str(row["feature_name"]))
        minimum_support = min(minimum_support, support)
        maximum_uncertainty = max(maximum_uncertainty, uncertainty)
    return {
        "row_count": len(rows),
        "feature_count": len(features),
        "minimum_effective_support": 0.0 if not rows else minimum_support,
        "maximum_uncertainty": maximum_uncertainty,
    }
