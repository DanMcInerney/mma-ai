"""Fail-closed semantic portfolio preregistration and inner selection seams."""

from __future__ import annotations

from collections import defaultdict
from datetime import date
from typing import Any, Mapping, Sequence

from .hashing import canonical_sha256


MEASUREMENT_GROUP_IDS = (
    "demographics-experience",
    "global-striking-pace-efficiency",
    "head-body-leg-targeting",
    "range-clinch-ground-position",
    "takedown-control-submission",
    "damage-finish",
    "opponent-style-strength-of-schedule",
)


class SemanticPortfolioError(ValueError):
    """A semantic candidate or its selection evidence is inadmissible."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise SemanticPortfolioError(message)


def validate_preregistered_profile(
    profile: Mapping[str, Any],
    *,
    source_header: Sequence[str],
) -> dict[str, Any]:
    """Validate lineage and the complete, pre-score maximum-eight menu."""

    frozen_source = profile.get("frozen_source", {})
    source_sha256 = str(frozen_source.get("sha256", ""))
    header_sha256 = canonical_sha256(list(source_header))
    # The source header commitment deliberately uses a line-oriented identity so
    # it can be reproduced without parsing any data rows.
    import hashlib

    line_header_sha256 = hashlib.sha256("\n".join(source_header).encode()).hexdigest().upper()
    _require(len(source_sha256) == 64, "frozen source SHA-256 is missing")
    _require(
        frozen_source.get("ordered_header_sha256") == line_header_sha256,
        "frozen source header identity mismatch",
    )
    cutoff = date.fromisoformat(str(frozen_source.get("cutoff")))
    _require(
        tuple(profile.get("measurement_group_ids", ())) == MEASUREMENT_GROUP_IDS,
        "measurement groups differ from the preregistered order",
    )

    candidates = profile.get("candidate_features")
    _require(isinstance(candidates, list) and candidates, "candidate features are missing")
    names: set[str] = set()
    semantics: set[str] = set()
    candidate_by_name: dict[str, Mapping[str, Any]] = {}
    header = set(source_header)
    lineage_fields = {
        "name",
        "semantic_id",
        "measurement_group",
        "source_file_sha256",
        "source_header_sha256",
        "formula",
        "available_by",
        "domain_redundancy_rank",
    }
    for candidate in candidates:
        _require(isinstance(candidate, Mapping), "candidate lineage must be an object")
        _require(lineage_fields <= set(candidate), "candidate has missing lineage")
        name = str(candidate["name"])
        semantic_id = str(candidate["semantic_id"])
        _require(name not in names, f"duplicate candidate feature: {name}")
        _require(semantic_id not in semantics, f"duplicate semantic identity: {semantic_id}")
        _require(name in header, f"candidate is absent from the source header: {name}")
        _require(
            candidate["measurement_group"] in MEASUREMENT_GROUP_IDS,
            f"candidate has unknown measurement group: {name}",
        )
        _require(
            candidate["source_file_sha256"] == source_sha256
            and candidate["source_header_sha256"] == line_header_sha256,
            f"candidate source lineage mismatch: {name}",
        )
        _require(
            date.fromisoformat(str(candidate["available_by"])) <= cutoff,
            f"candidate is post-cutoff: {name}",
        )
        _require(str(candidate["formula"]).strip() != "", f"candidate formula is missing: {name}")
        _require(
            isinstance(candidate["domain_redundancy_rank"], int)
            and candidate["domain_redundancy_rank"] > 0,
            f"candidate redundancy rank is invalid: {name}",
        )
        names.add(name)
        semantics.add(semantic_id)
        candidate_by_name[name] = candidate

    profiles = profile.get("measurement_profiles")
    _require(isinstance(profiles, list) and profiles, "measurement profiles are missing")
    _require(len(profiles) <= 8, "maximum eight measurement profiles may be preregistered")
    profile_ids: set[str] = set()
    for measurement_profile in profiles:
        profile_id = str(measurement_profile.get("id", ""))
        _require(profile_id and profile_id not in profile_ids, "duplicate measurement profile ID")
        included_groups = tuple(measurement_profile.get("included_groups", ()))
        _require(
            len(included_groups) == len(set(included_groups))
            and set(included_groups) <= set(MEASUREMENT_GROUP_IDS),
            f"measurement profile has invalid groups: {profile_id}",
        )
        ordered_features = measurement_profile.get("ordered_features")
        _require(
            isinstance(ordered_features, list)
            and len(ordered_features) == len(set(ordered_features))
            and set(ordered_features) <= names,
            f"measurement profile has invalid ordered features: {profile_id}",
        )
        _require(
            all(candidate_by_name[name]["measurement_group"] in included_groups for name in ordered_features),
            f"measurement profile feature/group mismatch: {profile_id}",
        )
        _require(
            measurement_profile.get("ordered_feature_sha256") == canonical_sha256(ordered_features),
            f"measurement profile feature hash mismatch: {profile_id}",
        )
        profile_ids.add(profile_id)

    selection = profile.get("selection", {})
    _require(selection.get("evidence_role") == "inner-chronological", "selection must be inner-only")
    _require(
        selection.get("combined_row_importance_role") == "non-selection",
        "combined-row importance must be explicitly non-selection",
    )
    _require(0.5 < float(selection.get("stability_threshold", 0)) <= 1.0, "invalid stability threshold")
    _require(int(selection.get("minimum_fold_support", 0)) >= 2, "invalid minimum fold support")
    _require(int(selection.get("domain_redundancy_cap", 0)) >= 1, "invalid domain redundancy cap")
    return {
        "candidate_count": len(candidates),
        "profile_count": len(profiles),
        "source_sha256": source_sha256,
        "source_header_sha256": line_header_sha256,
        "canonical_header_sha256": header_sha256,
        "candidate_feature_sha256": canonical_sha256([item["name"] for item in candidates]),
        "measurement_profile_sha256": canonical_sha256(profiles),
    }


def select_stable_features(
    evidence: Sequence[Mapping[str, Any]],
    *,
    profile: Mapping[str, Any],
    outer_year: int,
) -> dict[str, Any]:
    """Select candidates from strictly prior chronological fold evidence."""

    candidates = list(profile["candidate_features"])
    candidate_order = {item["name"]: index for index, item in enumerate(candidates)}
    candidate_group = {item["name"]: item["measurement_group"] for item in candidates}
    by_feature: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    seen: set[tuple[str, int]] = set()
    for row in evidence:
        feature = str(row.get("feature", ""))
        fold = int(row.get("fold", outer_year))
        _require(feature in candidate_order, f"unknown evidence feature: {feature}")
        _require(
            row.get("role") == "inner-chronological" and fold < outer_year,
            "outer or future selection evidence is forbidden",
        )
        _require((feature, fold) not in seen, "duplicate feature/fold selection evidence")
        _require(int(row.get("direction", 0)) in (-1, 1), "feature direction must be signed")
        _require(
            float(row.get("drop_column_delta", 0.0)) == float(row.get("drop_column_delta", 0.0)),
            "drop-column evidence must be finite",
        )
        seen.add((feature, fold))
        by_feature[feature].append(row)

    selection = profile["selection"]
    minimum_support = int(selection["minimum_fold_support"])
    threshold = float(selection["stability_threshold"])
    minimum_improvement = float(selection["drop_column_min_improvement"])
    eligible = []
    statistics = []
    for candidate in candidates:
        feature = candidate["name"]
        rows = sorted(by_feature.get(feature, ()), key=lambda row: int(row["fold"]))
        support = len(rows)
        positive = sum(int(row["direction"]) > 0 for row in rows)
        negative = support - positive
        stability = max(positive, negative) / support if support else 0.0
        mean_delta = (
            sum(float(row["drop_column_delta"]) for row in rows) / support if support else 0.0
        )
        accepted = (
            support >= minimum_support
            and stability >= threshold
            and mean_delta > minimum_improvement
        )
        statistic = {
            "feature": feature,
            "measurement_group": candidate_group[feature],
            "folds": [int(row["fold"]) for row in rows],
            "support": support,
            "positive_direction_count": positive,
            "negative_direction_count": negative,
            "stability": stability,
            "mean_drop_column_delta": mean_delta,
            "eligible_before_redundancy_cap": accepted,
        }
        statistics.append(statistic)
        if accepted:
            eligible.append(statistic)

    cap = int(selection["domain_redundancy_cap"])
    selected_set: set[str] = set()
    for group in MEASUREMENT_GROUP_IDS:
        group_candidates = [item for item in eligible if item["measurement_group"] == group]
        group_candidates.sort(
            key=lambda item: (-item["mean_drop_column_delta"], candidate_order[item["feature"]])
        )
        selected_set.update(item["feature"] for item in group_candidates[:cap])
    selected = [item["name"] for item in candidates if item["name"] in selected_set]
    fit_folds = sorted({int(row["fold"]) for rows in by_feature.values() for row in rows})
    return {
        "outer_year": outer_year,
        "fit_role": "inner-chronological",
        "fit_folds": fit_folds,
        "selected_features": selected,
        "selected_feature_sha256": canonical_sha256(selected),
        "feature_statistics": statistics,
        "outer_label_selection_count": 0,
        "gate_selection_count": 0,
        "combined_row_importance_used": False,
    }
