"""Directional matchup features and complete fighter-role reversal guards."""

from __future__ import annotations

from copy import deepcopy
from math import isclose
from typing import Any, Mapping, Sequence

from .hashing import canonical_sha256


class MatchupGeometryError(ValueError):
    pass


ROLE_PAIRS = (
    ("fighter1_id", "fighter2_id"),
    ("fighter1_name", "fighter2_name"),
    ("fighter1_url", "fighter2_url"),
    ("fighter1_label", "fighter2_label"),
    ("fighter1_odds", "fighter2_odds"),
    ("fighter1_features", "fighter2_features"),
    ("fighter1_lineage", "fighter2_lineage"),
    ("fighter1_market_probability", "fighter2_market_probability"),
)
ANTISYMMETRIC_FIELDS = ("offense_minus_defense",)
FORBIDDEN_MODEL_FEATURE_TOKENS = ("id", "name", "url", "label", "odds")


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise MatchupGeometryError(message)


def swap_matchup_row(row: Mapping[str, Any]) -> dict[str, Any]:
    """Return a complete role reversal; partial swaps fail before prediction."""

    required = {"fight_id", "event_id", "event_date", "y_true"}
    required.update(key for pair in ROLE_PAIRS for key in pair)
    _require(required <= set(row), "complete role swap fields are required")
    first_features = row["fighter1_features"]
    second_features = row["fighter2_features"]
    first_lineage = row["fighter1_lineage"]
    second_lineage = row["fighter2_lineage"]
    _require(
        isinstance(first_features, Mapping)
        and isinstance(second_features, Mapping)
        and set(first_features) == set(second_features),
        "complete role swap requires identical paired feature keys",
    )
    _require(
        isinstance(first_lineage, Mapping)
        and isinstance(second_lineage, Mapping)
        and set(first_lineage) == set(second_lineage),
        "complete role swap requires identical paired lineage keys",
    )
    target = row["y_true"]
    _require(target in (0, 1), "complete role swap requires a binary target orientation")
    swapped = deepcopy(dict(row))
    for first, second in ROLE_PAIRS:
        swapped[first], swapped[second] = deepcopy(row[second]), deepcopy(row[first])
    swapped["y_true"] = 1 - int(target)
    for name in ANTISYMMETRIC_FIELDS:
        if name in row:
            swapped[name] = -float(row[name])
    return swapped


def build_directional_interactions(
    fighter1: Mapping[str, Any],
    fighter2: Mapping[str, Any],
    declarations: Sequence[Mapping[str, Any]],
) -> dict[str, float]:
    """Construct registered strength-versus-weakness cross-differences."""

    result: dict[str, float] = {}
    for declaration in declarations:
        name = str(declaration.get("name", ""))
        left = str(declaration.get("left", ""))
        right = str(declaration.get("right", ""))
        _require(name and left and right, "interaction declaration is incomplete")
        _require(
            declaration.get("formula") == "cross-difference"
            and declaration.get("swap_rule") == "negate",
            "eligible directional interactions must be registered antisymmetric cross-differences",
        )
        _require(
            left in fighter1 and right in fighter1 and left in fighter2 and right in fighter2,
            f"interaction {name} source feature is missing",
        )
        minimum_support = float(declaration.get("minimum_support", 0.0))
        _require(minimum_support > 0.0, "interaction support gate must be positive")
        _require(
            float(fighter1.get("effective_support", -1.0)) >= minimum_support
            and float(fighter2.get("effective_support", -1.0)) >= minimum_support,
            f"interaction {name} failed its support gate",
        )
        value = (
            float(fighter1[left])
            + float(fighter2[right])
            - float(fighter2[left])
            - float(fighter1[right])
        )
        result[name] = value
    return result


def validate_prediction_geometry(
    rows: Sequence[Mapping[str, Any]],
    *,
    tolerance: float,
) -> dict[str, Any]:
    """Validate per-fight original/swapped complementarity and identities."""

    _require(0.0 <= tolerance <= 1e-6, "registered geometry tolerance is invalid")
    seen: set[str] = set()
    residuals: list[float] = []
    for row in rows:
        required = {
            "fight_id",
            "original_prediction_id",
            "swapped_prediction_id",
            "original_probability",
            "swapped_probability",
            "averaged_probability",
            "invariance_residual",
        }
        _require(required <= set(row), "prediction geometry row is incomplete")
        fight_id = str(row["fight_id"])
        _require(fight_id not in seen, "prediction geometry contains duplicate fight IDs")
        seen.add(fight_id)
        original_identity = str(row["original_prediction_id"])
        swapped_identity = str(row["swapped_prediction_id"])
        _require(
            original_identity != swapped_identity
            and original_identity == f"{fight_id}:original"
            and swapped_identity == f"{fight_id}:swapped",
            "outer original and swapped prediction identities are invalid",
        )
        features = tuple(str(value).lower() for value in row.get("model_feature_names", ()))
        _require(
            not any(
                token in feature
                for feature in features
                for token in FORBIDDEN_MODEL_FEATURE_TOKENS
            ),
            "identity, URL, label, or odds leakage is forbidden",
        )
        original = float(row["original_probability"])
        swapped = float(row["swapped_probability"])
        _require(0.0 <= original <= 1.0 and 0.0 <= swapped <= 1.0, "probability is outside [0, 1]")
        residual = abs(original + swapped - 1.0)
        averaged = (original + (1.0 - swapped)) / 2.0
        _require(residual <= tolerance, "original and swapped predictions are not complementary")
        _require(
            isclose(float(row["averaged_probability"]), averaged, abs_tol=tolerance, rel_tol=0.0)
            and isclose(float(row["invariance_residual"]), residual, abs_tol=tolerance, rel_tol=0.0),
            "stored averaged probability or invariance residual differs",
        )
        residuals.append(residual)
    return {
        "row_count": len(rows),
        "tolerance": tolerance,
        "maximum_invariance_residual": max(residuals, default=0.0),
        "complementarity_failure_count": 0,
    }


def validate_preregistered_matchup_profiles(profile: Mapping[str, Any]) -> dict[str, Any]:
    profiles = profile.get("profiles")
    _require(isinstance(profiles, list) and 1 <= len(profiles) <= 8, "maximum eight profiles")
    profile_ids = tuple(str(item.get("id")) for item in profiles)
    _require(len(set(profile_ids)) == len(profile_ids), "profile IDs must be unique")
    _require(
        profile.get("normalization", {}).get("fit_scope") == "outer-train-only",
        "global-fit normalization is forbidden",
    )
    selection = profile.get("selection", {})
    _require(
        selection.get("fit_scope") == "prior-inner-only"
        and selection.get("outer_label_selection_count") == 0,
        "profile selection must be prior-inner-only",
    )
    registered = {
        str(item["name"]): item for item in profile.get("interaction_definitions", [])
    }
    _require(len(registered) == len(profile.get("interaction_definitions", [])), "interaction names must be unique")
    for item in registered.values():
        _require(
            item.get("formula") == "cross-difference"
            and item.get("swap_rule") == "negate",
            "eligible transforms must be registered antisymmetric",
        )
        _require(float(item.get("minimum_support", 0.0)) > 0.0, "sparse interaction lacks support gate")
    for item in profiles:
        interactions = list(item.get("interaction_names", []))
        _require(set(interactions) <= set(registered), "profile references an unregistered interaction")
        expected_hash = canonical_sha256(
            {
                "interaction_names": interactions,
                "prediction_geometry": item.get("prediction_geometry"),
            }
        )
        _require(item.get("ordered_interaction_sha256") == expected_hash, "profile interaction hash differs")
    return {"profile_count": len(profiles), "profile_ids": list(profile_ids)}


def select_matchup_profile(
    evidence: Sequence[Mapping[str, Any]],
    *,
    profile: Mapping[str, Any],
    outer_year: int,
) -> dict[str, Any]:
    validated = validate_preregistered_matchup_profiles(profile)
    profile_ids = tuple(validated["profile_ids"])
    scores: dict[str, float] = {}
    selection_years: set[int] = set()
    for profile_id in profile_ids:
        rows = [row for row in evidence if row.get("profile_id") == profile_id]
        _require(rows, "inner selection evidence is incomplete")
        years = {int(row["validation_year"]) for row in rows}
        _require(
            all(
                row.get("role") == "inner-chronological"
                and int(row["validation_year"]) < outer_year
                for row in rows
            ),
            "inner selection contains outer or future labels",
        )
        selection_years.update(years)
        scores[profile_id] = sum(float(row["validation_log_loss"]) for row in rows) / len(rows)
    selected = min(profile_ids, key=lambda value: (scores[value], profile_ids.index(value)))
    return {
        "outer_year": outer_year,
        "selected_profile_id": selected,
        "profile_scores": scores,
        "selection_years": sorted(selection_years),
        "outer_label_selection_count": 0,
    }
