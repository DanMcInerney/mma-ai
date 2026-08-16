"""Causal, count-aware multi-timescale fighter state construction."""

from __future__ import annotations

from collections import defaultdict
from datetime import date
from math import exp, log, log1p, sqrt
from typing import Any, Mapping, Sequence

from .hashing import canonical_sha256


class FighterStateError(ValueError):
    """A fighter-state profile or source fight is inadmissible."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise FighterStateError(message)


def validate_preregistered_profiles(profile: Mapping[str, Any]) -> dict[str, Any]:
    """Require a complete, bounded menu before any construction or fit."""

    source_sha = str(profile.get("source_sha256", ""))
    _require(len(source_sha) == 64, "source SHA-256 is missing")
    date.fromisoformat(str(profile.get("cutoff")))
    priors = profile.get("registered_priors")
    _require(isinstance(priors, Mapping) and priors, "registered priors are missing")
    for prior_id, prior in priors.items():
        _require(str(prior_id).strip() != "", "registered prior ID is missing")
        _require(
            float(prior.get("alpha", 0.0)) > 0.0 and float(prior.get("beta", 0.0)) > 0.0,
            "registered beta prior must be positive",
        )
    normalization = profile.get("normalization", {})
    _require(
        normalization.get("fit_scope") == "outer-train-only",
        "global-fit normalization is forbidden",
    )
    profiles = profile.get("profiles")
    _require(isinstance(profiles, list) and profiles, "feature profiles are missing")
    _require(len(profiles) <= 8, "maximum eight feature profiles may be preregistered")
    ids: set[str] = set()
    for item in profiles:
        profile_id = str(item.get("id", ""))
        _require(profile_id and profile_id not in ids, "duplicate feature profile ID")
        feature_names = item.get("feature_names")
        formula_versions = item.get("formula_versions")
        _require(
            isinstance(feature_names, list)
            and (feature_names or profile_id == "v8-retained-incumbent-control")
            and len(feature_names) == len(set(feature_names)),
            f"invalid feature order: {profile_id}",
        )
        _require(
            isinstance(formula_versions, list)
            and len(formula_versions) == len(feature_names)
            and all(str(value).strip() for value in formula_versions),
            f"missing formula version: {profile_id}",
        )
        ids.add(profile_id)
    return {
        "profile_count": len(profiles),
        "profile_ids": [item["id"] for item in profiles],
        "profile_sha256": canonical_sha256(profile),
    }


def _posterior_rate(wins: float, exposure: float, alpha: float, beta: float) -> tuple[float, float]:
    posterior_alpha = wins + alpha
    posterior_beta = exposure - wins + beta
    denominator = posterior_alpha + posterior_beta
    mean = posterior_alpha / denominator
    variance = posterior_alpha * posterior_beta / (denominator**2 * (denominator + 1.0))
    return mean, sqrt(variance)


def _weighted_rate(
    history: Sequence[Mapping[str, Any]],
    field: str,
    weights: Sequence[float],
    alpha: float,
    beta: float,
) -> tuple[float, float, float, float, float]:
    exposure = float(sum(weights))
    successes = float(sum(float(row[field]) * weight for row, weight in zip(history, weights, strict=True)))
    mean, uncertainty = _posterior_rate(successes, exposure, alpha, beta)
    raw = successes / exposure if exposure else alpha / (alpha + beta)
    return raw, uncertainty, successes + alpha, exposure + alpha + beta, exposure


def _weighted_location(
    history: Sequence[Mapping[str, Any]],
    field: str,
    weights: Sequence[float],
) -> tuple[float, float, float, float, float]:
    values = [float(row[field]) for row in history]
    if not values:
        return 0.0, 1.0, 0.0, 1.0, 0.0
    ordered = sorted(values)
    low = ordered[int((len(ordered) - 1) * 0.1)]
    high = ordered[int((len(ordered) - 1) * 0.9)]
    clipped = [min(high, max(low, value)) for value in values]
    denominator = float(sum(weights))
    numerator = float(sum(value * weight for value, weight in zip(clipped, weights, strict=True)))
    mean = numerator / denominator
    variance = sum(weight * (value - mean) ** 2 for value, weight in zip(clipped, weights, strict=True)) / denominator
    uncertainty = sqrt(variance / max(denominator, 1.0))
    return mean, uncertainty, numerator, denominator, denominator


def build_fighter_state_rows(
    fights: Sequence[Mapping[str, Any]],
    *,
    profile: Mapping[str, Any],
    artifact_sha256: str,
) -> list[dict[str, Any]]:
    """Build fighter-level state rows without exposing a same-date event."""

    validate_preregistered_profiles(profile)
    _require(len(artifact_sha256) == 64, "artifact SHA-256 is missing")
    ordered = sorted(fights, key=lambda row: (str(row["event_date"]), str(row["event_id"]), str(row["fight_id"])))
    fight_ids = [str(row["fight_id"]) for row in ordered]
    _require(len(fight_ids) == len(set(fight_ids)), "fight identity collision")
    for fight in ordered:
        _require(fight["fighter1_id"] != fight["fighter2_id"], "fighter identity collision")
        _require(int(fight["y_true"]) in (0, 1), "fight outcome is not binary")

    feature_names = {
        name for item in profile["profiles"] for name in item["feature_names"]
    }
    definitions = profile.get("feature_definitions", {})
    if definitions:
        _require(feature_names == set(definitions), "profile features differ from frozen definitions")
    else:
        _require(feature_names <= {"recent_win_rate"}, "unknown fighter-state feature")
    prior_id, prior = next(iter(profile["registered_priors"].items()))
    alpha, beta = float(prior["alpha"]), float(prior["beta"])
    histories: dict[str, list[dict[str, Any]]] = defaultdict(list)
    output: list[dict[str, Any]] = []
    by_date: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for fight in ordered:
        by_date[str(fight["event_date"])].append(fight)
    for event_date in sorted(by_date):
        daily = by_date[event_date]
        for fight in daily:
            participants = (
                (str(fight["fighter1_id"]), str(fight["fighter2_id"])),
                (str(fight["fighter2_id"]), str(fight["fighter1_id"])),
            )
            for fighter_id, opponent_id in participants:
                history = histories[fighter_id]
                target = date.fromisoformat(event_date)
                recent = history[-2:]
                medium = [
                    item for item in history
                    if 0 < (target - date.fromisoformat(item["event_date"])).days <= 548
                ]
                recent_weights = [1.0] * len(recent)
                medium_weights = [
                    exp(-log(2.0) * (target - date.fromisoformat(item["event_date"])).days / 180.0)
                    for item in medium
                ]
                career_weights = [1.0] * len(history)
                rate_values: dict[str, tuple[float, float, float, float, float, Sequence[Mapping[str, Any]]]] = {}
                for prefix, subset, weights in (
                    ("recent", recent, recent_weights),
                    ("decay18", medium, medium_weights),
                    ("career", history, career_weights),
                ):
                    for suffix, field in (("win_rate", "won"), ("ko_rate", "ko"), ("submission_rate", "submission")):
                        values = _weighted_rate(subset, field, weights, alpha, beta)
                        rate_values[f"{prefix}_{suffix}"] = (*values, subset)
                sparse_ko = (*_weighted_rate(medium, "ko", medium_weights, alpha, beta), medium)
                sparse_submission = (*_weighted_rate(medium, "submission", medium_weights, alpha, beta), medium)
                high_weights = [
                    exp(-log(2.0) * (target - date.fromisoformat(item["event_date"])).days / 365.0)
                    for item in history
                ]
                high_striking = (*_weighted_location(history, "striking", high_weights), history)
                high_head = (*_weighted_location(history, "head", high_weights), history)
                recent_win = rate_values["recent_win_rate"]
                medium_win = rate_values["decay18_win_rate"]
                career_win = rate_values["career_win_rate"]
                inactivity = min(
                    730.0,
                    float((target - date.fromisoformat(history[-1]["event_date"])).days) if history else 730.0,
                )
                if history:
                    latest_age = float(history[-1]["age"]) + (
                        target - date.fromisoformat(history[-1]["event_date"])
                    ).days / 365.25
                else:
                    latest_age = 30.0
                derived = {
                    **rate_values,
                    "win_trend_short_medium": (
                        recent_win[0] - medium_win[0], max(recent_win[1], medium_win[1]),
                        recent_win[0] - medium_win[0], max(recent_win[4], medium_win[4], 1.0),
                        max(recent_win[4], medium_win[4]), history,
                    ),
                    "win_trend_medium_career": (
                        medium_win[0] - career_win[0], max(medium_win[1], career_win[1]),
                        medium_win[0] - career_win[0], max(medium_win[4], career_win[4], 1.0),
                        max(medium_win[4], career_win[4]), history,
                    ),
                    "sparse_ko_posterior": sparse_ko,
                    "sparse_submission_posterior": sparse_submission,
                    "high_count_striking_state": high_striking,
                    "high_count_head_state": high_head,
                    "inactivity_days": (inactivity, 0.0, inactivity, 1.0, float(bool(history)), history[-1:] if history else []),
                    "age_win_interaction": (
                        ((latest_age - 30.0) / 10.0) * career_win[0], career_win[1],
                        ((latest_age - 30.0) / 10.0) * career_win[0], max(career_win[4], 1.0),
                        career_win[4], history,
                    ),
                    "state_effective_support": (
                        log1p(len(history)), 0.0, float(len(history)), 1.0, float(len(history)), history,
                    ),
                    "state_win_uncertainty": (
                        career_win[1], career_win[1], career_win[1], 1.0, career_win[4], history,
                    ),
                }
                for feature_name in sorted(feature_names):
                    value, uncertainty, numerator, denominator, support, sources = derived[feature_name]
                    definition = definitions.get(feature_name, {"formula_version": "recent-last-two-v1", "prior_id": prior_id})
                    output.append(
                        {
                            "event_id": str(fight["event_id"]),
                            "fight_id": str(fight["fight_id"]),
                            "fighter_id": fighter_id,
                            "opponent_id": opponent_id,
                            "event_date": event_date,
                            "cutoff": event_date,
                            "feature_name": feature_name,
                            "value": value,
                            "formula_version": definition["formula_version"],
                            "fit_scope": "prior-only",
                            "numerator": numerator,
                            "denominator": denominator,
                            "effective_support": support,
                            "uncertainty": uncertainty,
                            "prior_id": prior_id,
                            "source_row_ids": [item["fight_id"] for item in sources],
                            "source_event_ids": [item["event_id"] for item in sources],
                            "source_dates": [item["event_date"] for item in sources],
                            "artifact_sha256": artifact_sha256,
                        }
                    )
        for fight in daily:
            y_true = int(fight["y_true"])
            striking = float(fight.get("sig_str_land_state", 0.0))
            striking_diff = float(fight.get("sig_str_land_state_diff", 0.0))
            head = float(fight.get("head_land_state", 0.0))
            head_diff = float(fight.get("head_land_state_diff", 0.0))
            age = float(fight.get("age_state", 30.0))
            age_diff = float(fight.get("age_state_diff", 0.0))
            identities = (
                (str(fight["fighter1_id"]), y_true, striking, head, age),
                (str(fight["fighter2_id"]), 1 - y_true, striking - striking_diff, head - head_diff, age - age_diff),
            )
            method = str(fight["method"]).lower()
            for fighter_id, won, striking_value, head_value, age_value in identities:
                histories[fighter_id].append(
                    {
                        "fight_id": str(fight["fight_id"]),
                        "event_id": str(fight["event_id"]),
                        "event_date": str(fight["event_date"]),
                        "won": won,
                        "ko": int(bool(won) and "ko" in method),
                        "submission": int(bool(won) and "submission" in method),
                        "striking": striking_value,
                        "head": head_value,
                        "age": age_value,
                    }
                )
    return output


def select_state_profile(
    evidence: Sequence[Mapping[str, Any]],
    *,
    profile: Mapping[str, Any],
    outer_year: int,
) -> dict[str, Any]:
    """Select one preregistered profile using only prior inner years."""

    profile_ids = tuple(item["id"] for item in profile["profiles"])
    by_profile: dict[str, dict[int, float]] = {profile_id: {} for profile_id in profile_ids}
    for row in evidence:
        profile_id = str(row.get("profile_id", ""))
        validation_year = int(row.get("validation_year", outer_year))
        _require(profile_id in by_profile, "unknown profile selection evidence")
        _require(
            row.get("role") == "inner-chronological" and validation_year < outer_year,
            "outer or future selection evidence is forbidden",
        )
        _require(
            validation_year not in by_profile[profile_id],
            "duplicate profile/year selection evidence",
        )
        loss = float(row["validation_log_loss"])
        _require(loss == loss, "selection loss is not finite")
        by_profile[profile_id][validation_year] = loss
    required_support = int(profile["inner_validation_year_count"])
    _require(
        all(len(scores) == required_support for scores in by_profile.values()),
        "profile selection evidence support is incomplete",
    )
    years = sorted(next(iter(by_profile.values())))
    _require(
        all(sorted(scores) == years for scores in by_profile.values()),
        "profile selection years differ",
    )
    mean_losses = {
        profile_id: sum(scores.values()) / len(scores)
        for profile_id, scores in by_profile.items()
    }
    selected = min(profile_ids, key=lambda value: (mean_losses[value], profile_ids.index(value)))
    selected_profile = next(item for item in profile["profiles"] if item["id"] == selected)
    return {
        "outer_year": outer_year,
        "selection_role": "inner-chronological",
        "selection_years": years,
        "profile_scores": mean_losses,
        "selected_profile_id": selected,
        "selected_feature_names": selected_profile["feature_names"],
        "selected_ordered_feature_sha256": selected_profile["ordered_feature_sha256"],
        "outer_label_selection_count": 0,
    }
