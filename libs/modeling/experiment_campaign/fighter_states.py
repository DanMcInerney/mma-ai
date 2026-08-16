"""Causal, count-aware multi-timescale fighter state construction."""

from __future__ import annotations

from collections import defaultdict
from datetime import date
from math import sqrt
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
            and feature_names
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
                recent = histories[fighter_id][-2:]
                exposure = float(len(recent))
                wins = float(sum(item["won"] for item in recent))
                value, uncertainty = _posterior_rate(wins, exposure, alpha, beta)
                if "recent_win_rate" in feature_names:
                    output.append(
                        {
                            "event_id": str(fight["event_id"]),
                            "fight_id": str(fight["fight_id"]),
                            "fighter_id": fighter_id,
                            "opponent_id": opponent_id,
                            "event_date": event_date,
                            "cutoff": event_date,
                            "feature_name": "recent_win_rate",
                            "value": wins / exposure if exposure else alpha / (alpha + beta),
                            "formula_version": "recent-last-two-v1",
                            "fit_scope": "prior-only",
                            "numerator": wins + alpha,
                            "denominator": exposure + alpha + beta,
                            "effective_support": exposure,
                            "uncertainty": uncertainty,
                            "prior_id": prior_id,
                            "source_row_ids": [item["fight_id"] for item in recent],
                            "source_event_ids": [item["event_id"] for item in recent],
                            "source_dates": [item["event_date"] for item in recent],
                            "artifact_sha256": artifact_sha256,
                        }
                    )
        for fight in daily:
            y_true = int(fight["y_true"])
            identities = (
                (str(fight["fighter1_id"]), y_true),
                (str(fight["fighter2_id"]), 1 - y_true),
            )
            for fighter_id, won in identities:
                histories[fighter_id].append(
                    {
                        "fight_id": str(fight["fight_id"]),
                        "event_id": str(fight["event_id"]),
                        "event_date": str(fight["event_date"]),
                        "won": won,
                    }
                )
    return output
