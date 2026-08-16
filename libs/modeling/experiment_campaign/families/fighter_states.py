"""Family 6 multi-timescale fighter-state preregistration constants."""

from __future__ import annotations

import csv
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

from ..fighter_states import validate_preregistered_profiles
from ..feature_lineage import (
    build_development_safe_ids,
    decode_development_rows,
    validate_feature_lineage_rows,
)
from ..hashing import (
    canonical_sha256,
    file_sha256,
    read_json,
    tree_inventory,
    write_canonical_json,
)
from ..metrics import event_block_bootstrap_delta, metric_gap, reduce_predictions
from ..protocol import AccessLedger
from .semantic_portfolio import V8_FEATURES


EXPERIMENT_ID = "family-06-multiscale-count-aware-state"
RUN_ALIAS = "family-06-fighter-states"
FROZEN_SPEC_SHA256 = "93FB5CC31AD810B1867FFC8A250DD257AAF74732998D103D56AB8D3A2D309A23"
FROZEN_SOURCE_SHA256 = "157649B780965ECC585F18B3030199CDC0F4FE3013958FFA4095FCF665FDB1EA"
FROZEN_SOURCE = Path(
    r"C:\Users\danhm\mma-ai\worktrees\top10-20260815"
    r"\experiments\top10_20260815\artifacts\01-campaign-harness\frozen\training_data.csv"
)
FIXED_INCUMBENT_ARTIFACT = Path(
    r"C:\Users\danhm\mma-ai\worktrees\top10-20260815"
    r"\experiments\top10_20260815\artifacts\02-family-01-weighted-v8-control"
)
RUN_PATH = "runs/family-06-fighter-states"
ARTIFACT_PATH = "artifacts/07-family-06-fighter-states"
DATA_PATH = "data/experiments/top10_20260815/family-06-fighter-states/matched-state-table.csv"

STATE_FEATURES = (
    "recent_win_rate",
    "recent_ko_rate",
    "recent_submission_rate",
    "decay18_win_rate",
    "decay18_ko_rate",
    "decay18_submission_rate",
    "career_win_rate",
    "career_ko_rate",
    "career_submission_rate",
    "win_trend_short_medium",
    "win_trend_medium_career",
    "sparse_ko_posterior",
    "sparse_submission_posterior",
    "high_count_striking_state",
    "high_count_head_state",
    "inactivity_days",
    "age_win_interaction",
    "state_effective_support",
    "state_win_uncertainty",
)

PROFILE_IDS = (
    "v8-retained-incumbent-control",
    "recent-one-two-fight-state",
    "18-month-decayed-state",
    "career-state",
    "short-medium-career-trend-stack",
    "sparse-event-hierarchical-shrinkage",
    "robust-high-count-striking-state",
    "bounded-all-state-portfolio",
)


def _definition(name: str) -> dict[str, Any]:
    if name.startswith("recent_"):
        version, scale = "recent-last-two-v1", "last-two-prior-fights"
    elif name.startswith("decay18_"):
        version, scale = "decay-18m-half-life-180d-v1", "prior-548-days"
    elif name.startswith("career_"):
        version, scale = "career-count-aware-v1", "all-prior-fights"
    elif name.startswith("sparse_"):
        version, scale = "beta-binomial-hierarchical-v1", "prior-548-days"
    elif name.startswith("high_count_"):
        version, scale = "winsorized-decay-adjusted-performance-v1", "all-prior-snapshots"
    elif name.startswith("win_trend_"):
        version, scale = "bounded-rate-difference-v1", "cross-timescale-prior"
    elif name == "inactivity_days":
        version, scale = "capped-prior-gap-v1", "most-recent-prior-fight"
    elif name == "age_win_interaction":
        version, scale = "prior-age-x-career-win-v1", "all-prior-fights"
    elif name == "state_effective_support":
        version, scale = "log1p-career-exposure-v1", "all-prior-fights"
    else:
        version, scale = "career-beta-uncertainty-v1", "all-prior-fights"
    return {
        "formula_version": version,
        "timescale": scale,
        "exposure": "explicit-prior-fight-effective-count",
        "prior_id": "sparse-beta-1-4",
    }


def build_preregistered_profile() -> dict[str, Any]:
    """Return the exact maximum-eight menu without decoding a source row."""

    groups = (
        (),
        ("recent_win_rate", "recent_ko_rate", "recent_submission_rate", "inactivity_days", "age_win_interaction"),
        ("decay18_win_rate", "decay18_ko_rate", "decay18_submission_rate", "inactivity_days"),
        ("career_win_rate", "career_ko_rate", "career_submission_rate", "state_effective_support"),
        (
            "recent_win_rate",
            "decay18_win_rate",
            "career_win_rate",
            "win_trend_short_medium",
            "win_trend_medium_career",
        ),
        ("sparse_ko_posterior", "sparse_submission_posterior", "state_effective_support", "state_win_uncertainty"),
        ("high_count_striking_state", "high_count_head_state", "state_effective_support"),
        STATE_FEATURES,
    )
    definitions = {name: _definition(name) for name in STATE_FEATURES}
    profiles = []
    for profile_id, names in zip(PROFILE_IDS, groups, strict=True):
        names = list(names)
        profiles.append(
            {
                "id": profile_id,
                "base_feature_names": list(V8_FEATURES),
                "base_feature_sha256": canonical_sha256(list(V8_FEATURES)),
                "feature_names": names,
                "formula_versions": [definitions[name]["formula_version"] for name in names],
                "ordered_feature_sha256": canonical_sha256([*V8_FEATURES, *names]),
            }
        )
    profile = {
        "experiment_id": EXPERIMENT_ID,
        "family_number": 6,
        "frozen_spec_sha256": FROZEN_SPEC_SHA256,
        "source_sha256": FROZEN_SOURCE_SHA256,
        "source_path": "artifacts/01-campaign-harness/frozen/training_data.csv",
        "cutoff": "2025-12-31",
        "registered_priors": {
            "sparse-beta-1-4": {
                "family": "beta-binomial",
                "alpha": 1.0,
                "beta": 4.0,
                "fallback_mean": 0.2,
            }
        },
        "normalization": {"fit_scope": "outer-train-only", "method": "training-median-standard-scale"},
        "construction": {
            "same_date_policy": "exclude-entire-date",
            "recent_fight_count": 2,
            "decay_18m_days": 548,
            "decay_half_life_days": 180,
            "high_count_half_life_days": 365,
            "robust_winsor_limits": [0.1, 0.9],
            "inactivity_cap_days": 730,
            "age_interaction_center_years": 30.0,
        },
        "feature_definitions": definitions,
        "profiles": profiles,
        "outer_years": [2022, 2023, 2024, 2025],
        "inner_validation_year_count": 3,
        "selection": {
            "fit_scope": "prior-inner-only",
            "score": "mean-inner-log-loss",
            "tie_break": list(PROFILE_IDS),
            "outer_label_selection_count": 0,
        },
        "model": {
            "type": "logistic-regression",
            "imputation": "training-median",
            "scaling": "standard",
            "penalty": "l2",
            "C": 1.0,
            "solver": "lbfgs",
            "max_iter": 2000,
            "random_state": 20260815,
        },
        "bootstrap": {"iterations": 2000, "seed": 20260815},
        "database_access": {"used": False, "sql": None, "urls": []},
        "invocation": {"gpu_lease_count": 1, "retry_count": 0, "serialized": True},
    }
    validate_preregistered_profiles(profile)
    return profile


def write_preregistration(campaign_root: Path, *, source_revision: str) -> dict[str, Any]:
    """Persist the frozen menu while every construction and score path is absent."""

    campaign_root = Path(campaign_root)
    profile_path = campaign_root / "profiles/family-06-fighter-states.json"
    preregistration_path = campaign_root / "runs/family-06-fighter-states/preregistration.json"
    artifact_root = campaign_root / "artifacts/07-family-06-fighter-states"
    data_root = campaign_root.parents[1] / "data/experiments/top10_20260815/family-06-fighter-states"
    if profile_path.exists() or preregistration_path.exists() or artifact_root.exists() or data_root.exists():
        raise ValueError("family 6 preregistration destinations must all be absent")
    gate = AccessLedger(campaign_root).gate_status()
    if gate["state"] != "closed" or gate["protected_access_count"] != 0:
        raise ValueError("family 6 preregistration requires the gate closed with zero access")
    profile = build_preregistered_profile()
    write_canonical_json(profile_path, profile)
    preregistration = {
        "experiment_id": EXPERIMENT_ID,
        "family_number": 6,
        "source_revision": source_revision,
        "frozen_spec_sha256": FROZEN_SPEC_SHA256,
        "profile_path": "profiles/family-06-fighter-states.json",
        "profile_sha256": canonical_sha256(profile),
        "profile_file_sha256": file_sha256(profile_path),
        "preregistered_profile_ids": list(PROFILE_IDS),
        "ordered_profile_hashes": {
            item["id"]: item["ordered_feature_sha256"] for item in profile["profiles"]
        },
        "source_file_sha256": profile["source_sha256"],
        "registry_prefix_sha256_before": hashlib.sha256(
            (campaign_root / "registry.jsonl").read_bytes()
        ).hexdigest().upper(),
        "scoring_state": "not-started",
        "selection": profile["selection"],
        "database_access": profile["database_access"],
        "invocation": profile["invocation"],
        "gate_required_state": "closed-zero-access",
        "terminal_failure_rule": "Any lineage, chronology, source, menu, safety, or destination mismatch terminates without retry.",
    }
    write_canonical_json(preregistration_path, preregistration)
    return preregistration


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(
        b"".join(
            json.dumps(dict(row), sort_keys=True, separators=(",", ":")).encode() + b"\n"
            for row in rows
        )
    )


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in Path(path).read_bytes().splitlines()]


def _load_development_table(profile: Mapping[str, Any]):
    import pandas as pd

    if file_sha256(FROZEN_SOURCE) != profile["source_sha256"]:
        raise ValueError("frozen source bytes differ before family 6 construction")
    fold_manifest = read_json(FROZEN_SOURCE.parents[3] / "baseline/fold-manifest.json")
    safe_ids, retired_ids = build_development_safe_ids(fold_manifest)
    source_columns = [
        "fight_id",
        "event_id",
        "fighter1_id",
        "fighter2_id",
        "fighter1_name",
        "fighter2_name",
        "method",
        "event_date",
        "y_true",
        "weightclass_encoded",
        "sig_str_land_per_min_dec_adjperf_dec_avg",
        "sig_str_land_per_min_dec_adjperf_dec_avg_diff",
        "head_land_per_min_dec_adjperf_dec_avg",
        "head_land_per_min_dec_adjperf_dec_avg_diff",
        "age_dec_avg",
        "age_dec_avg_diff",
        *V8_FEATURES,
    ]
    with FROZEN_SOURCE.open(encoding="utf-8", newline="") as source:
        header = next(csv.reader(source))
    columns = list(dict.fromkeys(source_columns))
    indices = tuple(header.index(column) for column in columns)
    with FROZEN_SOURCE.open("rb") as source:
        source.readline()
        rows = decode_development_rows(
            source,
            safe_ids=safe_ids,
            retired_ids=retired_ids,
            indices=indices,
        )
    table = pd.DataFrame(rows, columns=columns)
    identity_columns = {
        "fight_id",
        "event_id",
        "fighter1_id",
        "fighter2_id",
        "fighter1_name",
        "fighter2_name",
        "method",
        "event_date",
    }
    numeric = [column for column in columns if column not in identity_columns]
    table[numeric] = table[numeric].apply(pd.to_numeric, errors="coerce")
    if (
        len(table) != 3_089
        or set(str(value) for value in table["fight_id"]) != set(safe_ids)
        or str(table["event_date"].max()) != "2025-12-13"
        or set(table["y_true"].dropna().astype(int).unique()) != {0, 1}
    ):
        raise ValueError("family 6 development population differs from the exact safe set")
    table["event_year"] = pd.to_datetime(table["event_date"], errors="raise").dt.year
    return table.sort_values(["event_date", "event_id", "fight_id"]).reset_index(drop=True)


def _state_tables(table, profile: Mapping[str, Any]):
    import pandas as pd

    fights = []
    for row in table.to_dict("records"):
        fights.append(
            {
                **{key: row[key] for key in (
                    "fight_id", "event_id", "fighter1_id", "fighter2_id",
                    "fighter1_name", "fighter2_name", "method", "event_date", "y_true",
                )},
                "sig_str_land_state": row["sig_str_land_per_min_dec_adjperf_dec_avg"],
                "sig_str_land_state_diff": row["sig_str_land_per_min_dec_adjperf_dec_avg_diff"],
                "head_land_state": row["head_land_per_min_dec_adjperf_dec_avg"],
                "head_land_state_diff": row["head_land_per_min_dec_adjperf_dec_avg_diff"],
                "age_state": row["age_dec_avg"],
                "age_state_diff": row["age_dec_avg_diff"],
            }
        )
    lineage_rows = build_fighter_state_rows(
        fights,
        profile=profile,
        artifact_sha256=profile["source_sha256"],
    )
    validate_feature_lineage_rows(
        lineage_rows,
        registered_prior_ids=set(profile["registered_priors"]),
    )
    values = {
        (row["fight_id"], row["fighter_id"], row["feature_name"]): float(row["value"])
        for row in lineage_rows
    }
    matched = []
    for fight in fights:
        record = {"fight_id": str(fight["fight_id"])}
        for feature in STATE_FEATURES:
            first = values[(record["fight_id"], str(fight["fighter1_id"]), feature)]
            second = values[(record["fight_id"], str(fight["fighter2_id"]), feature)]
            record[feature] = first - second
        matched.append(record)
    matched_table = pd.DataFrame(matched)
    return lineage_rows, table.merge(matched_table, on="fight_id", validate="one_to_one")


def _inner_evidence(table, profile: Mapping[str, Any], outer_year: int):
    from .semantic_portfolio import _fit_model, _log_loss

    evidence = []
    first_year = outer_year - int(profile["inner_validation_year_count"])
    for validation_year in range(first_year, outer_year):
        train = table[table["event_year"] < validation_year]
        validation = table[table["event_year"] == validation_year]
        if train.empty or validation.empty:
            raise ValueError("family 6 inner chronological split is empty")
        for item in profile["profiles"]:
            features = [*item["base_feature_names"], *item["feature_names"]]
            model = _fit_model(train, features, profile["model"])
            probability = model.predict_proba(validation[features])[:, 1]
            evidence.append(
                {
                    "outer_year": outer_year,
                    "validation_year": validation_year,
                    "profile_id": item["id"],
                    "role": "inner-chronological",
                    "fit_max_date": str(train["event_date"].max()),
                    "validation_min_date": str(validation["event_date"].min()),
                    "validation_max_date": str(validation["event_date"].max()),
                    "validation_row_count": len(validation),
                    "validation_log_loss": _log_loss(validation["y_true"], probability),
                }
            )
    from ..fighter_states import select_state_profile

    return evidence, select_state_profile(evidence, profile=profile, outer_year=outer_year)


def _append_registry(campaign_root: Path, payload: Mapping[str, Any]) -> dict[str, str]:
    registry_path = campaign_root / "registry.jsonl"
    head_path = campaign_root / "registry-head.json"
    before = registry_path.read_bytes()
    records = [json.loads(line) for line in before.splitlines()]
    if any(record["payload"]["experiment_id"] == EXPERIMENT_ID for record in records):
        raise ValueError("family 6 already exists in the registry")
    head = read_json(head_path)
    prefix_before = hashlib.sha256(before).hexdigest().upper()
    if prefix_before != head["registry_prefix_sha256"]:
        raise ValueError("registry head does not match the immutable family-5 prefix")
    record = {
        "payload": dict(payload),
        "prefix_sha256_before": prefix_before,
        "previous_record_sha256": head["last_record_sha256"],
        "sequence": head["record_count"],
    }
    record["record_sha256"] = canonical_sha256(record)
    after = before + json.dumps(record, sort_keys=True, separators=(",", ":")).encode() + b"\n"
    registry_path.write_bytes(after)
    write_canonical_json(
        head_path,
        {
            "last_record_sha256": record["record_sha256"],
            "record_count": len(records) + 1,
            "registry_bytes": len(after),
            "registry_prefix_sha256": hashlib.sha256(after).hexdigest().upper(),
        },
    )
    return {
        "record_sha256": record["record_sha256"],
        "registry_prefix_sha256_before": prefix_before,
        "registry_prefix_sha256_after": hashlib.sha256(after).hexdigest().upper(),
    }


def materialize_family_06(
    campaign_root: Path,
    *,
    source_revision: str,
    preregistration_commit: str,
) -> dict[str, Any]:
    """Construct and score the frozen menu exactly once."""

    from ..fighter_states import select_state_profile
    from .semantic_portfolio import (
        _fit_model,
        _metric_gaps,
        _model_identity,
    )

    campaign_root = Path(campaign_root)
    run_root = campaign_root / RUN_PATH
    artifact_root = campaign_root / ARTIFACT_PATH
    data_root = campaign_root.parents[1] / "data/experiments/top10_20260815/family-06-fighter-states"
    manifest_path = run_root / "manifest.json"
    if artifact_root.exists() or data_root.exists() or manifest_path.exists():
        raise ValueError("family 6 score destination already exists; retries are forbidden")
    inherited_database = [
        key for key in ("DATABASE_URL", "ODDS_DATABASE_URL") if os.environ.get(key)
    ]
    if inherited_database:
        raise ValueError("family 6 refuses inherited database URLs")
    gate = AccessLedger(campaign_root).gate_status()
    if gate["state"] != "closed" or gate["protected_access_count"] != 0:
        raise ValueError("family 6 requires the gate closed with zero access")
    profile_path = campaign_root / "profiles/family-06-fighter-states.json"
    preregistration_path = run_root / "preregistration.json"
    profile = read_json(profile_path)
    preregistration = read_json(preregistration_path)
    validate_preregistered_profiles(profile)
    registry_before = hashlib.sha256((campaign_root / "registry.jsonl").read_bytes()).hexdigest().upper()
    if (
        profile != build_preregistered_profile()
        or preregistration["scoring_state"] != "not-started"
        or preregistration["profile_file_sha256"] != file_sha256(profile_path)
        or preregistration["profile_sha256"] != canonical_sha256(profile)
        or preregistration["registry_prefix_sha256_before"] != registry_before
    ):
        raise ValueError("family 6 was not exactly preregistered before score")

    artifact_root.mkdir(parents=True)
    write_canonical_json(
        artifact_root / "gpu-lease-acquired.json",
        {"lease_id": "family-06-serialized-lease-1", "ordinal": 1, "pid": os.getpid(), "state": "acquired"},
    )
    table = _load_development_table(profile)
    lineage_rows, model_table = _state_tables(table, profile)
    lineage_path = artifact_root / "feature-lineage.jsonl"
    _write_jsonl(lineage_path, lineage_rows)
    data_root.mkdir(parents=True)
    data_path = data_root / "matched-state-table.csv"
    model_table.drop(columns=["event_year"]).to_csv(data_path, index=False, lineterminator="\n")

    staged_outputs = []
    attempts = []
    all_candidate: list[dict[str, Any]] = []
    all_incumbent: list[dict[str, Any]] = []
    all_train: list[dict[str, Any]] = []
    for outer_year in profile["outer_years"]:
        evidence, selection = _inner_evidence(model_table, profile, outer_year)
        selected = next(
            item for item in profile["profiles"]
            if item["id"] == selection["selected_profile_id"]
        )
        features = [*selected["base_feature_names"], *selected["feature_names"]]
        train = model_table[model_table["event_year"] < outer_year]
        outer = model_table[model_table["event_year"] == outer_year]
        model = _fit_model(train, features, profile["model"])
        outer_probability = model.predict_proba(outer[features])[:, 1]
        train_probability = model.predict_proba(train[features])[:, 1]
        incumbent = _read_jsonl(
            FIXED_INCUMBENT_ARTIFACT / f"fold-{outer_year}/outer-predictions.jsonl"
        )
        probability_by_id = {
            str(fight_id): float(probability)
            for fight_id, probability in zip(outer["fight_id"], outer_probability, strict=True)
        }
        if set(probability_by_id) != {str(row["fight_id"]) for row in incumbent}:
            raise ValueError(f"family 6 outer population differs for {outer_year}")
        fit_event_ids = sorted({str(value) for value in train["event_id"]}, key=int)
        predictions = []
        for incumbent_row in incumbent:
            prediction = dict(incumbent_row)
            prediction.update(
                {
                    "probability": probability_by_id[str(incumbent_row["fight_id"])],
                    "fit_event_ids": fit_event_ids,
                    "fit_max_date": str(train["event_date"].max()),
                    "selection_max_date": max(row["validation_max_date"] for row in evidence),
                    "selected_state_profile": selection["selected_profile_id"],
                    "selected_ordered_feature_sha256": selection["selected_ordered_feature_sha256"],
                }
            )
            predictions.append(prediction)
        train_predictions = []
        for (_, row), probability in zip(train.iterrows(), train_probability, strict=True):
            method = str(row["method"]).lower()
            train_predictions.append(
                {
                    "boundary": "Original",
                    "event_date": str(row["event_date"]),
                    "event_id": str(row["event_id"]),
                    "experience": "unknown",
                    "fight_id": str(row["fight_id"]),
                    "fit_scope": "prior-only",
                    "fold": f"train-{outer_year}",
                    "outcome_type": "decision" if "decision" in method else "finish",
                    "probability": float(probability),
                    "weight_class": str(row["weightclass_encoded"]),
                    "y_true": int(row["y_true"]),
                }
            )
        attempts.extend(
            {
                "fold": outer_year,
                "profile_id": profile_id,
                "state": "scored-inner-chronological",
                "mean_inner_log_loss": score,
                "selected": profile_id == selection["selected_profile_id"],
            }
            for profile_id, score in selection["profile_scores"].items()
        )
        staged_outputs.append(
            {
                "year": outer_year,
                "evidence": evidence,
                "selection": selection,
                "predictions": predictions,
                "train_predictions": train_predictions,
                "model": _model_identity(model, features),
            }
        )
        all_candidate.extend(predictions)
        all_incumbent.extend(incumbent)
        all_train.extend(train_predictions)

    metrics = reduce_predictions(all_candidate).as_dict()
    incumbent_metrics = reduce_predictions(all_incumbent).as_dict()
    train_metric_result = reduce_predictions(all_train)
    train_metrics = train_metric_result.as_dict()
    metric_deltas = {
        name: float(metrics[name]) - float(incumbent_metrics[name])
        for name in ("log_loss", "brier", "accuracy")
    }
    intervals = event_block_bootstrap_delta(
        all_candidate,
        all_incumbent,
        iterations=int(profile["bootstrap"]["iterations"]),
        seed=int(profile["bootstrap"]["seed"]),
    )
    calibration_gaps, subgroup_gaps = _metric_gaps(metrics, incumbent_metrics)
    train_gaps = metric_gap(train_metric_result, reduce_predictions(all_candidate))
    promote = metric_deltas["log_loss"] < 0.0 and intervals["log_loss_delta"]["upper"] < 0.0
    decision = {
        "action": "promote-family-06" if promote else "retain-family-01-weighted-v8-control",
        "incumbent_before": "family-01-weighted-v8-control",
        "incumbent_after": EXPERIMENT_ID if promote else "family-01-weighted-v8-control",
        "promoted": promote,
        "rule": "pooled log-loss delta and paired event-block interval upper bound must both be below zero",
    }
    support_summary = {}
    for feature in STATE_FEATURES:
        rows = [row for row in lineage_rows if row["feature_name"] == feature]
        supports = [float(row["effective_support"]) for row in rows]
        uncertainties = [float(row["uncertainty"]) for row in rows]
        support_summary[feature] = {
            "row_count": len(rows),
            "minimum_support": min(supports),
            "mean_support": sum(supports) / len(supports),
            "maximum_support": max(supports),
            "low_support_row_count": sum(value < 2.0 for value in supports),
            "mean_uncertainty": sum(uncertainties) / len(uncertainties),
            "maximum_uncertainty": max(uncertainties),
        }
    adaptive_signal = {
        "selected_profiles": [item["selection"]["selected_profile_id"] for item in staged_outputs],
        "selected_feature_hashes": [item["selection"]["selected_ordered_feature_sha256"] for item in staged_outputs],
        "pooled_log_loss_delta": metric_deltas["log_loss"],
        "low_support_fraction": sum(
            float(row["effective_support"]) < 2.0 for row in lineage_rows
        ) / len(lineage_rows),
    }

    fold_outputs = []
    for output in staged_outputs:
        year = output["year"]
        fold_root = artifact_root / f"fold-{year}"
        _write_jsonl(fold_root / "inner-evidence.jsonl", output["evidence"])
        write_canonical_json(fold_root / "selection.json", output["selection"])
        _write_jsonl(fold_root / "outer-predictions.jsonl", output["predictions"])
        _write_jsonl(fold_root / "train-predictions.jsonl", output["train_predictions"])
        write_canonical_json(fold_root / "model.json", output["model"])
        fold_outputs.append(
            {
                "year": year,
                "selected_profile_id": output["selection"]["selected_profile_id"],
                "selected_ordered_feature_sha256": output["selection"]["selected_ordered_feature_sha256"],
                "evidence_path": f"fold-{year}/inner-evidence.jsonl",
                "evidence_sha256": file_sha256(fold_root / "inner-evidence.jsonl"),
                "selection_path": f"fold-{year}/selection.json",
                "selection_sha256": canonical_sha256(output["selection"]),
                "prediction_path": f"fold-{year}/outer-predictions.jsonl",
                "prediction_sha256": file_sha256(fold_root / "outer-predictions.jsonl"),
                "prediction_row_count": len(output["predictions"]),
                "train_prediction_path": f"fold-{year}/train-predictions.jsonl",
                "train_prediction_sha256": file_sha256(fold_root / "train-predictions.jsonl"),
                "train_prediction_row_count": len(output["train_predictions"]),
                "model_path": f"fold-{year}/model.json",
                "model_sha256": canonical_sha256(output["model"]),
            }
        )
    write_canonical_json(artifact_root / "support-summary.json", support_summary)
    write_canonical_json(
        artifact_root / "source-lineage.json",
        {
            "source_path": profile["source_path"],
            "source_file_sha256": profile["source_sha256"],
            "development_safe_id_count": 3_089,
            "retired_id_count": 178,
            "feature_lineage_path": "feature-lineage.jsonl",
            "feature_lineage_sha256": file_sha256(lineage_path),
            "feature_lineage_row_count": len(lineage_rows),
            "matched_state_table_path": DATA_PATH,
            "matched_state_table_sha256": file_sha256(data_path),
            "outer_label_selection_count": 0,
            "gate_selection_count": 0,
            "normalization_fit_scope": "outer-train-only",
        },
    )
    write_canonical_json(
        artifact_root / "safety.json",
        {
            "database_access": profile["database_access"],
            "gpu_lease_count": 1,
            "production_attempt_count": 1,
            "retry_count": 0,
            "serialized": True,
            "gate_access_count": gate["protected_access_count"],
        },
    )
    write_canonical_json(
        artifact_root / "gpu-lease-released.json",
        {"lease_id": "family-06-serialized-lease-1", "ordinal": 1, "pid": os.getpid(), "state": "released"},
    )
    result = {
        "experiment_id": EXPERIMENT_ID,
        "status": "complete",
        "metrics": metrics,
        "incumbent_metrics": incumbent_metrics,
        "train_metrics": train_metrics,
        "metric_deltas": metric_deltas,
        "calibration_gaps": calibration_gaps,
        "subgroup_gaps": subgroup_gaps,
        "train_gaps": train_gaps,
        "paired_event_block_intervals": intervals,
        "support_summary": support_summary,
        "promotion_decision": decision,
        "adaptive_signal_for_family_07": adaptive_signal,
        "gate_access_count": gate["protected_access_count"],
    }
    write_canonical_json(artifact_root / "result.json", result)
    inventory = tree_inventory(artifact_root)
    _write_jsonl(run_root / "attempts.jsonl", attempts)
    (run_root / "decision.md").write_text(
        f"# Family 6 decision\n\n{decision['action']}: {decision['rule']}.\n",
        encoding="utf-8",
        newline="\n",
    )
    manifest = {
        **result,
        "kind": "family",
        "exit_state": "complete",
        "artifact_path": ARTIFACT_PATH,
        "artifact_tree_sha256": inventory.tree_sha256,
        "artifact_file_count": inventory.file_count,
        "data_path": DATA_PATH,
        "data_sha256": file_sha256(data_path),
        "profile_path": "profiles/family-06-fighter-states.json",
        "profile_sha256": canonical_sha256(profile),
        "profile_file_sha256": file_sha256(profile_path),
        "preregistration_path": f"{RUN_PATH}/preregistration.json",
        "preregistration_commit": preregistration_commit,
        "attempts_path": f"{RUN_PATH}/attempts.jsonl",
        "source_lineage_path": "source-lineage.json",
        "feature_lineage_path": "feature-lineage.jsonl",
        "support_summary_path": "support-summary.json",
        "folds": fold_outputs,
        "source_revision": source_revision,
        "outer_label_selection_count": 0,
        "invocation": profile["invocation"],
        "database_access": profile["database_access"],
        "terminal_failure": None,
    }
    write_canonical_json(manifest_path, manifest)
    registry = _append_registry(
        campaign_root,
        {
            "artifact_path": ARTIFACT_PATH,
            "artifact_tree_sha256": inventory.tree_sha256,
            "experiment_id": EXPERIMENT_ID,
            "kind": "family",
            "manifest_path": f"{RUN_PATH}/manifest.json",
            "manifest_sha256": canonical_sha256(manifest),
            "profile_path": manifest["profile_path"],
            "profile_sha256": manifest["profile_sha256"],
            "status": "complete",
        },
    )
    return {**result, **registry, "artifact_tree_sha256": inventory.tree_sha256}
