"""Family 5 stable semantic portfolio preregistration and materialization."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from ..hashing import canonical_sha256, file_sha256, read_json, tree_inventory, write_canonical_json
from ..metrics import event_block_bootstrap_delta, metric_gap, reduce_predictions
from ..protocol import AccessLedger
from ..semantic_portfolio import (
    MEASUREMENT_GROUP_IDS,
    select_stable_features,
    validate_preregistered_profile,
)


EXPERIMENT_ID = "family-05-stable-semantic-portfolio"
RUN_PATH = "runs/family-05-semantic-portfolio"
ARTIFACT_PATH = "artifacts/06-family-05-semantic-portfolio"
DATA_PATH = "../../data/experiments/top10_20260815/family-05-semantic-portfolio"
FROZEN_SPEC_SHA256 = "93FB5CC31AD810B1867FFC8A250DD257AAF74732998D103D56AB8D3A2D309A23"
FROZEN_SOURCE_SHA256 = "157649B780965ECC585F18B3030199CDC0F4FE3013958FFA4095FCF665FDB1EA"
V8_ORDERED_FEATURE_SHA256 = "13E545D762A3F1BE4D023D82B8E65D77E41589031051F1F6796D742F25223022"
DEVELOPMENT_FIGHT_COUNT = 3089
GATE_FIGHT_COUNT = 178
POPULATION_FIGHT_COUNT = DEVELOPMENT_FIGHT_COUNT + GATE_FIGHT_COUNT
DEVELOPMENT_MAX_DATE = "2025-12-13"
FROZEN_SOURCE = Path(
    r"C:\Users\danhm\mma-ai\worktrees\top10-20260815"
    r"\experiments\top10_20260815\artifacts\01-campaign-harness"
    r"\frozen\training_data.csv"
)

V8_FEATURES = (
    "age_dec_avg_diff",
    "age_ratio_diff",
    "reach_ratio_dec_avg_diff",
    "days_since_last_fight_dec_avg_diff",
    "sig_str_land_per_min_dec_adjperf_dec_avg_diff",
    "weightclass_encoded",
    "sig_str_land_ratio_dec_adjperf_dec_avg_diff",
    "clinch_def_dec_adjperf_dec_avg_diff",
    "sig_str_acc_dec_adjperf_dec_avg_diff",
    "head_land_ratio_dec_adjperf_dec_avg_diff",
    "head_def_dec_adjperf_dec_avg_diff",
    "head_acc_dec_adjperf_dec_avg_diff",
    "body_def_dec_adjperf_dec_avg_diff",
    "body_acc_dec_adjperf_dec_avg_diff",
    "leg_land_per_min_dec_adjperf_dec_avg_diff",
    "leg_acc_dec_adjperf_dec_avg_diff",
    "distance_land_ratio_dec_adjperf_dec_avg_diff",
    "sig_str_land_pressure_dec_adjperf_dec_avg_diff",
    "distance_acc_dec_adjperf_dec_avg_diff",
    "distance_land_per_min_dec_adjperf_dec_avg_diff",
    "body_land_ratio_dec_adjperf_dec_avg_diff",
    "clinch_land_ratio_dec_adjperf_dec_avg_diff",
    "clinch_acc_dec_adjperf_dec_avg_diff",
    "ground_acc_dec_adjperf_dec_avg_diff",
    "ground_land_per_ctrl_dec_avg_diff",
    "ko_per_sig_str_land_dec_adjperf_dec_avg_diff",
    "ko_ratio_dec_adjperf_dec_avg_diff",
    "sub_att_ratio_dec_adjperf_dec_avg_diff",
    "sub_att_dec_avg_diff",
    "sub_def_dec_adjperf_dec_avg_diff",
    "win_dec_adjperf_dec_avg_diff",
    "rev_per_ctrlopp_dec_adjperf_dec_avg_diff",
    "td_acc_dec_adjperf_dec_avg_diff",
    "td_def_dec_adjperf_dec_avg_diff",
    "ctrl_ratio_dec_adjperf_dec_avg_diff",
    "ctrl_per_min_opp_dec_avg_diff",
    "td_att_opp_dec_avg_diff",
    "td_att_rd1_opp_dec_avg_diff",
    "td_land_per_ctrl_dec_adjperf_dec_avg_diff",
    "td_per_sig_str_att_dec_adjperf_dec_avg_diff",
)

FEATURE_GROUPS = {
    "age_dec_avg_diff": "demographics-experience",
    "age_ratio_diff": "demographics-experience",
    "reach_ratio_dec_avg_diff": "demographics-experience",
    "days_since_last_fight_dec_avg_diff": "demographics-experience",
    "weightclass_encoded": "demographics-experience",
    "sig_str_land_per_min_dec_adjperf_dec_avg_diff": "global-striking-pace-efficiency",
    "sig_str_land_ratio_dec_adjperf_dec_avg_diff": "global-striking-pace-efficiency",
    "sig_str_acc_dec_adjperf_dec_avg_diff": "global-striking-pace-efficiency",
    "head_land_ratio_dec_adjperf_dec_avg_diff": "head-body-leg-targeting",
    "head_def_dec_adjperf_dec_avg_diff": "head-body-leg-targeting",
    "head_acc_dec_adjperf_dec_avg_diff": "head-body-leg-targeting",
    "body_def_dec_adjperf_dec_avg_diff": "head-body-leg-targeting",
    "body_acc_dec_adjperf_dec_avg_diff": "head-body-leg-targeting",
    "leg_land_per_min_dec_adjperf_dec_avg_diff": "head-body-leg-targeting",
    "leg_acc_dec_adjperf_dec_avg_diff": "head-body-leg-targeting",
    "body_land_ratio_dec_adjperf_dec_avg_diff": "head-body-leg-targeting",
    "distance_land_ratio_dec_adjperf_dec_avg_diff": "range-clinch-ground-position",
    "sig_str_land_pressure_dec_adjperf_dec_avg_diff": "range-clinch-ground-position",
    "distance_acc_dec_adjperf_dec_avg_diff": "range-clinch-ground-position",
    "distance_land_per_min_dec_adjperf_dec_avg_diff": "range-clinch-ground-position",
    "clinch_land_ratio_dec_adjperf_dec_avg_diff": "range-clinch-ground-position",
    "clinch_acc_dec_adjperf_dec_avg_diff": "range-clinch-ground-position",
    "ground_acc_dec_adjperf_dec_avg_diff": "range-clinch-ground-position",
    "ground_land_per_ctrl_dec_avg_diff": "range-clinch-ground-position",
    "sub_att_ratio_dec_adjperf_dec_avg_diff": "takedown-control-submission",
    "sub_att_dec_avg_diff": "takedown-control-submission",
    "sub_def_dec_adjperf_dec_avg_diff": "takedown-control-submission",
    "rev_per_ctrlopp_dec_adjperf_dec_avg_diff": "takedown-control-submission",
    "td_acc_dec_adjperf_dec_avg_diff": "takedown-control-submission",
    "td_def_dec_adjperf_dec_avg_diff": "takedown-control-submission",
    "ctrl_ratio_dec_adjperf_dec_avg_diff": "takedown-control-submission",
    "ctrl_per_min_opp_dec_avg_diff": "takedown-control-submission",
    "td_att_opp_dec_avg_diff": "takedown-control-submission",
    "td_att_rd1_opp_dec_avg_diff": "takedown-control-submission",
    "td_land_per_ctrl_dec_adjperf_dec_avg_diff": "takedown-control-submission",
    "td_per_sig_str_att_dec_adjperf_dec_avg_diff": "takedown-control-submission",
    "ko_per_sig_str_land_dec_adjperf_dec_avg_diff": "damage-finish",
    "ko_ratio_dec_adjperf_dec_avg_diff": "damage-finish",
    "clinch_def_dec_adjperf_dec_avg_diff": "opponent-style-strength-of-schedule",
    "win_dec_adjperf_dec_avg_diff": "opponent-style-strength-of-schedule",
}


def _source_header(source_path: Path) -> tuple[str, ...]:
    with Path(source_path).open(encoding="utf-8", newline="") as source:
        return tuple(next(csv.reader(source)))


def _header_sha256(header: tuple[str, ...]) -> str:
    return hashlib.sha256("\n".join(header).encode()).hexdigest().upper()


def build_preregistered_profile(source_path: Path = FROZEN_SOURCE) -> dict[str, Any]:
    """Build the exact eight-profile menu without reading any score or label row."""

    header = _source_header(source_path)
    if canonical_sha256(list(V8_FEATURES)) != V8_ORDERED_FEATURE_SHA256:
        raise ValueError("v8 feature anchor differs from the immutable ordered list")
    if set(FEATURE_GROUPS) != set(V8_FEATURES):
        raise ValueError("every v8 candidate must have exactly one authored semantic group")
    header_sha256 = _header_sha256(header)
    candidates = [
        {
            "name": feature,
            "semantic_id": feature,
            "measurement_group": FEATURE_GROUPS[feature],
            "source_file_sha256": FROZEN_SOURCE_SHA256,
            "source_header_sha256": header_sha256,
            "formula": f"identity({feature})",
            "available_by": "2014-01-01",
            "domain_redundancy_rank": sum(
                1
                for prior in V8_FEATURES[: index + 1]
                if FEATURE_GROUPS[prior] == FEATURE_GROUPS[feature]
            ),
        }
        for index, feature in enumerate(V8_FEATURES)
    ]
    profiles = []
    for profile_id, groups in (
        ("v8-control", MEASUREMENT_GROUP_IDS),
        *((group, (group,)) for group in MEASUREMENT_GROUP_IDS),
    ):
        features = [feature for feature in V8_FEATURES if FEATURE_GROUPS[feature] in groups]
        profiles.append(
            {
                "id": profile_id,
                "included_groups": list(groups),
                "ordered_features": features,
                "ordered_feature_sha256": canonical_sha256(features),
            }
        )
    profile = {
        "experiment_id": EXPERIMENT_ID,
        "family_number": 5,
        "frozen_spec_sha256": FROZEN_SPEC_SHA256,
        "frozen_source": {
            "path": "artifacts/01-campaign-harness/frozen/training_data.csv",
            "absolute_path": str(Path(source_path)),
            "sha256": FROZEN_SOURCE_SHA256,
            "ordered_header_sha256": header_sha256,
            "cutoff": "2025-12-31",
        },
        "v8_ordered_features": list(V8_FEATURES),
        "v8_ordered_feature_sha256": V8_ORDERED_FEATURE_SHA256,
        "measurement_group_ids": list(MEASUREMENT_GROUP_IDS),
        "candidate_features": candidates,
        "measurement_profiles": profiles,
        "outer_years": [2022, 2023, 2024, 2025],
        "inner_validation_year_count": 3,
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
        "selection": {
            "evidence_role": "inner-chronological",
            "profile_score": "mean-inner-log-loss",
            "profile_tie_break": [item["id"] for item in profiles],
            "stability_threshold": 2 / 3,
            "drop_column_min_improvement": 0.0,
            "minimum_fold_support": 3,
            "domain_redundancy_cap": 3,
            "tie_break": "profile-order-then-feature-order",
            "combined_row_importance_role": "non-selection",
        },
        "bootstrap": {"iterations": 2000, "seed": 20260815},
        "promotion_rule": "pooled log-loss delta and paired event-block interval upper bound must both be below zero",
        "incumbent_id": "family-01-weighted-v8-control",
        "historical_authored_documents_role": "group-membership-proposal-only",
        "outer_label_roles": ["final-metrics-only"],
        "gate_required_state": "closed-zero-access",
    }
    validate_preregistered_profile(profile, source_header=header)
    return profile


def write_preregistration(campaign_root: Path) -> dict[str, Any]:
    """Persist the frozen menu and not-started commitment before any score."""

    campaign_root = Path(campaign_root)
    gate = AccessLedger(campaign_root).gate_status()
    if gate["state"] != "closed" or gate["protected_access_count"] != 0:
        raise ValueError("family 5 preregistration requires a closed zero-access gate")
    profile_path = campaign_root / "profiles/family-05-semantic-portfolio.json"
    preregistration_path = campaign_root / RUN_PATH / "preregistration.json"
    if profile_path.exists() or preregistration_path.exists():
        raise ValueError("family 5 preregistration destination already exists")
    profile = build_preregistered_profile()
    write_canonical_json(profile_path, profile)
    registry_bytes = (campaign_root / "registry.jsonl").read_bytes()
    profiles = profile["measurement_profiles"]
    preregistration = {
        "experiment_id": EXPERIMENT_ID,
        "family_number": 5,
        "frozen_spec_sha256": FROZEN_SPEC_SHA256,
        "profile_path": "profiles/family-05-semantic-portfolio.json",
        "profile_file_sha256": file_sha256(profile_path),
        "profile_sha256": canonical_sha256(profile),
        "registry_prefix_sha256_before": hashlib.sha256(registry_bytes).hexdigest().upper(),
        "scoring_state": "not-started",
        "preregistered_profile_ids": [item["id"] for item in profiles],
        "ordered_profile_hashes": {
            item["id"]: item["ordered_feature_sha256"] for item in profiles
        },
        "source_file_sha256": FROZEN_SOURCE_SHA256,
        "source_header_sha256": profile["frozen_source"]["ordered_header_sha256"],
        "selection": profile["selection"],
        "outer_label_roles": profile["outer_label_roles"],
        "gate_required_state": profile["gate_required_state"],
        "terminal_failure_rule": "Any lineage, chronology, menu, source, gate, or destination mismatch terminates without retry.",
    }
    write_canonical_json(preregistration_path, preregistration)
    return preregistration


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in Path(path).read_bytes().splitlines()]


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(
        b"".join(
            json.dumps(dict(row), sort_keys=True, separators=(",", ":")).encode() + b"\n"
            for row in rows
        )
    )


def _fit_model(table, features: Sequence[str], model_profile: Mapping[str, Any]):
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    model = Pipeline(
        (
            ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
            ("scaler", StandardScaler()),
            (
                "classifier",
                LogisticRegression(
                    C=float(model_profile["C"]),
                    penalty=str(model_profile["penalty"]),
                    solver=str(model_profile["solver"]),
                    max_iter=int(model_profile["max_iter"]),
                    random_state=int(model_profile["random_state"]),
                ),
            ),
        )
    )
    model.fit(table[list(features)], table["y_true"].astype(int))
    return model


def _log_loss(labels, probabilities) -> float:
    from sklearn.metrics import log_loss

    return float(log_loss(labels.astype(int), probabilities, labels=[0, 1]))


def _model_identity(model, features: Sequence[str]) -> dict[str, Any]:
    imputer = model.named_steps["imputer"]
    scaler = model.named_steps["scaler"]
    classifier = model.named_steps["classifier"]
    return {
        "ordered_features": list(features),
        "ordered_feature_sha256": canonical_sha256(list(features)),
        "imputer_statistics": [float(value) for value in imputer.statistics_],
        "scaler_mean": [float(value) for value in scaler.mean_],
        "scaler_scale": [float(value) for value in scaler.scale_],
        "coefficients": [float(value) for value in classifier.coef_[0]],
        "intercept": float(classifier.intercept_[0]),
    }


def _partition_development_ids(
    fold_manifest: Mapping[str, Any],
) -> tuple[tuple[str, ...], frozenset[str]]:
    population_fight_ids = tuple(
        str(value) for value in fold_manifest["population_fight_ids"]
    )
    gate_roster = tuple(str(item["fight_id"]) for item in fold_manifest["gate_roster"])
    gate_ids = frozenset(gate_roster)
    safe_ids = tuple(
        fight_id for fight_id in population_fight_ids if fight_id not in gate_ids
    )
    if (
        len(population_fight_ids) != POPULATION_FIGHT_COUNT
        or len(set(population_fight_ids)) != POPULATION_FIGHT_COUNT
        or len(gate_roster) != GATE_FIGHT_COUNT
        or len(gate_ids) != GATE_FIGHT_COUNT
        or len(safe_ids) != DEVELOPMENT_FIGHT_COUNT
        or not set(safe_ids).isdisjoint(gate_ids)
        or set(safe_ids) | gate_ids != set(population_fight_ids)
    ):
        raise ValueError("development and gate identities are not the exact safe partition")
    return safe_ids, gate_ids


def _decode_full_row(raw: bytes, indices: Sequence[int]) -> list[str]:
    parsed = next(csv.reader([raw.decode("utf-8")]))
    return [parsed[index] for index in indices]


def _decode_safe_rows(
    raw_rows: Iterable[bytes],
    *,
    safe_ids: Sequence[str],
    gate_ids: frozenset[str],
    indices: Sequence[int],
) -> list[list[str]]:
    safe_ids = tuple(str(fight_id) for fight_id in safe_ids)
    safe_set = frozenset(safe_ids)
    gate_ids = frozenset(str(fight_id) for fight_id in gate_ids)
    if (
        len(safe_ids) != DEVELOPMENT_FIGHT_COUNT
        or len(safe_set) != DEVELOPMENT_FIGHT_COUNT
        or len(gate_ids) != GATE_FIGHT_COUNT
        or not safe_set.isdisjoint(gate_ids)
    ):
        raise ValueError("row decoder requires the exact disjoint safe partition")
    rows = []
    for raw in raw_rows:
        fight_id = raw.split(b",", 1)[0].strip(b'"').decode("utf-8")
        if fight_id in gate_ids:
            continue
        if fight_id in safe_set:
            rows.append(_decode_full_row(raw, indices))
    return rows


def _load_development_table(profile: Mapping[str, Any]):
    import pandas as pd

    source_path = Path(profile["frozen_source"]["absolute_path"])
    if file_sha256(source_path) != profile["frozen_source"]["sha256"]:
        raise ValueError("frozen source bytes differ before score")
    fold_manifest = read_json(source_path.parents[3] / "baseline/fold-manifest.json")
    safe_ids, gate_ids = _partition_development_ids(fold_manifest)
    columns = [
        "fight_id",
        "event_id",
        "method",
        "event_date",
        "y_true",
        *profile["v8_ordered_features"],
    ]
    header = _source_header(source_path)
    indices = [header.index(column) for column in columns]
    with source_path.open("rb") as source:
        source.readline()
        rows = _decode_safe_rows(
            source,
            safe_ids=safe_ids,
            gate_ids=gate_ids,
            indices=indices,
        )
    table = pd.DataFrame(rows, columns=columns)
    table["fight_id"] = table["fight_id"].astype("string")
    table["event_id"] = table["event_id"].astype("string")
    table["event_date"] = table["event_date"].astype("string")
    table["y_true"] = pd.to_numeric(table["y_true"], errors="raise")
    table[list(profile["v8_ordered_features"])] = table[
        list(profile["v8_ordered_features"])
    ].apply(pd.to_numeric, errors="coerce")
    table["event_year"] = pd.to_datetime(table["event_date"], errors="raise").dt.year
    if (
        len(table) != len(safe_ids)
        or set(str(value) for value in table["fight_id"]) != set(safe_ids)
        or set(table["y_true"].dropna().astype(int).unique()) != {0, 1}
    ):
        raise ValueError("frozen development population or labels differ")
    if str(table["event_date"].max()) != DEVELOPMENT_MAX_DATE:
        raise ValueError("development rows do not end at the frozen 2025-12-13 boundary")
    return table.sort_values(["event_date", "event_id", "fight_id"]).reset_index(drop=True)


def _inner_evidence(table, profile: Mapping[str, Any], outer_year: int) -> tuple[list[dict], dict]:
    profile_evidence: list[dict[str, Any]] = []
    profile_scores: dict[str, float] = {}
    stable: dict[str, dict[str, Any]] = {}
    first_year = outer_year - int(profile["inner_validation_year_count"])
    for measurement_profile in profile["measurement_profiles"]:
        profile_id = measurement_profile["id"]
        features = list(measurement_profile["ordered_features"])
        validation_scores = []
        evidence = []
        for validation_year in range(first_year, outer_year):
            train = table[table["event_year"] < validation_year]
            validation = table[table["event_year"] == validation_year]
            if train.empty or validation.empty:
                raise ValueError("inner chronological fold is empty")
            full_model = _fit_model(train, features, profile["model"])
            full_probability = full_model.predict_proba(validation[features])[:, 1]
            full_loss = _log_loss(validation["y_true"], full_probability)
            validation_scores.append(full_loss)
            coefficients = full_model.named_steps["classifier"].coef_[0]
            for index, feature in enumerate(features):
                dropped = [candidate for candidate in features if candidate != feature]
                dropped_model = _fit_model(train, dropped, profile["model"])
                dropped_probability = dropped_model.predict_proba(validation[dropped])[:, 1]
                row = {
                    "profile_id": profile_id,
                    "feature": feature,
                    "fold": validation_year,
                    "direction": 1 if float(coefficients[index]) >= 0 else -1,
                    "drop_column_delta": _log_loss(validation["y_true"], dropped_probability)
                    - full_loss,
                    "validation_log_loss": full_loss,
                    "train_row_count": len(train),
                    "validation_row_count": len(validation),
                    "train_max_date": str(train["event_date"].max()),
                    "validation_min_date": str(validation["event_date"].min()),
                    "validation_max_date": str(validation["event_date"].max()),
                    "role": "inner-chronological",
                }
                evidence.append(row)
                profile_evidence.append(row)
        profile_scores[profile_id] = sum(validation_scores) / len(validation_scores)
        stable[profile_id] = select_stable_features(
            evidence,
            profile=profile,
            outer_year=outer_year,
        )
    eligible = [
        item["id"]
        for item in profile["measurement_profiles"]
        if stable[item["id"]]["selected_features"]
    ]
    if not eligible:
        raise ValueError("all preregistered profiles failed stability selection")
    tie_break = tuple(profile["selection"]["profile_tie_break"])
    selected_profile = min(
        eligible,
        key=lambda profile_id: (profile_scores[profile_id], tie_break.index(profile_id)),
    )
    selection = {
        **stable[selected_profile],
        "selected_profile_id": selected_profile,
        "profile_scores": profile_scores,
        "eligible_profile_ids": eligible,
        "scored_profile_count": len(profile_scores),
        "selection_basis": "minimum mean inner log-loss among stability-eligible profiles",
    }
    return profile_evidence, selection


def _metric_gaps(candidate: Mapping[str, Any], incumbent: Mapping[str, Any]) -> tuple[dict, dict]:
    calibration = {
        name: float(candidate[name]) - float(incumbent[name])
        for name in ("calibration_intercept", "calibration_slope", "ece")
    }
    subgroup = {
        group: {
            metric: float(candidate["subgroup_metrics"][group][metric])
            - float(incumbent["subgroup_metrics"][group][metric])
            for metric in ("log_loss", "brier", "accuracy")
        }
        for group in sorted(candidate["subgroup_metrics"])
    }
    return calibration, subgroup


def promotion_decision(metric_deltas: Mapping[str, float], intervals: Mapping[str, Any]) -> dict:
    promote = (
        float(metric_deltas["log_loss"]) < 0.0
        and float(intervals["log_loss_delta"]["upper"]) < 0.0
    )
    return {
        "action": "promote-family-05" if promote else "retain-family-01-weighted-v8-control",
        "incumbent_before": "family-01-weighted-v8-control",
        "incumbent_after": EXPERIMENT_ID if promote else "family-01-weighted-v8-control",
        "promoted": promote,
        "rule": "pooled log-loss delta and paired event-block interval upper bound must both be below zero",
    }


def _append_registry(campaign_root: Path, payload: Mapping[str, Any]) -> dict[str, str]:
    registry_path = campaign_root / "registry.jsonl"
    head_path = campaign_root / "registry-head.json"
    before = registry_path.read_bytes()
    records = [json.loads(line) for line in before.splitlines()]
    if any(record["payload"]["experiment_id"] == EXPERIMENT_ID for record in records):
        raise ValueError("family 5 already exists in the registry")
    head = read_json(head_path)
    prefix_before = hashlib.sha256(before).hexdigest().upper()
    if prefix_before != head["registry_prefix_sha256"]:
        raise ValueError("registry head does not match the immutable prefix")
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


def materialize_family_05(
    campaign_root: Path,
    *,
    source_revision: str,
    preregistration_commit: str,
) -> dict[str, Any]:
    """Score the frozen menu once and append one terminal family result."""

    campaign_root = Path(campaign_root)
    artifact_root = campaign_root / ARTIFACT_PATH
    data_root = campaign_root.parents[1] / "data/experiments/top10_20260815/family-05-semantic-portfolio"
    run_root = campaign_root / RUN_PATH
    manifest_path = run_root / "manifest.json"
    if artifact_root.exists() or data_root.exists() or manifest_path.exists():
        raise ValueError("family 5 score destination already exists; retries are forbidden")
    gate = AccessLedger(campaign_root).gate_status()
    if gate["state"] != "closed" or gate["protected_access_count"] != 0:
        raise ValueError("family 5 requires the gate closed with zero access")
    profile_path = campaign_root / "profiles/family-05-semantic-portfolio.json"
    preregistration_path = run_root / "preregistration.json"
    profile = read_json(profile_path)
    preregistration = read_json(preregistration_path)
    header = _source_header(Path(profile["frozen_source"]["absolute_path"]))
    validate_preregistered_profile(profile, source_header=header)
    if (
        preregistration["scoring_state"] != "not-started"
        or preregistration["profile_file_sha256"] != file_sha256(profile_path)
        or preregistration["profile_sha256"] != canonical_sha256(profile)
        or preregistration["registry_prefix_sha256_before"]
        != hashlib.sha256((campaign_root / "registry.jsonl").read_bytes()).hexdigest().upper()
    ):
        raise ValueError("family 5 profile was not preregistered at the current prefix")

    table = _load_development_table(profile)
    fixed_incumbent = FROZEN_SOURCE.parents[2] / "02-family-01-weighted-v8-control"
    fold_outputs = []
    all_candidate: list[dict[str, Any]] = []
    all_incumbent: list[dict[str, Any]] = []
    all_train: list[dict[str, Any]] = []
    attempts = []
    staged_outputs = []
    for outer_year in profile["outer_years"]:
        evidence, selection = _inner_evidence(table, profile, outer_year)
        features = selection["selected_features"]
        train = table[table["event_year"] < outer_year]
        outer = table[table["event_year"] == outer_year]
        model = _fit_model(train, features, profile["model"])
        outer_probability = model.predict_proba(outer[features])[:, 1]
        train_probability = model.predict_proba(train[features])[:, 1]
        incumbent = _read_jsonl(fixed_incumbent / f"fold-{outer_year}/outer-predictions.jsonl")
        outer_by_id = {
            str(fight_id): probability
            for fight_id, probability in zip(outer["fight_id"], outer_probability, strict=True)
        }
        if set(outer_by_id) != {str(row["fight_id"]) for row in incumbent}:
            raise ValueError(f"family 5 outer population differs for {outer_year}")
        fit_event_ids = sorted({str(value) for value in train["event_id"]}, key=lambda value: int(value))
        predictions = []
        for incumbent_row in incumbent:
            row = dict(incumbent_row)
            row.update(
                {
                    "probability": float(outer_by_id[str(row["fight_id"])]),
                    "fit_event_ids": fit_event_ids,
                    "fit_max_date": str(train["event_date"].max()),
                    "selection_max_date": max(item["validation_max_date"] for item in evidence),
                    "selected_measurement_profile": selection["selected_profile_id"],
                    "selected_feature_sha256": selection["selected_feature_sha256"],
                }
            )
            predictions.append(row)
        train_rows = []
        for (_, source_row), probability in zip(train.iterrows(), train_probability, strict=True):
            method = str(source_row["method"]).lower()
            train_rows.append(
                {
                    "boundary": "Original",
                    "event_date": str(source_row["event_date"]),
                    "event_id": str(source_row["event_id"]),
                    "experience": "unknown",
                    "fight_id": str(source_row["fight_id"]),
                    "fit_scope": "prior-only",
                    "fold": f"train-{outer_year}",
                    "outcome_type": "decision" if "decision" in method else "finish",
                    "probability": float(probability),
                    "weight_class": str(source_row["weightclass_encoded"]),
                    "y_true": int(source_row["y_true"]),
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
                "train_predictions": train_rows,
                "model": _model_identity(model, features),
            }
        )
        all_candidate.extend(predictions)
        all_incumbent.extend(incumbent)
        all_train.extend(train_rows)

    metrics = reduce_predictions(all_candidate).as_dict()
    incumbent_metrics = reduce_predictions(all_incumbent).as_dict()
    train_metrics = reduce_predictions(all_train).as_dict()
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
    train_gaps = metric_gap(reduce_predictions(all_train), reduce_predictions(all_candidate))
    decision = promotion_decision(metric_deltas, intervals)
    adaptive_signal = {
        "selected_profiles": [item["selection"]["selected_profile_id"] for item in staged_outputs],
        "selected_feature_hashes": [item["selection"]["selected_feature_sha256"] for item in staged_outputs],
        "selected_feature_counts": [len(item["selection"]["selected_features"]) for item in staged_outputs],
        "pooled_log_loss_delta": metric_deltas["log_loss"],
        "pooled_ece_delta": calibration_gaps["ece"],
    }

    data_root.mkdir(parents=True)
    data_path = data_root / "development-table.csv"
    table.drop(columns=["event_year"]).to_csv(data_path, index=False, lineterminator="\n")
    lineage = {
        "source_path": profile["frozen_source"]["path"],
        "source_file_sha256": profile["frozen_source"]["sha256"],
        "source_header_sha256": profile["frozen_source"]["ordered_header_sha256"],
        "development_table_path": "data/experiments/top10_20260815/family-05-semantic-portfolio/development-table.csv",
        "development_table_sha256": file_sha256(data_path),
        "candidate_feature_sha256": canonical_sha256(profile["v8_ordered_features"]),
        "candidate_count": len(profile["candidate_features"]),
        "outer_label_selection_count": 0,
        "gate_selection_count": 0,
        "combined_row_importance_used": False,
        "historical_authored_documents_role": profile["historical_authored_documents_role"],
    }
    write_canonical_json(artifact_root / "source-lineage.json", lineage)
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
                "selected_features": output["selection"]["selected_features"],
                "selected_feature_sha256": output["selection"]["selected_feature_sha256"],
                "selection_path": f"fold-{year}/selection.json",
                "selection_sha256": canonical_sha256(output["selection"]),
                "evidence_path": f"fold-{year}/inner-evidence.jsonl",
                "evidence_sha256": file_sha256(fold_root / "inner-evidence.jsonl"),
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
        "promotion_decision": decision,
        "adaptive_signal_for_family_06": adaptive_signal,
        "gate_access_count": gate["protected_access_count"],
    }
    write_canonical_json(artifact_root / "result.json", result)
    inventory = tree_inventory(artifact_root)
    _write_jsonl(run_root / "attempts.jsonl", attempts)
    (run_root / "decision.md").write_text(
        f"# Family 5 decision\n\n{decision['action']}: {decision['rule']}.\n",
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
        "data_path": lineage["development_table_path"],
        "data_sha256": lineage["development_table_sha256"],
        "profile_path": "profiles/family-05-semantic-portfolio.json",
        "profile_sha256": canonical_sha256(profile),
        "profile_file_sha256": file_sha256(profile_path),
        "preregistration_path": f"{RUN_PATH}/preregistration.json",
        "preregistration_commit": preregistration_commit,
        "attempts_path": f"{RUN_PATH}/attempts.jsonl",
        "source_lineage_path": "source-lineage.json",
        "folds": fold_outputs,
        "source_revision": source_revision,
        "outer_label_selection_count": 0,
        "combined_row_importance_used": False,
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
