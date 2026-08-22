"""Deterministically replay and verify one generated prediction CSV."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from autogluon.tabular import TabularPredictor

from libs.modeling.inference_contract import (
    FORBIDDEN_PREDICTION_COLUMNS,
    validate_weighted_v8_inference_contract,
)
from libs.modeling.portable_artifacts import (
    install_pathlib_pickle_compatibility,
    load_joblib_artifact,
    load_tabular_predictor,
    pathlib_pickle_compatibility,
)


def _require_equal(actual, expected, field: str) -> None:
    if actual != expected:
        raise AssertionError(f"{field}: expected {expected!r}, got {actual!r}")


def _american_probability(odds: int) -> float:
    return abs(odds) / (abs(odds) + 100) if odds < 0 else 100 / (odds + 100)


def _american_profit(odds: int) -> float:
    return 100 / abs(odds) if odds < 0 else odds / 100


def _fair_american_odds(probability: float) -> int:
    decimal = 1 / probability
    if decimal >= 2.0:
        return int((decimal - 1) * 100)
    return int(-100 / (decimal - 1))


def _ai_odds(probability: float) -> str:
    if probability > 0.5:
        return str(int(-100 * probability / (1 - probability)))
    return f"+{int(100 * (1 - probability) / probability)}"


def _model_frame(stats_row: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    values = {}
    for feature in features:
        if feature in stats_row:
            source = feature
        elif f"fighter1_{feature}" in stats_row:
            source = f"fighter1_{feature}"
        else:
            raise AssertionError(f"raw stats lack model feature {feature!r}")
        values[feature] = stats_row.iloc[0][source]
    return pd.DataFrame([values], columns=features)


def _positive_probability(model: TabularPredictor, frame: pd.DataFrame) -> float:
    with pathlib_pickle_compatibility():
        probabilities = model.predict_proba(frame)
    if isinstance(probabilities, pd.DataFrame):
        positive = probabilities[1] if 1 in probabilities.columns else probabilities.iloc[:, -1]
    else:
        array = np.asarray(probabilities, dtype=float)
        positive = array[:, -1] if array.ndim == 2 else array
    values = np.asarray(positive, dtype=float)
    if values.shape != (1,) or not np.isfinite(values[0]) or not 0 <= values[0] <= 1:
        raise AssertionError(f"model returned invalid positive probability {values!r}")
    return float(values[0])


def _formula_check(row: pd.Series, *, fighter1_probability: float) -> dict[str, object]:
    displayed_fighter1_probability = float(row["Fighter1_AI_Prob"])
    displayed_fighter2_probability = float(row["Fighter2_AI_Prob"])
    _require_equal(
        round(displayed_fighter1_probability + displayed_fighter2_probability, 1),
        100.0,
        "AI probability sum",
    )

    raw_fighter1_probability = float(fighter1_probability)
    raw_fighter2_probability = 1 - raw_fighter1_probability
    _require_equal(
        float(row["Fighter1_AI_Prob"]),
        round(raw_fighter1_probability * 100, 1),
        "replayed fighter1 AI probability",
    )
    _require_equal(
        float(row["Fighter2_AI_Prob"]),
        round(raw_fighter2_probability * 100, 1),
        "replayed fighter2 AI probability",
    )

    picked_fighter = row["Fighter1"] if raw_fighter1_probability > 0.5 else row["Fighter2"]
    confidence = max(raw_fighter1_probability, raw_fighter2_probability)
    _require_equal(str(row["AI_Pick"]), picked_fighter, "AI pick")
    _require_equal(float(row["Confidence"]), round(confidence * 100, 1), "confidence")

    ai_odds = _ai_odds(confidence)
    _require_equal(str(row["AI_Odds"]), ai_odds, "AI odds")

    fighter1_odds = int(row["Fighter1_Odds"])
    fighter2_odds = int(row["Fighter2_Odds"])
    implied1 = _american_probability(fighter1_odds)
    implied2 = _american_probability(fighter2_odds)
    overround = implied1 + implied2
    fair1_odds = _fair_american_odds(implied1 / overround)
    fair2_odds = _fair_american_odds(implied2 / overround)
    market1 = round(_american_probability(fair1_odds) * 100, 1)
    market2 = round(_american_probability(fair2_odds) * 100, 1)
    _require_equal(float(row["Fighter1_Market_Prob"]), market1, "fighter1 market probability")
    _require_equal(float(row["Fighter2_Market_Prob"]), market2, "fighter2 market probability")

    picked_odds = fighter1_odds if picked_fighter == row["Fighter1"] else fighter2_odds
    expected_return = confidence * _american_profit(picked_odds) - (1 - confidence)
    ev = int(expected_return > 0)
    _require_equal(int(row["EV"]), ev, "EV flag")
    return {
        "fight": f"{row['Fighter1']} vs {row['Fighter2']}",
        "ai_odds": ai_odds,
        "fair_market_odds": [fair1_odds, fair2_odds],
        "market_probabilities": [market1, market2],
        "ev": ev,
        "expected_return_percent": round(expected_return * 100, 1),
    }


def verify(csv_path: Path, stats_path: Path, model_path: Path) -> dict[str, object]:
    output = pd.read_csv(csv_path, comment="#", dtype={"AI_Odds": "string"})
    stats = pd.read_csv(stats_path)
    if len(output) != len(stats):
        raise AssertionError(f"output rows {len(output)} != prediction stats rows {len(stats)}")
    if output.empty:
        raise AssertionError("prediction output cannot be empty")

    install_pathlib_pickle_compatibility()
    scaler = load_joblib_artifact(model_path / "scaler.pkl")
    contract = validate_weighted_v8_inference_contract(model_path, scaler)
    model = load_tabular_predictor(TabularPredictor, model_path)
    learner = model._learner
    _require_equal(getattr(learner, "sample_weight", None), "sample_weight", "training weight column")
    _require_equal(getattr(learner, "weight_evaluation", None), False, "evaluation weighting")
    features = [
        line.strip()
        for line in (model_path / "feats.txt").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    forbidden_prediction_columns = sorted(FORBIDDEN_PREDICTION_COLUMNS.intersection(features))
    if len(features) != 40 or forbidden_prediction_columns:
        raise AssertionError(
            "prediction input must be exactly 40 features with no weight/date/label columns"
        )
    expected_scale_features = [feature for feature in features if feature != "weightclass_encoded"]
    saved_scale_features = [str(value) for value in scaler.feature_names_in_]
    _require_equal(saved_scale_features, expected_scale_features, "saved scaler feature order")

    rows = []
    discrimination = None
    for _, row in output.iterrows():
        matching = stats[
            (stats["fighter1_name"] == row["Fighter1"])
            & (stats["fighter2_name"] == row["Fighter2"])
        ]
        if len(matching) != 1:
            fight = f"{row['Fighter1']} vs {row['Fighter2']}"
            raise AssertionError(f"expected one stats row for {fight}, found {len(matching)}")

        frame = _model_frame(matching, features)
        scaled = frame.copy()
        scaled.loc[:, saved_scale_features] = scaler.transform(frame[saved_scale_features])
        raw_fighter1_probability = _positive_probability(model, scaled)
        formula = _formula_check(row, fighter1_probability=raw_fighter1_probability)

        if discrimination is None:
            wrong_result = row.copy()
            wrong_result["EV"] = 1 - int(row["EV"])
            try:
                _formula_check(
                    wrong_result,
                    fighter1_probability=raw_fighter1_probability,
                )
            except AssertionError as exc:
                discrimination = {"status": "PASS", "wrong_result_failure": str(exc)}
            else:
                raise AssertionError("formula oracle did not reject a wrong EV result")

        rows.append(
            {
                **formula,
                "ai_probabilities": [
                    round(raw_fighter1_probability * 100, 1),
                    round((1 - raw_fighter1_probability) * 100, 1),
                ],
                "status": "PASS",
            }
        )

    return {
        "status": "PASS",
        "row_count": len(rows),
        "contract": contract,
        "prediction_input": {
            "feature_count": len(features),
            "forbidden_columns": forbidden_prediction_columns,
            "direct_predictor_calls_without_weights": len(rows),
        },
        "weight_boundary": {
            "training_weight_column": "sample_weight",
            "training_weight_rows": contract["training_weight_rows"],
            "validation_unit_weight_rows": contract["evaluation_weight_rows"],
            "weight_evaluation": False,
            "prediction_weight_columns": [],
        },
        "discrimination": discrimination,
        "rows": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=Path, required=True)
    parser.add_argument("--stats", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(verify(args.csv, args.stats, args.model), indent=2))


if __name__ == "__main__":
    main()
