"""Deterministically replay and verify one generated prediction CSV."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from libs.feature_store.inference import filter_features_for_model
from libs.modeling.inference_contract import (
    inference_features_to_scale,
    validate_weighted_v8_inference_contract,
)
from libs.modeling.portable_artifacts import (
    install_pathlib_pickle_compatibility,
    load_joblib_artifact,
)
from predict import (
    BFOLatestOddsOnly,
    american_odds_to_prob,
    convert_prob_to_american_odds,
    get_predictions,
    has_positive_ev,
    load_model_and_calibrator,
)


def _require_equal(actual, expected, field: str) -> None:
    if actual != expected:
        raise AssertionError(f"{field}: expected {expected!r}, got {actual!r}")


def _formula_check(row: pd.Series, *, discriminate: bool = False) -> dict[str, object]:
    fighter1_probability = float(row["Fighter1_AI_Prob"])
    fighter2_probability = float(row["Fighter2_AI_Prob"])
    _require_equal(round(fighter1_probability + fighter2_probability, 1), 100.0, "AI probability sum")

    picked_fighter = row["Fighter1"] if fighter1_probability > 50 else row["Fighter2"]
    confidence = max(fighter1_probability, fighter2_probability)
    _require_equal(str(row["AI_Pick"]), picked_fighter, "AI pick")
    _require_equal(float(row["Confidence"]), confidence, "confidence")

    ai_odds = convert_prob_to_american_odds(confidence / 100)
    _require_equal(str(row["AI_Odds"]), ai_odds, "AI odds")

    fighter1_odds = int(row["Fighter1_Odds"])
    fighter2_odds = int(row["Fighter2_Odds"])
    fair1, fair2 = BFOLatestOddsOnly().remove_vig(fighter1_odds, fighter2_odds)
    market1 = round(american_odds_to_prob(fair1) * 100, 1)
    market2 = round(american_odds_to_prob(fair2) * 100, 1)
    _require_equal(float(row["Fighter1_Market_Prob"]), market1, "fighter1 market probability")
    _require_equal(float(row["Fighter2_Market_Prob"]), market2, "fighter2 market probability")

    picked_odds = fighter1_odds if picked_fighter == row["Fighter1"] else fighter2_odds
    ev = int(has_positive_ev(ai_odds, picked_odds))
    if discriminate:
        ev = 1 - ev
    _require_equal(int(row["EV"]), ev, "EV flag")
    return {
        "fight": f"{row['Fighter1']} vs {row['Fighter2']}",
        "ai_odds": ai_odds,
        "market_probabilities": [market1, market2],
        "ev": ev,
    }


def verify(csv_path: Path, stats_path: Path, model_path: Path) -> dict[str, object]:
    output = pd.read_csv(csv_path, comment="#", dtype={"AI_Odds": "string"})
    stats = pd.read_csv(stats_path)
    if len(output) != len(stats):
        raise AssertionError(f"output rows {len(output)} != prediction stats rows {len(stats)}")

    try:
        _formula_check(output.iloc[0], discriminate=True)
    except AssertionError as exc:
        discrimination = {"status": "PASS", "wrong_result_failure": str(exc)}
    else:
        raise AssertionError("formula oracle did not reject an inverted EV result")

    install_pathlib_pickle_compatibility()
    scaler = load_joblib_artifact(model_path / "scaler.pkl")
    contract = validate_weighted_v8_inference_contract(model_path, scaler)
    model, calibrator = load_model_and_calibrator(model_path, False)
    features = [
        line.strip()
        for line in (model_path / "feats.txt").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    rows = []
    for _, row in output.iterrows():
        formula = _formula_check(row)
        matching = stats[
            (stats["fighter1_name"] == row["Fighter1"])
            & (stats["fighter2_name"] == row["Fighter2"])
        ]
        if len(matching) != 1:
            raise AssertionError(f"expected one stats row for {formula['fight']}, found {len(matching)}")

        frame = filter_features_for_model(matching, features)
        frame = frame.rename(
            columns={
                "fighter1_weightclass_encoded": "weightclass_encoded",
                "fighter1_days_since_last_fight_dec_avg": "days_since_last_fight_dec_avg",
            }
        )
        scaled = frame.copy()
        scale_features = inference_features_to_scale(scaled.columns)
        scaled.loc[:, scale_features] = scaler.transform(frame[scale_features])
        probabilities = get_predictions(model, calibrator, scaled, False)
        fighter1_probability = round(float(probabilities.iloc[0][1]) * 100, 1)
        fighter2_probability = round((1 - float(probabilities.iloc[0][1])) * 100, 1)
        _require_equal(float(row["Fighter1_AI_Prob"]), fighter1_probability, "replayed fighter1 AI probability")
        _require_equal(float(row["Fighter2_AI_Prob"]), fighter2_probability, "replayed fighter2 AI probability")
        rows.append(
            {
                **formula,
                "ai_probabilities": [fighter1_probability, fighter2_probability],
                "status": "PASS",
            }
        )

    return {
        "status": "PASS",
        "row_count": len(rows),
        "contract": contract,
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
