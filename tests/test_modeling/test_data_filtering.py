from pathlib import Path

import pandas as pd
import pytest

from libs.feature_store.features import vSeven_testing2
from libs.modeling.data_preparation import DataPreparation
from libs.modeling.data_utils import filter_fights


def _fight_rows() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "fight_id": [1, 2, 3],
            "event_date": ["2024-01-01", "2024-02-01", "2024-03-01"],
            "fighter1_id": [11, 12, 13],
            "fighter2_id": [21, 22, 23],
            "method": ["decision - unanimous"] * 3,
            "y_true": [1, 0, 1],
            "selected_feature": [0.1, 0.2, 0.3],
            "f1_sevenday_vigless_ip_opening_odds": [0.4, 0.5, 0.6],
        }
    )


def _filter(df: pd.DataFrame, required_features=None) -> pd.DataFrame:
    return filter_fights(
        df,
        threshold=0,
        date="2024-01-01",
        include_split_dec=False,
        required_features=required_features,
    )


def test_unused_all_null_column_does_not_change_eligibility():
    fights = _fight_rows()
    expected = _filter(fights, required_features=["selected_feature"])

    with_unused_null = fights.assign(unused_feature=pd.NA)
    actual = _filter(with_unused_null, required_features=["selected_feature"])

    pd.testing.assert_frame_equal(
        actual[expected.columns], expected, check_dtype=False
    )


def test_selected_feature_null_drops_row_but_unused_odds_null_does_not():
    fights = _fight_rows()
    fights.loc[0, "selected_feature"] = pd.NA
    fights["f1_sevenday_vigless_ip_opening_odds"] = pd.NA

    filtered = _filter(fights, required_features=["selected_feature"])

    assert filtered["fight_id"].tolist() == [2, 3]


def test_required_odds_nulls_are_reported_separately_and_dropped(capsys):
    fights = _fight_rows()
    fights.loc[0, "selected_feature"] = pd.NA
    fights.loc[1, "f1_sevenday_vigless_ip_opening_odds"] = pd.NA

    filtered = _filter(
        fights,
        required_features=[
            "selected_feature",
            "f1_sevenday_vigless_ip_opening_odds",
        ],
    )

    assert filtered["fight_id"].tolist() == [3]
    output = capsys.readouterr().out
    assert "required non-odds model features: 1" in output
    assert "required odds features: 1" in output


def test_missing_required_schema_reports_odds_separately():
    with pytest.raises(ValueError) as exc_info:
        _filter(
            _fight_rows().drop(
                columns=[
                    "selected_feature",
                    "f1_sevenday_vigless_ip_opening_odds",
                ]
            ),
            required_features=[
                "selected_feature",
                "f1_sevenday_vigless_ip_opening_odds",
            ],
        )

    message = str(exc_info.value)
    assert "Missing required non-odds model features: selected_feature" in message
    assert (
        "Missing required odds features: "
        "f1_sevenday_vigless_ip_opening_odds"
    ) in message


def test_data_preparation_preserves_odds_when_selected(tmp_path):
    fights = _fight_rows()
    fights.loc[0, "f1_sevenday_vigless_ip_opening_odds"] = pd.NA
    data_path = tmp_path / "fights.csv"
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    fights.to_csv(data_path, index=False)

    preparation = DataPreparation(
        data_path=str(data_path),
        feats=["selected_feature", "f1_sevenday_vigless_ip_opening_odds"],
        odds=False,
        start_date="2024-01-01",
        num_fights=0,
    )
    features, target = preparation.load_and_clean_data(str(model_dir))

    assert features.index.tolist() == [0, 1]
    assert "f1_sevenday_vigless_ip_opening_odds" in features.columns
    assert target.tolist() == [0, 1]


def test_training_snapshot_retains_selected_feature_complete_rows():
    from libs.modeling.train import DataLoader

    data_path = Path(__file__).parents[2] / "data" / "training_data.csv"
    filtered = DataLoader.load_and_filter_data(
        str(data_path),
        num_fights=2,
        start_date="2014-01-01",
        include_split_dec=True,
        required_features=vSeven_testing2,
    )

    assert len(filtered) == 3267
    assert filtered["event_date"].max() == pd.Timestamp("2026-08-08")
