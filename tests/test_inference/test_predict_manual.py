from datetime import datetime
from unittest.mock import MagicMock, patch

from predict import build_manual_odds, get_bfo_odds, get_manual_fight


def test_get_manual_fight_uses_prediction_pipeline_tuple_shape():
    fights, event_names = get_manual_fight("fighter one", "fighter two", "2026-05-25")

    assert fights == [(datetime(2026, 5, 25), "fighter one", "fighter two")]
    assert event_names == ["fighter one_vs_fighter two"]


def test_build_manual_odds_devigs_pair():
    fight_list = [(datetime(2026, 5, 25), "fighter one", "fighter two")]

    with patch("predict.BFOLatestOddsOnly") as mock_bfo_class:
        mock_bfo = MagicMock()
        mock_bfo.remove_vig.return_value = (-110, 110)
        mock_bfo_class.return_value = mock_bfo

        odds = build_manual_odds(fight_list, -120, 100)

    assert odds["fighter one"] == {"original": -120, "vigless": -110}
    assert odds["fighter two"] == {"original": 100, "vigless": 110}


@patch("predict.get_manual_fighter_odds")
@patch("predict.BFOLatestOddsOnly")
def test_get_bfo_odds_noninteractive_does_not_prompt(mock_bfo_class, mock_manual_odds):
    mock_bfo = MagicMock()
    mock_bfo.get_latest_fight_odds_no_db.side_effect = Exception("network down")
    mock_bfo_class.return_value = mock_bfo
    fight_list = [(datetime(2026, 5, 25), "fighter one", "fighter two")]

    odds = get_bfo_odds(fight_list, allow_manual_input=False)

    assert odds == {"fighter one": "N/A", "fighter two": "N/A"}
    mock_manual_odds.assert_not_called()
