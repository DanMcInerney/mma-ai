from datetime import datetime
from unittest.mock import MagicMock, patch

from predict import (
    build_manual_odds,
    get_bfo_odds,
    get_manual_fight,
    manual_odds_cover_fight_list,
    resolve_prediction_odds,
)


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


def test_manual_odds_cover_fight_list_is_case_insensitive_and_complete():
    fight_list = [
        (datetime(2026, 5, 25), "Fighter One", "Fighter Two"),
        (datetime(2026, 5, 25), "Fighter Three", "Fighter Four"),
    ]

    assert manual_odds_cover_fight_list(
        fight_list,
        {
            "fighter one": -120,
            "fighter two": 100,
            "fighter three": -150,
            "fighter four": 130,
        },
    )
    assert not manual_odds_cover_fight_list(
        fight_list,
        {
            "fighter one": -120,
            "fighter two": 100,
            "fighter three": -150,
        },
    )


def test_resolve_prediction_odds_skips_bfo_when_manual_event_odds_are_complete(capsys):
    fight_list = [
        (datetime(2026, 5, 25), "fighter one", "fighter two"),
        (datetime(2026, 5, 25), "fighter three", "fighter four"),
    ]
    manual_odds = {
        "fighter one": -120,
        "fighter two": 100,
        "fighter three": -150,
        "fighter four": 130,
    }

    with patch("predict.get_bfo_odds") as mock_get_bfo, patch("predict.BFOLatestOddsOnly") as mock_bfo_class:
        mock_bfo = MagicMock()
        mock_bfo.remove_vig.side_effect = [(-110, 110), (-140, 140)]
        mock_bfo_class.return_value = mock_bfo

        odds = resolve_prediction_odds(
            fight_list,
            odds_enabled=True,
            manual_odds=manual_odds,
            allow_manual_input=False,
        )

    mock_get_bfo.assert_not_called()
    assert odds["fighter one"] == {"original": -120, "vigless": -110}
    assert odds["fighter four"] == {"original": 130, "vigless": 140}
    assert "skipping BFO odds lookup" in capsys.readouterr().out


def test_resolve_prediction_odds_uses_bfo_when_manual_event_odds_are_partial():
    fight_list = [(datetime(2026, 5, 25), "fighter one", "fighter two")]

    with patch("predict.get_bfo_odds") as mock_get_bfo, patch("predict.BFOLatestOddsOnly") as mock_bfo_class:
        mock_get_bfo.return_value = {"fighter one": "N/A", "fighter two": 100}
        mock_bfo = MagicMock()
        mock_bfo.remove_vig.return_value = (-110, 110)
        mock_bfo_class.return_value = mock_bfo

        odds = resolve_prediction_odds(
            fight_list,
            odds_enabled=True,
            manual_odds={"fighter one": -120},
            allow_manual_input=False,
        )

    mock_get_bfo.assert_called_once_with(
        fight_list,
        allow_manual_input=False,
        use_flaresolverr=False,
    )
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
