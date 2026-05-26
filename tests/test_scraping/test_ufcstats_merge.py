from pathlib import Path

import pandas as pd

from libs.scraping.ufcstats import COMPETITION_FIELDS, FIGHTER_FIELDS, _merge_csv


def write_fighters(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows, columns=FIGHTER_FIELDS).to_csv(path, index=False)


def write_competitions(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows, columns=COMPETITION_FIELDS).to_csv(path, index=False)


def test_incremental_merge_adds_new_rows_without_replacing_existing_csv(tmp_path):
    existing = tmp_path / "individuals.csv"
    incoming = tmp_path / "individuals_new.csv"
    write_fighters(
        existing,
        [
            {
                "name": "Existing Fighter",
                "nickname": "--",
                "url": "http://ufcstats.com/fighter-details/existing",
                "dob": "Jan 1, 1990",
                "weight": "170 lbs.",
                "reach": "72\"",
                "height": "6' 0\"",
                "stance": "Orthodox",
            }
        ],
    )
    write_fighters(
        incoming,
        [
            {
                "name": "New Fighter",
                "nickname": "Fresh",
                "url": "http://ufcstats.com/fighter-details/new",
                "dob": "Feb 2, 1992",
                "weight": "155 lbs.",
                "reach": "70\"",
                "height": "5' 10\"",
                "stance": "Southpaw",
            }
        ],
    )

    total = _merge_csv(existing, incoming, FIGHTER_FIELDS, ["url"], replace=False)

    merged = pd.read_csv(existing)
    assert total == 2
    assert merged["url"].tolist() == [
        "http://ufcstats.com/fighter-details/existing",
        "http://ufcstats.com/fighter-details/new",
    ]


def test_incremental_merge_does_not_rewrite_csv_when_no_new_rows(tmp_path):
    existing = tmp_path / "individuals.csv"
    incoming = tmp_path / "individuals_new.csv"
    write_fighters(existing, [{"name": "Existing Fighter", "url": "existing"}])
    incoming.write_text(",".join(FIGHTER_FIELDS) + "\n", encoding="utf-8")
    original = existing.read_text(encoding="utf-8")

    total = _merge_csv(existing, incoming, FIGHTER_FIELDS, ["url"], replace=False)

    assert total == 1
    assert existing.read_text(encoding="utf-8") == original


def test_incremental_merge_preserves_existing_rows_when_incoming_duplicates(tmp_path):
    existing = tmp_path / "individuals.csv"
    incoming = tmp_path / "individuals_new.csv"
    write_fighters(
        existing,
        [
            {
                "name": "Existing Fighter",
                "nickname": "Original",
                "url": "http://ufcstats.com/fighter-details/existing",
            }
        ],
    )
    write_fighters(
        incoming,
        [
            {
                "name": "Changed Fighter",
                "nickname": "Incoming",
                "url": "http://ufcstats.com/fighter-details/existing",
            },
            {
                "name": "New Fighter",
                "nickname": "Fresh",
                "url": "http://ufcstats.com/fighter-details/new",
            },
        ],
    )

    total = _merge_csv(existing, incoming, FIGHTER_FIELDS, ["url"], replace=False)

    merged = pd.read_csv(existing)
    assert total == 2
    assert merged[["url", "name", "nickname"]].to_dict("records") == [
        {
            "url": "http://ufcstats.com/fighter-details/existing",
            "name": "Existing Fighter",
            "nickname": "Original",
        },
        {
            "url": "http://ufcstats.com/fighter-details/new",
            "name": "New Fighter",
            "nickname": "Fresh",
        },
    ]


def test_incremental_fight_merge_appends_new_fights_without_replacing_existing_rows(tmp_path):
    existing = tmp_path / "competitions.csv"
    incoming = tmp_path / "competitions_new.csv"
    key_columns = ["event_url", "player1_url", "player2_url"]
    write_competitions(
        existing,
        [
            {
                "event_url": "http://ufcstats.com/event-details/event-1",
                "player1": "Existing Winner",
                "player1_url": "http://ufcstats.com/fighter-details/existing-winner",
                "player2": "Existing Loser",
                "player2_url": "http://ufcstats.com/fighter-details/existing-loser",
                "method": "Decision",
            }
        ],
    )
    write_competitions(
        incoming,
        [
            {
                "event_url": "http://ufcstats.com/event-details/event-1",
                "player1": "Changed Winner",
                "player1_url": "http://ufcstats.com/fighter-details/existing-winner",
                "player2": "Existing Loser",
                "player2_url": "http://ufcstats.com/fighter-details/existing-loser",
                "method": "KO/TKO",
            },
            {
                "event_url": "http://ufcstats.com/event-details/event-2",
                "player1": "New Winner",
                "player1_url": "http://ufcstats.com/fighter-details/new-winner",
                "player2": "New Loser",
                "player2_url": "http://ufcstats.com/fighter-details/new-loser",
                "method": "Submission",
            },
        ],
    )

    total = _merge_csv(existing, incoming, COMPETITION_FIELDS, key_columns, replace=False)

    merged = pd.read_csv(existing)
    assert total == 2
    assert merged[["event_url", "player1", "method"]].to_dict("records") == [
        {
            "event_url": "http://ufcstats.com/event-details/event-1",
            "player1": "Existing Winner",
            "method": "Decision",
        },
        {
            "event_url": "http://ufcstats.com/event-details/event-2",
            "player1": "New Winner",
            "method": "Submission",
        },
    ]


def test_merge_drops_rows_with_blank_key_values(tmp_path):
    existing = tmp_path / "individuals.csv"
    incoming = tmp_path / "individuals_new.csv"
    write_fighters(existing, [{"name": "Existing Fighter", "url": "existing"}])
    write_fighters(
        incoming,
        [
            {"name": "Blank URL Fighter", "url": "   "},
            {"name": "New Fighter", "url": "new"},
        ],
    )

    total = _merge_csv(existing, incoming, FIGHTER_FIELDS, ["url"], replace=False)

    merged = pd.read_csv(existing)
    assert total == 2
    assert merged["url"].tolist() == ["existing", "new"]


def test_force_full_merge_replaces_existing_rows(tmp_path):
    existing = tmp_path / "individuals.csv"
    incoming = tmp_path / "individuals_new.csv"
    write_fighters(existing, [{"name": "Existing Fighter", "url": "existing"}])
    write_fighters(incoming, [{"name": "Replacement Fighter", "url": "replacement"}])

    total = _merge_csv(existing, incoming, FIGHTER_FIELDS, ["url"], replace=True)

    merged = pd.read_csv(existing)
    assert total == 1
    assert merged["url"].tolist() == ["replacement"]
