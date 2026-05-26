import pandas as pd
from bs4 import BeautifulSoup

from libs.upcoming_fights import UpcomingFights


def test_scheduled_events_are_sorted_by_wikipedia_date(monkeypatch):
    class FakeWikiTableScraper:
        table = object()

        def __init__(self, url, table_id):
            self.url = url
            self.table_id = table_id

        def get_table_links(self, column):
            return [
                "https://example.test/UFC_Later",
                "https://example.test/UFC_Next",
            ]

        def get_table_by_id(self):
            return pd.DataFrame(
                [
                    {"Event": "UFC Later", "Date": "Jul 1, 2026"},
                    {"Event": "UFC Next", "Date": "Jun 1, 2026"},
                ]
            )

    monkeypatch.setattr("libs.upcoming_fights.WikiTableScraper", FakeWikiTableScraper)

    scheduled_events = UpcomingFights(pd.DataFrame({"fighter_name": []}), 1).get_scheduled_events()

    assert [event["name"] for event in scheduled_events] == ["UFC Next", "UFC Later"]
    assert [event["url"] for event in scheduled_events] == [
        "https://example.test/UFC_Next",
        "https://example.test/UFC_Later",
    ]


def test_scheduled_events_keep_dates_and_links_aligned_when_rows_lack_links(monkeypatch):
    class FakeWikiTableScraper:
        def __init__(self, url, table_id):
            self.url = url
            self.table_id = table_id
            self.base_url = "https://example.test"
            self.table = BeautifulSoup(
                """
                <table>
                  <tr><th>Event</th><th>Date</th></tr>
                  <tr><td>UFC No Article Yet</td><td>Aug 29, 2026</td></tr>
                  <tr><td><a href="/wiki/UFC_Later">UFC Later</a></td><td>Jul 1, 2026</td></tr>
                  <tr><td><a href="/wiki/UFC_Next">UFC Next</a></td><td>Jun 1, 2026</td></tr>
                </table>
                """,
                "html.parser",
            ).find("table")

        def get_table_links(self, column):
            raise AssertionError("row parser should keep links aligned without a separate link list")

        def get_table_by_id(self):
            raise AssertionError("row parser should not need pandas table fallback")

    monkeypatch.setattr("libs.upcoming_fights.WikiTableScraper", FakeWikiTableScraper)

    scheduled_events = UpcomingFights(pd.DataFrame({"fighter_name": []}), 1).get_scheduled_events()

    assert [(event["name"], event["url"]) for event in scheduled_events] == [
        ("UFC Next", "https://example.test/wiki/UFC_Next"),
        ("UFC Later", "https://example.test/wiki/UFC_Later"),
    ]


def test_run_uses_date_sorted_scheduled_event_number(monkeypatch):
    captured = {}

    def fake_get_scheduled_events(self):
        return [
            {"url": "https://example.test/UFC_Next", "name": "UFC Next", "date": pd.Timestamp("2026-06-01")},
            {"url": "https://example.test/UFC_Later", "name": "UFC Later", "date": pd.Timestamp("2026-07-01")},
        ]

    def fake_get_upcoming_cards(self, links):
        captured["links"] = links
        return {"UFC Later": [(pd.Timestamp("2026-07-01"), "fighter one", "fighter two")]}

    monkeypatch.setattr(UpcomingFights, "get_scheduled_events", fake_get_scheduled_events)
    monkeypatch.setattr(UpcomingFights, "get_upcoming_cards", fake_get_upcoming_cards)

    result = UpcomingFights(pd.DataFrame({"fighter_name": ["fighter one", "fighter two"]}), 2).run()

    assert captured["links"] == ["https://example.test/UFC_Later"]
    assert list(result) == ["UFC Later"]
