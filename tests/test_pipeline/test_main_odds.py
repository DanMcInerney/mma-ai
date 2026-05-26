from main import refresh_odds_features


def test_refresh_odds_features_skips_external_work_when_disabled(monkeypatch, capsys):
    class ForbiddenBFOScraper:
        def __init__(self, _conn):
            raise AssertionError("BFO scraper should not be created when odds are disabled")

    class ForbiddenOddsCalculator:
        def __init__(self, _conn):
            raise AssertionError("Odds calculator should not be created when odds are disabled")

    monkeypatch.setattr("main.BFOScraper", ForbiddenBFOScraper)
    monkeypatch.setattr("main.OddsCalculator", ForbiddenOddsCalculator)

    result = refresh_odds_features(object(), enabled=False)

    assert result == {"enabled": False, "refresh_bfo": False, "records_scraped": None, "calculated": False}
    assert "Skipping BFO odds refresh" in capsys.readouterr().out


def test_refresh_odds_features_calculates_from_imported_odds_without_external_scrape(monkeypatch, capsys):
    calls = []

    class ForbiddenBFOScraper:
        def __init__(self, _conn):
            raise AssertionError("BFO scraper should not run for imported odds feature calculation")

    class FakeOddsCalculator:
        def __init__(self, conn):
            calls.append(("odds_init", conn))

        def run(self):
            calls.append(("odds_run", None))

    conn = object()
    monkeypatch.setattr("main.BFOScraper", ForbiddenBFOScraper)
    monkeypatch.setattr("main.OddsCalculator", FakeOddsCalculator)

    result = refresh_odds_features(conn, enabled=True, refresh_bfo=False)

    assert result == {"enabled": True, "refresh_bfo": False, "records_scraped": None, "calculated": True}
    assert calls == [("odds_init", conn), ("odds_run", None)]
    assert "configured odds database" in capsys.readouterr().out


def test_refresh_odds_features_runs_external_work_when_enabled(monkeypatch):
    calls = []

    class FakeBFOScraper:
        def __init__(self, conn):
            calls.append(("bfo_init", conn))

        def scrape_all_fighters(self):
            calls.append(("scrape_all_fighters", None))
            return 7

    class FakeOddsCalculator:
        def __init__(self, conn):
            calls.append(("odds_init", conn))

        def run(self):
            calls.append(("odds_run", None))

    conn = object()
    monkeypatch.setattr("main.BFOScraper", FakeBFOScraper)
    monkeypatch.setattr("main.OddsCalculator", FakeOddsCalculator)

    result = refresh_odds_features(conn, enabled=True)

    assert result == {"enabled": True, "refresh_bfo": True, "records_scraped": 7, "calculated": True}
    assert calls == [
        ("bfo_init", conn),
        ("scrape_all_fighters", None),
        ("odds_init", conn),
        ("odds_run", None),
    ]
