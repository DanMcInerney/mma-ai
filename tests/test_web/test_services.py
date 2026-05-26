from pathlib import Path
import json
import subprocess
import sys
from types import ModuleType, SimpleNamespace

import pandas as pd
import pytest

from libs.web.models import DataRefreshRequest, EventPredictionRequest, MatchupPredictionRequest, TrainingRequest
from libs.web.services import (
    TRAINING_RESULT_BEGIN,
    TRAINING_RESULT_END,
    _database_ready,
    _data_status_row_deltas,
    _run_logged_subprocess,
    get_data_status,
    get_readiness_status,
    list_fighters,
    list_models,
    list_upcoming_events,
    run_data_refresh,
    run_event_prediction,
    run_matchup_prediction,
    run_training,
    run_training_impl,
    validate_matchup_request,
)


def write_csv(path: Path, rows: list[dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)


def test_data_refresh_defaults_recreate_generated_schemas():
    request = DataRefreshRequest()

    assert request.scrape is True
    assert request.rebuild is True
    assert request.reset_db is True
    assert request.force_full is False


def test_get_data_status_counts_configured_csvs(monkeypatch, tmp_path):
    raw_dir = tmp_path / "raw"
    data_dir = tmp_path / "data"
    monkeypatch.setenv("MMA_AI_UFCSTATS_DIR", str(raw_dir))
    monkeypatch.setenv("MMA_AI_DATA_DIR", str(data_dir))
    monkeypatch.setenv("DATABASE_URL", "postgresql://postgres:secret@localhost:5432/mma-ai")

    write_csv(raw_dir / "competitions.csv", [{"event_url": "e1"}, {"event_url": "e2"}])
    write_csv(raw_dir / "individuals.csv", [{"url": "f1"}])
    write_csv(data_dir / "training_data.csv", [{"fight_id": 1}, {"fight_id": 2}, {"fight_id": 3}])

    status = get_data_status()

    assert status["raw_csvs"]["competitions"]["rows"] == 2
    assert status["raw_csvs"]["individuals"]["rows"] == 1
    assert status["model_csvs"]["training_data"]["rows"] == 3
    assert "secret" not in status["database_url"]


def test_data_status_row_deltas_compare_before_and_after_counts():
    before = {
        "raw_csvs": {
            "competitions": {"rows": 10},
            "individuals": {"rows": 5},
        },
        "model_csvs": {
            "prediction_data": {"rows": 20},
            "training_data": {"rows": 30},
            "training_data_dec": {"rows": None},
        },
    }
    after = {
        "raw_csvs": {
            "competitions": {"rows": 12},
            "individuals": {"rows": 5},
        },
        "model_csvs": {
            "prediction_data": {"rows": 21},
            "training_data": {"rows": 30},
            "training_data_dec": {"rows": 7},
        },
    }

    assert _data_status_row_deltas(before, after) == {
        "competitions": 2,
        "individuals": 0,
        "prediction_data": 1,
        "training_data": 0,
        "training_data_dec": None,
    }


def test_get_readiness_status_requires_seed_data_model_csvs_model_and_databases(monkeypatch, tmp_path):
    raw_dir = tmp_path / "raw"
    data_dir = tmp_path / "data"
    models_dir = tmp_path / "AutogluonModels"
    monkeypatch.setenv("MMA_AI_UFCSTATS_DIR", str(raw_dir))
    monkeypatch.setenv("MMA_AI_DATA_DIR", str(data_dir))
    monkeypatch.setenv("MMA_AI_MODELS_DIR", str(models_dir))
    monkeypatch.setenv("DATABASE_URL", "postgresql://postgres@localhost:5432/mma-ai")
    monkeypatch.setenv("ODDS_DATABASE_URL", "postgresql://postgres@localhost:5432/odds")

    write_csv(raw_dir / "competitions.csv", [{"event_url": "event-1"}])
    write_csv(raw_dir / "individuals.csv", [{"url": "fighter-1"}])
    write_csv(data_dir / "prediction_data.csv", [{"fighter_name": "fighter one"}])
    write_csv(data_dir / "training_data.csv", [{"fighter1_name": "fighter one", "target": 1}])
    write_csv(data_dir / "training_data_dec.csv", [{"fighter1_name": "fighter one", "decision_target": 0}])
    starter_model = models_dir / "ag-20260304_110750-win-extreme"
    starter_model.mkdir(parents=True)
    (starter_model / "feats.txt").write_text("feature\n", encoding="utf-8")
    (starter_model / "predictor.pkl").write_text("starter", encoding="utf-8")
    captured_database_checks = []

    def fake_database_ready(url, required_tables=None):
        captured_database_checks.append((url, required_tables))
        return {"ok": True, "url": url, "required_tables": required_tables}

    monkeypatch.setattr("libs.web.services._database_ready", fake_database_ready)

    readiness = get_readiness_status()

    assert readiness["ready"] is True
    assert readiness["status"] == "ok"
    assert readiness["checks"]["competitions_csv"]["rows"] == 1
    assert readiness["checks"]["prediction_data_csv"]["ok"] is True
    assert readiness["checks"]["training_data_dec_csv"]["ok"] is True
    assert readiness["checks"]["starter_model"]["expected"] == "ag-20260304_110750-win-extreme"
    assert readiness["checks"]["starter_model"]["models"] == ["ag-20260304_110750-win-extreme"]
    assert readiness["checks"]["database"]["ok"] is True
    assert readiness["checks"]["odds_database"]["ok"] is True
    assert captured_database_checks == [
        ("postgresql://postgres@localhost:5432/mma-ai", ["features.fight_mapping"]),
        ("postgresql://postgres@localhost:5432/odds", ["bestfightodds.bfo"]),
    ]


def test_get_readiness_status_reports_missing_prerequisites(monkeypatch, tmp_path):
    monkeypatch.setenv("MMA_AI_UFCSTATS_DIR", str(tmp_path / "raw"))
    monkeypatch.setenv("MMA_AI_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("MMA_AI_MODELS_DIR", str(tmp_path / "AutogluonModels"))
    monkeypatch.setattr("libs.web.services._database_ready", lambda url, required_tables=None: {"ok": False, "url": url, "error": "offline"})

    readiness = get_readiness_status()

    assert readiness["ready"] is False
    assert readiness["status"] == "not_ready"
    assert readiness["checks"]["competitions_csv"]["ok"] is False
    assert readiness["checks"]["prediction_data_csv"]["ok"] is False
    assert readiness["checks"]["training_data_dec_csv"]["ok"] is False
    assert readiness["checks"]["starter_model"]["ok"] is False
    assert readiness["checks"]["database"]["error"] == "offline"


def test_get_readiness_status_requires_configured_starter_model_name(monkeypatch, tmp_path):
    raw_dir = tmp_path / "raw"
    data_dir = tmp_path / "data"
    models_dir = tmp_path / "AutogluonModels"
    monkeypatch.setenv("MMA_AI_UFCSTATS_DIR", str(raw_dir))
    monkeypatch.setenv("MMA_AI_DATA_DIR", str(data_dir))
    monkeypatch.setenv("MMA_AI_MODELS_DIR", str(models_dir))

    write_csv(raw_dir / "competitions.csv", [{"event_url": "event-1"}])
    write_csv(raw_dir / "individuals.csv", [{"url": "fighter-1"}])
    write_csv(data_dir / "prediction_data.csv", [{"fighter_name": "fighter one"}])
    write_csv(data_dir / "training_data.csv", [{"fighter1_name": "fighter one", "target": 1}])
    write_csv(data_dir / "training_data_dec.csv", [{"fighter1_name": "fighter one", "decision_target": 0}])
    other_model = models_dir / "some-other-model"
    other_model.mkdir(parents=True)
    (other_model / "feats.txt").write_text("feature\n", encoding="utf-8")
    (other_model / "predictor.pkl").write_text("starter", encoding="utf-8")
    monkeypatch.setattr("libs.web.services._database_ready", lambda url, required_tables=None: {"ok": True, "url": url})

    readiness = get_readiness_status()

    assert readiness["ready"] is False
    assert readiness["checks"]["starter_model"]["ok"] is False
    assert readiness["checks"]["starter_model"]["expected"] == "ag-20260304_110750-win-extreme"
    assert readiness["checks"]["starter_model"]["models"] == ["some-other-model"]


def test_get_readiness_status_reports_missing_imported_database_tables(monkeypatch, tmp_path):
    raw_dir = tmp_path / "raw"
    data_dir = tmp_path / "data"
    models_dir = tmp_path / "AutogluonModels"
    monkeypatch.setenv("MMA_AI_UFCSTATS_DIR", str(raw_dir))
    monkeypatch.setenv("MMA_AI_DATA_DIR", str(data_dir))
    monkeypatch.setenv("MMA_AI_MODELS_DIR", str(models_dir))

    write_csv(raw_dir / "competitions.csv", [{"event_url": "event-1"}])
    write_csv(raw_dir / "individuals.csv", [{"url": "fighter-1"}])
    write_csv(data_dir / "prediction_data.csv", [{"fighter_name": "fighter one"}])
    write_csv(data_dir / "training_data.csv", [{"fighter1_name": "fighter one", "target": 1}])
    write_csv(data_dir / "training_data_dec.csv", [{"fighter1_name": "fighter one", "decision_target": 0}])
    starter_model = models_dir / "ag-20260304_110750-win-extreme"
    starter_model.mkdir(parents=True)
    (starter_model / "feats.txt").write_text("feature\n", encoding="utf-8")
    (starter_model / "predictor.pkl").write_text("starter", encoding="utf-8")

    def fake_database_ready(url, required_tables=None):
        return {"ok": False, "url": url, "missing_tables": required_tables or []}

    monkeypatch.setattr("libs.web.services._database_ready", fake_database_ready)

    readiness = get_readiness_status()

    assert readiness["ready"] is False
    assert readiness["checks"]["database"]["missing_tables"] == ["features.fight_mapping"]
    assert readiness["checks"]["odds_database"]["missing_tables"] == ["bestfightodds.bfo"]


def test_get_readiness_status_requires_loadable_starter_model(monkeypatch, tmp_path):
    raw_dir = tmp_path / "raw"
    data_dir = tmp_path / "data"
    models_dir = tmp_path / "AutogluonModels"
    monkeypatch.setenv("MMA_AI_UFCSTATS_DIR", str(raw_dir))
    monkeypatch.setenv("MMA_AI_DATA_DIR", str(data_dir))
    monkeypatch.setenv("MMA_AI_MODELS_DIR", str(models_dir))

    write_csv(raw_dir / "competitions.csv", [{"event_url": "event-1"}])
    write_csv(raw_dir / "individuals.csv", [{"url": "fighter-1"}])
    write_csv(data_dir / "prediction_data.csv", [{"fighter_name": "fighter one"}])
    write_csv(data_dir / "training_data.csv", [{"fighter1_name": "fighter one", "target": 1}])
    write_csv(data_dir / "training_data_dec.csv", [{"fighter1_name": "fighter one", "decision_target": 0}])
    starter_model = models_dir / "ag-20260304_110750-win-extreme"
    starter_model.mkdir(parents=True)
    (starter_model / "feats.txt").write_text("feature\n", encoding="utf-8")
    monkeypatch.setattr("libs.web.services._database_ready", lambda url, required_tables=None: {"ok": True, "url": url})

    readiness = get_readiness_status()

    assert readiness["ready"] is False
    assert readiness["checks"]["starter_model"]["ok"] is False
    assert readiness["checks"]["starter_model"]["models"] == []
    assert readiness["checks"]["starter_model"]["path"] is None


def test_database_ready_requires_postgres_tables(monkeypatch):
    captured_queries = []

    class FakeResult:
        def __init__(self, value):
            self.value = value

        def scalar(self):
            return self.value

    class FakeConnection:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def execute(self, statement, params=None):
            captured_queries.append((str(statement), params))
            if params == {"table_name": "features.fight_mapping"}:
                return FakeResult(False)
            return FakeResult(True)

    class FakeEngine:
        def connect(self):
            return FakeConnection()

        def dispose(self):
            pass

    monkeypatch.setattr("sqlalchemy.create_engine", lambda *_args, **_kwargs: FakeEngine())

    result = _database_ready(
        "postgresql://postgres:secret@localhost:5432/mma-ai",
        required_tables=["features.fight_mapping"],
    )

    assert result["ok"] is False
    assert result["url"] == "postgresql://postgres:***@localhost:5432/mma-ai"
    assert result["missing_tables"] == ["features.fight_mapping"]
    assert any("to_regclass" in query for query, _params in captured_queries)


def test_run_logged_subprocess_streams_output_and_returns_completed_process(capsys):
    command = [
        sys.executable,
        "-c",
        "import sys; print('hello stdout'); print('hello stderr', file=sys.stderr)",
    ]

    completed = _run_logged_subprocess(command, "unit-test")

    assert completed.returncode == 0
    assert completed.stdout == "hello stdout\n"
    assert completed.stderr == "hello stderr\n"
    captured = capsys.readouterr()
    assert "[unit-test] command:" in captured.out
    assert "[unit-test] stdout begin" in captured.out
    assert "hello stdout" in captured.out
    assert "[unit-test] exit_code=0" in captured.out
    assert "[unit-test] stderr begin" in captured.err
    assert "hello stderr" in captured.err


@pytest.mark.parametrize("odds_enabled", [False, True])
def test_run_data_refresh_runs_rebuild_as_subprocess(monkeypatch, tmp_path, odds_enabled):
    raw_dir = tmp_path / "raw"
    data_dir = tmp_path / "data"
    monkeypatch.setenv("MMA_AI_UFCSTATS_DIR", str(raw_dir))
    monkeypatch.setenv("MMA_AI_DATA_DIR", str(data_dir))
    monkeypatch.setenv("DATABASE_URL", "postgresql://postgres:postgres@localhost:5432/mma-ai")
    captured = {}

    def fake_run(command, log_prefix, **kwargs):
        captured["command"] = command
        captured["log_prefix"] = log_prefix
        captured["kwargs"] = kwargs
        return subprocess.CompletedProcess(command, 0, stdout="Finished rebuild\n", stderr="")

    monkeypatch.setattr("libs.web.services._run_logged_subprocess", fake_run)

    result = run_data_refresh(
        DataRefreshRequest(
            scrape=False,
            rebuild=True,
            reset_db=True,
            odds=odds_enabled,
        )
    )

    assert result["scrape_counts"] == {}
    assert "before_status" in result
    assert "row_deltas" in result
    assert result["row_deltas"]["training_data"] is None
    command = captured["command"]
    assert command[0] == sys.executable
    assert command[1].endswith("main.py")
    assert command[command.index("--raw-data-dir") + 1] == str(raw_dir)
    assert command[command.index("--output-data-dir") + 1] == str(data_dir)
    assert "--reset-db" in command
    assert ("--odds" in command) is odds_enabled
    assert "--db-url" not in command
    assert captured["log_prefix"] == "data-refresh"
    assert captured["kwargs"]["stdout_label"] == "rebuild stdout"
    assert captured["kwargs"]["stderr_label"] == "rebuild stderr"


def test_run_data_refresh_runs_scraper_as_subprocess(monkeypatch, tmp_path, capsys):
    raw_dir = tmp_path / "raw"
    data_dir = tmp_path / "data"
    monkeypatch.setenv("MMA_AI_UFCSTATS_DIR", str(raw_dir))
    monkeypatch.setenv("MMA_AI_DATA_DIR", str(data_dir))
    captured = {"commands": []}

    def fake_run(command, log_prefix, **kwargs):
        captured["commands"].append((command, log_prefix, kwargs))
        script = Path(command[1]).name
        if script == "main.py":
            return subprocess.CompletedProcess(command, 0, stdout="Finished rebuild\n", stderr="")
        return subprocess.CompletedProcess(
            command,
            0,
            stdout="Scrapy log line\nfighters: 42 total rows\nfights: 99 total rows\n",
            stderr="crawler debug stderr\n",
        )

    monkeypatch.setattr("libs.web.services._run_logged_subprocess", fake_run)

    result = run_data_refresh(
        DataRefreshRequest(
            scrape=True,
            rebuild=True,
            force_full=True,
            reset_db=True,
            log_level="DEBUG",
        )
    )

    scrape_command, scrape_prefix, scrape_kwargs = captured["commands"][0]
    rebuild_command, rebuild_prefix, rebuild_kwargs = captured["commands"][1]
    assert scrape_command[0] == sys.executable
    assert scrape_command[1].endswith("scripts\\scrape_ufcstats.py") or scrape_command[1].endswith("scripts/scrape_ufcstats.py")
    assert scrape_command[scrape_command.index("--output-dir") + 1] == str(raw_dir)
    assert scrape_command[scrape_command.index("--log-level") + 1] == "DEBUG"
    assert "--force-full" in scrape_command
    assert scrape_prefix == "data-refresh"
    assert scrape_kwargs["stdout_label"] == "scraper stdout"
    assert scrape_kwargs["stderr_label"] == "scraper stderr"
    assert rebuild_command[1].endswith("main.py")
    assert rebuild_command[rebuild_command.index("--raw-data-dir") + 1] == str(raw_dir)
    assert rebuild_command[rebuild_command.index("--output-data-dir") + 1] == str(data_dir)
    assert "--reset-db" in rebuild_command
    assert rebuild_prefix == "data-refresh"
    assert rebuild_kwargs["stdout_label"] == "rebuild stdout"
    assert rebuild_kwargs["stderr_label"] == "rebuild stderr"
    assert result["scrape_counts"] == {"fighters": 42, "fights": 99}
    output = capsys.readouterr().out
    assert "scraper subprocess finished" in output
    assert "feature-store rebuild subprocess finished" in output


def test_run_data_refresh_surfaces_scraper_failure(monkeypatch, tmp_path):
    monkeypatch.setenv("MMA_AI_UFCSTATS_DIR", str(tmp_path / "raw"))

    def fake_run(command, _log_prefix, **_kwargs):
        return subprocess.CompletedProcess(command, 1, stdout="partial stdout", stderr="reactor error")

    monkeypatch.setattr("libs.web.services._run_logged_subprocess", fake_run)

    with pytest.raises(RuntimeError, match="reactor error"):
        run_data_refresh(DataRefreshRequest(scrape=True, rebuild=False))


def test_run_data_refresh_surfaces_rebuild_failure(monkeypatch, tmp_path):
    monkeypatch.setenv("MMA_AI_UFCSTATS_DIR", str(tmp_path / "raw"))
    monkeypatch.setenv("MMA_AI_DATA_DIR", str(tmp_path / "data"))

    def fake_run(command, _log_prefix, **_kwargs):
        return subprocess.CompletedProcess(command, 2, stdout="partial rebuild stdout", stderr="schema error")

    monkeypatch.setattr("libs.web.services._run_logged_subprocess", fake_run)

    with pytest.raises(RuntimeError, match="schema error"):
        run_data_refresh(DataRefreshRequest(scrape=False, rebuild=True))


def test_list_fighters_supports_prediction_data_shapes(monkeypatch, tmp_path):
    monkeypatch.setenv("MMA_AI_DATA_DIR", str(tmp_path))
    prediction_csv = tmp_path / "prediction_data.csv"
    write_csv(
        prediction_csv,
        [
            {"fighter1_name": "alex", "fighter2_name": "bo"},
            {"fighter1_name": "casey", "fighter2_name": "alex"},
        ],
    )

    assert list_fighters(str(prediction_csv)) == ["alex", "bo", "casey"]


def test_list_models_discovers_huggingface_starter_model_shape(monkeypatch, tmp_path):
    models_dir = tmp_path / "AutogluonModels"
    starter_model = models_dir / "ag-20260304_110750-win-extreme"
    starter_model.mkdir(parents=True)
    for filename in ("predictor.pkl", "learner.pkl", "metadata.json", "feats.txt", "scaler.pkl"):
        (starter_model / filename).write_text("starter", encoding="utf-8")
    monkeypatch.setenv("MMA_AI_MODELS_DIR", str(models_dir))

    models = list_models()

    assert [model["name"] for model in models] == ["ag-20260304_110750-win-extreme"]
    assert models[0]["path"] == str(starter_model)
    assert models[0]["has_features"] is True
    assert models[0]["has_predictor"] is True
    assert models[0]["is_ensemble"] is False
    assert models[0]["has_scaler"] is True


def test_list_models_discovers_ensemble_model_shape(monkeypatch, tmp_path):
    models_dir = tmp_path / "AutogluonModels"
    ensemble_model = models_dir / "ag-20260304_110750-win-ensemble"
    (ensemble_model / "final_model").mkdir(parents=True)
    (ensemble_model / "feats.txt").write_text("feature\n", encoding="utf-8")
    (ensemble_model / "ensemble_info.txt").write_text("ensemble", encoding="utf-8")
    monkeypatch.setenv("MMA_AI_MODELS_DIR", str(models_dir))

    models = list_models()

    assert [model["name"] for model in models] == ["ag-20260304_110750-win-ensemble"]
    assert models[0]["has_predictor"] is False
    assert models[0]["is_ensemble"] is True


def test_list_models_ignores_incomplete_model_directories(monkeypatch, tmp_path):
    models_dir = tmp_path / "AutogluonModels"
    features_only = models_dir / "ag-20260304_110750-win-features-only"
    empty_ensemble = models_dir / "ag-20260304_110750-win-empty-ensemble"
    valid_model = models_dir / "ag-20260304_110750-win-valid"
    for model_dir in (features_only, empty_ensemble, valid_model):
        model_dir.mkdir(parents=True)
        (model_dir / "feats.txt").write_text("feature\n", encoding="utf-8")
    (empty_ensemble / "ensemble_info.txt").write_text("ensemble", encoding="utf-8")
    (valid_model / "predictor.pkl").write_text("predictor", encoding="utf-8")
    monkeypatch.setenv("MMA_AI_MODELS_DIR", str(models_dir))

    models = list_models("win")

    assert [model["name"] for model in models] == ["ag-20260304_110750-win-valid"]


def test_list_models_can_filter_by_prediction_target(monkeypatch, tmp_path):
    models_dir = tmp_path / "AutogluonModels"
    for model_name in ("ag-20260304_110750-win-extreme", "ag-20260304_110750-decision-best"):
        model_dir = models_dir / model_name
        model_dir.mkdir(parents=True)
        (model_dir / "feats.txt").write_text("feature\n", encoding="utf-8")
        (model_dir / "predictor.pkl").write_text("predictor", encoding="utf-8")
    monkeypatch.setenv("MMA_AI_MODELS_DIR", str(models_dir))

    assert [model["name"] for model in list_models("win")] == ["ag-20260304_110750-win-extreme"]
    assert [model["name"] for model in list_models("decision")] == ["ag-20260304_110750-decision-best"]
    assert {model["name"] for model in list_models()} == {
        "ag-20260304_110750-win-extreme",
        "ag-20260304_110750-decision-best",
    }


def test_list_upcoming_events_uses_wikipedia_scraper_adapter(monkeypatch, tmp_path):
    monkeypatch.setenv("MMA_AI_DATA_DIR", str(tmp_path))
    prediction_csv = tmp_path / "prediction_data.csv"
    write_csv(prediction_csv, [{"fighter_name": "fighter one"}, {"fighter_name": "fighter two"}])

    class FakeUpcomingFights:
        def __init__(self, df, upcoming_number):
            self.upcoming_number = upcoming_number

        def get_upcoming_event_links(self):
            return ["https://example.test/ufc-test-2", "https://example.test/ufc-test-1"]

        def get_upcoming_cards(self, links):
            event_number = links[0].rsplit("-", 1)[1]
            return {
                f"UFC Test {event_number}": [
                    (pd.Timestamp(f"2026-06-0{event_number}"), "fighter one", "fighter two"),
                ]
            }

    monkeypatch.setattr("libs.upcoming_fights.UpcomingFights", FakeUpcomingFights)

    result = list_upcoming_events(str(prediction_csv), limit=2)

    assert result["warning"] is None
    assert [event["upcoming_number"] for event in result["events"]] == [1, 2]
    assert [event["name"] for event in result["events"]] == ["UFC Test 1", "UFC Test 2"]
    assert result["events"][0]["fights"][0]["fighter1"] == "fighter one"


def test_list_upcoming_events_preserves_prediction_cli_numbers_after_date_sort(monkeypatch, tmp_path):
    monkeypatch.setenv("MMA_AI_DATA_DIR", str(tmp_path))
    prediction_csv = tmp_path / "prediction_data.csv"
    write_csv(prediction_csv, [{"fighter_name": "fighter one"}, {"fighter_name": "fighter two"}])

    class FakeUpcomingFights:
        def __init__(self, df, upcoming_number):
            self.upcoming_number = upcoming_number

        def get_upcoming_event_links(self):
            return [
                "https://example.test/ufc-test-3",
                "https://example.test/ufc-test-2",
                "https://example.test/ufc-test-1",
            ]

        def get_upcoming_cards(self, links):
            event_number = links[0].rsplit("-", 1)[1]
            dates = {"1": "2026-06-10", "2": "2026-06-01", "3": "2026-06-20"}
            return {
                f"UFC Test {event_number}": [
                    (pd.Timestamp(dates[event_number]), "fighter one", "fighter two"),
                ]
            }

    monkeypatch.setattr("libs.upcoming_fights.UpcomingFights", FakeUpcomingFights)

    result = list_upcoming_events(str(prediction_csv), limit=3)

    assert [event["name"] for event in result["events"]] == ["UFC Test 2", "UFC Test 1", "UFC Test 3"]
    assert [event["upcoming_number"] for event in result["events"]] == [2, 1, 3]


def test_list_upcoming_events_keeps_unmatched_event_names_visible(monkeypatch, tmp_path):
    monkeypatch.setenv("MMA_AI_DATA_DIR", str(tmp_path))
    prediction_csv = tmp_path / "prediction_data.csv"
    write_csv(prediction_csv, [{"fighter_name": "fighter one"}, {"fighter_name": "fighter two"}])

    class FakeUpcomingFights:
        def __init__(self, df, upcoming_number):
            self.upcoming_number = upcoming_number

        def get_upcoming_event_links(self):
            return ["https://example.test/UFC_Empty_Event"]

        def get_upcoming_cards(self, links):
            return {}

    monkeypatch.setattr("libs.upcoming_fights.UpcomingFights", FakeUpcomingFights)

    result = list_upcoming_events(str(prediction_csv), limit=1)

    assert result["events"][0]["name"] == "UFC Empty Event"
    assert result["events"][0]["fights"] == []
    assert result["events"][0]["upcoming_number"] == 1
    assert "no matched fights found" in result["warning"]


def test_list_upcoming_events_reports_missing_prediction_csv(monkeypatch, tmp_path):
    monkeypatch.setenv("MMA_AI_DATA_DIR", str(tmp_path))
    result = list_upcoming_events(str(tmp_path / "missing.csv"))

    assert result["events"] == []
    assert "Prediction data CSV not found" in result["warning"]


def test_validate_matchup_request_rejects_unknown_fighter(monkeypatch, tmp_path):
    monkeypatch.setenv("MMA_AI_DATA_DIR", str(tmp_path))
    prediction_csv = tmp_path / "prediction_data.csv"
    write_csv(prediction_csv, [{"fighter_name": "known fighter"}])

    request = MatchupPredictionRequest(
        prediction_data_csv=str(prediction_csv),
        fighter1="known fighter",
        fighter2="missing fighter",
    )

    with pytest.raises(ValueError, match="missing fighter"):
        validate_matchup_request(request)


def test_validate_matchup_request_rejects_blank_fighters(monkeypatch, tmp_path):
    monkeypatch.setenv("MMA_AI_DATA_DIR", str(tmp_path))
    prediction_csv = tmp_path / "prediction_data.csv"
    write_csv(prediction_csv, [{"fighter_name": "known fighter"}])

    request = MatchupPredictionRequest(
        prediction_data_csv=str(prediction_csv),
        fighter1=" ",
        fighter2="known fighter",
    )

    with pytest.raises(ValueError, match="Enter both fighter names"):
        validate_matchup_request(request)


def test_validate_matchup_request_accepts_known_fighters(monkeypatch, tmp_path):
    monkeypatch.setenv("MMA_AI_DATA_DIR", str(tmp_path))
    prediction_csv = tmp_path / "prediction_data.csv"
    write_csv(prediction_csv, [{"fighter_name": "fighter one"}, {"fighter_name": "fighter two"}])

    request = MatchupPredictionRequest(
        prediction_data_csv=str(prediction_csv),
        fighter1="fighter one",
        fighter2="fighter two",
    )

    assert validate_matchup_request(request)["status"] == "ready_for_prediction"


def test_validate_matchup_request_trims_known_fighters(monkeypatch, tmp_path):
    monkeypatch.setenv("MMA_AI_DATA_DIR", str(tmp_path))
    prediction_csv = tmp_path / "prediction_data.csv"
    write_csv(prediction_csv, [{"fighter_name": "fighter one"}, {"fighter_name": "fighter two"}])

    request = MatchupPredictionRequest(
        prediction_data_csv=str(prediction_csv),
        fighter1=" fighter one ",
        fighter2="\tfighter two\n",
        fight_date=" 2026-06-01 ",
    )

    result = validate_matchup_request(request)

    assert result["fighter1"] == "fighter one"
    assert result["fighter2"] == "fighter two"
    assert result["fight_date"] == "2026-06-01"


def test_validate_matchup_request_rejects_invalid_fight_date(monkeypatch, tmp_path):
    monkeypatch.setenv("MMA_AI_DATA_DIR", str(tmp_path))
    prediction_csv = tmp_path / "prediction_data.csv"
    write_csv(prediction_csv, [{"fighter_name": "fighter one"}, {"fighter_name": "fighter two"}])

    request = MatchupPredictionRequest(
        prediction_data_csv=str(prediction_csv),
        fighter1="fighter one",
        fighter2="fighter two",
        fight_date="06/01/2026",
    )

    with pytest.raises(ValueError, match="YYYY-MM-DD"):
        validate_matchup_request(request)


def test_run_matchup_prediction_uses_predict_cli_without_interactive_odds(monkeypatch, tmp_path):
    monkeypatch.setenv("MMA_AI_DATA_DIR", str(tmp_path))
    prediction_csv = tmp_path / "prediction_data.csv"
    write_csv(prediction_csv, [{"fighter_name": "fighter one"}, {"fighter_name": "fighter two"}])
    captured = {}

    def fake_run(command, log_prefix, **kwargs):
        captured["log_prefix"] = log_prefix
        captured["command"] = command
        output_dir = Path(command[command.index("--output-dir") + 1])
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "fight_predictions.csv").write_text(
            "# Fight Predictions using ORIGINAL model predictions\n"
            "Fighter1,Fighter2,Fighter1_Odds,Fighter2_Odds,Fighter1_AI_Prob,Fighter2_AI_Prob,"
            "Fighter1_Market_Prob,Fighter2_Market_Prob,AI_Pick,Confidence,AI_Odds,EV\n"
            "fighter one,fighter two,-120,100,55.0,45.0,52.0,48.0,fighter one,55.0,-122,1\n",
            encoding="utf-8",
        )
        return subprocess.CompletedProcess(command, 0, stdout="ok", stderr="")

    monkeypatch.setattr("libs.web.services._run_logged_subprocess", fake_run)
    request = MatchupPredictionRequest(
        prediction_data_csv=str(prediction_csv),
        output_dir="predictions/manual-custom",
        fighter1="fighter one",
        fighter2="fighter two",
        fight_date="2026-06-01",
        odds_fighter1=-120,
        odds_fighter2=100,
    )

    result = run_matchup_prediction(request)

    assert "--fighter1" in captured["command"]
    assert "--fighter2" in captured["command"]
    assert captured["command"][captured["command"].index("--output-dir") + 1] == str(tmp_path / "predictions" / "manual-custom")
    assert "--fighter1-odds" in captured["command"]
    assert "--fighter2-odds" in captured["command"]
    assert captured["command"][captured["command"].index("--fight-date") + 1] == "2026-06-01"
    assert "--no-manual-odds" not in captured["command"]
    assert captured["log_prefix"] == "prediction"
    assert result["output_dir"] == str(tmp_path / "predictions" / "manual-custom")
    assert result["predictions"][0]["AI_Pick"] == "fighter one"


def test_run_matchup_prediction_can_fetch_odds_noninteractively(monkeypatch, tmp_path):
    monkeypatch.setenv("MMA_AI_DATA_DIR", str(tmp_path))
    prediction_csv = tmp_path / "prediction_data.csv"
    write_csv(prediction_csv, [{"fighter_name": "fighter one"}, {"fighter_name": "fighter two"}])
    captured = {}

    def fake_run(command, log_prefix, **kwargs):
        captured["log_prefix"] = log_prefix
        captured["command"] = command
        output_dir = Path(command[command.index("--output-dir") + 1])
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "fight_predictions.csv").write_text(
            "# Fight Predictions using ORIGINAL model predictions\n"
            "Fighter1,Fighter2,Fighter1_Odds,Fighter2_Odds,Fighter1_AI_Prob,Fighter2_AI_Prob,"
            "Fighter1_Market_Prob,Fighter2_Market_Prob,AI_Pick,Confidence,AI_Odds,EV\n"
            "fighter one,fighter two,-120,100,55.0,45.0,52.0,48.0,fighter one,55.0,-122,1\n",
            encoding="utf-8",
        )
        return subprocess.CompletedProcess(command, 0, stdout="ok", stderr="")

    monkeypatch.setattr("libs.web.services._run_logged_subprocess", fake_run)
    request = MatchupPredictionRequest(
        prediction_data_csv=str(prediction_csv),
        fighter1="fighter one",
        fighter2="fighter two",
        odds=True,
    )

    result = run_matchup_prediction(request)

    assert "--fighter1-odds" not in captured["command"]
    assert "--fighter2-odds" not in captured["command"]
    assert "--odds" in captured["command"]
    assert "--no-manual-odds" in captured["command"]
    assert captured["log_prefix"] == "prediction"
    assert result["predictions"][0]["EV"] == "1"


def test_run_event_prediction_respects_prediction_knobs(monkeypatch, tmp_path):
    monkeypatch.setenv("MMA_AI_DATA_DIR", str(tmp_path))
    prediction_csv = tmp_path / "prediction_data.csv"
    write_csv(prediction_csv, [{"fighter_name": "fighter one"}, {"fighter_name": "fighter two"}])
    captured = {}

    def fake_run(command, log_prefix, **kwargs):
        captured["log_prefix"] = log_prefix
        captured["command"] = command
        output_dir = Path(command[command.index("--output-dir") + 1])
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "fight_predictions.csv").write_text(
            "# Fight Predictions using CALIBRATED model predictions\n"
            "Fighter1,Fighter2,Fighter1_Odds,Fighter2_Odds,Fighter1_AI_Prob,Fighter2_AI_Prob,"
            "Fighter1_Market_Prob,Fighter2_Market_Prob,AI_Pick,Confidence,AI_Odds,EV\n"
            "fighter one,fighter two,-120,100,55.0,45.0,52.0,48.0,fighter one,55.0,-122,1\n",
            encoding="utf-8",
        )
        return subprocess.CompletedProcess(command, 0, stdout="ok", stderr="")

    monkeypatch.setattr("libs.web.services._run_logged_subprocess", fake_run)
    request = EventPredictionRequest(
        model_type="decision",
        prediction_data_csv=str(prediction_csv),
        output_dir="predictions/event-custom",
        upcoming_number=3,
        odds=True,
        use_calibrated=True,
        shap=True,
    )

    result = run_event_prediction(request)

    assert captured["command"][captured["command"].index("--model-type") + 1] == "decision"
    assert captured["command"][captured["command"].index("--output-dir") + 1] == str(tmp_path / "predictions" / "event-custom")
    assert captured["command"][captured["command"].index("--upcoming-number") + 1] == "3"
    assert "--prediction-data-csv" in captured["command"]
    assert "--odds" in captured["command"]
    assert "--no-manual-odds" in captured["command"]
    assert "--use-calibrated" in captured["command"]
    assert "--no-shap" not in captured["command"]
    assert captured["log_prefix"] == "prediction"
    assert result["output_dir"] == str(tmp_path / "predictions" / "event-custom")
    assert result["predictions"][0]["EV"] == "1"


def test_run_event_prediction_passes_manual_odds_json(monkeypatch, tmp_path):
    monkeypatch.setenv("MMA_AI_DATA_DIR", str(tmp_path))
    prediction_csv = tmp_path / "prediction_data.csv"
    write_csv(prediction_csv, [{"fighter_name": "fighter one"}, {"fighter_name": "fighter two"}])
    captured = {}

    def fake_run(command, log_prefix, **kwargs):
        captured["log_prefix"] = log_prefix
        captured["command"] = command
        output_dir = Path(command[command.index("--output-dir") + 1])
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "fight_predictions.csv").write_text(
            "# Fight Predictions using ORIGINAL model predictions\n"
            "Fighter1,Fighter2,Fighter1_Odds,Fighter2_Odds,Fighter1_AI_Prob,Fighter2_AI_Prob,"
            "Fighter1_Market_Prob,Fighter2_Market_Prob,AI_Pick,Confidence,AI_Odds,EV\n"
            "fighter one,fighter two,-120,100,55.0,45.0,52.0,48.0,fighter one,55.0,-122,1\n",
            encoding="utf-8",
        )
        return subprocess.CompletedProcess(command, 0, stdout="manual odds ok", stderr="debug stderr")

    monkeypatch.setattr("libs.web.services._run_logged_subprocess", fake_run)
    request = EventPredictionRequest(
        prediction_data_csv=str(prediction_csv),
        upcoming_number=1,
        odds=True,
        manual_odds={"fighter one": -120, "fighter two": 100},
    )

    result = run_event_prediction(request)

    manual_json = captured["command"][captured["command"].index("--manual-odds-json") + 1]
    assert json.loads(manual_json) == {"fighter one": -120, "fighter two": 100}
    assert "--no-manual-odds" in captured["command"]
    assert captured["log_prefix"] == "prediction"
    assert result["stdout_tail"] == "manual odds ok"
    assert result["stderr_tail"] == "debug stderr"


def test_run_training_runs_dashboard_training_script_as_subprocess(monkeypatch, tmp_path):
    captured = {}

    def fake_run(command, log_prefix, **kwargs):
        captured["command"] = command
        captured["log_prefix"] = log_prefix
        captured["kwargs"] = kwargs
        payload = json.loads(kwargs["input"])
        result = {
            "model_path": str(tmp_path / "AutogluonModels" / "dashboard-run"),
            "used_script_defaults": True,
            "evaluation": {"available": True, "model_path": str(tmp_path / "AutogluonModels" / "dashboard-run")},
        }
        assert payload["model_type"] == "decision"
        assert payload["time_limit"] == 1200
        return subprocess.CompletedProcess(
            command,
            0,
            stdout=f"training logs\n{TRAINING_RESULT_BEGIN}\n{json.dumps(result)}\n{TRAINING_RESULT_END}\n",
            stderr="training warnings\n",
        )

    monkeypatch.setattr("libs.web.services._run_logged_subprocess", fake_run)

    result = run_training(TrainingRequest(model_type="decision", time_limit=1200))

    command = captured["command"]
    assert command[0] == sys.executable
    assert command[1].endswith("scripts\\train_dashboard.py") or command[1].endswith("scripts/train_dashboard.py")
    assert captured["log_prefix"] == "training"
    assert captured["kwargs"]["input"].endswith("\n")
    assert result["used_script_defaults"] is True
    assert result["evaluation"]["available"] is True


def test_run_training_surfaces_dashboard_script_failure(monkeypatch):
    def fake_run(command, log_prefix, **_kwargs):
        return subprocess.CompletedProcess(command, 7, stdout="partial training output", stderr="autogluon exploded")

    monkeypatch.setattr("libs.web.services._run_logged_subprocess", fake_run)

    with pytest.raises(RuntimeError, match="autogluon exploded"):
        run_training(TrainingRequest())


def test_run_training_requires_structured_dashboard_script_result(monkeypatch):
    def fake_run(command, log_prefix, **_kwargs):
        return subprocess.CompletedProcess(command, 0, stdout="finished without markers", stderr="")

    monkeypatch.setattr("libs.web.services._run_logged_subprocess", fake_run)

    with pytest.raises(RuntimeError, match="structured dashboard result"):
        run_training(TrainingRequest())


def test_run_training_uses_script_defaults_when_advanced_knobs_match(monkeypatch, tmp_path):
    captured = {}

    def fake_main(**kwargs):
        captured["kwargs"] = kwargs
        return SimpleNamespace(path=str(tmp_path / "AutogluonModels" / "script-default"))

    fake_train_module = ModuleType("libs.modeling.train")
    fake_train_module.main = fake_main
    monkeypatch.setitem(sys.modules, "libs.modeling.train", fake_train_module)
    monkeypatch.setattr(
        "libs.web.services.summarize_model_evaluation",
        lambda model_path: {"available": True, "model_path": model_path},
    )

    result = run_training_impl(
        TrainingRequest(
            model_type="decision",
            preset="best",
            time_limit=1200,
            split_strategy="walkforward",
            refit_full=False,
            use_script_defaults=True,
        )
    )

    assert result["used_script_defaults"] is True
    assert captured["kwargs"] == {
        "model_type": "decision",
        "time_limit": 1200,
        "preset": "best",
        "split_strategy": "walkforward",
        "refit_full": False,
    }


def test_run_training_passes_custom_knobs_to_training_config(monkeypatch, tmp_path):
    captured = {}

    class FakeTrainingConfig:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

    class FakeModelTrainer:
        def __init__(self, config):
            captured["config"] = config

        def train(self):
            return SimpleNamespace(path=str(tmp_path / "AutogluonModels" / "decision"))

    fake_train_module = SimpleNamespace(
        vSeven_testing2=["win_feature"],
        DECISION_TEST_FEATS4=["decision_feature"],
        TrainingConfig=FakeTrainingConfig,
        ModelTrainer=FakeModelTrainer,
    )
    import libs.modeling as modeling_package

    monkeypatch.setitem(sys.modules, "libs.modeling.train", fake_train_module)
    monkeypatch.setattr(modeling_package, "train", fake_train_module, raising=False)
    monkeypatch.setattr(
        "libs.web.services.summarize_model_evaluation",
        lambda model_path: {"available": True, "model_path": model_path},
    )

    request = TrainingRequest(
        model_type="decision",
        preset="best",
        time_limit=900,
        split_strategy="walkforward",
        walkforward_n_windows=6,
        walkforward_initial_year=2020,
        refit_full=False,
        refit_all=True,
        use_script_defaults=False,
        test_size="2025-01-01",
        val_date="2024-01-01",
        start_date="2016-01-01",
        num_fights=4,
        include_split_dec=False,
        normalize="zscore",
        use_recency_weights=False,
        decay_rate=0.3,
        calculate_importance=False,
        feature_list=["custom_feature_diff", "market_prob_diff"],
        included_strings=["diff"],
        excluded_strings=["leaky"],
        required_strings=["market_prob_diff"],
        included_model_types=["GBM", "CAT"],
    )

    result = run_training_impl(request)

    config = captured["config"]
    assert result["used_script_defaults"] is False
    assert result["evaluation"]["available"] is True
    assert config.model_type == "decision"
    assert config.preset == "best"
    assert config.time_limit == 900
    assert config.split_strategy == "walkforward"
    assert config.walkforward_n_windows == 6
    assert config.walkforward_initial_year == 2020
    assert config.refit_full is False
    assert config.refit_all is True
    assert config.test_size == "2025-01-01"
    assert config.val_date == "2024-01-01"
    assert config.features == ["custom_feature_diff", "market_prob_diff"]
    assert config.included_strings == ["diff"]
    assert config.excluded_strings == ["leaky"]
    assert config.required_strings == ["market_prob_diff"]
    assert config.start_date == "2016-01-01"
    assert config.num_fights == 4
    assert config.include_split_dec is False
    assert config.normalize == "zscore"
    assert config.use_recency_weights is False
    assert config.decay_rate == 0.3
    assert config.calculate_importance is False
    assert config.included_model_types == ["GBM", "CAT"]


def test_run_training_uses_custom_config_when_script_defaults_are_overridden(monkeypatch, tmp_path):
    captured = {}

    def fake_main(**_kwargs):
        raise AssertionError("train.main should not run when advanced knobs changed")

    class FakeTrainingConfig:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

    class FakeModelTrainer:
        def __init__(self, config):
            captured["config"] = config

        def train(self):
            return SimpleNamespace(path=str(tmp_path / "AutogluonModels" / "custom-from-script-request"))

    fake_train_module = ModuleType("libs.modeling.train")
    fake_train_module.main = fake_main
    fake_train_module.vSeven_testing2 = ["win_feature"]
    fake_train_module.DECISION_TEST_FEATS4 = ["decision_feature"]
    fake_train_module.TrainingConfig = FakeTrainingConfig
    fake_train_module.ModelTrainer = FakeModelTrainer
    import libs.modeling as modeling_package

    monkeypatch.setitem(sys.modules, "libs.modeling.train", fake_train_module)
    monkeypatch.setattr(modeling_package, "train", fake_train_module, raising=False)
    monkeypatch.setattr(
        "libs.web.services.summarize_model_evaluation",
        lambda model_path: {"available": True, "model_path": model_path},
    )

    result = run_training_impl(
        TrainingRequest(
            use_script_defaults=True,
            start_date="2016-01-01",
            walkforward_n_windows=8,
        )
    )

    assert result["used_script_defaults"] is False
    assert captured["config"].start_date == "2016-01-01"
    assert captured["config"].walkforward_n_windows == 8


def test_run_training_logs_evaluation_summary_failures(monkeypatch, tmp_path, capsys):
    def fake_main(**_kwargs):
        return SimpleNamespace(path=str(tmp_path / "AutogluonModels" / "script-default"))

    fake_train_module = ModuleType("libs.modeling.train")
    fake_train_module.main = fake_main
    monkeypatch.setitem(sys.modules, "libs.modeling.train", fake_train_module)

    def fail_summary(_model_path):
        raise RuntimeError("missing evals.txt")

    monkeypatch.setattr("libs.web.services.summarize_model_evaluation", fail_summary)

    result = run_training_impl(TrainingRequest())

    captured = capsys.readouterr()
    assert result["evaluation"] == {
        "available": False,
        "message": "missing evals.txt",
        "model_path": str(tmp_path / "AutogluonModels" / "script-default"),
    }
    assert "[training] evaluation summary failed: RuntimeError: missing evals.txt" in captured.out


def test_list_fighters_rejects_csv_outside_data_dir(monkeypatch, tmp_path):
    data_root = tmp_path / "data"
    outside = tmp_path / "outside" / "prediction_data.csv"
    monkeypatch.setenv("MMA_AI_DATA_DIR", str(data_root))
    write_csv(outside, [{"fighter_name": "fighter one"}])

    with pytest.raises(ValueError, match="prediction_data.csv path must be under"):
        list_fighters(str(outside))


def test_event_prediction_rejects_output_dir_outside_data_dir(monkeypatch, tmp_path):
    data_root = tmp_path / "data"
    outside = tmp_path / "outside" / "prediction-output"
    monkeypatch.setenv("MMA_AI_DATA_DIR", str(data_root))
    data_root.mkdir()

    request = EventPredictionRequest(output_dir=str(outside))

    with pytest.raises(ValueError, match="output directory must be under"):
        run_event_prediction(request)
