from pathlib import Path
import json
import subprocess
import sys
from types import ModuleType, SimpleNamespace

import pandas as pd
import pytest

from libs.web.models import DataRefreshRequest, EventPredictionRequest, MatchupPredictionRequest, TrainingRequest
from libs.web.services import (
    get_data_status,
    get_readiness_status,
    list_fighters,
    list_models,
    list_upcoming_events,
    run_data_refresh,
    run_event_prediction,
    run_matchup_prediction,
    run_training,
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


def test_get_readiness_status_requires_seed_data_model_csvs_model_and_databases(monkeypatch, tmp_path):
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
    monkeypatch.setattr("libs.web.services._database_ready", lambda url: {"ok": True, "url": url})

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


def test_get_readiness_status_reports_missing_prerequisites(monkeypatch, tmp_path):
    monkeypatch.setenv("MMA_AI_UFCSTATS_DIR", str(tmp_path / "raw"))
    monkeypatch.setenv("MMA_AI_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("MMA_AI_MODELS_DIR", str(tmp_path / "AutogluonModels"))
    monkeypatch.setattr("libs.web.services._database_ready", lambda url: {"ok": False, "url": url, "error": "offline"})

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
    monkeypatch.setattr("libs.web.services._database_ready", lambda url: {"ok": True, "url": url})

    readiness = get_readiness_status()

    assert readiness["ready"] is False
    assert readiness["checks"]["starter_model"]["ok"] is False
    assert readiness["checks"]["starter_model"]["expected"] == "ag-20260304_110750-win-extreme"
    assert readiness["checks"]["starter_model"]["models"] == ["some-other-model"]


@pytest.mark.parametrize("odds_enabled", [False, True])
def test_run_data_refresh_passes_odds_flag_to_rebuild(monkeypatch, tmp_path, odds_enabled):
    raw_dir = tmp_path / "raw"
    data_dir = tmp_path / "data"
    monkeypatch.setenv("MMA_AI_UFCSTATS_DIR", str(raw_dir))
    monkeypatch.setenv("MMA_AI_DATA_DIR", str(data_dir))
    monkeypatch.setenv("DATABASE_URL", "postgresql://postgres:postgres@localhost:5432/mma-ai")
    captured = {}

    def fake_rebuild_main(**kwargs):
        captured["kwargs"] = kwargs

    fake_main_module = ModuleType("main")
    fake_main_module.main = fake_rebuild_main
    monkeypatch.setitem(sys.modules, "main", fake_main_module)

    result = run_data_refresh(
        DataRefreshRequest(
            scrape=False,
            rebuild=True,
            reset_db=True,
            odds=odds_enabled,
        )
    )

    assert result["scrape_counts"] == {}
    assert captured["kwargs"]["odds"] is odds_enabled
    assert captured["kwargs"]["raw_data_dir"] == raw_dir
    assert captured["kwargs"]["output_data_dir"] == data_dir
    assert captured["kwargs"]["scrape"] is False
    assert captured["kwargs"]["reset_db"] is True


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
    assert models[0]["has_scaler"] is True


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

    def fake_run(command, **kwargs):
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

    monkeypatch.setattr("libs.web.services.subprocess.run", fake_run)
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
    assert result["output_dir"] == str(tmp_path / "predictions" / "manual-custom")
    assert result["predictions"][0]["AI_Pick"] == "fighter one"


def test_run_matchup_prediction_can_fetch_odds_noninteractively(monkeypatch, tmp_path):
    monkeypatch.setenv("MMA_AI_DATA_DIR", str(tmp_path))
    prediction_csv = tmp_path / "prediction_data.csv"
    write_csv(prediction_csv, [{"fighter_name": "fighter one"}, {"fighter_name": "fighter two"}])
    captured = {}

    def fake_run(command, **kwargs):
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

    monkeypatch.setattr("libs.web.services.subprocess.run", fake_run)
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
    assert result["predictions"][0]["EV"] == "1"


def test_run_event_prediction_respects_prediction_knobs(monkeypatch, tmp_path):
    monkeypatch.setenv("MMA_AI_DATA_DIR", str(tmp_path))
    prediction_csv = tmp_path / "prediction_data.csv"
    write_csv(prediction_csv, [{"fighter_name": "fighter one"}, {"fighter_name": "fighter two"}])
    captured = {}

    def fake_run(command, **kwargs):
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

    monkeypatch.setattr("libs.web.services.subprocess.run", fake_run)
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
    assert result["output_dir"] == str(tmp_path / "predictions" / "event-custom")
    assert result["predictions"][0]["EV"] == "1"


def test_run_event_prediction_passes_manual_odds_json(monkeypatch, tmp_path):
    monkeypatch.setenv("MMA_AI_DATA_DIR", str(tmp_path))
    prediction_csv = tmp_path / "prediction_data.csv"
    write_csv(prediction_csv, [{"fighter_name": "fighter one"}, {"fighter_name": "fighter two"}])
    captured = {}

    def fake_run(command, **kwargs):
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

    monkeypatch.setattr("libs.web.services.subprocess.run", fake_run)
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
    assert result["stdout_tail"] == "manual odds ok"
    assert result["stderr_tail"] == "debug stderr"


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

    result = run_training(
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

    result = run_training(request)

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

    result = run_training(
        TrainingRequest(
            use_script_defaults=True,
            start_date="2016-01-01",
            walkforward_n_windows=8,
        )
    )

    assert result["used_script_defaults"] is False
    assert captured["config"].start_date == "2016-01-01"
    assert captured["config"].walkforward_n_windows == 8


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
