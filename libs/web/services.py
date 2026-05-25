"""Service layer for web workflows.

The heavy training, scraping, and prediction modules are imported lazily so the
web app can start, render status, and run tests without AutoGluon or Scrapy side
effects.
"""

from __future__ import annotations

import csv
import json
import os
import subprocess
import sys
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from libs.paths import PROJECT_ROOT, data_dir, data_file, database_url, models_dir, raw_ufcstats_dir
from libs.web.evaluations import summarize_model_evaluation
from libs.web.models import DataRefreshRequest, EventPredictionRequest, MatchupPredictionRequest, TrainingRequest
from libs.web.path_safety import resolve_data_csv, resolve_data_output_dir, resolve_model_dir


@dataclass(frozen=True)
class DashboardDefaults:
    train: dict[str, Any]
    predict: dict[str, Any]
    data: dict[str, Any]


def _count_csv_rows(path: Path) -> int | None:
    if not path.exists():
        return None
    with path.open("r", newline="", encoding="utf-8", errors="replace") as handle:
        reader = csv.reader(handle)
        try:
            next(reader)
        except StopIteration:
            return 0
        return sum(1 for _row in reader)


def _redact_url(url: str) -> str:
    if "@" not in url or "://" not in url:
        return url
    scheme, rest = url.split("://", 1)
    credentials, host = rest.split("@", 1)
    username = credentials.split(":", 1)[0]
    return f"{scheme}://{username}:***@{host}"


def get_dashboard_defaults() -> dict[str, Any]:
    defaults = DashboardDefaults(
        data={
            "scrape": True,
            "rebuild": True,
            "force_full": False,
            "reset_db": True,
            "odds": False,
            "log_level": "INFO",
            "analytics_max_rows": 100,
        },
        train={
            "model_type": "win",
            "preset": "extreme",
            "time_limit": 3000,
            "split_strategy": "timeseries_split",
            "walkforward_n_windows": 4,
            "walkforward_initial_year": 2021,
            "refit_full": True,
            "refit_all": False,
            "start_date": "2014-01-01",
            "num_fights": 2,
            "include_split_dec": True,
            "normalize": "robust",
            "use_recency_weights": True,
            "decay_rate": 0.15,
            "calculate_importance": True,
            "feature_list": None,
            "included_strings": None,
            "excluded_strings": None,
            "required_strings": None,
            "included_model_types": ["TABICL", "MITRA", "TABM", "GBM_PREP", "CAT", "GBM", "REALTABPFN-V2"],
        },
        predict={
            "model_type": "win",
            "upcoming_number": 1,
            "odds": True,
            "use_calibrated": False,
            "shap": False,
        },
    )
    return asdict(defaults)


def get_data_status() -> dict[str, Any]:
    raw_dir = raw_ufcstats_dir()
    app_data_dir = data_dir()
    prediction_csv = data_file("prediction_data.csv")
    training_csv = data_file("training_data.csv")
    decision_csv = data_file("training_data_dec.csv")
    return {
        "project_root": str(PROJECT_ROOT),
        "database_url": _redact_url(database_url()),
        "raw_data_dir": str(raw_dir),
        "data_dir": str(app_data_dir),
        "raw_csvs": {
            "competitions": {
                "path": str(raw_dir / "competitions.csv"),
                "rows": _count_csv_rows(raw_dir / "competitions.csv"),
            },
            "individuals": {
                "path": str(raw_dir / "individuals.csv"),
                "rows": _count_csv_rows(raw_dir / "individuals.csv"),
            },
        },
        "model_csvs": {
            "prediction_data": {"path": str(prediction_csv), "rows": _count_csv_rows(prediction_csv)},
            "training_data": {"path": str(training_csv), "rows": _count_csv_rows(training_csv)},
            "training_data_dec": {"path": str(decision_csv), "rows": _count_csv_rows(decision_csv)},
        },
    }


def run_data_refresh(request: DataRefreshRequest) -> dict[str, Any]:
    counts: dict[str, int] = {}
    raw_dir = raw_ufcstats_dir()
    app_data_dir = data_dir()
    print(
        "[data-refresh] "
        f"scrape={request.scrape} rebuild={request.rebuild} reset_db={request.reset_db} "
        f"force_full={request.force_full} odds={request.odds} raw_dir={raw_dir} data_dir={app_data_dir}"
    )

    if request.scrape:
        from libs.scraping.ufcstats import scrape_ufcstats

        print("[data-refresh] starting UFCStats scraper")
        counts = scrape_ufcstats(
            output_dir=raw_dir,
            fighters=True,
            fights=True,
            force_full=request.force_full,
            log_level=request.log_level,
        )
        print(f"[data-refresh] scraper finished: {counts}")

    if request.rebuild:
        from main import main as rebuild_main

        print("[data-refresh] starting feature-store rebuild")
        rebuild_main(
            odds=request.odds,
            db_url=database_url(),
            raw_data_dir=raw_dir,
            output_data_dir=app_data_dir,
            scrape=False,
            reset_db=request.reset_db,
        )
        print("[data-refresh] feature-store rebuild finished")

    return {"scrape_counts": counts, "status": get_data_status()}


def list_models() -> list[dict[str, Any]]:
    root = models_dir()
    if not root.exists():
        return []

    summaries = []
    for path in sorted((p for p in root.iterdir() if p.is_dir()), key=lambda p: p.stat().st_mtime, reverse=True):
        marker_files = ["predictor.pkl", "learner.pkl", "metadata.json", "feats.txt"]
        if not any((path / marker).exists() for marker in marker_files):
            continue
        summaries.append(
            {
                "name": path.name,
                "path": str(path),
                "modified_at": path.stat().st_mtime,
                "has_features": (path / "feats.txt").exists(),
                "has_scaler": (path / "scaler.pkl").exists(),
                "has_calibrator": (path / "calibrator.pkl").exists(),
            }
        )
    return summaries


def list_fighters(prediction_data_csv: str | None = None) -> list[str]:
    path = resolve_data_csv(prediction_data_csv, "prediction_data.csv")
    if not path.exists():
        return []

    df = pd.read_csv(path, usecols=lambda column: column in {"fighter_name", "fighter1_name", "fighter2_name"})
    names: set[str] = set()
    for column in ("fighter_name", "fighter1_name", "fighter2_name"):
        if column in df.columns:
            names.update(str(name) for name in df[column].dropna().unique())
    return sorted(names, key=str.lower)


def list_upcoming_events(prediction_data_csv: str | None = None, limit: int = 5) -> dict[str, Any]:
    path = resolve_data_csv(prediction_data_csv, "prediction_data.csv")
    if not path.exists():
        return {"events": [], "warning": f"Prediction data CSV not found: {path}"}

    from libs.upcoming_fights import UpcomingFights

    df = pd.read_csv(path)
    events = []
    warnings = []
    for upcoming_number in range(1, limit + 1):
        try:
            event_map = UpcomingFights(df, upcoming_number).run()
        except Exception as exc:
            warnings.append(str(exc))
            break
        for event_name, fights in event_map.items():
            events.append(
                {
                    "upcoming_number": upcoming_number,
                    "name": event_name,
                    "fights": [
                        {
                            "date": fight[0].isoformat() if hasattr(fight[0], "isoformat") else str(fight[0]),
                            "fighter1": fight[1],
                            "fighter2": fight[2],
                        }
                        for fight in fights
                    ],
                }
            )
    return {"events": events, "warning": "; ".join(warnings) if warnings else None}


def run_training(request: TrainingRequest) -> dict[str, Any]:
    print(
        "[training] "
        f"model_type={request.model_type} preset={request.preset} time_limit={request.time_limit} "
        f"split_strategy={request.split_strategy} script_defaults={request.use_script_defaults}"
    )
    if _can_use_training_script_defaults(request):
        from libs.modeling.train import main as train_main

        print("[training] using libs.modeling.train.main defaults path")
        predictor = train_main(
            model_type=request.model_type,
            time_limit=request.time_limit,
            preset=request.preset,
            split_strategy=request.split_strategy,
            refit_full=request.refit_full,
        )
        model_path = str(getattr(predictor, "path", ""))
        print(f"[training] completed script-default training: model_path={model_path}")
        return {"model_path": model_path, "used_script_defaults": True, "evaluation": _safe_evaluation(model_path)}

    from libs.modeling import train as train_module
    print("[training] using custom TrainingConfig path")

    if request.model_type == "win":
        features = train_module.vSeven_testing2
        included_strings = None
        excluded_strings = None
        required_strings = None
    else:
        features = train_module.DECISION_TEST_FEATS4
        included_strings = ["time_sec", "decision", "sub", "ko", "kd", "win", "strikes_att", "distance_att", "td", "ctrl", "weightclass_encoded"]
        excluded_strings = ["total_avg"]
        required_strings = None

    features = request.feature_list or features
    included_strings = request.included_strings or included_strings
    excluded_strings = request.excluded_strings or excluded_strings
    required_strings = request.required_strings or required_strings

    config = train_module.TrainingConfig(
        model_type=request.model_type,
        preset=request.preset,
        time_limit=request.time_limit,
        test_size=request.test_size,
        val_date=request.val_date,
        features=features,
        included_strings=included_strings,
        excluded_strings=excluded_strings,
        required_strings=required_strings,
        start_date=request.start_date,
        num_fights=request.num_fights,
        include_split_dec=request.include_split_dec,
        normalize=request.normalize,
        use_recency_weights=request.use_recency_weights,
        decay_rate=request.decay_rate,
        split_strategy=request.split_strategy,
        walkforward_n_windows=request.walkforward_n_windows,
        walkforward_initial_year=request.walkforward_initial_year,
        calculate_importance=request.calculate_importance,
        refit_all=request.refit_all,
        refit_full=request.refit_full,
        included_model_types=request.included_model_types,
    )
    predictor = train_module.ModelTrainer(config).train()
    model_path = str(getattr(predictor, "path", ""))
    print(f"[training] completed custom training: model_path={model_path}")
    return {"model_path": model_path, "used_script_defaults": False, "evaluation": _safe_evaluation(model_path)}


def _can_use_training_script_defaults(request: TrainingRequest) -> bool:
    if not request.use_script_defaults:
        return False

    defaults = get_dashboard_defaults()["train"]
    advanced_matches = {
        "test_size": request.test_size is None,
        "val_date": request.val_date is None,
        "start_date": request.start_date == defaults["start_date"],
        "num_fights": request.num_fights == defaults["num_fights"],
        "include_split_dec": request.include_split_dec == defaults["include_split_dec"],
        "normalize": request.normalize == defaults["normalize"],
        "use_recency_weights": request.use_recency_weights == defaults["use_recency_weights"],
        "decay_rate": request.decay_rate == defaults["decay_rate"],
        "walkforward_n_windows": request.walkforward_n_windows == defaults["walkforward_n_windows"],
        "walkforward_initial_year": request.walkforward_initial_year == defaults["walkforward_initial_year"],
        "calculate_importance": request.calculate_importance == defaults["calculate_importance"],
        "feature_list": not request.feature_list,
        "included_strings": not request.included_strings,
        "excluded_strings": not request.excluded_strings,
        "required_strings": not request.required_strings,
        "refit_all": request.refit_all == defaults["refit_all"],
        "included_model_types": request.included_model_types in (None, defaults["included_model_types"]),
    }
    return all(advanced_matches.values())


def _safe_evaluation(model_path: str) -> dict[str, Any] | None:
    if not model_path:
        return None
    try:
        return summarize_model_evaluation(model_path)
    except Exception as exc:
        return {"available": False, "message": str(exc), "model_path": model_path}


def _base_prediction_command(model_type: str, output_dir: Path) -> list[str]:
    command = [
        sys.executable,
        str(PROJECT_ROOT / "predict.py"),
        "--model-type",
        model_type,
        "--output-dir",
        str(output_dir),
    ]
    return command


def _run_prediction_command(command: list[str], output_dir: Path) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"[prediction] command: {subprocess.list2cmdline(command)}")
    completed = subprocess.run(
        command,
        cwd=PROJECT_ROOT,
        text=True,
        capture_output=True,
        check=False,
        env=os.environ.copy(),
    )
    print(f"[prediction] exit_code={completed.returncode}")
    if completed.stdout:
        print("[prediction] stdout begin")
        print(completed.stdout.rstrip())
        print("[prediction] stdout end")
    if completed.stderr:
        print("[prediction] stderr begin")
        print(completed.stderr.rstrip())
        print("[prediction] stderr end")
    if completed.returncode != 0:
        raise RuntimeError(completed.stderr or completed.stdout or f"Prediction failed with exit code {completed.returncode}")

    csv_path = output_dir / "fight_predictions.csv"
    rows = _read_prediction_rows(csv_path)
    return {
        "output_dir": str(output_dir),
        "csv_path": str(csv_path) if csv_path.exists() else None,
        "predictions": rows,
        "stdout_tail": completed.stdout[-4000:],
        "stderr_tail": completed.stderr[-4000:],
    }


def _append_common_prediction_args(command: list[str], request: EventPredictionRequest | MatchupPredictionRequest) -> None:
    if request.model_path:
        command.extend(["--model-path", str(resolve_model_dir(request.model_path))])
    if request.prediction_data_csv:
        command.extend(["--prediction-data-csv", str(resolve_data_csv(request.prediction_data_csv, "prediction_data.csv"))])
    if request.training_data_csv:
        command.extend(["--training-data-csv", str(resolve_data_csv(request.training_data_csv, "training_data.csv"))])
    if request.manual_odds:
        command.extend(["--manual-odds-json", json.dumps(request.manual_odds, sort_keys=True)])
    if request.odds:
        command.append("--odds")
        command.append("--no-manual-odds")
    if request.use_calibrated:
        command.append("--use-calibrated")
    if not request.shap:
        command.append("--no-shap")


def run_event_prediction(request: EventPredictionRequest) -> dict[str, Any]:
    validate_event_prediction_request(request)
    output_dir = resolve_data_output_dir(request.output_dir, "predictions/latest")
    command = _base_prediction_command(request.model_type, output_dir)
    command.extend(["--upcoming-number", str(request.upcoming_number)])
    _append_common_prediction_args(command, request)
    return _run_prediction_command(command, output_dir)


def validate_event_prediction_request(request: EventPredictionRequest) -> dict[str, Any]:
    if request.model_path:
        resolve_model_dir(request.model_path)
    if request.prediction_data_csv:
        resolve_data_csv(request.prediction_data_csv, "prediction_data.csv")
    if request.training_data_csv:
        resolve_data_csv(request.training_data_csv, "training_data.csv")
    output_dir = resolve_data_output_dir(request.output_dir, "predictions/latest")
    return {
        "model_type": request.model_type,
        "model_path": request.model_path,
        "output_dir": str(output_dir),
        "upcoming_number": request.upcoming_number,
        "status": "ready_for_prediction",
    }


def validate_matchup_request(request: MatchupPredictionRequest) -> dict[str, Any]:
    if request.model_path:
        resolve_model_dir(request.model_path)
    if request.prediction_data_csv:
        resolve_data_csv(request.prediction_data_csv, "prediction_data.csv")
    if request.training_data_csv:
        resolve_data_csv(request.training_data_csv, "training_data.csv")
    output_dir = resolve_data_output_dir(request.output_dir, "predictions/manual")

    fighter1 = request.fighter1.strip()
    fighter2 = request.fighter2.strip()
    if not fighter1 or not fighter2:
        raise ValueError("Enter both fighter names before prediction.")
    if fighter1.lower() == fighter2.lower():
        raise ValueError("Choose two different fighters for a matchup prediction.")
    fight_date = request.fight_date.strip() if request.fight_date else None
    if fight_date:
        try:
            datetime.strptime(fight_date, "%Y-%m-%d")
        except ValueError as exc:
            raise ValueError("Fight date must use YYYY-MM-DD format.") from exc

    fighters = set(list_fighters(request.prediction_data_csv))
    missing = [name for name in (fighter1, fighter2) if name not in fighters]
    if missing:
        raise ValueError(f"Fighter not found in prediction data: {', '.join(missing)}")

    return {
        "fighter1": fighter1,
        "fighter2": fighter2,
        "fight_date": fight_date,
        "model_type": request.model_type,
        "model_path": request.model_path,
        "output_dir": str(output_dir),
        "status": "ready_for_prediction",
    }


def run_matchup_prediction(request: MatchupPredictionRequest) -> dict[str, Any]:
    validated = validate_matchup_request(request)
    output_dir = Path(validated["output_dir"])
    command = _base_prediction_command(request.model_type, output_dir)
    command.extend(["--fighter1", validated["fighter1"], "--fighter2", validated["fighter2"]])
    if validated["fight_date"]:
        command.extend(["--fight-date", validated["fight_date"]])
    if request.odds_fighter1 is not None:
        command.extend(["--fighter1-odds", str(request.odds_fighter1)])
    if request.odds_fighter2 is not None:
        command.extend(["--fighter2-odds", str(request.odds_fighter2)])
    _append_common_prediction_args(command, request)
    return _run_prediction_command(command, output_dir)


def _read_prediction_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8", errors="replace") as handle:
        non_comment_lines = [line for line in handle if not line.startswith("#")]
    if not non_comment_lines:
        return []
    reader = csv.DictReader(non_comment_lines)
    return [dict(row) for row in reader]
