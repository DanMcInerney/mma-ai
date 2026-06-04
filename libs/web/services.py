"""Service layer for web workflows.

The heavy training, scraping, and prediction modules are imported lazily so the
web app can start, render status, and run tests without AutoGluon or Scrapy side
effects.
"""

from __future__ import annotations

import csv
from copy import deepcopy
import json
import os
import re
import subprocess
import sys
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from threading import Lock, Thread
from time import monotonic
from typing import Any
from urllib.parse import unquote

import pandas as pd

from libs.modeling.discovery import is_loadable_prediction_model_dir
from libs.modeling.runtime_dependencies import assert_prediction_runtime_dependencies, prediction_runtime_dependency_report
from libs.paths import PROJECT_ROOT, data_dir, data_file, database_url, models_dir, odds_database_url, raw_ufcstats_dir
from libs.web.evaluations import summarize_model_evaluation, write_model_evaluation_report
from libs.web.llm import llm_config, llm_config_hint
from libs.web.models import DataRefreshRequest, EventPredictionRequest, MatchupPredictionRequest, TrainingRequest
from libs.web.path_safety import resolve_data_csv, resolve_data_output_dir, resolve_model_dir


STARTER_MODEL_NAME = "ag-20260304_110750-win-extreme"
TRAINING_RESULT_BEGIN = "<<<MMA_AI_TRAINING_RESULT_BEGIN>>>"
TRAINING_RESULT_END = "<<<MMA_AI_TRAINING_RESULT_END>>>"
UPCOMING_EVENTS_CACHE_TTL_SECONDS = int(os.getenv("MMA_AI_UPCOMING_EVENTS_CACHE_TTL_SECONDS", "900"))
READINESS_CSV_REQUIRED_COLUMNS = {
    ("raw_csvs", "competitions"): {"event_url"},
    ("raw_csvs", "individuals"): {"url"},
    ("model_csvs", "prediction_data"): {"fighter_name"},
    ("model_csvs", "training_data"): {"y_true"},
    ("model_csvs", "training_data_dec"): {"y_true"},
}
_upcoming_events_cache: dict[tuple[Any, ...], tuple[float, dict[str, Any]]] = {}
_upcoming_events_cache_lock = Lock()


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


def _csv_has_data_row(path: Path) -> int | None:
    """Return 1 when a CSV has at least one data row without scanning it."""
    if not path.exists():
        return None
    try:
        with path.open("r", newline="", encoding="utf-8", errors="replace") as handle:
            reader = csv.reader(handle)
            try:
                next(reader)
            except StopIteration:
                return 0
            try:
                next(reader)
            except StopIteration:
                return 0
    except (OSError, csv.Error):
        return None
    return 1


def _read_csv_header(path: Path) -> tuple[set[str] | None, str | None]:
    if not path.exists():
        return None, None
    try:
        with path.open("r", newline="", encoding="utf-8", errors="replace") as handle:
            reader = csv.reader(handle)
            try:
                header = next(reader)
            except StopIteration:
                return set(), None
    except (OSError, csv.Error) as exc:
        return set(), type(exc).__name__
    return {column.strip() for column in header if column.strip()}, None


def _csv_readiness_check(entry: dict[str, Any], required_columns: set[str]) -> dict[str, Any]:
    rows = entry["rows"]
    path = Path(entry["path"])
    header_columns, error = _read_csv_header(path)
    missing_columns = sorted(required_columns - (header_columns or set()))
    check = {
        "ok": rows is not None and rows > 0 and not missing_columns and not error,
        "rows": rows,
        "path": entry["path"],
        "required_columns": sorted(required_columns),
        "missing_columns": missing_columns,
    }
    if error:
        check["error"] = error
    return check


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
            "odds_features": True,
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
            "flaresolverr": False,
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


def get_readiness_status() -> dict[str, Any]:
    """Return whether the dashboard has the artifacts needed for first use."""
    checks: dict[str, dict[str, Any]] = {}
    raw_dir = raw_ufcstats_dir()
    readiness_csvs = {
        ("raw_csvs", "competitions"): raw_dir / "competitions.csv",
        ("raw_csvs", "individuals"): raw_dir / "individuals.csv",
        ("model_csvs", "prediction_data"): data_file("prediction_data.csv"),
        ("model_csvs", "training_data"): data_file("training_data.csv"),
        ("model_csvs", "training_data_dec"): data_file("training_data_dec.csv"),
    }

    for group, key in (
        ("raw_csvs", "competitions"),
        ("raw_csvs", "individuals"),
        ("model_csvs", "prediction_data"),
        ("model_csvs", "training_data"),
        ("model_csvs", "training_data_dec"),
    ):
        path = readiness_csvs[(group, key)]
        entry = {
            "path": str(path),
            "rows": _csv_has_data_row(path),
        }
        checks[f"{key}_csv"] = _csv_readiness_check(
            entry,
            READINESS_CSV_REQUIRED_COLUMNS[(group, key)],
        )

    models = list_models()
    starter_model = next((model for model in models if model["name"] == STARTER_MODEL_NAME), None)
    checks["starter_model"] = {
        "ok": bool(starter_model),
        "expected": STARTER_MODEL_NAME,
        "count": len(models),
        "models": [model["name"] for model in models[:5]],
        "path": starter_model["path"] if starter_model else None,
    }
    checks["database"] = _database_ready(database_url(), required_tables=["features.fight_mapping"])
    checks["odds_database"] = _database_ready(odds_database_url(), required_tables=["bestfightodds.bfo"])
    checks["prediction_runtime"] = prediction_runtime_dependency_report()

    ready = all(check["ok"] for check in checks.values())
    return {
        "status": "ok" if ready else "not_ready",
        "ready": ready,
        "checks": checks,
    }


def get_analytics_status() -> dict[str, Any]:
    """Return non-secret analytics LLM configuration status for the dashboard."""
    config = llm_config()
    configured = bool(config and config.is_configured)
    return {
        "configured": configured,
        "provider": config.provider if config else None,
        "model": config.model if config else None,
        "base_url": config.base_url if config and config.provider in {"local", "custom"} else None,
        "needs_api_key": config.needs_api_key if config else None,
        "mode": "llm" if configured else "sql_only",
        "hint": None if configured else llm_config_hint(),
    }


def _database_ready(url: str, required_tables: list[str] | None = None) -> dict[str, Any]:
    try:
        from sqlalchemy import create_engine, text

        engine_kwargs: dict[str, Any] = {"pool_pre_ping": True}
        if url.startswith("postgresql"):
            engine_kwargs["connect_args"] = {"connect_timeout": 3}
        engine = create_engine(url, **engine_kwargs)
        try:
            with engine.connect() as connection:
                connection.execute(text("select 1"))
                missing_tables: list[str] = []
                if required_tables and url.startswith("postgresql"):
                    for table_name in required_tables:
                        exists = connection.execute(
                            text("select to_regclass(:table_name) is not null"),
                            {"table_name": table_name},
                        ).scalar()
                        if not exists:
                            missing_tables.append(table_name)
                if missing_tables:
                    return {"ok": False, "url": _redact_url(url), "missing_tables": missing_tables}
        finally:
            engine.dispose()
    except Exception as exc:
        return {"ok": False, "url": _redact_url(url), "error": type(exc).__name__}

    return {"ok": True, "url": _redact_url(url)}


def run_data_refresh(request: DataRefreshRequest) -> dict[str, Any]:
    counts: dict[str, int] = {}
    raw_dir = raw_ufcstats_dir()
    app_data_dir = data_dir()
    before_status = get_data_status()
    print(
        "[data-refresh] "
        f"scrape={request.scrape} rebuild={request.rebuild} reset_db={request.reset_db} "
        f"force_full={request.force_full} odds_features={request.odds_features} "
        f"odds={request.odds} raw_dir={raw_dir} data_dir={app_data_dir}"
    )

    if request.scrape:
        print("[data-refresh] starting UFCStats scraper subprocess")
        counts = _run_scrape_command(request, raw_dir)
        print(f"[data-refresh] scraper subprocess finished: {counts}")

    if request.rebuild:
        print("[data-refresh] starting feature-store rebuild subprocess")
        _run_rebuild_command(request, raw_dir, app_data_dir)
        print("[data-refresh] feature-store rebuild subprocess finished")

    after_status = get_data_status()
    row_deltas = _data_status_row_deltas(before_status, after_status)
    print(f"[data-refresh] row deltas: {row_deltas}")
    return {
        "scrape_counts": counts,
        "before_status": before_status,
        "status": after_status,
        "row_deltas": row_deltas,
    }


def _data_status_row_deltas(before_status: dict[str, Any], after_status: dict[str, Any]) -> dict[str, int | None]:
    deltas: dict[str, int | None] = {}
    groups = (
        ("raw_csvs", ("competitions", "individuals")),
        ("model_csvs", ("prediction_data", "training_data", "training_data_dec")),
    )
    for group, entries in groups:
        for key in entries:
            before = ((before_status.get(group) or {}).get(key) or {}).get("rows")
            after = ((after_status.get(group) or {}).get(key) or {}).get("rows")
            deltas[key] = after - before if isinstance(before, int) and isinstance(after, int) else None
    return deltas


def _run_logged_subprocess(
    command: list[str],
    log_prefix: str,
    *,
    stdout_label: str = "stdout",
    stderr_label: str = "stderr",
    input: str | None = None,
) -> subprocess.CompletedProcess[str]:
    """Run a child process while streaming stdout/stderr into the job log."""
    print(f"[{log_prefix}] command: {subprocess.list2cmdline(command)}")
    child_env = os.environ.copy()
    child_env["PYTHONUNBUFFERED"] = "1"
    process = subprocess.Popen(
        command,
        cwd=PROJECT_ROOT,
        stdin=subprocess.PIPE if input is not None else None,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,
        env=child_env,
    )

    stdout_chunks: list[str] = []
    stderr_chunks: list[str] = []

    def drain_stream(stream: Any, chunks: list[str], label: str, destination: Any) -> None:
        began = False
        try:
            for line in iter(stream.readline, ""):
                if not began:
                    print(f"[{log_prefix}] {label} begin", file=destination)
                    began = True
                chunks.append(line)
                print(line, end="", file=destination, flush=True)
        finally:
            if began:
                print(f"[{log_prefix}] {label} end", file=destination)

    stdout_thread = Thread(target=drain_stream, args=(process.stdout, stdout_chunks, stdout_label, sys.stdout), daemon=True)
    stderr_thread = Thread(target=drain_stream, args=(process.stderr, stderr_chunks, stderr_label, sys.stderr), daemon=True)
    stdout_thread.start()
    stderr_thread.start()

    if input is not None and process.stdin is not None:
        try:
            process.stdin.write(input)
            process.stdin.close()
        except (BrokenPipeError, OSError):
            pass

    returncode = process.wait()
    stdout_thread.join()
    stderr_thread.join()
    print(f"[{log_prefix}] exit_code={returncode}")
    return subprocess.CompletedProcess(command, returncode, stdout="".join(stdout_chunks), stderr="".join(stderr_chunks))


def _run_scrape_command(request: DataRefreshRequest, raw_dir: Path) -> dict[str, int]:
    command = [
        sys.executable,
        str(PROJECT_ROOT / "scripts" / "scrape_ufcstats.py"),
        "--output-dir",
        str(raw_dir),
        "--log-level",
        request.log_level,
    ]
    if request.force_full:
        command.append("--force-full")

    completed = _run_logged_subprocess(
        command,
        "data-refresh",
        stdout_label="scraper stdout",
        stderr_label="scraper stderr",
    )
    if completed.returncode != 0:
        raise RuntimeError(completed.stderr or completed.stdout or f"UFCStats scrape failed with exit code {completed.returncode}")
    return _parse_scrape_counts(completed.stdout)


def _run_rebuild_command(request: DataRefreshRequest, raw_dir: Path, app_data_dir: Path) -> None:
    command = [
        sys.executable,
        str(PROJECT_ROOT / "main.py"),
        "--raw-data-dir",
        str(raw_dir),
        "--output-data-dir",
        str(app_data_dir),
    ]
    if request.reset_db:
        command.append("--reset-db")
    if request.odds_features:
        command.append("--odds-features")
    if request.odds:
        command.append("--odds")

    completed = _run_logged_subprocess(
        command,
        "data-refresh",
        stdout_label="rebuild stdout",
        stderr_label="rebuild stderr",
    )
    if completed.returncode != 0:
        raise RuntimeError(completed.stderr or completed.stdout or f"Feature-store rebuild failed with exit code {completed.returncode}")


def _parse_scrape_counts(stdout: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for line in stdout.splitlines():
        match = re.match(r"^\s*(fighters|fights):\s*(\d+)\s+total rows\s*$", line, flags=re.IGNORECASE)
        if match:
            counts[match.group(1).lower()] = int(match.group(2))
    return counts


def list_models(model_type: str | None = None) -> list[dict[str, Any]]:
    root = models_dir()
    if not root.exists():
        return []

    summaries = []
    for path in sorted((p for p in root.iterdir() if p.is_dir()), key=lambda p: p.stat().st_mtime, reverse=True):
        if model_type and f"-{model_type}-" not in path.name:
            continue
        if not is_loadable_prediction_model_dir(path):
            continue
        summaries.append(
            {
                "name": path.name,
                "path": str(path),
                "modified_at": path.stat().st_mtime,
                "has_features": (path / "feats.txt").exists(),
                "has_predictor": (path / "predictor.pkl").exists(),
                "is_ensemble": (path / "ensemble_info.txt").exists(),
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


def list_upcoming_events(
    prediction_data_csv: str | None = None,
    limit: int | None = None,
    *,
    force_refresh: bool = False,
) -> dict[str, Any]:
    path = resolve_data_csv(prediction_data_csv, "prediction_data.csv")
    normalized_limit = _normalize_upcoming_limit(limit)
    cache_key = _upcoming_events_cache_key(path, normalized_limit)
    if not force_refresh:
        cached = _cached_upcoming_events(cache_key)
        if cached is not None:
            return cached

    payload = _load_upcoming_events(path, normalized_limit)
    _store_upcoming_events(cache_key, payload)
    return deepcopy(payload)


def warm_up_upcoming_events(prediction_data_csv: str | None = None, limit: int | None = None) -> Thread:
    """Warm the Wikipedia-backed upcoming-event cache without blocking startup."""
    thread = Thread(
        target=_warm_up_upcoming_events_worker,
        args=(prediction_data_csv, limit),
        name="mma-ai-upcoming-events-warmup",
        daemon=True,
    )
    thread.start()
    return thread


def _warm_up_upcoming_events_worker(prediction_data_csv: str | None, limit: int | None) -> None:
    try:
        list_upcoming_events(prediction_data_csv, limit, force_refresh=True)
    except Exception as exc:
        print(f"[startup] upcoming event warmup failed: {type(exc).__name__}: {exc}")


def _normalize_upcoming_limit(limit: int | None) -> int | None:
    return max(1, int(limit)) if limit is not None else None


def _upcoming_events_cache_key(path: Path, limit: int | None) -> tuple[Any, ...]:
    try:
        stat = path.stat()
        return (str(path), limit, stat.st_mtime_ns, stat.st_size)
    except OSError:
        return (str(path), limit, None, None)


def _cached_upcoming_events(cache_key: tuple[Any, ...]) -> dict[str, Any] | None:
    with _upcoming_events_cache_lock:
        cached = _upcoming_events_cache.get(cache_key)
    if cached is None:
        return None
    loaded_at, payload = cached
    if monotonic() - loaded_at > UPCOMING_EVENTS_CACHE_TTL_SECONDS:
        with _upcoming_events_cache_lock:
            _upcoming_events_cache.pop(cache_key, None)
        return None
    return deepcopy(payload)


def _store_upcoming_events(cache_key: tuple[Any, ...], payload: dict[str, Any]) -> None:
    with _upcoming_events_cache_lock:
        _upcoming_events_cache[cache_key] = (monotonic(), deepcopy(payload))


def _load_upcoming_events(path: Path, limit: int | None = None) -> dict[str, Any]:
    from libs.upcoming_fights import UpcomingFights

    events = []
    warnings = []
    if path.exists():
        df = pd.read_csv(path)
        can_match_fights = True
    else:
        df = pd.DataFrame({"fighter_name": []})
        can_match_fights = False
        warnings.append(f"Prediction data CSV not found: {path}; showing Wikipedia event names without matched fights")

    scraper = UpcomingFights(df, 1)
    try:
        scheduled_events = _scheduled_events_from_scraper(scraper)
    except Exception as exc:
        scheduled_events = []
        warnings.append(f"Could not load scheduled event metadata from Wikipedia: {exc}")

    if scheduled_events:
        selected_events = [{**event, "_prefer_scheduled_name": True} for event in scheduled_events]
    else:
        try:
            event_links = scraper.get_upcoming_event_links()
        except Exception as exc:
            return {"events": [], "warning": f"Could not load upcoming UFC events from Wikipedia: {exc}"}
        selected_events = [{"url": link, "name": _event_name_from_url(link), "date": None} for link in reversed(event_links)]
    if limit is not None:
        selected_events = selected_events[:limit]

    for upcoming_number, scheduled_event in enumerate(selected_events, start=1):
        event_link = scheduled_event["url"]
        if not can_match_fights:
            events.append(
                _empty_upcoming_event(
                    event_link,
                    upcoming_number,
                    "Prediction data CSV is unavailable, so fight matching is disabled.",
                    name=scheduled_event.get("name"),
                    event_date=scheduled_event.get("date"),
                )
            )
            continue
        try:
            event_map = scraper.get_upcoming_cards([event_link])
        except Exception as exc:
            warnings.append(f"event {upcoming_number}: {exc}")
            events.append(
                _empty_upcoming_event(
                    event_link,
                    upcoming_number,
                    str(exc),
                    name=scheduled_event.get("name"),
                    event_date=scheduled_event.get("date"),
                )
            )
            continue
        if not event_map:
            warnings.append(f"event {upcoming_number}: no matched fights found")
            events.append(
                _empty_upcoming_event(
                    event_link,
                    upcoming_number,
                    "No matched fights found.",
                    name=scheduled_event.get("name"),
                    event_date=scheduled_event.get("date"),
                )
            )
            continue
        for event_name, fights in event_map.items():
            display_name = scheduled_event.get("name") if scheduled_event.get("_prefer_scheduled_name") else None
            events.append(
                {
                    "upcoming_number": upcoming_number,
                    "name": _clean_event_name(display_name or event_name),
                    "date": _iso_datetime(scheduled_event.get("date")),
                    "source_url": event_link,
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

    events.sort(key=_upcoming_event_sort_key)
    return {"events": events, "warning": "; ".join(warnings) if warnings else None}


def _scheduled_events_from_scraper(scraper: Any) -> list[dict[str, Any]]:
    get_scheduled_events = getattr(scraper, "get_scheduled_events", None)
    if not callable(get_scheduled_events):
        return []
    scheduled_events = []
    for item in get_scheduled_events() or []:
        event_link = item.get("url")
        if not event_link:
            continue
        scheduled_events.append(
            {
                "url": event_link,
                "name": _clean_event_name(item.get("name") or _event_name_from_url(event_link)),
                "date": item.get("date"),
            }
        )
    return sorted(scheduled_events, key=lambda event: (_parse_event_date(event.get("date")), event.get("name") or ""))


def _empty_upcoming_event(
    event_link: str,
    upcoming_number: int,
    warning: str,
    *,
    name: str | None = None,
    event_date: Any = None,
) -> dict[str, Any]:
    return {
        "upcoming_number": upcoming_number,
        "name": _clean_event_name(name) if name else _event_name_from_url(event_link),
        "date": _iso_datetime(event_date),
        "fights": [],
        "warning": warning,
        "source_url": event_link,
    }


def _event_name_from_url(event_link: str) -> str:
    slug = event_link.rstrip("/").rsplit("/", 1)[-1]
    name = _clean_event_name(slug)
    return name or f"Upcoming UFC event {event_link}"


def _clean_event_name(raw_name: str) -> str:
    return unquote(str(raw_name)).replace("_", " ").strip()


def _upcoming_event_sort_key(event: dict[str, Any]) -> tuple[datetime, int]:
    raw_date = event.get("date") or (event.get("fights") or [{}])[0].get("date")
    event_date = _parse_event_date(raw_date)
    return event_date, int(event.get("upcoming_number") or 0)


def _parse_event_date(raw_date: Any) -> datetime:
    try:
        if hasattr(raw_date, "to_pydatetime"):
            return raw_date.to_pydatetime()
        if hasattr(raw_date, "isoformat"):
            return datetime.fromisoformat(raw_date.isoformat())
        return datetime.fromisoformat(str(raw_date))
    except (TypeError, ValueError):
        return datetime.max


def _iso_datetime(raw_date: Any) -> str | None:
    event_date = _parse_event_date(raw_date)
    return None if event_date == datetime.max else event_date.isoformat()


def run_training(request: TrainingRequest) -> dict[str, Any]:
    return _run_training_command(request)


def run_training_impl(request: TrainingRequest) -> dict[str, Any]:
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


def _run_training_command(request: TrainingRequest) -> dict[str, Any]:
    command = [
        sys.executable,
        str(PROJECT_ROOT / "scripts" / "train_dashboard.py"),
    ]
    payload = json.dumps(_pydantic_dump(request), sort_keys=True)
    completed = _run_logged_subprocess(
        command,
        input=payload + "\n",
        log_prefix="training",
        stdout_label="stdout",
        stderr_label="stderr",
    )
    if completed.returncode != 0:
        raise RuntimeError(completed.stderr or completed.stdout or f"Training failed with exit code {completed.returncode}")
    return _parse_training_result(completed.stdout)


def _pydantic_dump(model: Any) -> dict[str, Any]:
    if hasattr(model, "model_dump"):
        return model.model_dump()
    return model.dict()


def _parse_training_result(stdout: str) -> dict[str, Any]:
    start = stdout.rfind(TRAINING_RESULT_BEGIN)
    end = stdout.rfind(TRAINING_RESULT_END)
    if start < 0 or end < 0 or end <= start:
        raise RuntimeError("Training script finished without a structured dashboard result.")
    raw_json = stdout[start + len(TRAINING_RESULT_BEGIN) : end].strip()
    try:
        result = json.loads(raw_json)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Training script emitted invalid dashboard result JSON: {exc}") from exc
    if not isinstance(result, dict):
        raise RuntimeError("Training script emitted a non-object dashboard result.")
    return result


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


def _safe_evaluation(model_path: str) -> dict[str, Any]:
    if not model_path:
        message = "Training finished but no model path was returned, so evaluation artifacts could not be located."
        print(f"[training] evaluation unavailable: {message}")
        return {"available": False, "message": message, "model_path": None}
    try:
        summary = summarize_model_evaluation(model_path)
        report_paths = write_model_evaluation_report(summary)
        if report_paths:
            summary["report_paths"] = report_paths
            print(f"[training] evaluation report written: {report_paths}")
        return summary
    except Exception as exc:
        print(f"[training] evaluation summary failed: {type(exc).__name__}: {exc}")
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
    completed = _run_logged_subprocess(
        command,
        "prediction",
        stdout_label="stdout",
        stderr_label="stderr",
    )
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
        if request.flaresolverr:
            command.append("--flaresolverr")
    if request.use_calibrated:
        command.append("--use-calibrated")
    if not request.shap:
        command.append("--no-shap")


def _validate_american_odds_value(value: int | None, label: str) -> None:
    if value is None:
        return
    if abs(value) < 100:
        raise ValueError(f"{label} must be American odds at +100 or longer, or -100 or shorter.")


def _validate_manual_odds_mapping(manual_odds: dict[str, int] | None) -> None:
    if not manual_odds:
        return
    for fighter_name, value in manual_odds.items():
        name = str(fighter_name).strip()
        if not name:
            raise ValueError("Manual odds fighter names cannot be blank.")
        _validate_american_odds_value(value, f"Manual odds for {name}")


def _require_prediction_model_available(model_type: str, model_path: str | None) -> None:
    try:
        assert_prediction_runtime_dependencies()
    except RuntimeError as exc:
        raise ValueError(str(exc)) from exc

    if model_path:
        resolve_model_dir(model_path)
        return
    if list_models(model_type):
        return
    raise FileNotFoundError(
        f"No loadable {model_type} model found in {models_dir()}. "
        f"Run setup again or train a {model_type} model before predicting."
    )


def run_event_prediction(request: EventPredictionRequest) -> dict[str, Any]:
    validate_event_prediction_request(request)
    output_dir = resolve_data_output_dir(request.output_dir, "predictions/latest")
    command = _base_prediction_command(request.model_type, output_dir)
    command.extend(["--upcoming-number", str(request.upcoming_number)])
    _append_common_prediction_args(command, request)
    return _run_prediction_command(command, output_dir)


def validate_event_prediction_request(request: EventPredictionRequest) -> dict[str, Any]:
    if request.prediction_data_csv:
        resolve_data_csv(request.prediction_data_csv, "prediction_data.csv")
    if request.training_data_csv:
        resolve_data_csv(request.training_data_csv, "training_data.csv")
    _validate_manual_odds_mapping(request.manual_odds)
    output_dir = resolve_data_output_dir(request.output_dir, "predictions/latest")
    _require_prediction_model_available(request.model_type, request.model_path)
    return {
        "model_type": request.model_type,
        "model_path": request.model_path,
        "output_dir": str(output_dir),
        "upcoming_number": request.upcoming_number,
        "status": "ready_for_prediction",
    }


def validate_matchup_request(request: MatchupPredictionRequest) -> dict[str, Any]:
    if request.prediction_data_csv:
        resolve_data_csv(request.prediction_data_csv, "prediction_data.csv")
    if request.training_data_csv:
        resolve_data_csv(request.training_data_csv, "training_data.csv")
    _validate_american_odds_value(request.odds_fighter1, "Fighter 1 odds")
    _validate_american_odds_value(request.odds_fighter2, "Fighter 2 odds")
    _validate_manual_odds_mapping(request.manual_odds)
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

    _require_prediction_model_available(request.model_type, request.model_path)
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
