"""FastAPI application for the MMA AI dashboard."""

from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncIterator

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from libs.paths import data_dir
from libs.web.analytics import run_analytics
from libs.web.analytics_prompt import ANALYTICS_AGENT_SYSTEM_PROMPT_VERSION, analytics_system_prompt
from libs.web.evaluations import summarize_model_evaluation
from libs.web.jobs import JobManager
from libs.web.models import (
    AnalyticsRequest,
    DataRefreshRequest,
    EventPredictionRequest,
    JobResponse,
    MatchupPredictionRequest,
    TrainingRequest,
)
from libs.web.services import (
    get_analytics_status,
    get_dashboard_defaults,
    get_data_status,
    get_readiness_status,
    list_fighters,
    list_models,
    list_upcoming_events,
    run_data_refresh,
    run_event_prediction,
    run_matchup_prediction,
    run_training,
    validate_event_prediction_request,
    validate_matchup_request,
    warm_up_upcoming_events,
)


STATIC_DIR = Path(__file__).resolve().parent / "static"
jobs = JobManager(log_dir_factory=lambda: data_dir() / "logs" / "jobs")


def _plotly_bundle_path() -> Path:
    import plotly

    path = Path(plotly.__file__).resolve().parent / "package_data" / "plotly.min.js"
    if not path.exists():
        raise FileNotFoundError(f"Plotly JavaScript bundle not found: {path}")
    return path


def create_app() -> FastAPI:
    @asynccontextmanager
    async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
        warm_up_upcoming_events()
        yield

    app = FastAPI(title="MMA AI", version="0.1.0", lifespan=lifespan)
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

    @app.get("/")
    def index() -> FileResponse:
        return FileResponse(STATIC_DIR / "index.html")

    @app.get("/vendor/plotly.min.js")
    def plotly_bundle() -> FileResponse:
        try:
            return FileResponse(_plotly_bundle_path(), media_type="application/javascript")
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @app.get("/api/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/api/readiness")
    def readiness() -> dict:
        payload = get_readiness_status()
        if not payload["ready"]:
            raise HTTPException(status_code=503, detail=payload)
        return payload

    @app.get("/api/defaults")
    def defaults() -> dict:
        return get_dashboard_defaults()

    @app.get("/api/data/status")
    def data_status() -> dict:
        return get_data_status()

    @app.post("/api/data/refresh", response_model=JobResponse)
    def data_refresh(request: DataRefreshRequest) -> JobResponse:
        job = jobs.start("data_refresh", lambda: run_data_refresh(request))
        return JobResponse(job_id=job.id, state=job.state.value, message=job.message)

    @app.post("/api/data/analytics")
    def analytics(request: AnalyticsRequest) -> dict:
        try:
            return run_analytics(request.question, request.sql, request.max_rows)
        except (RuntimeError, ValueError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.get("/api/data/analytics/status")
    def analytics_status() -> dict:
        return get_analytics_status()

    @app.get("/api/data/analytics/system-prompt")
    def analytics_prompt() -> dict:
        return {"version": ANALYTICS_AGENT_SYSTEM_PROMPT_VERSION, "system_prompt": analytics_system_prompt()}

    @app.get("/api/train/defaults")
    def train_defaults() -> dict:
        return get_dashboard_defaults()["train"]

    @app.get("/api/train/evaluations")
    def train_evaluations(model_path: str | None = None) -> dict:
        try:
            return summarize_model_evaluation(model_path)
        except (FileNotFoundError, ValueError) as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @app.post("/api/train", response_model=JobResponse)
    def train(request: TrainingRequest) -> JobResponse:
        job = jobs.start("training", lambda: run_training(request))
        return JobResponse(job_id=job.id, state=job.state.value, message=job.message)

    @app.get("/api/predict/models")
    def models(model_type: str | None = None) -> dict:
        if model_type not in {None, "win", "decision"}:
            raise HTTPException(status_code=400, detail="model_type must be win or decision.")
        return {"models": list_models(model_type)}

    @app.get("/api/predict/fighters")
    def fighters(prediction_data_csv: str | None = None) -> dict:
        try:
            return {"fighters": list_fighters(prediction_data_csv)}
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.get("/api/predict/upcoming")
    def upcoming(prediction_data_csv: str | None = None, limit: int | None = None) -> dict:
        try:
            return list_upcoming_events(prediction_data_csv, limit)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/api/predict/event", response_model=JobResponse)
    def predict_event(request: EventPredictionRequest) -> JobResponse:
        try:
            validate_event_prediction_request(request)
            job = jobs.start("event_prediction", lambda: run_event_prediction(request))
        except (FileNotFoundError, ValueError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return JobResponse(job_id=job.id, state=job.state.value, message=job.message)

    @app.post("/api/predict/matchup", response_model=JobResponse)
    def predict_matchup(request: MatchupPredictionRequest) -> JobResponse:
        try:
            validate_matchup_request(request)
            job = jobs.start("matchup_prediction", lambda: run_matchup_prediction(request))
        except (FileNotFoundError, ValueError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return JobResponse(job_id=job.id, state=job.state.value, message=job.message)

    @app.get("/api/jobs")
    def list_jobs() -> dict:
        return {"jobs": [job.as_dict() for job in jobs.list()]}

    @app.get("/api/jobs/{job_id}")
    def get_job(job_id: str) -> dict:
        job = jobs.get(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="Job not found")
        return job.as_dict()

    @app.get("/api/jobs/{job_id}/log")
    def get_job_log(job_id: str) -> dict:
        job = jobs.get(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="Job not found")
        return {"job_id": job_id, "log_path": job.log_path, "log": jobs.read_log(job_id)}

    return app


app = create_app()
