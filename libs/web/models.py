"""Pydantic request and response models for the dashboard API."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class JobResponse(BaseModel):
    job_id: str
    state: str
    message: str


class DataRefreshRequest(BaseModel):
    scrape: bool = True
    rebuild: bool = True
    force_full: bool = False
    reset_db: bool = True
    odds_features: bool = True
    odds: bool = False
    log_level: str = "INFO"


class AnalyticsRequest(BaseModel):
    question: str = Field(..., min_length=3)
    sql: str | None = None
    max_rows: int = Field(default=100, ge=1, le=1000)


class TrainingRequest(BaseModel):
    model_type: Literal["win", "decision"] = "win"
    preset: Literal["extreme", "best"] = "extreme"
    time_limit: int = Field(default=3000, ge=30, le=172800)
    split_strategy: Literal["standard", "timeseries_split", "walkforward"] = "timeseries_split"
    walkforward_n_windows: int = Field(default=4, ge=1, le=24)
    walkforward_initial_year: int = Field(default=2021, ge=1993, le=2100)
    refit_full: bool = True
    refit_all: bool = False
    use_script_defaults: bool = True

    test_size: str | None = None
    val_date: str | None = None
    start_date: str = "2014-01-01"
    num_fights: int = Field(default=2, ge=0, le=20)
    include_split_dec: bool = True
    normalize: Literal["robust", "zscore", "none"] = "robust"
    use_recency_weights: bool = True
    decay_rate: float = Field(default=0.15, ge=0.0, le=1.0)
    calculate_importance: bool = True
    feature_list: list[str] | None = None
    included_strings: list[str] | None = None
    excluded_strings: list[str] | None = None
    required_strings: list[str] | None = None
    included_model_types: list[str] | None = None


class EventPredictionRequest(BaseModel):
    model_type: Literal["win", "decision"] = "win"
    model_path: str | None = None
    prediction_data_csv: str | None = None
    training_data_csv: str | None = None
    output_dir: str | None = None
    upcoming_number: int = Field(default=1, ge=1)
    odds: bool = True
    manual_odds: dict[str, int] | None = None
    flaresolverr: bool = False
    use_calibrated: bool = False
    shap: bool = False


class MatchupPredictionRequest(BaseModel):
    model_type: Literal["win", "decision"] = "win"
    model_path: str | None = None
    prediction_data_csv: str | None = None
    training_data_csv: str | None = None
    output_dir: str | None = None
    fighter1: str
    fighter2: str
    fight_date: str | None = None
    odds_fighter1: int | None = None
    odds_fighter2: int | None = None
    odds: bool = False
    manual_odds: dict[str, int] | None = None
    flaresolverr: bool = False
    use_calibrated: bool = False
    shap: bool = False


class ApiError(BaseModel):
    detail: str


JsonDict = dict[str, Any]
