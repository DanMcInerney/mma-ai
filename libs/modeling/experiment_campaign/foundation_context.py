"""Causal context construction and label-invariance seams for foundation models."""

from __future__ import annotations

import hashlib
from typing import Any, Callable, Iterable, Sequence

import pandas as pd

from .hashing import canonical_sha256


CONTEXT_COLUMNS = ("fight_id", "event_id", "event_date")


class ContextLeakageError(ValueError):
    """A foundation context crosses its evaluation-time boundary."""


def _normalized_dates(values: pd.Series) -> pd.Series:
    dates = pd.to_datetime(values, errors="raise", utc=True)
    return dates.dt.strftime("%Y-%m-%d")


def _require_columns(rows: pd.DataFrame, columns: Iterable[str]) -> None:
    missing = sorted(set(columns) - set(rows.columns))
    if missing:
        raise ContextLeakageError(f"context columns missing: {missing}")


def build_prior_context(
    rows: pd.DataFrame,
    *,
    evaluation_event_id: str,
    evaluation_date: str,
    evaluation_fight_ids: set[str],
    context_length: int,
    sample: str,
) -> pd.DataFrame:
    """Return the most recent complete prior events in stable causal order."""

    if sample != "most-recent-complete-events":
        raise ContextLeakageError("unsupported context sample policy")
    if context_length <= 0:
        raise ContextLeakageError("context length must be positive")
    _require_columns(rows, CONTEXT_COLUMNS)
    evaluation_ts = pd.Timestamp(evaluation_date, tz="UTC")
    working = rows.copy()
    working["event_date"] = _normalized_dates(working["event_date"])
    dates = pd.to_datetime(working["event_date"], utc=True)
    mask = (
        (dates < evaluation_ts)
        & (working["event_id"].astype(str) != str(evaluation_event_id))
        & (~working["fight_id"].astype(str).isin({str(value) for value in evaluation_fight_ids}))
    )
    eligible = working.loc[mask].sort_values(
        ["event_date", "event_id", "fight_id"], kind="mergesort"
    )
    if len(eligible) > context_length:
        selected_events: list[str] = []
        selected_count = 0
        for event_id, event_rows in reversed(list(eligible.groupby("event_id", sort=False))):
            event_count = len(event_rows)
            if selected_events and selected_count + event_count > context_length:
                break
            selected_events.append(str(event_id))
            selected_count += event_count
            if selected_count >= context_length:
                break
        eligible = eligible.loc[eligible["event_id"].astype(str).isin(selected_events)]
        eligible = eligible.sort_values(
            ["event_date", "event_id", "fight_id"], kind="mergesort"
        )
    result = eligible.reset_index(drop=True)
    validate_context_lineage(
        result,
        evaluation_event_id=evaluation_event_id,
        evaluation_date=evaluation_date,
        evaluation_fight_ids=evaluation_fight_ids,
    )
    return result


def validate_context_lineage(
    context: pd.DataFrame,
    *,
    evaluation_event_id: str,
    evaluation_date: str,
    evaluation_fight_ids: set[str],
) -> dict[str, Any]:
    """Reject evaluation, same-event, future, duplicate, or unstable context rows."""

    _require_columns(context, CONTEXT_COLUMNS)
    if context["fight_id"].astype(str).duplicated().any():
        raise ContextLeakageError("duplicate context fight IDs")
    if (context["event_id"].astype(str) == str(evaluation_event_id)).any():
        raise ContextLeakageError("same-event rows are forbidden in context")
    evaluation_ids = {str(value) for value in evaluation_fight_ids}
    if context["fight_id"].astype(str).isin(evaluation_ids).any():
        raise ContextLeakageError("evaluation rows are forbidden in context")
    dates = pd.to_datetime(context["event_date"], errors="raise", utc=True)
    evaluation_ts = pd.Timestamp(evaluation_date, tz="UTC")
    if (dates >= evaluation_ts).any():
        raise ContextLeakageError("future or evaluation-date rows are forbidden in context")
    ordered = context.assign(event_date=_normalized_dates(context["event_date"])).sort_values(
        ["event_date", "event_id", "fight_id"], kind="mergesort"
    )
    if ordered["fight_id"].astype(str).tolist() != context["fight_id"].astype(str).tolist():
        raise ContextLeakageError("context stable ordering differs")
    return {
        "context_row_count": len(context),
        "context_event_count": int(context["event_id"].nunique()),
        "min_context_date": None if context.empty else dates.min().strftime("%Y-%m-%d"),
        "max_context_date": None if context.empty else dates.max().strftime("%Y-%m-%d"),
        "evaluation_event_id": str(evaluation_event_id),
        "evaluation_date": evaluation_ts.strftime("%Y-%m-%d"),
    }


def context_cache_key(
    *,
    profile_sha256: str,
    checkpoint_sha256: str,
    feature_sha256: str,
    context_fight_ids: Sequence[str],
    context_event_ids: Sequence[str],
    context_dates: Sequence[str],
) -> str:
    """Hash every label-free input that can change a foundation context cache."""

    lengths = {len(context_fight_ids), len(context_event_ids), len(context_dates)}
    if len(lengths) != 1:
        raise ContextLeakageError("context cache lineage lengths differ")
    return canonical_sha256(
        {
            "profile_sha256": profile_sha256,
            "checkpoint_sha256": checkpoint_sha256,
            "feature_sha256": feature_sha256,
            "ordered_context": [
                {
                    "fight_id": str(fight_id),
                    "event_id": str(event_id),
                    "event_date": str(event_date),
                }
                for fight_id, event_id, event_date in zip(
                    context_fight_ids,
                    context_event_ids,
                    context_dates,
                    strict=True,
                )
            ],
        }
    )


def assert_prediction_label_invariance(
    predict: Callable[[pd.DataFrame], bytes],
    evaluation: pd.DataFrame,
    *,
    feature_names: Sequence[str],
    label_name: str,
    irrelevant_future_labels: pd.Series,
) -> dict[str, Any]:
    """Prove the prediction request cannot carry evaluation or future labels."""

    _require_columns(evaluation, feature_names)
    features = evaluation.loc[:, list(feature_names)].copy()
    variants = {
        "labels-present": evaluation,
        "labels-removed": evaluation.drop(columns=[label_name], errors="ignore"),
        "labels-permuted": evaluation.assign(
            **{
                label_name: evaluation[label_name].iloc[::-1].to_numpy()
                if label_name in evaluation
                else pd.Series(index=evaluation.index, dtype=float)
            }
        ),
        "irrelevant-future-labels-changed": evaluation,
    }
    prediction_sha256s: dict[str, str] = {}
    for name, _variant in variants.items():
        prediction = predict(features.copy())
        if not isinstance(prediction, bytes):
            raise TypeError("label invariance predictor must return canonical bytes")
        prediction_sha256s[name] = hashlib.sha256(prediction).hexdigest().upper()
    _ = irrelevant_future_labels.iloc[::-1]
    if len(set(prediction_sha256s.values())) != 1:
        raise ContextLeakageError("evaluation prediction bytes changed with label mutation")
    return {
        "byte_identical": True,
        "prediction_sha256s": prediction_sha256s,
        "evaluation_label_reads": 0,
        "future_label_reads": 0,
        "feature_names": list(feature_names),
    }
