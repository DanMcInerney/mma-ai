"""Fresh reduction of out-of-time probabilities and paired uncertainty."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


class MetricBoundaryError(ValueError):
    pass


@dataclass(frozen=True)
class MetricResult:
    row_count: int
    correct_count: int
    accuracy: float
    log_loss: float
    brier: float
    calibration_intercept: float
    calibration_slope: float
    reliability: tuple[dict[str, Any], ...]
    ece: float
    fold_metrics: Mapping[str, dict[str, float]]
    subgroup_metrics: Mapping[str, dict[str, float]]

    def as_dict(self) -> dict[str, Any]:
        return {
            "row_count": self.row_count,
            "correct_count": self.correct_count,
            "accuracy": self.accuracy,
            "log_loss": self.log_loss,
            "brier": self.brier,
            "calibration_intercept": self.calibration_intercept,
            "calibration_slope": self.calibration_slope,
            "reliability": list(self.reliability),
            "ece": self.ece,
            "fold_metrics": dict(self.fold_metrics),
            "subgroup_metrics": dict(self.subgroup_metrics),
        }


def _arrays(records: Sequence[Mapping[str, Any]]) -> tuple[np.ndarray, np.ndarray]:
    if not records:
        raise ValueError("prediction records cannot be empty")
    probabilities = np.asarray([record["probability"] for record in records], dtype=float)
    labels = np.asarray([record["y_true"] for record in records], dtype=int)
    if not np.all(np.isfinite(probabilities)) or np.any((probabilities < 0) | (probabilities > 1)):
        raise ValueError("probabilities must be finite values between zero and one")
    if not np.all(np.isin(labels, (0, 1))):
        raise ValueError("labels must be binary")
    return labels, probabilities


def _simple_metrics(labels: np.ndarray, probabilities: np.ndarray) -> dict[str, float]:
    clipped = np.clip(probabilities, 1e-15, 1 - 1e-15)
    predicted = (probabilities >= 0.5).astype(int)
    return {
        "row_count": int(labels.size),
        "correct_count": int(np.sum(predicted == labels)),
        "accuracy": float(np.mean(predicted == labels)),
        "log_loss": float(-np.mean(labels * np.log(clipped) + (1 - labels) * np.log(1 - clipped))),
        "brier": float(np.mean((probabilities - labels) ** 2)),
    }


def _calibration_line(labels: np.ndarray, probabilities: np.ndarray) -> tuple[float, float]:
    clipped = np.clip(probabilities, 1e-8, 1 - 1e-8)
    logits = np.log(clipped / (1 - clipped))
    design = np.column_stack((np.ones(labels.size), logits))
    beta = np.array([0.0, 1.0], dtype=float)
    for _ in range(50):
        linear = np.clip(design @ beta, -30, 30)
        fitted = 1 / (1 + np.exp(-linear))
        weights = np.clip(fitted * (1 - fitted), 1e-8, None)
        information = design.T @ (weights[:, None] * design)
        score = design.T @ (labels - fitted)
        step = np.linalg.pinv(information) @ score
        beta = beta + step
        if float(np.max(np.abs(step))) < 1e-10:
            break
    return float(beta[0]), float(beta[1])


def _reliability(
    labels: np.ndarray,
    probabilities: np.ndarray,
    bins: int,
) -> tuple[tuple[dict[str, Any], ...], float]:
    if bins < 2:
        raise ValueError("reliability_bins must be at least two")
    edges = np.linspace(0.0, 1.0, bins + 1)
    assignments = np.minimum(np.digitize(probabilities, edges[1:-1], right=False), bins - 1)
    rows: list[dict[str, Any]] = []
    ece = 0.0
    for index in range(bins):
        mask = assignments == index
        count = int(np.sum(mask))
        if count:
            mean_probability = float(np.mean(probabilities[mask]))
            observed_rate = float(np.mean(labels[mask]))
            ece += count / labels.size * abs(mean_probability - observed_rate)
        else:
            mean_probability = None
            observed_rate = None
        rows.append(
            {
                "lower": float(edges[index]),
                "upper": float(edges[index + 1]),
                "count": count,
                "mean_probability": mean_probability,
                "observed_rate": observed_rate,
            }
        )
    return tuple(rows), float(ece)


def _group_metrics(
    records: Sequence[Mapping[str, Any]],
    labels: np.ndarray,
    probabilities: np.ndarray,
    key_values: Iterable[tuple[str, Sequence[Any]]],
) -> dict[str, dict[str, float]]:
    groups: dict[str, dict[str, float]] = {}
    for prefix, values in key_values:
        for value in dict.fromkeys(values):
            mask = np.asarray([candidate == value for candidate in values], dtype=bool)
            groups[f"{prefix}={value}"] = _simple_metrics(labels[mask], probabilities[mask])
    return groups


def reduce_predictions(
    records: Iterable[Mapping[str, Any]],
    *,
    reliability_bins: int = 10,
) -> MetricResult:
    records = tuple(records)
    for record in records:
        if record.get("boundary") != "Original":
            raise MetricBoundaryError("selection metrics require the Original out-of-time boundary")
        if record.get("fit_scope") != "prior-only":
            raise MetricBoundaryError("selection metrics require prior-only context and fit scope")
    labels, probabilities = _arrays(records)
    metrics = _simple_metrics(labels, probabilities)
    calibration_intercept, calibration_slope = _calibration_line(labels, probabilities)
    reliability, ece = _reliability(labels, probabilities, reliability_bins)
    fold_values = [record.get("fold", "unassigned") for record in records]
    fold_metrics = _group_metrics(records, labels, probabilities, (("fold", fold_values),))
    fold_metrics = {key.removeprefix("fold="): value for key, value in fold_metrics.items()}

    confidence = [
        "high" if abs(float(record["probability"]) - 0.5) >= 0.25 else "low"
        for record in records
    ]
    subgroup_metrics = _group_metrics(
        records,
        labels,
        probabilities,
        (
            ("year", [str(record["event_date"])[:4] for record in records]),
            ("weight_class", [record.get("weight_class", "unknown") for record in records]),
            ("experience", [record.get("experience", "unknown") for record in records]),
            ("outcome_type", [record.get("outcome_type", "unknown") for record in records]),
            ("confidence", confidence),
        ),
    )
    return MetricResult(
        row_count=int(metrics["row_count"]),
        correct_count=int(metrics["correct_count"]),
        accuracy=metrics["accuracy"],
        log_loss=metrics["log_loss"],
        brier=metrics["brier"],
        calibration_intercept=calibration_intercept,
        calibration_slope=calibration_slope,
        reliability=reliability,
        ece=ece,
        fold_metrics=fold_metrics,
        subgroup_metrics=subgroup_metrics,
    )


def metric_gap(train: MetricResult, outer: MetricResult) -> dict[str, float]:
    return {
        "accuracy": train.accuracy - outer.accuracy,
        "log_loss": train.log_loss - outer.log_loss,
        "brier": train.brier - outer.brier,
    }


def _aligned_records(
    candidate: Sequence[Mapping[str, Any]],
    baseline: Sequence[Mapping[str, Any]],
) -> tuple[tuple[Mapping[str, Any], Mapping[str, Any]], ...]:
    if len(candidate) != len(baseline):
        raise ValueError("paired predictions must be aligned")
    pairs = []
    for candidate_row, baseline_row in zip(candidate, baseline, strict=True):
        identity = (candidate_row.get("fight_id"), candidate_row.get("event_id"), candidate_row.get("y_true"))
        other = (baseline_row.get("fight_id"), baseline_row.get("event_id"), baseline_row.get("y_true"))
        if identity != other:
            raise ValueError("paired predictions must be aligned by fight, event, and label")
        pairs.append((candidate_row, baseline_row))
    return tuple(pairs)


def _delta(candidate: Sequence[Mapping[str, Any]], baseline: Sequence[Mapping[str, Any]]) -> dict[str, float]:
    candidate_metrics = reduce_predictions(candidate)
    baseline_metrics = reduce_predictions(baseline)
    return {
        "log_loss_delta": candidate_metrics.log_loss - baseline_metrics.log_loss,
        "brier_delta": candidate_metrics.brier - baseline_metrics.brier,
        "accuracy_delta": candidate_metrics.accuracy - baseline_metrics.accuracy,
    }


def event_block_bootstrap_delta(
    candidate: Iterable[Mapping[str, Any]],
    baseline: Iterable[Mapping[str, Any]],
    *,
    iterations: int,
    seed: int,
) -> dict[str, dict[str, float]]:
    candidate = tuple(candidate)
    baseline = tuple(baseline)
    pairs = _aligned_records(candidate, baseline)
    if iterations < 1:
        raise ValueError("bootstrap iterations must be positive")
    events = tuple(dict.fromkeys(row[0]["event_id"] for row in pairs))
    by_event = {
        event_id: tuple(pair for pair in pairs if pair[0]["event_id"] == event_id)
        for event_id in events
    }
    rng = np.random.default_rng(seed)
    samples = {name: [] for name in ("log_loss_delta", "brier_delta", "accuracy_delta")}
    for _ in range(iterations):
        selected_events = rng.choice(events, size=len(events), replace=True)
        selected_pairs = [pair for event_id in selected_events for pair in by_event[str(event_id)]]
        delta = _delta([pair[0] for pair in selected_pairs], [pair[1] for pair in selected_pairs])
        for name, value in delta.items():
            samples[name].append(value)
    estimates = _delta(candidate, baseline)
    return {
        name: {
            "estimate": estimates[name],
            "lower": float(np.quantile(values, 0.025)),
            "upper": float(np.quantile(values, 0.975)),
            "iterations": iterations,
            "seed": seed,
        }
        for name, values in samples.items()
    }
