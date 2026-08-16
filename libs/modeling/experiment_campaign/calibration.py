"""Leakage-resistant temporal probability calibration seams."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
import math
from typing import Any, Callable, Mapping, Sequence

import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression


CALIBRATION_VARIANT_IDS = (
    "identity",
    "sigmoid-platt",
    "temperature",
    "conservative-isotonic",
)


class CalibrationError(ValueError):
    """Raised when calibration history violates the frozen temporal contract."""


@dataclass(frozen=True)
class FittedCalibrator:
    variant_id: str
    fit_summary: dict[str, Any]
    _transform: Callable[[Sequence[float]], list[float]]

    def transform(self, probabilities: Sequence[float]) -> list[float]:
        values = list(probabilities)
        _validate_probabilities(values)
        return self._transform(values)


def _validate_probabilities(probabilities: Sequence[float]) -> None:
    if any(not math.isfinite(float(value)) or not 0.0 <= float(value) <= 1.0 for value in probabilities):
        raise CalibrationError("probabilities must be finite and in range [0, 1]")


def _as_date(value: date | str) -> date:
    return value if isinstance(value, date) else date.fromisoformat(value)


def _validate_history(
    *,
    probabilities: Sequence[float],
    labels: Sequence[int],
    fit_ids: Sequence[str],
    fit_dates: Sequence[date | str],
    model_fit_ids: Sequence[str],
    outer_ids: Sequence[str],
    outer_min_date: date | str,
) -> tuple[list[float], list[int]]:
    lengths = {len(probabilities), len(labels), len(fit_ids), len(fit_dates)}
    if len(lengths) != 1:
        raise CalibrationError("calibration history fields have unequal lengths")
    if not probabilities:
        raise CalibrationError("calibration history is empty")
    values = [float(value) for value in probabilities]
    targets = [int(value) for value in labels]
    _validate_probabilities(values)
    if set(targets) != {0, 1}:
        raise CalibrationError("calibration history must contain both classes")
    if len(set(fit_ids)) != len(fit_ids):
        raise CalibrationError("calibration history contains duplicate fit IDs")
    fit_id_set = set(fit_ids)
    if fit_id_set.intersection(model_fit_ids):
        raise CalibrationError("calibration fit IDs overlap model-fit IDs")
    if fit_id_set.intersection(outer_ids):
        raise CalibrationError("calibration fit IDs overlap outer IDs")
    dates = [_as_date(value) for value in fit_dates]
    if dates != sorted(dates):
        raise CalibrationError("calibration history must be chronological, not shuffled")
    outer_start = _as_date(outer_min_date)
    if any(value >= outer_start for value in dates):
        raise CalibrationError("calibration history contains future IDs")
    return values, targets


def _logit(probabilities: Sequence[float], epsilon: float) -> np.ndarray:
    clipped = np.clip(np.asarray(probabilities, dtype=float), epsilon, 1.0 - epsilon)
    return np.log(clipped / (1.0 - clipped))


def _identity(values: Sequence[float]) -> list[float]:
    return list(values)


def _fit_platt(
    probabilities: list[float], labels: list[int], config: Mapping[str, Any]
) -> tuple[Callable[[Sequence[float]], list[float]], dict[str, Any]]:
    epsilon = float(config["clipping_epsilon"])
    model = LogisticRegression(
        C=float(config["inverse_regularization_strength"]),
        max_iter=int(config["max_iterations"]),
        random_state=int(config["seed"]),
        solver="lbfgs",
    )
    model.fit(_logit(probabilities, epsilon).reshape(-1, 1), labels)

    def transform(values: Sequence[float]) -> list[float]:
        return model.predict_proba(_logit(values, epsilon).reshape(-1, 1))[:, 1].tolist()

    return transform, {
        "coefficient": float(model.coef_[0, 0]),
        "intercept": float(model.intercept_[0]),
    }


def _fit_temperature(
    probabilities: list[float], labels: list[int], config: Mapping[str, Any]
) -> tuple[Callable[[Sequence[float]], list[float]], dict[str, Any]]:
    epsilon = float(config["clipping_epsilon"])
    logits = _logit(probabilities, epsilon)
    targets = np.asarray(labels, dtype=float)
    minimum = float(config["minimum_temperature"])
    maximum = float(config["maximum_temperature"])
    step = float(config["grid_step"])
    regularization = float(config["regularization_strength"])
    count = int(round((maximum - minimum) / step))
    temperatures = [minimum + index * step for index in range(count + 1)]

    def objective(temperature: float) -> float:
        calibrated = 1.0 / (1.0 + np.exp(-(logits / temperature)))
        calibrated = np.clip(calibrated, epsilon, 1.0 - epsilon)
        loss = -np.mean(targets * np.log(calibrated) + (1.0 - targets) * np.log(1.0 - calibrated))
        return float(loss + regularization * (math.log(temperature) ** 2))

    temperature = min(temperatures, key=lambda value: (objective(value), value))

    def transform(values: Sequence[float]) -> list[float]:
        calibrated = 1.0 / (1.0 + np.exp(-(_logit(values, epsilon) / temperature)))
        return calibrated.tolist()

    return transform, {"temperature": temperature, "objective": objective(temperature)}


def _fit_isotonic(
    probabilities: list[float], labels: list[int], config: Mapping[str, Any]
) -> tuple[Callable[[Sequence[float]], list[float]], dict[str, Any]]:
    class_counts = {label: labels.count(label) for label in (0, 1)}
    checks = (
        (len(probabilities) < int(config["minimum_support"]), "minimum_support"),
        (min(class_counts.values()) < int(config["minimum_class_support"]), "minimum_class_support"),
        (
            len(set(probabilities)) < int(config["minimum_unique_probabilities"]),
            "minimum_unique_probabilities",
        ),
    )
    fallback_reason = next((reason for failed, reason in checks if failed), None)
    if fallback_reason is not None:
        if config["fallback"] != "identity":
            raise CalibrationError("unsupported conservative isotonic fallback")
        return _identity, {"fallback": "identity", "fallback_reason": fallback_reason}
    model = IsotonicRegression(y_min=0.0, y_max=1.0, increasing=True, out_of_bounds="clip")
    model.fit(probabilities, labels)
    weight = float(config["shrinkage_weight"])

    def transform(values: Sequence[float]) -> list[float]:
        raw = np.asarray(values, dtype=float)
        isotonic = model.predict(raw)
        return ((1.0 - weight) * raw + weight * isotonic).tolist()

    return transform, {"fallback": None, "fallback_reason": None}


def fit_temporal_calibrator(
    *,
    variant_id: str,
    config: Mapping[str, Any],
    probabilities: Sequence[float],
    labels: Sequence[int],
    fit_ids: Sequence[str],
    fit_dates: Sequence[date | str],
    model_fit_ids: Sequence[str],
    outer_ids: Sequence[str],
    outer_min_date: date | str,
) -> FittedCalibrator:
    if variant_id not in CALIBRATION_VARIANT_IDS:
        raise CalibrationError(f"unknown calibration variant: {variant_id}")
    values, targets = _validate_history(
        probabilities=probabilities,
        labels=labels,
        fit_ids=fit_ids,
        fit_dates=fit_dates,
        model_fit_ids=model_fit_ids,
        outer_ids=outer_ids,
        outer_min_date=outer_min_date,
    )
    parameters: dict[str, Any]
    if variant_id == "identity":
        transform, parameters = _identity, {"fallback": None, "fallback_reason": None}
    elif variant_id == "sigmoid-platt":
        transform, parameters = _fit_platt(values, targets, config)
    elif variant_id == "temperature":
        transform, parameters = _fit_temperature(values, targets, config)
    else:
        transform, parameters = _fit_isotonic(values, targets, config)
    summary = {
        "variant_id": variant_id,
        "fit_row_count": len(values),
        "fit_id_count": len(set(fit_ids)),
        **parameters,
    }
    return FittedCalibrator(variant_id, summary, transform)
