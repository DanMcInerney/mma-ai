"""Family 3 temporal calibration source-lineage audit and execution."""

from __future__ import annotations

from datetime import date
import math
from typing import Any, Mapping, Sequence

from ..calibration import CALIBRATION_VARIANT_IDS, fit_temporal_calibrator


class SourceLineageError(ValueError):
    """A registered prediction source is inadmissible for calibration."""

    def __init__(self, message: str, audit: Mapping[str, Any]) -> None:
        super().__init__(message)
        self.audit = dict(audit)


def audit_registered_rows(
    inner_rows: Sequence[Mapping[str, Any]],
    outer_rows: Sequence[Mapping[str, Any]],
    *,
    outer_year: int,
) -> dict[str, Any]:
    fit_ids = [str(row["fight_id"]) for row in inner_rows]
    calibration_event_ids = {str(row["event_id"]) for row in inner_rows}
    model_fit_event_ids = {
        str(event_id) for row in inner_rows for event_id in row.get("fit_event_ids", [])
    }
    outer_ids = {str(row["fight_id"]) for row in outer_rows}
    outer_event_ids = {str(row["event_id"]) for row in outer_rows}
    same_fit_rows = [
        row for row in inner_rows if str(row["event_id"]) in {str(value) for value in row.get("fit_event_ids", [])}
    ]
    audit = {
        "outer_year": outer_year,
        "calibration_fit_row_count": len(inner_rows),
        "calibration_fit_id_count": len(set(fit_ids)),
        "calibration_fit_event_count": len(calibration_event_ids),
        "model_fit_event_count": len(model_fit_event_ids),
        "outer_row_count": len(outer_rows),
        "outer_event_count": len(outer_event_ids),
        "calibration_model_fit_overlap_count": len(same_fit_rows),
        "same_fit_row_count": len(same_fit_rows),
        "calibration_outer_id_overlap_count": len(set(fit_ids).intersection(outer_ids)),
        "calibration_outer_event_overlap_count": len(calibration_event_ids.intersection(outer_event_ids)),
        "variant_fit_count": 0,
        "variant_score_count": 0,
    }

    def reject(message: str) -> None:
        audit["status"] = "ineligible"
        audit["reason"] = message
        raise SourceLineageError(message, audit)

    if not inner_rows or not outer_rows:
        reject("registered calibration or outer history is empty")
    boundaries = {row.get("boundary") for row in inner_rows}
    if len(boundaries) != 1 or not boundaries.issubset({"InnerSelection", "Original"}):
        reject("calibration history has an unsupported or shuffled fold boundary")
    if any(row.get("boundary") != "Original" for row in outer_rows):
        reject("outer history must contain Original probabilities")
    if len(fit_ids) != len(set(fit_ids)):
        reject("calibration history contains duplicate IDs")
    if set(fit_ids).intersection(outer_ids):
        reject("calibration fit IDs overlap outer IDs")
    if same_fit_rows:
        reject("calibration event IDs overlap base model-fit event IDs")
    if calibration_event_ids.intersection(outer_event_ids):
        reject("calibration event IDs overlap outer event IDs")
    probabilities = [float(row["probability"]) for row in inner_rows]
    if any(not math.isfinite(value) or not 0.0 <= value <= 1.0 for value in probabilities):
        reject("calibration probabilities are outside range [0, 1]")
    if {int(row["y_true"]) for row in inner_rows} != {0, 1}:
        reject("calibration history must contain both classes")
    fit_dates = [date.fromisoformat(str(row["event_date"])) for row in inner_rows]
    if fit_dates != sorted(fit_dates):
        reject("calibration history is shuffled rather than chronological")
    outer_dates = [date.fromisoformat(str(row["event_date"])) for row in outer_rows]
    if any(value >= min(outer_dates) for value in fit_dates):
        reject("calibration history contains future IDs")
    if any(value.year != outer_year for value in outer_dates):
        reject("outer history year does not match its declared fold")
    audit["status"] = "eligible"
    audit["reason"] = None
    return audit


def _positive_log_loss(labels: Sequence[int], probabilities: Sequence[float]) -> float:
    epsilon = 1e-15
    losses = []
    for label, probability in zip(labels, probabilities, strict=True):
        value = min(max(float(probability), epsilon), 1.0 - epsilon)
        losses.append(-(int(label) * math.log(value) + (1 - int(label)) * math.log(1.0 - value)))
    return sum(losses) / len(losses)


def _chronological_event_split(
    rows: Sequence[Mapping[str, Any]], fraction: float = 0.7
) -> tuple[list[Mapping[str, Any]], list[Mapping[str, Any]]]:
    events: list[str] = []
    for row in rows:
        event_id = str(row["event_id"])
        if not events or events[-1] != event_id:
            events.append(event_id)
    if len(events) < 2:
        raise SourceLineageError(
            "calibration history has fewer than two chronological event blocks",
            {"status": "ineligible", "event_count": len(events)},
        )
    cut = min(max(int(len(events) * fraction), 1), len(events) - 1)
    fit_events = set(events[:cut])
    fit_rows = [row for row in rows if str(row["event_id"]) in fit_events]
    score_rows = [row for row in rows if str(row["event_id"]) not in fit_events]
    return fit_rows, score_rows


def _fit_variant(
    variant_id: str,
    config: Mapping[str, Any],
    fit_rows: Sequence[Mapping[str, Any]],
    score_rows: Sequence[Mapping[str, Any]],
):
    return fit_temporal_calibrator(
        variant_id=variant_id,
        config=config,
        probabilities=[float(row["probability"]) for row in fit_rows],
        labels=[int(row["y_true"]) for row in fit_rows],
        fit_ids=[str(row["fight_id"]) for row in fit_rows],
        fit_dates=[str(row["event_date"]) for row in fit_rows],
        model_fit_ids=[],
        outer_ids=[str(row["fight_id"]) for row in score_rows],
        outer_min_date=min(str(row["event_date"]) for row in score_rows),
    )


def select_and_calibrate_outer(
    history_rows: Sequence[Mapping[str, Any]],
    outer_rows: Sequence[Mapping[str, Any]],
    *,
    outer_year: int,
    variant_configs: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    if tuple(variant_configs) != CALIBRATION_VARIANT_IDS:
        raise ValueError("variant configuration order differs from the frozen menu")
    if not history_rows:
        predictions = [
            {
                **row,
                "original_probability": row["probability"],
                "selected_calibration_variant": "identity",
                "calibration_fit_row_count": 0,
                "calibration_fit_max_date": None,
            }
            for row in outer_rows
        ]
        return {
            "selection": {
                "variant_id": "identity",
                "selection_basis": "identity-only-no-fit",
                "fit_row_count": 0,
                "score_row_count": 0,
                "selection_max_date": None,
                "variant_scores": {},
            },
            "predictions": predictions,
        }
    audit = audit_registered_rows(history_rows, outer_rows, outer_year=outer_year)
    fit_rows, score_rows = _chronological_event_split(history_rows)
    scores: dict[str, float] = {}
    selection_fits: dict[str, Any] = {}
    for variant_id in CALIBRATION_VARIANT_IDS:
        fitted = _fit_variant(variant_id, variant_configs[variant_id], fit_rows, score_rows)
        calibrated = fitted.transform([float(row["probability"]) for row in score_rows])
        scores[variant_id] = _positive_log_loss(
            [int(row["y_true"]) for row in score_rows], calibrated
        )
        selection_fits[variant_id] = fitted.fit_summary
    selected = min(CALIBRATION_VARIANT_IDS, key=lambda value: (scores[value], CALIBRATION_VARIANT_IDS.index(value)))
    final_fit = _fit_variant(selected, variant_configs[selected], history_rows, outer_rows)
    calibrated_outer = final_fit.transform([float(row["probability"]) for row in outer_rows])
    fit_max_date = max(str(row["event_date"]) for row in history_rows)
    predictions = [
        {
            **row,
            "original_probability": row["probability"],
            "probability": calibrated,
            "selected_calibration_variant": selected,
            "calibration_fit_row_count": len(history_rows),
            "calibration_fit_max_date": fit_max_date,
        }
        for row, calibrated in zip(outer_rows, calibrated_outer, strict=True)
    ]
    return {
        "selection": {
            "variant_id": selected,
            "selection_basis": "chronological-event-block-tail",
            "fit_row_count": len(history_rows),
            "selection_fit_row_count": len(fit_rows),
            "score_row_count": len(score_rows),
            "selection_max_date": max(str(row["event_date"]) for row in score_rows),
            "variant_scores": scores,
            "variant_fit_summaries": selection_fits,
            "final_fit_summary": final_fit.fit_summary,
            "lineage_audit": audit,
        },
        "predictions": predictions,
    }
