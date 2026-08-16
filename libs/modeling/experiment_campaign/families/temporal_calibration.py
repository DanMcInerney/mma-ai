"""Family 3 temporal calibration source-lineage audit and execution."""

from __future__ import annotations

from datetime import date
import math
from typing import Any, Mapping, Sequence


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
    overlap = calibration_event_ids.intersection(model_fit_event_ids)
    audit = {
        "outer_year": outer_year,
        "calibration_fit_row_count": len(inner_rows),
        "calibration_fit_id_count": len(set(fit_ids)),
        "calibration_fit_event_count": len(calibration_event_ids),
        "model_fit_event_count": len(model_fit_event_ids),
        "outer_row_count": len(outer_rows),
        "outer_event_count": len(outer_event_ids),
        "calibration_model_fit_overlap_count": len(overlap),
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
    if any(row.get("boundary") != "InnerSelection" for row in inner_rows):
        reject("calibration history has an unsupported or shuffled fold boundary")
    if any(row.get("boundary") != "Original" for row in outer_rows):
        reject("outer history must contain Original probabilities")
    if len(fit_ids) != len(set(fit_ids)):
        reject("calibration history contains duplicate IDs")
    if set(fit_ids).intersection(outer_ids):
        reject("calibration fit IDs overlap outer IDs")
    if overlap:
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
