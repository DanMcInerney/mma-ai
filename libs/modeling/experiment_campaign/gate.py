"""Development seal and outcome-unknown prospective recording.

The campaign's 2026 historical period is retired.  This module deliberately
has no function that can read it: final evidence is limited to the already
fixed incident metadata and development prediction identities.
"""

from __future__ import annotations

import hashlib
import json
from datetime import date, datetime
from pathlib import Path
from typing import Any, Mapping

from .hashing import canonical_json_bytes, canonical_sha256, read_json, write_canonical_json
from .protocol import AccessLedger


FINAL_CANDIDATE_ID = "family-01-weighted-v8-control"
FINAL_PREDICTION_SHA256 = "6536FEEF899FEF40E0FC7979ECE96B7653EEEB603493120D7C89D8176419CF14"
EXACT_TEN_REGISTRY_SHA256 = "A1DA8BB50D1E1685061222CCFF73F83B38E00EFB65CE1CA97B4D4E751B08A6DB"
RETIRED_GATE_ID = "historically_exposed_campaign_gate"
RETIRED_GATE_ROWS = 178
RETIRED_GATE_RANGE = ("2026-01-01", "2026-08-08")
PROSPECTIVE_AFTER = date(2026, 8, 8)
FINAL_EXPERIMENT_ID = "development-candidate-seal"

_OUTCOME_KEYS = {
    "correct",
    "label",
    "outcome",
    "result",
    "target",
    "winner",
    "y",
    "y_true",
}
_PROSPECTIVE_KEYS = {
    "fight_id",
    "event_id",
    "event_date",
    "fighter_1_id",
    "fighter_2_id",
    "probability_fighter_1",
    "candidate_prediction_sha256",
    "source_revision",
    "data_identity",
}


class FinalProtocolError(ValueError):
    """A finalization action violates the frozen development-only protocol."""


def _json_text(value: Any) -> str:
    return json.dumps(value, sort_keys=True, ensure_ascii=True).lower()


def _reject_2026_boundary_claims(value: Any) -> None:
    text = _json_text(value)
    forbidden = ("2026 holdout", "2026 untouched", "untouched 2026")
    if any(phrase in text for phrase in forbidden):
        raise FinalProtocolError("forbidden 2026 boundary claim")


def validate_candidate_seal(payload: Mapping[str, Any]) -> dict[str, Any]:
    value = dict(payload)
    candidate = value.get("candidate", {})
    if value.get("kind") != "development-candidate-seal":
        raise FinalProtocolError("candidate seal kind is not development-only")
    if value.get("registry_prefix_sha256") != EXACT_TEN_REGISTRY_SHA256:
        raise FinalProtocolError("candidate seal does not follow the exact-ten registry")
    if candidate.get("experiment_id") != FINAL_CANDIDATE_ID:
        raise FinalProtocolError("candidate is not the fixed development incumbent")
    if candidate.get("prediction_sha256") != FINAL_PREDICTION_SHA256:
        raise FinalProtocolError("candidate prediction identity changed")
    if candidate.get("boundary") != "Original" or candidate.get("row_count") != 1108:
        raise FinalProtocolError("candidate must use the 1,108-row Original boundary")
    if candidate.get("outer_years") != [2022, 2023, 2024, 2025]:
        raise FinalProtocolError("candidate years are not the fixed development folds")
    gate = value.get("gate", {})
    if gate.get("gate_id") != RETIRED_GATE_ID:
        raise FinalProtocolError("candidate seal names the wrong retired period")
    if gate.get("status") != "retired-compromised-unscored":
        raise FinalProtocolError("retired period must remain compromised and unscored")
    if gate.get("software_access_count") != 0:
        raise FinalProtocolError("candidate seal requires zero software gate access")
    if gate.get("metric") is not None:
        raise FinalProtocolError("retired period must not have a metric")
    try:
        datetime.fromisoformat(str(value["sealed_at"]))
    except (KeyError, ValueError) as exc:
        raise FinalProtocolError("candidate seal needs an ISO timestamp") from exc
    _reject_2026_boundary_claims(value)
    return value


def validate_compromise_record(payload: Mapping[str, Any]) -> dict[str, Any]:
    value = dict(payload)
    population = value.get("population", {})
    if value.get("incident_id") != "family-05-preflight-2026-label-decode":
        raise FinalProtocolError("unexpected gate-compromise incident identity")
    if value.get("gate_id") != RETIRED_GATE_ID:
        raise FinalProtocolError("incident names the wrong retired period")
    if population.get("row_count") != RETIRED_GATE_ROWS:
        raise FinalProtocolError("incident row count changed")
    if population.get("date_range") != list(RETIRED_GATE_RANGE):
        raise FinalProtocolError("incident date range changed")
    if value.get("classification") != "protocol-compromise-retired-unscored":
        raise FinalProtocolError("incident classification changed")
    if value.get("software_access_count_before") != 0 or value.get(
        "software_access_count_after"
    ) != 0:
        raise FinalProtocolError("incident record must preserve zero software ledger access")
    if value.get("prediction_identity") is not None or value.get("metric") is not None:
        raise FinalProtocolError("incident must contain neither prediction nor metric")
    _reject_2026_boundary_claims(value)
    return value


def _registry_prefix(campaign_root: Path, count: int = 11) -> str:
    lines = (campaign_root / "registry.jsonl").read_bytes().splitlines(keepends=True)
    if len(lines) < count:
        raise FinalProtocolError("registry does not contain experiment zero plus ten families")
    return hashlib.sha256(b"".join(lines[:count])).hexdigest().upper()


def validate_preseal_state(campaign_root: Path) -> dict[str, Any]:
    campaign_root = Path(campaign_root)
    lines = (campaign_root / "registry.jsonl").read_bytes().splitlines(keepends=True)
    if len(lines) != 11 or _registry_prefix(campaign_root) != EXACT_TEN_REGISTRY_SHA256:
        raise FinalProtocolError("preseal state is not the exact-ten registry")
    head = read_json(campaign_root / "registry-head.json")
    if head.get("record_count") != 11 or head.get("registry_prefix_sha256") != EXACT_TEN_REGISTRY_SHA256:
        raise FinalProtocolError("preseal registry head changed")
    gate = AccessLedger(campaign_root).gate_status()
    if gate.get("state") != "closed" or gate.get("protected_access_count") != 0:
        raise FinalProtocolError("retired period must stay closed with zero software access")
    return {"registry_prefix_sha256": EXACT_TEN_REGISTRY_SHA256, "gate": gate}


def build_candidate_seal(
    campaign_root: Path,
    *,
    sealed_at: str,
    pre_gate_code_revision: str,
) -> dict[str, Any]:
    validate_preseal_state(campaign_root)
    family_1 = read_json(campaign_root / "runs/family-01-weighted-v8-control/manifest.json")
    family_10 = read_json(campaign_root / "runs/family-10-outcome-decomposition/manifest.json")
    incumbent = family_10["development_final_incumbent_identity"]
    prediction = incumbent["candidate_prediction_identity"]
    if prediction.get("sha256") != FINAL_PREDICTION_SHA256:
        raise FinalProtocolError("family-10 decision does not derive the fixed candidate")
    baseline = read_json(campaign_root / "baseline/manifest.json")
    accepted = baseline["model_identities"]["accepted"]
    payload = {
        "kind": "development-candidate-seal",
        "sealed_at": sealed_at,
        "pre_gate_code_revision": pre_gate_code_revision,
        "registry_prefix_sha256": EXACT_TEN_REGISTRY_SHA256,
        "candidate": {
            "experiment_id": FINAL_CANDIDATE_ID,
            "boundary": "Original",
            "row_count": 1108,
            "outer_years": [2022, 2023, 2024, 2025],
            "prediction_sha256": prediction["sha256"],
            "profile_path": family_1["profile_path"],
            "profile_sha256": family_1["profile_sha256"],
            "model_source_name": accepted["source_name"],
            "model_complete_tree_sha256": accepted["complete_tree_sha256"],
            "model_native_tree_sha256": accepted["native_tree_sha256"],
            "family_artifact_tree_sha256": family_1["artifact_tree_sha256"],
            "development_metrics": incumbent["development_metrics"],
        },
        "gate": {
            "gate_id": RETIRED_GATE_ID,
            "status": "retired-compromised-unscored",
            "software_access_count": 0,
            "metric": None,
        },
    }
    return validate_candidate_seal(payload)


def write_candidate_seal(
    campaign_root: Path,
    *,
    sealed_at: str,
    pre_gate_code_revision: str,
) -> dict[str, Any]:
    campaign_root = Path(campaign_root)
    path = campaign_root / "final/candidate-seal.json"
    if path.exists():
        raise FinalProtocolError("candidate seal already exists")
    payload = build_candidate_seal(
        campaign_root, sealed_at=sealed_at, pre_gate_code_revision=pre_gate_code_revision
    )
    write_canonical_json(path, payload)
    return payload


def build_compromise_record() -> dict[str, Any]:
    return validate_compromise_record(
        {
            "incident_id": "family-05-preflight-2026-label-decode",
            "gate_id": RETIRED_GATE_ID,
            "population": {"row_count": RETIRED_GATE_ROWS, "date_range": list(RETIRED_GATE_RANGE)},
            "classification": "protocol-compromise-retired-unscored",
            "software_access_count_before": 0,
            "software_access_count_after": 0,
            "prediction_identity": None,
            "metric": None,
            "facts": [
                "A failed Family-5 preflight decoded target values before aborting.",
                "No prediction, score, persistence, printed label, or selection use occurred.",
                "The historical period is permanently retired from acceptance and confirmation.",
            ],
        }
    )


def write_compromise_record(campaign_root: Path) -> dict[str, Any]:
    campaign_root = Path(campaign_root)
    if AccessLedger(campaign_root).gate_status().get("protected_access_count") != 0:
        raise FinalProtocolError("cannot record incident after additional software gate access")
    path = campaign_root / "runs/historically-exposed-campaign-gate/incident.json"
    if path.exists():
        raise FinalProtocolError("gate-compromise incident already exists")
    payload = build_compromise_record()
    write_canonical_json(path, payload)
    return payload


def record_prospective_prediction(root: Path, record: Mapping[str, Any]) -> dict[str, Any]:
    root = Path(root)
    value = dict(record)
    lower_keys = {str(key).lower() for key in value}
    if lower_keys & _OUTCOME_KEYS:
        raise FinalProtocolError("prospective input contains an outcome")
    missing = sorted(_PROSPECTIVE_KEYS - set(value))
    extra = sorted(set(value) - _PROSPECTIVE_KEYS)
    if missing or extra:
        raise FinalProtocolError(f"prospective schema mismatch: missing={missing}, extra={extra}")
    try:
        event_date = date.fromisoformat(str(value["event_date"])[:10])
    except ValueError as exc:
        raise FinalProtocolError("prospective event date is invalid") from exc
    if event_date <= PROSPECTIVE_AFTER:
        raise FinalProtocolError("prospective event must be after 2026-08-08")
    probability = value["probability_fighter_1"]
    if isinstance(probability, bool) or not isinstance(probability, (int, float)) or not 0 <= probability <= 1:
        raise FinalProtocolError("prospective probability must be between zero and one")
    if value["candidate_prediction_sha256"] != FINAL_PREDICTION_SHA256:
        raise FinalProtocolError("prospective record uses a different candidate")
    if not value["fight_id"] or value["fighter_1_id"] == value["fighter_2_id"]:
        raise FinalProtocolError("prospective fight identity is invalid")
    enriched = {**value, "record_sha256": canonical_sha256(value)}
    path = root / "records" / f"{value['fight_id']}.json"
    if path.exists():
        raise FinalProtocolError("prospective prediction already exists")
    write_canonical_json(path, enriched)
    return enriched


def verify_prospective_seam(campaign_root: Path) -> dict[str, Any]:
    root = Path(campaign_root) / "prospective"
    fixture = read_json(root / "immutability-fixture.json")
    if fixture.get("policy") != "append-only-outcome-unknown-after-2026-08-08":
        raise FinalProtocolError("prospective immutability fixture changed")
    records = []
    for path in sorted((root / "records").glob("*.json")) if (root / "records").is_dir() else []:
        value = read_json(path)
        expected = value.pop("record_sha256", None)
        if canonical_sha256(value) != expected:
            raise FinalProtocolError("prospective record hash mismatch")
        if set(value) & _OUTCOME_KEYS:
            raise FinalProtocolError("stored prospective record contains an outcome")
        if date.fromisoformat(str(value["event_date"])[:10]) <= PROSPECTIVE_AFTER:
            raise FinalProtocolError("stored prospective record is not future-only")
        records.append(expected)
    return {
        "policy": fixture["policy"],
        "fixture_sha256": canonical_sha256(fixture),
        "record_count": len(records),
        "record_sha256s": records,
    }


def append_seal_registry_record(campaign_root: Path) -> dict[str, Any]:
    """Append the sole post-family record without relaxing the family registry API."""
    campaign_root = Path(campaign_root)
    seal_path = campaign_root / "final/candidate-seal.json"
    seal = validate_candidate_seal(read_json(seal_path))
    registry_path = campaign_root / "registry.jsonl"
    before = registry_path.read_bytes()
    lines = before.splitlines(keepends=True)
    if len(lines) != 11 or hashlib.sha256(before).hexdigest().upper() != EXACT_TEN_REGISTRY_SHA256:
        raise FinalProtocolError("seal registry append requires the exact-ten prefix")
    previous = json.loads(lines[-1])["record_sha256"]
    core = {
        "sequence": 11,
        "prefix_sha256_before": EXACT_TEN_REGISTRY_SHA256,
        "previous_record_sha256": previous,
        "payload": {
            "kind": "development-candidate-seal",
            "experiment_id": FINAL_EXPERIMENT_ID,
            "status": "complete",
            "seal_path": "final/candidate-seal.json",
            "seal_sha256": canonical_sha256(seal),
            "candidate_prediction_sha256": FINAL_PREDICTION_SHA256,
            "gate_metric": None,
        },
    }
    record = {**core, "record_sha256": canonical_sha256(core)}
    appended = before + canonical_json_bytes(record) + b"\n"
    registry_path.write_bytes(appended)
    write_canonical_json(
        campaign_root / "registry-head.json",
        {
            "record_count": 12,
            "registry_bytes": len(appended),
            "registry_prefix_sha256": hashlib.sha256(appended).hexdigest().upper(),
            "last_record_sha256": record["record_sha256"],
        },
    )
    return record


def validate_final_registry(campaign_root: Path) -> dict[str, Any]:
    campaign_root = Path(campaign_root)
    raw = (campaign_root / "registry.jsonl").read_bytes()
    lines = raw.splitlines(keepends=True)
    if len(lines) != 12 or _registry_prefix(campaign_root) != EXACT_TEN_REGISTRY_SHA256:
        raise FinalProtocolError("final registry must be exact-ten plus one seal")
    seal_record = json.loads(lines[-1])
    if seal_record["sequence"] != 11 or seal_record["payload"].get("kind") != "development-candidate-seal":
        raise FinalProtocolError("final registry tail is not the candidate seal")
    if seal_record["payload"].get("gate_metric") is not None:
        raise FinalProtocolError("final registry must not contain a gate metric")
    seal = validate_candidate_seal(read_json(campaign_root / seal_record["payload"]["seal_path"]))
    if canonical_sha256(seal) != seal_record["payload"]["seal_sha256"]:
        raise FinalProtocolError("registered candidate seal hash mismatch")
    head = read_json(campaign_root / "registry-head.json")
    actual = hashlib.sha256(raw).hexdigest().upper()
    if head.get("record_count") != 12 or head.get("registry_prefix_sha256") != actual:
        raise FinalProtocolError("final registry head mismatch")
    gate = AccessLedger(campaign_root).gate_status()
    if gate.get("state") != "closed" or gate.get("protected_access_count") != 0:
        raise FinalProtocolError("retired period must remain closed with zero software access")
    incident = validate_compromise_record(
        read_json(campaign_root / "runs/historically-exposed-campaign-gate/incident.json")
    )
    return {
        "record_count": 12,
        "exact_ten_registry_prefix_sha256": EXACT_TEN_REGISTRY_SHA256,
        "final_registry_sha256": actual,
        "seal_record_sha256": seal_record["record_sha256"],
        "candidate_prediction_sha256": FINAL_PREDICTION_SHA256,
        "incident_id": incident["incident_id"],
        "gate_state": gate["state"],
        "protected_access_count": gate["protected_access_count"],
        "gate_metric": None,
    }
