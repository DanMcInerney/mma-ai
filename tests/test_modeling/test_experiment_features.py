from __future__ import annotations

import json
from copy import deepcopy

import pytest

from libs.modeling.experiment_campaign.feature_lineage import (
    FeatureLineageError,
    build_development_safe_ids,
    decode_development_rows,
    validate_feature_lineage_rows,
)


def _valid_row() -> dict:
    return {
        "event_id": "target-event",
        "fight_id": "target-fight",
        "fighter_id": "fighter-a",
        "opponent_id": "fighter-b",
        "event_date": "2025-05-03",
        "cutoff": "2025-05-03",
        "feature_name": "submission_rate_18m",
        "value": 0.2,
        "formula_version": "beta-binomial-v1",
        "fit_scope": "prior-only",
        "numerator": 1.0,
        "denominator": 4.0,
        "effective_support": 4.0,
        "uncertainty": 0.16,
        "prior_id": "sparse-beta-1-4",
        "source_row_ids": ["source-fight"],
        "source_event_ids": ["source-event"],
        "source_dates": ["2025-04-01"],
        "artifact_sha256": "A" * 64,
    }


def test_exact_safe_partition_precedes_full_row_decode(monkeypatch) -> None:
    fold_manifest = json.loads(
        open(
            "experiments/top10_20260815/baseline/fold-manifest.json",
            encoding="utf-8",
        ).read()
    )
    safe_ids, retired_ids = build_development_safe_ids(fold_manifest)
    assert len(safe_ids) == 3_089
    assert len(retired_ids) == 178
    assert set(safe_ids).isdisjoint(retired_ids)

    decoded: list[str] = []

    def tripwire(raw: bytes, indices: tuple[int, ...]) -> list[str]:
        fight_id = raw.split(b",", 1)[0].decode()
        assert fight_id not in retired_ids
        decoded.append(fight_id)
        return [fight_id]

    monkeypatch.setattr(
        "libs.modeling.experiment_campaign.feature_lineage._decode_full_row",
        tripwire,
    )
    population = tuple(str(value) for value in fold_manifest["population_fight_ids"])
    rows = (f"{fight_id},secret\n".encode() for fight_id in population)
    result = decode_development_rows(
        rows,
        safe_ids=safe_ids,
        retired_ids=retired_ids,
        indices=(0,),
    )
    assert len(result) == len(decoded) == 3_089
    assert tuple(decoded) == safe_ids


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda row: row.update(source_dates=["2025-05-03"]), "same-event or future"),
        (lambda row: row.update(source_dates=["2025-06-01"]), "same-event or future"),
        (lambda row: row.update(source_event_ids=["target-event"]), "same-event or future"),
        (lambda row: row.update(opponent_id="fighter-a"), "identity collision"),
        (lambda row: row.pop("denominator"), "missing lineage"),
        (lambda row: row.update(denominator=0.0), "zero denominator"),
        (lambda row: row.update(fit_scope="global"), "global-fit"),
        (lambda row: row.update(prior_id="invented-prior"), "unregistered prior"),
        (lambda row: row.pop("artifact_sha256"), "missing lineage"),
    ],
)
def test_wrong_result_lineage_fixtures_fail(mutate, message: str) -> None:
    row = deepcopy(_valid_row())
    mutate(row)
    with pytest.raises(FeatureLineageError, match=message):
        validate_feature_lineage_rows(
            [row],
            registered_prior_ids={"sparse-beta-1-4"},
        )


def test_valid_sparse_lineage_exposes_support_and_uncertainty() -> None:
    result = validate_feature_lineage_rows(
        [_valid_row()],
        registered_prior_ids={"sparse-beta-1-4"},
    )
    assert result == {
        "row_count": 1,
        "feature_count": 1,
        "minimum_effective_support": 4.0,
        "maximum_uncertainty": 0.16,
    }
