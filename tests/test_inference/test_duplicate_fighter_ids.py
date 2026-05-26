import builtins
from datetime import datetime

import pandas as pd
import pytest

from libs.feature_store.create_inference_data import CreateInferenceData
from libs.feature_store.inference.loaders.base import select_duplicate_fighter_id
from libs.feature_store.inference.loaders.static_loader import StaticDataLoader


def _duplicate_id_rows() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "fighter_id": "old-id",
                "fighter_name": "duplicate fighter",
                "fighter_dob": datetime(1990, 1, 1),
                "event_id": "old-event",
                "event_date": "2020-01-01",
                "age": 30,
                "reach": 70,
            },
            {
                "fighter_id": "new-id",
                "fighter_name": "duplicate fighter",
                "fighter_dob": datetime(1990, 1, 1),
                "event_id": "new-event-1",
                "event_date": "2024-01-01",
                "age": 34,
                "reach": 71,
            },
            {
                "fighter_id": "new-id",
                "fighter_name": "duplicate fighter",
                "fighter_dob": datetime(1990, 1, 1),
                "event_id": "new-event-2",
                "event_date": "2025-01-01",
                "age": 35,
                "reach": 71,
            },
            {
                "fighter_id": "opponent-id",
                "fighter_name": "opponent",
                "fighter_dob": datetime(1991, 1, 1),
                "event_id": "opp-event-1",
                "event_date": "2024-02-01",
                "age": 33,
                "reach": 72,
            },
            {
                "fighter_id": "opponent-id",
                "fighter_name": "opponent",
                "fighter_dob": datetime(1991, 1, 1),
                "event_id": "opp-event-2",
                "event_date": "2025-02-01",
                "age": 34,
                "reach": 72,
            },
        ]
    )


@pytest.fixture
def forbid_input(monkeypatch):
    def fail_input(prompt=""):
        raise AssertionError(f"Unexpected interactive prompt: {prompt}")

    monkeypatch.setattr(builtins, "input", fail_input)


def test_select_duplicate_fighter_id_prefers_most_recent_history():
    df = _duplicate_id_rows()

    selected = select_duplicate_fighter_id(df[df["fighter_name"] == "duplicate fighter"], "duplicate fighter")

    assert selected == "new-id"


def test_static_loader_duplicate_fighter_ids_do_not_prompt(forbid_input):
    loader = StaticDataLoader(
        _duplicate_id_rows(),
        [("2026-01-01", "duplicate fighter", "opponent")],
    )

    fighter_data = loader.load_fighter_data("duplicate fighter")

    assert not fighter_data.empty
    assert set(fighter_data["fighter_id"]) == {"new-id"}


def test_legacy_create_inference_data_duplicate_fighter_ids_do_not_prompt(forbid_input, tmp_path):
    csv_path = tmp_path / "prediction_data.csv"
    _duplicate_id_rows().to_csv(csv_path, index=False)
    inference = CreateInferenceData(
        csv_path=str(csv_path),
        feats=["age_diff", "reach_diff"],
        fight_list=[("2026-01-01", "duplicate fighter", "opponent")],
    )

    inference.load_static_data()

    assert "duplicate fighter" in inference.static_fighter_dfs
    assert set(inference.static_fighter_dfs["duplicate fighter"]["fighter_id"]) == {"new-id"}
