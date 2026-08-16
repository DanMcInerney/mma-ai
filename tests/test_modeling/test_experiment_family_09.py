from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path

import pandas as pd
import pytest

from libs.modeling.experiment_campaign.foundation_context import (
    ContextLeakageError,
    assert_prediction_label_invariance,
    build_prior_context,
    context_cache_key,
    validate_context_lineage,
)
from libs.modeling.experiment_campaign.families.capacity_foundation import (
    CANDIDATE_IDS,
    MENU_IDS,
    CapacityFoundationError,
    build_preregistered_profile,
    materialize_family_09,
    validate_preregistered_profile,
    write_preregistration,
)
from libs.modeling.experiment_campaign.hashing import (
    canonical_sha256,
    file_sha256,
    write_canonical_json,
)
from libs.modeling.experiment_campaign.protocol import initialize_gate
from libs.modeling.experiment_campaign.runner import (
    audit_campaign_safety,
    replay_campaign_decisions,
    validate_terminal_campaign,
    verify_family_run,
)


def _context_rows() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"fight_id": "a", "event_id": "e1", "event_date": "2023-01-01", "y_true": 1},
            {"fight_id": "b", "event_id": "e2", "event_date": "2023-02-01", "y_true": 0},
            {"fight_id": "c", "event_id": "e2", "event_date": "2023-02-01", "y_true": 1},
            {"fight_id": "d", "event_id": "e3", "event_date": "2023-03-01", "y_true": 0},
            {"fight_id": "z", "event_id": "future", "event_date": "2023-05-01", "y_true": 1},
        ]
    )


def test_exact_eight_menu_materializes_control_sentinel_and_six_candidates() -> None:
    profile = build_preregistered_profile()
    validated = validate_preregistered_profile(profile)

    assert tuple(item["id"] for item in profile["menu"]) == MENU_IDS
    assert tuple(item["id"] for item in profile["candidates"]) == CANDIDATE_IDS
    assert validated["menu_count"] == 8
    assert validated["candidate_fit_count"] == 6
    assert profile["menu"][0]["status"] == "executable-control"
    assert profile["menu"][1]["status"] == "unavailable-inconclusive-sentinel"
    assert profile["menu"][1]["negative_evidence"] is False
    assert {item["model_family"] for item in profile["candidates"]} == {
        "FASTAI",
        "MITRA",
        "TABICL",
    }
    assert {item["capacity"] for item in profile["candidates"]} == {"low", "medium"}
    assert all(item["profile_sha256"] == canonical_sha256({
        key: value for key, value in item.items() if key != "profile_sha256"
    }) for item in profile["candidates"])


def test_all_capacity_context_checkpoint_and_runtime_defaults_are_explicit() -> None:
    profile = build_preregistered_profile()
    for candidate in profile["candidates"]:
        assert set(candidate) == {
            "id",
            "model_family",
            "capacity",
            "features",
            "architecture",
            "optimization",
            "context",
            "checkpoint",
            "runtime",
            "selection",
            "profile_sha256",
        }
        assert set(candidate["architecture"]) == {
            "layers",
            "width",
            "dropout",
            "embedding_dropout",
            "ensemble_size",
        }
        assert set(candidate["optimization"]) == {
            "learning_rate",
            "weight_decay",
            "epochs",
            "early_stopping_patience",
            "batch_size",
            "seed",
        }
        assert set(candidate["context"]) == {
            "length",
            "sample",
            "ordering",
            "event_boundary",
            "evaluation_labels",
            "same_event_rows",
            "future_rows",
            "cache_mode",
        }
        assert set(candidate["checkpoint"]) == {
            "repository",
            "revision",
            "filename",
            "sha256",
            "allow_auto_download",
        }
        assert candidate["runtime"] == {
            "num_gpus": 1,
            "gpu_device": "0",
            "serialized": True,
            "refit_full": False,
            "time_limit_seconds": 2400,
        }
        assert candidate["selection"] == {
            "outer_label_selection_count": 0,
            "same_row_score_selection": False,
            "metric": "log_loss",
        }


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda p: p["menu"].append(deepcopy(p["menu"][-1])), "maximum eight"),
        (lambda p: p["candidates"].append(deepcopy(p["candidates"][-1])), "six candidate"),
        (lambda p: p["candidates"][0]["optimization"].pop("weight_decay"), "materialized"),
        (lambda p: p["candidates"][0]["runtime"].__setitem__("refit_full", True), "refit_full"),
        (lambda p: p["candidates"][2]["context"].__setitem__("evaluation_labels", "included"), "evaluation labels"),
        (lambda p: p["candidates"][4]["context"].__setitem__("same_event_rows", "included"), "same-event"),
        (lambda p: p["candidates"][3].__setitem__("model_family", "TABPFN"), "forbidden model"),
        (lambda p: p["candidates"][5]["checkpoint"].__setitem__("repository", "TabDPT/checkpoint"), "forbidden model"),
    ],
)
def test_profile_rejects_leaky_forbidden_or_unmaterialized_variants(mutate, message: str) -> None:
    profile = build_preregistered_profile()
    mutate(profile)
    with pytest.raises(CapacityFoundationError, match=message):
        validate_preregistered_profile(profile)


def test_prior_context_is_stably_ordered_and_excludes_same_event_eval_and_future() -> None:
    rows = _context_rows()
    context = build_prior_context(
        rows,
        evaluation_event_id="e3",
        evaluation_date="2023-03-01",
        evaluation_fight_ids={"d"},
        context_length=3,
        sample="most-recent-complete-events",
    )
    assert context["fight_id"].tolist() == ["a", "b", "c"]
    lineage = validate_context_lineage(
        context,
        evaluation_event_id="e3",
        evaluation_date="2023-03-01",
        evaluation_fight_ids={"d"},
    )
    assert lineage["context_row_count"] == 3
    assert lineage["max_context_date"] == "2023-02-01"

    unstable = context.iloc[::-1].reset_index(drop=True)
    with pytest.raises(ContextLeakageError, match="stable ordering"):
        validate_context_lineage(
            unstable,
            evaluation_event_id="e3",
            evaluation_date="2023-03-01",
            evaluation_fight_ids={"d"},
        )
    same_event = pd.concat([context, rows.loc[rows["fight_id"] == "d"]], ignore_index=True)
    with pytest.raises(ContextLeakageError, match="same-event"):
        validate_context_lineage(
            same_event,
            evaluation_event_id="e3",
            evaluation_date="2023-03-01",
            evaluation_fight_ids={"d"},
        )


def test_context_cache_key_covers_order_checkpoint_profile_and_never_labels() -> None:
    rows = _context_rows().iloc[:3]
    common = {
        "profile_sha256": "A" * 64,
        "checkpoint_sha256": "B" * 64,
        "feature_sha256": "C" * 64,
        "context_fight_ids": rows["fight_id"].tolist(),
        "context_event_ids": rows["event_id"].tolist(),
        "context_dates": rows["event_date"].tolist(),
    }
    key = context_cache_key(**common)
    assert key == context_cache_key(**common)
    assert key != context_cache_key(**{**common, "checkpoint_sha256": "D" * 64})
    assert key != context_cache_key(**{**common, "context_fight_ids": ["b", "a", "c"]})
    changed_labels = rows.assign(y_true=[0, 1, 0])
    assert key == context_cache_key(
        **{**common, "context_fight_ids": changed_labels["fight_id"].tolist()}
    )


def test_predictions_are_byte_identical_after_eval_label_removal_permutation_and_future_change() -> None:
    evaluation = pd.DataFrame(
        {
            "fight_id": ["x", "y"],
            "f1": [0.25, -0.5],
            "f2": [2.0, 3.0],
            "y_true": [0, 1],
        }
    )

    def predict(features: pd.DataFrame) -> bytes:
        assert list(features) == ["f1", "f2"]
        return json.dumps((features["f1"] + features["f2"]).tolist()).encode()

    evidence = assert_prediction_label_invariance(
        predict,
        evaluation,
        feature_names=("f1", "f2"),
        label_name="y_true",
        irrelevant_future_labels=pd.Series([1, 0, 1]),
    )
    assert evidence["byte_identical"] is True
    assert evidence["evaluation_label_reads"] == 0
    assert len(set(evidence["prediction_sha256s"].values())) == 1


def test_preregistration_commits_exact_menu_before_launch(tmp_path: Path) -> None:
    campaign = tmp_path / "experiments/top10_20260815"
    campaign.mkdir(parents=True)
    initialize_gate(campaign, expected_family_ids=())
    (campaign / "registry.jsonl").write_bytes(b"fixed-prefix\n")

    preregistration = write_preregistration(campaign, source_revision="before-fit")
    profile_path = campaign / "profiles/family-09-capacity-foundation.json"
    prereg_path = campaign / "runs/family-09-capacity-foundation/preregistration.json"
    assert profile_path.is_file() and prereg_path.is_file()
    assert preregistration["launch_state"] == "not-started"
    assert preregistration["candidate_fit_count"] == 6
    assert preregistration["menu_ids"] == list(MENU_IDS)
    assert preregistration["profile_file_sha256"] == file_sha256(profile_path)
    assert preregistration["gate_required_state"] == "closed-zero-access"
    with pytest.raises(ValueError, match="destinations must all be absent"):
        write_preregistration(campaign, source_revision="retry")


def test_actual_campaign_is_preregistered_and_terminally_recomputable() -> None:
    campaign = Path("experiments/top10_20260815")
    profile = json.loads(
        (campaign / "profiles/family-09-capacity-foundation.json").read_text(encoding="utf-8")
    )
    prereg = json.loads(
        (campaign / "runs/family-09-capacity-foundation/preregistration.json").read_text(
            encoding="utf-8"
        )
    )
    manifest = json.loads(
        (campaign / "runs/family-09-capacity-foundation/manifest.json").read_text(
            encoding="utf-8"
        )
    )
    assert profile == build_preregistered_profile()
    assert prereg["launch_state"] == "not-started"
    assert manifest["development_safe_population"] == {
        "asserted_before_target_decode": True,
        "development_safe_id_count": 3_089,
        "development_max_date": "2025-12-13",
        "retired_id_count": 178,
    }
    assert manifest["gate_access_count"] == 0
    assert manifest["candidate_fit_count"] <= 6
    verified = verify_family_run(campaign, "family-09-capacity-foundation", recompute_all=True)
    assert verified["status"] in {"complete", "failed"}
    assert verified["gate_access_count"] == 0
    replayed = replay_campaign_decisions(campaign, through="family-09-capacity-foundation")
    assert len(replayed["decisions"]) == 9
    safety = audit_campaign_safety(
        campaign,
        through="family-09-capacity-foundation",
        require_gate_closed=True,
    )
    assert safety["gpu_lease_count"] == 1
    assert safety["retry_count"] == 0
    terminal = validate_terminal_campaign(
        campaign,
        expect_terminal_through=9,
        require_gate_closed=True,
    )
    assert len(terminal["family_ids"]) == 9
    assert terminal["protected_gate_access_count"] == 0


def test_one_shot_materializer_refuses_a_changed_registry_prefix(tmp_path: Path) -> None:
    campaign = tmp_path / "experiments/top10_20260815"
    campaign.mkdir(parents=True)
    initialize_gate(campaign, expected_family_ids=())
    (campaign / "registry.jsonl").write_bytes(b"wrong-prefix\n")
    write_preregistration(campaign, source_revision="before-fit")
    (campaign / "registry.jsonl").write_bytes(b"changed-after-preregistration\n")
    with pytest.raises(ValueError, match="registry prefix"):
        materialize_family_09(
            campaign,
            source_revision="scorer",
            preregistration_commit="preregistered",
        )
