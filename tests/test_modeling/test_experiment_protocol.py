from datetime import date, timedelta

import pytest

from libs.modeling.experiment_campaign.protocol import (
    AccessLedger,
    GateError,
    LearnedStep,
    ProtocolError,
    build_development_folds,
    initialize_gate,
    seal_candidate,
    validate_fold,
    validate_learned_step,
)


def _rows():
    rows = []
    for year in range(2019, 2027):
        for event_number, day in enumerate((8, 22), start=1):
            event_id = f"event-{year}-{event_number}"
            for fight_number in range(2):
                rows.append(
                    {
                        "fight_id": f"fight-{year}-{event_number}-{fight_number}",
                        "event_id": event_id,
                        "event_date": date(year, 1, day),
                    }
                )
    return rows


def test_frozen_outer_and_inner_folds_are_expanding_event_grouped_and_embargoed():
    folds = build_development_folds(_rows(), years=(2022, 2023, 2024, 2025), embargo_days=7)
    assert [fold.test_year for fold in folds] == [2022, 2023, 2024, 2025]
    assert [len(fold.test_ids) for fold in folds] == [4, 4, 4, 4]
    assert [len(fold.train_ids) for fold in folds] == sorted(len(fold.train_ids) for fold in folds)
    for fold in folds:
        validate_fold(fold)
        assert max(fold.train_dates) <= min(fold.test_dates) - timedelta(days=7)
        assert set(fold.train_event_ids).isdisjoint(fold.test_event_ids)
        assert set(fold.inner_train_event_ids).isdisjoint(fold.inner_validation_event_ids)
        assert max(fold.inner_train_dates) <= min(fold.inner_validation_dates) - timedelta(days=7)


def test_fold_validation_rejects_same_event_future_and_asof_crossing():
    fold = build_development_folds(_rows(), years=(2022,), embargo_days=7)[0]
    with pytest.raises(ProtocolError, match="same event"):
        validate_fold(fold._replace(train_event_ids=fold.train_event_ids + (fold.test_event_ids[0],)))
    with pytest.raises(ProtocolError, match="embargo"):
        validate_fold(fold._replace(train_dates=fold.train_dates + (min(fold.test_dates),)))


@pytest.mark.parametrize(
    "step",
    [
        LearnedStep("calibration", "outer", "chronological", True),
        LearnedStep("ensemble", "inner", "shuffled", False),
        LearnedStep("context", "inner", "chronological", False),
        LearnedStep("feature-selection", "future", "chronological", True),
    ],
)
def test_every_learned_step_is_inner_only_chronological_and_prior_only(step):
    with pytest.raises(ProtocolError):
        validate_learned_step(step)
    validate_learned_step(LearnedStep("calibration", "inner", "chronological", True))


def test_gate_fails_closed_and_data_access_is_audited_without_calling_reader(tmp_path):
    campaign = tmp_path / "campaign"
    initialize_gate(campaign, expected_family_ids=tuple(f"family-{i:02d}" for i in range(1, 11)))
    ledger = AccessLedger(campaign)
    ledger.record(
        purpose="fold-membership",
        columns=("fight_id", "event_id", "event_date"),
        min_date=date(2022, 1, 1),
        max_date=date(2026, 8, 8),
        protected_gate_labels=False,
    )
    state = ledger.gate_status()
    assert state["state"] == "closed"
    assert state["protected_access_count"] == 0

    called = False

    def forbidden_reader():
        nonlocal called
        called = True

    with pytest.raises(GateError, match="sealed"):
        ledger.read_protected_gate(forbidden_reader)
    assert called is False
    assert ledger.gate_status()["protected_access_count"] == 0


def test_gate_requires_exactly_ten_terminal_families_and_blocks_post_gate_adaptation(tmp_path):
    campaign = tmp_path / "campaign"
    families = tuple(f"family-{i:02d}" for i in range(1, 11))
    initialize_gate(campaign, expected_family_ids=families)
    with pytest.raises(GateError, match="exactly ten"):
        seal_candidate(campaign, family_ids=families[:-1], candidate_id="candidate")
    seal_candidate(campaign, family_ids=families, candidate_id="candidate")
    ledger = AccessLedger(campaign)
    assert ledger.gate_status()["candidate_id"] == "candidate"
    with pytest.raises(GateError, match="post-gate adaptation"):
        ledger.record_adaptation("changed-profile")


def test_gate_access_is_consumed_before_reader_can_fail(tmp_path):
    campaign = tmp_path / "campaign"
    families = tuple(f"family-{i:02d}" for i in range(1, 11))
    initialize_gate(campaign, expected_family_ids=families)
    seal_candidate(campaign, family_ids=families, candidate_id="candidate")
    ledger = AccessLedger(campaign)

    def failed_reader():
        raise RuntimeError("fixture failure before any label value is returned")

    with pytest.raises(RuntimeError, match="fixture failure"):
        ledger.read_protected_gate(failed_reader)
    state = ledger.gate_status()
    assert state["state"] == "open"
    assert state["protected_access_count"] == 1
    with pytest.raises(GateError, match="exactly once"):
        ledger.read_protected_gate(lambda: None)
