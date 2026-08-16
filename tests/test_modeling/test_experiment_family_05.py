from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from libs.modeling.experiment_campaign import runner as campaign_runner
from libs.modeling.experiment_campaign.families import semantic_portfolio as family_05
from libs.modeling.experiment_campaign.semantic_portfolio import (
    MEASUREMENT_GROUP_IDS,
    SemanticPortfolioError,
    select_stable_features,
    validate_preregistered_profile,
)
from libs.modeling.experiment_campaign.runner import (
    replay_campaign_decisions,
    verify_family_run,
)


SOURCE_SHA256 = "A" * 64
HEADER = ("feature_a", "feature_b", "event_date", "target")
HEADER_SHA256 = hashlib.sha256("\n".join(HEADER).encode()).hexdigest().upper()


def _profile() -> dict:
    candidates = [
        {
            "name": "feature_a",
            "semantic_id": "pace-a",
            "measurement_group": "global-striking-pace-efficiency",
            "source_file_sha256": SOURCE_SHA256,
            "source_header_sha256": HEADER_SHA256,
            "formula": "identity(feature_a)",
            "available_by": "2021-12-31",
            "domain_redundancy_rank": 1,
        },
        {
            "name": "feature_b",
            "semantic_id": "pace-b",
            "measurement_group": "global-striking-pace-efficiency",
            "source_file_sha256": SOURCE_SHA256,
            "source_header_sha256": HEADER_SHA256,
            "formula": "identity(feature_b)",
            "available_by": "2021-12-31",
            "domain_redundancy_rank": 2,
        },
    ]
    ordered_hash = hashlib.sha256(
        b'["feature_a","feature_b"]'
    ).hexdigest().upper()
    return {
        "experiment_id": "family-05-stable-semantic-portfolio",
        "frozen_source": {
            "path": "frozen/training_data.csv",
            "sha256": SOURCE_SHA256,
            "ordered_header_sha256": HEADER_SHA256,
            "cutoff": "2025-12-31",
        },
        "measurement_group_ids": list(MEASUREMENT_GROUP_IDS),
        "candidate_features": candidates,
        "measurement_profiles": [
            {
                "id": "v8-control",
                "included_groups": list(MEASUREMENT_GROUP_IDS),
                "ordered_features": ["feature_a", "feature_b"],
                "ordered_feature_sha256": ordered_hash,
            }
        ],
        "selection": {
            "evidence_role": "inner-chronological",
            "stability_threshold": 0.75,
            "drop_column_min_improvement": 0.0,
            "minimum_fold_support": 2,
            "domain_redundancy_cap": 1,
            "tie_break": "profile-order-then-feature-order",
            "combined_row_importance_role": "non-selection",
        },
    }


def test_preregistration_requires_exact_lineage_groups_and_maximum_eight_profiles() -> None:
    profile = _profile()
    validated = validate_preregistered_profile(profile, source_header=HEADER)
    assert validated["candidate_count"] == 2
    assert validated["profile_count"] == 1

    duplicate_semantic = _profile()
    duplicate_semantic["candidate_features"][1]["semantic_id"] = "pace-a"
    with pytest.raises(SemanticPortfolioError, match="duplicate semantic"):
        validate_preregistered_profile(duplicate_semantic, source_header=HEADER)

    missing_lineage = _profile()
    missing_lineage["candidate_features"][0]["name"] = "not_in_header"
    with pytest.raises(SemanticPortfolioError, match="source header"):
        validate_preregistered_profile(missing_lineage, source_header=HEADER)

    too_many = _profile()
    too_many["measurement_profiles"] *= 9
    with pytest.raises(SemanticPortfolioError, match="maximum eight"):
        validate_preregistered_profile(too_many, source_header=HEADER)


def test_stability_selection_is_inner_only_directional_and_redundancy_capped() -> None:
    profile = _profile()
    evidence = [
        {"feature": "feature_a", "fold": 2019, "direction": 1, "drop_column_delta": 0.02, "role": "inner-chronological"},
        {"feature": "feature_a", "fold": 2020, "direction": 1, "drop_column_delta": 0.03, "role": "inner-chronological"},
        {"feature": "feature_b", "fold": 2019, "direction": 1, "drop_column_delta": 0.01, "role": "inner-chronological"},
        {"feature": "feature_b", "fold": 2020, "direction": 1, "drop_column_delta": 0.01, "role": "inner-chronological"},
    ]
    selected = select_stable_features(evidence, profile=profile, outer_year=2022)
    assert selected["selected_features"] == ["feature_a"]
    assert selected["fit_folds"] == [2019, 2020]
    assert selected["outer_label_selection_count"] == 0
    assert selected["combined_row_importance_used"] is False

    future = evidence + [
        {"feature": "feature_a", "fold": 2022, "direction": 1, "drop_column_delta": 1.0, "role": "outer"}
    ]
    with pytest.raises(SemanticPortfolioError, match="outer or future"):
        select_stable_features(future, profile=profile, outer_year=2022)

    unstable = [dict(row) for row in evidence]
    unstable[1]["direction"] = -1
    result = select_stable_features(unstable, profile=profile, outer_year=2022)
    assert result["selected_features"] == ["feature_b"]


def test_gate_roster_is_subtracted_before_any_full_row_decode(monkeypatch) -> None:
    manifest_path = family_05.FROZEN_SOURCE.parents[3] / "baseline/fold-manifest.json"
    fold_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    population_fight_ids = tuple(
        str(fight_id) for fight_id in fold_manifest["population_fight_ids"]
    )
    gate_ids = frozenset(
        str(item["fight_id"]) for item in fold_manifest["gate_roster"]
    )
    safe_ids = tuple(
        fight_id for fight_id in population_fight_ids if fight_id not in gate_ids
    )
    assert len(safe_ids) == 3089
    assert len(gate_ids) == 178
    assert family_05._partition_development_ids(fold_manifest) == (safe_ids, gate_ids)

    decoded_ids: list[str] = []

    def decode_tripwire(raw: bytes, indices: list[int]) -> list[str]:
        fight_id = raw.split(b",", 1)[0].decode("utf-8")
        assert fight_id not in gate_ids, f"gate ID reached full-row decode: {fight_id}"
        decoded_ids.append(fight_id)
        return [fight_id, "synthetic-development-payload"]

    monkeypatch.setattr(family_05, "_decode_full_row", decode_tripwire)
    synthetic_rows = (
        f"{fight_id},synthetic-payload\n".encode("utf-8")
        for fight_id in population_fight_ids
    )
    decoded = family_05._decode_safe_rows(
        synthetic_rows,
        safe_ids=safe_ids,
        gate_ids=gate_ids,
        indices=[0, 1],
    )

    assert len(decoded) == 3089
    assert tuple(decoded_ids) == safe_ids


def test_actual_campaign_has_frozen_eight_profile_preregistration() -> None:
    campaign = Path("experiments/top10_20260815")
    profile_path = campaign / "profiles/family-05-semantic-portfolio.json"
    profile = json.loads(profile_path.read_text(encoding="utf-8"))
    source_path = Path(profile["frozen_source"]["absolute_path"])
    source_header = source_path.open(encoding="utf-8", newline="").readline().rstrip("\r\n").split(",")
    validated = validate_preregistered_profile(profile, source_header=source_header)
    assert validated["candidate_count"] == 40
    assert validated["profile_count"] == 8
    assert profile["v8_ordered_feature_sha256"] == "13E545D762A3F1BE4D023D82B8E65D77E41589031051F1F6796D742F25223022"


def test_actual_campaign_has_terminal_semantic_portfolio_result() -> None:
    campaign = Path("experiments/top10_20260815")
    result = verify_family_run(
        campaign,
        "family-05-stable-semantic-portfolio",
        recompute_all=True,
    )
    assert result["status"] == "complete"
    assert result["gate_access_count"] == 0
    assert result["profile_count"] == 8
    assert result["outer_years"] == [2022, 2023, 2024, 2025]
    assert result["outer_label_selection_count"] == 0
    assert result["combined_row_importance_used"] is False
    assert all(result["selected_features_by_fold"])


def test_ticket_named_verify_run_alias_resolves_terminal_family() -> None:
    result = verify_family_run(
        Path("experiments/top10_20260815"),
        "family-05-semantic-portfolio",
        recompute_all=False,
    )
    assert result["experiment_id"] == "family-05-stable-semantic-portfolio"
    assert result["status"] == "complete"


def test_ticket_named_replay_alias_resolves_terminal_family() -> None:
    result = replay_campaign_decisions(
        Path("experiments/top10_20260815"),
        through="family-05-semantic-portfolio",
    )
    assert len(result["decisions"]) == 5
    assert result["decisions"][-1]["experiment_id"] == (
        "family-05-stable-semantic-portfolio"
    )
    assert result["gate_access_count"] == 0


def test_joined_windows_checkout_preserves_development_table_identity() -> None:
    joined_root = Path(r"C:\Users\danhm\mma-ai\worktrees\top10-20260815")
    joined_campaign = joined_root / "experiments/top10_20260815"
    manifest = json.loads(
        (
            joined_campaign / "runs/family-05-semantic-portfolio/manifest.json"
        ).read_text(encoding="utf-8")
    )
    joined_data = (joined_root / manifest["data_path"]).read_bytes()
    assert joined_data.count(b"\r\n") == 3090

    result = verify_family_run(
        joined_campaign,
        "family-05-semantic-portfolio",
        recompute_all=False,
    )
    assert result["status"] == "complete"
    assert result["artifact_tree_sha256"] == (
        "346318577146874AFCD823516956346D53E23C87CD2BA752BF7A598610F8BC1B"
    )


def test_checkout_text_identity_normalizes_only_line_endings(tmp_path: Path) -> None:
    canonical = b"fight_id,event_date,value\n1,2025-12-13,alpha\n"
    expected = hashlib.sha256(canonical).hexdigest().upper()
    lf_path = tmp_path / "lf.csv"
    crlf_path = tmp_path / "crlf.csv"
    changed_path = tmp_path / "changed.csv"
    bare_cr_path = tmp_path / "bare-cr.csv"
    lf_path.write_bytes(canonical)
    crlf_path.write_bytes(canonical.replace(b"\n", b"\r\n"))
    changed_path.write_bytes(canonical.replace(b"alpha", b"omega"))
    bare_cr_path.write_bytes(canonical.replace(b"alpha", b"alpha\rbeta"))

    assert campaign_runner._canonical_checkout_text_sha256(lf_path) == expected
    assert campaign_runner._canonical_checkout_text_sha256(crlf_path) == expected
    assert campaign_runner._canonical_checkout_text_sha256(changed_path) != expected
    with pytest.raises(ValueError, match="bare carriage returns"):
        campaign_runner._canonical_checkout_text_sha256(bare_cr_path)
