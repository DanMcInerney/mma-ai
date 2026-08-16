from __future__ import annotations

import copy
import hashlib
import json
import subprocess
from dataclasses import asdict
from pathlib import Path

import pandas as pd
import pytest
from autogluon.core.metrics import get_metric
from autogluon.tabular.trainer.model_presets.presets import get_preset_models

from libs.modeling import train, training_profiles


REPO_ROOT = Path(__file__).resolve().parents[2]
MANIFEST_PATH = REPO_ROOT / "experiments/split_refit_20260816/rollback-manifest.json"
ROLLBACK_BRANCH = "codex/weighted-v8-67-baseline"
ROLLBACK_REVISION = "545441975b86caf0abb6136e099e44e6b93caf22"
ROLLBACK_TREE = "82305ddf6160338bfab8e1e8e4e6dc3b82efc7bf"
EXPECTED_RESULT_PATHS = {
    "experiments/split_refit_20260816/rollback-manifest.json",
    "tests/test_modeling/test_split_refit_rollback.py",
}
EXPECTED_FEATURES = [
    "age_dec_avg_diff",
    "age_ratio_diff",
    "reach_ratio_dec_avg_diff",
    "days_since_last_fight_dec_avg_diff",
    "sig_str_land_per_min_dec_adjperf_dec_avg_diff",
    "weightclass_encoded",
    "sig_str_land_ratio_dec_adjperf_dec_avg_diff",
    "clinch_def_dec_adjperf_dec_avg_diff",
    "sig_str_acc_dec_adjperf_dec_avg_diff",
    "head_land_ratio_dec_adjperf_dec_avg_diff",
    "head_def_dec_adjperf_dec_avg_diff",
    "head_acc_dec_adjperf_dec_avg_diff",
    "body_def_dec_adjperf_dec_avg_diff",
    "body_acc_dec_adjperf_dec_avg_diff",
    "leg_land_per_min_dec_adjperf_dec_avg_diff",
    "leg_acc_dec_adjperf_dec_avg_diff",
    "distance_land_ratio_dec_adjperf_dec_avg_diff",
    "sig_str_land_pressure_dec_adjperf_dec_avg_diff",
    "distance_acc_dec_adjperf_dec_avg_diff",
    "distance_land_per_min_dec_adjperf_dec_avg_diff",
    "body_land_ratio_dec_adjperf_dec_avg_diff",
    "clinch_land_ratio_dec_adjperf_dec_avg_diff",
    "clinch_acc_dec_adjperf_dec_avg_diff",
    "ground_acc_dec_adjperf_dec_avg_diff",
    "ground_land_per_ctrl_dec_avg_diff",
    "ko_per_sig_str_land_dec_adjperf_dec_avg_diff",
    "ko_ratio_dec_adjperf_dec_avg_diff",
    "sub_att_ratio_dec_adjperf_dec_avg_diff",
    "sub_att_dec_avg_diff",
    "sub_def_dec_adjperf_dec_avg_diff",
    "win_dec_adjperf_dec_avg_diff",
    "rev_per_ctrlopp_dec_adjperf_dec_avg_diff",
    "td_acc_dec_adjperf_dec_avg_diff",
    "td_def_dec_adjperf_dec_avg_diff",
    "ctrl_ratio_dec_adjperf_dec_avg_diff",
    "ctrl_per_min_opp_dec_avg_diff",
    "td_att_opp_dec_avg_diff",
    "td_att_rd1_opp_dec_avg_diff",
    "td_land_per_ctrl_dec_adjperf_dec_avg_diff",
    "td_per_sig_str_att_dec_adjperf_dec_avg_diff",
]
EXPECTED_PROFILE = {
    "model_type": "win",
    "preset": "hybrid",
    "time_limit": 3000,
    "test_size": None,
    "val_date": None,
    "features": EXPECTED_FEATURES,
    "included_strings": None,
    "excluded_strings": None,
    "required_strings": None,
    "start_date": "2014-01-01",
    "num_fights": 2,
    "include_split_dec": True,
    "normalize": "robust",
    "use_recency_weights": True,
    "decay_rate": 0.15,
    "calculate_importance": True,
    "included_model_types": None,
    "split_strategy": "timeseries_split",
    "walkforward_n_windows": 4,
    "walkforward_initial_year": 2021,
    "timeseries_split": None,
    "refit_all": False,
    "refit_full": True,
}
EXPECTED_CANDIDATES = [
    "CatBoost",
    "PrepLightGBM",
    "LightGBM_r8",
    "ExtraTreesGini",
    "RandomForestGini",
    "NeuralNetFastAI",
    "RealMLP_r9",
    "XGBoost",
    "Mitra",
    "TabICL",
]
EXPECTED_NESTED_METRICS = {
    "accuracy": 0.6552346570397112,
    "brier": 0.21542436485338948,
    "calibration_intercept": -0.04680187261689956,
    "calibration_slope": 1.2131088302121558,
    "correct_count": 726,
    "ece": 0.026405311636761235,
    "log_loss": 0.6195954814877112,
    "row_count": 1108,
}
EXPECTED_FOLD_METRICS = {
    "2022": {
        "accuracy": 0.6382978723404256,
        "brier": 0.21808865933232288,
        "correct_count": 180,
        "log_loss": 0.6249679373720841,
        "row_count": 282,
    },
    "2023": {
        "accuracy": 0.6454183266932271,
        "brier": 0.21402384491897009,
        "correct_count": 162,
        "log_loss": 0.6142896201666029,
        "row_count": 251,
    },
    "2024": {
        "accuracy": 0.6757679180887372,
        "brier": 0.20954389452085065,
        "correct_count": 198,
        "log_loss": 0.6065511289974385,
        "row_count": 293,
    },
    "2025": {
        "accuracy": 0.6595744680851063,
        "brier": 0.2201164828247154,
        "correct_count": 186,
        "log_loss": 0.6324987932318777,
        "row_count": 282,
    },
}


def _canonical_json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest().upper()


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(repo), *args],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _rollback_worktree(repo: Path, branch: str) -> Path:
    records = _git(repo, "worktree", "list", "--porcelain").split("\n\n")
    branch_ref = f"branch refs/heads/{branch}"
    for record in records:
        lines = record.splitlines()
        if branch_ref in lines:
            return Path(lines[0].removeprefix("worktree "))
    raise AssertionError(f"no dedicated worktree resolves {branch}")


def _resolved_candidate_names(tmp_path: Path) -> list[str]:
    config = training_profiles.get_training_profile("v8-hybrid-weighted")
    fit_kwargs = train.build_training_fit_kwargs(
        config,
        train_data=pd.DataFrame({"feature": [0, 1], "y_true": [0, 1]}),
        tuning_data=pd.DataFrame({"feature": [2], "y_true": [1]}),
    )
    resolved, _ = get_preset_models(
        path=str(tmp_path),
        problem_type="binary",
        eval_metric=get_metric("log_loss"),
        hyperparameters=copy.deepcopy(fit_kwargs["hyperparameters"]),
    )
    return [model.name for model in resolved]


def _validate_result_range(repo: Path, head: str = "HEAD") -> None:
    changed = set(
        filter(
            None,
            _git(repo, "diff", "--name-only", f"{ROLLBACK_REVISION}..{head}").splitlines(),
        )
    )
    assert changed == EXPECTED_RESULT_PATHS
    subprocess.run(
        ["git", "-C", str(repo), "diff", "--check", f"{ROLLBACK_REVISION}..{head}"],
        check=True,
        capture_output=True,
        text=True,
    )


def _validate_manifest(path: Path, tmp_path: Path, repo: Path = REPO_ROOT) -> dict:
    assert path.is_file(), "rollback manifest is absent"
    raw = path.read_text(encoding="utf-8")
    manifest = json.loads(raw)
    assert raw == _canonical_json(manifest) + "\n"
    assert set(manifest) == {
        "evaluation",
        "model_identity",
        "reproduction",
        "rollback",
        "schema_version",
        "source_identities",
    }
    assert manifest["schema_version"] == 1

    rollback = manifest["rollback"]
    assert rollback == {
        "branch": ROLLBACK_BRANCH,
        "revision": ROLLBACK_REVISION,
        "tree": ROLLBACK_TREE,
        "worktree": "C:/Users/danhm/mma-ai/worktrees/weighted-v8-67-baseline",
    }
    assert _git(repo, "rev-parse", f"refs/heads/{rollback['branch']}") == rollback["revision"]
    assert _git(repo, "rev-parse", f"{rollback['revision']}^{{tree}}") == rollback["tree"]
    rollback_worktree = _rollback_worktree(repo, rollback["branch"])
    assert rollback_worktree.resolve() == Path(rollback["worktree"]).resolve()
    assert _git(rollback_worktree, "status", "--porcelain") == ""

    reproduction = manifest["reproduction"]
    assert reproduction["named_seam"] == {
        "call": 'libs.modeling.training_profiles.train_profile("v8-hybrid-weighted")',
        "forbidden_call": "libs.modeling.train.main()",
    }
    resolved_profile = asdict(training_profiles.get_training_profile("v8-hybrid-weighted"))
    assert len(resolved_profile) == 23
    assert resolved_profile == EXPECTED_PROFILE
    assert reproduction["profile"] == {
        "canonical_sha256": "55B750C16528AC07ECF0B9E8D9AD557308F4D9087A9A5DA86E24D8A62E8684A0",
        "field_count": 23,
        "fields": EXPECTED_PROFILE,
        "name": "v8-hybrid-weighted",
    }
    assert _canonical_sha256(resolved_profile) == reproduction["profile"]["canonical_sha256"]

    accepted_profile_path = repo / reproduction["accepted_evaluation_profile"]["path"]
    accepted_profile = json.loads(accepted_profile_path.read_text(encoding="utf-8"))
    assert accepted_profile == {**EXPECTED_PROFILE, "calculate_importance": False, "refit_full": False}
    assert reproduction["accepted_evaluation_profile"] == {
        "canonical_sha256": "5667837233CBCCF5C9AF7B8D3FE3A990BF8AB3B77D72B02411849BFF8CC15E90",
        "field_count": 23,
        "fields": accepted_profile,
        "path": "experiments/top10_20260815/profiles/family-01-weighted-v8-control.json",
    }
    assert _canonical_sha256(accepted_profile) == reproduction["accepted_evaluation_profile"]["canonical_sha256"]

    assert reproduction["features"] == {
        "count": 40,
        "ordered_names": EXPECTED_FEATURES,
        "ordered_sha256": "13E545D762A3F1BE4D023D82B8E65D77E41589031051F1F6796D742F25223022",
    }
    assert resolved_profile["features"] == reproduction["features"]["ordered_names"]
    assert _canonical_sha256(resolved_profile["features"]) == reproduction["features"]["ordered_sha256"]

    inventory = reproduction["candidate_inventory"]
    assert inventory == {
        "count": 10,
        "forbidden_context_models": ["NORI", "TABDPT", "TABPFN", "REALTABPFN"],
        "names": EXPECTED_CANDIDATES,
    }
    assert _resolved_candidate_names(tmp_path) == inventory["names"]
    assert not set(inventory["forbidden_context_models"]).intersection(inventory["names"])

    baseline_manifest = json.loads(
        (repo / "experiments/top10_20260815/baseline/manifest.json").read_text(encoding="utf-8")
    )
    source_identities = manifest["source_identities"]
    assert source_identities == {
        "accepted_direct_evaluation_evidence_sha256": "6665DF5DE0A9CABEFAE52304B8ADC135F064446A9FDB3763C0833D7D09E8ED69",
        "accepted_final_reverification_sha256": "41CB2A246A0C4BE936C1295A5D3882981F9DD03472ED719FA10AB32F26D82C54",
        "accepted_training_snapshot_sha256": "A25A127D99B2F49C535FF7B1941EB548B11F875FD7F5C773C9577EA27A39EBF6",
        "features_py_sha256": "A5B46F303EEFC36BB2FDEC419D32F9F11C01AB75303C8CB3E5A4DDE692F9BD41",
        "frozen_source_csv": {
            "column_count": 2430,
            "columns_sha256": "842668655BDBE7823584C4BFDC1AFE5E53B336726B0E4B7A55124A1A0A396AC9",
            "eligible_population": 3267,
            "raw_row_count": 7704,
            "sha256": "157649B780965ECC585F18B3030199CDC0F4FE3013958FFA4095FCF665FDB1EA",
        },
        "training_profiles_py_sha256": "686949C70E9A6D208D49BC1A9D42E49598F12A93086DBF938CE25A7D38FC7C3A",
    }
    assert _file_sha256(repo / "libs/modeling/training_profiles.py") == source_identities["training_profiles_py_sha256"]
    assert _file_sha256(repo / "libs/feature_store/features.py") == source_identities["features_py_sha256"]
    assert baseline_manifest["frozen_csv"]["sha256"] == source_identities["frozen_source_csv"]["sha256"]
    assert baseline_manifest["frozen_csv"]["schema"] == {
        "column_count": source_identities["frozen_source_csv"]["column_count"],
        "columns_sha256": source_identities["frozen_source_csv"]["columns_sha256"],
        "row_count": source_identities["frozen_source_csv"]["raw_row_count"],
    }
    assert baseline_manifest["population"]["total"] == source_identities["frozen_source_csv"]["eligible_population"]
    fixed_hashes = baseline_manifest["source_inventory"]["fixed_file_hashes"]
    assert fixed_hashes["accepted_evidence/direct-evaluation.json"] == source_identities["accepted_direct_evaluation_evidence_sha256"]
    assert fixed_hashes["accepted_evidence/final-reverification.md"] == source_identities["accepted_final_reverification_sha256"]
    accepted_snapshot = next(
        item
        for item in baseline_manifest["artifact_inventory"]["files"]
        if item["path"] == "models/accepted/training_data.csv"
    )
    assert accepted_snapshot["sha256"] == source_identities["accepted_training_snapshot_sha256"]

    model_identity = manifest["model_identity"]
    assert model_identity == {
        "complete_tree_sha256": "55445E804973B96B43AB6EC86E856A37390FF4937EAC968DC01106E71A257091",
        "file_count": 56,
        "native_tree_sha256": "2B90CD505809E7624B8A8701A170BCA41220A937F6C2C24513F30C073D8D2346",
        "source_name": "ag-20260815_090928-win-hybrid",
    }
    assert baseline_manifest["model_identities"]["accepted"] == model_identity

    evaluation = manifest["evaluation"]
    assert evaluation["accepted_direct_validation"] == {
        "accuracy": 0.6717391304347826,
        "boundary": "historical 2025 direct validation",
        "correct_count": 309,
        "positive_log_loss": 0.6131854280928061,
        "row_count": 460,
    }
    assert evaluation["nested_historical"] == {
        "boundary": "historical nested whole-event folds 2022-2025",
        "fold_metrics": EXPECTED_FOLD_METRICS,
        "metrics": EXPECTED_NESTED_METRICS,
        "prediction_sha256": "6536FEEF899FEF40E0FC7979ECE96B7653EEEB603493120D7C89D8176419CF14",
    }
    family_manifest = json.loads(
        (repo / "experiments/top10_20260815/runs/family-01-weighted-v8-control/manifest.json").read_text(encoding="utf-8")
    )
    assert {
        key: family_manifest["metrics"][key] for key in EXPECTED_NESTED_METRICS
    } == evaluation["nested_historical"]["metrics"]
    assert family_manifest["metrics"]["fold_metrics"] == evaluation["nested_historical"]["fold_metrics"]
    assert baseline_manifest["model_identities"]["accepted"]["complete_tree_sha256"] == model_identity["complete_tree_sha256"]
    return manifest


def test_rollback_manifest_resolves_and_fails_closed(tmp_path: Path) -> None:
    manifest = _validate_manifest(MANIFEST_PATH, tmp_path / "models")
    _validate_result_range(REPO_ROOT)

    clone = tmp_path / "wrong-result-clone"
    subprocess.run(
        ["git", "clone", "--shared", "--quiet", str(REPO_ROOT), str(clone)],
        check=True,
        capture_output=True,
        text=True,
    )
    assert _git(clone, "rev-list", "--count", "HEAD") == _git(
        REPO_ROOT, "rev-list", "--count", "HEAD"
    )
    wrong_manifest_path = clone / MANIFEST_PATH.relative_to(REPO_ROOT)

    wrong_manifests = []
    wrong = copy.deepcopy(manifest)
    wrong["reproduction"]["named_seam"]["call"] = 'libs.modeling.training_profiles.train_profile("v7")'
    wrong_manifests.append(wrong)
    wrong = copy.deepcopy(manifest)
    wrong["reproduction"]["features"]["ordered_names"][0:2] = reversed(
        wrong["reproduction"]["features"]["ordered_names"][0:2]
    )
    wrong_manifests.append(wrong)
    wrong = copy.deepcopy(manifest)
    wrong["rollback"]["branch"] = "codex/moved-rollback"
    wrong_manifests.append(wrong)
    wrong = copy.deepcopy(manifest)
    wrong["model_identity"]["native_tree_sha256"] = "0" * 64
    wrong_manifests.append(wrong)
    wrong = copy.deepcopy(manifest)
    del wrong["evaluation"]["nested_historical"]["metrics"]["ece"]
    wrong_manifests.append(wrong)

    for wrong in wrong_manifests:
        wrong_manifest_path.write_text(_canonical_json(wrong) + "\n", encoding="utf-8")
        with pytest.raises((AssertionError, KeyError)):
            _validate_manifest(wrong_manifest_path, tmp_path / "wrong-models", repo=REPO_ROOT)

    wrong_manifest_path.write_text(_canonical_json(manifest) + "\n", encoding="utf-8")
    extra_path = clone / "beside-tree-extra.txt"
    extra_path.write_text("outside ticket scope\n", encoding="utf-8")
    _git(clone, "config", "user.name", "rollback-fixture")
    _git(clone, "config", "user.email", "rollback-fixture@example.invalid")
    _git(clone, "add", "beside-tree-extra.txt")
    _git(clone, "commit", "-m", "add out-of-scope fixture")
    with pytest.raises(AssertionError):
        _validate_result_range(clone)
