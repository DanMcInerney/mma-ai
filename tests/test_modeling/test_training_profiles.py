from dataclasses import asdict, fields

import pytest

from libs.modeling import training_profiles
from libs.modeling.train import TrainingConfig


EXPECTED_V8_FEATURES = (
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
)

EXPECTED_WORKING_PROFILE = {
    "model_type": "win",
    "preset": "hybrid",
    "time_limit": 3000,
    "test_size": None,
    "val_date": None,
    "features": EXPECTED_V8_FEATURES,
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


def test_working_profile_explicitly_matches_every_training_config_field():
    profile = training_profiles.WIN_V8_HYBRID_WORKING_PROFILE

    assert dict(profile) == EXPECTED_WORKING_PROFILE
    assert tuple(profile) == tuple(field.name for field in fields(TrainingConfig))
    assert len(profile["features"]) == 40
    with pytest.raises(TypeError):
        profile["time_limit"] = 1
    with pytest.raises(TypeError):
        profile["features"][0] = "mutated"


def test_no_recency_profile_differs_only_at_the_weighting_switch():
    working = dict(training_profiles.WIN_V8_HYBRID_WORKING_PROFILE)
    no_recency = dict(training_profiles.WIN_V8_HYBRID_NO_RECENCY_PROFILE)
    differences = {
        key: (working[key], no_recency[key])
        for key in working
        if working[key] != no_recency[key]
    }

    assert tuple(no_recency) == tuple(field.name for field in fields(TrainingConfig))
    assert differences == {"use_recency_weights": (True, False)}
    assert no_recency["decay_rate"] == 0.15
    with pytest.raises(TypeError):
        training_profiles.WIN_V8_HYBRID_NO_RECENCY_PROFILE["decay_rate"] = 0.1


@pytest.mark.parametrize(
    ("name", "expected_profile"),
    (
        (
            training_profiles.WIN_V8_HYBRID_WORKING_PROFILE_NAME,
            EXPECTED_WORKING_PROFILE,
        ),
        (
            training_profiles.WIN_V8_HYBRID_NO_RECENCY_PROFILE_NAME,
            {**EXPECTED_WORKING_PROFILE, "use_recency_weights": False},
        ),
    ),
)
def test_profile_builder_returns_fresh_valid_config_and_feature_list(
    name, expected_profile
):
    first = training_profiles.get_training_profile(name)
    second = training_profiles.get_training_profile(name)
    expected_config = {**expected_profile, "features": list(expected_profile["features"])}

    assert isinstance(first, TrainingConfig)
    assert asdict(first) == expected_config
    assert asdict(second) == expected_config
    assert first is not second
    assert first.features is not second.features
    first.features.append("local-only-mutation")
    assert "local-only-mutation" not in second.features
    assert tuple(training_profiles.WIN_V8_HYBRID_WORKING_PROFILE["features"]) == (
        EXPECTED_V8_FEATURES
    )


def test_no_recency_training_seam_builds_once_and_trains_once(monkeypatch):
    calls = []

    class StubTrainer:
        def __init__(self, config):
            calls.append(("init", config))

        def train(self):
            calls.append(("train",))
            return "stub-predictor"

    monkeypatch.setattr(training_profiles, "ModelTrainer", StubTrainer)

    result = training_profiles.train_profile(
        training_profiles.WIN_V8_HYBRID_NO_RECENCY_PROFILE_NAME
    )

    assert result == "stub-predictor"
    assert [call[0] for call in calls] == ["init", "train"]
    assert asdict(calls[0][1]) == {
        **EXPECTED_WORKING_PROFILE,
        "features": list(EXPECTED_V8_FEATURES),
        "use_recency_weights": False,
    }
