"""Named, reusable training profiles for accepted model configurations."""

from types import MappingProxyType
from typing import Mapping

from libs.feature_store.features import vEight_testing
from libs.modeling.train import ModelTrainer, TrainingConfig


WIN_V8_HYBRID_WORKING_PROFILE_NAME = "v8-hybrid-weighted"
WIN_V8_HYBRID_NO_RECENCY_PROFILE_NAME = "v8-hybrid-no-recency"

_V8_FEATURES = tuple(vEight_testing)

WIN_V8_HYBRID_WORKING_PROFILE: Mapping[str, object] = MappingProxyType(
    {
        "model_type": "win",
        "preset": "hybrid",
        "time_limit": 3000,
        "test_size": None,
        "val_date": None,
        "features": _V8_FEATURES,
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
)

WIN_V8_HYBRID_NO_RECENCY_PROFILE: Mapping[str, object] = MappingProxyType(
    {
        "model_type": "win",
        "preset": "hybrid",
        "time_limit": 3000,
        "test_size": None,
        "val_date": None,
        "features": _V8_FEATURES,
        "included_strings": None,
        "excluded_strings": None,
        "required_strings": None,
        "start_date": "2014-01-01",
        "num_fights": 2,
        "include_split_dec": True,
        "normalize": "robust",
        "use_recency_weights": False,
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
)

_TRAINING_PROFILES: Mapping[str, Mapping[str, object]] = MappingProxyType(
    {
        WIN_V8_HYBRID_WORKING_PROFILE_NAME: WIN_V8_HYBRID_WORKING_PROFILE,
        WIN_V8_HYBRID_NO_RECENCY_PROFILE_NAME: WIN_V8_HYBRID_NO_RECENCY_PROFILE,
    }
)


def get_training_profile(name: str) -> TrainingConfig:
    """Build a fresh ``TrainingConfig`` for a named immutable profile."""
    try:
        profile = _TRAINING_PROFILES[name]
    except KeyError as exc:
        available = ", ".join(sorted(_TRAINING_PROFILES))
        raise ValueError(f"Unknown training profile {name!r}; choose from: {available}") from exc

    values = dict(profile)
    values["features"] = list(profile["features"])
    return TrainingConfig(**values)


def train_profile(name: str):
    """Train exactly once using a fresh config from a named profile."""
    return ModelTrainer(get_training_profile(name)).train()
