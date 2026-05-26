import os
import pathlib
import pickle

import pandas as pd
import pytest

from libs.modeling.portable_artifacts import (
    install_pathlib_pickle_compatibility,
    pathlib_pickle_compatibility,
)
from predict import get_predictions


WINDOWS_PATH_PICKLE = (
    b"\x80\x04\x953\x00\x00\x00\x00\x00\x00\x00\x8c\x07pathlib\x94"
    b"\x8c\x0bWindowsPath\x94\x93\x94\x8c\x03C:\\\x94\x8c\x06models"
    b"\x94\x8c\x02ag\x94\x87\x94R\x94."
)


def test_pathlib_pickle_compatibility_loads_windows_paths_on_posix():
    if os.name == "nt":
        pytest.skip("WindowsPath pickles already load natively on Windows.")

    with pytest.raises(NotImplementedError):
        pickle.loads(WINDOWS_PATH_PICKLE)

    with pathlib_pickle_compatibility():
        loaded = pickle.loads(WINDOWS_PATH_PICKLE)

    assert isinstance(loaded, pathlib.PosixPath)
    assert loaded.parts[-2:] == ("models", "ag")


def test_install_pathlib_pickle_compatibility_supports_lazy_prediction_unpickles():
    class LazyArtifactModel:
        def predict_proba(self, data):
            pickle.loads(WINDOWS_PATH_PICKLE)
            return pd.DataFrame({0: [0.4], 1: [0.6]}, index=data.index)

    if os.name != "nt":
        with pytest.raises(NotImplementedError):
            LazyArtifactModel().predict_proba(pd.DataFrame({"feature": [1]}))

    source_name = "PosixPath" if os.name == "nt" else "WindowsPath"
    original_path_class = getattr(pathlib, source_name)
    try:
        install_pathlib_pickle_compatibility()
        result = get_predictions(
            LazyArtifactModel(),
            calibrator=None,
            scaled_X_df=pd.DataFrame({"feature": [1]}),
            use_calibrated=False,
        )
    finally:
        setattr(pathlib, source_name, original_path_class)

    assert result.iloc[0][1] == 0.6
