"""Compatibility helpers for model artifacts serialized on another OS."""

from __future__ import annotations

import os
import pathlib
from contextlib import contextmanager
from typing import Iterator


def _pathlib_pickle_mapping() -> tuple[str, type[pathlib.Path]] | None:
    """Return the concrete pathlib class that should be remapped while unpickling."""
    if os.name == "nt":
        return "PosixPath", pathlib.WindowsPath
    return "WindowsPath", pathlib.PosixPath


def install_pathlib_pickle_compatibility() -> str | None:
    """
    Make pathlib pickles from the opposite OS load in this process.

    AutoGluon lazily unpickles child model artifacts during prediction, so a
    process-level compatibility install is more reliable than only wrapping the
    initial predictor load.
    """
    mapping = _pathlib_pickle_mapping()
    if mapping is None:
        return None

    source_name, replacement = mapping
    if getattr(pathlib, source_name) is not replacement:
        setattr(pathlib, source_name, replacement)
    return source_name


@contextmanager
def pathlib_pickle_compatibility() -> Iterator[None]:
    """Temporarily remap concrete pathlib classes while loading pickle artifacts."""
    mapping = _pathlib_pickle_mapping()
    if mapping is None:
        yield
        return

    source_name, replacement = mapping
    original = getattr(pathlib, source_name)
    setattr(pathlib, source_name, replacement)
    try:
        yield
    finally:
        setattr(pathlib, source_name, original)


def load_tabular_predictor(predictor_cls, path, **kwargs):
    """Load an AutoGluon TabularPredictor with cross-OS pathlib pickle support."""
    with pathlib_pickle_compatibility():
        return predictor_cls.load(path, **kwargs)


def load_joblib_artifact(path):
    """Load a joblib artifact with cross-OS pathlib pickle support."""
    import joblib

    with pathlib_pickle_compatibility():
        return joblib.load(path)
