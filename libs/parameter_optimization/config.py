"""
DEPRECATED: This module has been moved to config.parameters

Please update imports:
    OLD: from libs.parameter_optimization.config import CACHE_PATH
    NEW: from config.parameters import CACHE_PATH

This file will be removed in a future version.
"""

import warnings

# Re-export from new location for backward compatibility
from config.parameters import (
    TRAINING_START_DATE,
    TRAINING_END_DATE,
    TRAINING_PERIOD,
    PROJECT_ROOT,
    CACHE_PATH,
    FIGHT_COUNT_TOLERANCE
)

warnings.warn(
    "libs.parameter_optimization.config is deprecated. "
    "Please import from config.parameters instead.",
    DeprecationWarning,
    stacklevel=2
)
