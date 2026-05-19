"""
DEPRECATED: This module has been moved to config.decay

Please update imports:
    OLD: from libs.feature_store.config import get_decay_rate
    NEW: from config.decay import get_decay_rate

This file will be removed in a future version.
"""

import warnings

# Re-export from new location for backward compatibility
from config.decay import (
    get_decay_half_life_years,
    get_decay_rate,
    get_decay_rate_sql_constant,
    DECAY_HALF_LIFE_YEARS,
    DECAY_RATE,
    DECAY_RATE_SQL
)

warnings.warn(
    "libs.feature_store.config is deprecated. "
    "Please import from config.decay instead.",
    DeprecationWarning,
    stacklevel=2
)

