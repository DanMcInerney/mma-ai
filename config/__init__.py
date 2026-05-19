"""
Centralized configuration for MMA AI project.

This module provides a single source of truth for all configuration values
used across the project.

Usage:
    from config import DECAY_HALF_LIFE_YEARS, TRAINING_START_DATE
    from config.parameters import CACHE_PATH
    from config.decay import get_decay_rate
"""

# Re-export commonly used values for convenience
from .decay import (
    DECAY_HALF_LIFE_YEARS,
    DECAY_RATE,
    DECAY_RATE_SQL,
    get_decay_half_life_years,
    get_decay_rate,
    get_decay_rate_sql_constant
)

from .parameters import (
    TRAINING_START_DATE,
    TRAINING_END_DATE,
    TRAINING_PERIOD,
    CACHE_PATH,
    PROJECT_ROOT,
    FIGHT_COUNT_TOLERANCE,
    BASELINE_ADJPERF_PARAMS
)

__all__ = [
    # Decay configuration
    'DECAY_HALF_LIFE_YEARS',
    'DECAY_RATE',
    'DECAY_RATE_SQL',
    'get_decay_half_life_years',
    'get_decay_rate',
    'get_decay_rate_sql_constant',

    # Parameter optimization configuration
    'TRAINING_START_DATE',
    'TRAINING_END_DATE',
    'TRAINING_PERIOD',
    'CACHE_PATH',
    'PROJECT_ROOT',
    'FIGHT_COUNT_TOLERANCE',
]
