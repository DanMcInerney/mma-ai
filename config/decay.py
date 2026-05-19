"""
Time decay configuration for feature calculations.

This module provides configuration for time-decay calculations used in
features like time-decayed averages, standard deviations, and slopes.
"""

import json
import logging
import os
from math import log
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

def get_decay_half_life_years() -> float:
    """
    Get the time decay half-life in years.

    This value controls how quickly historical data is weighted down
    in time-decay calculations. A smaller value means more recent data
    is weighted more heavily.

    Priority:
    1. Environment variable DECAY_HALF_LIFE_YEARS (highest - for manual overrides)
    2. Optimized value from data/comprehensive_tuning/optimized_decay.json
    3. Default: 1.25 years (lowest)

    Returns:
        Half-life in years (e.g., 1.25 for 1.25 year half-life)
    """
    default = 2.0

    # First priority: environment variable (for manual overrides)
    env_value = os.getenv('DECAY_HALF_LIFE_YEARS')
    if env_value is not None:
        try:
            value = float(env_value)
            logger.debug(f"Using decay half-life from environment variable: {value} years")
            return value
        except ValueError:
            logger.warning(f"Invalid DECAY_HALF_LIFE_YEARS value '{env_value}', ignoring")

    # Second priority: optimized value from decay rate optimization
    try:
        project_root = Path(__file__).parent.parent
        optimized_path = project_root / 'data' / 'comprehensive_tuning' / 'optimized_decay.json'

        if optimized_path.exists():
            with open(optimized_path, 'r') as f:
                params = json.load(f)

            if 'decay_half_life_years' in params:
                optimized_value = params['decay_half_life_years']
                logger.debug(f"Using optimized decay half-life: {optimized_value} years")
                return float(optimized_value)
    except (json.JSONDecodeError, KeyError, ValueError) as e:
        logger.debug(f"Could not load optimized decay rate: {e}")

    # Third priority: default fallback
    logger.debug(f"Using default decay half-life: {default} years")
    return default


def get_decay_rate() -> float:
    """
    Get the decay rate constant for exponential decay calculations.

    This is calculated from the half-life: decay_rate = log(2) / half_life_years

    Returns:
        Decay rate constant for use in EXP(-decay_rate * time) calculations
    """
    half_life = get_decay_half_life_years()
    return log(2) / half_life


def get_decay_rate_sql_constant() -> str:
    """
    Get the SQL expression for the decay rate constant.

    This is used in SQL queries for time-decay calculations.
    The value is pre-calculated and formatted as a string for SQL.

    Returns:
        SQL expression string (e.g., "0.5545" for 1.25 year half-life)
    """
    decay_rate = get_decay_rate()
    return f"{decay_rate:.4f}"


# Export commonly used values
DECAY_HALF_LIFE_YEARS = get_decay_half_life_years()
DECAY_RATE = get_decay_rate()
DECAY_RATE_SQL = get_decay_rate_sql_constant()
