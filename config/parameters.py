"""
Parameter optimization configuration.

This module provides configuration for the parameter optimization system,
including training periods, cache paths, and validation thresholds.
"""

from pathlib import Path

# Training period for parameter optimization (no data leakage)
TRAINING_START_DATE = '2014-01-01'
TRAINING_END_DATE = '2023-01-01'
TRAINING_PERIOD = f"{TRAINING_START_DATE} to {TRAINING_END_DATE}"

# Project root and cache paths
PROJECT_ROOT = Path(__file__).parent.parent
CACHE_PATH = PROJECT_ROOT / 'config' / 'optimized_parameters.json'

# Cache validation thresholds
FIGHT_COUNT_TOLERANCE = 100  # ±100 fights before cache invalidation

# =========================================================================
# Baseline AdjPerf Parameters
# =========================================================================
# These are the default values used if optimization hasn't run or is disabled

BASELINE_ADJPERF_PARAMS = {
    'winsorization_limits': {
        'healthy': 7.0,
        'sparse': [3.0, 5.0],  # Interpolated based on MAD
        'degenerate': 2.5
    },
    'mad_floor_percentile': 0.10,  # 10th percentile
    'quality_thresholds': {
        'mad_degenerate': 0.010,
        'mad_sparse': 0.030,
        'mode_frequency_degenerate': 0.45
    }
}
