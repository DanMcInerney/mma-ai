"""
Integration helpers for main.py pipeline.

Provides functions to check cache validity, run optimization, and create parameter loaders.
"""

import os
import logging
from typing import Tuple

from ..storage.json_store import JSONParameterStore
from ..storage.cache_manager import CacheManager
from ..loaders.parameter_loader import ParameterLoader

logger = logging.getLogger(__name__)


def should_run_optimization(conn) -> Tuple[bool, str]:
    """
    Check if parameter optimization needs to run.

    Args:
        conn: Database connection for cache validation

    Returns:
        Tuple of (should_run, reason)
        - should_run: True if optimization should run
        - reason: String explaining the decision
    """
    store = JSONParameterStore()
    cache_mgr = CacheManager(store, conn)

    # is_cache_valid returns (is_valid, reason)
    # We invert it because we need (should_run, reason)
    is_valid, reason = cache_mgr.is_cache_valid()

    return (not is_valid, reason)


def run_parameter_optimization(conn):
    """
    Run smoothing parameter optimization.

    Optimizes tau values for:
    - Beta-binomial smoothing (binary outcomes: ko, win, decision, etc.)
    - Poisson-gamma smoothing (count stats: sig_str, td, sub_att, etc.)
    - Time-series cross-validation

    Results are saved to config/optimized_parameters.json

    Args:
        conn: Database connection (passed through but tuners use their own connections)

    Raises:
        RuntimeError: If optimization fails
    """
    try:
        logger.info("="*60)
        logger.info("STARTING SMOOTHING PARAMETER OPTIMIZATION")
        logger.info("="*60)

        # Run smoothing optimization
        logger.info("\nOptimizing smoothing parameters (tau values)...")
        from tuning.comprehensive_likelihood_tuner import main as run_smoothing_tuner
        run_smoothing_tuner()
        logger.info("✓ Smoothing optimization complete")

        logger.info("\n" + "="*60)
        logger.info("SMOOTHING PARAMETER OPTIMIZATION COMPLETE")
        logger.info("="*60)

    except ImportError as e:
        raise RuntimeError(
            f"Could not import optimization modules: {e}. "
            "Make sure tuning/ modules are available."
        )
    except Exception as e:
        raise RuntimeError(f"Parameter optimization failed: {e}")


def get_parameter_mode() -> str:
    """
    Get parameter mode from environment variable.

    Returns:
        'baseline' or 'optimized' (default: 'optimized')

    Raises:
        ValueError: If PARAM_MODE has invalid value
    """
    mode = os.getenv('PARAM_MODE', 'optimized').lower()

    if mode not in ('baseline', 'optimized'):
        raise ValueError(
            f"Invalid PARAM_MODE='{mode}'. Must be 'baseline' or 'optimized'"
        )

    return mode


def get_default_parameter_loader() -> ParameterLoader:
    """
    Create a parameter loader with default settings.

    Uses:
    - Mode from PARAM_MODE environment variable (default: 'optimized')
    - Default JSONParameterStore

    Returns:
        ParameterLoader instance

    Raises:
        RuntimeError: If mode is 'optimized' but no parameters exist
        ValueError: If PARAM_MODE has invalid value
    """
    mode = get_parameter_mode()
    store = JSONParameterStore()

    logger.info(f"Creating parameter loader with mode='{mode}'")

    return ParameterLoader(store, mode=mode)
