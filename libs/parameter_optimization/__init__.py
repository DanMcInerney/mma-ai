"""
Unified Parameter Optimization Library

Provides intelligent parameter loading and caching for smoothing calculators.

Usage:
    from libs.parameter_optimization import get_default_parameter_loader

    # Get parameter loader (uses PARAM_MODE env var)
    param_loader = get_default_parameter_loader()

    # Use in calculators
    BetaBinomialCalculator(conn, param_loader=param_loader).run()
    PoissonGammaCalculator(conn, param_loader=param_loader).run()

Environment Variables:
    PARAM_MODE: 'baseline' or 'optimized' (default: 'optimized')
    FORCE_REOPTIMIZE: '1' to force reoptimization even if cache is valid
"""

from .integration import (
    should_run_optimization,
    run_parameter_optimization,
    get_parameter_mode,
    get_default_parameter_loader
)
from .loaders import ParameterLoader
from .storage import JSONParameterStore, CacheManager

__all__ = [
    'should_run_optimization',
    'run_parameter_optimization',
    'get_parameter_mode',
    'get_default_parameter_loader',
    'ParameterLoader',
    'JSONParameterStore',
    'CacheManager'
]

__version__ = '1.0.0'
