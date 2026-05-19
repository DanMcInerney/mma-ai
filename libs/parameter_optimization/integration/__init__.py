"""Integration module for main.py pipeline."""

from .main_integration import (
    should_run_optimization,
    run_parameter_optimization,
    get_parameter_mode,
    get_default_parameter_loader
)

__all__ = [
    'should_run_optimization',
    'run_parameter_optimization',
    'get_parameter_mode',
    'get_default_parameter_loader'
]
