"""
Parameter loader with intelligent resolution.

Resolves parameters with fallback chain:
- Per-weightclass optimized → Global optimized → Baseline hardcoded
"""

import logging
from typing import Optional

from ..storage.json_store import JSONParameterStore

logger = logging.getLogger(__name__)

# Baseline parameters (hardcoded from calculator files)
# These match the current values in BetaBinomialCalculator and PoissonGammaCalculator

BASELINE_BETA_BINOMIAL = {
    'ko': 7.29,
    'ko_rd1': 8.0,
    'sub_land': 13.98,
    'sub_land_rd1': 6.47,
    'win': 43.11,
    'win_rd1': 7.29,
    'decision': 60.0,
    'ctrl': 50.0,
    'ctrl_rd1': 50.0,
    'default': 15.5
}

BASELINE_POISSON_GAMMA = {
    'sig_str': 0.98,
    'sig_str_rd1': 0.81,
    'head': 0.98,
    'head_rd1': 0.81,
    'body': 2.83,
    'body_rd1': 2.38,
    'leg': 1.92,
    'leg_rd1': 1.75,
    'distance': 0.98,
    'distance_rd1': 0.98,
    'clinch': 1.34,
    'clinch_rd1': 0.98,
    'ground': 0.80,
    'ground_rd1': 0.72,
    'td': 6.67,
    'td_rd1': 8.33,
    'sub': 10.83,
    'sub_rd1': 7.5,
    'kd': 20.0,
    'kd_rd1': 12.86,
    'rev': 38.42,
    'rev_rd1': 80.0,
    'default': 8.0
}


class ParameterLoader:
    """
    Loads and resolves parameters with intelligent fallback.

    Resolution order for mode='optimized':
    1. per_weightclass[weightclass][stat_name] (if weightclass provided)
    2. global[stat_name]
    3. BASELINE_PARAMS[stat_name]

    For mode='baseline':
    - Always returns BASELINE_PARAMS[stat_name]
    """

    def __init__(self, store: JSONParameterStore, mode: str = 'optimized'):
        """
        Initialize parameter loader.

        Args:
            store: JSONParameterStore instance
            mode: 'baseline' or 'optimized'

        Raises:
            ValueError: If mode is not 'baseline' or 'optimized'
        """
        if mode not in ('baseline', 'optimized'):
            raise ValueError(f"Invalid mode: {mode}. Must be 'baseline' or 'optimized'")

        self.store = store
        self.mode = mode
        self._params = None

        # Load parameters if in optimized mode
        if mode == 'optimized':
            self._load_params()

    def _load_params(self) -> None:
        """Load parameters from store."""
        self._params = self.store.load()
        if self._params is None:
            raise RuntimeError(
                "No optimized parameters found. Run optimization first or use PARAM_MODE=baseline"
            )
        logger.info(f"Loaded optimized parameters (mode={self.mode})")

    def get_beta_binomial_params(self, stat_name: str, weightclass: Optional[str] = None) -> float:
        """
        Get beta-binomial parameter (pseudo-count) for a stat.

        Args:
            stat_name: Name of stat (e.g., 'ko', 'win', 'sub_land')
            weightclass: Optional weightclass (e.g., 'flyweight', 'heavyweight')

        Returns:
            Pseudo-count (tau) for the stat

        Raises:
            KeyError: If stat_name not found in any parameter source (fail-fast)
        """
        if self.mode == 'baseline':
            return self._get_baseline_beta_binomial(stat_name)

        # Mode is 'optimized' - try resolution chain
        # 1. Try per-weightclass
        if weightclass:
            per_class_params = self._params.get('beta_binomial', {}).get('per_weightclass', {})
            wc_params = per_class_params.get(weightclass, {})
            if stat_name in wc_params:
                logger.debug(f"Using per-weightclass param: {stat_name} ({weightclass}) = {wc_params[stat_name]}")
                return float(wc_params[stat_name])

        # 2. Try global optimized
        global_params = self._params.get('beta_binomial', {}).get('global', {})
        if stat_name in global_params:
            logger.debug(f"Using global optimized param: {stat_name} = {global_params[stat_name]}")
            return float(global_params[stat_name])

        # 3. Fall back to baseline
        return self._get_baseline_beta_binomial(stat_name)

    def get_poisson_gamma_params(self, stat_name: str, weightclass: Optional[str] = None) -> float:
        """
        Get poisson-gamma parameter (pseudo-minutes) for a stat.

        Args:
            stat_name: Name of stat (e.g., 'sig_str', 'td', 'sub')
            weightclass: Optional weightclass (e.g., 'flyweight', 'heavyweight')

        Returns:
            Pseudo-minutes (tau) for the stat

        Raises:
            KeyError: If stat_name not found in any parameter source (fail-fast)
        """
        if self.mode == 'baseline':
            return self._get_baseline_poisson_gamma(stat_name)

        # Mode is 'optimized' - try resolution chain
        # 1. Try per-weightclass
        if weightclass:
            per_class_params = self._params.get('poisson_gamma', {}).get('per_weightclass', {})
            wc_params = per_class_params.get(weightclass, {})
            if stat_name in wc_params:
                logger.debug(f"Using per-weightclass param: {stat_name} ({weightclass}) = {wc_params[stat_name]}")
                return float(wc_params[stat_name])

        # 2. Try global optimized
        global_params = self._params.get('poisson_gamma', {}).get('global', {})
        if stat_name in global_params:
            logger.debug(f"Using global optimized param: {stat_name} = {global_params[stat_name]}")
            return float(global_params[stat_name])

        # 3. Fall back to baseline
        return self._get_baseline_poisson_gamma(stat_name)

    def _get_baseline_beta_binomial(self, stat_name: str) -> float:
        """Get baseline beta-binomial parameter."""
        if stat_name in BASELINE_BETA_BINOMIAL:
            return BASELINE_BETA_BINOMIAL[stat_name]
        # Use default if stat not found
        logger.warning(f"Stat '{stat_name}' not found in baseline params, using default")
        return BASELINE_BETA_BINOMIAL['default']

    def _get_baseline_poisson_gamma(self, stat_name: str) -> float:
        """Get baseline poisson-gamma parameter."""
        if stat_name in BASELINE_POISSON_GAMMA:
            return BASELINE_POISSON_GAMMA[stat_name]
        # Use default if stat not found
        logger.warning(f"Stat '{stat_name}' not found in baseline params, using default")
        return BASELINE_POISSON_GAMMA['default']

    def get_adjperf_params(self) -> dict:
        """
        Get adjperf parameters (winsorization limits and MAD percentile).

        Returns:
            Dictionary with adjperf parameters:
            {
                'winsorization_limits': {'healthy': 7.0, 'sparse': [3.0, 5.0], 'degenerate': 2.5},
                'mad_floor_percentile': 0.10,
                'quality_thresholds': {'mad_degenerate': 0.010, ...}
            }
        """
        if self.mode == 'baseline':
            return self._get_baseline_adjperf()

        # Mode is 'optimized' - try to load from optimized parameters
        adjperf_params = self._params.get('adjperf', {})
        if adjperf_params:
            logger.debug(f"Using optimized adjperf params")
            return adjperf_params

        # Fall back to baseline
        logger.info("No optimized adjperf params found, using baseline")
        return self._get_baseline_adjperf()

    def _get_baseline_adjperf(self) -> dict:
        """Get baseline adjperf parameters."""
        from config.parameters import BASELINE_ADJPERF_PARAMS
        return BASELINE_ADJPERF_PARAMS
