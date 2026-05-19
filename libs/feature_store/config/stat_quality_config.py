"""
Statistical Quality Configuration for Adjperf Calculations

This module defines empirically-derived thresholds for classifying distribution quality
and determining adaptive winsorization limits.

Thresholds are based on analysis of 1,962 stat × weightclass combinations from the
2014-2023 training period, documented in data/distribution_quality_analysis.json.

Key Findings:
- 22.1% of distributions are degenerate (MAD < 0.01 or >45% at mode)
- 7.3% are sparse (MAD 0.01-0.03)
- 72.8% are healthy (MAD ≥ 0.03)

References:
- docs/ADJPERF_DEGENERATE_DISTRIBUTIONS_FIX_PLAN.md
- scripts/analyze_distribution_quality.py
- data/distribution_quality_analysis.json
"""

from typing import Dict, Tuple, Literal

# Quality tier type
QualityTier = Literal['healthy', 'sparse', 'degenerate']


class StatQualityConfig:
    """Configuration for statistical quality classification and adaptive winsorization."""

    # =========================================================================
    # Distribution Quality Thresholds
    # =========================================================================

    # MAD thresholds for classification (empirically derived)
    MAD_DEGENERATE_THRESHOLD: float = 0.010
    """
    Distributions with wc_mad < 0.010 are classified as degenerate.

    These distributions have collapsed to priors with minimal cross-sectional variance.
    Empirical finding: ~10% of distributions have MAD = 0.0 (truly degenerate).

    Examples:
    - ground_acc (heavyweight): MAD = 0.0, 50.8% at mode
    - td_def (heavyweight): MAD = 0.0, 50.6% at prior
    """

    MAD_SPARSE_THRESHOLD: float = 0.030
    """
    Distributions with 0.010 ≤ wc_mad < 0.030 are classified as sparse.

    These distributions have limited cross-sectional variance, typically due to:
    - High rate of zero attempts (e.g., heavyweight ground strikes)
    - Small sample sizes (e.g., catchweight divisions)
    - Low variance stats (e.g., round 1 ratios)

    Empirical finding: 25th percentile of MAD = 0.0268, close to this threshold.

    Examples:
    - leg_acc (heavyweight): MAD = 0.021
    - sig_str_att_rd1_ratio (flyweight): MAD = 0.027
    """

    # Mode frequency threshold for degeneracy detection
    MODE_FREQUENCY_DEGENERATE_THRESHOLD: float = 0.45
    """
    Distributions with >45% of values at the mode are classified as degenerate.

    This catches cases where MAD might be slightly above zero but distribution
    is still effectively collapsed.

    Empirical finding: Degenerate examples have mode_frequency 0.51-0.97.
    """

    # Minimum sample size for reliable distribution
    MIN_UNIQUE_VALUES_HEALTHY: int = 50
    """
    Distributions with fewer than 50 unique values may lack reliability.

    This doesn't directly affect quality tier classification but can be used
    for additional validation or warnings.
    """

    # =========================================================================
    # Adaptive Winsorization Limits
    # =========================================================================

    # Winsorization limits by quality tier
    WINSOR_LIMIT_DEGENERATE: float = 2.5
    """
    Tight clipping at ±2.5 for degenerate distributions.

    Rationale:
    - When wc_mad ≈ 0, adjperf is dominated by noise (division by mad_floor)
    - Empirical analysis shows 95th percentile of degenerate adjperf ≈ ±8 to ±15
    - This is pure noise amplification, not signal
    - Clip tightly to preserve direction (+/-) while limiting magnitude
    - At ±2.5, we keep directional information but prevent extreme outliers

    Impact: Reduces noise from 181x amplification (when mad = 0) to manageable levels.
    """

    WINSOR_LIMIT_SPARSE: Tuple[float, float] = (3.0, 5.0)
    """
    Adaptive clipping between ±3.0 and ±5.0 for sparse distributions.

    Calculation:
        limit = 3.0 + (wc_mad / 0.030) * 2.0

    Examples:
    - wc_mad = 0.010: limit = 3.67
    - wc_mad = 0.020: limit = 4.33
    - wc_mad = 0.029: limit = 4.93

    Rationale:
    - Sparse distributions have some signal but limited variance
    - Scale clipping proportional to actual variance
    - Empirical analysis shows 95th percentile ≈ ±4 to ±6 for sparse stats
    """

    WINSOR_LIMIT_HEALTHY: float = 7.0
    """
    Standard clipping at ±7.0 for healthy distributions.

    Rationale:
    - Healthy distributions have sufficient cross-sectional variance
    - Empirical analysis shows 95th percentile ≈ ±5 to ±7
    - Standard ±7 clipping preserves outliers while preventing extreme values
    - Maintains backward compatibility with existing behavior
    """

    # =========================================================================
    # Reliability Score Weights
    # =========================================================================

    RELIABILITY_DEGENERATE: float = 0.3
    """
    Low reliability (0.3) for degenerate distribution adjperf values.

    These values are heavily influenced by noise amplification and should be
    down-weighted in downstream models.
    """

    RELIABILITY_SPARSE: float = 0.6
    """
    Medium reliability (0.6) for sparse distribution adjperf values.

    These values have some signal but limited cross-sectional variance.
    """

    RELIABILITY_HEALTHY: float = 1.0
    """
    Full reliability (1.0) for healthy distribution adjperf values.

    These values have sufficient variance and sample size for robust estimation.
    """

    # =========================================================================
    # Floor Calculation Parameters
    # =========================================================================

    LONGITUDINAL_FLOOR_PERCENTILE: float = 0.10
    """
    Use 10th percentile of rolling MAD for longitudinal floor.

    This floor is used for opponent history shrinkage (fighter consistency over time).
    Changed from 5th to 10th percentile to reduce extreme amplification.
    """

    CROSS_SECTIONAL_FLOOR_PERCENTILE: float = 0.10
    """
    Use 10th percentile of wc_mad across weight classes for cross-sectional floor.

    This floor is used for weightclass prior shrinkage (fighter differences at one time).
    Conceptually distinct from longitudinal floor.
    """

    # =========================================================================
    # Helper Methods
    # =========================================================================

    @classmethod
    def classify_distribution_quality(
        cls,
        wc_mad: float,
        mode_frequency: float,
        unique_count: int
    ) -> QualityTier:
        """
        Classify distribution quality based on multiple signals.

        Args:
            wc_mad: Weightclass median absolute deviation
            mode_frequency: Proportion of values at the mode
            unique_count: Number of unique values in distribution

        Returns:
            Quality tier: 'healthy', 'sparse', or 'degenerate'

        Examples:
            >>> StatQualityConfig.classify_distribution_quality(0.0, 0.51, 100)
            'degenerate'

            >>> StatQualityConfig.classify_distribution_quality(0.085, 0.10, 500)
            'healthy'

            >>> StatQualityConfig.classify_distribution_quality(0.021, 0.20, 200)
            'sparse'
        """
        # Check for degeneracy
        if wc_mad < cls.MAD_DEGENERATE_THRESHOLD:
            return 'degenerate'
        if mode_frequency > cls.MODE_FREQUENCY_DEGENERATE_THRESHOLD:
            return 'degenerate'

        # Check for sparsity
        if wc_mad < cls.MAD_SPARSE_THRESHOLD:
            return 'sparse'

        # Default to healthy
        return 'healthy'

    @classmethod
    def calculate_winsorization_limit(
        cls,
        quality_tier: QualityTier,
        wc_mad: float = None
    ) -> float:
        """
        Calculate adaptive winsorization limit based on quality tier.

        Args:
            quality_tier: Distribution quality classification
            wc_mad: Weightclass MAD (required for sparse tier)

        Returns:
            Winsorization limit (applied as ±limit)

        Examples:
            >>> StatQualityConfig.calculate_winsorization_limit('degenerate')
            2.5

            >>> StatQualityConfig.calculate_winsorization_limit('sparse', 0.020)
            4.333333333333333

            >>> StatQualityConfig.calculate_winsorization_limit('healthy')
            7.0
        """
        if quality_tier == 'degenerate':
            return cls.WINSOR_LIMIT_DEGENERATE

        elif quality_tier == 'sparse':
            if wc_mad is None:
                # If MAD not provided, use middle of range
                return (cls.WINSOR_LIMIT_SPARSE[0] + cls.WINSOR_LIMIT_SPARSE[1]) / 2
            # Interpolate between 3.0 and 5.0 based on MAD
            min_limit, max_limit = cls.WINSOR_LIMIT_SPARSE
            mad_ratio = wc_mad / cls.MAD_SPARSE_THRESHOLD
            return min_limit + mad_ratio * (max_limit - min_limit)

        else:  # healthy
            return cls.WINSOR_LIMIT_HEALTHY

    @classmethod
    def get_reliability_score(cls, quality_tier: QualityTier) -> float:
        """
        Get reliability score for a quality tier.

        Args:
            quality_tier: Distribution quality classification

        Returns:
            Reliability score (0.0 to 1.0)

        Examples:
            >>> StatQualityConfig.get_reliability_score('degenerate')
            0.3

            >>> StatQualityConfig.get_reliability_score('sparse')
            0.6

            >>> StatQualityConfig.get_reliability_score('healthy')
            1.0
        """
        if quality_tier == 'degenerate':
            return cls.RELIABILITY_DEGENERATE
        elif quality_tier == 'sparse':
            return cls.RELIABILITY_SPARSE
        else:
            return cls.RELIABILITY_HEALTHY

    @classmethod
    def get_summary_dict(cls) -> Dict[str, any]:
        """
        Get a dictionary summary of all configuration values.

        Returns:
            Dictionary with all configuration parameters
        """
        return {
            'thresholds': {
                'mad_degenerate': cls.MAD_DEGENERATE_THRESHOLD,
                'mad_sparse': cls.MAD_SPARSE_THRESHOLD,
                'mode_frequency_degenerate': cls.MODE_FREQUENCY_DEGENERATE_THRESHOLD,
                'min_unique_values_healthy': cls.MIN_UNIQUE_VALUES_HEALTHY,
            },
            'winsorization_limits': {
                'degenerate': cls.WINSOR_LIMIT_DEGENERATE,
                'sparse': cls.WINSOR_LIMIT_SPARSE,
                'healthy': cls.WINSOR_LIMIT_HEALTHY,
            },
            'reliability_scores': {
                'degenerate': cls.RELIABILITY_DEGENERATE,
                'sparse': cls.RELIABILITY_SPARSE,
                'healthy': cls.RELIABILITY_HEALTHY,
            },
            'floor_percentiles': {
                'longitudinal': cls.LONGITUDINAL_FLOOR_PERCENTILE,
                'cross_sectional': cls.CROSS_SECTIONAL_FLOOR_PERCENTILE,
            }
        }


# =========================================================================
# Empirical Validation Data (from distribution_quality_analysis.json)
# =========================================================================

EMPIRICAL_VALIDATION = {
    'analysis_date': '2026-01-01',
    'total_distributions_analyzed': 1962,
    'date_range': ('2014-01-01', '2023-01-01'),

    'mad_distribution': {
        'min': 0.0,
        'p05': 0.0,
        'p10': 0.0,
        'p25': 0.0268,
        'median': 0.0853,
        'p75': 0.1845,
        'p90': 0.3082,
        'p95': 1.0137,
        'max': 372.5,
    },

    'quality_tier_counts': {
        'degenerate': 433,  # 22.1%
        'sparse': 143,      # 7.3%
        'healthy': 1429,    # 72.8%
    },

    'degenerate_examples': [
        'ground_acc (heavyweight): MAD=0.0, mode_freq=50.8%',
        'ground_def (heavyweight): MAD=0.0, mode_freq=50.8%',
        'td_def (heavyweight): MAD=0.0 (from prior investigation)',
        'ground_rd1_acc (all weightclasses): MAD=0.0, mode_freq=55-64%',
    ],

    'sparse_examples': [
        'leg_acc (heavyweight): MAD=0.021',
        'sig_str_att_rd1_ratio (flyweight): MAD=0.027',
        'strikes_att_rd1_ratio (featherweight): MAD=0.029',
    ],
}
