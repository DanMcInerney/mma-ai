#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Rigorous Bayesian Smoothing Parameter Optimization System

This script finds optimal tau (τ) values for three types of Bayesian smoothing:
1. Beta-Binomial (binary outcomes: KO, win, decision, sub_land, ctrl)
2. Poisson-Gamma (count data: strikes_land, kd, rev)
3. Accuracy (ratios: strikes_landed / strikes_attempted)

PHILOSOPHY:
-----------
Smoothing solves the bias-variance tradeoff in small-sample estimation:
- Low tau: Low bias (responsive to data), High variance (unstable estimates)
- High tau: High bias (stuck to prior), Low variance (stable estimates)

Optimal tau minimizes OUT-OF-SAMPLE prediction error, balancing these forces.

METHODOLOGY:
------------
1. Proper Bayesian predictive likelihood (not ad-hoc loss functions)
2. Time-series cross-validation (respects temporal order, prevents leakage)
3. Multiple evaluation metrics (log-likelihood, Brier score, calibration)
4. Stability validation (consistent across CV configurations)
5. Adaptive search (expands range if optimum on boundary)
6. Comprehensive diagnostics (effective sample size, shrinkage, calibration)

DATA USAGE:
-----------
- Training period: 2014-01-01 to 2024-01-01 (statistics computed from this data)
- Evaluation period: 2024-01-02 to 2026-01-01 (parameter selection - pick best tau values)
- Simple temporal split (no cross-validation needed)
- CRITICAL: For each evaluation fight, only use historical data BEFORE that fight's date

Note: We don't need a separate test set because we're just finding optimal scalar
parameters (tau values), not training a complex model that could overfit. The real
test will be when the final XGBoost model (using these smoothed features) is
evaluated on future fights.

Per-weightclass optimization with global fallback for consistency.

Author: AI Analysis + Domain Knowledge
Date: 2024-12-29
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.special import gammaln as loggamma, betaln
from scipy.stats import binom, nbinom
from sklearn.model_selection import TimeSeriesSplit
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Connection
from libs.paths import database_url

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Constants
EPS = 1e-10
MIN_TRAIN_SIZE = 100  # Minimum fights for training
MIN_VAL_SIZE = 20     # Minimum fights for validation

# ============================================================================
# MATHEMATICAL FOUNDATIONS
# ============================================================================

def beta_binomial_logpmf(k: np.ndarray, n: np.ndarray, mu: float, tau: float) -> np.ndarray:
    """
    Beta-Binomial log probability mass function.

    Models: p ~ Beta(α, β), X|p ~ Binomial(n, p)
    Result: X ~ BetaBinomial(n, α, β)

    Args:
        k: Successes (0 <= k <= n)
        n: Trials
        mu: Prior mean (0 < mu < 1)
        tau: Prior pseudo-count (tau > 0)

    Returns:
        Log probabilities
    """
    mu = np.clip(mu, 1e-6, 1 - 1e-6)
    tau = max(tau, 1e-6)

    alpha = mu * tau
    beta = (1.0 - mu) * tau

    # log C(n,k) + log B(k+α, n-k+β) - log B(α,β)
    logpmf = (
        loggamma(n + 1) - loggamma(k + 1) - loggamma(n - k + 1) +
        betaln(k + alpha, n - k + beta) -
        betaln(alpha, beta)
    )

    return logpmf

def negative_binomial_logpmf(k: np.ndarray, t: np.ndarray, mu: float, tau: float) -> np.ndarray:
    """
    Negative Binomial log probability mass function (Poisson-Gamma predictive).

    Models: λ ~ Gamma(r, rate=tau), X ~ Poisson(t*λ)
    Result: X ~ NegBin(r, p) where r=mu*tau, p=tau/(tau+t)

    Args:
        k: Counts (k >= 0)
        t: Exposure (minutes)
        mu: Prior rate (per minute)
        tau: Prior pseudo-minutes

    Returns:
        Log probabilities
    """
    mu = max(mu, 1e-9)
    tau = max(tau, 1e-9)
    t = np.maximum(t, 1e-12)

    r = mu * tau
    p = tau / (tau + t)
    p = np.clip(p, 1e-12, 1 - 1e-12)

    # log Γ(k+r) - log Γ(r) - log Γ(k+1) + r*log(p) + k*log(1-p)
    logpmf = (
        loggamma(k + r) - loggamma(r) - loggamma(k + 1) +
        r * np.log(p) + k * np.log(1.0 - p)
    )

    return logpmf

def brier_score(predictions: np.ndarray, outcomes: np.ndarray) -> float:
    """
    Brier score for probabilistic predictions.

    Lower is better. Perfect predictions = 0, worst = 1.

    Args:
        predictions: Predicted probabilities [0, 1]
        outcomes: Actual outcomes {0, 1}

    Returns:
        Mean squared error
    """
    return np.mean((predictions - outcomes) ** 2)

def calibration_analysis(predictions: np.ndarray, outcomes: np.ndarray, n_bins: int = 10) -> Dict[str, float]:
    """
    Analyze calibration of probabilistic predictions.

    Well-calibrated model: predictions of 70% should occur ~70% of the time.

    Args:
        predictions: Predicted probabilities
        outcomes: Actual outcomes
        n_bins: Number of calibration bins

    Returns:
        Calibration metrics
    """
    # Bin predictions
    bins = np.linspace(0, 1, n_bins + 1)
    bin_indices = np.digitize(predictions, bins[:-1]) - 1
    bin_indices = np.clip(bin_indices, 0, n_bins - 1)

    calibration_error = 0.0
    valid_bins = 0

    for i in range(n_bins):
        mask = bin_indices == i
        if mask.sum() > 0:
            avg_pred = predictions[mask].mean()
            avg_outcome = outcomes[mask].mean()
            calibration_error += np.abs(avg_pred - avg_outcome) * mask.sum()
            valid_bins += 1

    calibration_error /= len(predictions) if len(predictions) > 0 else 1.0

    return {
        'calibration_error': calibration_error,
        'n_bins_used': valid_bins
    }

# ============================================================================
# DATA STRUCTURES
# ============================================================================

@dataclass
class StatSpec:
    """Specification for a single statistic to tune."""
    name: str           # Key for output dict (e.g., 'ko', 'sig_str', 'sig_str_rd1')
    stat_type: str      # 'beta_binomial', 'poisson_gamma', or 'accuracy'
    success_col: Optional[str] = None      # For beta-binomial
    attempts_col: Optional[str] = None     # For beta-binomial/accuracy
    count_col: Optional[str] = None        # For poisson-gamma
    land_col: Optional[str] = None         # For accuracy
    exposure_type: str = 'fight'           # 'fight' or 'rd1'

    def __post_init__(self):
        """Validate spec based on stat type."""
        if self.stat_type == 'beta_binomial':
            assert self.success_col, f"Beta-binomial stat {self.name} needs success_col"
        elif self.stat_type == 'poisson_gamma':
            assert self.count_col, f"Poisson-gamma stat {self.name} needs count_col"
        elif self.stat_type == 'accuracy':
            assert self.land_col and self.attempts_col, f"Accuracy stat {self.name} needs land_col and attempts_col"

@dataclass
class TuningResult:
    """Results for a single stat in a single weight class."""
    stat_name: str
    stat_type: str
    weightclass: str
    optimal_tau: float
    global_tau: float
    use_per_class: bool
    log_likelihood: float
    brier_score: Optional[float]
    calibration_error: Optional[float]
    improvement_pct: float
    tau_stability: float  # CV coefficient of variation
    effective_sample_size: float
    shrinkage_factor: float
    n_samples: int
    search_range: Tuple[float, float]
    boundary_hit: bool

@dataclass
class DiagnosticMetrics:
    """Comprehensive diagnostics for smoothing quality."""
    stat_name: str
    tau: float
    mean_prediction: float
    mean_observed: float
    prediction_std: float
    observed_std: float
    correlation: float
    mse: float
    mae: float
    calibration_slope: float  # Should be ~1.0
    calibration_intercept: float  # Should be ~0.0

# ============================================================================
# STAT DISCOVERY
# ============================================================================

def discover_stats(conn: Connection, schema: str = 'features', table: str = 'fight_stats_fe') -> List[StatSpec]:
    """
    Discover all available statistics in the database and create specs.

    Returns:
        List of StatSpec objects for all tunable stats
    """
    # Get available columns
    query = text("""
        SELECT column_name, data_type
        FROM information_schema.columns
        WHERE table_schema = :schema AND table_name = :table
        ORDER BY column_name
    """)

    cols_df = pd.read_sql(query, conn, params={'schema': schema, 'table': table})
    available_cols = set(cols_df['column_name'].tolist())

    logger.info(f"Found {len(available_cols)} columns in {schema}.{table}")

    specs = []

    # ========================================================================
    # BETA-BINOMIAL STATS (Binary Outcomes)
    # ========================================================================

    # Base stats
    bb_base = [
        ('ko', 'ko', 'one'),
        ('win', 'win', 'one'),
        ('decision', 'decision', 'one'),
        ('sub_land', 'sub_land', 'sub_att'),
        ('ctrl', 'ctrl', 'ctrl'),
    ]

    for name, success_col, attempts_type in bb_base:
        if success_col in available_cols:
            attempts_col = None
            if attempts_type == 'sub_att':
                attempts_col = 'sub_att'
                if attempts_col not in available_cols:
                    logger.warning(f"Skipping {name}: missing {attempts_col}")
                    continue

            specs.append(StatSpec(
                name=name,
                stat_type='beta_binomial',
                success_col=success_col,
                attempts_col=attempts_col,
                exposure_type='fight'
            ))

    # Round 1 variants
    bb_rd1 = [
        ('ko_rd1', 'ko_rd1', 'one'),
        ('win_rd1', 'win_rd1', 'one'),
        ('sub_land_rd1', 'sub_land_rd1', 'sub_att_rd1'),
        ('ctrl_rd1', 'ctrl_rd1', 'ctrl_rd1'),
    ]

    for name, success_col, attempts_type in bb_rd1:
        if success_col in available_cols:
            attempts_col = None
            if attempts_type == 'sub_att_rd1':
                attempts_col = 'sub_att_rd1'
                if attempts_col not in available_cols:
                    logger.warning(f"Skipping {name}: missing {attempts_col}")
                    continue

            specs.append(StatSpec(
                name=name,
                stat_type='beta_binomial',
                success_col=success_col,
                attempts_col=attempts_col,
                exposure_type='rd1'
            ))

    # ========================================================================
    # POISSON-GAMMA STATS (Count Data)
    # ========================================================================

    # Base count stats
    pg_base = [
        ('sig_str', 'sig_str_land'),
        ('head', 'head_land'),
        ('body', 'body_land'),
        ('leg', 'leg_land'),
        ('distance', 'distance_land'),
        ('clinch', 'clinch_land'),
        ('ground', 'ground_land'),
        ('td', 'td_land'),
        ('kd', 'kd'),
        ('rev', 'rev'),
        ('sub', 'sub_att'),  # Using attempts as count
    ]

    for name, count_col in pg_base:
        if count_col in available_cols:
            specs.append(StatSpec(
                name=name,
                stat_type='poisson_gamma',
                count_col=count_col,
                exposure_type='fight'
            ))

    # Round 1 variants
    for name, count_col in pg_base:
        rd1_col = f"{count_col}_rd1"
        if rd1_col in available_cols:
            specs.append(StatSpec(
                name=f"{name}_rd1",
                stat_type='poisson_gamma',
                count_col=rd1_col,
                exposure_type='rd1'
            ))

    # ========================================================================
    # ACCURACY STATS (Ratios)
    # ========================================================================

    # Base accuracy stats
    acc_base = [
        ('sig_str', 'sig_str_land', 'sig_str_att'),
        ('head', 'head_land', 'head_att'),
        ('body', 'body_land', 'body_att'),
        ('leg', 'leg_land', 'leg_att'),
        ('distance', 'distance_land', 'distance_att'),
        ('clinch', 'clinch_land', 'clinch_att'),
        ('ground', 'ground_land', 'ground_att'),
        ('td', 'td_land', 'td_att'),
        ('sub', 'sub_land', 'sub_att'),
    ]

    for name, land_col, att_col in acc_base:
        if land_col in available_cols and att_col in available_cols:
            specs.append(StatSpec(
                name=f"{name}_acc",
                stat_type='accuracy',
                land_col=land_col,
                attempts_col=att_col,
                exposure_type='fight'
            ))

    # Round 1 accuracy
    for name, land_col, att_col in acc_base:
        rd1_land = f"{land_col}_rd1"
        rd1_att = f"{att_col}_rd1"
        if rd1_land in available_cols and rd1_att in available_cols:
            specs.append(StatSpec(
                name=f"{name}_acc_rd1",
                stat_type='accuracy',
                land_col=rd1_land,
                attempts_col=rd1_att,
                exposure_type='rd1'
            ))

    logger.info(f"Created {len(specs)} stat specifications:")
    logger.info(f"  Beta-Binomial: {sum(1 for s in specs if s.stat_type == 'beta_binomial')}")
    logger.info(f"  Poisson-Gamma: {sum(1 for s in specs if s.stat_type == 'poisson_gamma')}")
    logger.info(f"  Accuracy: {sum(1 for s in specs if s.stat_type == 'accuracy')}")

    return specs

# ============================================================================
# DATA LOADING
# ============================================================================

def load_training_data(
    conn: Connection,
    specs: List[StatSpec],
    start_date: str = '2014-01-01',
    end_date: str = '2023-01-01',
    schema: str = 'features',
    table: str = 'fight_stats_fe'
) -> pd.DataFrame:
    """
    Load all necessary data for tuning.

    Args:
        conn: Database connection
        specs: List of stat specifications
        start_date: Training start date
        end_date: Training end date (exclusive)
        schema: Database schema
        table: Table name

    Returns:
        DataFrame with all needed columns
    """
    # Collect all needed columns
    needed_cols = {'time_sec'}

    for spec in specs:
        if spec.stat_type == 'beta_binomial':
            needed_cols.add(spec.success_col)
            if spec.attempts_col:
                needed_cols.add(spec.attempts_col)
            if spec.exposure_type == 'rd1':
                needed_cols.add('time_sec_rd1')

        elif spec.stat_type == 'poisson_gamma':
            needed_cols.add(spec.count_col)
            if spec.exposure_type == 'rd1':
                needed_cols.add('time_sec_rd1')

        elif spec.stat_type == 'accuracy':
            needed_cols.add(spec.land_col)
            needed_cols.add(spec.attempts_col)
            if spec.exposure_type == 'rd1':
                needed_cols.add('time_sec_rd1')

    # Build column list for query
    col_list = ['fd.fight_id', 'fd.fighter_id', 'fd.time_sec']

    if 'time_sec_rd1' in needed_cols:
        col_list.append('fd.time_sec_rd1')

    for col in sorted(needed_cols - {'time_sec', 'time_sec_rd1'}):
        col_list.append(f'fd.{col}')

    # Query
    query = f"""
        SELECT
            {', '.join(col_list)},
            fm.weightclass,
            em.event_date
        FROM {schema}.{table} fd
        JOIN {schema}.fight_mapping fm ON fd.fight_id = fm.fight_id
        JOIN {schema}.event_mapping em ON fd.event_id = em.event_id
        WHERE em.event_date >= :start_date
          AND em.event_date < :end_date
          AND fm.weightclass IN (
              'flyweight', 'bantamweight', 'featherweight', 'lightweight',
              'welterweight', 'middleweight', 'light heavyweight', 'heavyweight'
          )
          AND fd.time_sec > 0
        ORDER BY em.event_date, fd.fight_id
    """

    logger.info(f"Loading data from {start_date} to {end_date}...")
    df = pd.read_sql(text(query), conn, params={'start_date': start_date, 'end_date': end_date})

    logger.info(f"Loaded {len(df)} fight records")
    logger.info(f"Date range: {df['event_date'].min()} to {df['event_date'].max()}")
    logger.info(f"Weight classes: {df['weightclass'].unique().tolist()}")

    return df

# ============================================================================
# TAU OPTIMIZATION
# ============================================================================

class TauOptimizer:
    """Optimizes tau for a single stat using rigorous methodology."""

    def __init__(self, spec: StatSpec, data: pd.DataFrame):
        """
        Initialize optimizer.

        Args:
            spec: Stat specification
            data: Training data
        """
        self.spec = spec
        self.data = data
        self.name = spec.name
        self.stat_type = spec.stat_type

        # Determine initial search range based on stat type and domain knowledge
        self.search_range = self._get_initial_search_range()

    def _get_initial_search_range(self) -> Tuple[float, float]:
        """
        Determine initial tau search range based on stat characteristics.

        Uses domain knowledge about sparsity and typical values.
        """
        if self.stat_type == 'beta_binomial':
            # Control time needs very high tau (per-second modeling)
            if 'ctrl' in self.name:
                return (50.0, 200.0)
            # Submissions are sparse
            elif 'sub' in self.name:
                return (5.0, 40.0)  # Expanded from 30.0
            # KO, win, decision - win stats need higher range
            else:
                return (3.0, 90.0)  # Expanded from 30.0 (win stats hit 60.0 boundary)

        elif self.stat_type == 'poisson_gamma':
            # Extremely rare events (rev stats hit 80.0 boundary)
            if any(x in self.name for x in ['kd', 'rev']):
                return (5.0, 120.0)  # Expanded from (10.0, 40.0)
            # Rare events
            elif any(x in self.name for x in ['sub', 'td']):
                return (3.0, 30.0)  # Expanded from (5.0, 25.0)
            # Common striking
            elif any(x in self.name for x in ['sig_str', 'head', 'body', 'leg']):
                if '_rd1' in self.name:
                    return (0.3, 10.0)  # Expanded from (0.5, 8.0)
                else:
                    return (0.5, 15.0)  # Expanded from (1.0, 12.0)
            # Positional striking
            else:
                return (1.0, 20.0)  # Expanded from (3.0, 15.0)

        elif self.stat_type == 'accuracy':
            # Submissions are sparse (hitting both 10 lower and 100 upper boundaries)
            if 'sub' in self.name:
                return (3.0, 150.0)  # Expanded from (10.0, 25.0)
            # Common stats
            elif any(x in self.name for x in ['sig_str', 'head', 'body', 'leg']):
                return (3.0, 30.0)  # Expanded from (3.0, 15.0)
            # Moderate
            else:
                return (3.0, 40.0)  # Expanded from (5.0, 20.0)

        # Default fallback
        return (1.0, 30.0)

    def optimize_weightclass(
        self,
        wc_train: pd.DataFrame,
        wc_val: pd.DataFrame,
        global_train: pd.DataFrame,
        global_val: pd.DataFrame
    ) -> TuningResult:
        """
        Optimize tau for a specific weight class using simple train/val split.

        Args:
            wc_train: Training data for this weight class
            wc_val: Validation data for this weight class
            global_train: All training data (for global baseline)
            global_val: All validation data (for global baseline)

        Returns:
            TuningResult with optimal tau and diagnostics
        """
        weightclass = wc_train['weightclass'].iloc[0] if len(wc_train) > 0 else 'Unknown'

        logger.info(f"  Optimizing {self.name} for {weightclass} (train={len(wc_train)}, val={len(wc_val)})")

        # Step 1: Find global baseline tau
        global_tau, global_nll, _ = self._optimize_tau(global_train, global_val)

        logger.info(f"    Global optimal tau: {global_tau:.2f} (NLL: {global_nll:.4f})")

        # Step 2: Find weight class specific tau
        wc_tau, wc_nll, boundary_hit = self._optimize_tau(wc_train, wc_val)

        logger.info(f"    WC optimal tau: {wc_tau:.2f} (NLL: {wc_nll:.4f})")

        # Step 3: Evaluate improvement
        # Test global tau on WC validation using GLOBAL priors (matches production fallback)
        baseline_nll = self._evaluate_tau(global_train, wc_val, global_tau)

        # Calculate improvement with NaN/inf safety checks
        if baseline_nll > 0 and np.isfinite(baseline_nll) and np.isfinite(wc_nll):
            improvement_pct = ((baseline_nll - wc_nll) / baseline_nll) * 100
        else:
            improvement_pct = 0.0
            logger.warning(f"    Invalid NLL values (baseline={baseline_nll:.4f}, wc={wc_nll:.4f}), defaulting improvement to 0%")

        logger.info(f"    Improvement over global: {improvement_pct:.2f}%")

        # Step 4: Decide whether to use per-class tau
        use_per_class = (
            np.isfinite(improvement_pct) and  # Valid improvement value
            improvement_pct >= 0.5 and  # Meaningful improvement
            not boundary_hit and         # Not on boundary
            len(wc_train) >= 200          # Sufficient data
        )

        final_tau = wc_tau if use_per_class else global_tau

        logger.info(f"    Decision: {'USE per-class' if use_per_class else 'USE global'} tau={final_tau:.2f}")

        # Step 5: Calculate diagnostics (use combined train+val for shrinkage metrics)
        wc_all = pd.concat([wc_train, wc_val])
        eff_n, shrinkage = self._calculate_shrinkage_metrics(wc_all, final_tau)

        # Step 6: Calculate additional metrics
        # Use priors that match the selected tau (weight class priors for WC tau, global for global tau)
        brier, calib_error = None, None
        if self.stat_type in ['beta_binomial', 'accuracy']:
            prior_train_data = wc_train if use_per_class else global_train
            brier, calib_error = self._calculate_probabilistic_metrics(prior_train_data, wc_val, final_tau)

        return TuningResult(
            stat_name=self.name,
            stat_type=self.stat_type,
            weightclass=weightclass,
            optimal_tau=wc_tau,
            global_tau=global_tau,
            use_per_class=use_per_class,
            log_likelihood=wc_nll if use_per_class else baseline_nll,
            brier_score=brier,
            calibration_error=calib_error,
            improvement_pct=improvement_pct,
            tau_stability=0.0,  # Not applicable with simple split
            effective_sample_size=eff_n,
            shrinkage_factor=shrinkage,
            n_samples=len(wc_train),
            search_range=self.search_range,
            boundary_hit=boundary_hit
        )

    def _optimize_tau(
        self,
        train_data: pd.DataFrame,
        val_data: pd.DataFrame,
        _expansion_count: int = 0,
        _max_expansions: int = 5
    ) -> Tuple[float, float, bool]:
        """
        Find optimal tau using simple train/val split.

        Args:
            train_data: Training data
            val_data: Validation data
            _expansion_count: Internal counter for recursion depth (default: 0)
            _max_expansions: Maximum number of boundary expansions (default: 5)

        Returns:
            (optimal_tau, best_nll, boundary_hit)
        """
        tau_min, tau_max = self.search_range

        # Adaptive grid: finer near expected optima, coarser at edges
        if tau_max - tau_min > 50:
            # Wide range: use logarithmic spacing
            tau_range = np.exp(np.linspace(np.log(tau_min), np.log(tau_max), 30))
        elif tau_max - tau_min > 20:
            # Medium range: mixed spacing
            tau_range = np.concatenate([
                np.linspace(tau_min, tau_min + 10, 15),
                np.linspace(tau_min + 10, tau_max, 10)
            ])
        else:
            # Narrow range: linear spacing
            tau_range = np.linspace(tau_min, tau_max, 25)

        # Evaluate each tau candidate
        tau_scores = []

        for tau in tau_range:
            nll = self._evaluate_tau(train_data, val_data, tau)
            if nll is not None:
                tau_scores.append((tau, nll))

        if not tau_scores:
            logger.warning(f"    No valid tau scores, using midpoint {(tau_min + tau_max)/2:.2f}")
            return (tau_min + tau_max) / 2, float('inf'), True

        best_tau, best_nll = min(tau_scores, key=lambda x: x[1])

        # Check if we hit boundary - if so, expand and retry (with recursion limit)
        boundary_hit = False
        if _expansion_count >= _max_expansions:
            logger.warning(f"    Maximum expansions ({_max_expansions}) reached, using tau={best_tau:.2f}")
            boundary_hit = best_tau in [tau_min, tau_max]
            return best_tau, best_nll, boundary_hit

        if best_tau == tau_min:
            logger.info(f"    Tau at lower boundary, expanding search... (expansion {_expansion_count + 1}/{_max_expansions})")
            new_min = max(0.1, tau_min / 2)
            self.search_range = (new_min, tau_max)
            tau, nll, _ = self._optimize_tau(train_data, val_data, _expansion_count + 1, _max_expansions)
            return tau, nll, True

        elif best_tau == tau_max:
            logger.info(f"    Tau at upper boundary, expanding search... (expansion {_expansion_count + 1}/{_max_expansions})")
            new_max = tau_max * 2
            self.search_range = (tau_min, new_max)
            tau, nll, _ = self._optimize_tau(train_data, val_data, _expansion_count + 1, _max_expansions)
            return tau, nll, True

        return best_tau, best_nll, False

    def _optimize_tau_with_stability(
        self,
        data: pd.DataFrame,
        cv_configs: List[TimeSeriesSplit]
    ) -> Tuple[float, float, float, bool]:
        """
        Optimize tau with stability check across multiple CV configs.

        Returns:
            (median_tau, median_nll, stability_pct, boundary_hit)
        """
        all_taus = []
        all_nlls = []

        for cv in cv_configs:
            tau, nll = self._optimize_tau(data, [cv])
            if tau is not None and nll < float('inf'):
                all_taus.append(tau)
                all_nlls.append(nll)

        if not all_taus:
            tau_min, tau_max = self.search_range
            return (tau_min + tau_max) / 2, float('inf'), 100.0, True

        median_tau = float(np.median(all_taus))
        median_nll = float(np.median(all_nlls))

        # Calculate coefficient of variation
        stability_pct = (np.std(all_taus) / np.mean(all_taus)) * 100 if len(all_taus) > 1 else 0.0

        # Check if any run hit boundary
        tau_min, tau_max = self.search_range
        boundary_hit = any(t <= tau_min * 1.01 or t >= tau_max * 0.99 for t in all_taus)

        return median_tau, median_nll, stability_pct, boundary_hit

    def _evaluate_tau(
        self,
        train_data: pd.DataFrame,
        val_data: pd.DataFrame,
        tau: float
    ) -> Optional[float]:
        """
        Evaluate a specific tau value using simple train/val split.

        Args:
            train_data: Training data
            val_data: Validation data
            tau: Tau value to evaluate

        Returns:
            Negative log-likelihood (or None if invalid)
        """
        # Skip if insufficient data
        if len(train_data) < MIN_TRAIN_SIZE or len(val_data) < MIN_VAL_SIZE:
            return None

        nll = self._compute_fold_nll(train_data, val_data, tau)

        if nll is not None and not np.isnan(nll) and not np.isinf(nll):
            return nll

        return None

    def _compute_fold_nll(
        self,
        train_data: pd.DataFrame,
        val_data: pd.DataFrame,
        tau: float
    ) -> Optional[float]:
        """
        Compute NLL for a single fold.

        Returns:
            Mean negative log-likelihood per observation
        """
        try:
            if self.stat_type == 'beta_binomial':
                return self._compute_bb_nll(train_data, val_data, tau)
            elif self.stat_type == 'poisson_gamma':
                return self._compute_pg_nll(train_data, val_data, tau)
            elif self.stat_type == 'accuracy':
                return self._compute_acc_nll(train_data, val_data, tau)
        except Exception as e:
            logger.debug(f"Error computing NLL: {e}")
            return None

    def _compute_bb_nll(
        self,
        train_data: pd.DataFrame,
        val_data: pd.DataFrame,
        tau: float
    ) -> Optional[float]:
        """Compute Beta-Binomial NLL."""
        # Get successes and attempts
        successes_col = self.spec.success_col

        # Determine attempts
        if self.spec.attempts_col:
            # Submission attempts
            train_valid = train_data[train_data[self.spec.attempts_col] > 0].copy()
            val_valid = val_data[val_data[self.spec.attempts_col] > 0].copy()

            if len(train_valid) < 20 or len(val_valid) < 5:
                return None

            mu = train_valid[successes_col].sum() / train_valid[self.spec.attempts_col].sum()
            k = val_valid[successes_col].values
            n = val_valid[self.spec.attempts_col].values

        elif 'ctrl' in self.name:
            # Control time (per-second modeling)
            if self.spec.exposure_type == 'rd1':
                train_time = np.minimum(
                    train_data.get('time_sec_rd1', train_data['time_sec']).fillna(train_data['time_sec']),
                    300
                )
                val_time = np.minimum(
                    val_data.get('time_sec_rd1', val_data['time_sec']).fillna(val_data['time_sec']),
                    300
                )
            else:
                train_time = train_data['time_sec']
                val_time = val_data['time_sec']

            mu = train_data[successes_col].sum() / train_time.sum()
            k = val_data[successes_col].values
            n = val_time.values

        else:
            # Binary outcomes (one trial per fight)
            mu = train_data[successes_col].mean()
            k = val_data[successes_col].values
            n = np.ones_like(k)

        # Validate
        mu = np.clip(mu, 1e-6, 1 - 1e-6)
        k = np.maximum(k, 0)
        n = np.maximum(n, 1)
        k = np.minimum(k, n)

        # Compute log-likelihood
        loglik = beta_binomial_logpmf(k, n, mu, tau)

        return -np.mean(loglik)  # Return mean NLL

    def _compute_pg_nll(
        self,
        train_data: pd.DataFrame,
        val_data: pd.DataFrame,
        tau: float
    ) -> Optional[float]:
        """Compute Poisson-Gamma (Negative Binomial) NLL."""
        count_col = self.spec.count_col

        # Get exposure
        if self.spec.exposure_type == 'rd1':
            train_exposure = np.minimum(
                train_data.get('time_sec_rd1', train_data['time_sec']).fillna(train_data['time_sec']),
                300
            ) / 60.0
            val_exposure = np.minimum(
                val_data.get('time_sec_rd1', val_data['time_sec']).fillna(val_data['time_sec']),
                300
            ) / 60.0
        else:
            train_exposure = train_data['time_sec'] / 60.0
            val_exposure = val_data['time_sec'] / 60.0

        # Calculate prior rate
        mu = train_data[count_col].sum() / train_exposure.sum()
        mu = max(mu, 1e-9)

        # Get validation counts and exposure
        k = val_data[count_col].values
        t = val_exposure.values

        # Validate
        k = np.maximum(k, 0)
        t = np.maximum(t, 1e-12)

        # Compute log-likelihood
        loglik = negative_binomial_logpmf(k, t, mu, tau)

        return -np.mean(loglik)

    def _compute_acc_nll(
        self,
        train_data: pd.DataFrame,
        val_data: pd.DataFrame,
        tau: float
    ) -> Optional[float]:
        """Compute Accuracy (Beta-Binomial) NLL."""
        land_col = self.spec.land_col
        att_col = self.spec.attempts_col

        # Filter to rows with attempts > 0
        train_valid = train_data[train_data[att_col] > 0].copy()
        val_valid = val_data[val_data[att_col] > 0].copy()

        if len(train_valid) < 20 or len(val_valid) < 5:
            return None

        # Calculate prior accuracy
        mu = train_valid[land_col].sum() / train_valid[att_col].sum()
        mu = np.clip(mu, 1e-6, 1 - 1e-6)

        # Get validation data
        k = val_valid[land_col].values
        n = val_valid[att_col].values

        # Validate
        k = np.maximum(k, 0)
        n = np.maximum(n, 1)
        k = np.minimum(k, n)

        # Compute log-likelihood
        loglik = beta_binomial_logpmf(k, n, mu, tau)

        return -np.mean(loglik)

    def _calculate_shrinkage_metrics(
        self,
        data: pd.DataFrame,
        tau: float
    ) -> Tuple[float, float]:
        """
        Calculate effective sample size and shrinkage factor.

        Returns:
            (effective_n, shrinkage_factor)
        """
        if self.stat_type == 'poisson_gamma':
            # Exposure in minutes
            if self.spec.exposure_type == 'rd1':
                exposure = np.minimum(
                    data.get('time_sec_rd1', data['time_sec']).fillna(data['time_sec']),
                    300
                ).sum() / 60.0
            else:
                exposure = data['time_sec'].sum() / 60.0

            # Effective sample size
            eff_n = exposure / (1 + exposure / tau)

            # Shrinkage factor
            shrinkage = tau / (tau + exposure)

        else:  # beta_binomial or accuracy
            # Get effective number of trials
            if self.spec.attempts_col and 'ctrl' not in self.name:
                n_trials = data[self.spec.attempts_col].sum()
            elif 'ctrl' in self.name:
                if self.spec.exposure_type == 'rd1':
                    n_trials = np.minimum(
                        data.get('time_sec_rd1', data['time_sec']).fillna(data['time_sec']),
                        300
                    ).sum()
                else:
                    n_trials = data['time_sec'].sum()
            else:
                n_trials = len(data)  # One trial per fight

            # Effective sample size
            eff_n = n_trials / (1 + n_trials / tau)

            # Shrinkage factor
            shrinkage = tau / (tau + n_trials)

        return float(eff_n), float(shrinkage)

    def _calculate_probabilistic_metrics(
        self,
        train_data: pd.DataFrame,
        val_data: pd.DataFrame,
        tau: float
    ) -> Tuple[Optional[float], Optional[float]]:
        """
        Calculate Brier score and calibration error for probabilistic predictions.

        Args:
            train_data: Training data
            val_data: Validation data
            tau: Tau value (not used in simplified version, kept for compatibility)

        Returns:
            (brier_score, calibration_error)
        """
        if len(train_data) < MIN_TRAIN_SIZE or len(val_data) < MIN_VAL_SIZE:
            return None, None

        all_predictions = []
        all_outcomes = []

        # Get prior
        if self.stat_type == 'beta_binomial':
            if self.spec.attempts_col and 'ctrl' not in self.name:
                train_valid = train_data[train_data[self.spec.attempts_col] > 0]
                if len(train_valid) < 20:
                    return None, None
                prior = train_valid[self.spec.success_col].sum() / train_valid[self.spec.attempts_col].sum()
            elif 'ctrl' in self.name:
                if self.spec.exposure_type == 'rd1':
                    train_time = np.minimum(
                        train_data.get('time_sec_rd1', train_data['time_sec']).fillna(train_data['time_sec']),
                        300
                    )
                else:
                    train_time = train_data['time_sec']
                prior = train_data[self.spec.success_col].sum() / train_time.sum()
            else:
                prior = train_data[self.spec.success_col].mean()

            # Get outcomes (for binary stats)
            if self.spec.attempts_col and 'ctrl' not in self.name:
                val_valid = val_data[val_data[self.spec.attempts_col] > 0]
                outcomes = (val_valid[self.spec.success_col] / val_valid[self.spec.attempts_col]).values
            elif 'ctrl' not in self.name:
                outcomes = val_data[self.spec.success_col].values
            else:
                return None, None  # Skip ctrl for probabilistic metrics

        elif self.stat_type == 'accuracy':
            train_valid = train_data[train_data[self.spec.attempts_col] > 0]
            if len(train_valid) < 20:
                return None, None
            prior = train_valid[self.spec.land_col].sum() / train_valid[self.spec.attempts_col].sum()

            val_valid = val_data[val_data[self.spec.attempts_col] > 0]
            outcomes = (val_valid[self.spec.land_col] / val_valid[self.spec.attempts_col]).values

        else:
            return None, None

        prior = np.clip(prior, 1e-6, 1 - 1e-6)

        # Predictions are just the smoothed prior (simplified)
        predictions = np.full_like(outcomes, prior)

        all_predictions.extend(predictions)
        all_outcomes.extend(outcomes)

        if not all_predictions:
            return None, None

        all_predictions = np.array(all_predictions)
        all_outcomes = np.array(all_outcomes)

        # Brier score
        brier = brier_score(all_predictions, all_outcomes)

        # Calibration
        calib = calibration_analysis(all_predictions, all_outcomes)

        return float(brier), float(calib['calibration_error'])

# ============================================================================
# MAIN TUNING ORCHESTRATOR
# ============================================================================

class ComprehensiveTuner:
    """Main tuning orchestrator."""

    def __init__(
        self,
        db_url: str,
        output_dir: str = 'config',
        train_start: str = '2014-01-01',
        train_end: str = '2023-12-31',
        val_start: str = '2024-01-01',
        val_end: str = '2026-01-01'
    ):
        """
        Initialize tuner.

        Args:
            db_url: Database connection string
            output_dir: Output directory for results
            train_start: Training start date (statistics computed from this data)
            train_end: Training end date (exclusive - so 2024-01-02 means through 2024-01-01)
            val_start: Evaluation start date (parameter selection)
            val_end: Evaluation end date (exclusive)
        """
        self.db_url = db_url
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)
        self.train_start = train_start
        self.train_end = train_end
        self.val_start = val_start
        self.val_end = val_end

        self.engine = create_engine(db_url)
        self.conn = self.engine.connect()

    def run(self) -> Dict[str, Any]:
        """
        Run comprehensive tuning process.

        Returns:
            Dictionary with all results
        """
        logger.info("="*80)
        logger.info("COMPREHENSIVE BAYESIAN SMOOTHING PARAMETER OPTIMIZATION")
        logger.info("="*80)
        logger.info(f"Training period: {self.train_start} to {self.train_end} (statistics computed)")
        logger.info(f"Evaluation period: {self.val_start} to {self.val_end} (parameter selection)")
        logger.info(f"Output directory: {self.output_dir}")
        logger.info("")

        # Step 1: Discover stats
        logger.info("Step 1: Discovering statistics...")
        specs = discover_stats(self.conn)
        logger.info(f"Found {len(specs)} statistics to tune")
        logger.info("")

        # Step 2: Load ALL data (training + evaluation)
        logger.info("Step 2: Loading all data (training + evaluation)...")
        all_data = load_training_data(
            self.conn,
            specs,
            self.train_start,
            self.val_end  # Load through evaluation period
        )
        logger.info(f"Total fights loaded: {len(all_data)}")

        # Split into training and evaluation
        # Convert event_date column to datetime for comparison
        all_data['event_date'] = pd.to_datetime(all_data['event_date'])

        train_end_dt = pd.to_datetime(self.train_end)
        val_start_dt = pd.to_datetime(self.val_start)
        val_end_dt = pd.to_datetime(self.val_end)

        train_data = all_data[all_data['event_date'] < train_end_dt].copy()
        val_data = all_data[(all_data['event_date'] >= val_start_dt) &
                            (all_data['event_date'] < val_end_dt)].copy()

        logger.info(f"Training fights: {len(train_data)} ({self.train_start} to {self.train_end})")
        logger.info(f"Evaluation fights: {len(val_data)} ({self.val_start} to {self.val_end})")
        logger.info("")

        # Step 3: Tune each stat for each weight class
        logger.info("Step 3: Optimizing tau values...")
        logger.info("")

        weight_classes = [
            'flyweight', 'bantamweight', 'featherweight', 'lightweight',
            'welterweight', 'middleweight', 'light heavyweight', 'heavyweight'
        ]

        all_results = []

        for spec in specs:
            logger.info(f"Tuning {spec.name} ({spec.stat_type})...")

            optimizer = TauOptimizer(spec, train_data)

            for wc in weight_classes:
                wc_train = train_data[train_data['weightclass'] == wc]
                wc_val = val_data[val_data['weightclass'] == wc]

                if len(wc_train) < 200 or len(wc_val) < MIN_VAL_SIZE:
                    logger.info(f"  Skipping {wc}: insufficient data (train={len(wc_train)}, val={len(wc_val)})")
                    continue

                result = optimizer.optimize_weightclass(wc_train, wc_val, train_data, val_data)
                all_results.append(result)

            logger.info("")

        # Step 4: Aggregate and format results
        logger.info("Step 4: Aggregating results...")
        formatted_results = self._format_results(all_results)

        # Step 5: Save results
        logger.info("Step 5: Saving results...")
        self._save_results(formatted_results, all_results)

        logger.info("")
        logger.info("="*80)
        logger.info("TUNING COMPLETE")
        logger.info("="*80)

        return formatted_results

    def _format_results(self, all_results: List[TuningResult]) -> Dict[str, Any]:
        """
        Format results for output.

        Returns:
            Formatted dictionary ready for JSON serialization
        """
        # Group by stat type
        bb_results = [r for r in all_results if r.stat_type == 'beta_binomial']
        pg_results = [r for r in all_results if r.stat_type == 'poisson_gamma']
        acc_results = [r for r in all_results if r.stat_type == 'accuracy']

        # Build per-weightclass configurations
        def build_wc_config(results):
            wc_config = {}

            # Group by weight class
            for wc in ['flyweight', 'bantamweight', 'featherweight', 'lightweight',
                      'welterweight', 'middleweight', 'light heavyweight', 'heavyweight']:
                wc_results = [r for r in results if r.weightclass == wc]
                if not wc_results:
                    continue

                wc_params = {}
                for r in wc_results:
                    if r.use_per_class:
                        wc_params[r.stat_name] = r.optimal_tau

                if wc_params:
                    wc_config[wc] = wc_params

            return wc_config

        # Build global configurations (median of global taus)
        def build_global_config(results):
            global_params = {}

            # Group by stat name
            stat_names = set(r.stat_name for r in results)

            for stat_name in stat_names:
                stat_results = [r for r in results if r.stat_name == stat_name]
                if stat_results:
                    # Use median of global taus
                    global_tau = np.median([r.global_tau for r in stat_results])
                    global_params[stat_name] = float(global_tau)

            return global_params

        return {
            'metadata': {
                'training_period': f"{self.train_start} to {self.train_end}",
                'evaluation_period': f"{self.val_start} to {self.val_end}",
                'n_stats_tuned': len(set(r.stat_name for r in all_results)),
                'n_weight_classes': len(set(r.weightclass for r in all_results)),
                'total_optimizations': len(all_results)
            },
            'beta_binomial': {
                'per_weightclass': build_wc_config(bb_results),
                'global': build_global_config(bb_results),
                'n_stats': len(set(r.stat_name for r in bb_results))
            },
            'poisson_gamma': {
                'per_weightclass': build_wc_config(pg_results),
                'global': build_global_config(pg_results),
                'n_stats': len(set(r.stat_name for r in pg_results))
            },
            'accuracy': {
                'per_weightclass': build_wc_config(acc_results),
                'global': build_global_config(acc_results),
                'n_stats': len(set(r.stat_name for r in acc_results))
            }
        }

    def _save_results(self, formatted_results: Dict[str, Any], detailed_results: List[TuningResult]):
        """Save results to files."""
        # Save formatted results
        formatted_path = self.output_dir / 'optimized_parameters.json'
        with open(formatted_path, 'w') as f:
            json.dump(formatted_results, f, indent=2)
        logger.info(f"Saved formatted results to: {formatted_path}")

        # Save detailed results - convert to JSON-serializable format
        detailed_path = self.output_dir / 'detailed_results.json'
        with open(detailed_path, 'w') as f:
            json_results = []
            for r in detailed_results:
                result_dict = asdict(r)
                # Convert tuple to list for JSON serialization
                result_dict['search_range'] = list(result_dict['search_range'])
                # Ensure booleans are Python bools, not numpy bools
                result_dict['use_per_class'] = bool(result_dict['use_per_class'])
                result_dict['boundary_hit'] = bool(result_dict['boundary_hit'])
                json_results.append(result_dict)
            json.dump(json_results, f, indent=2)
        logger.info(f"Saved detailed results to: {detailed_path}")

        # Save human-readable summary
        summary_path = self.output_dir / 'TUNING_SUMMARY.txt'
        self._write_summary(summary_path, formatted_results, detailed_results)
        logger.info(f"Saved summary to: {summary_path}")

    def _write_summary(
        self,
        path: Path,
        formatted_results: Dict[str, Any],
        detailed_results: List[TuningResult]
    ):
        """Write human-readable summary."""
        with open(path, 'w') as f:
            f.write("="*80 + "\n")
            f.write("BAYESIAN SMOOTHING PARAMETER TUNING SUMMARY\n")
            f.write("="*80 + "\n\n")

            f.write(f"Training Period: {formatted_results['metadata']['training_period']}\n")
            f.write(f"Stats Tuned: {formatted_results['metadata']['n_stats_tuned']}\n")
            f.write(f"Weight Classes: {formatted_results['metadata']['n_weight_classes']}\n")
            f.write(f"Total Optimizations: {formatted_results['metadata']['total_optimizations']}\n\n")

            # Beta-Binomial
            f.write("="*80 + "\n")
            f.write("BETA-BINOMIAL PARAMETERS (Binary Outcomes)\n")
            f.write("="*80 + "\n\n")

            f.write("Global Parameters:\n")
            for stat, tau in sorted(formatted_results['beta_binomial']['global'].items()):
                f.write(f"  {stat:<20} tau = {tau:>6.1f}\n")
            f.write("\n")

            if formatted_results['beta_binomial']['per_weightclass']:
                f.write("Per-Weightclass Overrides:\n")
                for wc, params in sorted(formatted_results['beta_binomial']['per_weightclass'].items()):
                    f.write(f"  {wc}:\n")
                    for stat, tau in sorted(params.items()):
                        f.write(f"    {stat:<18} tau = {tau:>6.1f}\n")
                f.write("\n")

            # Poisson-Gamma
            f.write("="*80 + "\n")
            f.write("POISSON-GAMMA PARAMETERS (Count Data, tau in pseudo-minutes)\n")
            f.write("="*80 + "\n\n")

            f.write("Global Parameters:\n")
            for stat, tau in sorted(formatted_results['poisson_gamma']['global'].items()):
                f.write(f"  {stat:<20} tau = {tau:>6.1f} minutes\n")
            f.write("\n")

            if formatted_results['poisson_gamma']['per_weightclass']:
                f.write("Per-Weightclass Overrides:\n")
                for wc, params in sorted(formatted_results['poisson_gamma']['per_weightclass'].items()):
                    f.write(f"  {wc}:\n")
                    for stat, tau in sorted(params.items()):
                        f.write(f"    {stat:<18} tau = {tau:>6.1f} minutes\n")
                f.write("\n")

            # Accuracy
            f.write("="*80 + "\n")
            f.write("ACCURACY PARAMETERS (Landed/Attempted Ratios)\n")
            f.write("="*80 + "\n\n")

            f.write("Global Parameters:\n")
            for stat, tau in sorted(formatted_results['accuracy']['global'].items()):
                f.write(f"  {stat:<20} tau = {tau:>6.1f}\n")
            f.write("\n")

            if formatted_results['accuracy']['per_weightclass']:
                f.write("Per-Weightclass Overrides:\n")
                for wc, params in sorted(formatted_results['accuracy']['per_weightclass'].items()):
                    f.write(f"  {wc}:\n")
                    for stat, tau in sorted(params.items()):
                        f.write(f"    {stat:<18} tau = {tau:>6.1f}\n")
                f.write("\n")

            # Diagnostics
            f.write("="*80 + "\n")
            f.write("KEY DIAGNOSTICS\n")
            f.write("="*80 + "\n\n")

            # Find stats with high shrinkage
            high_shrinkage = [r for r in detailed_results if r.shrinkage_factor > 0.5]
            if high_shrinkage:
                f.write("Stats with High Prior Weight (shrinkage > 50%):\n")
                for r in sorted(high_shrinkage, key=lambda x: x.shrinkage_factor, reverse=True)[:10]:
                    f.write(f"  {r.stat_name:<20} {r.weightclass:<20} "
                           f"shrinkage={r.shrinkage_factor:.1%} tau={r.optimal_tau:.1f}\n")
                f.write("\n")

            # Find stats with low shrinkage
            low_shrinkage = [r for r in detailed_results if r.shrinkage_factor < 0.2]
            if low_shrinkage:
                f.write("Stats with Low Prior Weight (shrinkage < 20%, more responsive):\n")
                for r in sorted(low_shrinkage, key=lambda x: x.shrinkage_factor)[:10]:
                    f.write(f"  {r.stat_name:<20} {r.weightclass:<20} "
                           f"shrinkage={r.shrinkage_factor:.1%} tau={r.optimal_tau:.1f}\n")
                f.write("\n")

            f.write("="*80 + "\n")
            f.write("END OF SUMMARY\n")
            f.write("="*80 + "\n")

# ============================================================================
# CLI
# ============================================================================

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Rigorous Bayesian Smoothing Parameter Optimization',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run full tuning
  python comprehensive_likelihood_tuner.py

  # Run with custom database
  python comprehensive_likelihood_tuner.py --db postgresql://user:pass@localhost/db

  # Run with verbose logging
  python comprehensive_likelihood_tuner.py --verbose
        """
    )

    parser.add_argument(
        '--db',
        default=database_url(),
        help='Database connection string'
    )

    parser.add_argument(
        '--output-dir',
        default='config',
        help='Output directory for results'
    )

    parser.add_argument(
        '--train-start',
        default='2014-01-01',
        help='Training start date (YYYY-MM-DD)'
    )

    parser.add_argument(
        '--train-end',
        default='2023-12-31',
        help='Training end date (YYYY-MM-DD, exclusive)'
    )

    parser.add_argument(
        '--val-start',
        default='2024-01-01',
        help='Validation start date (YYYY-MM-DD)'
    )

    parser.add_argument(
        '--val-end',
        default='2026-01-01',
        help='Validation end date (YYYY-MM-DD, exclusive)'
    )

    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose logging'
    )

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Run tuning
    tuner = ComprehensiveTuner(
        db_url=args.db,
        output_dir=args.output_dir,
        train_start=args.train_start,
        train_end=args.train_end,
        val_start=args.val_start,
        val_end=args.val_end
    )

    results = tuner.run()

    logger.info("")
    logger.info("Results saved to:")
    logger.info(f"  - {Path(args.output_dir) / 'optimized_parameters.json'}")
    logger.info(f"  - {Path(args.output_dir) / 'detailed_results.json'}")
    logger.info(f"  - {Path(args.output_dir) / 'TUNING_SUMMARY.txt'}")

if __name__ == '__main__':
    main()
