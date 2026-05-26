#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Decay Rate Half-Life Optimization System (vSeven_testing2 Edition)

This script finds the optimal decay half-life for exponential time-weighting
of historical fight data, using Mean Squared Error (MSE) as the objective.

WHAT IS DECAY:
--------------
Decay controls how much we weight recent vs. distant historical performance:
- Short half-life (e.g., 0.5 years): Recent fights heavily weighted
- Long half-life (e.g., 2.0 years): All history weighted more equally

Formula: weight = exp(-decay_rate * days_diff / 365.25)
where decay_rate = log(2) / half_life_years

OPTIMIZATION STRATEGY:
----------------------
1. Extract base stats from vSeven_testing2 feature list (20-40 stats)
2. Query SMOOTHED fight data from fight_stats_derived (pre-calculated smoothed values)
3. For each candidate decay half-life [1.6, 1.8, 2.0, ..., 3.4]:
   - Apply exponential decay weights to historical fights in Python
   - Calculate time-weighted averages (decayed stats) for each fighter
   - Create fight pairs with decayed average differences (fighter1 - fighter2)
   - Train logistic regression to predict win/loss (y_true) from decayed averages
   - Compute log-loss (negative log-likelihood) on validation data
4. Select decay rate with lowest validation log-loss

KEY CHANGES FROM PREVIOUS VERSION:
-----------------------------------
- Uses vSeven_testing2 features (actual production features)
- Loads smoothed values directly from fight_stats_derived
- Optimizes decay rate to predict fight outcomes (win/loss), not smoothed stats
- Uses logistic regression with log-loss to evaluate decay rate quality
- Directly optimizes for the end goal: predicting fight outcomes

DATA USAGE:
-----------
- Training period: 2014-01-01 to 2024-01-01 (statistics computed from this data)
- Evaluation period: 2024-01-02 to 2026-01-01 (parameter selection - pick best decay rate)
- CRITICAL: For each evaluation fight, only use historical data BEFORE that fight's date

Note: We don't need a separate test set because we're just finding optimal scalar
parameters (not training a complex model that could overfit). The real test will be
when the final XGBoost model (using these features) is evaluated on future fights.

NO DATABASE REGENERATION:
--------------------------
Unlike changing database features, this approach:
- Queries smoothed fight stats once
- Simulates different decay rates in Python
- Evaluates prediction quality without touching the database
- Completes in 2-4 hours instead of days

Author: AI Analysis + Domain Knowledge
Date: 2026-01-01
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Connection

# Project imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from libs.feature_store.features import vSeven_testing2
from libs.paths import database_url

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Constants
EPS = 1e-10
MIN_HISTORICAL_FIGHTS = 3  # Minimum fights needed for decayed average
DB_URL = database_url()

# ============================================================================
# DATA LOADING
# ============================================================================

def extract_base_stats_from_features(feature_list: List[str]) -> List[str]:
    """
    Extract base stat names from complex feature names.

    Examples:
        'sig_str_land_ratio_dec_adjperf_dec_avg_diff' -> 'sig_str_land'
        'head_land_ratio_dec_adjperf_dec_avg_diff' -> 'head_land'
        'ko_ratio_dec_adjperf_dec_avg_diff' -> 'ko'

    Args:
        feature_list: List of complex feature names

    Returns:
        List of unique base stat names
    """
    base_stats = set()

    # Common stat patterns
    stat_patterns = [
        'sig_str_land', 'sig_str_def', 'sig_str_acc', 'sig_str_att',
        'head_land', 'head_def', 'head_acc', 'head_att',
        'body_land', 'body_def', 'body_acc', 'body_att',
        'leg_land', 'leg_def', 'leg_acc', 'leg_att',
        'distance_land', 'distance_acc', 'distance_att',
        'clinch_land', 'clinch_acc', 'clinch_att',
        'ground_land', 'ground_acc', 'ground_att',
        'td_land', 'td_def', 'td_acc', 'td_att',
        'sub_land', 'sub_def', 'sub_att',
        'ctrl', 'rev', 'kd', 'ko', 'win', 'decision',
        'time_sec', 'reach', 'age', 'days_since_last_fight'
    ]

    for feature in feature_list:
        # Skip meta features
        if 'weightclass' in feature or 'encoded' in feature:
            continue

        # Try to match stat patterns
        for stat in stat_patterns:
            if feature.startswith(stat):
                base_stats.add(stat)
                break

    return sorted(list(base_stats))


def load_smoothed_fight_data(
    conn: Connection,
    stat_cols: List[str],
    start_date: str = '2014-01-01',
    end_date: str = '2024-01-01',
    schema: str = 'features',
    table: str = 'fight_stats_derived'
) -> pd.DataFrame:
    """
    Load SMOOTHED fight data from fight_stats_derived for decay optimization.
    Includes win column for outcome prediction.

    This uses pre-calculated smoothed values, avoiding the need for tau-based
    Bayesian smoothing during optimization.

    Args:
        conn: Database connection
        stat_cols: List of stat columns to load
        start_date: Start date (inclusive)
        end_date: End date (exclusive)
        schema: Database schema
        table: Table name (fight_stats_derived)

    Returns:
        DataFrame with smoothed fight stats, event dates, and win outcomes
    """
    # Build column list
    col_list = ['fd.fight_id', 'fd.fighter_id', 'fd.time_sec', 'fd.win']

    # Add requested stat columns
    for col in stat_cols:
        col_list.append(f'fd.{col}')

    query = f"""
        SELECT
            {', '.join(col_list)},
            fm.weightclass,
            fm.fighter1_id,
            fm.fighter2_id,
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

    logger.info(f"Loading smoothed fight data from {start_date} to {end_date}...")
    logger.info(f"  Stats to load: {len(stat_cols)} columns")

    df = pd.read_sql(text(query), conn, params={'start_date': start_date, 'end_date': end_date})

    logger.info(f"Loaded {len(df)} fight records")
    logger.info(f"Date range: {df['event_date'].min()} to {df['event_date'].max()}")
    logger.info(f"Unique fighters: {df['fighter_id'].nunique()}")
    logger.info(f"Weight classes: {sorted(df['weightclass'].unique().tolist())}")

    return df


# ============================================================================
# DECAY WEIGHTING
# ============================================================================

def apply_decay_weights(
    df: pd.DataFrame,
    decay_half_life_years: float,
    stat_cols: List[str]
) -> pd.DataFrame:
    """
    Apply exponential decay weights to historical data for each fighter.

    This simulates what time_dec_avg_calc.py does in SQL, but in Python
    to avoid regenerating features for each test.

    CRITICAL: Uses strict temporal ordering - only historical fights
    where event_date < current_fight_date are used.

    Args:
        df: Raw fight data sorted by event_date
        decay_half_life_years: Candidate half-life (e.g., 1.0, 1.25, 1.5)
        stat_cols: Columns to calculate decayed averages for

    Returns:
        DataFrame with decayed statistics added as new columns
    """
    decay_rate = np.log(2) / decay_half_life_years

    logger.info(f"  Applying decay weights (half-life={decay_half_life_years:.2f} years, rate={decay_rate:.4f})...")

    results = []
    fighter_groups = df.groupby('fighter_id')
    total_fighters = len(fighter_groups)

    for fighter_idx, (fighter_id, fighter_df) in enumerate(fighter_groups, 1):
        if fighter_idx % 500 == 0:
            logger.info(f"    Processed {fighter_idx}/{total_fighters} fighters...")

        # Sort by event_date to ensure temporal ordering
        fighter_df = fighter_df.sort_values('event_date').reset_index(drop=True)

        # For each fight, calculate decayed average from all PREVIOUS fights
        # Match production behavior: include same-day fights (b.event_date <= c.event_date)
        # but exclude the current fight itself to avoid self-reference
        for i in range(len(fighter_df)):
            current_row = fighter_df.iloc[i].copy()
            current_date = current_row['event_date']
            current_fight_id = current_row.get('fight_id')

            # CRITICAL: Include fights on or before current date, but exclude current fight
            # Production SQL uses: b.event_date <= c.event_date
            # We match this but exclude current fight to avoid self-reference
            historical = fighter_df[
                (fighter_df['event_date'] < current_date) |
                ((fighter_df['event_date'] == current_date) & 
                 (fighter_df.index < i))  # Same-day fights before current index
            ].copy()

            if len(historical) < MIN_HISTORICAL_FIGHTS:
                # Insufficient history - will use priors only
                for stat in stat_cols:
                    current_row[f'{stat}_dec_avg'] = None
                    current_row[f'{stat}_effective_n'] = 0
            else:
                # Verify no data leakage
                assert (historical['event_date'] <= current_date).all(), \
                    f"Data leakage detected: historical dates > current date {current_date}"
                # Verify current fight is not included (critical for no self-reference)
                if current_fight_id is not None:
                    assert current_fight_id not in historical['fight_id'].values, \
                        f"Data leakage detected: current fight {current_fight_id} included in historical"

                # Calculate decay weights
                days_diff = (current_date - historical['event_date']).dt.days
                weights = np.exp(-decay_rate * days_diff.values / 365.25)  # Convert to numpy array

                # Calculate weighted averages
                for stat in stat_cols:
                    values = historical[stat].values
                    valid_mask = ~pd.isna(values)

                    if valid_mask.sum() == 0:
                        current_row[f'{stat}_dec_avg'] = None
                        current_row[f'{stat}_effective_n'] = 0
                    else:
                        valid_values = values[valid_mask]
                        valid_weights = weights[valid_mask]  # Now both are numpy arrays

                        weighted_sum = np.sum(valid_values * valid_weights)
                        weight_sum = np.sum(valid_weights)

                        current_row[f'{stat}_dec_avg'] = weighted_sum / weight_sum

                        # Effective sample size (Kish's formula)
                        current_row[f'{stat}_effective_n'] = (weight_sum ** 2) / np.sum(valid_weights ** 2)

            results.append(current_row)

    result_df = pd.DataFrame(results)
    logger.info(f"  Decay weighting complete. Added {len(stat_cols)} decayed average columns.")

    return result_df


# ============================================================================
# PRE-FIGHT STAT SHIFTING
# ============================================================================

def shift_stats_backward(
    df: pd.DataFrame,
    stat_cols: List[str]
) -> pd.DataFrame:
    """
    Shift each fighter's decayed average stats backward by 1 fight.
    
    For each fighter, shifts stats so that fight N uses the decayed average
    from fight N-1 (pre-fight stats). This ensures we're using only historical
    data available BEFORE the fight.
    
    Args:
        df: DataFrame with decayed averages (one row per fighter per fight)
        stat_cols: List of stat columns to shift
        
    Returns:
        DataFrame with shifted stats (pre-fight stats)
    """
    result_rows = []
    
    for fighter_id, fighter_df in df.groupby('fighter_id'):
        # Sort by event_date to ensure temporal ordering
        fighter_df = fighter_df.sort_values('event_date').reset_index(drop=True)
        
        for i in range(len(fighter_df)):
            current_row = fighter_df.iloc[i].copy()
            
            # For fight i, use decayed averages from fight i-1 (if available)
            if i > 0:
                previous_row = fighter_df.iloc[i-1]
                # Copy decayed averages from previous fight
                for stat in stat_cols:
                    prev_dec_avg = previous_row.get(f'{stat}_dec_avg')
                    current_row[f'{stat}_dec_avg'] = prev_dec_avg
            else:
                # First fight: no previous fight, set to None
                for stat in stat_cols:
                    current_row[f'{stat}_dec_avg'] = None
            
            result_rows.append(current_row)
    
    return pd.DataFrame(result_rows)


# ============================================================================
# FIGHT PAIR CREATION
# ============================================================================

def create_fight_pairs(
    val_df: pd.DataFrame,
    stat_cols: List[str]
) -> pd.DataFrame:
    """
    Create fight pairs (fighter1 vs fighter2) with pre-fight decayed averages and outcomes.
    
    For each fight, creates a row with:
    - fighter1_<stat>_dec_avg: Pre-fight decayed average for fighter1
    - fighter2_<stat>_dec_avg: Pre-fight decayed average for fighter2
    - y_true = 1 if fighter1 won, 0 if fighter2 won
    
    Args:
        val_df: Validation data with shifted decayed averages (one row per fighter per fight)
        stat_cols: List of stat columns to create features for
        
    Returns:
        DataFrame with one row per fight, with fighter1/fighter2 features and y_true
    """
    fight_pairs = []
    
    for fight_id in val_df['fight_id'].unique():
        fight_data = val_df[val_df['fight_id'] == fight_id].copy()
        
        if len(fight_data) != 2:
            continue  # Skip if not exactly 2 fighters
        
        # Get fighter1_id and fighter2_id from fight_mapping (should be in first row)
        if 'fighter1_id' not in fight_data.columns or 'fighter2_id' not in fight_data.columns:
            continue  # Skip if fighter IDs not available
        
        fighter1_id = fight_data['fighter1_id'].iloc[0]
        fighter2_id = fight_data['fighter2_id'].iloc[0]
        
        # Get fighter1 and fighter2 data
        fighter1_mask = fight_data['fighter_id'] == fighter1_id
        fighter2_mask = fight_data['fighter_id'] == fighter2_id
        
        if not fighter1_mask.any() or not fighter2_mask.any():
            continue  # Skip if fighters not found
        
        fighter1_data = fight_data[fighter1_mask].iloc[0]
        fighter2_data = fight_data[fighter2_mask].iloc[0]
        
        # Create pair row
        pair_row = {
            'fight_id': fight_id,
            'event_date': fighter1_data['event_date'],
            'weightclass': fighter1_data['weightclass'],
            'fighter1_id': fighter1_data['fighter_id'],
            'fighter2_id': fighter2_data['fighter_id'],
            'y_true': fighter1_data['win']  # 1 if fighter1 won, 0 if fighter2 won
        }
        
        # Add separate fighter1 and fighter2 stats for each stat
        for stat in stat_cols:
            f1_dec_avg = fighter1_data.get(f'{stat}_dec_avg')
            f2_dec_avg = fighter2_data.get(f'{stat}_dec_avg')
            
            # Store as separate columns: fighter1_<stat> and fighter2_<stat>
            pair_row[f'fighter1_{stat}'] = f1_dec_avg
            pair_row[f'fighter2_{stat}'] = f2_dec_avg
        
        fight_pairs.append(pair_row)
    
    return pd.DataFrame(fight_pairs)


# ============================================================================
# NLL CALCULATION (Logistic Regression on Win/Loss)
# ============================================================================

def compute_nll_for_win_prediction(
    val_pairs: pd.DataFrame,
    decay_half_life_years: float,
    stat_cols: List[str]
) -> Tuple[float, Dict[str, float]]:
    """
    Compute negative log-likelihood for predicting win/loss using pre-fight decayed averages.
    
    Uses logistic regression to predict y_true (win/loss) from fighter1 and fighter2 stats.
    Features are stat differences: fighter1_<stat> - fighter2_<stat>
    Optimizes decay rate based on how well decayed averages predict fight outcomes.
    
    Args:
        val_pairs: Validation fight pairs with pre-fight decayed averages and y_true
        decay_half_life_years: Candidate half-life being tested
        stat_cols: List of stat columns to use as features
        
    Returns:
        (mean_nll, per_stat_nll_dict)
    """
    logger.info(f"\nEvaluating decay half-life: {decay_half_life_years:.2f} years")
    
    # Create feature columns: stat differences (fighter1 - fighter2)
    feature_cols = []
    for stat in stat_cols:
        f1_col = f'fighter1_{stat}'
        f2_col = f'fighter2_{stat}'
        
        # Create difference feature
        if f1_col in val_pairs.columns and f2_col in val_pairs.columns:
            val_pairs[f'{stat}_diff'] = val_pairs[f1_col] - val_pairs[f2_col]
            feature_cols.append(f'{stat}_diff')
    
    if len(feature_cols) == 0:
        logger.warning("  No valid feature columns found")
        return float('inf'), {}
    
    X = val_pairs[feature_cols].copy()
    y = val_pairs['y_true'].copy()
    
    # Remove rows with missing features or outcomes
    valid_mask = ~(X.isna().any(axis=1) | y.isna())
    X = X[valid_mask]
    y = y[valid_mask]
    
    if len(X) < 100:
        logger.warning(f"  Insufficient data: {len(X)} valid fights")
        return float('inf'), {}
    
    # Train logistic regression
    try:
        model = LogisticRegression(max_iter=1000, random_state=42)
        model.fit(X, y)
        
        # Predict probabilities
        y_pred_proba = model.predict_proba(X)[:, 1]  # Probability of fighter1 winning
        
        # Compute log-loss (negative log-likelihood)
        nll = log_loss(y, y_pred_proba)
        
        logger.info(f"  Log-loss (NLL): {nll:.6f} (n={len(X)} fights)")
        logger.info(f"  Accuracy: {(y == (y_pred_proba > 0.5)).mean():.4f}")
        
        # Per-stat contribution (approximate)
        per_stat_nll = {}
        for stat in stat_cols:
            diff_col = f'{stat}_diff'
            if diff_col in X.columns:
                # Use single-feature model to estimate contribution
                X_single = X[[diff_col]].copy()
                if X_single.notna().all().all():
                    try:
                        model_single = LogisticRegression(max_iter=1000, random_state=42)
                        model_single.fit(X_single, y)
                        y_pred_single = model_single.predict_proba(X_single)[:, 1]
                        per_stat_nll[stat] = log_loss(y, y_pred_single)
                    except:
                        per_stat_nll[stat] = float('inf')
                else:
                    per_stat_nll[stat] = float('inf')
            else:
                per_stat_nll[stat] = float('inf')
        
        return nll, per_stat_nll
        
    except Exception as e:
        logger.error(f"  Error training model: {e}")
        return float('inf'), {}


# ============================================================================
# MAIN OPTIMIZER CLASS
# ============================================================================

class DecayRateOptimizer:
    """Optimize decay half-life using NLL minimization."""

    def __init__(
        self,
        db_url: str = DB_URL,
        output_dir: str = 'config',
        train_start: str = '2014-01-01',
        train_end: str = '2023-12-31',
        val_start: str = '2024-01-01',
        val_end: str = '2026-01-01'
    ):
        """
        Initialize optimizer.

        Args:
            db_url: Database connection URL
            output_dir: Directory to save results
            train_start: Training start date (statistics computed from this data)
            train_end: Training end date (exclusive - so 2024-01-02 means through 2024-01-01)
            val_start: Evaluation start date (parameter selection)
            val_end: Evaluation end date (exclusive)
        """
        self.db_url = db_url
        self.output_dir = Path(output_dir)
        self.train_start = train_start
        self.train_end = train_end
        self.val_start = val_start
        self.val_end = val_end

        self.engine = create_engine(db_url)

    def run_optimization(self) -> Dict[str, Any]:
        """
        Run full decay rate optimization using vSeven_testing2 features.

        Returns:
            Results dictionary with optimal decay rate and diagnostics
        """
        logger.info("=" * 80)
        logger.info("DECAY RATE HALF-LIFE OPTIMIZATION")
        logger.info("=" * 80)

        # Step 1: Extract base stats from vSeven_testing2 features
        logger.info("\nStep 1: Extracting base stats from vSeven_testing2 features...")
        stat_cols = extract_base_stats_from_features(vSeven_testing2)
        logger.info(f"  Extracted {len(stat_cols)} unique base stats from {len(vSeven_testing2)} features")
        logger.info(f"  Stats: {', '.join(stat_cols[:10])}..." if len(stat_cols) > 10 else f"  Stats: {', '.join(stat_cols)}")

        # Step 2: Load smoothed fight data
        logger.info("\nStep 2: Loading smoothed fight data from fight_stats_derived...")
        with self.engine.connect() as conn:
            # Load all data (training and evaluation periods)
            all_data = load_smoothed_fight_data(
                conn,
                stat_cols=stat_cols,
                start_date=self.train_start,
                end_date=self.val_end
            )

        # Split into training and evaluation sets
        # Convert event_date column to datetime for comparison
        all_data['event_date'] = pd.to_datetime(all_data['event_date'])

        train_end_dt = pd.to_datetime(self.train_end)
        val_start_dt = pd.to_datetime(self.val_start)
        val_end_dt = pd.to_datetime(self.val_end)

        train_df = all_data[all_data['event_date'] < train_end_dt].copy()
        val_df = all_data[(all_data['event_date'] >= val_start_dt) &
                          (all_data['event_date'] < val_end_dt)].copy()

        logger.info(f"  Training: {len(train_df)} fights ({self.train_start} to {self.train_end})")
        logger.info(f"  Evaluation: {len(val_df)} fights ({self.val_start} to {self.val_end})")

        # Step 3: Grid search over decay rates
        logger.info("\nStep 3: Grid search over decay half-life candidates...")
        candidate_half_lives = [1.6, 1.8, 2.0, 2.2, 2.4, 2.6, 2.8, 3.0, 3.2, 3.4]

        results = []
        for half_life in candidate_half_lives:
            logger.info(f"\n{'='*80}")
            logger.info(f"Testing decay half-life: {half_life:.2f} years")
            logger.info(f"{'='*80}")

            # Apply decay weighting to validation data
            # Note: We apply to ALL data to compute historical averages correctly
            all_data_decayed = apply_decay_weights(all_data, half_life, stat_cols)

            # Extract validation subset with decayed averages
            val_df_decayed = all_data_decayed[
                (all_data_decayed['event_date'] >= val_start_dt) &
                (all_data_decayed['event_date'] < val_end_dt)
            ].copy()

            # Shift stats backward by 1 fight to get pre-fight stats
            val_df_prefight = shift_stats_backward(val_df_decayed, stat_cols)

            # Create fight pairs with pre-fight decayed averages and outcomes
            val_pairs = create_fight_pairs(val_df_prefight, stat_cols)
            
            if len(val_pairs) == 0:
                logger.warning(f"  No valid fight pairs found, skipping half-life {half_life}")
                continue

            # Compute log-loss for win/loss prediction
            nll, per_stat_nll = compute_nll_for_win_prediction(
                val_pairs=val_pairs,
                decay_half_life_years=half_life,
                stat_cols=stat_cols
            )

            results.append({
                'decay_half_life_years': half_life,
                'nll': nll,
                'per_stat_nll': per_stat_nll
            })

        # Step 4: Select best
        logger.info("\n" + "=" * 80)
        logger.info("OPTIMIZATION RESULTS")
        logger.info("=" * 80)

        results_df = pd.DataFrame([
            {'decay_half_life_years': r['decay_half_life_years'], 'nll': r['nll']}
            for r in results
        ])

        best_idx = results_df['nll'].idxmin()
        best_result = results_df.loc[best_idx]
        baseline_result = results_df[results_df['decay_half_life_years'] == 2.0].iloc[0]

        improvement_pct = (
            (baseline_result['nll'] - best_result['nll']) / baseline_result['nll'] * 100
        )

        logger.info(f"\nOptimal decay half-life: {best_result['decay_half_life_years']:.2f} years")
        logger.info(f"Optimal log-loss: {best_result['nll']:.6f}")
        logger.info(f"Baseline log-loss (2.0 years): {baseline_result['nll']:.6f}")
        logger.info(f"Improvement: {improvement_pct:.2f}%")

        # Boundary warning
        if best_result['decay_half_life_years'] in [candidate_half_lives[0], candidate_half_lives[-1]]:
            logger.warning("\n[WARNING] Optimal value is at search boundary!")
            logger.warning(f"Consider expanding search range beyond [{candidate_half_lives[0]}, {candidate_half_lives[-1]}]")

        # Create final results dict
        final_results = {
            'decay_half_life_years': float(best_result['decay_half_life_years']),
            'nll': float(best_result['nll']),
            'baseline_nll': float(baseline_result['nll']),
            'improvement_pct': float(improvement_pct),
            'all_results': [
                {
                    'decay_half_life_years': float(r['decay_half_life_years']),
                    'nll': float(r['nll']),
                    'per_stat_nll': {k: float(v) for k, v in r['per_stat_nll'].items()}
                }
                for r in results
            ],
            'optimization_metadata': {
                'training_period': f"{self.train_start} to {self.train_end} (statistics computed)",
                'evaluation_period': f"{self.val_start} to {self.val_end} (parameter selection)",
                'optimized_at': datetime.now().isoformat(),
                'n_candidates_tested': len(candidate_half_lives),
                'stats_evaluated': stat_cols,
                'n_stats_evaluated': len(stat_cols),
                'search_range': [float(candidate_half_lives[0]), float(candidate_half_lives[-1])]
            }
        }

        return final_results

    def save_results(self, results: Dict[str, Any]) -> None:
        """
        Save optimization results to JSON.

        Args:
            results: Results dictionary from run_optimization()
        """
        self.output_dir.mkdir(parents=True, exist_ok=True)

        output_path = self.output_dir / 'optimized_decay.json'
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)

        logger.info(f"\nResults saved to: {output_path}")


# ============================================================================
# CLI INTERFACE
# ============================================================================

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Optimize decay rate half-life for exponential time-weighting'
    )
    parser.add_argument(
        '--db',
        default=DB_URL,
        help='Database URL (default: DATABASE_URL from .env or repo default)'
    )
    parser.add_argument(
        '--output-dir',
        default='config',
        help='Output directory for results (default: config)'
    )
    parser.add_argument(
        '--train-start',
        default='2014-01-01',
        help='Training start date (default: 2014-01-01)'
    )
    parser.add_argument(
        '--train-end',
        default='2023-01-01',
        help='Training end date (default: 2023-01-01)'
    )
    parser.add_argument(
        '--val-start',
        default='2023-01-01',
        help='Validation start date (default: 2023-01-01)'
    )
    parser.add_argument(
        '--val-end',
        default='2024-01-01',
        help='Validation end date (default: 2024-01-01)'
    )

    args = parser.parse_args()

    # Run optimization
    optimizer = DecayRateOptimizer(
        db_url=args.db,
        output_dir=args.output_dir,
        train_start=args.train_start,
        train_end=args.train_end,
        val_start=args.val_start,
        val_end=args.val_end
    )

    results = optimizer.run_optimization()
    optimizer.save_results(results)

    logger.info("\n" + "=" * 80)
    logger.info("OPTIMIZATION COMPLETE")
    logger.info("=" * 80)


if __name__ == '__main__':
    main()
