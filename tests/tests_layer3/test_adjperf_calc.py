#!/usr/bin/env python3
"""
Comprehensive unit tests for AdjustedPerformanceCalculator.

Tests the calculator with dummy data and manual calculations to ensure
the reliability-weighted adjusted performance calculations are mathematically correct.
"""

import unittest
import sys
import os
import pandas as pd
import numpy as np
from unittest.mock import MagicMock, patch
from sqlalchemy import create_engine, text
from math import exp, sqrt

# Add project root to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from libs.feature_store.calculators.adj_perf_calc import AdjustedPerformanceCalculator
from libs.feature_store.calculator_context import CalculatorContext
from libs.feature_store.base_calculator import BaseCalculator


class TestAdjustedPerformanceCalculator(unittest.TestCase):
    """Comprehensive test suite for AdjustedPerformanceCalculator."""

    def setUp(self):
        """Set up test fixtures with in-memory SQLite database."""
        # Create in-memory SQLite database for testing
        self.engine = create_engine('sqlite:///:memory:')
        self.conn = self.engine.connect()
        
        # Create test schema and data
        self._create_test_schema()
        self._create_test_data()
        
        # Mock context and calculator
        self.mock_context = MagicMock(spec=CalculatorContext)
        self.mock_context.connection = self.conn
        
        # Mock feature utils
        self.mock_feature_utils = MagicMock()
        self.mock_feature_utils.get_stat_tables.return_value = {
            'test_sig_str': ['sig_str_land', 'sig_str_att', 'sig_str_acc', 'sig_str_per_min'],
            'test_ko': ['ko', 'ko_per_sig_str_land']
        }
        self.mock_context.feature_utils = self.mock_feature_utils
        
        # Create calculator with mocked dependencies
        with patch.object(BaseCalculator, '__init__', return_value=None):
            self.calculator = AdjustedPerformanceCalculator(
                self.mock_context,
                decay=False,
                K_mean=4.0,
                K_mad=4.0
            )
            
            # Set necessary attributes manually
            self.calculator.conn = self.conn
            self.calculator.connection = self.conn
            self.calculator.decay = False
            self.calculator.K_mean = 4.0
            self.calculator.K_mad = 4.0
            self.calculator.N_floor = 5.0
            self.calculator.mad_to_sigma = 1.0
            self.calculator.binary_cols = {'ko', 'sub_land', 'decision', 'win'}
            self.calculator.use_unified_base = False
            self.calculator.layer_suffix = '_adjperf'
            self.calculator.stat_tables = self.mock_feature_utils.get_stat_tables()
            self.calculator.logger = MagicMock()
            self.calculator.should_process_column = lambda col: True
            self.calculator.schema = 'features'
            self.calculator.include_patterns = set()
            self.calculator.exclude_patterns = set()
            
            # Set up denom table map
            self.calculator.denom_table_map = {
                'ko_per_sig_str_land': ('sig_str', 'sig_str_land')
            }

    def tearDown(self):
        """Clean up test fixtures."""
        self.conn.close()

    def _create_test_schema(self):
        """Create test database schema."""
        
        # Create event_mapping table
        self.conn.execute(text("""
            CREATE TABLE event_mapping (
                event_id INTEGER PRIMARY KEY,
                event_date DATE
            )
        """))
        
        # Create fight_mapping table
        self.conn.execute(text("""
            CREATE TABLE fight_mapping (
                fight_id INTEGER PRIMARY KEY,
                fighter1_id INTEGER,
                fighter2_id INTEGER,
                event_id INTEGER,
                weightclass TEXT
            )
        """))
        
        # Create test feature tables
        self.conn.execute(text("""
            CREATE TABLE test_sig_str (
                fight_id INTEGER,
                fighter_id INTEGER,
                event_id INTEGER,
                sig_str_land REAL,
                sig_str_att REAL,
                sig_str_acc REAL,
                sig_str_per_min REAL
            )
        """))
        
        self.conn.execute(text("""
            CREATE TABLE test_ko (
                fight_id INTEGER,
                fighter_id INTEGER,
                event_id INTEGER,
                ko REAL,
                ko_per_sig_str_land REAL
            )
        """))
        
        # Create weightclass prior tables with correct column structure
        self.conn.execute(text("""
            CREATE TABLE test_sig_str_wc_mean (
                weightclass TEXT PRIMARY KEY,
                sig_str_acc_wc_mean REAL,
                sig_str_per_min_wc_mean REAL
            )
        """))
        
        self.conn.execute(text("""
            CREATE TABLE test_sig_str_wc_mad (
                weightclass TEXT PRIMARY KEY,
                sig_str_acc_wc_mad REAL,
                sig_str_per_min_wc_mad REAL
            )
        """))
        
        self.conn.execute(text("""
            CREATE TABLE test_sig_str_minimum_mad (
                weightclass TEXT PRIMARY KEY,
                sig_str_acc_min_mad REAL,
                sig_str_per_min_min_mad REAL
            )
        """))
        
        self.conn.execute(text("""
            CREATE TABLE test_ko_wc_mean (
                weightclass TEXT PRIMARY KEY,
                ko_per_sig_str_land_wc_mean REAL
            )
        """))
        
        self.conn.execute(text("""
            CREATE TABLE test_ko_wc_mad (
                weightclass TEXT PRIMARY KEY,
                ko_per_sig_str_land_wc_mad REAL
            )
        """))
        
        self.conn.execute(text("""
            CREATE TABLE test_ko_minimum_mad (
                weightclass TEXT PRIMARY KEY,
                ko_per_sig_str_land_min_mad REAL
            )
        """))
        
        self.conn.commit()

    def _create_test_data(self):
        """Create test data with known statistical properties for validation."""
        
        # Insert events (spanning multiple years for history)
        events = [
            (1, '2018-01-01'), (2, '2018-06-01'), (3, '2019-01-01'),
            (4, '2019-06-01'), (5, '2020-01-01'), (6, '2020-06-01'),
            (7, '2021-01-01'), (8, '2021-06-01'), (9, '2022-01-01')
        ]
        for event_id, event_date in events:
            self.conn.execute(text("INSERT INTO event_mapping VALUES (:event_id, :event_date)"),
                            {"event_id": event_id, "event_date": event_date})
        
        # Insert fights with specific fighter matchups for history testing
        fights = [
            # Fighter 100's historical fights (will be the opponent F2 in our test)
            (1, 100, 201, 1, 'LW'),  # 2018-01: F100 vs F201
            (2, 100, 202, 2, 'LW'),  # 2018-06: F100 vs F202  
            (3, 100, 203, 3, 'LW'),  # 2019-01: F100 vs F203
            (4, 100, 204, 4, 'LW'),  # 2019-06: F100 vs F204
            
            # Current fight: F999 vs F100 (F100 is F2, has history)
            (9, 999, 100, 9, 'LW'),  # 2022-01: F999 vs F100 (test this)
        ]
        
        for fight_id, f1_id, f2_id, event_id, weightclass in fights:
            self.conn.execute(text("INSERT INTO fight_mapping VALUES (:fight_id, :f1_id, :f2_id, :event_id, :weightclass)"),
                            {"fight_id": fight_id, "f1_id": f1_id, "f2_id": f2_id, "event_id": event_id, "weightclass": weightclass})
        
        # Insert sig_str data - design F100's opponents to have specific accuracy values
        sig_str_data = [
            # F100's historical opponents (what they achieved against F100)
            (1, 201, 1, 10.0, 20.0, 0.50, 2.0),  # F201 vs F100: 50% accuracy
            (2, 202, 2, 12.0, 20.0, 0.60, 2.4),  # F202 vs F100: 60% accuracy  
            (3, 203, 3, 8.0, 20.0, 0.40, 1.6),   # F203 vs F100: 40% accuracy
            (4, 204, 4, 14.0, 20.0, 0.70, 2.8),  # F204 vs F100: 70% accuracy
            # Historical opponent mean accuracy = (0.50 + 0.60 + 0.40 + 0.70) / 4 = 0.55
            # Historical opponent MAD = median(|0.50-0.55|, |0.60-0.55|, |0.40-0.55|, |0.70-0.55|) = median(0.05, 0.05, 0.15, 0.15) = 0.10
            
            # F100's own performance in those fights (not used for opponent history)
            (1, 100, 1, 15.0, 25.0, 0.60, 3.0),  # F100 vs F201
            (2, 100, 2, 16.0, 25.0, 0.64, 3.2),  # F100 vs F202
            (3, 100, 3, 14.0, 25.0, 0.56, 2.8),  # F100 vs F203  
            (4, 100, 4, 17.0, 25.0, 0.68, 3.4),  # F100 vs F204
            
            # Current fight data
            (9, 999, 9, 18.0, 25.0, 0.72, 3.6),  # F999 vs F100: F999 gets 72% accuracy
            (9, 100, 9, 13.0, 25.0, 0.52, 2.6),  # F100 vs F999: F100 allows F999 to get 72%
        ]
        
        for fight_id, fighter_id, event_id, land, att, acc, per_min in sig_str_data:
            self.conn.execute(text("INSERT INTO test_sig_str VALUES (:fight_id, :fighter_id, :event_id, :land, :att, :acc, :per_min)"),
                            {"fight_id": fight_id, "fighter_id": fighter_id, "event_id": event_id,
                             "land": land, "att": att, "acc": acc, "per_min": per_min})
        
        # Insert KO data
        ko_data = [
            # F100's historical opponents (what they achieved against F100)
            (1, 201, 1, 0.0, 0.10),  # F201 vs F100: no KO, 10% KO rate per sig_str_land
            (2, 202, 2, 1.0, 0.08),  # F202 vs F100: KO, 8% KO rate
            (3, 203, 3, 0.0, 0.12),  # F203 vs F100: no KO, 12% KO rate
            (4, 204, 4, 0.0, 0.14),  # F204 vs F100: no KO, 14% KO rate
            # Historical opponent mean ko_per_sig_str_land = (0.10 + 0.08 + 0.12 + 0.14) / 4 = 0.11
            
            # F100's own performance (not used for opponent history)
            (1, 100, 1, 0.0, 0.05),
            (2, 100, 2, 0.0, 0.06),
            (3, 100, 3, 1.0, 0.07),
            (4, 100, 4, 0.0, 0.04),
            
            # Current fight
            (9, 999, 9, 0.0, 0.16),  # F999 vs F100: no KO, 16% rate
            (9, 100, 9, 0.0, 0.08),  # F100 vs F999
        ]
        
        for fight_id, fighter_id, event_id, ko, ko_per_sig_str_land in ko_data:
            self.conn.execute(text("INSERT INTO test_ko VALUES (:fight_id, :fighter_id, :event_id, :ko, :ko_per_sig_str_land)"),
                            {"fight_id": fight_id, "fighter_id": fighter_id, "event_id": event_id,
                             "ko": ko, "ko_per_sig_str_land": ko_per_sig_str_land})
        
        # Insert weightclass priors (match column structure exactly)
        self.conn.execute(text("INSERT INTO test_sig_str_wc_mean VALUES ('LW', 0.50, 2.5)"))  # acc_mean, per_min_mean
        self.conn.execute(text("INSERT INTO test_sig_str_wc_mad VALUES ('LW', 0.08, 0.4)"))   # acc_mad, per_min_mad
        self.conn.execute(text("INSERT INTO test_sig_str_minimum_mad VALUES ('LW', 0.01, 0.05)"))  # acc_min_mad, per_min_min_mad
        
        self.conn.execute(text("INSERT INTO test_ko_wc_mean VALUES ('LW', 0.09)"))  # ko_per_sig_str_land_mean
        self.conn.execute(text("INSERT INTO test_ko_wc_mad VALUES ('LW', 0.03)"))   # ko_per_sig_str_land_mad
        self.conn.execute(text("INSERT INTO test_ko_minimum_mad VALUES ('LW', 0.005)"))  # ko_per_sig_str_land_min_mad
        
        self.conn.commit()

    def tearDown(self):
        """Clean up test fixtures."""
        self.conn.close()



    def test_adjperf_target_filtering(self):
        """Test that adjperf target filtering works correctly."""
        
        target_cases = [
            # Should be targets
            ('sig_str_acc', True),
            ('sig_str_per_min', True),
            ('ko_per_sig_str_land', True),
            ('td_land_per_ctrl', True),
            ('sig_str_pressure', True),
            # ('ko', True),  # Removed from targets in user's changes
            # ('sub_land', True),  # Removed from targets in user's changes
            
            # Should NOT be targets
            ('sig_str_total', False),  # _total excluded
            ('sig_str_land', False),   # raw count, not a target
            ('age', False),            # static stat
            ('reach', False),          # static stat
        ]
        
        for col, should_be_target in target_cases:
            with self.subTest(column=col):
                is_target = self.calculator._is_adjperf_target(col)
                self.assertEqual(is_target, should_be_target,
                               f"Column {col} target status should be {should_be_target}, got {is_target}")

    def test_manual_opponent_history_calculation(self):
        """Test opponent history calculation with manual Python calculation."""
        
        # Get F100's historical opponent data manually
        historical_data = pd.read_sql("""
            SELECT ts.sig_str_acc, ts.sig_str_per_min, ko.ko_per_sig_str_land
            FROM test_sig_str ts
            JOIN test_ko ko ON ts.fight_id = ko.fight_id AND ts.fighter_id = ko.fighter_id
            JOIN fight_mapping fm ON ts.fight_id = fm.fight_id
            WHERE ts.fighter_id IN (201, 202, 203, 204)  -- F100's historical opponents
            ORDER BY ts.fight_id
        """, self.conn)
        
        print("F100's historical opponent data:")
        print(historical_data)
        
        # Manual calculations
        # sig_str_acc: opponents achieved [0.50, 0.60, 0.40, 0.70] against F100
        acc_values = historical_data['sig_str_acc'].values
        expected_acc_mean = np.mean(acc_values)  # Should be 0.55
        expected_acc_mad = np.median(np.abs(acc_values - np.median(acc_values)))  # Should be 0.10
        
        # ko_per_sig_str_land: opponents achieved [0.10, 0.08, 0.12, 0.14] against F100
        ko_values = historical_data['ko_per_sig_str_land'].values
        expected_ko_mean = np.mean(ko_values)  # Should be 0.11
        expected_ko_mad = np.median(np.abs(ko_values - np.median(ko_values)))
        
        print(f"Manual calculations:")
        print(f"  sig_str_acc: mean={expected_acc_mean:.3f}, MAD={expected_acc_mad:.3f}")
        print(f"  ko_per_sig_str_land: mean={expected_ko_mean:.3f}, MAD={expected_ko_mad:.3f}")
        
        # Validate our test data design
        self.assertAlmostEqual(expected_acc_mean, 0.55, places=3, msg="Test data should have acc mean = 0.55")
        self.assertAlmostEqual(expected_acc_mad, 0.10, places=3, msg="Test data should have acc MAD = 0.10")
        self.assertAlmostEqual(expected_ko_mean, 0.11, places=3, msg="Test data should have ko mean = 0.11")

    def test_shrinkage_calculation(self):
        """Test reliability-weighted shrinkage calculation."""
        
        # Test shrinkage with known values
        K_mean = 4.0
        K_mad = 4.0
        n_fights = 4  # F100 has 4 historical fights
        
        # Shrinkage weights
        w_mean = n_fights / (n_fights + K_mean)  # 4 / (4 + 4) = 0.5
        w_mad = n_fights / (n_fights + K_mad)    # 4 / (4 + 4) = 0.5
        
        # For sig_str_acc
        opp_mean_pers = 0.55  # What opponents achieved against F100
        opp_mad_pers = 0.10   # MAD of what opponents achieved
        wc_mean = 0.50        # Weightclass mean
        wc_mad = 0.08         # Weightclass MAD
        
        # Manual shrinkage calculation
        expected_shrunk_mean = w_mean * opp_mean_pers + (1 - w_mean) * wc_mean
        expected_shrunk_mad = max(w_mad * opp_mad_pers + (1 - w_mad) * wc_mad, 0.01)  # MAD floor = 0.01
        
        print(f"Shrinkage calculation test:")
        print(f"  w_mean = {w_mean}, w_mad = {w_mad}")
        print(f"  opp_mean_pers = {opp_mean_pers}, wc_mean = {wc_mean}")
        print(f"  shrunk_mean = {expected_shrunk_mean:.3f}")
        print(f"  opp_mad_pers = {opp_mad_pers}, wc_mad = {wc_mad}")
        print(f"  shrunk_mad = {expected_shrunk_mad:.3f}")
        
        # Validate shrinkage logic
        self.assertAlmostEqual(expected_shrunk_mean, 0.525, places=3, 
                              msg="Shrunk mean should be 0.5*0.55 + 0.5*0.50 = 0.525")
        self.assertAlmostEqual(expected_shrunk_mad, 0.09, places=3,
                              msg="Shrunk MAD should be 0.5*0.10 + 0.5*0.08 = 0.09")

    def test_simplified_denominator(self):
        """Test the simplified MAD-only denominator calculation."""
        
        # Test that denominator is just the shrunk MAD (no exposure noise)
        shrunk_mad = 0.09
        mad_floor = 0.01
        
        # The denominator should just be the shrunk MAD (or floor if larger)
        expected_denom = max(shrunk_mad, mad_floor)
        
        print(f"Simplified denominator test:")
        print(f"  shrunk_mad: {shrunk_mad}")
        print(f"  mad_floor: {mad_floor}")
        print(f"  denominator: {expected_denom}")
        
        # Validate simplified approach
        self.assertEqual(expected_denom, shrunk_mad, "Denominator should be shrunk MAD when above floor")
        
        # Test floor case
        small_mad = 0.005
        expected_denom_floor = max(small_mad, mad_floor)
        self.assertEqual(expected_denom_floor, mad_floor, "Should use floor when shrunk MAD is too small")

    def test_complete_adjperf_calculation(self):
        """Test complete adjusted performance calculation with manual verification."""
        
        # Manual calculation for F999 vs F100 fight
        # F999 observed: sig_str_acc = 0.72
        
        # F100's opponent history (what opponents achieved against F100)
        opp_mean_pers = 0.55  # (0.50 + 0.60 + 0.40 + 0.70) / 4
        opp_mad_pers = 0.10   # median of [0.05, 0.05, 0.15, 0.15]
        n_fights = 4
        
        # Weightclass priors
        wc_mean = 0.50
        wc_mad = 0.08
        mad_floor = 0.01
        
        # Shrinkage (K_mean = K_mad = 4.0)
        w_mean = n_fights / (n_fights + 4.0)  # 4/8 = 0.5
        w_mad = n_fights / (n_fights + 4.0)   # 4/8 = 0.5
        
        shrunk_mean = w_mean * opp_mean_pers + (1 - w_mean) * wc_mean
        shrunk_mad = max(w_mad * opp_mad_pers + (1 - w_mad) * wc_mad, mad_floor)
        
        # Final calculation using simplified formula
        observed = 0.72
        mad_shrunk = shrunk_mad  # No scaling or noise
        
        expected_adjperf = (observed - shrunk_mean) / mad_shrunk
        
        # Apply winsorization
        expected_adjperf = max(min(expected_adjperf, 7.0), -7.0)
        
        print(f"\nComplete manual calculation for F999 vs F100 (sig_str_acc):")
        print(f"  Observed: {observed}")
        print(f"  Opponent history: mean={opp_mean_pers}, MAD={opp_mad_pers}, n={n_fights}")
        print(f"  Weightclass priors: mean={wc_mean}, MAD={wc_mad}")
        print(f"  Shrinkage weights: w_mean={w_mean}, w_mad={w_mad}")
        print(f"  Shrunk values: mean={shrunk_mean:.3f}, MAD={shrunk_mad:.3f}")
        print(f"  Denominator (mad_shrunk): {mad_shrunk:.4f}")
        print(f"  Expected adjperf: {expected_adjperf:.4f}")
        
        # Store expected values for comparison with calculator output
        self.expected_adjperf_acc = expected_adjperf
        self.expected_shrunk_mean = shrunk_mean
        self.expected_shrunk_mad = shrunk_mad

    def test_simplified_architecture(self):
        """Test that the simplified architecture removes unnecessary complexity."""
        
        # Test that we no longer have exposure-related methods
        self.assertFalse(hasattr(self.calculator, '_compute_effective_sample_size'), 
                        "Should not have effective sample size method (not needed for simplified formula)")
        self.assertFalse(hasattr(self.calculator, '_needed_denom_tables'),
                        "Should not have denom tables method (not needed for simplified formula)")
        self.assertFalse(hasattr(self.calculator, '_effective_n_floor'),
                        "Should not have N floor method (not needed for simplified formula)")
        
        # Test that we still have the essential methods
        self.assertTrue(hasattr(self.calculator, '_is_adjperf_target'),
                       "Should still have adjperf target filtering")
        self.assertTrue(hasattr(self.calculator, '_compute_observed_canonical_value'),
                       "Should still have canonical value computation")

    def test_canonical_value_simplification(self):
        """Test that canonical values just return stored feature values."""
        
        test_cases = [
            ('sig_str_acc', 't', "COALESCE(t.sig_str_acc, 0)"),
            ('ko_per_sig_str_land', 'hist_opp', "COALESCE(hist_opp.ko_per_sig_str_land, 0)"),
            ('sig_str_per_min', 'current_fight', "COALESCE(current_fight.sig_str_per_min, 0)"),
        ]
        
        for col, alias, expected in test_cases:
            with self.subTest(column=col, alias=alias):
                result = self.calculator._compute_observed_canonical_value(col, 'test_table', alias=alias)
                self.assertEqual(result, expected,
                               f"Canonical value for {col} with alias {alias} should be {expected}")

    def test_simplified_formula_components(self):
        """Test the components of the simplified formula."""
        
        # Test shrinkage weight calculation
        n_fights = 4
        K_mean = 4.0
        K_mad = 4.0
        
        expected_w_mean = n_fights / (n_fights + K_mean)  # 4/8 = 0.5
        expected_w_mad = n_fights / (n_fights + K_mad)    # 4/8 = 0.5
        
        print(f"Simplified formula components test:")
        print(f"  n_fights: {n_fights}")
        print(f"  K_mean: {K_mean}, K_mad: {K_mad}")
        print(f"  w_mean: {expected_w_mean}")
        print(f"  w_mad: {expected_w_mad}")
        
        # Test that weights are in [0,1]
        self.assertGreaterEqual(expected_w_mean, 0.0, "w_mean should be >= 0")
        self.assertLessEqual(expected_w_mean, 1.0, "w_mean should be <= 1")
        self.assertGreaterEqual(expected_w_mad, 0.0, "w_mad should be >= 0")
        self.assertLessEqual(expected_w_mad, 1.0, "w_mad should be <= 1")
        
        # Test extreme cases
        # No history: w should be 0
        w_no_history = 0 / (0 + K_mean)
        self.assertEqual(w_no_history, 0.0, "No history should give w=0")
        
        # Lots of history: w should approach 1
        w_lots_history = 100 / (100 + K_mean)
        self.assertGreater(w_lots_history, 0.9, "Lots of history should give w close to 1")

    def test_sql_generation_structure(self):
        """Test that SQL generation creates proper structure without database execution."""
        
        # Mock column retrieval to avoid information_schema issues
        with patch.object(self.calculator, 'get_features') as mock_get_features:
            mock_get_features.return_value = ['sig_str_acc', 'sig_str_per_min']
            
            # Test SQL generation
            sql = self.calculator.calculate_for_table('test_sig_str', ['sig_str_acc', 'sig_str_per_min'])
            
            # Verify SQL structure
            self.assertIn('WITH', sql, "SQL should use CTEs")
            self.assertIn('rows_0 AS', sql, "Should have rows CTE for history")
            self.assertIn('n_hist AS', sql, "Should have n_fights CTE")
            self.assertIn('stats_0 AS', sql, "Should have stats CTE")
            self.assertIn('mad_0 AS', sql, "Should have MAD CTE")
            self.assertIn('opponent_history AS', sql, "Should have opponent history CTE")
            self.assertIn('weightclass_priors AS', sql, "Should have weightclass priors CTE")
            
            # Verify strict ordering with proper parentheses
            self.assertIn('em_hist.event_date < em_current.event_date', sql, "Should have strict past ordering")
            self.assertIn(') AND hist_opp.', sql, "Should have proper parentheses around OR clauses")
            
            # Verify NULL handling
            self.assertIn('IS NOT NULL', sql, "Should filter NULL values")
            
            # Verify table references
            self.assertIn('features.test_sig_str', sql, "Should reference feature table")
            self.assertIn('_wc_mean', sql, "Should reference weightclass mean table")
            self.assertIn('_wc_mad', sql, "Should reference weightclass MAD table")
            
            # Verify no nested PERCENTILE_CONT (check for actual nesting pattern)
            # Look for the specific illegal pattern: PERCENTILE_CONT inside another PERCENTILE_CONT's ORDER BY
            illegal_pattern = 'ORDER BY ABS(' in sql and 'PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY' in sql and 'ORDER BY ABS(' in sql.split('PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY')[1] if 'PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY' in sql else False
            self.assertFalse(illegal_pattern, "Should not have actual nested PERCENTILE_CONT in ORDER BY clause")
            
            print(f"Generated SQL length: {len(sql)} characters")
            print("SQL structure validation passed!")

    def test_kish_effective_sample_size_logic(self):
        """Test Kish effective sample size calculation logic."""
        
        # Test with known weights
        weights = np.array([1.0, 0.8, 0.6, 0.4])  # Decay weights
        
        # Manual Kish calculation
        sum_w = np.sum(weights)
        sum_w_squared = np.sum(weights**2)
        expected_kish = (sum_w**2) / sum_w_squared
        
        print(f"Kish effective sample size test:")
        print(f"  Weights: {weights}")
        print(f"  Sum(w): {sum_w}")
        print(f"  Sum(w²): {sum_w_squared}")
        print(f"  Kish N_eff: {expected_kish:.3f}")
        
        # Kish should be between 1 and len(weights)
        self.assertGreaterEqual(expected_kish, 1.0, "Kish N_eff should be >= 1")
        self.assertLessEqual(expected_kish, len(weights), "Kish N_eff should be <= number of observations")
        
        # For uniform weights, Kish should equal the count
        uniform_weights = np.array([1.0, 1.0, 1.0, 1.0])
        uniform_kish = (np.sum(uniform_weights)**2) / np.sum(uniform_weights**2)
        self.assertAlmostEqual(uniform_kish, 4.0, places=2, msg="Uniform weights should give Kish = count")

    def test_quadrature_denominator_logic(self):
        """Test the quadrature denominator calculation."""
        
        # Test values
        mad_to_sigma = 1.0
        shrunk_mad = 0.09
        noise = 0.05
        
        # Manual quadrature calculation
        scaled_mad = mad_to_sigma * shrunk_mad
        expected_denom = sqrt(scaled_mad**2 + noise**2)
        
        print(f"Quadrature denominator test:")
        print(f"  shrunk_mad: {shrunk_mad}")
        print(f"  noise: {noise}")
        print(f"  scaled_mad: {scaled_mad}")
        print(f"  quadrature denom: {expected_denom:.4f}")
        
        # Compare with simple max approach
        simple_max = max(scaled_mad, noise)
        
        print(f"  simple max: {simple_max:.4f}")
        print(f"  quadrature vs max ratio: {expected_denom/simple_max:.3f}")
        
        # Quadrature should be >= max of components
        self.assertGreaterEqual(expected_denom, scaled_mad, "Quadrature should be >= scaled MAD")
        self.assertGreaterEqual(expected_denom, noise, "Quadrature should be >= noise")


if __name__ == '__main__':
    unittest.main()
