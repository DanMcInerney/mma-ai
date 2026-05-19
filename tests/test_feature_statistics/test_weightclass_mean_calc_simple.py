#!/usr/bin/env python3
"""
Comprehensive unit tests for WeightclassMeanCalculator.

Tests the calculator with dummy data to ensure weightclass means are calculated correctly.
"""

import unittest
import sys
import os
import pandas as pd
import numpy as np
from unittest.mock import MagicMock, patch
from sqlalchemy import create_engine, text

# Add project root to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from libs.feature_store.calculators.weightclass_mean_calc import WeightclassMeanCalculator
from libs.feature_store.calculator_context import CalculatorContext
from libs.feature_store.base_calculator import BaseCalculator


class TestWeightclassMeanCalculatorSimple(unittest.TestCase):
    """Test suite for WeightclassMeanCalculator with simplified setup."""

    def setUp(self):
        """Set up test fixtures with in-memory SQLite database."""
        # Create in-memory SQLite database for testing
        self.engine = create_engine('sqlite:///:memory:')
        self.conn = self.engine.connect()
        
        # Create all tables in correct order
        self._create_all_tables()
        self._insert_test_data()
        
        # Mock context and calculator
        self.mock_context = MagicMock(spec=CalculatorContext)
        self.mock_context.connection = self.conn
        
        # Mock feature utils
        self.mock_feature_utils = MagicMock()
        self.mock_feature_utils.get_stat_tables.return_value = {
            'test_strikes': ['strikes_land', 'strikes_att', 'strikes_acc', 'strikes_per_min', 'strikes_pressure']
        }
        self.mock_context.feature_utils = self.mock_feature_utils
        
        # Create calculator with mocked dependencies
        with patch.object(BaseCalculator, '__init__', return_value=None):
            self.calculator = WeightclassMeanCalculator(self.mock_context)
            
            # Set necessary attributes manually
            self.calculator.conn = self.conn
            self.calculator.connection = self.conn
            self.calculator.table_suffix = '_wc_mean'
            self.calculator.start_date = '2014-01-01'
            self.calculator.end_date = '2023-01-01'
            self.calculator.stat_tables = self.mock_feature_utils.get_stat_tables()
            self.calculator.logger = MagicMock()
            self.calculator.should_process_column = lambda col: True
            self.calculator.schema = 'features'
            
            # Override the minimum count for testing (original uses 10, we'll use 3 for testing)
            self.calculator.min_sample_size = 3

    def tearDown(self):
        """Clean up test fixtures."""
        self.conn.close()

    def _create_all_tables(self):
        """Create all required tables in correct order."""
        
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
        
        # Create test feature table
        self.conn.execute(text("""
            CREATE TABLE test_strikes (
                fight_id INTEGER,
                fighter_id INTEGER,
                event_id INTEGER,
                strikes_land REAL,
                strikes_att REAL,
                strikes_acc REAL,
                strikes_per_min REAL,
                strikes_pressure REAL
            )
        """))
        
        self.conn.commit()

    def _insert_test_data(self):
        """Insert test data with known expected results."""
        
        # Create events
        events_data = [
            (1, '2020-01-01'),
            (2, '2020-06-01'),
            (3, '2021-01-01'),
            (4, '2021-06-01'),
            (5, '2022-01-01')
        ]
        
        for event_id, event_date in events_data:
            self.conn.execute(text("""
                INSERT INTO event_mapping (event_id, event_date) 
                VALUES (:event_id, :event_date)
            """), {"event_id": event_id, "event_date": event_date})
        
        # Create fights with different weightclasses
        fights_data = [
            # Lightweight fights (weightclass 'LW') - 6 fighters
            (1, 101, 102, 1, 'LW'),
            (2, 103, 104, 2, 'LW'),
            (3, 105, 106, 3, 'LW'),
            
            # Welterweight fights (weightclass 'WW') - 4 fighters  
            (4, 201, 202, 4, 'WW'),
            (5, 203, 204, 5, 'WW'),
            
            # Heavyweight fights (weightclass 'HW') - only 2 fighters (should be excluded due to COUNT < 10)
            (6, 301, 302, 5, 'HW'),
        ]
        
        for fight_id, fighter1_id, fighter2_id, event_id, weightclass in fights_data:
            self.conn.execute(text("""
                INSERT INTO fight_mapping (fight_id, fighter1_id, fighter2_id, event_id, weightclass) 
                VALUES (:fight_id, :fighter1_id, :fighter2_id, :event_id, :weightclass)
            """), {
                "fight_id": fight_id, 
                "fighter1_id": fighter1_id, 
                "fighter2_id": fighter2_id, 
                "event_id": event_id, 
                "weightclass": weightclass
            })
        
        # Create test strikes data with known values for validation
        strikes_data = [
            # Lightweight fighters - designed to have LW mean = 10.0 for strikes_land
            (1, 101, 1, 8.0, 20.0, 0.4, 2.0, 1.5),   # Fighter 101: strikes_land=8
            (1, 102, 1, 12.0, 25.0, 0.48, 3.0, 2.0), # Fighter 102: strikes_land=12
            (2, 103, 2, 9.0, 22.0, 0.41, 2.2, 1.8),  # Fighter 103: strikes_land=9
            (2, 104, 2, 11.0, 24.0, 0.46, 2.8, 1.9), # Fighter 104: strikes_land=11
            (3, 105, 3, 10.0, 23.0, 0.43, 2.5, 1.7), # Fighter 105: strikes_land=10
            (3, 106, 3, 10.0, 21.0, 0.48, 2.3, 1.6), # Fighter 106: strikes_land=10
            # LW mean for strikes_land = (8+12+9+11+10+10)/6 = 60/6 = 10.0
            
            # Welterweight fighters - designed to have WW mean = 15.0 for strikes_land
            (4, 201, 4, 14.0, 30.0, 0.47, 3.5, 2.2), # Fighter 201: strikes_land=14
            (4, 202, 4, 16.0, 32.0, 0.5, 4.0, 2.5),  # Fighter 202: strikes_land=16
            (5, 203, 5, 15.0, 31.0, 0.48, 3.8, 2.3), # Fighter 203: strikes_land=15
            (5, 204, 5, 15.0, 29.0, 0.52, 3.7, 2.4), # Fighter 204: strikes_land=15
            # WW mean for strikes_land = (14+16+15+15)/4 = 60/4 = 15.0
            
            # Heavyweight fighters - only one fight (should be excluded)
            (6, 301, 5, 20.0, 40.0, 0.5, 5.0, 3.0),  # Fighter 301: strikes_land=20
            (6, 302, 5, 22.0, 44.0, 0.5, 5.5, 3.2),  # Fighter 302: strikes_land=22
        ]
        
        for fight_id, fighter_id, event_id, strikes_land, strikes_att, strikes_acc, strikes_per_min, strikes_pressure in strikes_data:
            self.conn.execute(text("""
                INSERT INTO test_strikes (fight_id, fighter_id, event_id, strikes_land, strikes_att, strikes_acc, strikes_per_min, strikes_pressure) 
                VALUES (:fight_id, :fighter_id, :event_id, :strikes_land, :strikes_att, :strikes_acc, :strikes_per_min, :strikes_pressure)
            """), {
                "fight_id": fight_id,
                "fighter_id": fighter_id, 
                "event_id": event_id,
                "strikes_land": strikes_land,
                "strikes_att": strikes_att,
                "strikes_acc": strikes_acc,
                "strikes_per_min": strikes_per_min,
                "strikes_pressure": strikes_pressure
            })
        
        self.conn.commit()

    def test_simple_weightclass_mean_calculation(self):
        """Test basic weightclass mean calculation with known data."""
        
        # Test strikes table
        self.calculator._create_weightclass_mean_table('test_strikes', [])
        
        # Verify the table was created
        result = self.conn.execute(text("SELECT name FROM sqlite_master WHERE type='table' AND name='test_strikes_wc_mean'")).fetchone()
        self.assertIsNotNone(result, "Weightclass mean table should be created")
        
        # Check the calculated means
        means_df = pd.read_sql("SELECT * FROM test_strikes_wc_mean ORDER BY weightclass", self.conn)
        
        # Should have 2 weightclasses (LW and WW), HW excluded due to COUNT < min_sample_size
        self.assertEqual(len(means_df), 2, "Should have 2 weightclasses (LW and WW)")
        self.assertIn('LW', means_df['weightclass'].values)
        self.assertIn('WW', means_df['weightclass'].values)
        self.assertNotIn('HW', means_df['weightclass'].values, "HW should be excluded due to low count")
        
        # Validate specific means for strikes_land
        lw_row = means_df[means_df['weightclass'] == 'LW'].iloc[0]
        ww_row = means_df[means_df['weightclass'] == 'WW'].iloc[0]
        
        # Test strikes_land means
        self.assertAlmostEqual(lw_row['strikes_land_wc_mean'], 10.0, places=1, 
                              msg="LW strikes_land mean should be 10.0")
        self.assertAlmostEqual(ww_row['strikes_land_wc_mean'], 15.0, places=1,
                              msg="WW strikes_land mean should be 15.0")
        
        # Test strikes_acc means (should be around 0.44 for LW, 0.49 for WW)
        self.assertAlmostEqual(lw_row['strikes_acc_wc_mean'], 0.44, places=2,
                              msg="LW strikes_acc mean should be ~0.44")
        self.assertAlmostEqual(ww_row['strikes_acc_wc_mean'], 0.49, places=2,
                              msg="WW strikes_acc mean should be ~0.49")

    def test_mathematical_accuracy(self):
        """Test mathematical accuracy of mean calculations."""
        
        self.calculator._create_weightclass_mean_table('test_strikes', [])
        
        # Get the calculated means
        means_df = pd.read_sql("SELECT * FROM test_strikes_wc_mean ORDER BY weightclass", self.conn)
        
        # Manually calculate expected means for validation
        raw_data = pd.read_sql("""
            SELECT fm.weightclass, ts.* 
            FROM test_strikes ts
            JOIN fight_mapping fm ON ts.fight_id = fm.fight_id
            WHERE fm.weightclass IN ('LW', 'WW')
        """, self.conn)
        
        # Calculate expected means by weightclass
        expected_means = raw_data.groupby('weightclass').agg({
            'strikes_land': 'mean',
            'strikes_att': 'mean',
            'strikes_acc': 'mean',
            'strikes_per_min': 'mean',
            'strikes_pressure': 'mean'
        })
        
        # Compare calculated vs expected
        for weightclass in ['LW', 'WW']:
            calc_row = means_df[means_df['weightclass'] == weightclass].iloc[0]
            expected_row = expected_means.loc[weightclass]
            
            for stat in ['strikes_land', 'strikes_att', 'strikes_acc', 'strikes_per_min', 'strikes_pressure']:
                calc_value = calc_row[f'{stat}_wc_mean']
                expected_value = expected_row[stat]
                
                self.assertAlmostEqual(calc_value, expected_value, places=6,
                                     msg=f"{weightclass} {stat} mean mismatch: {calc_value} vs {expected_value}")

    def test_column_naming_convention(self):
        """Test that column naming follows the _wc_mean convention."""
        
        self.calculator._create_weightclass_mean_table('test_strikes', [])
        
        # Get column names from created table
        columns_result = self.conn.execute(text("""
            PRAGMA table_info(test_strikes_wc_mean)
        """)).fetchall()
        
        column_names = [row[1] for row in columns_result]  # SQLite pragma returns (cid, name, type, ...)
        
        # Check that all non-weightclass columns end with _wc_mean
        for col_name in column_names:
            if col_name != 'weightclass':
                self.assertTrue(col_name.endswith('_wc_mean'), 
                               f"Column {col_name} should end with _wc_mean")

    def test_weightclass_filtering(self):
        """Test that weightclasses with insufficient data are filtered out."""
        
        self.calculator._create_weightclass_mean_table('test_strikes', [])
        
        # Get the results
        means_df = pd.read_sql("SELECT weightclass FROM test_strikes_wc_mean", self.conn)
        
        # Should only have LW and WW (HW has only 2 fighters, less than minimum sample size)
        self.assertEqual(len(means_df), 2, "Should have exactly 2 weightclasses")
        self.assertIn('LW', means_df['weightclass'].values, "Should include LW")
        self.assertIn('WW', means_df['weightclass'].values, "Should include WW") 
        self.assertNotIn('HW', means_df['weightclass'].values, "Should exclude HW (insufficient data)")

    def _create_all_tables(self):
        """Create all required tables."""
        
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
        
        # Create test feature table
        self.conn.execute(text("""
            CREATE TABLE test_strikes (
                fight_id INTEGER,
                fighter_id INTEGER,
                event_id INTEGER,
                strikes_land REAL,
                strikes_att REAL,
                strikes_acc REAL,
                strikes_per_min REAL,
                strikes_pressure REAL
            )
        """))
        
        self.conn.commit()

    def _insert_test_data(self):
        """Insert test data with known expected results."""
        
        # Create events
        events_data = [
            (1, '2020-01-01'),
            (2, '2020-06-01'),
            (3, '2021-01-01'),
            (4, '2021-06-01'),
            (5, '2022-01-01')
        ]
        
        for event_id, event_date in events_data:
            self.conn.execute(text("""
                INSERT INTO event_mapping (event_id, event_date) 
                VALUES (:event_id, :event_date)
            """), {"event_id": event_id, "event_date": event_date})
        
        # Create fights
        fights_data = [
            (1, 101, 102, 1, 'LW'),
            (2, 103, 104, 2, 'LW'),
            (3, 105, 106, 3, 'LW'),
            (4, 201, 202, 4, 'WW'),
            (5, 203, 204, 5, 'WW'),
            (6, 301, 302, 5, 'HW'),  # Only 1 fight for HW
        ]
        
        for fight_id, fighter1_id, fighter2_id, event_id, weightclass in fights_data:
            self.conn.execute(text("""
                INSERT INTO fight_mapping (fight_id, fighter1_id, fighter2_id, event_id, weightclass) 
                VALUES (:fight_id, :fighter1_id, :fighter2_id, :event_id, :weightclass)
            """), {
                "fight_id": fight_id, 
                "fighter1_id": fighter1_id, 
                "fighter2_id": fighter2_id, 
                "event_id": event_id, 
                "weightclass": weightclass
            })
        
        # Create strikes data with known means
        strikes_data = [
            # LW fighters: strikes_land mean = 10.0
            (1, 101, 1, 8.0, 20.0, 0.4, 2.0, 1.5),
            (1, 102, 1, 12.0, 25.0, 0.48, 3.0, 2.0),
            (2, 103, 2, 9.0, 22.0, 0.41, 2.2, 1.8),
            (2, 104, 2, 11.0, 24.0, 0.46, 2.8, 1.9),
            (3, 105, 3, 10.0, 23.0, 0.43, 2.5, 1.7),
            (3, 106, 3, 10.0, 21.0, 0.48, 2.3, 1.6),
            
            # WW fighters: strikes_land mean = 15.0
            (4, 201, 4, 14.0, 30.0, 0.47, 3.5, 2.2),
            (4, 202, 4, 16.0, 32.0, 0.5, 4.0, 2.5),
            (5, 203, 5, 15.0, 31.0, 0.48, 3.8, 2.3),
            (5, 204, 5, 15.0, 29.0, 0.52, 3.7, 2.4),
            
            # HW fighters: should be excluded
            (6, 301, 5, 20.0, 40.0, 0.5, 5.0, 3.0),
            (6, 302, 5, 22.0, 44.0, 0.5, 5.5, 3.2),
        ]
        
        for fight_id, fighter_id, event_id, strikes_land, strikes_att, strikes_acc, strikes_per_min, strikes_pressure in strikes_data:
            self.conn.execute(text("""
                INSERT INTO test_strikes (fight_id, fighter_id, event_id, strikes_land, strikes_att, strikes_acc, strikes_per_min, strikes_pressure) 
                VALUES (:fight_id, :fighter_id, :event_id, :strikes_land, :strikes_att, :strikes_acc, :strikes_per_min, :strikes_pressure)
            """), {
                "fight_id": fight_id,
                "fighter_id": fighter_id, 
                "event_id": event_id,
                "strikes_land": strikes_land,
                "strikes_att": strikes_att,
                "strikes_acc": strikes_acc,
                "strikes_per_min": strikes_per_min,
                "strikes_pressure": strikes_pressure
            })
        
        self.conn.commit()


if __name__ == '__main__':
    unittest.main()
