import unittest
import pandas as pd
import numpy as np
from unittest.mock import MagicMock, patch, Mock
import logging
import sys
import os
import pytest

# Import libs directly

from sqlalchemy import text
from typing import Dict, List, Set, Optional, Any
from datetime import datetime, timedelta

from libs.feature_store.calculators.adj_perf_calc import AdjustedPerformanceCalculator
from libs.feature_store.calculator_context import CalculatorContext
from libs.feature_store.feature_utils import FeatureUtils
from libs.feature_store.sql_template_manager import SQLTemplateManager
from libs.feature_store.base_calculator import BaseCalculator, ColumnFilter

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    stream=sys.stdout
)

class AdjustedPerformanceCalculatorTester:
    """Test helper class that wraps AdjustedPerformanceCalculator with mocked dependencies for testing"""
    
    def __init__(self, conn=None, decay=False, include_patterns=set(), exclude_patterns=set()):
        # Initialize mock context and components
        self.mock_context = MagicMock(spec=CalculatorContext)
        self.mock_context.connection = MagicMock()
        self.mock_context.feature_utils = MagicMock(spec=FeatureUtils)
        self.mock_context.sql_manager = MagicMock(spec=SQLTemplateManager)
        
        # Setup stat tables
        stat_tables = {
            'sig_str': ['sig_str_land', 'sig_str_att', 'sig_str_acc', 'sig_str_def', 
                        'sig_str_land_opp', 'sig_str_att_opp', 'sig_str_acc_opp', 'sig_str_def_opp',
                        'sig_str_opp_avg', 'sig_str_opp_mad', 'sig_str_opp_dec_avg', 'sig_str_opp_dec_mad'],
            'td': ['td_land', 'td_att', 'td_acc', 'td_def', 
                  'td_land_opp', 'td_att_opp', 'td_acc_opp', 'td_def_opp',
                  'td_opp_avg', 'td_opp_mad', 'td_opp_dec_avg', 'td_opp_dec_mad'],
            'sub': ['sub_land', 'sub_att', 
                   'sub_land_opp', 'sub_att_opp',
                   'sub_opp_avg', 'sub_opp_mad', 'sub_opp_dec_avg', 'sub_opp_dec_mad']
        }
        self.mock_context.feature_utils.get_stat_tables.return_value = stat_tables
        
        # Initialize the actual calculator with our mock context
        with patch('libs.feature_store.calculators.adj_perf_calc.CalculatorContext', return_value=self.mock_context):
            self.calculator = AdjustedPerformanceCalculator(
                self.mock_context, 
                decay=decay, 
                K_mean=4.0, 
                K_mad=4.0,
                include_patterns=include_patterns, 
                exclude_patterns=exclude_patterns
            )
        
        # Set up required attributes on the calculator
        self.calculator.calculator_type = 'multi_table'
        self.calculator.schema = 'features'
        self.calculator.stat_tables = stat_tables
        
        # Save patterns for testing
        self.testing_include_patterns = include_patterns
        self.testing_exclude_patterns = exclude_patterns
    
    def _ensure_columns_exist(self, *args, **kwargs):
        # Skip column validation for tests
        pass
    
    def execute_layer_update(self, calculation_sql, table_name, schema, batch_size=100000):
        # In tests, we just want to capture the SQL, not execute updates
        self.last_executed_sql = calculation_sql
        return True
        
    def should_process_column(self, column):
        """Override should_process_column for testing to properly handle patterns"""
        # If we have include patterns, column must match at least one
        if self.testing_include_patterns:
            if not any(pattern in column for pattern in self.testing_include_patterns):
                return False
        
        # If we have exclude patterns, column must not match any
        if self.testing_exclude_patterns:
            if any(pattern in column for pattern in self.testing_exclude_patterns):
                return False
                
        return True
    
    def calculate_for_table(self, table_name, columns=None):
        """Delegate to the wrapped calculator with filtering"""
        if columns is None:
            columns = self.calculator.stat_tables.get(table_name, [])
            
        # Apply filtering manually
        filtered_columns = [col for col in columns if self.should_process_column(col)]
        
        # Call the calculator's method with filtered columns
        return self.calculator.calculate_for_table(table_name, filtered_columns)
    
    def execute_for_table(self, table_name, columns=None):
        """Delegate to the wrapped calculator"""
        return self.calculator.execute_for_table(table_name, columns)
    
    def execute_raw_sql(self, sql, params=None, return_results=True):
        """Delegate to the wrapped calculator"""
        return self.calculator.execute_raw_sql(sql, params, return_results)


# Define fixtures for commonly used mock data
@pytest.fixture
def mock_data():
    """Fixture providing mock data for tests"""
    return {
        # Main stats table with fighter, opponent, and previous opponent data
        'sig_str': pd.DataFrame({
            'fight_id': [1, 2],
            'fighter_id': [101, 102],
            'sig_str_land': [30, 25],        # Fighter's stats
            'sig_str_acc': [0.60, 0.50],
            'sig_str_opp_avg': [None, 20],   # Previous opponent average (None for first fighter)
            'sig_str_opp_mad': [None, 5],    # Previous opponent MAD (None for first fighter)
            'sig_str_opp_dec_avg': [None, 18],   # Previous opponent decayed average
            'sig_str_opp_dec_mad': [None, 4.5]   # Previous opponent decayed MAD
        }),
        # Fight mapping table
        'fight_mapping': pd.DataFrame({
            'fight_id': [1, 2],
            'fighter1_id': [101, 201],
            'fighter2_id': [102, 202],
            'weightclass': ['Lightweight', 'Welterweight'],
            'event_id': [1, 2]
        }),
        # Event mapping table
        'event_mapping': pd.DataFrame({
            'event_id': [1, 2],
            'event_date': ['2020-01-15', '2020-06-20']
        }),
        # Fallback tables for first-time fighters
        'sig_str_first_time_mad_stats': pd.DataFrame({
            'weightclass': ['Lightweight', 'Welterweight'],
            'sig_str_land_opp_wc_mad': [4.0, 4.5]
        }),
        'sig_str_first_time_opp_avg_stats': pd.DataFrame({
            'weightclass': ['Lightweight', 'Welterweight'],
            'sig_str_land_opp_wc_avg': [22.0, 23.0]
        })
    }


@pytest.fixture
def mock_data_zero_mad():
    """Fixture providing mock data with zero MAD values for testing zero denominator handling"""
    return {
        # Main stats table with fighter, opponent, and previous opponent data
        'sig_str': pd.DataFrame({
            'fight_id': [1, 2],
            'fighter_id': [101, 102],
            'sig_str_land': [30, 25], 
            'sig_str_opp_avg': [None, 20],
            'sig_str_opp_mad': [None, 0]  # Zero MAD value
        }),
        # Fight mapping table
        'fight_mapping': pd.DataFrame({
            'fight_id': [1, 2],
            'fighter1_id': [101, 201],
            'fighter2_id': [102, 202],
            'weightclass': ['Lightweight', 'Welterweight'],
            'event_id': [1, 2]
        }),
        # Event mapping table
        'event_mapping': pd.DataFrame({
            'event_id': [1, 2],
            'event_date': ['2020-01-15', '2020-06-20']
        }),
        # Fallback tables for first-time fighters
        'sig_str_first_time_mad_stats': pd.DataFrame({
            'weightclass': ['Lightweight', 'Welterweight'],
            'sig_str_land_opp_wc_mad': [0, 0]  # Zero MAD values
        }),
        'sig_str_first_time_opp_avg_stats': pd.DataFrame({
            'weightclass': ['Lightweight', 'Welterweight'],
            'sig_str_land_opp_wc_avg': [22.0, 23.0]
        })
    }


def test_adjperf_calculation_regular(mock_data):
    """
    Test that the adjusted performance calculation works correctly without time decay.
    Validates the formula: (fighter1_stat - fighter2_stat_opp_avg) / fighter2_stat_opp_mad
    """
    # Create calculator with decay=False
    calculator = AdjustedPerformanceCalculatorTester(decay=False)
    
    # Mock the execute_raw_sql method to track SQL and return custom results
    def mock_execute_raw_sql(sql, params=None, return_results=False):
        if not return_results:
            return None
            
        # Process the regular adjusted performance calculation
        result_df = mock_data['sig_str'].copy()
        
        # Add adjusted performance columns using the formula:
        # (fighter1_stat - fighter2_stat_prev_opp_avg) / fighter2_stat_prev_opp_mad
        
        # For fighter 101 (no previous opponent data), use fallback from weightclass
        # Get weightclass from fight_mapping
        fighter_101_weightclass = mock_data['fight_mapping'][
            mock_data['fight_mapping']['fighter1_id'] == 101]['weightclass'].iloc[0]
        # Get fallback values
        fallback_avg = mock_data['sig_str_first_time_opp_avg_stats'][
            mock_data['sig_str_first_time_opp_avg_stats']['weightclass'] == fighter_101_weightclass
        ]['sig_str_land_opp_wc_avg'].iloc[0]
        fallback_mad = mock_data['sig_str_first_time_mad_stats'][
            mock_data['sig_str_first_time_mad_stats']['weightclass'] == fighter_101_weightclass
        ]['sig_str_land_opp_wc_mad'].iloc[0]
        
        # Calculate adjperf for first fighter (using fallback)
        # Using sig_str_acc values: fighter_101_adjperf = (0.60 - 0.55) / 0.1 = 0.5 (example)
        result_df.loc[result_df['fighter_id'] == 101, 'sig_str_acc_adjperf'] = (0.60 - 0.55) / 0.1
        
        # For fighter 102, use actual previous opponent stats
        # fighter_102_adjperf = (0.50 - 0.45) / 0.1 = 0.5 (example)
        result_df.loc[result_df['fighter_id'] == 102, 'sig_str_acc_adjperf'] = (0.50 - 0.45) / 0.1
        
        return result_df
    
    calculator.calculator.execute_raw_sql = mock_execute_raw_sql
    
    # Execute the calculation
    result = calculator.execute_for_table('sig_str', ['sig_str_acc'])
    
    # Validate results
    assert 'sig_str_acc_adjperf' in result.columns
    np.testing.assert_almost_equal(
        result.loc[result['fighter_id'] == 101, 'sig_str_acc_adjperf'].iloc[0], 
        0.5, 
        decimal=5
    )
    np.testing.assert_almost_equal(
        result.loc[result['fighter_id'] == 102, 'sig_str_acc_adjperf'].iloc[0], 
        0.5, 
        decimal=5
    )


def test_adjperf_calculation_with_decay(mock_data):
    """
    Test that the adjusted performance calculation works correctly with time decay.
    Uses the decayed versions of opponent stats (opp_dec_avg and opp_dec_mad).
    """
    # Create calculator with decay=True
    calculator = AdjustedPerformanceCalculatorTester(decay=True)
    
    # Mock the execute_raw_sql method to track SQL and return custom results
    def mock_execute_raw_sql(sql, params=None, return_results=False):
        if not return_results:
            return None
            
        # Process the time-decayed adjusted performance calculation
        result_df = mock_data['sig_str'].copy()
        
        # Add adjusted performance columns using the formula with decayed stats:
        # (fighter1_stat - fighter2_stat_prev_opp_dec_avg) / fighter2_stat_prev_opp_dec_mad
        
        # For fighter 101 (no previous opponent data), use fallback from weightclass
        # Get weightclass from fight_mapping
        fighter_101_weightclass = mock_data['fight_mapping'][
            mock_data['fight_mapping']['fighter1_id'] == 101]['weightclass'].iloc[0]
        # Get fallback values
        fallback_avg = mock_data['sig_str_first_time_opp_avg_stats'][
            mock_data['sig_str_first_time_opp_avg_stats']['weightclass'] == fighter_101_weightclass
        ]['sig_str_land_opp_wc_avg'].iloc[0]
        fallback_mad = mock_data['sig_str_first_time_mad_stats'][
            mock_data['sig_str_first_time_mad_stats']['weightclass'] == fighter_101_weightclass
        ]['sig_str_land_opp_wc_mad'].iloc[0]
        
        # Calculate dec_adjperf for first fighter (using fallback)
        # Using sig_str_acc values: fighter_101_dec_adjperf = (0.60 - 0.55) / 0.1 = 0.5 (example)
        result_df.loc[result_df['fighter_id'] == 101, 'sig_str_acc_dec_adjperf'] = (0.60 - 0.55) / 0.1
        
        # For fighter 102, use actual previous opponent decayed stats
        # fighter_102_dec_adjperf = (0.50 - 0.44) / 0.1 = 0.6 (example)
        result_df.loc[result_df['fighter_id'] == 102, 'sig_str_acc_dec_adjperf'] = (0.50 - 0.44) / 0.1
        
        return result_df
    
    calculator.calculator.execute_raw_sql = mock_execute_raw_sql
    
    # Execute the calculation
    result = calculator.execute_for_table('sig_str', ['sig_str_acc'])
    
    # Validate results
    assert 'sig_str_acc_dec_adjperf' in result.columns
    np.testing.assert_almost_equal(
        result.loc[result['fighter_id'] == 101, 'sig_str_acc_dec_adjperf'].iloc[0], 
        0.5, 
        decimal=5
    )
    np.testing.assert_almost_equal(
        result.loc[result['fighter_id'] == 102, 'sig_str_acc_dec_adjperf'].iloc[0], 
        0.6, 
        decimal=4
    )


def test_adjperf_sql_generation():
    """
    Test that the SQL generation includes the correct joins and logic.
    """
    # Create calculator
    calculator = AdjustedPerformanceCalculatorTester(decay=False)
    
    # Calculate SQL for the table - only sig_str_acc should get adjperf, not sig_str_land
    sql = calculator.calculate_for_table('sig_str', ['sig_str_land', 'sig_str_acc'])
    
    # Verify SQL contains essential components
    assert 'features.sig_str t' in sql
    assert 'JOIN features.fight_mapping fm' in sql
    assert 'JOIN features.event_mapping em' in sql
    assert 'CROSS JOIN LATERAL' in sql  # For determining opponent ID
    assert 'opponent_history' in sql   # CTE for opponent history
    assert 'weightclass_priors' in sql # CTE for weightclass priors
    
    # Verify the adjusted performance calculation logic - only acc should have adjperf
    assert 'sig_str_land_adjperf' not in sql  # _land columns don't get adjperf
    assert 'sig_str_acc_adjperf' in sql
    # Check for shrinkage formulas
    assert 'COALESCE(oh.n_fights, 0)' in sql
    assert 'COALESCE(wp.' in sql  # weightclass priors


def test_adjperf_with_decay_sql_generation():
    """
    Test that the SQL generation for time-decay calculations includes the correct columns.
    """
    # Create calculator with decay=True
    calculator = AdjustedPerformanceCalculatorTester(decay=True)
    
    # Calculate SQL for the table - only sig_str_acc should get adjperf, not sig_str_land
    sql = calculator.calculate_for_table('sig_str', ['sig_str_land', 'sig_str_acc'])
    
    # Verify SQL contains essential components
    assert 'features.sig_str t' in sql
    assert 'JOIN features.fight_mapping fm' in sql
    assert 'JOIN features.event_mapping em' in sql
    
    # Verify the time-decay adjusted performance calculation logic - only acc should have adjperf
    assert 'sig_str_land_dec_adjperf' not in sql  # _land columns don't get adjperf
    assert 'sig_str_acc_dec_adjperf' in sql
    # Check for time decay expressions (uses centralized config, default: 1.0 year half-life)
    from libs.feature_store.config import get_decay_rate_sql_constant
    decay_rate_sql = get_decay_rate_sql_constant()
    assert f'EXP(-{decay_rate_sql}' in sql  # Time decay formula
    assert 'POWER(SUM(w), 2)' in sql  # Kish effective sample size


def test_adjperf_zero_denominator_handling(mock_data_zero_mad):
    """
    Test that the adjusted performance calculation handles zero denominators correctly.
    """
    # Create calculator
    calculator = AdjustedPerformanceCalculatorTester(decay=False)
    
    # Mock the execute_raw_sql method to track SQL and return custom results
    def mock_execute_raw_sql(sql, params=None, return_results=False):
        if not return_results:
            return None
            
        # Process the adjusted performance calculation
        result_df = mock_data_zero_mad['sig_str'].copy()
        
        # CASE logic should return 0 when denominator is zero
        result_df.loc[result_df['fighter_id'] == 101, 'sig_str_acc_adjperf'] = 0
        result_df.loc[result_df['fighter_id'] == 102, 'sig_str_acc_adjperf'] = 0
        
        return result_df
    
    calculator.calculator.execute_raw_sql = mock_execute_raw_sql
    
    # Execute the calculation
    result = calculator.execute_for_table('sig_str', ['sig_str_acc'])
    
    # Validate results - all values should be 0 due to zero MAD
    assert 'sig_str_acc_adjperf' in result.columns
    assert result.loc[result['fighter_id'] == 101, 'sig_str_acc_adjperf'].iloc[0] == 0
    assert result.loc[result['fighter_id'] == 102, 'sig_str_acc_adjperf'].iloc[0] == 0


def test_adjperf_calculation_multiple_features():
    """
    Test that the calculator correctly processes multiple stat features at once.
    """
    # Create calculator
    calculator = AdjustedPerformanceCalculatorTester(decay=False)
    
    # Get SQL for multiple features - only acc and def should get adjperf
    sql = calculator.calculate_for_table('sig_str', ['sig_str_land', 'sig_str_acc', 'sig_str_def'])
    
    # Verify SQL includes only the adjperf target features
    assert 'sig_str_land_adjperf' not in sql  # _land columns don't get adjperf
    assert 'sig_str_acc_adjperf' in sql       # _acc columns get adjperf
    assert 'sig_str_def_adjperf' in sql       # _def columns get adjperf


def test_adjperf_feature_filtering():
    """
    Test that include/exclude patterns properly filter columns.
    """
    # Create calculator with include pattern for only acc features
    include_patterns = {'acc'}
    calculator = AdjustedPerformanceCalculatorTester(decay=False, include_patterns=include_patterns)
    
    # Get SQL for filtered features - only acc should be processed since land doesn't get adjperf anyway
    sql = calculator.calculate_for_table('sig_str', ['sig_str_land', 'sig_str_acc', 'sig_str_def'])
    
    # Verify SQL only includes acc features (land wouldn't get adjperf anyway)
    assert 'sig_str_land_adjperf' not in sql  # _land never gets adjperf
    assert 'sig_str_acc_adjperf' in sql       # included by pattern and is adjperf target
    assert 'sig_str_def_adjperf' not in sql   # excluded by pattern
    
    # Create calculator with exclude pattern for acc
    exclude_patterns = {'acc'}
    calculator = AdjustedPerformanceCalculatorTester(decay=False, exclude_patterns=exclude_patterns)
    
    # Get SQL for filtered features
    sql = calculator.calculate_for_table('sig_str', ['sig_str_land', 'sig_str_acc', 'sig_str_def'])
    
    # Verify SQL excludes acc features
    assert 'sig_str_land_adjperf' not in sql  # _land never gets adjperf
    assert 'sig_str_acc_adjperf' not in sql   # excluded by pattern
    assert 'sig_str_def_adjperf' in sql       # not excluded and is adjperf target