import pytest
import pandas as pd
import numpy as np
from unittest.mock import MagicMock, patch, call
from sqlalchemy import create_engine, text
from libs.feature_store.base_calculator import BaseCalculator
from libs.feature_store.calculator_context import CalculatorContext
from libs.feature_store.base import BaseFeatureStore


class TestBaseCalculatorExpert:
    """
    Expert-level unit tests for BaseCalculator to identify data corruption bugs.
    """
    
    def setup_method(self):
        """Set up test fixtures for each test"""
        # Create mock connection
        self.mock_conn = MagicMock()
        
        # Create a concrete test implementation of BaseCalculator
        class ConcreteCalculator(BaseCalculator):
            def __init__(self, conn, calculator_type='single_table'):
                super().__init__(conn, calculator_type)
                self.schema = 'features'
                
            def get_features(self, table_name: str = None):
                return ['test_feature']
                
            def save_for_table(self, table_name: str, columns=None, result_df=None):
                return pd.DataFrame()
        
        self.calculator = ConcreteCalculator(self.mock_conn)
        
        # Mock the logger to avoid issues
        self.calculator.logger = MagicMock()
        
        # Create test data that matches the PerCalculator issue
        self.test_data = pd.DataFrame({
            'fight_id': [1, 2, 3, 4, 5],
            'fighter_id': [101, 102, 103, 104, 105],
            'event_id': [1, 1, 2, 2, 3],
            'ko_per_sig_str_land': [0.013704, 0.025000, 0.000000, 0.554090, 0.100000],
            'clinch_per_sig_str_land': [0.138278, 0.200000, 0.777507, 0.050000, 0.300000],
            'body_leg_per_sig_str_land': [0.381401, 0.500000, 0.881990, 0.200000, 0.600000],
            'head_per_sig_str_land': [0.627707, 0.800000, 0.962927, 0.400000, 0.750000],
            'distance_per_sig_str_land': [0.736311, 0.900000, 1.092937, 0.500000, 0.850000]
        })

    def test_execute_calculator_update_basic_functionality(self):
        """Test basic execute_calculator_update functionality."""
        # Mock execute_raw_sql to return our test data
        self.calculator.execute_raw_sql = MagicMock(return_value=self.test_data)
        
        # Mock _ensure_columns_exist to avoid DB calls
        self.calculator._ensure_columns_exist = MagicMock()
        
        # Mock bulk_update_dataframe to capture what gets passed to it
        self.calculator.bulk_update_dataframe = MagicMock()
        
        # Test columns to update
        new_columns = ['ko_per_sig_str_land', 'clinch_per_sig_str_land']
        
        # Execute the method
        result = self.calculator.execute_calculator_update(
            calculation_sql="SELECT * FROM test_table",
            table_name="test_table",
            new_columns=new_columns,
            schema="features"
        )
        
        # Verify the method was called correctly
        self.calculator.execute_raw_sql.assert_called_once()
        self.calculator._ensure_columns_exist.assert_called_once()
        self.calculator.bulk_update_dataframe.assert_called_once()
        
        # Check that the result DataFrame is returned
        assert result is not None
        assert not result.empty
        assert 'ko_per_sig_str_land' in result.columns
        assert 'clinch_per_sig_str_land' in result.columns

    def test_execute_calculator_update_type_conversion_bug(self):
        """Test the specific type conversion logic that might be corrupting data."""
        # Mock execute_raw_sql to return our test data
        self.calculator.execute_raw_sql = MagicMock(return_value=self.test_data.copy())
        
        # Mock _ensure_columns_exist
        self.calculator._ensure_columns_exist = MagicMock()
        
        # Mock bulk_update_dataframe to capture the DataFrame that gets passed
        captured_dataframes = []
        def capture_dataframe(df, table_name, schema, key_columns):
            captured_dataframes.append(df.copy())
            
        self.calculator.bulk_update_dataframe = MagicMock(side_effect=capture_dataframe)
        
        # Test with columns that have different suffix patterns
        test_cases = [
            # Test per_sig_str_land columns (the ones that are failing)
            (['ko_per_sig_str_land'], 'ko_per_sig_str_land'),
            (['clinch_per_sig_str_land'], 'clinch_per_sig_str_land'),
            (['body_leg_per_sig_str_land'], 'body_leg_per_sig_str_land'),
            
            # Test other patterns that might work differently
            (['head_per_sig_str_land'], 'head_per_sig_str_land'),
            (['distance_per_sig_str_land'], 'distance_per_sig_str_land'),
        ]
        
        for new_columns, test_column in test_cases:
            captured_dataframes.clear()
            
            # Execute the method
            result = self.calculator.execute_calculator_update(
                calculation_sql="SELECT * FROM test_table",
                table_name="test_table",
                new_columns=new_columns,
                schema="features"
            )
            
            # Check the original data
            original_values = self.test_data[test_column].values
            original_non_zero = (original_values > 0).sum()
            original_avg = original_values.mean()
            
            # Check the result data
            result_values = result[test_column].values
            result_non_zero = (result_values > 0).sum()
            result_avg = result_values.mean()
            
            # Check the captured DataFrame (what gets passed to bulk_update_dataframe)
            if captured_dataframes:
                captured_df = captured_dataframes[0]
                captured_values = captured_df[test_column].values
                captured_non_zero = (captured_values > 0).sum()
                captured_avg = captured_values.mean()
                
                print(f"\n=== TYPE CONVERSION TEST: {test_column} ===")
                print(f"Original: {original_non_zero} non-zero, avg={original_avg:.6f}")
                print(f"Result: {result_non_zero} non-zero, avg={result_avg:.6f}")
                print(f"Captured: {captured_non_zero} non-zero, avg={captured_avg:.6f}")
                print(f"Original dtype: {self.test_data[test_column].dtype}")
                print(f"Result dtype: {result[test_column].dtype}")
                print(f"Captured dtype: {captured_df[test_column].dtype}")
                
                # Assert that data is not corrupted
                assert original_non_zero == result_non_zero, f"Non-zero count changed for {test_column}: {original_non_zero} -> {result_non_zero}"
                assert abs(original_avg - result_avg) < 0.000001, f"Average changed for {test_column}: {original_avg} -> {result_avg}"
                assert original_non_zero == captured_non_zero, f"Captured non-zero count wrong for {test_column}: {original_non_zero} -> {captured_non_zero}"

    def test_execute_calculator_update_suffix_pattern_analysis(self):
        """Test how different column name suffixes are handled by type conversion."""
        # Mock execute_raw_sql
        self.calculator.execute_raw_sql = MagicMock(return_value=self.test_data.copy())
        self.calculator._ensure_columns_exist = MagicMock()
        self.calculator.bulk_update_dataframe = MagicMock()
        
        # Test different suffix patterns to see which ones trigger different type conversions
        test_columns = [
            'ko_per_sig_str_land',      # Should this be treated as float or int?
            'clinch_per_sig_str_land',  # Should this be treated as float or int?
            'td_per_sig_str_att',       # Should this be treated as float or int?
            'sig_str_acc',              # Should be float (accuracy)
            'td_land',                  # Should be int (count)
            'sig_str_land'              # Should be int (count)
        ]
        
        for col in test_columns:
            # Create test data for this column
            test_df = pd.DataFrame({
                'fight_id': [1, 2, 3],
                'fighter_id': [101, 102, 103],
                col: [0.123456, 0.789012, 0.456789]  # Precise float values
            })
            
            self.calculator.execute_raw_sql.return_value = test_df
            
            # Execute the method
            result = self.calculator.execute_calculator_update(
                calculation_sql="SELECT * FROM test_table",
                table_name="test_table", 
                new_columns=[col],
                schema="features"
            )
            
            # Check if the values were corrupted by type conversion
            original_values = test_df[col].values
            result_values = result[col].values
            
            print(f"\n=== SUFFIX PATTERN TEST: {col} ===")
            print(f"Original values: {original_values}")
            print(f"Result values: {result_values}")
            print(f"Original dtype: {test_df[col].dtype}")
            print(f"Result dtype: {result[col].dtype}")
            
            # Check for data corruption
            if not np.array_equal(original_values, result_values):
                print(f"🚨 DATA CORRUPTION DETECTED in {col}!")
                print(f"Difference: {original_values - result_values}")

    def test_execute_calculator_update_per_sig_str_land_specific_bug(self):
        """Test specifically for the per_sig_str_land suffix pattern bug."""
        # Create data that exactly matches what we see in the debug output
        bug_test_data = pd.DataFrame({
            'fight_id': [1, 2, 3, 4, 5],
            'fighter_id': [101, 102, 103, 104, 105],
            'event_id': [1, 1, 2, 2, 3],
            'ko_per_sig_str_land': [0.013704, 0.025000, 0.000000, 0.554090, 0.100000],
            'clinch_per_sig_str_land': [0.138278, 0.200000, 0.777507, 0.050000, 0.300000],
            'body_leg_per_sig_str_land': [0.381401, 0.500000, 0.881990, 0.200000, 0.600000]
        })
        
        # Mock the dependencies
        self.calculator.execute_raw_sql = MagicMock(return_value=bug_test_data.copy())
        self.calculator._ensure_columns_exist = MagicMock()
        
        # Capture what gets passed to bulk_update_dataframe
        captured_df = None
        def capture_bulk_update(df, table_name, schema, key_columns):
            nonlocal captured_df
            captured_df = df.copy()
            
        self.calculator.bulk_update_dataframe = MagicMock(side_effect=capture_bulk_update)
        
        # Test each problematic column individually
        for col in ['ko_per_sig_str_land', 'clinch_per_sig_str_land', 'body_leg_per_sig_str_land']:
            captured_df = None
            
            # Execute the method
            result = self.calculator.execute_calculator_update(
                calculation_sql=f"SELECT fight_id, fighter_id, event_id, {col} FROM test_table",
                table_name="test_table",
                new_columns=[col],
                schema="features"
            )
            
            # Get the original and result values
            original_values = bug_test_data[col].values
            result_values = result[col].values
            
            # Check if captured DataFrame has the corruption
            if captured_df is not None:
                captured_values = captured_df[col].values
                
                print(f"\n=== BUG REPRODUCTION TEST: {col} ===")
                print(f"Original: non-zero={np.sum(original_values > 0)}, avg={np.mean(original_values):.6f}")
                print(f"Result: non-zero={np.sum(result_values > 0)}, avg={np.mean(result_values):.6f}")
                print(f"Captured: non-zero={np.sum(captured_values > 0)}, avg={np.mean(captured_values):.6f}")
                print(f"Original dtype: {bug_test_data[col].dtype}")
                print(f"Result dtype: {result[col].dtype}")
                print(f"Captured dtype: {captured_df[col].dtype}")
                
                # Check if values match exactly
                if not np.allclose(original_values, result_values, rtol=1e-10):
                    print(f"🚨 RESULT CORRUPTION: Values changed during execute_calculator_update!")
                    print(f"Original sample: {original_values[:3]}")
                    print(f"Result sample: {result_values[:3]}")
                    
                if not np.allclose(original_values, captured_values, rtol=1e-10):
                    print(f"🚨 BULK UPDATE CORRUPTION: Values changed before bulk_update_dataframe!")
                    print(f"Original sample: {original_values[:3]}")
                    print(f"Captured sample: {captured_values[:3]}")
                    
                # The test should fail if data is corrupted
                assert np.allclose(original_values, result_values, rtol=1e-10), f"Data corruption in {col} result"
                assert np.allclose(original_values, captured_values, rtol=1e-10), f"Data corruption in {col} before bulk update"

    def test_type_conversion_logic_analysis(self):
        """Test the specific type conversion logic that might be causing corruption."""
        # Test data with precise float values
        test_df = pd.DataFrame({
            'fight_id': [1, 2, 3],
            'fighter_id': [101, 102, 103],
            'test_per_sig_str_land': [0.123456789, 0.987654321, 0.555555555],  # Precise floats
            'test_acc': [0.123456789, 0.987654321, 0.555555555],               # Should stay float
            'test_land': [0.123456789, 0.987654321, 0.555555555],              # Might get converted to int
            'test_ratio': [0.123456789, 0.987654321, 0.555555555],             # Should stay float
        })
        
        # Test the type conversion logic directly
        for col in ['test_per_sig_str_land', 'test_acc', 'test_land', 'test_ratio']:
            original_values = test_df[col].copy()
            test_col_df = test_df[[col]].copy()
            
            # Apply the same type conversion logic as execute_calculator_update
            try:
                # This is the exact logic from lines 251-262 in base.py
                test_col_df[col] = pd.to_numeric(test_col_df[col], errors='coerce')
                
                # Check for integer columns based on suffix patterns
                if any(suffix in col for suffix in ['_acc', '_def', '_ratio', '_per_min']):
                    # These are typically percentage/ratio fields stored as float
                    test_col_df[col] = test_col_df[col].astype(float)
                elif col.endswith(('_land', '_att')):
                    # These are typically count fields that should be integers
                    test_col_df[col] = test_col_df[col].fillna(0).astype(int)
                    
            except Exception as e:
                print(f"Error converting {col}: {e}")
                
            converted_values = test_col_df[col].values
            
            print(f"\n=== TYPE CONVERSION ANALYSIS: {col} ===")
            print(f"Original: {original_values.values}")
            print(f"Converted: {converted_values}")
            print(f"Original dtype: {original_values.dtype}")
            print(f"Converted dtype: {test_col_df[col].dtype}")
            
            # Check for precision loss or corruption
            if col.endswith('_land'):
                # This should be converted to int, so we expect truncation
                expected_values = original_values.fillna(0).astype(int)
                print(f"Expected (int): {expected_values.values}")
                if not np.array_equal(converted_values, expected_values):
                    print(f"🚨 UNEXPECTED CONVERSION for {col}!")
            else:
                # This should stay as float
                if not np.allclose(original_values, converted_values, rtol=1e-10):
                    print(f"🚨 FLOAT PRECISION LOSS for {col}!")
                    print(f"Difference: {original_values.values - converted_values}")

    def test_per_sig_str_land_suffix_pattern_bug(self):
        """Test if columns ending with 'per_sig_str_land' are being misclassified."""
        # The bug might be that 'ko_per_sig_str_land' ends with '_land' 
        # and is being treated as a count field (integer) instead of a ratio field (float)
        
        test_columns = [
            'ko_per_sig_str_land',      # Ends with '_land' - might be treated as integer!
            'clinch_per_sig_str_land',  # Ends with '_land' - might be treated as integer!
            'body_leg_per_sig_str_land', # Ends with '_land' - might be treated as integer!
            'head_per_sig_str_land',    # Ends with '_land' - might be treated as integer!
            'distance_per_sig_str_land' # Ends with '_land' - might be treated as integer!
        ]
        
        for col in test_columns:
            # Test the suffix pattern logic
            ends_with_land = col.endswith('_land')
            ends_with_att = col.endswith('_att')
            has_ratio_patterns = any(suffix in col for suffix in ['_acc', '_def', '_ratio', '_per_min'])
            
            print(f"\n=== SUFFIX ANALYSIS: {col} ===")
            print(f"Ends with '_land': {ends_with_land}")
            print(f"Ends with '_att': {ends_with_att}")
            print(f"Has ratio patterns: {has_ratio_patterns}")
            
            if ends_with_land or ends_with_att:
                print(f"🚨 POTENTIAL BUG: {col} will be converted to INTEGER!")
                print(f"   This is WRONG - per_sig_str_land features should be FLOAT ratios!")
            
            # This is the exact bug! Columns like 'ko_per_sig_str_land' end with '_land'
            # so they get converted to integers, which truncates 0.013704 to 0!
            if col.endswith('_land') and 'per_sig_str' in col:
                print(f"🎯 BUG CONFIRMED: {col} is a ratio but treated as integer count!")

    def test_reproduce_exact_corruption_scenario(self):
        """Reproduce the exact scenario from the debug output."""
        # Create data that exactly matches the debug output
        debug_data = pd.DataFrame({
            'fight_id': [1, 2, 3, 4, 5],
            'fighter_id': [101, 102, 103, 104, 105],
            'event_id': [1, 1, 2, 2, 3],
            'ko_per_sig_str_land': [0.013704, 0.025000, 0.000000, 0.554090, 0.100000]
        })
        
        # Mock dependencies
        self.calculator.execute_raw_sql = MagicMock(return_value=debug_data.copy())
        self.calculator._ensure_columns_exist = MagicMock()
        self.calculator.bulk_update_dataframe = MagicMock()
        
        # Execute with the problematic column
        result = self.calculator.execute_calculator_update(
            calculation_sql="SELECT fight_id, fighter_id, event_id, ko_per_sig_str_land FROM test",
            table_name="ko",
            new_columns=['ko_per_sig_str_land'],
            schema="features"
        )
        
        # Check the exact values
        original_avg = debug_data['ko_per_sig_str_land'].mean()
        result_avg = result['ko_per_sig_str_land'].mean()
        
        original_non_zero = (debug_data['ko_per_sig_str_land'] > 0).sum()
        result_non_zero = (result['ko_per_sig_str_land'] > 0).sum()
        
        print(f"\n=== EXACT CORRUPTION REPRODUCTION ===")
        print(f"Original avg: {original_avg:.6f} ({original_non_zero} non-zero)")
        print(f"Result avg: {result_avg:.6f} ({result_non_zero} non-zero)")
        print(f"Original values: {debug_data['ko_per_sig_str_land'].values}")
        print(f"Result values: {result['ko_per_sig_str_land'].values}")
        print(f"Original dtype: {debug_data['ko_per_sig_str_land'].dtype}")
        print(f"Result dtype: {result['ko_per_sig_str_land'].dtype}")
        
        # This should expose the exact corruption mechanism
        if original_avg != result_avg or original_non_zero != result_non_zero:
            print(f"🚨 CORRUPTION REPRODUCED!")
            
            # Check if it's the _land suffix causing integer conversion
            if result['ko_per_sig_str_land'].dtype == 'int64' or result['ko_per_sig_str_land'].dtype == 'int32':
                print(f"🎯 ROOT CAUSE: Column ending with '_land' was converted to INTEGER!")
                print(f"   Float values like 0.013704 get truncated to 0!")
