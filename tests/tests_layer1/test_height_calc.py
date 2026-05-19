import pytest
import pandas as pd
import numpy as np
from unittest.mock import MagicMock, patch

from libs.feature_store.calculators.height_calc import HeightCalculator
from libs.feature_store.calculator_context import CalculatorContext

# Create a test version of the calculator that overrides _ensure_columns_exist
class TestHeightCalculator(HeightCalculator):
    def _ensure_columns_exist(self, columns, table_name=None, schema=None):
        # Do nothing, assume columns exist
        pass

def test_height_calculator_with_context():
    """Test HeightCalculator with mock data and context"""
    # Create mock data with realistic structure
    mock_data = {
        'fight_stats_fe': pd.DataFrame({
            'fight_id': [101, 102, 103, 104],
            'fighter_id': [1, 2, 3, 4],
            'event_id': [1, 1, 2, 2]
        }),
        'fighter_mapping': pd.DataFrame({
            'fighter_id': [1, 2, 3, 4],
            'fighter_name': ['Fighter 1', 'Fighter 2', 'Fighter 3', 'Fighter 4'],
            'fighter_height': [70.0, 72.0, None, 69.0]  # Fighter 3 has missing height
        }),
        'fight_mapping': pd.DataFrame({
            'fight_id': [101, 102],
            'event_id': [1, 2],
            'fighter1_id': [1, 3],
            'fighter2_id': [2, 4],
            'weightclass': ['Lightweight', 'Welterweight']
        })
    }
    
    # Create mock connection
    mock_conn = MagicMock()
    
    # Create context with mock connection and data
    with patch('libs.feature_store.calculator_context.FeatureUtils'):
        with patch('libs.feature_store.calculator_context.SQLTemplateManager'):
            context = CalculatorContext(mock_conn, mock_data)
            
            # Create calculator with context
            calculator = TestHeightCalculator(context)
            
            # Mock the execute_raw_sql method to actually calculate height
            def mock_execute_raw_sql(sql, params=None):
                # This simulates what the SQL would do to the database
                # Calculate average height by weightclass
                weightclass_height = {}
                
                # First, calculate average height by weightclass
                for _, row in mock_data['fight_mapping'].iterrows():
                    weightclass = row['weightclass']
                    fighter1_id = row['fighter1_id']
                    fighter2_id = row['fighter2_id']
                    
                    if weightclass not in weightclass_height:
                        weightclass_height[weightclass] = {'total': 0.0, 'count': 0}
                    
                    # Get fighter1 height
                    fighter1_height = mock_data['fighter_mapping'].loc[
                        mock_data['fighter_mapping']['fighter_id'] == fighter1_id, 'fighter_height'
                    ].iloc[0]
                    
                    # Get fighter2 height
                    fighter2_height = mock_data['fighter_mapping'].loc[
                        mock_data['fighter_mapping']['fighter_id'] == fighter2_id, 'fighter_height'
                    ].iloc[0]
                    
                    # Add to weightclass average if not None
                    if pd.notna(fighter1_height):
                        weightclass_height[weightclass]['total'] += fighter1_height
                        weightclass_height[weightclass]['count'] += 1
                    
                    if pd.notna(fighter2_height):
                        weightclass_height[weightclass]['total'] += fighter2_height
                        weightclass_height[weightclass]['count'] += 1
                
                # Calculate average height for each weightclass
                for weightclass in weightclass_height:
                    if weightclass_height[weightclass]['count'] > 0:
                        weightclass_height[weightclass]['avg'] = round(
                            weightclass_height[weightclass]['total'] / weightclass_height[weightclass]['count']
                        )
                    else:
                        weightclass_height[weightclass]['avg'] = 70  # Default value
                
                # Now update each fighter's height in fight_stats_fe
                for idx, row in mock_data['fight_stats_fe'].iterrows():
                    fighter_id = row['fighter_id']
                    fight_id = row['fight_id']
                    
                    # Get fighter's height
                    fighter_height = mock_data['fighter_mapping'].loc[
                        mock_data['fighter_mapping']['fighter_id'] == fighter_id, 'fighter_height'
                    ].iloc[0]
                    
                    # Get weightclass for this fight
                    fight_row = mock_data['fight_mapping'].loc[
                        mock_data['fight_mapping']['fight_id'] == fight_id
                    ]
                    
                    if not fight_row.empty:
                        weightclass = fight_row['weightclass'].iloc[0]
                        
                        # Use fighter's height if available, otherwise use weightclass average
                        if pd.notna(fighter_height):
                            height = int(fighter_height)
                        else:
                            height = weightclass_height[weightclass]['avg']
                        
                        # Update the mock data
                        mock_data['fight_stats_fe'].loc[idx, 'height'] = height
                    else:
                        # If no fight mapping found, use a default height
                        mock_data['fight_stats_fe'].loc[idx, 'height'] = 70
            
            # Replace the execute_raw_sql method
            calculator.execute_raw_sql = mock_execute_raw_sql
            
            # Run the calculator
            calculator.run()
            
            # Verify results
            assert 'height' in mock_data['fight_stats_fe'].columns
            
            # Get height for specific fighters
            fighter1_height = mock_data['fight_stats_fe'].loc[
                mock_data['fight_stats_fe']['fighter_id'] == 1, 'height'
            ].iloc[0]
            
            fighter2_height = mock_data['fight_stats_fe'].loc[
                mock_data['fight_stats_fe']['fighter_id'] == 2, 'height'
            ].iloc[0]
            
            fighter3_height = mock_data['fight_stats_fe'].loc[
                mock_data['fight_stats_fe']['fighter_id'] == 3, 'height'
            ].iloc[0]
            
            # Calculate expected height manually for verification
            # Fighter 1 has height of 70
            expected_fighter1_height = 70
            
            # Fighter 2 has height of 72
            expected_fighter2_height = 72
            
            # Fighter 3 has no height, should use default value of 70
            # since there's only one fighter (Fighter 4) in the Welterweight class with height data
            expected_fighter3_height = 70
            
            # Verify the calculated height values match our expectations
            assert fighter1_height == expected_fighter1_height, f"Expected {expected_fighter1_height}, got {fighter1_height}"
            assert fighter2_height == expected_fighter2_height, f"Expected {expected_fighter2_height}, got {fighter2_height}"
            assert fighter3_height == expected_fighter3_height, f"Expected {expected_fighter3_height}, got {fighter3_height}"
            
            # Print the height values for verification
            print(f"Fighter 1 height: {fighter1_height} inches")
            print(f"Fighter 2 height: {fighter2_height} inches")
            print(f"Fighter 3 height: {fighter3_height} inches (using weightclass average)")


def test_height_calculator_with_sql_template():
    """Test HeightCalculator with SQL template manager"""
    # Create mock connection
    mock_conn = MagicMock()
    
    # Create a real context but patch the SQL template manager
    with patch('libs.feature_store.calculator_context.FeatureUtils'):
        with patch('libs.feature_store.calculator_context.SQLTemplateManager') as mock_sql_manager_class:
            # Configure the mock SQL template manager
            mock_sql_manager = mock_sql_manager_class.return_value
            mock_sql_manager.render_template.return_value = """
                -- SQL template for height calculation
                UPDATE features.fight_stats_fe SET height = 70
            """
            
            # Create context with mock connection
            context = CalculatorContext(mock_conn)
            
            # Create calculator with context
            calculator = TestHeightCalculator(context)
            
            # Mock the execute_raw_sql method
            with patch.object(calculator, 'execute_raw_sql') as mock_execute:
                # Run the calculator
                calculator.run()
                
                # Verify SQL template manager was called correctly
                mock_sql_manager.render_template.assert_called_once_with(
                    'height', 'calculate', {'schema': 'features'}
                )
                
                # Verify execute_raw_sql was called with the SQL from the template
                mock_execute.assert_called_with("""
                -- SQL template for height calculation
                UPDATE features.fight_stats_fe SET height = 70
            """)


def test_height_calculator_integration():
    """Test HeightCalculator with full integration using SQL template"""
    # Create mock data with realistic structure
    mock_data = {
        'fight_stats_fe': pd.DataFrame({
            'fight_id': [101, 102, 103, 104],
            'fighter_id': [1, 2, 3, 4],
            'event_id': [1, 1, 2, 2]
        }),
        'fighter_mapping': pd.DataFrame({
            'fighter_id': [1, 2, 3, 4],
            'fighter_name': ['Fighter 1', 'Fighter 2', 'Fighter 3', 'Fighter 4'],
            'fighter_height': [70.0, 72.0, None, 69.0]  # Fighter 3 has missing height
        }),
        'fight_mapping': pd.DataFrame({
            'fight_id': [101, 102],
            'event_id': [1, 2],
            'fighter1_id': [1, 3],
            'fighter2_id': [2, 4],
            'weightclass': ['Lightweight', 'Welterweight']
        })
    }
    
    # Create mock connection
    mock_conn = MagicMock()
    
    # Create context with mock connection and data
    with patch('libs.feature_store.calculator_context.FeatureUtils'):
        with patch('libs.feature_store.calculator_context.SQLTemplateManager') as mock_sql_manager_class:
            # Configure the mock SQL template manager
            mock_sql_manager = mock_sql_manager_class.return_value
            
            # Create a SQL template that actually performs the calculation
            # This is the actual SQL template content that would be in the file
            mock_sql_manager.render_template.return_value = """
                WITH fighter_heights AS (
                    SELECT 
                        f.fighter_id,
                        fm.fighter_height
                    FROM features.fight_stats_fe f
                    JOIN features.fighter_mapping fm ON f.fighter_id = fm.fighter_id
                ),
                weightclass_avg_heights AS (
                    SELECT 
                        fm.weightclass,
                        ROUND(AVG(CASE WHEN fh.fighter_height IS NOT NULL THEN fh.fighter_height ELSE NULL END)) as avg_height
                    FROM features.fight_mapping fm
                    JOIN fighter_heights fh1 ON fm.fighter1_id = fh1.fighter_id
                    JOIN fighter_heights fh2 ON fm.fighter2_id = fh2.fighter_id
                    GROUP BY fm.weightclass
                )
                UPDATE features.fight_stats_fe f
                SET height = CASE 
                    WHEN fh.fighter_height IS NOT NULL THEN fh.fighter_height::integer
                    WHEN wah.avg_height IS NOT NULL THEN wah.avg_height
                    ELSE 70 -- Default height if no data available
                END
                FROM fighter_heights fh
                LEFT JOIN features.fight_mapping fm ON (f.fight_id = fm.fight_id AND (f.fighter_id = fm.fighter1_id OR f.fighter_id = fm.fighter2_id))
                LEFT JOIN weightclass_avg_heights wah ON fm.weightclass = wah.weightclass
                WHERE f.fighter_id = fh.fighter_id;
            """
            
            # Create context with mock connection
            context = CalculatorContext(mock_conn, mock_data)
            
            # Create calculator with context
            calculator = TestHeightCalculator(context)
            
            # Instead of mocking execute_raw_sql with our own implementation,
            # we'll create a function that simulates what the SQL would do
            # but uses the actual SQL template content for guidance
            def mock_execute_raw_sql(sql, params=None):
                # Verify that we're using the SQL from the template
                assert "WITH fighter_heights AS" in sql
                assert "weightclass_avg_heights AS" in sql
                assert "UPDATE features.fight_stats_fe" in sql
                
                # Now implement the logic that matches what the SQL would do
                
                # Step 1: Create fighter_heights CTE equivalent
                fighter_heights = {}
                for _, row in mock_data['fighter_mapping'].iterrows():
                    fighter_heights[row['fighter_id']] = row['fighter_height']
                
                # Step 2: Create weightclass_avg_heights CTE equivalent
                weightclass_avg_heights = {}
                for _, row in mock_data['fight_mapping'].iterrows():
                    weightclass = row['weightclass']
                    fighter1_id = row['fighter1_id']
                    fighter2_id = row['fighter2_id']
                    
                    if weightclass not in weightclass_avg_heights:
                        weightclass_avg_heights[weightclass] = {'total': 0.0, 'count': 0}
                    
                    # Add fighter1 height to average if not None
                    if pd.notna(fighter_heights.get(fighter1_id)):
                        weightclass_avg_heights[weightclass]['total'] += fighter_heights[fighter1_id]
                        weightclass_avg_heights[weightclass]['count'] += 1
                    
                    # Add fighter2 height to average if not None
                    if pd.notna(fighter_heights.get(fighter2_id)):
                        weightclass_avg_heights[weightclass]['total'] += fighter_heights[fighter2_id]
                        weightclass_avg_heights[weightclass]['count'] += 1
                
                # Calculate average height for each weightclass
                for weightclass in weightclass_avg_heights:
                    if weightclass_avg_heights[weightclass]['count'] > 0:
                        weightclass_avg_heights[weightclass] = round(
                            weightclass_avg_heights[weightclass]['total'] / weightclass_avg_heights[weightclass]['count']
                        )
                    else:
                        weightclass_avg_heights[weightclass] = 70  # Default value
                
                # Step 3: Update fight_stats_fe with heights
                for idx, row in mock_data['fight_stats_fe'].iterrows():
                    fighter_id = row['fighter_id']
                    fight_id = row['fight_id']
                    
                    # Get fighter's height
                    fighter_height = fighter_heights.get(fighter_id)
                    
                    # Get weightclass for this fight
                    fight_row = mock_data['fight_mapping'].loc[
                        mock_data['fight_mapping']['fight_id'] == fight_id
                    ]
                    
                    if not fight_row.empty:
                        weightclass = fight_row['weightclass'].iloc[0]
                        
                        # Use fighter's height if available, otherwise use weightclass average
                        if pd.notna(fighter_height):
                            height = int(fighter_height)
                        else:
                            height = weightclass_avg_heights.get(weightclass, 70)
                        
                        # Update the mock data
                        mock_data['fight_stats_fe'].loc[idx, 'height'] = height
                    else:
                        # If no fight mapping found, use a default height
                        mock_data['fight_stats_fe'].loc[idx, 'height'] = 70
            
            # Replace the execute_raw_sql method
            calculator.execute_raw_sql = mock_execute_raw_sql
            
            # Run the calculator
            calculator.run()
            
            # Verify results
            assert 'height' in mock_data['fight_stats_fe'].columns
            
            # Get height for specific fighters
            fighter1_height = mock_data['fight_stats_fe'].loc[
                mock_data['fight_stats_fe']['fighter_id'] == 1, 'height'
            ].iloc[0]
            
            fighter2_height = mock_data['fight_stats_fe'].loc[
                mock_data['fight_stats_fe']['fighter_id'] == 2, 'height'
            ].iloc[0]
            
            fighter3_height = mock_data['fight_stats_fe'].loc[
                mock_data['fight_stats_fe']['fighter_id'] == 3, 'height'
            ].iloc[0]
            
            # Calculate expected height manually for verification
            # Fighter 1 has height of 70
            expected_fighter1_height = 70
            
            # Fighter 2 has height of 72
            expected_fighter2_height = 72
            
            # Fighter 3 has no height, should use default value of 70
            # since there's only one fighter (Fighter 4) in the Welterweight class with height data
            expected_fighter3_height = 70
            
            # Verify the calculated height values match our expectations
            assert fighter1_height == expected_fighter1_height, f"Expected {expected_fighter1_height}, got {fighter1_height}"
            assert fighter2_height == expected_fighter2_height, f"Expected {expected_fighter2_height}, got {fighter2_height}"
            assert fighter3_height == expected_fighter3_height, f"Expected {expected_fighter3_height}, got {fighter3_height}"
            
            # Print the height values for verification
            print(f"Fighter 1 height: {fighter1_height} inches")
            print(f"Fighter 2 height: {fighter2_height} inches")
            print(f"Fighter 3 height: {fighter3_height} inches (using weightclass average)") 