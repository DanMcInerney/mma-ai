from typing import List, Dict, Optional, Any
import pandas as pd
from libs.feature_store.base_calculator import BaseCalculator
import numpy as np
from libs.feature_store.calculator_context import CalculatorContext


class TimeSecCalculator(BaseCalculator):
    """
    Calculator for computing time_sec features.
    
    This calculator computes the time in seconds for each round of a fight,
    based on the time_format and end_time from the fight_mapping table.
    """

    def __init__(self, conn_or_context, calculator_type='single_table'):
        """
        Initialize with either a connection or a calculator context.
        
        Args:
            conn_or_context: SQLAlchemy connection or CalculatorContext
            calculator_type: Type of calculator ('single_table', 'multi_table', 'cross_table')
                             Single_table is used for this calculator
        """
        # Handle both connection and context for backward compatibility
        if isinstance(conn_or_context, CalculatorContext):
            self.context = conn_or_context
            super().__init__(conn_or_context.connection, calculator_type)
        else:
            self.context = CalculatorContext(conn_or_context)
            super().__init__(conn_or_context, calculator_type)
            
        self.fight_mapping = None
        self.features = None
        self.feature_type = 'time_sec'  # Used for SQL template identifier
        
        # Required for single_table calculator type
        self.table_name = 'fight_stats_fe'
        
        # Configure column filtering patterns
        self.add_include_pattern('time_sec')  # Only process time-related columns
        
        # Setup execution plan
        self.execution_plan.add_operation(
            'load_features', 
            self.get_features
        )
        self.execution_plan.add_operation(
            'calculate_features',
            self.calculate
        )
        self.execution_plan.add_operation(
            'save_features',
            self.save
        )

    def get_features(self, table_name: str = None) -> List[str]:
        """
        Load fight mapping data with required columns using SQL template
        
        Args:
            table_name: Optional table name (defaults to self.table_name)
            
        Returns:
            List of feature column names
        """
        # Use table_name if provided, otherwise use default
        if table_name is None:
            table_name = self.table_name
            
        # Use SQL template if available
        if hasattr(self.context, 'sql_manager') and self.context.sql_manager:
            query = self.context.sql_manager.render_template(
                self.feature_type, 
                'get_features',
                {'schema': self.context.schema}
            )
            self.fight_mapping = pd.read_sql(query, self.conn)
        else:
            # Fallback to direct SQL for backward compatibility
            self.fight_mapping = pd.read_sql("""
                SELECT 
                    fm.fight_id,
                    fs.fighter_id,
                    fm.end_round,
                    fm.end_time,
                    fm.time_format
                FROM features.fight_mapping fm
                JOIN features.fight_stats_core fs ON fm.fight_id = fs.fight_id
            """, self.conn)
        
        # Determine maximum number of rounds to prepare column list
        max_rounds = max(tf.count(',') + 1 for tf in self.fight_mapping['time_format'])
        # Return list of feature columns created by this calculator
        return [f'time_sec_rd{round_num}' for round_num in range(1, max_rounds + 1)]

    def calculate(self, table_name: str = None, columns: List[str] = None) -> Dict[str, Any]:
        """
        Calculate time_sec for each round based on time_format and end_time
        
        Args:
            table_name: Optional table name (defaults to self.table_name)
            columns: Optional list of columns to calculate (not used in this calculator)
            
        Returns:
            Dictionary of operation results
        """
        # Use table_name if provided, otherwise use default
        if table_name is None:
            table_name = self.table_name
            
        self.features = self.fight_mapping[['fight_id', 'fighter_id']].copy()
        
        # First, determine the maximum number of rounds in the dataset
        max_rounds = max(tf.count(',') + 1 for tf in self.fight_mapping['time_format'])

        # Create a 2D array with consistent shape, padding with zeros
        round_seconds = np.zeros((len(self.fight_mapping), max_rounds))
        for i, tf in enumerate(self.fight_mapping['time_format']):
            values = [int(x) for x in tf.split(',')]
            round_seconds[i, :len(values)] = values
        
        # Create columns for each round
        for round_num in range(1, max_rounds + 1):
            col_name = f'time_sec_rd{round_num}'
            
            # For completed rounds before the final round, use the full round time
            is_earlier_round = self.fight_mapping['end_round'] > round_num
            self.features.loc[is_earlier_round, col_name] = round_seconds[is_earlier_round, round_num - 1]
            
            # For the final round of each fight, use the end_time
            is_final_round = self.fight_mapping['end_round'] == round_num
            self.features.loc[is_final_round, col_name] = self.fight_mapping.loc[is_final_round, 'end_time']
            
            # Rounds after the fight ends should be 0
            is_future_round = self.fight_mapping['end_round'] < round_num
            self.features.loc[is_future_round, col_name] = 0
            
        return {"status": "success", "feature_count": len(self.features), "table_name": table_name}
        
    def save(self, table_name: str = None, result_df: pd.DataFrame = None) -> pd.DataFrame:
        """
        Save the calculated time_sec features
        
        Args:
            table_name: Table to save features to (defaults to self.table_name)
            result_df: Optional DataFrame with results (uses self.features if None)
            
        Returns:
            DataFrame with saved features
        """
        # Use table_name if provided, otherwise use default
        if table_name is None:
            table_name = self.table_name
            
        if result_df is None:
            result_df = self.features
            
        # Use context for update if available
        if hasattr(self.context, 'update_table'):
            # Use the context's update_table method which works better with testing
            self.context.update_table(table_name, result_df)
        else:
            # Fallback to direct update for backward compatibility
            self.bulk_update_dataframe(
                result_df,
                table_name,
            )
            
        return result_df
        
    def calculate_for_table(self, table_name: str, columns: Optional[List[str]] = None) -> str:
        """
        Implementation of the required calculate_for_table method for the single_table calculator
        
        For TimeSecCalculator, this method doesn't actually generate SQL but instead
        uses Python processing. We'll still execute the calculation and store results
        to be consistent with the BaseCalculator API.
        
        Args:
            table_name: Name of the feature table
            columns: Optional list of time_sec columns to calculate
            
        Returns:
            Empty string as we don't use SQL generation for this calculator
        """
        # We first ensure we have the fight_mapping data
        if self.fight_mapping is None:
            self.get_features(table_name)
        
        # Then perform the calculation
        self.calculate(table_name, columns)
        
        # Return empty string since we don't generate SQL
        return ""
    
    def execute_for_table(self, table_name: str, columns: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Override the execute_for_table method to handle our custom execution flow
        
        Args:
            table_name: The target table name
            columns: Optional specific columns to process
            
        Returns:
            DataFrame with results
        """
        # Since we've already run the calculation in calculate_for_table,
        # we just need to return the features DataFrame
        if self.features is None:
            # Call calculate_for_table if we haven't run it yet
            self.calculate_for_table(table_name, columns)
            
        return self.features
        
    # Support for BaseCalculator .run() method when using single_table type
    def _run_sequential(self):
        """
        Internal method used by tests to run the calculator's operations in sequence
        """
        self.get_features()
        self.calculate()
        return {"fight_stats_fe": self.save()}

    def save_for_table(self, table_name: str) -> pd.DataFrame:
        """
        Override the save_for_table method to use our custom approach
        that works better with testing.
        
        Args:
            table_name: Name of the table to save to
            
        Returns:
            DataFrame with the saved features
        """
        # Make sure we have calculated features
        if self.features is None:
            # Call calculate_for_table if we haven't run it yet
            self.calculate_for_table(table_name)
            
        # Use the context's update_table method directly if available
        if hasattr(self.context, 'update_table'):
            self.context.update_table(table_name, self.features)
        else:
            # Get key columns based on data structure
            key_columns = ['fight_id', 'fighter_id'] if 'fighter_id' in self.features.columns else ['fight_id']
            
            # Skip bulk update if there are no columns to update
            update_columns = [col for col in self.features.columns if col not in key_columns]
            if not update_columns:
                return self.features
                
            # Direct database update for non-test environments
            self.bulk_update_dataframe(
                self.features,
                table_name,
                self.schema,
                key_columns
            )
            
        return self.features