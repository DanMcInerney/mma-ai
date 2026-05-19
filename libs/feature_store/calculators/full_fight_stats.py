from typing import List, Dict, Optional, Any
import pandas as pd
import numpy as np
from libs.feature_store.base_calculator import BaseCalculator
from libs.feature_store.feature_config import FeatureConfig
from libs.feature_store.calculator_context import CalculatorContext
from sqlalchemy import text


class FullFightStatsCalculator(BaseCalculator):
    """
    Calculator for computing total fight statistics by summing round-specific values.
    
    This calculator takes round-specific statistics (e.g., sig_str_land_rd1, sig_str_land_rd2)
    and calculates the total values across all rounds (e.g., sig_str_land).
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
            
        self.table_name = 'fight_stats_fe'
        self.schema = self.context.schema
        self.id_cols = ['fight_id', 'fighter_id']
        self.feature_type = 'full_fight_stats'  # Used for SQL template identifier
        self.features = None
        
        # Initialize feature config if needed
        if hasattr(self.context, 'feature_utils') and self.context.feature_utils:
            self.feature_config = FeatureConfig(self.conn)
        else:
            # For testing, we'll handle this differently
            self.feature_config = None
            
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
        Load fight stats data with all round-specific columns using SQL template
        
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
                {'schema': self.schema, 'table': table_name}
            )
            self.features = pd.read_sql(query, self.conn)
        else:
            # Get columns dynamically
            if self.feature_config:
                self.all_cols = self.feature_config.get_columns_from_table(self.schema, table_name)
                self.base_cols = self.feature_config.base_stat_prefixes
            else:
                # For testing, get columns from context mock data
                self.all_cols = list(self.context.mock_data.get(table_name, pd.DataFrame()).columns)
                self.base_cols = []
            
            cols_str = ', '.join(self.all_cols)
            
            # Build and execute SQL query
            query = f"SELECT {cols_str} FROM {self.schema}.{table_name}"
            self.features = pd.read_sql(query, self.conn)
            
        # Get all columns that contain round information
        round_cols = [col for col in self.features.columns if '_rd' in col]
        
        # Group columns by their base stat (handling both land and att)
        base_stats = {}
        for col in round_cols:
            # Split on _rd and take the prefix
            base_name = col.rsplit('_rd', 1)[0]
            if base_name not in base_stats:
                base_stats[base_name] = []
            
        # Return the list of base names that we'll calculate
        return list(base_stats.keys())

    def calculate(self, table_name: str = None, columns: List[str] = None) -> Dict[str, Any]:
        """
        Calculate total stats by summing across rounds
        
        Args:
            table_name: Optional table name (defaults to self.table_name)
            columns: Optional list of columns to calculate
            
        Returns:
            Dictionary of operation results
        """
        # Use table_name if provided, otherwise use default
        if table_name is None:
            table_name = self.table_name
            
        # Make sure features are loaded
        if self.features is None:
            self.get_features(table_name)
        
        # Get all columns that contain round information
        round_cols = [col for col in self.features.columns if '_rd' in col]
        
        # Group columns by their base stat (handling both land and att)
        base_stats = {}
        for col in round_cols:
            # Split on _rd and take the prefix
            base_name = col.rsplit('_rd', 1)[0]
            if base_name not in base_stats:
                base_stats[base_name] = []
            base_stats[base_name].append(col)
        
        # First, apply submission clamping logic before summing
        self._clamp_submission_stats()
        
        # For each base stat, sum up rounds 1-5
        for base_name, cols in base_stats.items():
            # If columns is specified, only calculate those requested
            if columns is not None and base_name not in columns:
                continue
                
            # Replace NaN with 0 before summing
            total_value = self.features[cols].fillna(0).sum(axis=1)
            
            # No minimum constraint applied
                
            self.features[base_name] = total_value
        
        return {
            "status": "success", 
            "feature_count": len(self.features), 
            "table_name": table_name,
            "columns_processed": len(base_stats)
        }
    
    def _clamp_submission_stats(self):
        """
        Clamp submission statistics to ensure data consistency.
        If sub_land == 1 and sub_att == 0 for any round, set sub_att to 1.
        """
        corrections_made = 0
        
        # Find all submission-related columns for each round
        for round_num in range(1, 6):  # Rounds 1-5
            sub_land_col = f'sub_land_rd{round_num}'
            sub_att_col = f'sub_att_rd{round_num}'
            
            # Check if both columns exist in the dataframe
            if sub_land_col in self.features.columns and sub_att_col in self.features.columns:
                # Find rows where sub_land == 1 and sub_att == 0
                mask = (self.features[sub_land_col].fillna(0) == 1) & (self.features[sub_att_col].fillna(0) == 0)
                
                # Set sub_att to 1 for these rows
                self.features.loc[mask, sub_att_col] = 1
                
                # Count corrections
                if mask.sum() > 0:
                    corrections_made += mask.sum()
        
        # Log total corrections if any were made
        if corrections_made > 0:
            print(f"FullFightStatsCalculator: Corrected {corrections_made} submission inconsistencies (set sub_att = 1 where sub_land = 1 and sub_att = 0)")

    def save(self, table_name: str = None, result_df: pd.DataFrame = None) -> pd.DataFrame:
        """
        Save the calculated total stats
        
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
                self.schema,
                self.id_cols
            )
            
        return result_df
        
    def calculate_for_table(self, table_name: str, columns: Optional[List[str]] = None) -> str:
        """
        Implementation of the required calculate_for_table method for the single_table calculator
        
        For FullFightStatsCalculator, this method doesn't actually generate SQL but instead
        uses Python processing. We'll still execute the calculation and store results
        to be consistent with the BaseCalculator API.
        
        Args:
            table_name: Name of the feature table
            columns: Optional list of columns to calculate
            
        Returns:
            Empty string as we don't use SQL generation for this calculator
        """
        # We first ensure we have the data loaded
        if self.features is None:
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
            # Skip bulk update if there are no columns to update
            update_columns = [col for col in self.features.columns if col not in self.id_cols]
            if not update_columns:
                return self.features
                
            # Direct database update for non-test environments
            self.bulk_update_dataframe(
                self.features,
                table_name,
                self.schema,
                self.id_cols
            )
            
        return self.features
        
    # Support for BaseCalculator .run() method when using single_table type
    def _run_sequential(self):
        """
        Internal method used by tests to run the calculator's operations in sequence
        """
        self.get_features()
        self.calculate()
        return {self.table_name: self.save()}

    # Method expected by tests - this is patched in the tests
    def run_sequential(self):
        """
        Run the calculator sequentially (named to match what the tests expect)
        """
        return self._run_sequential()
        
    def run(self, parallel: bool = False, max_workers: int = 4) -> Dict[str, pd.DataFrame]:
        """
        Run the full fight stats calculator with either sequential or parallel execution
        
        Args:
            parallel: Whether to use parallel execution (ignored for single_table calculator)
            max_workers: Maximum number of parallel workers if using parallel execution
            
        Returns:
            Dictionary mapping table names to DataFrames with results
        """
        # For single_table calculator, always call run_sequential
        # This ensures the method that gets patched in tests is called
        return self.run_sequential()
