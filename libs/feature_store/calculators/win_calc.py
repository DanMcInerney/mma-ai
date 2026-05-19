from typing import List, Dict, Optional, Any
import pandas as pd
from libs.feature_store.base_calculator import BaseCalculator
import numpy as np
from libs.feature_store.calculator_context import CalculatorContext


class WinCalculator(BaseCalculator):
    """
    Calculator for computing win features.
    
    This calculator determines whether a fighter won a fight
    based on the result from the fight_mapping table.
    It also tracks which round the win occurred in.
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
        self.columns_dict = None
        self.features = None
        self.feature_type = 'win'  # Used for SQL template identifier
        
        # Required for single_table calculator type
        self.table_name = 'fight_stats_fe'
        
        # Configure column filtering patterns
        self.add_include_pattern('win')  # Only process win-related columns
        
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
                    fm.result,
                    fm.fighter1_id,
                    fm.fighter2_id,
                    fm.end_round
                FROM features.fight_mapping fm
                JOIN features.fight_stats_core fs ON fm.fight_id = fs.fight_id
            """, self.conn)
        
        # Return list of feature columns created by this calculator
        return ['win', 'win_rd1', 'win_rd2', 'win_rd3', 'win_rd4', 'win_rd5']

    def calculate(self, table_name: str = None, columns: List[str] = None) -> Dict[str, Any]:
        """
        Calculate win feature based on result and track which round the win occurred in
        
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
        
        # Initialize win columns with 0
        self.features['win'] = 0
        self.features['win_rd1'] = 0
        self.features['win_rd2'] = 0
        self.features['win_rd3'] = 0
        self.features['win_rd4'] = 0
        self.features['win_rd5'] = 0
        
        # Set win=1 for fighters who won their fights
        for fight_id in self.features['fight_id'].unique():
            fight_rows = self.fight_mapping[self.fight_mapping['fight_id'] == fight_id]
            
            # Get result to determine winner
            result = fight_rows['result'].iloc[0]
            fighter1_id = fight_rows['fighter1_id'].iloc[0]
            fighter2_id = fight_rows['fighter2_id'].iloc[0]
            
            # Skip if result is not 0 or 1 (draw, no contest, etc.)
            if result != 0 and result != 1:
                continue
                
            # Determine winner based on result
            if result == 1:
                winner_id = fighter1_id
            else:  # result == 0
                winner_id = fighter2_id
                
            # Get the round the fight ended in
            end_round = fight_rows['end_round'].iloc[0]
            
            # Set win=1 for the winner
            winner_mask = (self.features['fight_id'] == fight_id) & (self.features['fighter_id'] == winner_id)
            self.features.loc[winner_mask, 'win'] = 1
            
            # Set the appropriate round-specific win column
            if end_round == 1:
                self.features.loc[winner_mask, 'win_rd1'] = 1
            elif end_round == 2:
                self.features.loc[winner_mask, 'win_rd2'] = 1
            elif end_round == 3:
                self.features.loc[winner_mask, 'win_rd3'] = 1
            elif end_round == 4:
                self.features.loc[winner_mask, 'win_rd4'] = 1
            elif end_round == 5:
                self.features.loc[winner_mask, 'win_rd5'] = 1
                
        return {"status": "success", "feature_count": len(self.features), "table_name": table_name}
        
    def save(self, table_name: str = None, result_df: pd.DataFrame = None) -> pd.DataFrame:
        """
        Save the calculated win features
        
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
        Generate SQL for calculating win features (if using SQL-based approach)
        
        Args:
            table_name: Name of the table to calculate for
            columns: Optional list of columns to calculate
            
        Returns:
            SQL query string for the calculation or empty string
        """
        # For WinCalculator, we don't use SQL for calculation but rather Python logic
        # Return a query that just fetches the data we need for processing
        return f"""
            SELECT 
                fm.fight_id,
                fs.fighter_id,
                fm.result,
                fm.fighter1_id,
                fm.fighter2_id
            FROM {self.schema}.fight_mapping fm
            JOIN {self.schema}.{table_name} fs ON fm.fight_id = fs.fight_id
        """
    
    def execute_for_table(self, table_name: str, columns: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Execute win feature calculation for a specific table and return results
        
        Args:
            table_name: Name of the table to calculate for
            columns: Optional list of columns to calculate
            
        Returns:
            DataFrame with calculation results
        """
        # Get the fight data we need
        sql = self.calculate_for_table(table_name, columns)
        self.fight_mapping = self.execute_raw_sql(sql, return_results=True)
        
        # Create a results dataframe for the win features
        result_df = self.fight_mapping[['fight_id', 'fighter_id']].copy()
        
        # Initialize win column with 0
        result_df['win'] = 0
        
        # Set win=1 for the fighter who won each fight
        for fight_id in result_df['fight_id'].unique():
            fight_rows = self.fight_mapping[self.fight_mapping['fight_id'] == fight_id]
            
            # Skip if empty (for tests)
            if fight_rows.empty:
                continue
                
            # Get result to determine winner
            # result = 1 means fighter1 won, result = 0 means fighter2 won
            result = fight_rows['result'].iloc[0]
            
            # Skip if result is not 0 or 1 (draw, no contest, etc.)
            if result != 0 and result != 1:
                continue
                
            # Get fighter IDs
            fighter1_id = fight_rows['fighter1_id'].iloc[0]
            fighter2_id = fight_rows['fighter2_id'].iloc[0]
            
            # Determine winner based on result
            if result == 1:
                winner_id = fighter1_id
            else:  # result == 0
                winner_id = fighter2_id
                
            # Set win=1 for the winner
            winner_mask = (result_df['fight_id'] == fight_id) & (result_df['fighter_id'] == winner_id)
            result_df.loc[winner_mask, 'win'] = 1
        
        # Store results for later use
        self.features = result_df
        
        return result_df
    
    def save_for_table(self, table_name: str) -> pd.DataFrame:
        """
        Save win calculation results to the specified table
        
        Args:
            table_name: Name of the table to save to
            
        Returns:
            DataFrame with saved results
        """
        # Calculate features if not already done
        if self.features is None:
            self.execute_for_table(table_name)
        
        # Use context's update_table method if available (for testing)
        if hasattr(self.context, 'update_table'):
            self.context.update_table(table_name, self.features)
            return self.features
        
        # Use bulk update for efficiency
        self.bulk_update_dataframe(
            self.features, 
            table_name, 
            self.schema, 
            ['fight_id', 'fighter_id']
        )
        
        return self.features
        
    # Support for BaseCalculator .run() method when using single_table type
    def _run_sequential(self):
        """
        Internal method used by tests to run the calculator's operations in sequence
        """
        self.get_features()
        self.calculate()
        return {"fight_stats_fe": self.save()}
    
    # Method expected by tests - this is patched in the tests
    def run_sequential(self):
        """
        Run the calculator sequentially (named to match what the tests expect)
        """
        return self._run_sequential()
        
    def run(self, parallel: bool = False, max_workers: int = 4) -> Dict[str, pd.DataFrame]:
        """
        Run the win calculator with either sequential or parallel execution
        
        Args:
            parallel: Whether to use parallel execution (ignored for single_table calculator)
            max_workers: Maximum number of parallel workers if using parallel execution
            
        Returns:
            Dictionary mapping table names to DataFrames with results
        """
        # For single_table calculator, always call run_sequential
        # This ensures the method that gets patched in tests is called
        return self.run_sequential()
