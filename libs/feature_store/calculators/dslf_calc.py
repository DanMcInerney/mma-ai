from typing import List, Dict, Optional, Any
import pandas as pd
from sqlalchemy import text
from libs.feature_store.base_calculator import BaseCalculator
from libs.feature_store.calculator_context import CalculatorContext

class DaysSinceLastFightCalculator(BaseCalculator):
    """
    Calculator for computing days since a fighter's last fight.
    
    This calculator determines the number of days between each fight for a fighter,
    using the event dates from the event_mapping table.
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
            
        self.features = None
        self.feature_type = 'dslf'  # Used for SQL template identifier
        
        # Required for single_table calculator type
        self.table_name = 'fight_stats_fe'
        
        # Configure column filtering patterns
        self.add_include_pattern('days_since_last_fight')  # Only process DSLF column
        
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
    
    def execute_sql_template(self, template_name, **params):
        """
        Execute a SQL template with parameters
        
        Args:
            template_name: Name of the template to execute
            **params: Parameters to pass to the template
            
        Returns:
            Result of the SQL execution
        """
        # Get SQL from template manager
        sql = self.context.sql_manager.render_template(
            self.feature_type, 
            template_name,
            params or {'schema': self.context.schema}
        )
        
        # Execute the SQL
        return self.execute_raw_sql(sql)
    
    def get_features(self, table_name: str = None) -> List[str]:
        """
        Get the days_since_last_fight feature column
        
        Args:
            table_name: Optional table name (defaults to self.table_name)
            
        Returns:
            List containing the 'days_since_last_fight' column
        """
        # We have a single feature column for days since last fight
        return ['days_since_last_fight']
    
    def calculate(self, table_name: str = None, columns: List[str] = None) -> Dict[str, Any]:
        """
        Calculate days since last fight for each fighter
        
        Args:
            table_name: Optional table name (defaults to self.table_name)
            columns: Optional list of columns to calculate (not used in this calculator)
            
        Returns:
            Dictionary of operation results
        """
        # Use table_name if provided, otherwise use default
        if table_name is None:
            table_name = self.table_name
            
        # No need to fetch data in advance since calculation happens directly in SQL
        # Just return success status for this operation
        return {'status': 'success', 'message': 'Days since last fight calculation prepared'}
        
    def save(self, table_name: str = None, result_df: pd.DataFrame = None) -> pd.DataFrame:
        """
        Save calculated days since last fight values to the database
        
        Args:
            table_name: Optional table name (defaults to self.table_name)
            result_df: Optional DataFrame (not used in this calculator as we calculate directly in SQL)
            
        Returns:
            Empty DataFrame (operation handled by database)
        """
        # Use table_name if provided, otherwise use default
        if table_name is None:
            table_name = self.table_name
            
        # Ensure the days_since_last_fight column exists
        self._ensure_columns_exist({'days_since_last_fight': 'INTEGER'})
            
        # Use SQL template if available
        if hasattr(self.context, 'sql_manager') and self.context.sql_manager:
            self.execute_sql_template('calculate')
        else:
            # Fallback to direct SQL for backward compatibility
            self.execute_raw_sql('''
                WITH ordered_fights AS (
                    SELECT 
                        f.fighter_id,
                        f.fight_id,
                        f.event_id,
                        e.event_date,
                        LAG(e.event_date) OVER (
                            PARTITION BY f.fighter_id 
                            ORDER BY e.event_date ASC, f.fight_id ASC
                        ) as prev_fight_date,
                        ROW_NUMBER() OVER (
                            PARTITION BY f.fighter_id 
                            ORDER BY e.event_date ASC, f.fight_id ASC
                        ) as fight_order
                    FROM features.fight_stats_fe f
                    JOIN features.event_mapping e ON e.event_id = f.event_id
                )
                UPDATE features.fight_stats_fe f
                SET days_since_last_fight = 
                    CASE 
                        WHEN of.fight_order = 1 THEN 120  -- first fight
                        ELSE CAST((of.event_date - of.prev_fight_date) AS INTEGER)
                    END
                FROM ordered_fights of
                WHERE f.fight_id = of.fight_id
                AND f.fighter_id = of.fighter_id
            ''')
            
        return pd.DataFrame()  # Return empty DataFrame since operation handled by database
    
    def calculate_for_table(self, table_name: str, columns: Optional[List[str]] = None) -> str:
        """
        Generate SQL for calculating days since last fight
        
        Args:
            table_name: Name of the table to calculate for
            columns: Optional list of columns (not used for this calculator)
            
        Returns:
            SQL query string for the calculation
        """
        # We use a window function to get the previous fight date for each fighter
        sql = f'''
            WITH ordered_fights AS (
                SELECT 
                    f.fighter_id,
                    f.fight_id,
                    f.event_id,
                    e.event_date,
                    LAG(e.event_date) OVER (
                        PARTITION BY f.fighter_id 
                        ORDER BY e.event_date ASC, f.fight_id ASC
                    ) as prev_fight_date,
                    ROW_NUMBER() OVER (
                        PARTITION BY f.fighter_id 
                        ORDER BY e.event_date ASC, f.fight_id ASC
                    ) as fight_order
                FROM {self.schema}.{table_name} f
                JOIN {self.schema}.event_mapping e ON e.event_id = f.event_id
            )
            SELECT 
                f.fight_id,
                f.fighter_id,
                CASE 
                    WHEN of.fight_order = 1 THEN 120  -- first fight
                    ELSE CAST((of.event_date - of.prev_fight_date) AS INTEGER)
                END as days_since_last_fight
            FROM {self.schema}.{table_name} f
            JOIN ordered_fights of ON 
                f.fight_id = of.fight_id AND 
                f.fighter_id = of.fighter_id
        '''
        return sql
    
    def execute_for_table(self, table_name: str, columns: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Execute days since last fight calculation for a specific table and return results
        
        Args:
            table_name: Name of the table to calculate for
            columns: Optional list of columns (not used for this calculator)
            
        Returns:
            DataFrame with calculation results
        """
        # Generate SQL
        sql = self.calculate_for_table(table_name, columns)
        
        # Execute SQL and get results
        result_df = self.execute_raw_sql(sql, return_results=True)
        
        # Store results for later use
        self.features = result_df
        
        return result_df
    
    def save_for_table(self, table_name: str) -> pd.DataFrame:
        """
        Save days since last fight calculation results to the specified table
        
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
            self.context.update_table(self.features, table_name, self.schema, ['fight_id', 'fighter_id'])
            return self.features
        
        # Determine key columns for the update
        key_columns = ['fight_id', 'fighter_id'] if 'fighter_id' in self.features.columns else ['fight_id']
        
        # Skip bulk update if there are no columns to update
        if len(self.features.columns) <= len(key_columns):
            return self.features
            
        # Use bulk update for efficiency
        self.bulk_update_dataframe(
            self.features, 
            table_name, 
            self.schema, 
            key_columns
        )
        
        return self.features
    
    def _run_sequential(self):
        """
        Run the calculator sequentially
        
        Returns:
            Dictionary of operation results
        """
        return self.execution_plan.execute_sequential()
        
    def run(self):
        """
        Run the days since last fight calculator (for backward compatibility)
        """
        return self._run_sequential()
