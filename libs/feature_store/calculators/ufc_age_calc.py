from typing import List, Dict, Optional, Any
import pandas as pd
from libs.feature_store.base_calculator import BaseCalculator
from libs.feature_store.calculator_context import CalculatorContext

class UfcAgeCalculator(BaseCalculator):
    """
    Calculator for computing fighter UFC age (time since UFC debut).
    
    This calculator determines how long a fighter has been in the UFC by calculating
    the time between their first UFC fight and the current fight.
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
        self.feature_type = 'ufc_age'  # Used for SQL template identifier
        
        # Required for single_table calculator type
        self.table_name = 'fight_stats_fe'
        
        # Configure column filtering patterns
        self.add_include_pattern('ufcage')  # Only process UFC age column
        
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
        Get the UFC age feature column
        
        Args:
            table_name: Optional table name (defaults to self.table_name)
            
        Returns:
            List containing the 'ufcage' column
        """
        # We have a single feature column for UFC age
        return ['ufcage']
    
    def calculate(self, table_name: str = None, columns: List[str] = None) -> Dict[str, Any]:
        """
        Calculate UFC age for each fighter
        
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
        return {'status': 'success', 'message': 'UFC age calculation prepared'}
        
    def save(self, table_name: str = None, result_df: pd.DataFrame = None) -> pd.DataFrame:
        """
        Save calculated UFC age values to the database
        
        Args:
            table_name: Optional table name (defaults to self.table_name)
            result_df: Optional DataFrame (not used in this calculator as we calculate directly in SQL)
            
        Returns:
            Empty DataFrame (operation handled by database)
        """
        # Use table_name if provided, otherwise use default
        if table_name is None:
            table_name = self.table_name
            
        # Ensure the ufcage column exists
        self._ensure_columns_exist({'ufcage': 'DOUBLE PRECISION'})
            
        # Use SQL template if available
        if hasattr(self.context, 'sql_manager') and self.context.sql_manager:
            self.execute_sql_template('calculate')
        else:
            # Fallback to direct SQL for backward compatibility
            self.execute_raw_sql('''
                WITH first_fights AS (
                    -- For each fighter, determine the debut (first UFC fight date)
                    SELECT 
                        f.fighter_id,
                        MIN(e.event_date) AS first_ufc_fight_date
                    FROM features.fight_stats_fe f
                    JOIN features.event_mapping e 
                      ON f.event_id = e.event_id
                    GROUP BY f.fighter_id
                ),
                ufc_age_calc AS (
                    -- For each fight, calculate the time in years since the fighter's debut.
                    SELECT 
                        f.fight_id,
                        f.fighter_id,
                        ROUND(
                            EXTRACT(EPOCH FROM (e.event_date::timestamp - ff.first_ufc_fight_date::timestamp))
                            / (365.25 * 24 * 60 * 60)::DECIMAL,
                            3
                        ) AS calculated_ufc_age
                    FROM features.fight_stats_fe f
                    JOIN features.event_mapping e 
                      ON f.event_id = e.event_id
                    JOIN first_fights ff 
                      ON f.fighter_id = ff.fighter_id
                )
                UPDATE features.fight_stats_fe f
                SET ufcage = (
                    SELECT calculated_ufc_age 
                    FROM ufc_age_calc ua 
                    WHERE ua.fight_id = f.fight_id 
                      AND ua.fighter_id = f.fighter_id
                )
            ''')
            
        return pd.DataFrame()  # Return empty DataFrame since operation handled by database
    
    def calculate_for_table(self, table_name: str, columns: Optional[List[str]] = None) -> str:
        """
        Generate SQL for calculating UFC age features
        
        Args:
            table_name: Name of the table to calculate for
            columns: Optional list of columns (not used for this calculator)
            
        Returns:
            SQL query string for the calculation
        """
        # We use multiple CTEs to calculate UFC age
        sql = f'''
            WITH first_fights AS (
                -- For each fighter, determine the debut (first UFC fight date)
                SELECT 
                    f.fighter_id,
                    MIN(e.event_date) AS first_ufc_fight_date
                FROM {self.schema}.{table_name} f
                JOIN {self.schema}.event_mapping e 
                  ON f.event_id = e.event_id
                GROUP BY f.fighter_id
            ),
            ufc_age_calc AS (
                -- For each fight, calculate the time in years since the fighter's debut.
                SELECT 
                    f.fight_id,
                    f.fighter_id,
                    ROUND(
                        EXTRACT(EPOCH FROM (e.event_date::timestamp - ff.first_ufc_fight_date::timestamp))
                        / (365.25 * 24 * 60 * 60)::DECIMAL,
                        3
                    ) AS calculated_ufc_age
                FROM {self.schema}.{table_name} f
                JOIN {self.schema}.event_mapping e 
                  ON f.event_id = e.event_id
                JOIN first_fights ff 
                  ON f.fighter_id = ff.fighter_id
            )
            SELECT 
                f.fight_id,
                f.fighter_id,
                ua.calculated_ufc_age as ufcage
            FROM {self.schema}.{table_name} f
            JOIN ufc_age_calc ua ON 
                ua.fight_id = f.fight_id AND 
                ua.fighter_id = f.fighter_id
        '''
        return sql
    
    def execute_for_table(self, table_name: str, columns: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Execute UFC age calculation for a specific table and return results
        
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
        Save UFC age calculation results to the specified table
        
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
        Run the UFC age calculator (for backward compatibility)
        """
        return self._run_sequential()
