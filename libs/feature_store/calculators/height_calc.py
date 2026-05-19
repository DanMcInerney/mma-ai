from typing import List, Dict, Optional, Any
import pandas as pd
from libs.feature_store.base_calculator import BaseCalculator
from libs.feature_store.calculator_context import CalculatorContext

class HeightCalculator(BaseCalculator):
    """
    Calculator for computing fighter height.
    
    This calculator retrieves fighter height from the fighter_mapping table
    and fills in missing values with the average height for the fighter's weight class.
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
        self.feature_type = 'height'  # Used for SQL template identifier
        
        # Required for single_table calculator type
        self.table_name = 'fight_stats_fe'
        
        # Configure column filtering patterns
        self.add_include_pattern('height')  # Only process height-related columns
        
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
        Get the height feature column
        
        Args:
            table_name: Optional table name (defaults to self.table_name)
            
        Returns:
            List containing the 'height' column
        """
        # We have a single feature column for height
        return ['height']
    
    def calculate(self, table_name: str = None, columns: List[str] = None) -> Dict[str, Any]:
        """
        Calculate height for each fighter
        
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
        return {'status': 'success', 'message': 'Height calculation prepared'}
        
    def save(self, table_name: str = None, result_df: pd.DataFrame = None) -> pd.DataFrame:
        """
        Save calculated height values to the database
        
        Args:
            table_name: Optional table name (defaults to self.table_name)
            result_df: Optional DataFrame (not used in this calculator as we calculate directly in SQL)
            
        Returns:
            Empty DataFrame (operation handled by database)
        """
        # Use table_name if provided, otherwise use default
        if table_name is None:
            table_name = self.table_name
            
        # Ensure the height column exists
        self._ensure_columns_exist({'height': 'INTEGER'})
            
        # Use SQL template if available
        if hasattr(self.context, 'sql_manager') and self.context.sql_manager:
            self.execute_sql_template('calculate')
        else:
            # Fallback to direct SQL for backward compatibility
            self.execute_raw_sql('''
                WITH weightclass_height AS (
                    SELECT 
                        fm.weightclass,
                        ROUND(AVG(fmap.fighter_height))::INTEGER as avg_height
                    FROM features.fight_mapping fm
                    JOIN features.fighter_mapping fmap ON 
                        fm.fighter1_id = fmap.fighter_id OR 
                        fm.fighter2_id = fmap.fighter_id
                    WHERE fmap.fighter_height IS NOT NULL
                    GROUP BY fm.weightclass
                )
                UPDATE features.fight_stats_fe f
                SET height = CAST(
                    COALESCE(
                        m.fighter_height,
                        wr.avg_height
                    ) AS INTEGER
                )
                FROM features.fighter_mapping m,
                     features.fight_mapping fm,
                     weightclass_height wr
                WHERE f.fighter_id = m.fighter_id
                AND f.fight_id = fm.fight_id
                AND wr.weightclass = fm.weightclass
            ''')
            
        return pd.DataFrame()  # Return empty DataFrame since operation handled by database
    
    def calculate_for_table(self, table_name: str, columns: Optional[List[str]] = None) -> str:
        """
        Generate SQL for calculating height features
        
        Args:
            table_name: Name of the table to calculate for
            columns: Optional list of columns (not used for this calculator)
            
        Returns:
            SQL query string for the calculation
        """
        # We use a WITH clause to calculate weight class average heights
        sql = f'''
            WITH weightclass_height AS (
                SELECT 
                    fm.weightclass,
                    ROUND(AVG(fmap.fighter_height))::INTEGER as avg_height
                FROM {self.schema}.fight_mapping fm
                JOIN {self.schema}.fighter_mapping fmap ON 
                    fm.fighter1_id = fmap.fighter_id OR 
                    fm.fighter2_id = fmap.fighter_id
                WHERE fmap.fighter_height IS NOT NULL
                GROUP BY fm.weightclass
            )
            SELECT 
                f.fight_id,
                f.fighter_id,
                CAST(
                    COALESCE(
                        m.fighter_height,
                        wr.avg_height
                    ) AS INTEGER
                ) as height
            FROM {self.schema}.{table_name} f
            JOIN {self.schema}.fighter_mapping m ON f.fighter_id = m.fighter_id
            JOIN {self.schema}.fight_mapping fm ON f.fight_id = fm.fight_id
            JOIN weightclass_height wr ON wr.weightclass = fm.weightclass
        '''
        return sql
    
    def execute_for_table(self, table_name: str, columns: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Execute height calculation for a specific table and return results
        
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
        Save height calculation results to the specified table
        
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
        Run the height calculator (for backward compatibility)
        """
        return self._run_sequential()
