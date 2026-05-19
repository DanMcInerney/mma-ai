from libs.feature_store.base_calculator import BaseCalculator
from libs.feature_store.sql_template_manager import SQLTemplateManager
from typing import List, Dict, Optional, Any
import pandas as pd
from libs.feature_store.calculator_context import CalculatorContext


class PressureCalculator(BaseCalculator):
    """
    Calculator for computing pressure features.
    
    Calculates sig_str_land_pressure = sig_str_land_rd1 / sig_str_land
    This represents how much of a fighter's significant strikes are landed in round 1.
    """
    
    def __init__(self, conn_or_context, calculator_type='single_table'):
        """
        Initialize with either a connection or a calculator context.
        
        Args:
            conn_or_context: SQLAlchemy connection or CalculatorContext
            calculator_type: Type of calculator ('single_table', 'multi_table', 'cross_table')
        """
        # Handle both connection and context for backward compatibility
        if isinstance(conn_or_context, CalculatorContext):
            self.context = conn_or_context
            super().__init__(conn_or_context.connection, calculator_type)
        else:
            self.context = CalculatorContext(conn_or_context)
            super().__init__(conn_or_context, calculator_type)
            
        self.table_name = 'fight_stats_derived'
        self.schema = 'features'
        self.features = ['sig_str_land']  # Only calculate for sig_str_land
        self.feature_type = 'pressure'  # Used for SQL template identifier
        
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
        Get the sig_str_land feature for pressure calculation.
        
        Args:
            table_name: Optional table name (defaults to self.table_name)
            
        Returns:
            List of feature column names
        """
        # Always return sig_str_land for pressure calculation
        return ['sig_str_land']

    def calculate(self, table_name: str = None, columns: List[str] = None) -> Dict[str, Any]:
        """
        Calculate pressure feature: sig_str_land_pressure = sig_str_land_rd1 / sig_str_land.
        
        Args:
            table_name: Optional table name (defaults to self.table_name)
            columns: Optional list of columns to calculate (not used, always calculates sig_str_land_pressure)
            
        Returns:
            Dictionary of operation results
        """
        # Use table_name if provided, otherwise use default
        if table_name is None:
            table_name = self.table_name
            
        # Generate SQL for pressure calculation
        sql = f"""
        SELECT 
            fight_id,
            fighter_id,
            event_id,
            CASE 
                WHEN sig_str_land > 0 THEN 
                    CAST(sig_str_land_rd1 AS FLOAT) / CAST(sig_str_land AS FLOAT)
                ELSE 0.0
            END AS sig_str_land_pressure
        FROM {self.schema}.{table_name}
        WHERE sig_str_land IS NOT NULL 
        AND sig_str_land_rd1 IS NOT NULL
        """
        
        return {"status": "success", "sql": sql, "feature_count": 1, "table_name": table_name}

    def save(self, table_name: str = None, result_df: pd.DataFrame = None) -> pd.DataFrame:
        """
        Execute the calculation SQL and save results to the database.
        
        Args:
            table_name: Table to save features to (defaults to self.table_name)
            result_df: Optional DataFrame with results (not used in this calculator)
            
        Returns:
            DataFrame with saved features
        """
        # Use table_name if provided, otherwise use default
        if table_name is None:
            table_name = self.table_name
            
        # Calculate SQL
        calc_result = self.calculate(table_name)
        sql = calc_result.get("sql", "")
        
        if not sql:
            return pd.DataFrame()
        
        # Define the new column
        pressure_columns = ['sig_str_land_pressure']
        
        # Use context to update table if available
        if hasattr(self.context, 'execute_calculator_update'):
            result = self.context.execute_calculator_update(
                calculation_sql=sql,
                table_name=table_name,
                new_columns=pressure_columns,
                schema=self.schema
            )
        else:
            # Fallback for direct update
            result = self.execute_calculator_update(
                calculation_sql=sql,
                table_name=table_name,
                new_columns=pressure_columns,
                schema=self.schema
            )
        
        return result
        
    def calculate_for_table(self, table_name: str, columns: Optional[List[str]] = None) -> str:
        """
        Generate SQL for calculating pressure features for a specific table.
        
        Args:
            table_name: Name of the table to calculate for
            columns: Optional list of columns to calculate (not used)
            
        Returns:
            SQL query string for the calculation
        """
        calc_result = self.calculate(table_name)
        return calc_result.get("sql", "")
        
    def execute_for_table(self, table_name: str, columns: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Execute calculation for a specific table and return results.
        
        Args:
            table_name: Name of the table to calculate for
            columns: Optional list of columns to calculate (not used)
            
        Returns:
            DataFrame with calculation results
        """
        try:
            # Calculate SQL query
            sql = self.calculate_for_table(table_name, columns)
            
            if not sql:
                self.logger.warning(f"No SQL generated for {table_name}")
                return pd.DataFrame()
                
            # Execute SQL and return results
            return self.execute_raw_sql(sql, return_results=True)
        except Exception as e:
            self.logger.error(f"Error executing calculation for {table_name}: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
            return pd.DataFrame()
        
    def save_for_table(self, table_name: str, columns: Optional[List[str]] = None, 
                   result_df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """
        Save calculation results for a specific table.
        
        Args:
            table_name: Name of the table to save to
            columns: Optional list of columns to calculate (not used)
            result_df: Optional DataFrame with results (executes calculation if None)
            
        Returns:
            DataFrame with saved results
        """
        # Get result if not provided
        if result_df is None:
            result_df = self.execute_for_table(table_name, columns)
            
        if result_df.empty:
            return pd.DataFrame()
            
        # Define the pressure column
        pressure_columns = ['sig_str_land_pressure']
        
        # Check if the expected column exists
        if not all(col in result_df.columns for col in pressure_columns):
            missing = [col for col in pressure_columns if col not in result_df.columns]
            self.logger.warning(f"Missing columns in result_df: {missing}")
            return result_df
                
        # Use context to update table if available
        if hasattr(self.context, 'update_table'):
            self.context.update_table(table_name, result_df)
        else:
            # Fallback to direct update
            self.bulk_update_dataframe(
                result_df, 
                table_name,
                self.schema,
                ['fight_id', 'fighter_id']
            )
            
        return result_df
        
    def run_sequential(self) -> Dict[str, pd.DataFrame]:
        """
        Run the calculator sequentially (for testing purposes).
        
        Returns:
            Dictionary of table names to result DataFrames
        """
        try:
            return self._run_sequential()
        except Exception as e:
            self.logger.error(f"Error in run_sequential: {str(e)}")
            # For testing, return empty result
            return {}
            
    def _run_sequential(self) -> Dict[str, pd.DataFrame]:
        """
        Internal implementation of sequential execution.
        
        Returns:
            Dictionary of table names to result DataFrames
        """
        results = {}
        
        # For single_table calculators, just run on the table_name
        if self.calculator_type == 'single_table':
            features = self.get_features(self.table_name)
            result_df = self.execute_for_table(self.table_name)
            saved_df = self.save_for_table(self.table_name, result_df=result_df)
            results[self.table_name] = saved_df
            
        return results
