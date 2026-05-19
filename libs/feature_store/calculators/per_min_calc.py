from libs.feature_store.base_calculator import BaseCalculator
from libs.feature_store.sql_template_manager import SQLTemplateManager
from typing import List, Dict, Optional, Any
import pandas as pd
from libs.feature_store.calculator_context import CalculatorContext


class PerMinCalculator(BaseCalculator):
    """
    Calculator for computing per-minute rate features.
    
    Divides event counts by the time in minutes to create
    rate statistics like strikes_per_min, td_per_min, etc.
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
        self.features = None
        self.feature_type = 'per_min'  # Used for SQL template identifier
        
        # Exclude certain columns that aren't event counts
        self.exclude_patterns = {
            '_id', 'time_sec', '_acc', '_def', '_per_min',
            'age', 'days_since_last_fight', 'reach', 'ape', 'total', '_raw'
        }
        
        # Add exclude patterns to column filter
        for pattern in self.exclude_patterns:
            self.add_exclude_pattern(pattern)
        
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
        Get all relevant columns that represent event counts for per-minute calculations.
        
        Args:
            table_name: Optional table name (defaults to self.table_name)
            
        Returns:
            List of feature column names
        """
        # Use table_name if provided, otherwise use default
        if table_name is None:
            table_name = self.table_name
            
        # Get feature columns using context if available
        if hasattr(self.context, 'feature_utils') and self.context.feature_utils:
            self.features = self.context.feature_utils.get_columns_from_table(
                self.schema,
                table_name,
                exclude_strs=self.exclude_patterns
            )
        else:
            # Fallback for tests
            self.features = []
            
        return self.features

    def calculate(self, table_name: str = None, columns: List[str] = None) -> Dict[str, Any]:
        """
        Calculate simple per-minute rates without Bayesian smoothing.
        Simply divides the count by the time in minutes.
        
        Args:
            table_name: Optional table name (defaults to self.table_name)
            columns: Optional list of columns to calculate
            
        Returns:
            Dictionary of operation results
        """
        # Use table_name if provided, otherwise use default
        if table_name is None:
            table_name = self.table_name
            
        # Get features if not already loaded or if specific columns provided
        if not self.features or columns:
            if columns:
                self.features = columns
            else:
                self.get_features(table_name)
            
        if not self.features:
            return {"status": "skipped", "message": "No features to calculate", "table_name": table_name}
            
        # Use SQL template if available via context
        sql = self.execute_sql_template(
            template_name=self.feature_type,
            operation='calculate',
            params={
                'schema': self.schema,
                'table_name': table_name,
                'features': self.features
            }
        )
        
        return {"status": "success", "sql": sql, "feature_count": len(self.features), "table_name": table_name}

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
            
        # Get features if not already loaded
        if not self.features:
            self.get_features(table_name)
            
        if not self.features:
            return pd.DataFrame()
            
        # Generate per-minute column names
        per_min_columns = [feat + '_per_min' for feat in self.features]
        
        # Calculate SQL if we need it
        calc_result = self.calculate(table_name)
        sql = calc_result.get("sql", "")
        
        # Use context to update table if available
        if hasattr(self.context, 'execute_calculator_update'):
            result = self.context.execute_calculator_update(
                calculation_sql=sql,
                table_name=table_name,
                new_columns=per_min_columns,
                schema=self.schema
            )
        else:
            # Fallback for direct update
            result = self.execute_calculator_update(
                calculation_sql=sql,
                table_name=table_name,
                new_columns=per_min_columns,
                schema=self.schema
            )
        
        return result
        
    def execute_sql_template(self, template_name: str, operation: str, params: Dict) -> str:
        """
        Execute a SQL template using the context's SQL manager if available.
        
        Args:
            template_name: Name of the template category
            operation: Operation name ('get_features', 'calculate', etc.)
            params: Parameters to pass to the template
            
        Returns:
            Rendered SQL string
        """
        try:
            sql = ""
            
            # First try using context's SQL manager
            if hasattr(self.context, 'sql_manager') and self.context.sql_manager:
                try:
                    sql = self.context.sql_manager.render_template(
                        template_name,
                        operation,
                        params
                    )
                except Exception as e:
                    self.logger.warning(f"Error using context SQL manager: {str(e)}")
                    
            # Fall back to instance SQL template manager if available
            if not sql and hasattr(self, 'sql_template_manager') and self.sql_template_manager:
                try:
                    sql = self.sql_template_manager.render_template(
                        template_name,
                        operation,
                        params
                    )
                except Exception as e:
                    self.logger.warning(f"Error using instance SQL template manager: {str(e)}")
                    
            # Ensure we're casting to numeric for per_min values
            if operation == 'calculate' and sql:
                # Check if SQL contains per_min calculations
                if '_per_min' in sql and 'CASE WHEN' in sql:
                    # Find all patterns like "CASE WHEN ... END AS col_per_min"
                    import re
                    sql = re.sub(
                        r'(CASE WHEN.*?END) AS ([a-z_]+_per_min)',
                        r'CAST(\1 AS FLOAT) AS \2',
                        sql,
                        flags=re.DOTALL
                    )
            
            return sql
            
        except Exception as e:
            self.logger.error(f"Error executing SQL template: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
            return ""
        
    def calculate_for_table(self, table_name: str, columns: Optional[List[str]] = None) -> str:
        """
        Generate SQL for calculating per-minute features for a specific table.
        
        Args:
            table_name: Name of the table to calculate for
            columns: Optional list of columns to calculate
            
        Returns:
            SQL query string for the calculation
        """
        # Get features if not already loaded or if specific columns provided
        if not self.features or columns:
            if columns:
                self.features = columns
            else:
                self.get_features(table_name)
                
        # If no features, return empty string
        if not self.features:
            return ""
            
        # Try to use SQL template first
        sql = self.execute_sql_template(
            template_name=self.feature_type,
            operation='calculate',
            params={
                'schema': self.schema,
                'table_name': table_name,
                'features': self.features
            }
        )
            
        return sql
        
    def execute_for_table(self, table_name: str, columns: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Execute calculation for a specific table and return results.
        
        Args:
            table_name: Name of the table to calculate for
            columns: Optional list of columns to calculate
            
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
            columns: Optional list of columns to calculate
            result_df: Optional DataFrame with results (executes calculation if None)
            
        Returns:
            DataFrame with saved results
        """
        # Get features if not already loaded or if specific columns provided
        if not self.features or columns:
            if columns:
                self.features = columns
            else:
                self.get_features(table_name)
                
        # Get result if not provided
        if result_df is None:
            result_df = self.execute_for_table(table_name, columns)
            
        if result_df.empty:
            return pd.DataFrame()
            
        # Generate per-minute column names
        per_min_columns = [feat + '_per_min' for feat in self.features]
        
        # Check if we need to filter the result DataFrame
        if len(per_min_columns) > 0 and not all(col in result_df.columns for col in per_min_columns):
            # Some expected columns are missing, log a warning
            missing = [col for col in per_min_columns if col not in result_df.columns]
            self.logger.warning(f"Missing columns in result_df: {missing}")
            
            # Only keep columns that exist
            per_min_columns = [col for col in per_min_columns if col in result_df.columns]
            
        # If no columns to update, just return the DataFrame
        if not per_min_columns:
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