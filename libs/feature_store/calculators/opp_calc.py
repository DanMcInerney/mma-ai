from libs.feature_store.base_calculator import BaseCalculator
from typing import List, Dict, Optional, Set, Tuple
import pandas as pd
import logging
from sqlalchemy import text
from libs.feature_store.sql_template_manager import SQLTemplateManager
from libs.feature_store.calculator_context import CalculatorContext

class OpponentCalculator(BaseCalculator):
    """
    Calculate opponent features by joining fighter stats with their opponent's stats.
    For each input column, creates a new column with '_opp' suffix containing the opponent's value.
    Works with feature-specific tables created by create_feature_specific_tables().
    
    Args:
        conn_or_context: Database connection or CalculatorContext
        include_patterns: Optional list of patterns - only columns containing these patterns will be processed
    """

    def __init__(self, conn_or_context, include_patterns=None):
        # Handle both connection and context objects for flexibility
        if isinstance(conn_or_context, CalculatorContext):
            self.context = conn_or_context
            super().__init__(conn_or_context.connection, calculator_type='multi_table')
        else:
            self.context = CalculatorContext(conn_or_context)
            super().__init__(conn_or_context, calculator_type='multi_table')
        
        # Simple include pattern filtering
        self.include_patterns = include_patterns or []
        
        # Set core properties
        self.exclude_patterns.update(['_opp', '_id'])  # Don't create opponent stats for IDs or existing opponent stats
        
        # Set up logging
        self.logger = logging.getLogger(__name__)
        
        # Get feature tables with exclusions
        exclude_patterns = ['_mapping', 'fight_stats_core', 'fight_stats_fe', 
                          'fight_stats_derived', 'training_data_raw', 
                          'training_data', 'model_data', '_first_time_opp_avg_stats']
        
        # Load feature tables
        self.feature_tables = self.get_feature_tables(exclude_patterns=exclude_patterns)
        self.logger.info(f"Initialized with {len(self.feature_tables)} feature tables")
        
        # Initialize SQL template manager if not available from context
        if not hasattr(self.context, 'sql_manager') or not self.context.sql_manager:
            self.sql_template_manager = SQLTemplateManager()

    def get_features(self, table_name: str) -> List[str]:
        """
        Get all columns for a specific feature table that should have opponent stats calculated.
        
        Args:
            table_name: Name of the feature table
            
        Returns:
            List of column names
        """
        try:
            # Get all columns from the table excluding our patterns
            if hasattr(self.context, 'feature_utils') and self.context.feature_utils:
                columns = self.context.feature_utils.get_columns_from_table(
                    self.schema,
                    table_name,
                    exclude_strs=self.exclude_patterns
                )
            else:
                # Fallback to basic implementation
                columns = self.feature_utils.get_columns_from_table(
                    self.schema,
                    table_name,
                    exclude_strs=self.exclude_patterns
                )
            
            # Simple include pattern filtering
            if self.include_patterns:
                columns = [col for col in columns if any(pattern in col for pattern in self.include_patterns)]
            
            return columns
        except Exception as e:
            self.logger.error(f"Error getting features for table {table_name}: {str(e)}")
            return []

    def _validate_inputs(self, table_name: str, columns: List[str]) -> bool:
        """
        Validate input columns before calculation.
        
        Args:
            table_name: Name of the feature table
            columns: List of columns to validate
            
        Returns:
            True if validation passes, raises exception otherwise
        """
        try:
            if not columns:
                raise ValueError(f"No columns provided for opponent calculation in table {table_name}")
                
            # Check that none of the columns already have _opp suffix
            invalid_columns = [col for col in columns if any(pat in col for pat in self.exclude_patterns)]
            if invalid_columns:
                self.logger.warning(f"Skipping columns with excluded patterns in {table_name}: {invalid_columns}")
                
            # Verify table exists
            table_exists = self.execute_raw_sql(f"""
                SELECT EXISTS (
                    SELECT FROM information_schema.tables 
                    WHERE table_schema = '{self.schema}' 
                    AND table_name = '{table_name}'
                )
            """, return_results=True).iloc[0, 0]
            
            if not table_exists:
                raise ValueError(f"Table {self.schema}.{table_name} does not exist")
                
            return True
                
        except Exception as e:
            self.logger.error(f"Input validation error for {table_name}: {str(e)}")
            raise

    def _validate_outputs(self, table_name: str, columns: List[str], result_df: pd.DataFrame) -> bool:
        """
        Validate calculation results.
        
        Args:
            table_name: Name of the feature table
            columns: List of columns that should have opponent values
            result_df: DataFrame with calculation results
            
        Returns:
            True if validation passes, raises exception otherwise
        """
        try:
            if result_df is None or result_df.empty:
                raise ValueError(f"No results returned from opponent calculation for {table_name}")
                
            # Verify all expected opponent columns are present
            expected_opp_columns = [feat + '_opp' for feat in columns]
            missing_columns = [col for col in expected_opp_columns if col not in result_df.columns]
            
            if missing_columns:
                self.logger.warning(f"Missing expected opponent columns in {table_name}: {missing_columns}")
                
            return True
                
        except Exception as e:
            self.logger.error(f"Output validation error for {table_name}: {str(e)}")
            raise

    def _prepare_column_selects(self, columns: List[str]) -> str:
        """
        Prepare column selection SQL with proper formatting to prevent syntax errors.
        
        Args:
            columns: List of columns to get opponent values for
            
        Returns:
            SQL fragment for column selection
        """
        # Create proper column selection syntax
        column_selects = []
        for col in columns:
            column_selects.append(f"opp.{col} AS {col}_opp")
        
        # Join with commas and ensure no trailing comma
        return ",\n    ".join(column_selects)

    def execute_sql_template(self, template_name: str, operation: str, params: Dict) -> str:
        """
        Execute a SQL template using the context's SQL manager or instance's SQL template manager.
        
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
            
            return sql
            
        except Exception as e:
            self.logger.warning(f"Error executing SQL template: {str(e)}")
            return ""

    def calculate_for_table(self, table_name: str, columns: Optional[List[str]] = None) -> str:
        """
        Calculate opponent stats for a specific feature table.
        
        Args:
            table_name: Name of the feature table
            columns: Optional list of columns (if None, will fetch all columns)
            
        Returns:
            SQL query string for the calculation
        """
        try:
            # Get columns if not provided
            if columns is None:
                columns = self.get_features(table_name)
                
            # Validate inputs
            self._validate_inputs(table_name, columns)
            
            # Prepare formatted column selects to prevent syntax errors
            column_selects = self._prepare_column_selects(columns)
            
            # Prepare template parameters
            template_params = {
                'schema': self.schema,
                'table_name': table_name,
                'columns': columns,
                'column_selects': column_selects,
                'exclude_patterns': self.exclude_patterns
            }
            
            # Render the SQL template
            sql = self.execute_sql_template(
                'opponent',
                'calculate',
                template_params
            )
            
            # Validate SQL syntax
            if sql:
                # Check for common syntax errors
                # 1. Fix missing spaces before FROM clause
                sql = sql.replace('_oppFROM', '_opp\nFROM')
                
                # Log debug information if requested
                self.logger.debug(f"Generated SQL for {table_name}: {sql}")
            else:
                self.logger.warning(f"Empty SQL generated for {table_name}")
            
            return sql
            
        except Exception as e:
            self.logger.error(f"Error calculating for table {table_name}: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
            return ""

    def execute_for_table(self, table_name: str, columns: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Execute the calculation SQL for a specific feature table and return results as DataFrame.
        
        Args:
            table_name: Name of the feature table
            columns: Optional list of columns (if None, will fetch all columns)
            
        Returns:
            DataFrame with opponent stat columns
        """
        try:
            # Generate SQL
            sql = self.calculate_for_table(table_name, columns)
            
            # Execute SQL using parent class method
            result_df = self.execute_raw_sql(sql, return_results=True)
            
            # Validate outputs
            columns = columns or self.get_features(table_name)
            self._validate_outputs(table_name, columns, result_df)
            
            return result_df
            
        except Exception as e:
            self.logger.error(f"Error executing opponent calculation for {table_name}: {str(e)}")
            return pd.DataFrame()

    def save_for_table(self, table_name: str, columns: Optional[List[str]] = None, 
                      result_df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """
        Execute the calculation SQL and save results to the specified feature table.
        
        Args:
            table_name: Name of the feature table
            columns: Optional list of columns to process
            result_df: Optional pre-computed result DataFrame
            
        Returns:
            DataFrame with updated opponent stat columns
        """
        try:
            # Get columns if not provided
            if columns is None:
                columns = self.get_features(table_name)
            
            if not columns:
                self.logger.warning(f"No suitable columns found for opponent calculation in {table_name}")
                return pd.DataFrame()
                
            # Get list of new opponent columns
            opp_columns = [feat + '_opp' for feat in columns]
            
            # Execute calculation if results not provided
            if result_df is None:
                # Generate SQL
                sql = self.calculate_for_table(table_name, columns)
                
                self.logger.info(f"Executing opponent calculation for {table_name} ({len(opp_columns)} columns)")
                
                # Use either context or parent class method to update the table
                if hasattr(self.context, 'execute_calculator_update'):
                    result = self.context.execute_calculator_update(
                        calculation_sql=sql,
                        table_name=table_name,
                        new_columns=opp_columns,
                        schema=self.schema
                    )
                else:
                    result = self.execute_calculator_update(
                        calculation_sql=sql,
                        table_name=table_name,
                        new_columns=opp_columns,
                        schema=self.schema
                    )
            else:
                # Use the provided results
                result = result_df
            
            self.logger.info(f"Successfully updated {table_name} with opponent stats")
            return result
            
        except Exception as e:
            self.logger.error(f"Error saving opponent stats for {table_name}: {str(e)}")
            if hasattr(self.conn, 'rollback'):
                self.conn.rollback()  # Rollback transaction on error
            return pd.DataFrame()

    # Legacy method implementations to delegate to the new architecture
    def run(self, parallel: bool = False, max_workers: int = 4) -> Dict[str, pd.DataFrame]:
        """
        Execute opponent calculations for all feature tables.
        Extends parent method to provide specific logging for opponent calculator.
        
        Args:
            parallel: Whether to use parallel execution
            max_workers: Maximum number of parallel workers
            
        Returns:
            Dictionary mapping table names to their result DataFrames
        """
        self.logger.info("Starting opponent calculations for all feature tables")
        
        # Use the parent class run method to handle execution
        results = super().run(parallel=parallel, max_workers=max_workers)
        
        # Count successful and failed tables
        successful_tables = [name for name, df in results.items() if df is not None and not df.empty]
        failed_tables = [name for name, df in results.items() if df is None or df.empty]
        
        self.logger.info(f"Opponent calculations completed. Processed {len(successful_tables)} tables successfully.")
        
        if failed_tables:
            self.logger.warning(f"Failed to process {len(failed_tables)} tables")
            
        return results
    
    # Legacy methods for backward compatibility
    def calculate(self) -> str:
        """Legacy method for backward compatibility. Use run() instead."""
        self.logger.warning("calculate() is deprecated, use run() instead")
        self.run()
        return "Opponent calculations completed for all feature tables"

    def save(self) -> pd.DataFrame:
        """Legacy method for backward compatibility. Use run() instead."""
        self.logger.warning("save() is deprecated, use run() instead")
        self.run()
        return pd.DataFrame({"status": ["completed"], "message": ["Opponent calculations completed for all feature tables"]})