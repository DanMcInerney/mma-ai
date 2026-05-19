from libs.feature_store.base import BaseFeatureStore
from typing import List, Dict, Union, Set, Optional, Tuple, Any, Callable
from sqlalchemy.engine import Connection
from abc import abstractmethod, ABC
import pandas as pd
from sqlalchemy import text
from libs.feature_store.feature_utils import FeatureUtils
from libs.feature_store.sql_template_manager import SQLTemplateManager
import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

class ColumnFilter:
    """
    Utility class for handling column inclusion/exclusion patterns.
    Provides a standardized way to filter columns across all calculator types.
    """
    
    def __init__(self):
        self.include_patterns: Set[str] = set()
        self.exclude_patterns: Set[str] = set()
        
    def add_include_pattern(self, pattern: str) -> None:
        """Add a pattern for columns to include"""
        self.include_patterns.add(pattern)
        
    def add_exclude_pattern(self, pattern: str) -> None:
        """Add a pattern for columns to exclude"""
        self.exclude_patterns.add(pattern)
        
    def should_process_column(self, column_name: str) -> bool:
        """
        Determine if a column should be processed based on include/exclude patterns
        
        Args:
            column_name: Name of the column to check
            
        Returns:
            True if column should be processed, False otherwise
        """
        # If include patterns are specified, column must match at least one
        if self.include_patterns:
            matches_include = any(pattern in column_name for pattern in self.include_patterns)
            if not matches_include:
                return False
                
        # Column must not match any exclude patterns
        if self.exclude_patterns:
            matches_exclude = any(pattern in column_name for pattern in self.exclude_patterns)
            if matches_exclude:
                return False
                
        return True
        
    def filter_columns(self, columns: List[str]) -> List[str]:
        """
        Filter a list of columns based on include/exclude patterns
        
        Args:
            columns: List of column names to filter
            
        Returns:
            Filtered list of columns
        """
        return [col for col in columns if self.should_process_column(col)]

class ExecutionPlan:
    """
    Represents a plan for executing calculator operations.
    Supports sequential or parallel execution of multiple operations.
    """
    
    def __init__(self):
        self.operations: List[Tuple[str, Callable, Dict[str, Any]]] = []
        
    def add_operation(self, name: str, func: Callable, **kwargs) -> None:
        """
        Add an operation to the execution plan
        
        Args:
            name: Operation name for logging/tracking
            func: Callable function to execute
            **kwargs: Arguments to pass to the function
        """
        self.operations.append((name, func, kwargs))
        
    def execute_sequential(self) -> Dict[str, Any]:
        """
        Execute operations sequentially
        
        Returns:
            Dictionary mapping operation names to their results
        """
        results = {}
        
        for name, func, kwargs in self.operations:
            start_time = time.time()
            results[name] = func(**kwargs)
            elapsed = time.time() - start_time
            logger = logging.getLogger(__name__)
            logger.info(f"Operation '{name}' completed in {elapsed:.2f} seconds")
            
        return results
        
    def execute_parallel(self, max_workers: int = 4) -> Dict[str, Any]:
        """
        Execute operations in parallel using a thread pool
        
        Args:
            max_workers: Maximum number of parallel workers
            
        Returns:
            Dictionary mapping operation names to their results
        """
        results = {}
        logger = logging.getLogger(__name__)
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all operations
            future_to_name = {
                executor.submit(func, **kwargs): name
                for name, func, kwargs in self.operations
            }
            
            # Process results as they complete
            for future in as_completed(future_to_name):
                name = future_to_name[future]
                try:
                    results[name] = future.result()
                    logger.info(f"Operation '{name}' completed successfully")
                except Exception as e:
                    logger.error(f"Operation '{name}' failed: {str(e)}")
                    results[name] = None
                    raise
                    
        return results

class BaseCalculator(BaseFeatureStore, ColumnFilter):
    """
    Enhanced base class for all feature calculators with support for different calculator types.
    Provides standardized methods for SQL execution, column filtering, and execution planning.
    """
    
    CALCULATOR_TYPES = ['single_table', 'multi_table', 'cross_table']
    
    def __init__(self, conn: Connection, calculator_type: str = 'single_table'):
        """
        Initialize calculator with connection and type
        
        Args:
            conn: SQLAlchemy connection object or CalculatorContext
            calculator_type: Type of calculator ('single_table', 'multi_table', or 'cross_table')
        """
        if calculator_type not in self.CALCULATOR_TYPES:
            raise ValueError(f"Invalid calculator type: {calculator_type}. "
                            f"Must be one of {self.CALCULATOR_TYPES}")
                            
        # Handle the case where a CalculatorContext is passed instead of a connection
        from libs.feature_store.calculator_context import CalculatorContext
        if isinstance(conn, CalculatorContext):
            actual_conn = conn.connection
        else:
            actual_conn = conn
                            
        # Initialize parent classes
        BaseFeatureStore.__init__(self, actual_conn)
        ColumnFilter.__init__(self)
        
        # Set calculator properties
        self.calculator_type = calculator_type
        self.feature_utils = FeatureUtils(actual_conn)
        self.sql_template_manager = SQLTemplateManager(actual_conn)
        self.logger = logging.getLogger(__name__)
        self.schema = 'features'  # Default schema
        self.execution_plan = ExecutionPlan()
        
        # Set default key columns for tables
        self.key_columns = {
            'fight_stats_derived': ['fight_id', 'fighter_id'],
            'fight_stats_core': ['fight_id', 'fighter_id'],
            'fight_stats_fe': ['fight_id', 'fighter_id'],
            'sig_str': ['fight_id', 'fighter_id'],
            'head': ['fight_id', 'fighter_id'],
            'body': ['fight_id', 'fighter_id'],
            'leg': ['fight_id', 'fighter_id'],
            'distance': ['fight_id', 'fighter_id'],
            'clinch': ['fight_id', 'fighter_id'],
            'ground': ['fight_id', 'fighter_id'],
            'td': ['fight_id', 'fighter_id'],
            'sub': ['fight_id', 'fighter_id'],
            'sig_str_rd1': ['fight_id', 'fighter_id'],
            'head_rd1': ['fight_id', 'fighter_id'],
            'body_rd1': ['fight_id', 'fighter_id'],
            'leg_rd1': ['fight_id', 'fighter_id'],
        }
        
        # Track processed tables and stats
        self.processed_tables = set()
        self.processed_stats = {}
        
    def get_feature_tables(self, exclude_patterns: Optional[List[str]] = None) -> List[str]:
        """
        Get all feature-specific tables from the schema.
        Enhanced to use standardized SQL execution and improved filtering.
        
        Args:
            exclude_patterns: List of patterns to exclude (e.g., ['_mapping', 'meta'])
            
        Returns:
            List of table names matching criteria
        """
        if exclude_patterns is None:
            exclude_patterns = ['_mapping', 'fight_stats_core', 'fight_stats_fe', 
                               'fight_stats_derived', 'first_time']
            
        try:
            # Build the exclusion SQL
            exclusion_sql = []
            for pattern in exclude_patterns:
                if pattern.startswith('_'):
                    exclusion_sql.append(f"table_name NOT LIKE '%{pattern}'")
                else:
                    exclusion_sql.append(f"table_name != '{pattern}'")
                    
            # Create query with cleaner formatting
            query = f"""
                SELECT table_name 
                FROM information_schema.tables 
                WHERE table_schema = '{self.schema}'
                AND {" AND ".join(exclusion_sql)}
            """
            
            # Use the enhanced execute_raw_sql method with result fetching
            result_df = self.execute_raw_sql(query, return_results=True)
            
            if result_df.empty:
                self.logger.warning(f"No tables found in schema '{self.schema}'")
                return []
                
            tables = result_df['table_name'].tolist()
            self.logger.info(f"Found {len(tables)} feature-specific tables")
            return tables
            
        except Exception as e:
            self.logger.error(f"Error getting feature tables: {str(e)}")
            raise
    
    def get_features(self, table_name: str) -> List[str]:
        """
        Get all columns for a specific feature table that match filtering criteria.
        
        Args:
            table_name: Name of the feature table
            
        Returns:
            List of column names
        """
        try:
            # Get all columns from the table
            columns = self.feature_utils.get_columns_from_table(
                self.schema,
                table_name
            )
            
            # Apply column filtering
            filtered_columns = self.filter_columns(columns)
            
            self.logger.debug(f"Found {len(filtered_columns)} columns for table {table_name}")
            return filtered_columns
            
        except Exception as e:
            self.logger.error(f"Error getting features for table {table_name}: {str(e)}")
            raise
    
    def calculate_for_table(self, table_name: str, columns: Optional[List[str]] = None) -> str:
        """
        Generate SQL for calculating features for a specific table.
        Must be implemented by subclasses.
        
        Args:
            table_name: Name of the feature table
            columns: Optional list of columns (if None, will fetch all matching columns)
            
        Returns:
            SQL query string for the calculation
        """
        raise NotImplementedError("Subclasses must implement calculate_for_table()")
    
    def execute_for_table(self, table_name: str, columns: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Execute calculation for a specific table and return results.
        
        Args:
            table_name: Name of the feature table
            columns: Optional list of columns (if None, will fetch all matching columns)
            
        Returns:
            DataFrame with calculation results
        """
        try:
            # Get columns if not provided
            if columns is None:
                columns = self.get_features(table_name)
                
            if not columns:
                self.logger.warning(f"No columns to process for table {table_name}")
                return pd.DataFrame()
                
            # Generate SQL
            sql = self.calculate_for_table(table_name, columns)
            
            # Execute SQL and get results
            result_df = self.execute_raw_sql(sql, return_results=True)
            
            # Track processed table
            self.processed_tables.add(table_name)
            
            return result_df
            
        except Exception as e:
            self.logger.error(f"Error executing calculation for {table_name}: {str(e)}")
            raise
    
    def save_for_table(self, table_name: str, columns: Optional[List[str]] = None, 
                       result_df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """
        Save calculation results to a table.
        
        Args:
            table_name: Name of the feature table
            columns: Optional list of columns (if None, will fetch all matching columns)
            result_df: Optional pre-computed result DataFrame (if None, will execute calculation)
            
        Returns:
            DataFrame with saved results
        """
        try:
            # Execute calculation if results not provided
            if result_df is None:
                result_df = self.execute_for_table(table_name, columns)
                
            if result_df is None or result_df.empty:
                self.logger.warning(f"No results to save for table {table_name}")
                return pd.DataFrame()
                
            # Get columns to update
            default_key_columns = ['fight_id', 'fighter_id']
            
            # Get key columns for this table, with fallbacks
            if hasattr(self, 'key_columns') and self.key_columns:
                # Use table-specific key columns if available
                table_key_columns = self.key_columns.get(table_name, default_key_columns)
            else:
                # Fallback to default key columns
                self.logger.warning(f"No key_columns dictionary defined, using default key columns for {table_name}")
                table_key_columns = default_key_columns
                
            # Verify key columns exist in the result DataFrame
            available_key_columns = [col for col in table_key_columns if col in result_df.columns]
            if not available_key_columns:
                # Last resort: use whatever columns appear to be IDs
                self.logger.warning(f"No expected key columns found in results for {table_name}, looking for ID columns")
                available_key_columns = [col for col in result_df.columns if '_id' in col.lower()]
                
            if not available_key_columns:
                self.logger.error(f"Cannot determine key columns for table {table_name}")
                return result_df
                
            # Filter out key columns to get update columns
            update_columns = [col for col in result_df.columns 
                             if col not in available_key_columns]
            
            if not update_columns:
                self.logger.warning(f"No update columns found in results for {table_name}")
                return result_df
                
            self.logger.info(f"Updating table {table_name} with key columns: {available_key_columns}")
            
            # Update the table
            updated_count = self.bulk_update_dataframe(
                df=result_df,
                table_name=table_name,
                schema=self.schema,
                key_columns=available_key_columns
            )
            
            self.logger.info(f"Updated {len(result_df)} rows in {table_name}")
            return result_df
            
        except Exception as e:
            self.logger.error(f"Error saving results for {table_name}: {str(e)}")
            # Log the full error traceback for debugging
            import traceback
            self.logger.error(traceback.format_exc())
            # Return empty DataFrame but don't re-raise to allow processing to continue
            return pd.DataFrame()
    
    def run_sequential(self) -> Dict[str, pd.DataFrame]:
        """
        Process tables sequentially based on calculator type.
        
        Returns:
            Dictionary mapping table names to their result DataFrames
        """
        results = {}
        
        try:
            if self.calculator_type == 'single_table':
                # For single_table calculators, process the specific table
                if hasattr(self, 'table_name') and self.table_name:
                    table_name = self.table_name
                    self.logger.info(f"Processing single table: {table_name}")
                    try:
                        results[table_name] = self.save_for_table(table_name)
                    except Exception as e:
                        self.logger.error(f"Error processing table {table_name}: {str(e)}")
                        import traceback
                        self.logger.error(traceback.format_exc())
                        results[table_name] = pd.DataFrame()
                else:
                    self.logger.error("single_table calculator must define table_name")
                    raise ValueError("single_table calculator must define table_name")
                    
            elif self.calculator_type == 'multi_table':
                # For multi_table calculators, process all feature tables
                try:
                    tables = self.get_feature_tables()
                    self.logger.info(f"Processing {len(tables)} tables")
                    
                    for table_name in tables:
                        try:
                            self.logger.info(f"Processing table: {table_name}")
                            results[table_name] = self.save_for_table(table_name)
                        except Exception as e:
                            self.logger.error(f"Error processing table {table_name}: {str(e)}")
                            import traceback
                            self.logger.error(traceback.format_exc())
                            results[table_name] = pd.DataFrame()
                except Exception as e:
                    self.logger.error(f"Error getting feature tables: {str(e)}")
                    import traceback
                    self.logger.error(traceback.format_exc())
            
            elif self.calculator_type == 'cross_table':
                # For cross_table calculators, use custom_run method
                if hasattr(self, 'custom_run') and callable(self.custom_run):
                    self.logger.info("Running cross-table processing")
                    results = self.custom_run()
                else:
                    self.logger.error("cross_table calculators must implement custom_run")
                    raise NotImplementedError("cross_table calculators must implement custom_run")
                
            return results
        except Exception as e:
            self.logger.error(f"Error in run_sequential: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
            raise  # Re-raise the exception instead of returning results
    
    def run_parallel(self, max_workers: int = 4) -> Dict[str, pd.DataFrame]:
        """
        Process tables in parallel based on calculator type.
        
        Args:
            max_workers: Maximum number of parallel workers
            
        Returns:
            Dictionary mapping table names to their result DataFrames
        """
        if self.calculator_type == 'single_table':
            # No parallelization for single table
            return self.run_sequential()
            
        elif self.calculator_type == 'multi_table':
            # Set up execution plan for parallel processing
            tables = self.get_feature_tables()
            self.logger.info(f"Processing {len(tables)} tables in parallel (max {max_workers} workers)")
            
            # Create execution plan
            plan = ExecutionPlan()
            for table_name in tables:
                plan.add_operation(
                    name=table_name,
                    func=self.save_for_table,
                    table_name=table_name
                )
                
            # Execute plan in parallel
            return plan.execute_parallel(max_workers=max_workers)
            
        elif self.calculator_type == 'cross_table':
            # For cross_table calculators, use custom implementation
            self.logger.info("Using custom cross-table parallel processing")
            if hasattr(self, 'custom_run_parallel'):
                return self.custom_run_parallel(max_workers)
            else:
                self.logger.warning("No custom_run_parallel() found, falling back to sequential execution")
                return self.run_sequential()
    
    def run(self, parallel: bool = False, max_workers: int = 4) -> Dict[str, pd.DataFrame]:
        """
        Run the calculator using either sequential or parallel execution.
        
        Args:
            parallel: Whether to use parallel execution
            max_workers: Maximum number of parallel workers if using parallel execution
            
        Returns:
            Dictionary mapping table names or operation names to their results
        """
        try:
            start_time = time.time()
            self.logger.info(f"Starting {self.calculator_type} calculator")
            
            if parallel and self.calculator_type != 'single_table':
                results = self.run_parallel(max_workers)
            else:
                results = self.run_sequential()
                
            elapsed = time.time() - start_time
            self.logger.info(f"Calculator completed in {elapsed:.2f} seconds")
            return results
            
        except Exception as e:
            self.logger.error(f"Error running calculator: {str(e)}")
            raise
    
    # Legacy support methods
    def calculate(self):
        """Legacy method for backward compatibility. Use run() instead."""
        self.logger.warning("calculate() is deprecated, use run() instead")
        self.run()
        return "Calculation completed"

    def save(self):
        """Legacy method for backward compatibility. Use run() instead."""
        self.logger.warning("save() is deprecated, use run() instead")
        results = self.run()
        if isinstance(results, dict) and results:
            # Try to return the first DataFrame result
            for result in results.values():
                if isinstance(result, pd.DataFrame):
                    return result
        # Return a status DataFrame if no result DataFrames
        return pd.DataFrame({"status": ["completed"], "message": ["Calculation completed for all tables"]})