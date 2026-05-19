import logging
import pandas as pd
from typing import Set, List, Dict, Optional, Any
from sqlalchemy import text
from libs.feature_store.base_calculator import BaseCalculator
from libs.feature_store.calculator_context import CalculatorContext
from libs.feature_store.feature_utils import FeatureUtils

class WeightclassMeanCalculator(BaseCalculator):
    """
    Precompute weightclass mean values for each stat by weightclass.
    
    For each table in self.stat_tables, compute the mean of each stat's values
    by weightclass, which will serve as priors for the new adjusted performance 
    calculations with reliability-weighted shrinkage.
    
    The results are stored in a table:
         features.<table>_wc_mean
         
    with each computed column named as <stat>_wc_mean.
    """

    def __init__(self, context_or_conn, include_patterns: Set[str] = set(), exclude_patterns: Set[str] = set()):
        """
        Initialize the weightclass mean calculator.
        
        Args:
            context_or_conn: CalculatorContext or database connection
            include_patterns: Set of patterns to include in calculation
            exclude_patterns: Set of patterns to exclude from calculation
        """
        # Handle context initialization
        try:
            # Try to check if it's a CalculatorContext object or has the required attributes
            if hasattr(context_or_conn, 'connection') and hasattr(context_or_conn, 'feature_utils'):
                self.context = context_or_conn
                conn = self.context.connection
            else:
                conn = context_or_conn
                self.context = CalculatorContext(conn)
        except (TypeError, AttributeError):
            # Fallback for cases where context_or_conn is a connection or mock
            conn = context_or_conn
            self.context = CalculatorContext(conn)
        
        # Initialize BaseCalculator with multi_table calculator type
        super().__init__(conn, calculator_type='multi_table')
        
        self.table_suffix = '_wc_mean'
        self.include_patterns = include_patterns
        self.exclude_patterns = exclude_patterns
        self.logger = logging.getLogger(__name__)
        
        # Date range (matching with other calculators)
        self.start_date = '2014-01-01'
        self.end_date = '2023-01-01'
        
        # Minimum sample size for reliable statistics (can be overridden for testing)
        self.min_sample_size = 10
        
        # Set up stat tables and include/exclude patterns
        self.stat_tables = self.context.feature_utils.get_stat_tables()

        for pattern in include_patterns:
            self.add_include_pattern(pattern)
        for pattern in exclude_patterns:
            self.add_exclude_pattern(pattern)

    def execute_sql_template(self, template_name: str, operation: str, params: Dict[str, Any]) -> str:
        """Execute a SQL template and return the results.
        
        Args:
            template_name: Name of the template to use
            operation: Operation to perform (e.g., 'calculate')
            params: Parameters to pass to the template
            
        Returns:
            Rendered SQL query string
        """
        try:
            if hasattr(self.context, 'sql_manager') and self.context.sql_manager is not None:
                sql = self.context.sql_manager.render_template(
                    template_name, operation, params
                )
            else:
                sql = self.sql_template_manager.render_template(
                    template_name, operation, params
                )
                
            # Validate SQL syntax
            if not sql or 'ERROR' in sql.upper():
                self.logger.error(f"Invalid SQL template: {sql}")
                raise ValueError(f"Invalid SQL template for {template_name}.{operation}")
                
            return sql
        except Exception as e:
            self.logger.error(f"Error executing SQL template {template_name}.{operation}: {str(e)}")
            raise



    def _create_weightclass_mean_table(self, table_name: str, columns: List[str]) -> None:
        """Create a new table for storing weightclass mean values."""
        try:
            # Get a list of actual column names from the table (database-agnostic)
            try:
                # Try PostgreSQL-style query first
                column_query = f"""
                SELECT column_name 
                FROM information_schema.columns 
                WHERE table_schema = 'features' AND table_name = '{table_name}'
                """
                all_table_columns = [row[0] for row in self.conn.execute(text(column_query)).fetchall()]
            except Exception:
                # Fallback to SQLite-style query
                column_query = f"PRAGMA table_info({table_name})"
                pragma_result = self.conn.execute(text(column_query)).fetchall()
                all_table_columns = [row[1] for row in pragma_result]  # Column name is at index 1
            
            # Determine schema prefix (PostgreSQL uses 'features.' but SQLite doesn't)
            schema_prefix = ""
            table_prefix = ""
            try:
                # Check if we're using PostgreSQL by trying a PostgreSQL-specific query
                self.conn.execute(text("SELECT 1 FROM information_schema.tables LIMIT 1"))
                schema_prefix = "features."
                table_prefix = "features."
            except Exception:
                # We're using SQLite or another database without schemas
                schema_prefix = ""
                table_prefix = ""

            # Drop existing table if it exists
            drop_sql = text(f"DROP TABLE IF EXISTS {schema_prefix}{table_name}{self.table_suffix}")
            self.conn.execute(drop_sql)
            self.conn.commit()

            # Include ALL columns except IDs (keep it simple)
            relevant_columns = []
            for col in all_table_columns:
                if col in ['fight_id', 'fighter_id', 'event_id']:
                    continue  # Skip ID columns
                    
                if self.should_process_column(col):
                    relevant_columns.append(col)
            
            if not relevant_columns:
                self.logger.warning(f"No relevant columns found in {table_name}")
                return

            # Build mean expressions for each column (simple average)
            mean_selects = []
            for col in relevant_columns:
                mean_selects.append(f"""
                    CAST(AVG(t.{col}) AS real) AS {col}_wc_mean
                """)
            
            if not mean_selects:
                self.logger.warning(f"No mean expressions generated for {table_name}")
                return
                
            mean_sql_str = ",\n    ".join(mean_selects)

            # Use inline SQL for reliability
            self.logger.info(f"Creating weightclass mean table for {table_name}")
            create_sql = f"""
            CREATE TABLE {schema_prefix}{table_name}{self.table_suffix} AS
            WITH fighter_fights AS (
                SELECT
                    t.fighter_id,
                    fm.weightclass,
                    t.*
                FROM {table_prefix}{table_name} t
                JOIN {table_prefix}fight_mapping fm ON fm.fight_id = t.fight_id
                JOIN {table_prefix}event_mapping e ON fm.event_id = e.event_id
                WHERE e.event_date BETWEEN '{self.start_date}' AND '{self.end_date}'
            )
            SELECT
                weightclass,
                {mean_sql_str}
            FROM fighter_fights t
            GROUP BY weightclass
            HAVING COUNT(*) >= {self.min_sample_size}  -- Minimum sample size for reliable statistics
            ORDER BY weightclass;
            """
            
            # Execute the SQL
            self.conn.execute(text(create_sql))
            self.conn.commit()

            # Create index on weightclass
            index_sql = f"""
            CREATE INDEX IF NOT EXISTS idx_{table_name}{self.table_suffix}_wclass
            ON {schema_prefix}{table_name}{self.table_suffix}(weightclass);
            """
            self.conn.execute(text(index_sql))
            self.conn.commit()
            
            self.logger.info(f"Successfully created weightclass mean table: {table_name}{self.table_suffix}")

        except Exception as e:
            self.logger.error(f"Error creating weightclass mean table for {table_name}: {str(e)}")
            self.conn.rollback()
            raise

    def _validate_weightclass_mean_stats(self, table_name: str) -> None:
        """Validate the computed weightclass mean statistics."""
        try:
            # Determine schema prefix
            schema_prefix = ""
            try:
                self.conn.execute(text("SELECT 1 FROM information_schema.tables LIMIT 1"))
                schema_prefix = "features."
            except Exception:
                schema_prefix = ""
                
            # Query the computed stats
            validation_sql = f"""
            SELECT * FROM {schema_prefix}{table_name}{self.table_suffix}
            """
            stats_df = pd.read_sql(validation_sql, self.conn)
            
            # Validation checks
            if stats_df.empty:
                self.logger.warning(f"No statistics computed for {table_name}")
                return
                
            if stats_df['weightclass'].nunique() < 8:
                self.logger.warning(f"Missing weightclasses in {table_name} (found {stats_df['weightclass'].nunique()} of expected 8+)")
                
            # Check for invalid values
            for col in stats_df.columns:
                if col != 'weightclass':
                    if stats_df[col].isnull().any():
                        self.logger.warning(f"NULL values found in {table_name}.{col}")
                                    # Basic validation - just check for negative values where they shouldn't exist
                if '_per_min' in col and stats_df[col].min() < 0:
                    self.logger.warning(f"Negative value in {table_name}.{col}: {stats_df[col].min():.3f}")
            
            self.logger.info(f"Validated {table_name}{self.table_suffix}: {len(stats_df)} rows, {len(stats_df.columns)} columns")
        except Exception as e:
            self.logger.error(f"Error validating stats for {table_name}: {str(e)}")

    def precompute_weightclass_mean_for_all_tables(self) -> Dict[str, pd.DataFrame]:
        """
        Precompute weightclass mean statistics for all feature tables.
        
        Returns:
            Dictionary of precomputed statistics DataFrames by table
        """
        results = {}
        tables_to_process = list(self.stat_tables.keys())
        total_tables = len(tables_to_process)
        
        self.logger.info(f"Starting weightclass mean calculation for {total_tables} tables")
        print(f"\n=== Starting weightclass mean calculation for {total_tables} tables ===")
        
        # For each stat table, precompute weightclass means
        for i, table_name in enumerate(tables_to_process, 1):
            try:
                self.logger.info(f"[{i}/{total_tables}] Computing weightclass mean stats for {table_name}")
                print(f"  └─ [{i}/{total_tables}] Processing {table_name}")
                
                # Create the weightclass mean table
                self._create_weightclass_mean_table(table_name, [])  # Pass empty columns list, we'll query the actual columns
                
                # Validate the results
                self._validate_weightclass_mean_stats(table_name)
                
                # Store computed results
                try:
                    # Determine schema prefix
                    schema_prefix = ""
                    try:
                        self.conn.execute(text("SELECT 1 FROM information_schema.tables LIMIT 1"))
                        schema_prefix = "features."
                    except Exception:
                        schema_prefix = ""
                    
                    stats_df = pd.read_sql(f"SELECT * FROM {schema_prefix}{table_name}{self.table_suffix}", self.conn)
                    results[table_name] = stats_df
                    print(f"  └─ ✓ Completed {table_name}: {len(stats_df)} rows")
                except Exception as e:
                    self.logger.warning(f"Could not load stats for {table_name}: {str(e)}")
                    print(f"  └─ ✗ Could not load stats for {table_name}")
                
            except Exception as e:
                self.logger.error(f"Error processing {table_name}: {str(e)}")
                print(f"  └─ ✗ Error processing {table_name}: {str(e)}")
                # Continue with other tables
                
        print(f"=== Weightclass mean calculation completed ===\n")
        self.logger.info(f"Completed weightclass mean calculation for {len(results)} tables")
        return results

    def calculate_for_table(self, table: str, columns: Optional[List[str]] = None) -> str:
        """Not used for this calculator as we handle all tables at once."""
        return ""
        
    def execute_for_table(self, table_name: str, columns: Optional[List[str]] = None) -> pd.DataFrame:
        """Not used for this calculator as we handle all tables at once."""
        return pd.DataFrame()
        
    def run(self, parallel: bool = False, max_workers: int = 4, table_pattern: str = "") -> Dict[str, pd.DataFrame]:
        """
        Run the weightclass mean calculator for all tables.
        
        Args:
            parallel: Whether to run in parallel (not used)
            max_workers: Number of workers for parallel execution (not used)
            table_pattern: Optional pattern to filter tables
            
        Returns:
            Dictionary of precomputed statistics by table
        """
        return self.precompute_weightclass_mean_for_all_tables()
