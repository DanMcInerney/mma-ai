from abc import ABC, abstractmethod
from typing import List, Dict, Any, Union, Optional
import pandas as pd
import numpy as np
import io
import psycopg2.extras
from sqlalchemy import text
from sqlalchemy.engine import Connection

class BaseFeatureStore(ABC):
    def __init__(self, conn: Connection, chunk_size: int = 10000):
        """Initialize with an open connection instead of engine"""
        self.conn = conn
        self.chunk_size = chunk_size
        self._validated_tables = set()  # Track which table+columns we've validated
        self.schema = 'features'

    def _ensure_columns_exist(self, columns: Dict[str, str], table_name: str = 'fight_stats_fe', schema: str = 'features'):
        """
        Ensure columns exist in target table, create if missing
        
        Args:
            columns: Dict mapping column names to their SQL types
                    e.g. {'time_sec_rd1': 'INTEGER', 'some_text': 'VARCHAR(255)'}
            table_name: Target table name (without schema)
            schema: Target schema name
        """
        # Strip schema from table_name if present to avoid duplication
        table_name_clean = table_name.split('.')[-1] if '.' in table_name else table_name
        
        validation_key = f"{schema}.{table_name_clean}"
        if validation_key not in self._validated_tables:
            # Only check existing columns once per table
            self._existing_columns = pd.read_sql(f"""
                SELECT column_name 
                FROM information_schema.columns 
                WHERE table_schema = '{schema}' 
                AND table_name = '{table_name_clean}'
            """, self.conn)['column_name'].tolist()
            self._validated_tables.add(validation_key)

        # Add any missing columns
        for col_name, col_type in columns.items():
            if col_name not in self._existing_columns:
                #print(f"Adding column {col_name} ({col_type}) to {schema}.{table_name_clean}")
                self.execute_raw_sql(f"""
                    ALTER TABLE {schema}.{table_name_clean}
                    ADD COLUMN {col_name} {col_type}
                """)
                self._existing_columns.append(col_name)


    def bulk_insert_dataframe(
        self, 
        df: pd.DataFrame, 
        table_name: str,
        schema: str = 'features',
        if_exists: str = 'append'
    ) -> int:
        """
        Fastest method for pandas DataFrame insertion using copy_expert
        """
        # Convert DataFrame to CSV format in memory
        data = io.StringIO()
        df.to_csv(data, header=True, index=False, na_rep='')
        data.seek(0)
        
        # Get raw connection from SQLAlchemy connection
        raw_conn = self.conn.connection
        try:
            with raw_conn.cursor() as cur:
                columns = ', '.join([f'"{col}"' for col in df.columns.tolist()])
                table_path = f'"{schema}"."{table_name}"'
                copy_sql = f"COPY {table_path} ({columns}) FROM STDIN WITH CSV HEADER DELIMITER ',' QUOTE '\"'"
                
                cur.copy_expert(copy_sql, data)
                self.conn.commit()
                return len(df)
        except Exception as e:
            print(f"\nError during bulk insert: {str(e)}")
            raise

    def bulk_insert_records(
        self,
        records: List[Dict[str, Any]],
        table_name: str,
        schema: str = 'features'
    ) -> int:
        """
        Fastest method for dictionary records using execute_values
        """
        if not records:
            return 0
            
        # Extract column names from first record and ensure they exist
        columns = list(records[0].keys())
        self._ensure_columns_exist({col: 'INTEGER' for col in columns})
        values = [[record[column] for column in columns] for record in records]
        
        # Construct the INSERT query
        query = f"""
            INSERT INTO {schema}.{table_name} ({','.join(columns)})
            VALUES %s
            ON CONFLICT DO NOTHING
        """
        
        # Use psycopg2.extras.execute_values for optimal batch insertion
        raw_conn = self.conn.connection
        try:
            with raw_conn.cursor() as cur:
                psycopg2.extras.execute_values(
                    cur,
                    query,
                    values,
                    template=None,
                    page_size=self.chunk_size
                )
                self.conn.commit()
            return len(records)
        except Exception as e:
            print(f"\nError during bulk insert: {str(e)}")
            raise

    def bulk_insert_numpy(
        self,
        array: np.ndarray,
        columns: List[str],
        table_name: str,
        schema: str = 'features'
    ) -> int:
        """
        Optimized for NumPy array insertion
        """
        # Ensure columns exist
        self._ensure_columns_exist({col: 'INTEGER' for col in columns})

        # Convert to binary format for fastest possible insertion
        buffer = io.BytesIO()
        np.save(buffer, array)
        buffer.seek(0)
        
        raw_conn = self.conn.connection
        try:
            with raw_conn.cursor() as cur:
                cur.copy_expert(
                    f"COPY {schema}.{table_name} ({','.join(columns)}) FROM STDIN WITH (FORMAT binary)",
                    buffer
                )
                self.conn.commit()
            return len(array)
        except Exception as e:
            print(f"\nError during numpy insert: {str(e)}")
            raise

    def execute_raw_sql(self, sql: str, params: Dict[str, Any] = None, return_results: bool = False) -> Union[None, pd.DataFrame]:
        """
        Execute raw SQL using the existing connection with improved error handling and results fetching.
        
        Args:
            sql: SQL query string (can be plain text or SQLAlchemy text object)
            params: Optional dictionary of parameter values for SQL
            return_results: If True, fetch and return results as DataFrame
            
        Returns:
            None if return_results is False, otherwise pandas DataFrame with results
        """
        try:
            # Convert string SQL to SQLAlchemy text object if needed
            if isinstance(sql, str):
                sql = text(sql)
                
            # Execute the query
            result = self.conn.execute(sql, params or {})
            
            # Commit changes if not a read-only query
            if not sql.text.strip().upper().startswith(('SELECT', 'WITH')):
                self.conn.commit()
                
            # Return results if requested
            if return_results:
                # Convert result to DataFrame
                if result.returns_rows:
                    # Get column names
                    columns = result.keys()
                    # Fetch all rows
                    rows = result.fetchall()
                    # Create DataFrame
                    return pd.DataFrame(rows, columns=columns)
                else:
                    # Return empty DataFrame if no rows returned
                    return pd.DataFrame()
            return None
            
        except Exception as e:
            # Handle transaction errors
            if hasattr(self.conn, 'rollback'):
                self.conn.rollback()
            raise RuntimeError(f"SQL execution error: {str(e)}\nQuery: {sql}")
            
    def execute_template(self, template_manager, template_group: str, template_name: str, 
                         template_params: Dict[str, Any], return_results: bool = False) -> Union[None, pd.DataFrame]:
        """
        Execute SQL generated from a template using the template manager.
        
        Args:
            template_manager: SQLTemplateManager instance
            template_group: Template group name (e.g., 'accuracy', 'opponent')
            template_name: Template name (e.g., 'calculate', 'validate')
            template_params: Dictionary of parameters for template rendering
            return_results: If True, fetch and return results as DataFrame
            
        Returns:
            None if return_results is False, otherwise pandas DataFrame with results
        """
        # Render the SQL template
        sql = template_manager.render_template(template_group, template_name, template_params)
        
        # Execute the SQL and return results if requested
        return self.execute_raw_sql(sql, return_results=return_results)

    def execute_calculator_update(self, calculation_sql: str, table_name: str, 
                                 new_columns: List[str], schema: str = 'features') -> pd.DataFrame:
        """
        Execute a calculator SQL query and update the specified columns in the target table.
        Returns the updated data as a DataFrame.
        
        Args:
            calculation_sql: SQL query to execute for calculation
            table_name: Target table name to update
            new_columns: List of new column names to add/update
            schema: Target schema name
            
        Returns:
            DataFrame with the updated data
        """
        try:
            # Ensure columns exist in the target table
            column_types = {col: 'FLOAT' for col in new_columns}
            self._ensure_columns_exist(column_types, table_name, schema)
            
            # Execute the calculation SQL to get the updated values
            result_df = self.execute_raw_sql(calculation_sql, return_results=True)
            
            if result_df.empty:
                raise ValueError(f"No results returned from calculation for {table_name}")
                
            # Convert string columns to numeric types
            for col in new_columns:
                if col in result_df.columns:
                    try:
                        # Try to convert to numeric, with errors='coerce' to handle non-numeric values
                        result_df[col] = pd.to_numeric(result_df[col], errors='coerce')
                        
                        # Check for integer columns based on suffix patterns
                        # PostgreSQL is very strict about types, so we need to ensure integer columns
                        # don't get float values
                        if any(suffix in col for suffix in ['_acc', '_def', '_ratio', '_per_min', '_per_']):
                            # These are typically percentage/ratio fields stored as float
                            result_df[col] = result_df[col].astype(float)
                        elif col.endswith(('_land', '_att')) and '_per_' not in col:
                            # These are typically count fields that should be integers
                            # BUT exclude per_ ratio columns that happen to end with _land
                            result_df[col] = result_df[col].fillna(0).astype(int)
                    except Exception as e:
                        self.logger.warning(f"Could not convert column {col} to numeric: {str(e)}")
                
            # Map the key columns to use for updating
            key_columns = []
            if 'fight_id' in result_df.columns:
                key_columns.append('fight_id')
            if 'fighter_id' in result_df.columns:
                key_columns.append('fighter_id')
                
            if not key_columns:
                raise ValueError(f"No key columns (fight_id, fighter_id) found in calculation results")
                
            # Update the target table with the new calculated values
            self.bulk_update_dataframe(
                df=result_df,
                table_name=table_name,
                schema=schema,
                key_columns=key_columns
            )
            
            return result_df
            
        except Exception as e:
            if hasattr(self.conn, 'rollback'):
                self.conn.rollback()
            raise RuntimeError(f"Calculator update error: {str(e)}")

    def bulk_update_dataframe(
        self,
        df: pd.DataFrame,
        table_name: str,
        schema: str = 'features',
        batch_size: int = 50000,
        key_columns: List[str] = None
    ) -> int:
        """
        Efficiently update existing rows in a table using DataFrame values.
        Uses batch processing and optimized PostgreSQL syntax for maximum performance.
        
        Args:
            df: DataFrame containing update values
            table_name: Target table name
            schema: Target schema name
            batch_size: Number of rows to update in each batch
            key_columns: List of columns that uniquely identify a row
        """
        # Strip schema from table_name if present
        table_name_clean = table_name.split('.')[-1] if '.' in table_name else table_name
        
        # Get default key columns if not provided
        if key_columns is None:
            key_columns = self.get_table_key_columns(table_name)

        # Ensure all columns exist with proper types
        column_types = {}
        for col in df.columns:
            if col not in key_columns:
                # Force FLOAT type for accuracy and similar metrics regardless of dtype
                if any(col.endswith(suffix) for suffix in ['_acc', '_ratio', '_per_min', '_def', '_adjperf', '_sdev', '_avg']):
                    column_types[col] = 'FLOAT'
                else:
                    # Use dtype-based logic for other columns
                    column_types[col] = 'FLOAT' if df[col].dtype.kind in 'fc' else 'INTEGER'
        
        self._ensure_columns_exist(column_types, table_name_clean, schema)
        
        # Prepare column list (excluding key columns)
        update_columns = [col for col in df.columns if col not in key_columns]
        if not update_columns:
            return 0
        
        total_updated = 0
        for start_idx in range(0, len(df), batch_size):
            batch_df = df.iloc[start_idx:start_idx + batch_size]
            
            # Create temporary table for batch (without schema prefix)
            temp_table = f"temp_update_{table_name_clean}_{start_idx}"
            try:
                # Create temp table in the public schema
                batch_df.to_sql(temp_table, self.conn, if_exists='replace', index=False)
                
                # Construct efficient UPDATE using JOIN with explicit type casting
                update_sets = []
                for col in update_columns:
                    # Determine appropriate type cast based on column name pattern
                    # Check for _per_ patterns FIRST before checking _land/_att patterns
                    if any(suffix in col for suffix in ['_acc', '_def', '_ratio', '_per_min', '_per_']):
                        # Ratio/percentage fields - cast to float
                        update_sets.append(f'"{col}" = CAST(tmp."{col}" AS FLOAT)')
                    elif any(col.endswith(suffix) for suffix in ['_land', '_att']) and '_per_' not in col:
                        # Count fields - cast to integer (but exclude _per_ fields that happen to end with _land)
                        update_sets.append(f'"{col}" = CAST(tmp."{col}" AS INTEGER)')
                    else:
                        # Default - use column as is
                        update_sets.append(f'"{col}" = tmp."{col}"')
                        
                where_clause = ' AND '.join([f't."{col}" = tmp."{col}"' for col in key_columns])
                
                update_query = f"""
                    UPDATE {schema}.{table_name_clean} t
                    SET {', '.join(update_sets)}
                    FROM {temp_table} tmp
                    WHERE {where_clause}
                """
                
                # Execute update
                self.execute_raw_sql(update_query)
                
                total_updated += len(batch_df)
                
            finally:
                # Cleanup temp table
                try:
                    self.execute_raw_sql(f"DROP TABLE IF EXISTS {temp_table}")
                except Exception as e:
                    print(f"Warning: Failed to drop temporary table: {e}")
                
        return total_updated

    def execute_layer_update(
        self,
        calculation_sql: str,
        table_name: str,
        schema: str = 'features',
        batch_size: int = 50000
    ) -> None:
        """
        Ultra-high performance update method optimized for layer calculations.
        Uses UNLOGGED tables, parallel operations, and minimal indexing.
        
        Args:
            calculation_sql: SQL query that generates the new columns
            table_name: Target table name (e.g., 'strikes', 'sig_str')
            schema: Target schema name
            batch_size: Number of rows to process in each batch
        """
        # Strip schema if present
        temp_table = f"temp_{table_name}_layer"
        
        try:
            # Create temporary table with calculations
            create_temp_sql = f"""
                DROP TABLE IF EXISTS {temp_table};
                CREATE UNLOGGED TABLE {temp_table} AS 
                {calculation_sql};
                
                -- Create minimal index only on keys
                CREATE INDEX idx_{temp_table}_keys 
                ON {temp_table}(fight_id, fighter_id);
                
                -- Quick analyze for query planner
                ANALYZE {temp_table};
            """
            self.execute_raw_sql(create_temp_sql)
            
            # Get columns from temp table (excluding key columns)
            columns = pd.read_sql(f"""
                SELECT column_name 
                FROM information_schema.columns 
                WHERE table_name = '{temp_table}'
                AND column_name NOT IN ('fight_id', 'fighter_id', 'event_id')
            """, self.conn)['column_name'].tolist()
            
            # Ensure columns exist in target table
            column_types = {col: 'FLOAT' for col in columns}  # Most layer calcs will be FLOAT
            self._ensure_columns_exist(column_types, table_name, schema)
            
            # Get total rows for progress tracking
            total_rows = self.conn.execute(text(f"SELECT COUNT(*) FROM {temp_table}")).scalar()
            
            # Process in batches for better memory management
            for offset in range(0, total_rows, batch_size):
                update_query = f"""
                    UPDATE {schema}.{table_name} t
                    SET {', '.join(f'"{col}" = tmp."{col}"' for col in columns)}
                    FROM (
                        SELECT *
                        FROM {temp_table}
                        ORDER BY fight_id, fighter_id
                        LIMIT {batch_size}
                        OFFSET {offset}
                    ) tmp
                    WHERE t.fight_id = tmp.fight_id 
                    AND t.fighter_id = tmp.fighter_id;
                """
                
                # Execute update
                self.execute_raw_sql(update_query)
                
                print(f"Updated {table_name}: {offset:,} to {min(offset + batch_size, total_rows):,} of {total_rows:,}")
                
        except Exception as e:
            import traceback
            error_msg = f"\nError during layer update:\n{str(e)}\n{traceback.format_exc()}"
            print(error_msg)
            print(f"\nError during layer update: {str(e)}")
            raise Exception(error_msg) from e
        
        finally:
            # Cleanup
            self.execute_raw_sql(f"DROP TABLE IF EXISTS {temp_table}")

    def get_table_key_columns(self, table_name: str) -> List[str]:
        """Get default key columns for a standard feature table."""
        table_name_clean = table_name.split('.')[-1]
        
        # Check if we have predefined key columns for this table
        if hasattr(self, 'key_columns') and table_name_clean in self.key_columns:
            return self.key_columns[table_name_clean]
            
        # Common defaults for standard tables
        if table_name_clean == 'fighter_mapping':
            key_columns = ['fighter_id']
        elif table_name_clean == 'fight_mapping':
            key_columns = ['event_id', 'player1_id', 'player2_id']
        elif table_name_clean == 'event_mapping':
            key_columns = ['event_id']
        elif 'fight_stats' in table_name_clean or table_name_clean.startswith(('sig_str', 'head', 'body', 'leg')):
            # Default for most stat tables
            key_columns = ['fight_id', 'fighter_id']
        else:
            # Last resort fallback
            self.logger.warning(f"No predefined key columns for table {table_name}, defaulting to ['fight_id', 'fighter_id']")
            key_columns = ['fight_id', 'fighter_id']

        return key_columns