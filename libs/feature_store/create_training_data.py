from sqlalchemy import text
import pandas as pd
from typing import Set, List
from libs.feature_store.features import BASE_STATIC_FEATS

class CreateTrainingData:
    def __init__(self, conn, include_patterns: Set[str] = None, exclude_patterns: Set[str] = None, required_features: Set[str] = None):
        """Initialize the CreateTrainingData class.
        
        Args:
            conn: Database connection
            include_pattern: Set of patterns to include in column selection
            exclude_pattern: Set of patterns to exclude from column selection (optional)
            required_features: Set of features that must be included
        """
        self.conn = conn
        self.static_stats = set(BASE_STATIC_FEATS)
        self.include_patterns = include_patterns
        self.exclude_patterns = exclude_patterns
        self.required_features = required_features

    def build_dataframe(self):
        """Create a single dataframe containing all training data.
        
        Returns:
            pandas.DataFrame: A dataframe containing all training data
        """
        print("Building training dataframe...")
        
        # Get all stats in a single dataframe
        df = self._get_feature_dataframe(
            include_patterns=self.include_patterns,
            exclude_patterns=self.exclude_patterns
        )
        df['event_date'] = pd.to_datetime(df['event_date'])
        
        print(f"Final dataframe shape: {df.shape}")
        return df

    def _get_feature_dataframe(self, include_patterns: Set[str] = set(), exclude_patterns: Set[str] = set()):
        """Build a dataframe with features based on column pattern matching."""
        print(f"Building dataframe with include patterns: {include_patterns} and exclude patterns: {exclude_patterns}")
        
        # Ensure patterns are sets, even if None is passed
        if include_patterns is None:
            include_patterns = set()
        if exclude_patterns is None:
            exclude_patterns = set()
            
        try:
            # Get all relevant tables and their matching columns
            tables = self._get_feature_tables()
            
            # Debug print
            print("\nRequired features:")
            print(self.required_features)
            
            # Filter to just stats tables
            non_stat_tables = set(['fight_', 'event_', 'fighter_', 'first_time', "minimum"])
            tables = [table for table in tables if not any(pre in table for pre in non_stat_tables)]
            
            # Debug print
            print("\nStats tables after filtering:")
            print(tables)
            
            # For each table, get columns matching patterns
            table_columns = {}
            for table in tables:
                columns = self._get_table_columns(table)
                
                # Debug print
                print(f"\nProcessing table {table}")
                
                # Filter columns based on patterns
                filtered_cols = []
                for col in columns:
                    # Skip ID columns
                    if "_id" in col:
                        continue

                    # **Always** include required features even if they match exclude_patterns.
                    if self.required_features and col in self.required_features:
                        filtered_cols.append(col)
                        print(f"Added required feature: {col}")
                        continue

                    # Otherwise, revert to pattern matching
                    matches_include = any(pattern in col for pattern in include_patterns) if include_patterns else True
                    matches_exclude = any(pattern in col for pattern in exclude_patterns)
                    
                    if matches_include and not matches_exclude:
                        filtered_cols.append(col)
                        #print(f"Added pattern-matched feature: {col}")
                
                if filtered_cols:
                    table_columns[table] = filtered_cols
                    print(f"\nFiltered columns for {table}: {filtered_cols}")
                else:
                    print(f"No matching columns found for {table}")
            
            # Build column list for SQL query
            all_columns = []
            for table, columns in table_columns.items():
                alias = f"t_{table}"
                all_columns.extend([(table, alias, col) for col in columns])
            
            # Debug: Print total columns found
            print(f"\nFound {len(all_columns)} matching columns across {len(table_columns)} tables")
            
            # PostgreSQL has a limit of 1664 columns per query
            # If we exceed this, we need to split into multiple queries
            MAX_COLUMNS_PER_QUERY = 1500  # Setting a bit lower than the limit for safety
            
            if not all_columns:
                print("No columns matched the criteria. Returning empty dataframe.")
                return pd.DataFrame()
            
            # Get the base dataframe with IDs and metadata first
            # Include all important metadata columns including fighter1_id and fighter2_id to preserve fight ordering
            base_query = """
                SELECT 
                    fsd.fight_id,
                    fsd.fighter_id,
                    fsd.event_id,
                    e.event_date,
                    fm.method,
                    fm.result,
                    fm.weightclass,
                    fm.weightclass_encoded,
                    f.fighter_name,
                    f.fighter_dob,
                    fm.fighter1_id,
                    fm.fighter2_id
                FROM features.fight_stats_derived fsd
                LEFT JOIN features.event_mapping e ON fsd.event_id = e.event_id
                LEFT JOIN features.fight_mapping fm ON fsd.fight_id = fm.fight_id
                LEFT JOIN features.fighter_mapping f ON fsd.fighter_id = f.fighter_id;
            """
            base_df = pd.read_sql_query(base_query, self.conn)
            print(f"Loaded base dataframe with {len(base_df)} rows and the following metadata columns: {base_df.columns.tolist()}")
            
            # Process columns in chunks to avoid the PostgreSQL column limit
            column_chunks = [all_columns[i:i + MAX_COLUMNS_PER_QUERY] 
                            for i in range(0, len(all_columns), MAX_COLUMNS_PER_QUERY)]
            
            print(f"Split columns into {len(column_chunks)} chunks")
            
            # Start with base dataframe as the result
            result_df = base_df.copy()
            
            # Process each chunk
            for i, column_chunk in enumerate(column_chunks):
                print(f"Processing chunk {i+1}/{len(column_chunks)} with {len(column_chunk)} columns")
                
                # Build column list and table joins for this chunk
                column_list = []
                join_clauses = set()  # Use a set to avoid duplicate joins
                
                for table, alias, col_original in column_chunk:
                    # Ensure we are using the renamed version if applicable
                    col = col_original.replace("7day_", "sevenday_") # Explicitly replace just in case
                    
                    # Quoting logic (no longer strictly needed for sevenday_, but safe)
                    select_col = f'"{col}"' if col[0].isdigit() else col
                    alias_col = f'"{col}"' if col[0].isdigit() else col
                    
                    # Use the potentially renamed 'col' variable
                    column_list.append(f"{alias}.{select_col} as {alias_col}")
                    join_clauses.add(
                        f"LEFT JOIN features.{table} {alias} "
                        f"ON fsd.fight_id = {alias}.fight_id "
                        f"AND fsd.fighter_id = {alias}.fighter_id"
                    )
                
                # Build SELECT query for this chunk - include only the IDs and new columns
                select_sql = f"""
                    SELECT 
                        fsd.fight_id,
                        fsd.fighter_id,
                        fsd.event_id,
                        {', '.join(column_list)}
                    FROM features.fight_stats_derived fsd
                    {' '.join(join_clauses)};
                """
                
                # Debug: Print the first chunk's SQL query to see what's being executed
                if i == 0:
                    print(f"\nDEBUG - First chunk SQL query: {select_sql}")
                
                # Execute query and load results into dataframe
                chunk_df = pd.read_sql_query(select_sql, self.conn)
                print(f"Loaded chunk with {len(chunk_df)} rows and {len(chunk_df.columns)} columns")
                
                # Merge feature columns with result dataframe
                # Use a safer direct merge with a left join to ensure row alignment
                result_df = pd.merge(
                    result_df,
                    chunk_df,
                    on=['fight_id', 'fighter_id', 'event_id'],
                    how='left',
                    suffixes=('', f'_{i}')  # Add unique suffix based on chunk number
                )
                
                # Remove any duplicate ID columns created during the merge
                # These would have the suffix from the current merge
                duplicate_cols = [col for col in result_df.columns if col.endswith(f'_{i}')]
                if duplicate_cols:
                    result_df = result_df.drop(columns=duplicate_cols)
                    print(f"Removed {len(duplicate_cols)} duplicate columns")
                
                print(f"Merged chunk {i+1}, dataframe now has {len(result_df)} rows and {len(result_df.columns)} columns")
            
            # Debug: Check for fighters with fight_id=71 (Volkanovski vs Rodriguez)
            volk_rows = result_df[result_df['fight_id'] == 71]
            if not volk_rows.empty:
                print(f"\nDEBUG - Found {len(volk_rows)} rows for fight_id=71")
                for _, row in volk_rows.iterrows():
                    fighter_name = row['fighter_name']
                    non_null_count = row.count()
                    print(f"Fighter: {fighter_name}, Non-null columns: {non_null_count} out of {len(row)}")
            else:
                print("DEBUG - No rows found for fight_id=71 (Volkanovski vs Rodriguez)")
            
            print(f"Final dataframe has {len(result_df)} rows and {len(result_df.columns)} columns")
            return result_df
            
        except Exception as e:
            print(f"\nError building dataframe: {str(e)}")
            raise
        
        print("Dataframe creation complete!")

    def _get_feature_tables(self) -> list:
        """Get all tables in the features schema."""
        exclude_patterns = [
            '_first_time_opp_stats',
            '_first_time_avg_stats',
            'fight_stats',
            '_mapping',
            '_minimum',
            '_wc_mean',
            '_wc_mad', # min_sdev and min_dec_sdev
        ]
        
        # Always include these tables regardless of exclusion patterns
        include_tables = ['odds']
        
        # Build the NOT LIKE conditions for each pattern
        exclude_conditions = " AND ".join([
            f"t.table_name NOT LIKE '%{pattern}%'"
            for pattern in exclude_patterns
        ])
        
        tables_query = f"""
            SELECT DISTINCT t.table_name 
            FROM information_schema.tables t
            JOIN information_schema.columns c 
                ON t.table_name = c.table_name 
                AND t.table_schema = c.table_schema
            WHERE t.table_schema = 'features' 
            AND t.table_type = 'BASE TABLE'
            AND {exclude_conditions}
            ORDER BY t.table_name;
        """

        # Debug print
        print("\nGetting feature tables...")
        print(f"Query: {tables_query}")
        
        res = [row[0] for row in self.conn.execute(text(tables_query)).fetchall()]
        
        # Add any include_tables that might have been excluded by the query
        for table in include_tables:
            if table not in res:
                # Check if the table actually exists before adding it
                check_query = f"""
                    SELECT 1 
                    FROM information_schema.tables 
                    WHERE table_schema = 'features' 
                    AND table_name = '{table}';
                """
                if self.conn.execute(text(check_query)).fetchone():
                    res.append(table)
                    print(f"Added table to include list: {table}")
        
        # Debug print
        print(f"Found tables: {res}")
        return res

    def _get_table_columns(self, table: str) -> list:
        """Get all columns for a specific table."""
        cols_query = f"""
            SELECT column_name 
            FROM information_schema.columns 
            WHERE table_schema = 'features' 
            AND table_name = '{table}'
            ORDER BY column_name;
        """
        res = [row[0] for row in self.conn.execute(text(cols_query)).fetchall()]
        return res

    def create_training_data(self):
        """Create the training data dataframe.
        
        Returns:
            pandas.DataFrame: A dataframe with all training features
        """
        return self.build_dataframe()
