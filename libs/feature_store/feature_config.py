from typing import List
import pandas as pd
from sqlalchemy.engine import Connection


class FeatureConfig:
    def __init__(self, conn: Connection):
        self.conn = conn
        self.base_stat_prefixes = [
            'strikes', 'sig_str', 'head', 'body', 'leg', 'distance',
            'clinch', 'ground', 'td', 'sub', 'ko', 'kd', 'decision',
            'ctrl', 'win', 'rev', 'time_sec'
        ]


    def get_columns(self, tables: List[str], include_strs: List[str] = None,
                    exclude_strs: List[str] = None, require_strs: List[str] = None) -> List[str]:
        """
        Returns a list of columns based on inclusion, exclusion, and required prefixes.
        """
        include_strs = include_strs or self.base_stat_prefixes
        exclude_strs = exclude_strs or []
        require_strs = require_strs or []

        selected_columns = {}

        # First collect all columns from tables
        for table in tables:
            cols = self.get_columns_from_table(table)
            filtered_cols = []
            # Filter columns based on prefixes
            for col in cols:
                if any(col.startswith(prefix) for prefix in include_strs):
                    if not any(col.startswith(prefix) for prefix in exclude_strs):
                        if all(prefix in col for prefix in require_strs):
                            filtered_cols.append(col)
            
            selected_columns[table] = filtered_cols

        return selected_columns

    def add_suffix(self, stat_name: str, suffix: str) -> str:
        """
        Adds a suffix to a stat name.
        """
        return f"{stat_name}_{suffix}"
    
    def get_columns_from_table(self, schema: str, table: str, exclude_strs: List[str] = []) -> List[str]:
        """
        Returns a list of columns from a table.
        """
        cols = pd.read_sql(
            f"""
            SELECT column_name 
            FROM information_schema.columns 
            WHERE table_schema = '{schema}' 
            AND table_name = '{table}'
            """, 
            self.conn
        )['column_name'].tolist()
        filtered_cols = [col for col in cols if not any(exclude in col for exclude in exclude_strs)]
        return filtered_cols
