from typing import Dict, List, Any, Optional, Union
import pandas as pd
import re
from sqlalchemy.engine import Connection
from libs.feature_store.feature_utils import FeatureUtils
from libs.feature_store.sql_template_manager import SQLTemplateManager
from libs.feature_store.base import BaseFeatureStore

class CalculatorContext:
    """
    Context for feature calculators that provides dependency injection.
    
    This class encapsulates:
    1. Database connection
    2. Feature utilities
    3. SQL template manager
    4. Mock data for testing
    
    It provides a consistent interface for calculators to access these dependencies,
    making it easier to test calculators with mock data.
    """
    
    def __init__(
        self, 
        connection: Optional[Connection] = None, 
        mock_data: Optional[Dict[str, pd.DataFrame]] = None,
        schema: str = 'features'
    ):
        """
        Initialize the calculator context.
        
        Args:
            connection: SQLAlchemy connection object (required for real execution)
            mock_data: Dictionary of mock DataFrames for testing (optional)
            schema: Database schema to use (default: 'features')
        """
        self.connection = connection
        self.mock_data = mock_data or {}
        self.schema = schema
        
        # Initialize utilities if we have a connection
        if connection is not None:
            self.feature_utils = FeatureUtils(connection)
            self.sql_manager = SQLTemplateManager(connection)
            self.feature_store = BaseFeatureStore(connection)
        else:
            # For testing without a connection
            self.feature_utils = None
            self.sql_manager = None
            self.feature_store = None
    
    def execute_query(self, query: str, params: Optional[Dict[str, Any]] = None) -> pd.DataFrame:
        """
        Execute a SQL query and return the results as a DataFrame.
        
        If mock_data is provided and contains a key matching the table name in the query,
        return the mock data instead of executing the query.
        
        Args:
            query: SQL query to execute
            params: Parameters for the query
            
        Returns:
            DataFrame with query results
        """
        if self.connection is None:
            raise ValueError("No connection available for query execution")
        
        # Check if we should use mock data
        if self.mock_data:
            # Extract table name from query (simple approach)
            table_match = re.search(r'FROM\s+(\w+\.)?(\w+)', query, re.IGNORECASE)
            if table_match:
                table_name = table_match.group(2)
                if table_name in self.mock_data:
                    return self.mock_data[table_name]
        
        # Execute real query
        return pd.read_sql(query, self.connection, params=params)
    
    def get_columns(
        self, 
        table: str, 
        include_strs: Optional[List[str]] = None, 
        exclude_strs: Optional[List[str]] = None
    ) -> List[str]:
        """
        Get columns from a table that match include/exclude patterns.
        
        Args:
            table: Table name
            include_strs: Strings that must be in column names
            exclude_strs: Strings that must not be in column names
            
        Returns:
            List of column names
        """
        if self.feature_utils is None:
            # For testing, return columns from mock data
            if table in self.mock_data:
                df = self.mock_data[table]
                columns = list(df.columns)
                
                # Apply include/exclude filters
                if include_strs:
                    columns = [c for c in columns if any(s in c for s in include_strs)]
                if exclude_strs:
                    columns = [c for c in columns if not any(s in c for s in exclude_strs)]
                
                return columns
            return []
        
        # Use feature utils for real execution
        return self.feature_utils.get_columns_from_table(
            self.schema, table, include_strs, exclude_strs
        )
    
    def update_table(
        self, 
        table_name: str, 
        data: Union[pd.DataFrame, str], 
        new_columns: Optional[List[str]] = None
    ) -> None:
        """
        Update a table with calculated data.
        
        Args:
            table_name: Name of the table to update
            data: DataFrame with data or SQL query to execute
            new_columns: List of new columns to add to the table
        """
        if isinstance(data, str):
            # It's a SQL query
            if self.connection is None:
                raise ValueError("No connection available for table update")
            
            # Execute the SQL directly
            self.connection.execute(data)
        else:
            # It's a DataFrame
            if table_name in self.mock_data:
                # Update mock data
                mock_df = self.mock_data[table_name]
                
                # Add new columns if needed
                if new_columns:
                    for col in new_columns:
                        if col not in mock_df.columns:
                            mock_df[col] = None
                
                # Update values
                for _, row in data.iterrows():
                    # Find matching rows in mock data
                    mask = (mock_df['fight_id'] == row['fight_id']) & (mock_df['fighter_id'] == row['fighter_id'])
                    
                    # Update each column
                    for col in data.columns:
                        if col not in ['fight_id', 'fighter_id']:
                            mock_df.loc[mask, col] = row[col]
            
            elif self.connection is not None and self.feature_store is not None:
                # Use bulk update for real execution
                self.feature_store.bulk_update_dataframe(
                    data,
                    table_name,
                    schema=self.schema
                )
            else:
                raise ValueError("No connection or mock data available for table update") 