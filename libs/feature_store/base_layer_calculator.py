from libs.feature_store.base_calculator import BaseCalculator
from typing import List, Dict, Any, Set
from sqlalchemy import text

class BaseLayerCalculator(BaseCalculator):
    """Base class for layered feature calculations."""
    
    def __init__(self, conn, calculator_type='multi_table'):
        super().__init__(conn, calculator_type=calculator_type)
        self.stat_tables = self.feature_utils.get_stat_tables()
        self.layer_suffix = ''  # Override in child classes (e.g., '_ratio', '_sdev')
        self.exclude_patterns: Set[str] = set()
        self.include_patterns: Set[str] = set()
        
    def add_exclude_pattern(self, pattern: str) -> None:
        """Add a pattern to exclude from calculations."""
        self.exclude_patterns.add(pattern)
        
    def add_include_pattern(self, pattern: str) -> None:
        """Add a pattern to include in calculations."""
        self.include_patterns.add(pattern)
        
    def should_process_column(self, column: str) -> bool:
        """Check if a column should be processed based on include/exclude patterns.
        
        If include patterns are specified, column must match at least one.
        If exclude patterns are specified, column must not match any.
        If both are specified, both conditions must be met.
        If neither are specified, all columns are processed.
        """
        # Check exclude patterns first
        if self.exclude_patterns and any(pattern in column for pattern in self.exclude_patterns):
            return False
            
        # If include patterns exist, column must match at least one
        if self.include_patterns:
            return any(pattern in column for pattern in self.include_patterns)
            
        # If no include patterns, and we passed exclude check, process the column
        return True
        
    def get_features(self) -> List[str]:
        """Get list of all features to be calculated."""
        self.features = []
        for table, columns in self.stat_tables.items():
            for col in columns:
                if self.should_process_column(col):
                    self.features.append(f"{col}{self.layer_suffix}")
        return self.features

    def calculate_for_table(self, table: str, columns: List[str]) -> str:
        """Calculate layered features for a specific table.
        
        Args:
            table: The source table name
            columns: List of columns to calculate features for
            
        Returns:
            SQL query string for the calculations
        """
        raise NotImplementedError("Child classes must implement calculate_for_table")

    def calculate(self) -> None:
        """Calculate all layered features."""
        for table, columns in self.stat_tables.items():
            # Filter columns based on exclude patterns
            filtered_columns = [col for col in columns if self.should_process_column(col)]
            if not filtered_columns:
                continue
                
            print(f"\nGenerating {self.layer_suffix} calculations for {table} with {len(filtered_columns)} columns...")
            table_sql = self.calculate_for_table(table, filtered_columns)
            if table_sql:
                self.execute_layer_update(
                    calculation_sql=table_sql,
                    table_name=table
                )

    def save(self) -> None:
        """Not needed as calculate() handles updates directly"""
        pass
