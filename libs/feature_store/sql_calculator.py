from typing import Dict, List, Any, Optional
from sqlalchemy import text
from libs.feature_store.base import BaseFeatureStore
from libs.feature_store.calculator_context import CalculatorContext


class SQLCalculator(BaseFeatureStore):
    """
    Base class for calculators that primarily use SQL for their calculations.
    
    This class extends BaseFeatureStore and adds support for CalculatorContext,
    allowing SQL-based calculators to be used with the dependency injection pattern.
    """
    
    def __init__(self, conn_or_context, calculator_type=None):
        """
        Initialize with either a connection or a calculator context.
        
        Args:
            conn_or_context: SQLAlchemy connection or CalculatorContext
            calculator_type: Type of calculator (used for SQL templates)
        """
        # Handle both connection and context for backward compatibility
        if isinstance(conn_or_context, CalculatorContext):
            self.context = conn_or_context
            super().__init__(conn_or_context.connection)
            self.schema = conn_or_context.schema
        else:
            self.context = CalculatorContext(conn_or_context)
            super().__init__(conn_or_context)
            self.schema = 'features'
            
        self.calculator_type = calculator_type
        self.table_name = 'fight_stats_fe'
        
    def execute_sql_template(self, template_name: str, params: Optional[Dict[str, Any]] = None) -> None:
        """
        Execute a SQL template with the given parameters.
        
        Args:
            template_name: Name of the SQL template
            params: Parameters to pass to the template
        """
        if not self.calculator_type:
            raise ValueError("calculator_type must be set to use SQL templates")
            
        # Use SQL template if available
        if hasattr(self.context, 'sql_manager') and self.context.sql_manager:
            template_params = {'schema': self.schema}
            if params:
                template_params.update(params)
                
            query = self.context.sql_manager.render_template(
                self.calculator_type, 
                template_name,
                template_params
            )
            self.execute_raw_sql(query)
        else:
            raise ValueError("SQL template manager not available in context")
            
    def _ensure_columns_exist(self, columns: Dict[str, str], table_name: str = None, schema: str = None):
        """
        Ensure columns exist in target table, create if missing.
        Uses the context schema if not specified.
        
        Args:
            columns: Dict mapping column names to their SQL types
            table_name: Target table name (without schema)
            schema: Target schema name
        """
        table_name = table_name or self.table_name
        schema = schema or self.schema
        super()._ensure_columns_exist(columns, table_name, schema) 