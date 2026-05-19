import os
import re
from typing import Dict, Any, Optional
from jinja2 import Template, Environment, FileSystemLoader
from sqlalchemy import text
from sqlalchemy.engine import Connection

class SQLTemplateManager:
    """
    Manages SQL templates for feature calculators.
    
    This class provides functionality to:
    1. Load SQL templates from files
    2. Render templates with parameters
    3. Validate SQL syntax
    4. Cache templates for performance
    """
    
    def __init__(self, conn: Connection, template_dir: Optional[str] = None):
        """
        Initialize the SQL template manager.
        
        Args:
            conn: SQLAlchemy connection object or CalculatorContext
            template_dir: Directory containing SQL templates (default: libs/feature_store/sql_templates)
        """
        # Handle the case where a CalculatorContext is passed instead of a connection
        from libs.feature_store.calculator_context import CalculatorContext
        if isinstance(conn, CalculatorContext):
            self.conn = conn.connection
        else:
            self.conn = conn
            
        self.template_dir = template_dir or os.path.join('libs', 'feature_store', 'sql_templates')
        self.templates_cache: Dict[str, Template] = {}
        
        # Set up Jinja2 environment
        self.env = Environment(
            loader=FileSystemLoader(self.template_dir),
            trim_blocks=True,
            lstrip_blocks=True
        )
        
    def get_template_path(self, calculator_type: str, template_name: str) -> str:
        """
        Get the path to a template file.
        
        Args:
            calculator_type: Type of calculator (e.g., 'time_sec', 'age')
            template_name: Name of the template (e.g., 'get_features', 'calculate')
            
        Returns:
            Path to the template file
        """
        # Use forward slashes for Jinja2 template paths
        return f"{calculator_type}/{template_name}.sql"
    
    def get_template(self, calculator_type: str, template_name: str) -> Template:
        """
        Get a template from the cache or load it from disk.
        
        Args:
            calculator_type: Type of calculator (e.g., 'time_sec', 'age')
            template_name: Name of the template (e.g., 'get_features', 'calculate')
            
        Returns:
            Jinja2 Template object
        """
        template_path = self.get_template_path(calculator_type, template_name)
        
        # Check if template is already cached
        if template_path in self.templates_cache:
            return self.templates_cache[template_path]
        
        # Load template from file
        try:
            template = self.env.get_template(template_path)
            self.templates_cache[template_path] = template
            return template
        except Exception as e:
            raise ValueError(f"Error loading template {template_path}: {str(e)}")
    
    def render_template(self, calculator_type: str, template_name: str, params: Dict[str, Any]) -> str:
        """
        Render a template with parameters.
        
        Args:
            calculator_type: Type of calculator (e.g., 'time_sec', 'age')
            template_name: Name of the template (e.g., 'get_features', 'calculate')
            params: Dictionary of parameters to pass to the template
            
        Returns:
            Rendered SQL string
        """
        template = self.get_template(calculator_type, template_name)
        return template.render(**params)
    
    def validate_sql(self, sql: str) -> bool:
        """
        Validate SQL syntax without executing it.
        
        Args:
            sql: SQL string to validate
            
        Returns:
            True if SQL is valid, False otherwise
        """
        try:
            # Prepare the SQL statement without executing it
            self.conn.execute(text(f"EXPLAIN {sql}"))
            return True
        except Exception as e:
            print(f"SQL validation error: {str(e)}")
            return False
    
    def execute_sql(self, sql: str, params: Optional[Dict[str, Any]] = None) -> Any:
        """
        Execute SQL with parameters.
        
        Args:
            sql: SQL string to execute
            params: Dictionary of parameters to pass to the SQL
            
        Returns:
            Result of the SQL execution
        """
        return self.conn.execute(text(sql), params or {}) 