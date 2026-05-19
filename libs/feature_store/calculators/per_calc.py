from libs.feature_store.base_calculator import BaseCalculator
from typing import List, Dict, Optional, Any
import pandas as pd
import logging
from libs.feature_store.calculator_context import CalculatorContext


class PerCalculator(BaseCalculator):
    """
    Calculator for computing various derived 'per' features.
    
    Creates derived features like ko_power, sig_str_perc, distance_range, etc.
    Each feature is placed in the appropriate feature-specific table.
    """
    
    def __init__(self, conn_or_context, calculator_type='multi_table'):
        """
        Initialize with either a connection or a calculator context.
        
        Args:
            conn_or_context: SQLAlchemy connection or CalculatorContext
            calculator_type: Type of calculator ('single_table', 'multi_table', 'cross_table')
        """
        # Handle both connection and context for backward compatibility
        if isinstance(conn_or_context, CalculatorContext):
            self.context = conn_or_context
            super().__init__(conn_or_context.connection, calculator_type)
        else:
            self.context = CalculatorContext(conn_or_context)
            super().__init__(conn_or_context, calculator_type)
            
        self.schema = 'features'
        self.feature_type = 'per'  # Used for SQL template identifier
        
        # Set up logging
        self.logger = logging.getLogger(__name__)
        
        # Define feature mappings: feature_name -> target_table
        self.feature_mappings = {
            # Power features -> ko table
            'ko_per_sig_str_land': 'ko',
            
            # Strike features -> sig_str table  
            'sig_str_per_str_att': 'sig_str',
            
            # Range features -> distance/clinch/ground tables
            'distance_per_sig_str_land': 'distance',
            'clinch_per_sig_str_land': 'clinch', 
            'ground_per_sig_str_land': 'ground',
            
            # Target features -> head/body tables
            'head_per_sig_str_land': 'head',
            'body_leg_per_sig_str_land': 'body',  # Put in body table since it combines body+leg
            
            # TD features -> td table
            'td_per_sig_str_att': 'td',
            'td_land_per_ctrl': 'td',  # Takedowns landed per control time
            
            # Ground features -> ground table
            'ground_land_per_ctrl': 'ground',  # Ground strikes per control time
            'ground_land_per_td_land': 'ground',  # Ground strikes per takedown landed
            
            # Submission features -> sub table
            'sub_att_per_ctrl': 'sub',  # Submission attempts per control time
            'sub_per_all_ctrl': 'sub',  # Submissions per combined control time

            # Reversal features -> rev table
            'rev_per_ctrlopp': 'rev',  # Reversals per opponent control time
            
            # Finishing features -> ko table (since they involve ko)
            'ko_sub_rd1_per_win': 'ko',
            'ko_sub_per_win': 'ko'
        }
        
        # Setup execution plan
        self.execution_plan.add_operation(
            'calculate_features',
            self.calculate
        )
        self.execution_plan.add_operation(
            'save_features', 
            self.save
        )

    def get_features(self, table_name: str = None) -> List[str]:
        """
        Get features that this calculator creates for a specific table.
        
        Args:
            table_name: Name of the feature table (if None, returns all features)
        
        Returns:
            List of feature names for the specified table
        """
        if table_name is None:
            return list(self.feature_mappings.keys())
        
        # Return only features mapped to this specific table
        return [feature for feature, target_table in self.feature_mappings.items() 
                if target_table == table_name]

    def calculate(self, table_name: str = None, columns: List[str] = None) -> Dict[str, Any]:
        """
        Calculate all per-features and organize by target table.
        
        Returns:
            Dictionary with SQL queries organized by target table
        """
        # Group features by target table
        table_features = {}
        for feature, target_table in self.feature_mappings.items():
            if target_table not in table_features:
                table_features[target_table] = []
            table_features[target_table].append(feature)
        
        # Generate SQL for each target table
        table_sqls = {}
        for target_table, features in table_features.items():
            sql = self._generate_sql_for_table(target_table, features)
            table_sqls[target_table] = sql
            
        return {
            "status": "success", 
            "table_sqls": table_sqls,
            "feature_count": len(self.feature_mappings),
            "table_count": len(table_features)
        }

    def _generate_sql_for_table(self, target_table: str, features: List[str]) -> str:
        """
        Generate SQL for calculating features for a specific target table.
        
        Args:
            target_table: Name of the target table (ko, sig_str, etc.)
            features: List of feature names to calculate for this table
            
        Returns:
            SQL query string
        """
        # Check if any features need opponent data
        needs_opponent_data = any(feature in ['rev_per_ctrlopp', 'sub_per_all_ctrl'] for feature in features)
        
        # Build the feature calculations
        feature_calcs = []
        
        for feature in features:
            calc_sql = self._get_feature_calculation(feature, needs_opponent_data)
            if calc_sql:
                feature_calcs.append(f"    {calc_sql} AS {feature}")
        
        if not feature_calcs:
            return ""
            
        # Build the SQL with or without opponent joins
        if needs_opponent_data:
            sql = f"""
        SELECT 
            f1.fight_id,
            f1.fighter_id,
            f1.event_id,
{','.join(feature_calcs)}
        FROM {self.schema}.fight_stats_derived f1
        JOIN {self.schema}.fight_mapping fm ON f1.fight_id = fm.fight_id
        LEFT JOIN {self.schema}.fight_stats_derived f2 ON f1.fight_id = f2.fight_id 
            AND f2.fighter_id = CASE 
                WHEN f1.fighter_id = fm.fighter1_id THEN fm.fighter2_id 
                ELSE fm.fighter1_id 
            END
        WHERE 1=1
        """
        else:
            sql = f"""
        SELECT 
            fight_id,
            fighter_id,
            event_id,
{','.join(feature_calcs)}
        FROM {self.schema}.fight_stats_derived
        WHERE 1=1
        """
        
        return sql

    def _get_feature_calculation(self, feature_name: str, needs_opponent_data: bool = False) -> str:
        """
        Get the SQL calculation for a specific feature.
        
        Args:
            feature_name: Name of the feature to calculate
            needs_opponent_data: Whether this calculation needs opponent data (affects table prefixes)
            
        Returns:
            SQL calculation string
        """
        # Use table prefixes when opponent data is needed
        prefix = "f1." if needs_opponent_data else ""
        
        calculations = {
            # Power
            'ko_per_sig_str_land': f"""
                CASE 
                    WHEN {prefix}sig_str_land > 0 THEN 
                        CAST({prefix}ko AS FLOAT) / CAST({prefix}sig_str_land AS FLOAT)
                    ELSE 0.0
                END""",
            
            # Power shot ratio
            'sig_str_per_str_att': f"""
                CASE 
                    WHEN {prefix}strikes_att > 0 THEN 
                        CAST({prefix}sig_str_land AS FLOAT) / CAST({prefix}strikes_att AS FLOAT)
                    ELSE 0.0
                END""",
            
            # Range features
            'distance_per_sig_str_land': f"""
                CASE 
                    WHEN {prefix}sig_str_land > 0 THEN 
                        CAST({prefix}distance_land AS FLOAT) / CAST({prefix}sig_str_land AS FLOAT)
                    ELSE 0.0
                END""",
            
            'clinch_per_sig_str_land': f"""
                CASE 
                    WHEN {prefix}sig_str_land > 0 THEN 
                        CAST({prefix}clinch_land AS FLOAT) / CAST({prefix}sig_str_land AS FLOAT)
                    ELSE 0.0
                END""",
            
            'ground_per_sig_str_land': f"""
                CASE 
                    WHEN {prefix}sig_str_land > 0 THEN 
                        CAST({prefix}ground_land AS FLOAT) / CAST({prefix}sig_str_land AS FLOAT)
                    ELSE 0.0
                END""",
            
            # Target features
            'head_per_sig_str_land': f"""
                CASE 
                    WHEN {prefix}sig_str_land > 0 THEN 
                        CAST({prefix}head_land AS FLOAT) / CAST({prefix}sig_str_land AS FLOAT)
                    ELSE 0.0
                END""",
            
            'body_leg_per_sig_str_land': f"""
                CASE 
                    WHEN {prefix}sig_str_land > 0 THEN 
                        CAST(({prefix}body_land + {prefix}leg_land) AS FLOAT) / CAST({prefix}sig_str_land AS FLOAT)
                    ELSE 0.0
                END""",
            
            # TD features
            'td_per_sig_str_att': f"""
                CASE 
                    WHEN {prefix}sig_str_att > 0 THEN 
                        CAST({prefix}td_att AS FLOAT) / CAST({prefix}sig_str_att AS FLOAT)
                    ELSE 0.0
                END""",
            
            'td_land_per_ctrl': f"""
                CASE 
                    WHEN {prefix}ctrl > 0 THEN 
                        CAST({prefix}td_land AS FLOAT) / CAST({prefix}ctrl AS FLOAT)
                    ELSE 0.0
                END""",
            
            'ground_land_per_ctrl': f"""
                CASE 
                    WHEN {prefix}ctrl > 0 THEN 
                        CAST({prefix}ground_land AS FLOAT) / CAST({prefix}ctrl AS FLOAT)
                    ELSE 0.0
                END""",
            
            'ground_land_per_td_land': f"""
                CASE 
                    WHEN {prefix}td_land > 0 THEN 
                        CAST({prefix}ground_land AS FLOAT) / CAST({prefix}td_land AS FLOAT)
                    ELSE 0.0
                END""",
            
            'sub_att_per_ctrl': f"""
                CASE 
                    WHEN {prefix}ctrl > 0 THEN 
                        CAST({prefix}sub_att AS FLOAT) / CAST({prefix}ctrl AS FLOAT)
                    ELSE 0.0
                END""",
            
            # Grappling - uses opponent's ctrl via JOIN
            'rev_per_ctrlopp': """
                CASE 
                    WHEN f2.ctrl > 0 THEN 
                        CAST(f1.rev AS FLOAT) / CAST(f2.ctrl AS FLOAT)
                    ELSE 0.0
                END""",
            
            # Finishing features
            'ko_sub_rd1_per_win': f"""
                CASE 
                    WHEN {prefix}win > 0 THEN 
                        CAST(({prefix}ko_rd1 + {prefix}sub_land_rd1) AS FLOAT) / CAST({prefix}win AS FLOAT)
                    ELSE 0.0
                END""",
            
            'ko_sub_per_win': f"""
                CASE 
                    WHEN {prefix}win > 0 THEN 
                        CAST(({prefix}ko + {prefix}sub_land) AS FLOAT) / CAST({prefix}win AS FLOAT)
                    ELSE 0.0
                END""",
            
            # Sub motivation - uses combined ctrl from both fighters via JOIN
            'sub_per_all_ctrl': """
                CASE 
                    WHEN (f1.ctrl + COALESCE(f2.ctrl, 0)) > 0 THEN 
                        CAST(f1.sub_att AS FLOAT) / CAST((f1.ctrl + COALESCE(f2.ctrl, 0)) AS FLOAT)
                    ELSE 0.0
                END"""
        }
        
        return calculations.get(feature_name, "")
        
    def calculate_for_table(self, table_name: str, columns: Optional[List[str]] = None) -> str:
        """
        Generate SQL for calculating features for a specific table.
        
        Args:
            table_name: Name of the target table to calculate for
            columns: Optional list of columns to calculate
            
        Returns:
            SQL query string for the calculation
        """
        # Get features for this table
        table_features = [feature for feature, target_table in self.feature_mappings.items() 
                         if target_table == table_name]
        
        if columns:
            # Filter to only requested columns
            table_features = [f for f in table_features if f in columns]
            
        if not table_features:
            return ""
            
        return self._generate_sql_for_table(table_name, table_features)
        
    def execute_for_table(self, table_name: str, columns: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Execute calculation for a specific table and return results.
        
        Args:
            table_name: Name of the target table to calculate for
            columns: Optional list of columns to calculate
            
        Returns:
            DataFrame with calculation results
        """
        try:
            # Calculate SQL query
            sql = self.calculate_for_table(table_name, columns)
            
            if not sql:
                self.logger.warning(f"No SQL generated for {table_name}")
                return pd.DataFrame()
                
            # Execute SQL and return results
            return self.execute_raw_sql(sql, return_results=True)
        except Exception as e:
            self.logger.error(f"Error executing calculation for {table_name}: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
            return pd.DataFrame()
        
    def get_table_features(self, table_name: str) -> List[str]:
        """
        Get features that belong to a specific table.
        
        Args:
            table_name: Name of the target table
            
        Returns:
            List of feature names for that table
        """
        return [feature for feature, target_table in self.feature_mappings.items() 
                if target_table == table_name]
