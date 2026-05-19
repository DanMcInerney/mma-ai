from libs.feature_store.base_calculator import BaseCalculator
from typing import List, Dict

class KDPowerCalculator(BaseCalculator):
    def __init__(self, conn):
        super().__init__(conn)
        self.table_name = 'fight_stats_derived'
        self.schema = 'features'
        self.power_stats = None
        self.rounds = ['', '_rd1']  # Only using rd1 and total

    def get_features(self) -> List[str]:
        """Load power statistics from the database"""
        # Compute global power stats per weight class
        table = 'fight_stats_derived'
        self.power_stats = self.feature_utils.compute_kd_power_stats(table, pprint=False)
        
        # Return empty list since we're not tracking specific features
        return []

    def calculate(self):
        """Generate SQL for KD power calculations using Beta-Binomial smoothing."""
        # Generate a VALUES clause for alpha/beta per weightclass
        wc_values = []
        for wc, params in self.power_stats.items():
            wc_values.append(f"('{wc}', {params['alpha']}, {params['beta']})")

        wc_values_clause = ",\n".join(wc_values)
        
        power_calcs = []
        for round_suffix in self.rounds:
            kd_col = f"kd{round_suffix}_total"
            distance_col = f"distance_att{round_suffix}_total"
            clinch_col = f"clinch_att{round_suffix}_total"
            total_attempts_col = f"({distance_col} + {clinch_col})"
            
            output_col = f"kd_power_score{round_suffix}"

            calc = f"""
                ROUND(CAST(
                    CASE 
                        WHEN fm.weightclass IS NOT NULL THEN
                            ((wc.alpha + CAST(fs.{kd_col} AS FLOAT)) 
                            / (wc.alpha + wc.beta + CAST({total_attempts_col} AS FLOAT))) * 1000.0
                        ELSE
                            ((params.alpha + CAST(fs.{kd_col} AS FLOAT)) 
                            / (params.alpha + params.beta + CAST({total_attempts_col} AS FLOAT))) * 1000.0
                    END
                AS NUMERIC), 3) as {output_col}
            """
            power_calcs.append(calc)

        # Note the WITH clause defining params and then CROSS JOINing it below
        return f"""
            WITH params AS (
                SELECT 
                    AVG(alpha) AS alpha,
                    AVG(beta) AS beta
                FROM (
                    VALUES
                        {wc_values_clause}
                ) as p(weightclass, alpha, beta)
            )
            SELECT 
                fs.fight_id,
                fs.fighter_id,
                {','.join(power_calcs)}
            FROM {self.schema}.{self.table_name} fs
            LEFT JOIN features.fight_mapping fm ON fs.fight_id = fm.fight_id
            CROSS JOIN params
            LEFT JOIN (
                VALUES 
                    {wc_values_clause}
            ) as wc(weightclass, alpha, beta)
            ON fm.weightclass = wc.weightclass
        """


    def save(self):
        """Save calculated power scores to the database"""
        # Get list of new power score columns
        power_columns = [f"kd_power_score{suffix}" for suffix in self.rounds]
        
        # Execute the calculation SQL and update the table
        self.execute_calculator_update(
            calculation_sql=self.calculate(),
            table_name=self.table_name,
            new_columns=power_columns,
            schema=self.schema
        )
