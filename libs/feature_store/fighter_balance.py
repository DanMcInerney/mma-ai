"""
Fighter Balance Module
Ensures fighter1 wins approximately 50% of the time by swapping fighter data
"""
import pandas as pd
from sqlalchemy import text
from typing import Dict, List, Tuple
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FighterBalancer:
    """Balances fighter1 win rate to approximately 50% by swapping fighter data"""
    
    def __init__(self, conn):
        self.conn = conn
        
    def get_current_stats(self) -> Dict[str, int]:
        """Get current win/loss statistics for fighter1"""
        query = text("""
            SELECT 
                COUNT(*) FILTER (WHERE fm.result = 1) as fighter1_wins,
                COUNT(*) FILTER (WHERE fm.result = 0) as fighter1_losses,
                COUNT(*) FILTER (WHERE fm.result IN (0, 1)) as total_decisive,
                COUNT(*) FILTER (WHERE fm.result IN (2, 3)) as no_decisions,
                COUNT(*) as total_fights
            FROM features.fight_mapping fm
            WHERE EXISTS (
                SELECT 1 FROM features.fight_stats_core fc
                WHERE fc.fight_id = fm.fight_id
            )
        """)
        
        result = self.conn.execute(query).fetchone()
        return {
            'fighter1_wins': result[0],
            'fighter1_losses': result[1],
            'total_decisive': result[2],
            'no_decisions': result[3],
            'total_fights': result[4],
            'current_win_rate': result[0] / result[2] if result[2] > 0 else 0
        }
    
    def calculate_swaps_needed(self, stats: Dict[str, int]) -> int:
        """Calculate how many fights need to be swapped"""
        target_wins = stats['total_decisive'] / 2
        current_wins = stats['fighter1_wins']
        swaps_needed = int(current_wins - target_wins)
        
        logger.info(f"Current fighter1 wins: {current_wins}")
        logger.info(f"Target fighter1 wins: {target_wins}")
        logger.info(f"Swaps needed: {abs(swaps_needed)}")
        
        return swaps_needed
    
    def get_fights_to_swap(self, swaps_needed: int) -> List[int]:
        """Get list of fight_ids to swap based on current imbalance"""
        if swaps_needed == 0:
            return []
        
        # If fighter1 wins too much, swap some wins to losses
        if swaps_needed > 0:
            query = text("""
                SELECT fm.fight_id
                FROM features.fight_mapping fm
                WHERE fm.result = 1  -- fighter1 wins
                ORDER BY RANDOM()
                LIMIT :limit
            """)
        # If fighter1 loses too much, swap some losses to wins
        else:
            query = text("""
                SELECT fm.fight_id
                FROM features.fight_mapping fm
                WHERE fm.result = 0  -- fighter1 loses
                ORDER BY RANDOM()
                LIMIT :limit
            """)
        
        result = self.conn.execute(query, {'limit': abs(swaps_needed)})
        return [row[0] for row in result]
    
    def swap_fight_mapping(self, fight_ids: List[int]) -> None:
        """Swap fighter1_id and fighter2_id in fight_mapping table"""
        if not fight_ids:
            return
            
        # Create a mapping of current fighter IDs before swapping
        backup_query = text("""
            SELECT fight_id, fighter1_id, fighter2_id
            FROM features.fight_mapping
            WHERE fight_id = ANY(:fight_ids)
        """)
        
        backup_data = self.conn.execute(backup_query, {'fight_ids': fight_ids}).fetchall()
        logger.info(f"Backing up {len(backup_data)} fight mappings before swap")
        
        # Perform the swap
        swap_query = text("""
            UPDATE features.fight_mapping
            SET 
                fighter1_id = fighter2_id,
                fighter2_id = fighter1_id
            WHERE fight_id = ANY(:fight_ids)
        """)
        
        self.conn.execute(swap_query, {'fight_ids': fight_ids})
        logger.info(f"Swapped fighter IDs for {len(fight_ids)} fights in fight_mapping")
    
    def get_stats_columns(self) -> List[str]:
        """Get all statistic columns that need to be swapped"""
        # Get all columns from fight_stats_core except metadata columns
        query = text("""
            SELECT column_name
            FROM information_schema.columns
            WHERE table_schema = 'features' 
            AND table_name = 'fight_stats_core'
            AND column_name NOT IN ('fight_id', 'fighter_id', 'event_id', 'result')
            ORDER BY ordinal_position
        """)
        
        result = self.conn.execute(query)
        return [row[0] for row in result]
    
    def swap_fight_stats(self, fight_ids: List[int]) -> None:
        """Swap all statistics between fighter1 and fighter2"""
        if not fight_ids:
            return
        
        stats_columns = self.get_stats_columns()
        
        # First, swap the result in fight_mapping table
        swap_result_query = text("""
            UPDATE features.fight_mapping
            SET result = CASE 
                WHEN result = 1 THEN 0
                WHEN result = 0 THEN 1
                ELSE result
            END
            WHERE fight_id = ANY(:fight_ids)
            AND result IN (0, 1)
        """)
        self.conn.execute(swap_result_query, {'fight_ids': fight_ids})
        
        # Create temporary table with swapped data
        create_temp_query = text("""
            CREATE TEMP TABLE swap_data AS
            WITH fight_pairs AS (
                SELECT 
                    fm.fight_id,
                    fm.fighter1_id as new_f1_id,
                    fm.fighter2_id as new_f2_id,
                    fm.fighter2_id as old_f1_id,  -- After fight_mapping swap
                    fm.fighter1_id as old_f2_id   -- After fight_mapping swap
                FROM features.fight_mapping fm
                WHERE fm.fight_id = ANY(:fight_ids)
            )
            SELECT 
                fp.fight_id,
                fp.new_f1_id,
                fp.new_f2_id
            FROM fight_pairs fp
        """)
        
        self.conn.execute(create_temp_query, {'fight_ids': fight_ids})
        
        # Now add the stats columns from both fighters
        # Get the stats columns and add them to the temp table
        for col in stats_columns:
            alter_query = text(f"""
                ALTER TABLE swap_data 
                ADD COLUMN {col} INTEGER,
                ADD COLUMN {col}_1 INTEGER
            """)
            self.conn.execute(alter_query)
        
        # Update the temp table with the actual stats
        update_query = text(f"""
            UPDATE swap_data sd
            SET 
                {', '.join(f'{col} = f2.{col}' for col in stats_columns)},
                {', '.join(f'{col}_1 = f1.{col}' for col in stats_columns)}
            FROM 
                features.fight_stats_core f1,
                features.fight_stats_core f2
            WHERE 
                sd.fight_id = f1.fight_id AND f1.fighter_id = sd.new_f2_id
                AND sd.fight_id = f2.fight_id AND f2.fighter_id = sd.new_f1_id
        """)
        self.conn.execute(update_query)
        
        # Build dynamic update queries for swapping stats
        set_clauses_f1_to_f2 = [f"{col} = sd.{col}" for col in stats_columns]
        set_clauses_f2_to_f1 = [f"{col} = sd.{col}_1" for col in stats_columns]
        
        # Update what was originally fighter1 data (now fighter2 after mapping swap)
        update_f1_query = f"""
            UPDATE features.fight_stats_core fc
            SET 
                {', '.join(set_clauses_f1_to_f2)}
            FROM swap_data sd
            WHERE fc.fight_id = sd.fight_id 
            AND fc.fighter_id = sd.new_f2_id
        """
        
        # Update what was originally fighter2 data (now fighter1 after mapping swap)
        update_f2_query = f"""
            UPDATE features.fight_stats_core fc
            SET 
                {', '.join(set_clauses_f2_to_f1)}
            FROM swap_data sd
            WHERE fc.fight_id = sd.fight_id 
            AND fc.fighter_id = sd.new_f1_id
        """
        
        self.conn.execute(text(update_f1_query))
        self.conn.execute(text(update_f2_query))
        
        # Clean up temp table
        self.conn.execute(text("DROP TABLE IF EXISTS swap_data"))
        
        logger.info(f"Swapped statistics for {len(fight_ids)} fights")
    
    def validate_swap(self, fight_ids: List[int]) -> Dict[str, any]:
        """Validate that the swap was successful"""
        if not fight_ids:
            return {'status': 'No swaps performed', 'valid': True}
        
        # Check that fight_mapping and fight_stats_core are consistent
        validation_query = text("""
            WITH validation AS (
                SELECT 
                    fm.fight_id,
                    fm.fighter1_id,
                    fm.fighter2_id,
                    fc1.fighter_id as stats_f1_id,
                    fc2.fighter_id as stats_f2_id,
                    fm.result,
                    CASE 
                        WHEN fm.fighter1_id = fc1.fighter_id 
                        AND fm.fighter2_id = fc2.fighter_id 
                        THEN 'consistent'
                        ELSE 'inconsistent'
                    END as consistency
                FROM features.fight_mapping fm
                JOIN features.fight_stats_core fc1 
                    ON fm.fight_id = fc1.fight_id AND fc1.fighter_id = fm.fighter1_id
                JOIN features.fight_stats_core fc2 
                    ON fm.fight_id = fc2.fight_id AND fc2.fighter_id = fm.fighter2_id
                WHERE fm.fight_id = ANY(:fight_ids)
            )
            SELECT 
                COUNT(*) as total_checked,
                COUNT(*) FILTER (WHERE consistency = 'consistent') as consistent_count,
                COUNT(*) FILTER (WHERE result = 1) as fighter1_wins,
                COUNT(*) FILTER (WHERE result = 0) as fighter1_losses
            FROM validation
        """)
        
        result = self.conn.execute(validation_query, {'fight_ids': fight_ids}).fetchone()
        
        return {
            'total_checked': result[0],
            'consistent_mappings': result[1],
            'valid_win_loss_pairs': result[2] + result[3],
            'valid': result[0] == result[1] and (result[2] + result[3]) == result[0]
        }
    
    def balance_fighters(self) -> Dict[str, any]:
        """Main method to balance fighter1 win rate to 50%"""
        logger.info("Starting fighter balancing process...")
        
        # Get current statistics
        initial_stats = self.get_current_stats()
        logger.info(f"Initial stats: {initial_stats}")
        
        # Calculate swaps needed
        swaps_needed = self.calculate_swaps_needed(initial_stats)
        
        if swaps_needed == 0:
            logger.info("Fighter1 win rate is already at 50%. No swaps needed.")
            return {
                'initial_stats': initial_stats,
                'final_stats': initial_stats,
                'swaps_performed': 0,
                'success': True
            }
        
        # Get fights to swap
        fight_ids = self.get_fights_to_swap(swaps_needed)
        logger.info(f"Selected {len(fight_ids)} fights to swap")
        
        # Perform swaps
        try:
            # Start transaction
            self.conn.execute(text("BEGIN"))
            
            # Swap fight_mapping first
            self.swap_fight_mapping(fight_ids)
            
            # Then swap fight_stats
            self.swap_fight_stats(fight_ids)
            
            # Validate the swap
            validation = self.validate_swap(fight_ids)
            
            if not validation['valid']:
                raise ValueError(f"Swap validation failed: {validation}")
            
            # Commit if validation passes
            self.conn.commit()
            logger.info("Fighter balancing completed successfully")
            
        except Exception as e:
            # Rollback on any error
            self.conn.rollback()
            logger.error(f"Error during fighter balancing: {e}")
            raise
        
        # Get final statistics
        final_stats = self.get_current_stats()
        logger.info(f"Final stats: {final_stats}")
        
        return {
            'initial_stats': initial_stats,
            'final_stats': final_stats,
            'swaps_performed': len(fight_ids),
            'fight_ids_swapped': fight_ids,
            'validation': validation,
            'success': True
        }
    
    def run_integrity_check(self) -> Dict[str, any]:
        """Run comprehensive integrity checks on the database"""
        checks = {}
        
        # Check 1: Fighter mapping consistency
        mapping_check = text("""
            SELECT COUNT(*) as inconsistent_mappings
            FROM features.fight_mapping fm
            WHERE NOT EXISTS (
                SELECT 1 FROM features.fight_stats_core fc1
                WHERE fc1.fight_id = fm.fight_id AND fc1.fighter_id = fm.fighter1_id
            )
            OR NOT EXISTS (
                SELECT 1 FROM features.fight_stats_core fc2
                WHERE fc2.fight_id = fm.fight_id AND fc2.fighter_id = fm.fighter2_id
            )
        """)
        
        result = self.conn.execute(mapping_check).fetchone()
        checks['mapping_consistency'] = result[0] == 0
        
        # Check 2: Result consistency (W-L pairs)
        result_check = text("""
            WITH fight_results AS (
                SELECT 
                    fm.fight_id,
                    fm.result,
                    COUNT(*) FILTER (WHERE fc.fighter_id = fm.fighter1_id) as f1_count,
                    COUNT(*) FILTER (WHERE fc.fighter_id = fm.fighter2_id) as f2_count
                FROM features.fight_mapping fm
                JOIN features.fight_stats_core fc 
                    ON fm.fight_id = fc.fight_id
                WHERE fm.result IN (0, 1)
                GROUP BY fm.fight_id, fm.result
            )
            SELECT 
                COUNT(*) as total_decisive,
                COUNT(*) FILTER (WHERE f1_count = 1 AND f2_count = 1) as valid_pairs
            FROM fight_results
        """)
        
        result = self.conn.execute(result_check).fetchone()
        checks['result_consistency'] = result[0] == result[1]
        checks['result_details'] = {
            'total_decisive_fights': result[0],
            'valid_win_loss_pairs': result[1]
        }
        
        # Check 3: Duplicate fighter check
        duplicate_check = text("""
            SELECT fight_id, COUNT(*) as fighter_count
            FROM features.fight_stats_core
            GROUP BY fight_id
            HAVING COUNT(*) != 2
        """)
        
        result = self.conn.execute(duplicate_check).fetchall()
        checks['no_duplicate_fighters'] = len(result) == 0
        
        # Check 4: Final win rate
        stats = self.get_current_stats()
        checks['final_win_rate'] = stats['current_win_rate']
        checks['win_rate_deviation'] = abs(0.5 - stats['current_win_rate'])
        
        return checks 