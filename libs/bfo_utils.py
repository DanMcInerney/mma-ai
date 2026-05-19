import pandas as pd
import numpy as np
from sqlalchemy import create_engine, text
from sqlalchemy_utils import database_exists, create_database
from thefuzz import fuzz, process
from contextlib import contextmanager
from tqdm import tqdm
import logging
from libs.paths import database_url, odds_database_url

class BFOUtils:
    """
    Utility class for managing and matching fighter names between ufcstats.com
    and bestfightodds.com data sources.
    """
    
    def __init__(self, mma_conn, odds_db_url=None):
        """
        Initialize the BFOUtils.
        
        Args:
            mma_conn: Database connection object to the main MMA database
            odds_db_url: Connection string for the odds database
        """
        self.mma_conn = mma_conn
        self.odds_db_url = odds_db_url or odds_database_url()
        self.odds_engine = self._create_odds_engine()
        self._initialize_name_matching_table()
        
    def _create_odds_engine(self):
        """Create and configure the database engine for the odds database"""
        if not database_exists(self.odds_db_url):
            create_database(self.odds_db_url)
        
        return create_engine(
            self.odds_db_url,
            pool_size=5,
            max_overflow=10,
            pool_timeout=30,
            pool_recycle=14400
        )
    
    @contextmanager
    def _get_odds_connection(self):
        """Context manager for connections to the odds database"""
        connection = self.odds_engine.connect()
        try:
            yield connection
        finally:
            connection.close()
    
    def _initialize_name_matching_table(self):
        """Create the name_matching table in the bestfightodds schema"""
        try:
            with self._get_odds_connection() as conn:
                # Create bestfightodds schema if it doesn't exist
                conn.execute(text('CREATE SCHEMA IF NOT EXISTS bestfightodds;'))
                
                # Create name_matching table
                conn.execute(text('''
                    CREATE TABLE IF NOT EXISTS bestfightodds.name_matching (
                        id SERIAL PRIMARY KEY,
                        ufcstats_name VARCHAR(255) NOT NULL,
                        bestfightodds_name VARCHAR(255) NOT NULL,
                        match_score INTEGER NOT NULL,
                        UNIQUE(ufcstats_name)
                    );
                    
                    CREATE INDEX IF NOT EXISTS idx_nm_ufcstats_name ON bestfightodds.name_matching(ufcstats_name);
                    CREATE INDEX IF NOT EXISTS idx_nm_bestfightodds_name ON bestfightodds.name_matching(bestfightodds_name);
                '''))
                
                conn.commit()
                print("Name matching table initialized successfully.")
        
        except Exception as e:
            print(f"Error initializing name matching table: {str(e)}")
            raise
    
    def get_ufcstats_fighter_names(self):
        """Get all unique fighter names from features.fighter_mapping"""
        try:
            result = self.mma_conn.execute(text("""
                SELECT DISTINCT fighter_name 
                FROM features.fighter_mapping
                ORDER BY fighter_name
            """))
            
            return [row[0] for row in result.fetchall()]
        except Exception as e:
            print(f"Error getting UFC Stats fighter names: {str(e)}")
            return []
    
    def get_bestfightodds_fighter_names(self):
        """Get all unique opponent names from bestfightodds.bfo table"""
        try:
            with self._get_odds_connection() as conn:
                result = conn.execute(text("""
                    SELECT DISTINCT opponent FROM bestfightodds.bfo
                    ORDER BY opponent
                """))
                
                return [row[0] for row in result.fetchall()]
        except Exception as e:
            print(f"Error getting Best Fight Odds opponent names: {str(e)}")
            return []
    
    def match_fighter_names(self, threshold=65, skip_fighters=[]):
        """
        Match fighter names between features.fighter_mapping and bestfightodds.bfo
        using fuzzy string matching.
        
        Args:
            threshold: Minimum fuzzy match score to consider a match (0-100)
            skip_fighters: List of UFC Stats fighter names to skip during matching
            
        Returns:
            Number of matches found and saved
        """
        try:
            # Get all UFC Stats fighter names
            ufcstats_names = self.get_ufcstats_fighter_names()
            print(f"Found {len(ufcstats_names)} unique UFC Stats fighter names")
            
            # Get all Best Fight Odds opponent names
            bfo_names = self.get_bestfightodds_fighter_names()
            print(f"Found {len(bfo_names)} unique Best Fight Odds opponent names")
            
            # Perform fuzzy matching
            match_results = []
            no_match_count = 0
            perfect_match_count = 0
            
            for ufcstats_name in tqdm(ufcstats_names, desc="Matching fighter names"):
                if ufcstats_name.lower() in skip_fighters:
                    continue
                
                # Find the best match
                best_matches = process.extract(ufcstats_name.lower(), 
                                              [name.lower() for name in bfo_names],
                                              limit=5,
                                              scorer=fuzz.token_sort_ratio)
                
                best_match, score = best_matches[0]
                
                if score == 100:
                    # Perfect match, just print it out and don't save to database
                    perfect_match_count += 1
                    original_case_match = [name for name in bfo_names if name.lower() == best_match][0]
                    
                elif score >= threshold:
                    # Good match above threshold, save to database
                    original_case_match = [name for name in bfo_names if name.lower() == best_match][0]
                    match_results.append({
                        'ufcstats_name': ufcstats_name,
                        'bestfightodds_name': original_case_match,
                        'match_score': score
                    })
                    print(f"Saved match for {ufcstats_name} and {original_case_match} with score {score}")
                else:
                    # No good match found, print top 5 closest matches
                    no_match_count += 1
                    original_case_matches = []
                    for match, match_score in best_matches:
                        orig_case = [name for name in bfo_names if name.lower() == match][0]
                        original_case_matches.append((orig_case, match_score))
                    
                    #matches_str = ", ".join([f"'{m[0]}' ({m[1]})" for m in original_case_matches])
                    #print(f"WARNING: No good match found for '{ufcstats_name}'. Top 5 matches: {matches_str}")
            
            print(f"Found {len(match_results)} matches above threshold {threshold}")
            print(f"Found {perfect_match_count} perfect matches (score=100)")
            print(f"No good match for {no_match_count} UFC Stats fighter names")
            
            # Save matches to the database
            if match_results:
                self.save_matches_to_database(match_results)
            
            return len(match_results)
            
        except Exception as e:
            print(f"Error matching fighter names: {str(e)}")
            return 0
    
    def save_matches_to_database(self, matches):
        """
        Save fighter name matches to the bestfightodds.name_matching table.
        
        Args:
            matches: List of dictionaries with ufcstats_name, bestfightodds_name, and match_score
            
        Returns:
            Number of matches saved
        """
        try:
            # Insert data into the odds database
            with self._get_odds_connection() as conn:
                # Clear existing matches first
                conn.execute(text("TRUNCATE TABLE bestfightodds.name_matching RESTART IDENTITY;"))
                
                # Insert each match
                inserted = 0
                for match in matches:
                    try:
                        conn.execute(
                            text("""
                                INSERT INTO bestfightodds.name_matching (ufcstats_name, bestfightodds_name, match_score)
                                VALUES (:ufcstats_name, :bestfightodds_name, :match_score)
                                ON CONFLICT (ufcstats_name) DO UPDATE SET
                                    bestfightodds_name = EXCLUDED.bestfightodds_name,
                                    match_score = EXCLUDED.match_score;
                            """),
                            {
                                'ufcstats_name': match['ufcstats_name'],
                                'bestfightodds_name': match['bestfightodds_name'],
                                'match_score': match['match_score']
                            }
                        )
                        inserted += 1
                    except Exception as e:
                        print(f"Error inserting match {match}: {str(e)}")
                        continue
                        
                conn.commit()
                print(f"Saved {inserted} name matches to the database")
                return inserted
                
        except Exception as e:
            print(f"Error saving matches to database: {str(e)}")
            return 0
    
    def get_bestfightodds_name(self, ufcstats_name):
        """
        Get the corresponding Best Fight Odds name for a UFC Stats fighter name.
        
        Args:
            ufcstats_name: UFC Stats fighter name
            
        Returns:
            The corresponding Best Fight Odds name or None if not found
        """
        try:
            with self._get_odds_connection() as conn:
                result = conn.execute(
                    text("""
                        SELECT bestfightodds_name 
                        FROM bestfightodds.name_matching 
                        WHERE ufcstats_name = :ufcstats_name
                        LIMIT 1
                    """),
                    {"ufcstats_name": ufcstats_name}
                )
                row = result.fetchone()
                return row[0] if row else None
        except Exception as e:
            print(f"Error getting Best Fight Odds name for {ufcstats_name}: {str(e)}")
            return None

    def add_manual_mappings(self, mappings):
        """
        Add manual mappings between UFC Stats names and Best Fight Odds names.
        
        Args:
            mappings: List of dictionaries with ufcstats_name, bestfightodds_name, and optional match_score
            
        Returns:
            Number of mappings added
        """
        try:
            with self._get_odds_connection() as conn:
                # Insert each mapping
                inserted = 0
                for mapping in mappings:
                    score = mapping.get('match_score', 100)  # Default to 100 if not provided
                    try:
                        conn.execute(
                            text("""
                                INSERT INTO bestfightodds.name_matching (ufcstats_name, bestfightodds_name, match_score)
                                VALUES (:ufcstats_name, :bestfightodds_name, :match_score)
                                ON CONFLICT (ufcstats_name) DO UPDATE SET
                                    bestfightodds_name = EXCLUDED.bestfightodds_name,
                                    match_score = EXCLUDED.match_score;
                            """),
                            {
                                'ufcstats_name': mapping['ufcstats_name'],
                                'bestfightodds_name': mapping['bestfightodds_name'],
                                'match_score': score
                            }
                        )
                        inserted += 1
                        print(f"Added manual mapping: '{mapping['ufcstats_name']}' → '{mapping['bestfightodds_name']}'")
                    except Exception as e:
                        print(f"Error inserting manual mapping {mapping}: {str(e)}")
                        continue
                        
                conn.commit()
                print(f"Added {inserted} manual mappings to the database")
                return inserted
                
        except Exception as e:
            print(f"Error adding manual mappings: {str(e)}")
            return 0

# Build the odds.bestfightodds.name_matching table
if __name__ == "__main__":
    import argparse
    import sys
    
    def create_db_engine(db_url, create_if_not_exists=True):
        """
        Create the SQLAlchemy database engine.
        
        Args:
            db_url: Database connection URL
            create_if_not_exists: Whether to create the database if it doesn't exist
            
        Returns:
            SQLAlchemy engine
        """
        if create_if_not_exists:
            from sqlalchemy_utils import database_exists, create_database
            
            # Create the database if it doesn't exist
            engine = create_engine(db_url)
            if not database_exists(engine.url):
                create_database(engine.url)
        else:
            engine = create_engine(db_url)
        
        return engine

    @contextmanager
    def get_db_connection(engine):
        """
        Context manager for database connections.
        
        Args:
            engine: SQLAlchemy engine
            
        Yields:
            Database connection
        """
        connection = engine.connect()
        try:
            yield connection
        finally:
            connection.close()
    
    # Set up argument parsing
    parser = argparse.ArgumentParser(description='Match fighter names between UFC Stats and Best Fight Odds')
    parser.add_argument('--threshold', type=int, default=89,
                      help='Minimum fuzzy match score to consider a match (0-100)')
    parser.add_argument('--mma-db-url', type=str, 
                      default=database_url(),
                      help='Main MMA database connection URL')
    parser.add_argument('--odds-db-url', type=str,
                      default=odds_database_url(),
                      help='Odds database connection URL')
    parser.add_argument('--no-auto-match', action='store_true',
                      help='Skip automatic matching and only add manual mappings')
    
    args = parser.parse_args()
    
    try:
        # Create database engines
        mma_engine = create_db_engine(args.mma_db_url, create_if_not_exists=False)  # Don't create the main DB
        
        # Match fighter names
        with get_db_connection(mma_engine) as mma_conn:
            # Initialize BFOUtils with both connections
            utils = BFOUtils(mma_conn, odds_db_url=args.odds_db_url)
            
            # Add manual mappings
            manual_mappings = [
                {'ufcstats_name': 'jacare souza', 'bestfightodds_name': 'Ronaldo Souza'},
                {'ufcstats_name': 'aj Fonseca', 'bestfightodds_name': 'A.J. Fonseca'},
                {'ufcstats_name': 'jc Cottrell', 'bestfightodds_name': 'J.C. Cottrell'},
                {'ufcstats_name': 'tj dillashaw', 'bestfightodds_name': 'T.J. Dillashaw'}
            ]
            utils.add_manual_mappings(manual_mappings)
            
            # Define UFC Stats fighter names to skip during matching
            skip_fighters = [
                'chris price',
                'dustin moore',
                'henrique da silva lopes',
                'john Donaldson',
                'yoshihiro Takayama',
                'sammy morgan',
                'rob macdonald',
                'rainy Martinez',
                'masakatsu okuda',
                'marques daniels',
                'kiyoshi tamura',
                'kenyon Jackson',
                'karl willis',
                'jon murphy',
                'henrique da silva lopes',
                'cory peterson'
            ]
            
            # Match fighter names if not disabled
            if not args.no_auto_match:
                print(f"Matching fighter names with threshold {args.threshold}...")
                match_count = utils.match_fighter_names(threshold=args.threshold, skip_fighters=skip_fighters)
                print(f"Total matches found and saved: {match_count}")
    
    except Exception as e:
        print(f"Error in main function: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1) 
