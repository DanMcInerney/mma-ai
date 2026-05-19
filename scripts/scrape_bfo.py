#!/usr/bin/env python3
"""
Script to scrape betting odds data from bestfightodds.com and store in the odds database.
"""

import argparse
import sys
import os

# Add the parent directory to sys.path to ensure imports work correctly
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sqlalchemy import create_engine
from sqlalchemy_utils import database_exists, create_database
from contextlib import contextmanager
from libs.bfo_scraper import BFOScraper
from libs.paths import database_url, odds_database_url

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

def main():
    parser = argparse.ArgumentParser(description='Scrape fighter odds data from bestfightodds.com')
    parser.add_argument('--fighter', type=str, help='Specific fighter name to scrape (optional)')
    parser.add_argument('--mma-db-url', type=str, 
                      default=database_url(),
                      help='Main MMA database connection URL')
    parser.add_argument('--odds-db-url', type=str,
                      default=odds_database_url(),
                      help='Odds database connection URL')
    args = parser.parse_args()
    
    # Create database engines
    mma_engine = create_db_engine(args.mma_db_url, create_if_not_exists=False)  # Don't create the main DB
    
    # Scrape the odds data
    with get_db_connection(mma_engine) as mma_conn:
        # Initialize scraper with both connections
        scraper = BFOScraper(mma_conn, odds_db_url=args.odds_db_url)
        
        if args.fighter:
            print(f"Scraping odds data for {args.fighter}...")
            records = scraper.scrape_fighter(args.fighter)
            print(f"Saved {records} odds records for {args.fighter}")
        else:
            print("Scraping odds data for all fighters in the database...")
            records = scraper.scrape_all_fighters()
            print(f"Total odds records saved: {records}")

if __name__ == "__main__":
    main() 
