#!/usr/bin/env python
"""
Script to calculate time-decayed standard deviations for UFC fight statistics.
This script demonstrates the usage of the TimedecStdDevCalculator with the optimized architecture.

Usage:
    python calculate_time_dec_sdev.py [--table_pattern PATTERN] [--sample] [--db_uri DB_URI]

Options:
    --table_pattern PATTERN   Filter tables by pattern (e.g., 'sig_str' or 'td')
    --sample                  Run in sample mode (process only 5%% of data for faster testing)
    --db_uri DB_URI           Database URI (default: DATABASE_URL from .env or repo default)
"""

import argparse
import logging
import sys
import time
from sqlalchemy import create_engine
from libs.feature_store.calculators.time_dec_sdev_calc import TimedecStdDevCalculator
from libs.paths import database_url

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    stream=sys.stdout
)
logger = logging.getLogger(__name__)

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Calculate time-decayed standard deviations for UFC fight statistics')
    parser.add_argument('--table_pattern', type=str, default='', help='Filter tables by pattern (e.g., "sig_str" or "td")')
    parser.add_argument('--sample', action='store_true', help='Run in sample mode (process only 5%% of data)')
    parser.add_argument('--db_uri', type=str, default=database_url(),
                        help='Database URI')
    args = parser.parse_args()
    
    # Connect to database
    logger.info(f"Connecting to database: {args.db_uri}")
    engine = create_engine(args.db_uri)
    conn = engine.connect()
    
    try:
        # Create calculator with 1.5 year half-life
        logger.info("Creating TimedecStdDevCalculator with 1.5 year half-life")
        calculator = TimedecStdDevCalculator(
            conn,
            decay_rate_years=1.5,  # 1.5 year half-life
            include_patterns=set(),  # Process all columns by default
            exclude_patterns=set(['opp']),  # Exclude opponent columns
            sample_mode=args.sample  # Use sample mode if specified
        )
        
        # Start timer
        start_time = time.time()
        
        # Run calculator with table pattern filtering
        logger.info(f"Running calculator with table pattern: '{args.table_pattern}'")
        results = calculator.run(table_pattern=args.table_pattern)
        
        # Calculate elapsed time
        elapsed_time = time.time() - start_time
        
        # Count successful tables
        successful_tables = sum(1 for df in results.values() if not df.empty)
        total_tables = len(results)
        
        # Print summary
        print("\n=== Time-Decayed Standard Deviation Calculation Summary ===")
        print(f"Total tables processed: {total_tables}")
        print(f"Successfully processed: {successful_tables}")
        print(f"Failed tables: {total_tables - successful_tables}")
        print(f"Total time elapsed: {elapsed_time:.2f} seconds")
        print(f"Average time per table: {elapsed_time / total_tables:.2f} seconds")
        print("==========================================================\n")
        
        logger.info(f"Calculation completed in {elapsed_time:.2f} seconds")
        
    except Exception as e:
        logger.error(f"Error calculating time-decayed standard deviations: {str(e)}")
        raise
    finally:
        # Close database connection
        conn.close()
        logger.info("Database connection closed")

if __name__ == "__main__":
    main() 
