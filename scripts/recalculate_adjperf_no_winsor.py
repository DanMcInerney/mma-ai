#!/usr/bin/env python3
"""
Script to copy the database and recalculate adjperf features without winsorization.

This script:
1. Copies the database from 'mma-ai' to 'mma-ai-no-winsor'
2. Deletes all columns with 'adjperf' in their name
3. Recalculates adjperf features without winsorization
4. Recalculates dec_adjperf, dec_adjperf_dec_avg, and related features
"""

import sys
import subprocess
import os
from pathlib import Path
from sqlalchemy import create_engine, text
from sqlalchemy.engine import make_url
from sqlalchemy_utils import database_exists, create_database, drop_database
from contextlib import contextmanager
import pandas as pd

# Set UTF-8 encoding for Windows console
if sys.platform == 'win32':
    os.environ['PYTHONIOENCODING'] = 'utf-8'
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')
    if hasattr(sys.stderr, 'reconfigure'):
        sys.stderr.reconfigure(encoding='utf-8')

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from libs.feature_store.calculators.adj_perf_calc import AdjustedPerformanceCalculator
from libs.feature_store.calculators.time_dec_avg_calc import TimedecAvgCalculator
from libs.feature_store.calculators.avg_calc import AverageCalculator
from libs.feature_store.calculators.minimum_mad_calc import MinimumMadCalculator
from config.decay import DECAY_HALF_LIFE_YEARS
from libs.paths import database_url, no_winsor_database_url

SOURCE_DB_URL = database_url()
TARGET_DB_URL = no_winsor_database_url()

def create_db_engine(db_url):
    """Create and configure the database engine"""
    return create_engine(
        db_url,
        pool_size=5,
        max_overflow=10,
        pool_timeout=30,
        pool_recycle=14400
    )

@contextmanager
def get_db_connection(engine):
    """Context manager for database connections"""
    connection = engine.connect()
    try:
        yield connection
    finally:
        connection.close()

def find_pg_bin_path():
    """Find PostgreSQL bin directory"""
    import os
    possible_paths = [
        r"C:\Program Files\PostgreSQL",
        r"C:\Program Files (x86)\PostgreSQL",
    ]
    
    # Check common installation paths
    for base_path in possible_paths:
        if os.path.exists(base_path):
            # Look for versioned directories
            versions = [d for d in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, d))]
            for version in sorted(versions, reverse=True):  # Try newest first
                bin_path = os.path.join(base_path, version, "bin")
                pg_dump_path = os.path.join(bin_path, "pg_dump.exe")
                if os.path.exists(pg_dump_path):
                    return bin_path
    
    # Check if pg_dump is already in PATH
    import shutil
    pg_dump_path = shutil.which("pg_dump")
    if pg_dump_path:
        return os.path.dirname(pg_dump_path)
    
    return None


def pg_cli_parts(db_url: str) -> dict[str, str]:
    """Return pg_dump/pg_restore connection parts from a SQLAlchemy database URL."""
    url = make_url(db_url)
    if not url.database:
        raise ValueError(f"Database URL must include a database name: {db_url}")
    return {
        "user": url.username or "postgres",
        "password": url.password or "",
        "host": url.host or "localhost",
        "port": str(url.port or 5432),
        "database": url.database,
    }

def copy_database():
    """Copy the source database to target database using pg_dump/pg_restore"""
    print("=" * 80)
    print("STEP 1: Copying database")
    print("=" * 80)
    
    # Drop target database if it exists
    if database_exists(TARGET_DB_URL):
        print(f"Dropping existing database: {TARGET_DB_URL}")
        drop_database(TARGET_DB_URL)
    
    # Create target database
    print(f"Creating target database: {TARGET_DB_URL}")
    create_database(TARGET_DB_URL)
    
    # Find PostgreSQL bin directory
    print("Looking for PostgreSQL installation...")
    pg_bin = find_pg_bin_path()
    if not pg_bin:
        raise RuntimeError(
            "Could not find pg_dump/pg_restore. Please either:\n"
            "1. Add PostgreSQL bin directory to PATH, or\n"
            "2. Install PostgreSQL and ensure pg_dump.exe is accessible"
        )
    
    print(f"Found PostgreSQL at: {pg_bin}")
    
    # Use pg_dump and pg_restore to copy the database
    print("Copying database structure and data...")
    
    source = pg_cli_parts(SOURCE_DB_URL)
    target = pg_cli_parts(TARGET_DB_URL)
    if source["password"]:
        os.environ['PGPASSWORD'] = source["password"]
    
    # Build paths to pg_dump and pg_restore
    pg_dump_exe = os.path.join(pg_bin, "pg_dump.exe")
    pg_restore_exe = os.path.join(pg_bin, "pg_restore.exe")
    
    # Run pg_dump to dump the source database
    dump_cmd = [
        pg_dump_exe,
        '-h', source["host"],
        '-p', source["port"],
        '-U', source["user"],
        '-d', source["database"],
        '-F', 'c',  # Custom format
        '-f', 'temp_dump.dump'
    ]
    
    print(f"Running: {' '.join(dump_cmd)}")
    result = subprocess.run(dump_cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Error dumping database: {result.stderr}")
        raise RuntimeError(f"pg_dump failed: {result.stderr}")
    
    # Run pg_restore to restore to target database
    if target["password"]:
        os.environ['PGPASSWORD'] = target["password"]
    restore_cmd = [
        pg_restore_exe,
        '-h', target["host"],
        '-p', target["port"],
        '-U', target["user"],
        '-d', target["database"],
        '--no-owner',
        '--no-acl',
        'temp_dump.dump'
    ]
    
    print(f"Running: {' '.join(restore_cmd)}")
    result = subprocess.run(restore_cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Error restoring database: {result.stderr}")
        raise RuntimeError(f"pg_restore failed: {result.stderr}")
    
    # Clean up dump file
    if Path('temp_dump.dump').exists():
        Path('temp_dump.dump').unlink()
    
    print("OK Database copied successfully")
    print()

def find_adjperf_columns(conn):
    """Find all columns with 'adjperf' or 'dec_adjperf' in their name"""
    print("=" * 80)
    print("STEP 2: Finding adjperf columns")
    print("=" * 80)
    
    query = text("""
        SELECT table_schema, table_name, column_name
        FROM information_schema.columns
        WHERE table_schema = 'features'
        AND (column_name LIKE '%adjperf%' OR column_name LIKE '%dec_adjperf%')
        ORDER BY table_name, column_name
    """)
    
    df = pd.read_sql(query, conn)
    
    if df.empty:
        print("No adjperf columns found!")
        return []
    
    print(f"Found {len(df)} columns with 'adjperf' or 'dec_adjperf' in their name")
    print(f"Across {df['table_name'].nunique()} tables")
    
    # Show sample of columns
    if len(df) <= 20:
        print("\nColumns to be dropped:")
        for _, row in df.iterrows():
            print(f"  {row['table_name']}.{row['column_name']}")
    else:
        print(f"\nSample columns (showing first 10):")
        for _, row in df.head(10).iterrows():
            print(f"  {row['table_name']}.{row['column_name']}")
        print(f"  ... and {len(df) - 10} more")
    
    return df

def drop_adjperf_columns(conn, adjperf_df):
    """Drop all columns with 'adjperf' in their name"""
    print("=" * 80)
    print("STEP 3: Dropping adjperf columns")
    print("=" * 80)
    
    if adjperf_df.empty:
        print("No columns to drop")
        return
    
    # Group by table
    tables = adjperf_df.groupby('table_name')
    
    total_dropped = 0
    for table_name, group in tables:
        columns = group['column_name'].tolist()
        print(f"\nDropping {len(columns)} columns from {table_name}...")
        
        for col in columns:
            try:
                drop_query = text(f"""
                    ALTER TABLE features."{table_name}"
                    DROP COLUMN IF EXISTS "{col}"
                """)
                conn.execute(drop_query)
                total_dropped += 1
                print(f"  Dropped: {col}")
            except Exception as e:
                print(f"  Error dropping {col}: {e}")
        
        conn.commit()
    
    print(f"\nOK Dropped {total_dropped} columns total")
    print()

class NonWinsorizedAdjustedPerformanceCalculator(AdjustedPerformanceCalculator):
    """Modified AdjustedPerformanceCalculator without winsorization"""
    
    def _generate_adjperf_expression(self, col: str, table_name: str) -> str:
        """Generate adjusted performance expression WITHOUT winsorization"""
        # Only process adjperf target columns
        if not self._is_adjperf_target(col):
            return "0"
        
        # Use the same observed value expression as parent
        obs_expr = self._compute_observed_canonical_value(col, table_name, alias='t')
        
        # Shrinkage weights (matching parent class structure)
        w_mean_expr = f"COALESCE(oh.n_fights, 0) / (COALESCE(oh.n_fights, 0) + {self.K_mean:.1f})"
        w_mad_expr = f"COALESCE(oh.n_fights, 0) / (COALESCE(oh.n_fights, 0) + {self.K_mad:.1f})"
        
        # Shrunk mean: w_mean * opp_mean + (1 - w_mean) * wc_mean
        mu_shrunk_expr = f"""
        ({w_mean_expr}) * COALESCE(oh.{col}_opp_mean_pers, 0) + 
        (1 - ({w_mean_expr})) * COALESCE(wp.{col}_wc_mean, 0)
        """
        
        # Shrunk MAD: max( w_mad * opp_mad + (1 - w_mad) * wc_mad , mad_floor )
        mad_shrunk_expr = f"""
        GREATEST(
            ({w_mad_expr}) * COALESCE(oh.{col}_opp_mad_pers, 0) + 
            (1 - ({w_mad_expr})) * COALESCE(wp.{col}_wc_mad, 0),
            COALESCE(wp.{col}_mad_floor, 0.001)
        )
        """
        
        # Final adjusted performance WITHOUT clipping (removed GREATEST/LEAST)
        adjperf_expr = f"""
        CASE 
            WHEN ({mad_shrunk_expr}) = 0 THEN 0
            ELSE (({obs_expr}) - ({mu_shrunk_expr})) / ({mad_shrunk_expr})
        END
        """
        
        return adjperf_expr

def recalculate_minimum_mad(conn):
    """Recalculate minimum MAD tables (required before adjperf calculation)"""
    print("=" * 80)
    print("STEP 4: Recalculating minimum MAD (required for adjperf)")
    print("=" * 80)
    
    # Calculate minimum mad (decay=False) - this uses the 10th percentile now
    print("\nCalculating minimum mad (decay=False)...")
    exclude_patterns = set(['_avg'])  # we only need mad for denominator of adjperf so no avg
    calc = MinimumMadCalculator(conn, decay=False, exclude_patterns=exclude_patterns)
    calc.run()
    print("OK minimum mad calculation complete")
    print()

def recalculate_adjperf_features(conn):
    """Recalculate adjperf features without winsorization"""
    print("=" * 80)
    print("STEP 5: Recalculating adjperf features (no winsorization)")
    print("=" * 80)
    
    # Calculate adjperf (decay=False)
    print("\nCalculating adjperf (decay=False)...")
    exclude_patterns = set(['_adjperf'])
    calc = NonWinsorizedAdjustedPerformanceCalculator(
        conn, 
        decay=False, 
        exclude_patterns=exclude_patterns
    )
    calc.run()
    print("OK adjperf calculation complete")
    
    # Calculate dec_adjperf (decay=True)
    print("\nCalculating dec_adjperf (decay=True)...")
    exclude_patterns = set(['_adjperf'])
    calc = NonWinsorizedAdjustedPerformanceCalculator(
        conn, 
        decay=True, 
        exclude_patterns=exclude_patterns
    )
    calc.run()
    print("OK dec_adjperf calculation complete")
    print()

def recalculate_dec_avg_features(conn):
    """Recalculate dec_avg features for adjperf columns"""
    print("=" * 80)
    print("STEP 6: Recalculating dec_avg for adjperf features")
    print("=" * 80)
    
    decay_rate_years = DECAY_HALF_LIFE_YEARS
    
    # Calculate dec_avg for _adjperf columns (this creates _adjperf_dec_avg)
    print("\nCalculating dec_avg for _adjperf columns...")
    include_patterns = set(['_adjperf'])
    exclude_patterns = set(['_dec_avg', '_mad', '_dec_adjperf'])  # Exclude existing _dec_avg, _mad, and _dec_adjperf
    calc = TimedecAvgCalculator(
        conn, 
        decay_rate_years, 
        include_patterns=include_patterns,
        exclude_patterns=exclude_patterns
    )
    calc.run()
    print("OK _adjperf_dec_avg calculation complete")
    
    # Calculate dec_avg for _dec_adjperf columns (this creates _dec_adjperf_dec_avg)
    print("\nCalculating dec_avg for _dec_adjperf columns...")
    include_patterns = set(['_dec_adjperf'])
    exclude_patterns = set(['_dec_avg', '_mad'])  # Exclude existing _dec_avg and _mad
    calc = TimedecAvgCalculator(
        conn, 
        decay_rate_years, 
        include_patterns=include_patterns,
        exclude_patterns=exclude_patterns
    )
    calc.run()
    print("OK _dec_adjperf_dec_avg calculation complete")
    print()

def recalculate_avg_features(conn):
    """Recalculate avg features for adjperf columns"""
    print("=" * 80)
    print("STEP 7: Recalculating avg for adjperf features")
    print("=" * 80)
    
    # Calculate avg for _dec_adjperf columns
    print("\nCalculating avg for _dec_adjperf columns...")
    include_patterns = set(['_dec_adjperf'])
    exclude_patterns = set(['_avg', '_mad'])  # Exclude existing _avg and _mad
    calc = AverageCalculator(
        conn,
        include_patterns=include_patterns,
        exclude_patterns=exclude_patterns
    )
    calc.run()
    print("OK dec_adjperf_dec_avg calculation complete")
    print()

def main():
    """Main execution function"""
    print("=" * 80)
    print("RECALCULATE ADJPERF WITHOUT WINSORIZATION")
    print("=" * 80)
    print(f"Source DB: {SOURCE_DB_URL}")
    print(f"Target DB: {TARGET_DB_URL}")
    print()
    
    try:
        # Step 1: Copy database
        copy_database()
        
        # Step 2-6: Work with target database
        target_engine = create_db_engine(TARGET_DB_URL)
        
        with get_db_connection(target_engine) as conn:
            # Step 2: Find adjperf columns
            adjperf_df = find_adjperf_columns(conn)
            
            # Step 3: Drop adjperf columns
            drop_adjperf_columns(conn, adjperf_df)
            
            # Step 4: Recalculate minimum MAD (required before adjperf)
            recalculate_minimum_mad(conn)
            
            # Step 5: Recalculate adjperf features (no winsorization)
            recalculate_adjperf_features(conn)
            
            # Step 6: Recalculate dec_avg for adjperf
            recalculate_dec_avg_features(conn)
            
            # Step 7: Recalculate avg for adjperf
            recalculate_avg_features(conn)
        
        print("=" * 80)
        print("SUCCESS!")
        print("=" * 80)
        print(f"Database '{TARGET_DB_URL}' has been created with adjperf features")
        print("recalculated without winsorization.")
        print()
        print("You can now use this database for training/testing.")
        
    except Exception as e:
        print("\n" + "=" * 80)
        print("ERROR!")
        print("=" * 80)
        print(f"An error occurred: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == '__main__':
    main()

