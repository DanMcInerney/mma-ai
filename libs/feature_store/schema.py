import pandas as pd
from sqlalchemy import text
from libs.feature_store.feature_utils import FeatureUtils
from typing import Set

def initialize_schema(conn):
    print("Starting schema initialization...")
    try:
        print("Creating schema and tables...")
        # Create schema
        conn.execute(text('CREATE SCHEMA IF NOT EXISTS features;'))
        conn.commit()

        # Debug: Check if schema exists after
        result = conn.execute(text("SELECT schema_name FROM information_schema.schemata WHERE schema_name = 'features'"))
        print(f"Schema exists after creation: {bool(result.scalar())}")

        # Create core mapping tables with enhanced indexing
        print("Creating mapping tables...")

        # Fighter mapping table
        conn.execute(text('''
            CREATE TABLE IF NOT EXISTS features.fighter_mapping (
                fighter_id INTEGER GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
                fighter_url VARCHAR(255) NOT NULL UNIQUE,
                fighter_name VARCHAR(255) NOT NULL,
                fighter_nickname VARCHAR(255),
                fighter_stance VARCHAR(20),
                fighter_weight VARCHAR(50),
                fighter_dob DATE,
                fighter_height INTEGER,
                fighter_reach INTEGER
            );
            
            CREATE INDEX IF NOT EXISTS idx_fighter_mapping_url ON features.fighter_mapping(fighter_url);
        '''))
        conn.commit()
        
        # Create event mapping table
        conn.execute(text('''
            CREATE TABLE IF NOT EXISTS features.event_mapping (
                event_id INTEGER GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
                event_url VARCHAR(255) NOT NULL UNIQUE,
                event_date DATE NOT NULL,
                event_location VARCHAR(255)
            );
            
            CREATE INDEX IF NOT EXISTS idx_event_mapping_date ON features.event_mapping(event_date);
        '''))
        conn.commit()
        
        # Create fight mapping table
        conn.execute(text('''
            CREATE TABLE IF NOT EXISTS features.fight_mapping (
                fight_id INTEGER GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
                event_id INTEGER NOT NULL,
                fighter1_id INTEGER NOT NULL,
                fighter2_id INTEGER NOT NULL,
                weightclass VARCHAR(50),
                weightclass_encoded INTEGER,
                method VARCHAR(50),
                details VARCHAR(255),
                end_round INTEGER,
                end_time INTEGER,
                time_format VARCHAR(20),
                result INTEGER,
                CONSTRAINT fk_event FOREIGN KEY (event_id) REFERENCES features.event_mapping(event_id),
                CONSTRAINT fk_fighter1 FOREIGN KEY (fighter1_id) REFERENCES features.fighter_mapping(fighter_id),
                CONSTRAINT fk_fighter2 FOREIGN KEY (fighter2_id) REFERENCES features.fighter_mapping(fighter_id),
                CONSTRAINT uq_fight UNIQUE(event_id, fighter1_id, fighter2_id)
            );
            
            CREATE INDEX IF NOT EXISTS idx_fight_mapping_event ON features.fight_mapping(event_id);
            CREATE INDEX IF NOT EXISTS idx_fight_mapping_fighters ON features.fight_mapping(fighter1_id, fighter2_id);
            CREATE INDEX IF NOT EXISTS idx_fight_mapping_weightclass_event ON features.fight_mapping(weightclass, event_id);
        '''))
        conn.commit()
        
        # Add weightclass_encoded column if it doesn't exist (migration for existing databases)
        try:
            conn.execute(text('''
                DO $$ 
                BEGIN
                    IF NOT EXISTS (
                        SELECT 1 FROM information_schema.columns 
                        WHERE table_schema = 'features' 
                        AND table_name = 'fight_mapping' 
                        AND column_name = 'weightclass_encoded'
                    ) THEN
                        ALTER TABLE features.fight_mapping ADD COLUMN weightclass_encoded INTEGER;
                    END IF;
                END $$;
            '''))
            conn.commit()
            print("Ensured weightclass_encoded column exists in fight_mapping")
        except Exception as e:
            print(f"Note: Could not add weightclass_encoded column (may already exist): {e}")
            conn.rollback()

        # Create core tables
        conn.execute(text('''
            CREATE TABLE IF NOT EXISTS features.fight_stats_core (
                fight_id INTEGER NOT NULL,
                fighter_id INTEGER NOT NULL,
                event_id INTEGER NOT NULL,
                -- Round stats
                kd_rd1 INTEGER, kd_rd2 INTEGER, kd_rd3 INTEGER, kd_rd4 INTEGER, kd_rd5 INTEGER,
                sig_str_land_rd1 INTEGER, sig_str_att_rd1 INTEGER,
                sig_str_land_rd2 INTEGER, sig_str_att_rd2 INTEGER,
                sig_str_land_rd3 INTEGER, sig_str_att_rd3 INTEGER,
                sig_str_land_rd4 INTEGER, sig_str_att_rd4 INTEGER,
                sig_str_land_rd5 INTEGER, sig_str_att_rd5 INTEGER,
                strikes_land_rd1 INTEGER, strikes_att_rd1 INTEGER,
                strikes_land_rd2 INTEGER, strikes_att_rd2 INTEGER,
                strikes_land_rd3 INTEGER, strikes_att_rd3 INTEGER,
                strikes_land_rd4 INTEGER, strikes_att_rd4 INTEGER,
                strikes_land_rd5 INTEGER, strikes_att_rd5 INTEGER,
                td_land_rd1 INTEGER, td_att_rd1 INTEGER,
                td_land_rd2 INTEGER, td_att_rd2 INTEGER,
                td_land_rd3 INTEGER, td_att_rd3 INTEGER,
                td_land_rd4 INTEGER, td_att_rd4 INTEGER,
                td_land_rd5 INTEGER, td_att_rd5 INTEGER,
                sub_att_rd1 INTEGER, sub_att_rd2 INTEGER, sub_att_rd3 INTEGER, sub_att_rd4 INTEGER, sub_att_rd5 INTEGER,
                rev_rd1 INTEGER, rev_rd2 INTEGER, rev_rd3 INTEGER, rev_rd4 INTEGER, rev_rd5 INTEGER,
                ctrl_rd1 INTEGER, ctrl_rd2 INTEGER, ctrl_rd3 INTEGER, ctrl_rd4 INTEGER, ctrl_rd5 INTEGER,
                -- Strike locations
                head_land_rd1 INTEGER, head_att_rd1 INTEGER,
                head_land_rd2 INTEGER, head_att_rd2 INTEGER,
                head_land_rd3 INTEGER, head_att_rd3 INTEGER,
                head_land_rd4 INTEGER, head_att_rd4 INTEGER,
                head_land_rd5 INTEGER, head_att_rd5 INTEGER,
                body_land_rd1 INTEGER, body_att_rd1 INTEGER,
                body_land_rd2 INTEGER, body_att_rd2 INTEGER,
                body_land_rd3 INTEGER, body_att_rd3 INTEGER,
                body_land_rd4 INTEGER, body_att_rd4 INTEGER,
                body_land_rd5 INTEGER, body_att_rd5 INTEGER,
                leg_land_rd1 INTEGER, leg_att_rd1 INTEGER,
                leg_land_rd2 INTEGER, leg_att_rd2 INTEGER,
                leg_land_rd3 INTEGER, leg_att_rd3 INTEGER,
                leg_land_rd4 INTEGER, leg_att_rd4 INTEGER,
                leg_land_rd5 INTEGER, leg_att_rd5 INTEGER,
                distance_land_rd1 INTEGER, distance_att_rd1 INTEGER,
                distance_land_rd2 INTEGER, distance_att_rd2 INTEGER,
                distance_land_rd3 INTEGER, distance_att_rd3 INTEGER,
                distance_land_rd4 INTEGER, distance_att_rd4 INTEGER,
                distance_land_rd5 INTEGER, distance_att_rd5 INTEGER,
                clinch_land_rd1 INTEGER, clinch_att_rd1 INTEGER,
                clinch_land_rd2 INTEGER, clinch_att_rd2 INTEGER,
                clinch_land_rd3 INTEGER, clinch_att_rd3 INTEGER,
                clinch_land_rd4 INTEGER, clinch_att_rd4 INTEGER,
                clinch_land_rd5 INTEGER, clinch_att_rd5 INTEGER,
                ground_land_rd1 INTEGER, ground_att_rd1 INTEGER,
                ground_land_rd2 INTEGER, ground_att_rd2 INTEGER,
                ground_land_rd3 INTEGER, ground_att_rd3 INTEGER,
                ground_land_rd4 INTEGER, ground_att_rd4 INTEGER,
                ground_land_rd5 INTEGER, ground_att_rd5 INTEGER,
                PRIMARY KEY (fight_id, fighter_id),
                CONSTRAINT fk_fight FOREIGN KEY (fight_id) REFERENCES features.fight_mapping(fight_id),
                CONSTRAINT fk_fighter FOREIGN KEY (fighter_id) REFERENCES features.fighter_mapping(fighter_id),
                CONSTRAINT fk_event FOREIGN KEY (event_id) REFERENCES features.event_mapping(event_id)
            );
            
            CREATE INDEX IF NOT EXISTS idx_fight_stats_fight ON features.fight_stats_core(fight_id);
            CREATE INDEX IF NOT EXISTS idx_fight_stats_fighter ON features.fight_stats_core(fighter_id);
            CREATE INDEX IF NOT EXISTS idx_fight_stats_event ON features.fight_stats_core(event_id);
        '''))
        conn.commit()

        # Create feature-engineered tables
        conn.execute(text('''
            CREATE TABLE IF NOT EXISTS features.fight_stats_fe (
                fight_id INTEGER REFERENCES features.fight_mapping(fight_id),
                fighter_id INTEGER REFERENCES features.fighter_mapping(fighter_id),
                event_id INTEGER REFERENCES features.event_mapping(event_id),
                -- Round stats
                kd_rd1 INTEGER, kd_rd2 INTEGER, kd_rd3 INTEGER, kd_rd4 INTEGER, kd_rd5 INTEGER,
                sig_str_land_rd1 INTEGER, sig_str_att_rd1 INTEGER,
                sig_str_land_rd2 INTEGER, sig_str_att_rd2 INTEGER,
                sig_str_land_rd3 INTEGER, sig_str_att_rd3 INTEGER,
                sig_str_land_rd4 INTEGER, sig_str_att_rd4 INTEGER,
                sig_str_land_rd5 INTEGER, sig_str_att_rd5 INTEGER,
                strikes_land_rd1 INTEGER, strikes_att_rd1 INTEGER,
                strikes_land_rd2 INTEGER, strikes_att_rd2 INTEGER,
                strikes_land_rd3 INTEGER, strikes_att_rd3 INTEGER,
                strikes_land_rd4 INTEGER, strikes_att_rd4 INTEGER,
                strikes_land_rd5 INTEGER, strikes_att_rd5 INTEGER,
                td_land_rd1 INTEGER, td_att_rd1 INTEGER,
                td_land_rd2 INTEGER, td_att_rd2 INTEGER,
                td_land_rd3 INTEGER, td_att_rd3 INTEGER,
                td_land_rd4 INTEGER, td_att_rd4 INTEGER,
                td_land_rd5 INTEGER, td_att_rd5 INTEGER,
                sub_att_rd1 INTEGER, sub_att_rd2 INTEGER, sub_att_rd3 INTEGER, sub_att_rd4 INTEGER, sub_att_rd5 INTEGER,
                rev_rd1 INTEGER, rev_rd2 INTEGER, rev_rd3 INTEGER, rev_rd4 INTEGER, rev_rd5 INTEGER,
                ctrl_rd1 INTEGER, ctrl_rd2 INTEGER, ctrl_rd3 INTEGER, ctrl_rd4 INTEGER, ctrl_rd5 INTEGER,
                -- Strike locations
                head_land_rd1 INTEGER, head_att_rd1 INTEGER,
                head_land_rd2 INTEGER, head_att_rd2 INTEGER,
                head_land_rd3 INTEGER, head_att_rd3 INTEGER,
                head_land_rd4 INTEGER, head_att_rd4 INTEGER,
                head_land_rd5 INTEGER, head_att_rd5 INTEGER,
                body_land_rd1 INTEGER, body_att_rd1 INTEGER,
                body_land_rd2 INTEGER, body_att_rd2 INTEGER,
                body_land_rd3 INTEGER, body_att_rd3 INTEGER,
                body_land_rd4 INTEGER, body_att_rd4 INTEGER,
                body_land_rd5 INTEGER, body_att_rd5 INTEGER,
                leg_land_rd1 INTEGER, leg_att_rd1 INTEGER,
                leg_land_rd2 INTEGER, leg_att_rd2 INTEGER,
                leg_land_rd3 INTEGER, leg_att_rd3 INTEGER,
                leg_land_rd4 INTEGER, leg_att_rd4 INTEGER,
                leg_land_rd5 INTEGER, leg_att_rd5 INTEGER,
                distance_land_rd1 INTEGER, distance_att_rd1 INTEGER,
                distance_land_rd2 INTEGER, distance_att_rd2 INTEGER,
                distance_land_rd3 INTEGER, distance_att_rd3 INTEGER,
                distance_land_rd4 INTEGER, distance_att_rd4 INTEGER,
                distance_land_rd5 INTEGER, distance_att_rd5 INTEGER,
                clinch_land_rd1 INTEGER, clinch_att_rd1 INTEGER,
                clinch_land_rd2 INTEGER, clinch_att_rd2 INTEGER,
                clinch_land_rd3 INTEGER, clinch_att_rd3 INTEGER,
                clinch_land_rd4 INTEGER, clinch_att_rd4 INTEGER,
                clinch_land_rd5 INTEGER, clinch_att_rd5 INTEGER,
                ground_land_rd1 INTEGER, ground_att_rd1 INTEGER,
                ground_land_rd2 INTEGER, ground_att_rd2 INTEGER,
                ground_land_rd3 INTEGER, ground_att_rd3 INTEGER,
                ground_land_rd4 INTEGER, ground_att_rd4 INTEGER,
                ground_land_rd5 INTEGER, ground_att_rd5 INTEGER,
                PRIMARY KEY (fight_id, fighter_id)
            );
            
            CREATE INDEX IF NOT EXISTS idx_fight_stats_fight ON features.fight_stats_fe(fight_id);
            CREATE INDEX IF NOT EXISTS idx_fight_stats_fighter ON features.fight_stats_fe(fighter_id);
            CREATE INDEX IF NOT EXISTS idx_fight_stats_event ON features.fight_stats_fe(event_id);
        '''))
        conn.commit()

        # Create derived tables
        conn.execute(text('''
            CREATE TABLE IF NOT EXISTS features.fight_stats_derived (
                fight_id INTEGER REFERENCES features.fight_mapping(fight_id),
                fighter_id INTEGER REFERENCES features.fighter_mapping(fighter_id),
                event_id INTEGER REFERENCES features.event_mapping(event_id),
                -- Round 1 stats
                kd_rd1 INTEGER,
                sig_str_land_rd1 INTEGER, 
                sig_str_att_rd1 INTEGER,
                strikes_land_rd1 INTEGER, 
                strikes_att_rd1 INTEGER,
                td_land_rd1 INTEGER, 
                td_att_rd1 INTEGER,
                sub_att_rd1 INTEGER,
                rev_rd1 INTEGER,
                ctrl_rd1 INTEGER,
                -- Strike locations round 1
                head_land_rd1 INTEGER, 
                head_att_rd1 INTEGER,
                body_land_rd1 INTEGER, 
                body_att_rd1 INTEGER,
                leg_land_rd1 INTEGER, 
                leg_att_rd1 INTEGER,
                distance_land_rd1 INTEGER, 
                distance_att_rd1 INTEGER,
                clinch_land_rd1 INTEGER, 
                clinch_att_rd1 INTEGER,
                ground_land_rd1 INTEGER, 
                ground_att_rd1 INTEGER,
                win_rd1 INTEGER,
                -- Derived features
                time_sec INTEGER,
                time_sec_rd1 INTEGER,
                ko INTEGER,
                ko_rd1 INTEGER,
                sub_land INTEGER,
                sub_land_rd1 INTEGER,
                decision INTEGER,
                -- Total stats (without round breakdown)
                kd INTEGER,
                sig_str_land INTEGER,
                sig_str_att INTEGER,
                strikes_land INTEGER,
                strikes_att INTEGER,
                td_land INTEGER,
                td_att INTEGER,
                sub_att INTEGER,
                rev INTEGER,
                ctrl INTEGER,
                head_land INTEGER,
                head_att INTEGER,
                body_land INTEGER,
                body_att INTEGER,
                leg_land INTEGER,
                leg_att INTEGER,
                distance_land INTEGER,
                distance_att INTEGER,
                clinch_land INTEGER,
                clinch_att INTEGER,
                ground_land INTEGER,
                ground_att INTEGER,
                win INTEGER,
                -- Static derived features
                age FLOAT,
                ufcage FLOAT,
                days_since_last_fight INTEGER,
                reach INTEGER,
                ape FLOAT,
                PRIMARY KEY (fight_id, fighter_id)
            );
            
            CREATE INDEX IF NOT EXISTS idx_fight_stats_derived_fight ON features.fight_stats_derived(fight_id);
            CREATE INDEX IF NOT EXISTS idx_fight_stats_derived_fighter ON features.fight_stats_derived(fighter_id);
            CREATE INDEX IF NOT EXISTS idx_fight_stats_derived_event ON features.fight_stats_derived(event_id);
            CREATE INDEX IF NOT EXISTS idx_fight_stats_derived_fight_fighter ON features.fight_stats_derived(fight_id, fighter_id);
        '''))
        conn.commit()

        # Debug: List all tables
        result = conn.execute(text("""
            SELECT table_schema, table_name 
            FROM information_schema.tables 
            WHERE table_schema = 'features';
        """))
        tables = result.fetchall()
        print("\nCreated tables:")
        for table in tables:
            print(f"- {table[0]}.{table[1]}")
            
        # Debug: Show fighter_mapping structure
        result = conn.execute(text("""
            SELECT column_name, data_type, is_nullable
            FROM information_schema.columns
            WHERE table_schema = 'features' 
            AND table_name = 'fighter_mapping'
            ORDER BY ordinal_position;
        """))
        columns = result.fetchall()
        print("\nfighter_mapping structure:")
        for col in columns:
            print(f"- {col[0]}: {col[1]} (nullable: {col[2]})")

    except Exception as e:
        print(f"\nError during schema initialization: {str(e)}")
        raise
        
    print("Schema initialization complete!")

def create_feature_specific_tables(conn):
    print("Creating feature-specific tables...")
    
    # Get column information from fight_stats_derived
    result = conn.execute(text("SELECT * FROM features.fight_stats_derived LIMIT 0"))
    columns = [(col, str(result.cursor.description[idx][1])) 
            for idx, col in enumerate(result.keys())]
    
    PG_TYPE_MAP = {
        '23': 'integer',      # int4
        '701': 'double precision',  # float8
    }

    # Group columns by feature type
    feature_groups = {}
    for col, dtype in columns:
        dtype = PG_TYPE_MAP.get(dtype, 'unknown')
        # Skip ID columns and non-feature columns
        if any(s in col for s in ['_id']):
            continue
            
        # Extract base feature name (e.g., 'strikes' from 'strikes_land')
        if 'days' in col:
            base_feature = 'days_since_last_fight'
        elif 'sig_str' in col:
            base_feature = 'sig_str'
        elif 'time_sec' in col:
            base_feature = 'time_sec'
        else:
            base_feature = col.split('_')[0]
        
        if base_feature not in feature_groups:
            feature_groups[base_feature] = []
        
        skip_static_rd1 = ['age', 'days_since_last_fight', 'reach', 'ape', 'ufcage']
        if base_feature not in skip_static_rd1:
            base_feature_rd1 = f"{base_feature}_rd1"
            if base_feature_rd1 not in feature_groups and base_feature not in skip_static_rd1:
                feature_groups[base_feature_rd1] = []
        
        if 'rd1' in col:
            feature_groups[base_feature_rd1].append((col, dtype))
        else:
            feature_groups[base_feature].append((col, dtype))


    # Create tables for each feature group
    for feature, columns in feature_groups.items():
        # Skip feature groups with no columns (e.g., decision_rd1 which was removed)
        if not columns:
            print(f"Skipping {feature} - no columns found")
            continue
            
        # Drop existing table if it exists
        conn.execute(text(f"DROP TABLE IF EXISTS features.{feature} CASCADE;"))
        conn.commit()

        # Create table
        create_table_sql = f"""
            CREATE TABLE IF NOT EXISTS features.{feature} (
                fight_id INTEGER NOT NULL,
                fighter_id INTEGER NOT NULL,
                event_id INTEGER NOT NULL,
                {', '.join(f'{col} {dtype}' for col, dtype in columns)},
                PRIMARY KEY (fight_id, fighter_id),
                CONSTRAINT fk_fight FOREIGN KEY (fight_id) REFERENCES features.fight_mapping(fight_id),
                CONSTRAINT fk_fighter FOREIGN KEY (fighter_id) REFERENCES features.fighter_mapping(fighter_id),
                CONSTRAINT fk_event FOREIGN KEY (event_id) REFERENCES features.event_mapping(event_id)
            );
            
            CREATE INDEX IF NOT EXISTS idx_{feature}_fight ON features.{feature}(fight_id);
            CREATE INDEX IF NOT EXISTS idx_{feature}_fighter ON features.{feature}(fighter_id);
            CREATE INDEX IF NOT EXISTS idx_{feature}_event ON features.{feature}(event_id);
        """
        conn.execute(text(create_table_sql))
        conn.commit()

    return feature_groups

def initialize_new_schema(conn, schema_name):
    """Create the schema for the feature store.
    
    Args:
        conn: Database connection
        schema_name: Name of the schema to create
    """
    print(f"Dropping and recreating {schema_name} schema...")
    try:
        # Drop schema if exists (CASCADE will remove all objects within the schema)
        conn.execute(text(f'DROP SCHEMA IF EXISTS {schema_name} CASCADE;'))
        conn.commit()
        
        # Create new schema
        conn.execute(text(f'CREATE SCHEMA {schema_name};'))
        conn.commit()
        
        # Debug: Check if schema exists
        result = conn.execute(text(f"SELECT schema_name FROM information_schema.schemata WHERE schema_name = '{schema_name}'"))
        print(f"Schema exists: {bool(result.scalar())}")
        
    except Exception as e:
        print(f"\nError recreating {schema_name} schema: {str(e)}")
        raise


def create_table(conn, schema_name, table_name, columns: Set[str] = set(), include_patterns: Set[str] = set(), exclude_patterns: Set[str] = set()):
    """Create the training table that combines selected columns based on patterns."""
    feature_utils = FeatureUtils(conn)
    
    try:
        # Debug print
        print(f"\nCreating table {schema_name}.{table_name}")
        print(f"Required columns: {columns if columns else 'None'}")
        print(f"Include patterns: {include_patterns if include_patterns else 'None'}")
        print(f"Exclude patterns: {exclude_patterns if exclude_patterns else 'None'}")
        
        # Get all stat tables and their columns
        if len(columns) == 0:
            required = False
            stat_tables = feature_utils.get_stat_tables()
        else:
            required = True
            stat_tables = {table_name: columns}
        
        # Define static stats
        static_stats = {'age', 'days_since_last_fight', 'reach', 'ape', 'ufcage'}
        
        # Collect columns matching include patterns and not matching exclude patterns
        training_columns = []
        for table, columns in stat_tables.items():
            filtered_cols = []
            for col in columns:
                # Debug print column processing
                #print(f"Processing column: {col}")
                
                # Determine if this is a static stat
                is_static = any(pattern in col for pattern in static_stats)
                
                # Skip if we're processing the wrong table type
                if table_name == 'stats_raw' and is_static:
                    continue
                if table_name == 'static_stats_raw' and not is_static:
                    continue

                # If both pattern sets are empty and no required features, include all columns
                if not include_patterns and not exclude_patterns and not columns:
                    filtered_cols.append(col)
                    continue

                # 1) If columns (required_features) is set, only use those features
                if columns and required:
                    if col in columns:
                        # Only exclude if it's in exclude_patterns
                        if not any(pattern in col for pattern in exclude_patterns):
                            filtered_cols.append(col)
                        continue
                    
                # 2) Otherwise, revert to pattern matching
                matches_include = not include_patterns or any(pattern in col for pattern in include_patterns)
                matches_exclude = exclude_patterns and any(pattern in col for pattern in exclude_patterns)
                
                if matches_include and not matches_exclude:
                    filtered_cols.append(col)
            
            if filtered_cols:
                training_columns.extend([f"{table}.{col}" for col in filtered_cols])

        if training_columns == []:
            print(f"No columns found for {table_name}")
            return

        # Create the training table SQL
        create_table_sql = f"""
            CREATE TABLE IF NOT EXISTS {schema_name}.{table_name} (
                fight_id INTEGER NOT NULL,
                fighter_id INTEGER NOT NULL,
                event_id INTEGER NOT NULL,
                {', '.join(f"{col.split('.')[-1]} FLOAT" for col in training_columns)},
                PRIMARY KEY (fight_id, fighter_id),
                CONSTRAINT fk_fight FOREIGN KEY (fight_id) REFERENCES features.fight_mapping(fight_id),
                CONSTRAINT fk_fighter FOREIGN KEY (fighter_id) REFERENCES features.fighter_mapping(fighter_id),
                CONSTRAINT fk_event FOREIGN KEY (event_id) REFERENCES features.event_mapping(event_id)
            );
            
            CREATE INDEX IF NOT EXISTS idx_{table_name}_fight ON {schema_name}.{table_name}(fight_id);
            CREATE INDEX IF NOT EXISTS idx_{table_name}_fighter ON {schema_name}.{table_name}(fighter_id);
            CREATE INDEX IF NOT EXISTS idx_{table_name}_event ON {schema_name}.{table_name}(event_id);
        """

        # Debug print final SQL
        print("\nExecuting SQL")
        
        # Drop existing table if it exists
        drop_sql = f"DROP TABLE IF EXISTS {schema_name}.{table_name} CASCADE;"
        print(f"\nDropping existing table: {drop_sql}")
        conn.execute(text(drop_sql))
        conn.commit()
        
        # Create new table
        print("\nCreating new table...")
        result = conn.execute(text(create_table_sql))
        conn.commit()
        
        # Verify table creation
        verify_sql = f"""
            SELECT EXISTS (
                SELECT FROM information_schema.tables 
                WHERE table_schema = '{schema_name}'
                AND table_name = '{table_name}'
            );
        """
        exists = conn.execute(text(verify_sql)).scalar()
        print(f"\nTable {schema_name}.{table_name} exists: {exists}")
        
        if exists:
            # Get column count
            col_count_sql = f"""
                SELECT COUNT(*) 
                FROM information_schema.columns 
                WHERE table_schema = '{schema_name}' 
                AND table_name = '{table_name}';
            """
            col_count = conn.execute(text(col_count_sql)).scalar()
            print(f"Number of columns created: {col_count}")
        
    except Exception as e:
        print(f"\nError creating table {schema_name}.{table_name}:")
        print(f"Error type: {type(e).__name__}")
        print(f"Error message: {str(e)}")
        raise
    
    print(f"\n{table_name} table creation complete!")