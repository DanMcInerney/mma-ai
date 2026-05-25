import pytest
import pandas as pd
import os
import sys
from sqlalchemy import create_engine, text
from typing import Set

# Add the parent directory to sys.path so we can import project modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from libs.feature_store.create_training_data import CreateTrainingData


@pytest.fixture(scope="module")
def db_engine():
    """Create a database engine once for all tests."""
    # Read database configuration from environment variables or use default test database
    db_url = os.getenv('TEST_DATABASE_URL', 'postgresql://postgres:postgres@localhost:5432/mma-ai')
    engine = create_engine(db_url)
    
    # Verify database connection is working
    try:
        with engine.connect() as conn:
            result = conn.execute(text("SELECT 1")).fetchone()
            if not result or result[0] != 1:
                pytest.skip("Database connection test failed")
    except Exception as e:
        pytest.skip(f"Could not connect to test database: {str(e)}")
    
    yield engine


@pytest.fixture
def db_conn(db_engine):
    """Create a database connection for each test."""
    with db_engine.connect() as conn:
        conn.execution_options(isolation_level="AUTOCOMMIT")
        yield conn


def test_init(db_conn):
    """Test that the CreateTrainingData class initializes correctly."""
    # Test with default parameters
    ctd = CreateTrainingData(db_conn)
    assert ctd.include_patterns is None
    assert ctd.exclude_patterns is None
    assert ctd.required_features is None
    
    # Test with custom parameters
    include_patterns = {'age', 'reach'}
    exclude_patterns = {'_opp'}
    required_features = {'win'}
    
    ctd = CreateTrainingData(
        db_conn,
        include_patterns=include_patterns,
        exclude_patterns=exclude_patterns,
        required_features=required_features
    )
    
    assert ctd.include_patterns == include_patterns
    assert ctd.exclude_patterns == exclude_patterns
    assert ctd.required_features == required_features


def test_get_feature_tables(db_conn):
    """Test the _get_feature_tables method returns expected tables."""
    ctd = CreateTrainingData(db_conn)
    tables = ctd._get_feature_tables()
    
    # Check that we got some tables
    assert isinstance(tables, list)
    assert len(tables) > 0
    
    # Check that none of the excluded tables are present
    excluded_patterns = [
        '_first_time_opp_stats',
        '_first_time_avg_stats',
        'fight_stats',
        '_mapping',
        '_minimum'
    ]
    
    for table in tables:
        for pattern in excluded_patterns:
            assert pattern not in table, f"Table {table} should have been excluded"


def test_get_table_columns(db_conn):
    """Test the _get_table_columns method returns expected columns."""
    # Get a list of tables first
    ctd = CreateTrainingData(db_conn)
    tables = ctd._get_feature_tables()
    
    # Pick the first table and get its columns
    if tables:
        first_table = tables[0]
        columns = ctd._get_table_columns(first_table)
        
        # Check that we got some columns
        assert isinstance(columns, list)
        assert len(columns) > 0
        
        # Check that fighter_id, fight_id, event_id are in the columns
        expected_id_columns = ['fighter_id', 'fight_id', 'event_id']
        for column in expected_id_columns:
            assert column in columns, f"Column {column} should be present in table {first_table}"


def test_create_training_data_no_filters(db_conn):
    """Test create_training_data without any filters."""
    ctd = CreateTrainingData(db_conn)
    df = ctd.create_training_data()
    
    # Check that we got a DataFrame
    assert isinstance(df, pd.DataFrame)
    assert len(df) > 0
    
    # Check for required columns
    required_columns = ['fight_id', 'fighter_id', 'event_id']
    for column in required_columns:
        assert column in df.columns, f"Column {column} should be present in the DataFrame"


def test_create_training_data_with_include_patterns(db_conn):
    """Test create_training_data with include_patterns."""
    # Test with common features like age and reach
    include_patterns = {'age', 'reach'}
    ctd = CreateTrainingData(db_conn, include_patterns=include_patterns)
    df = ctd.create_training_data()
    
    # Check that we got a DataFrame
    assert isinstance(df, pd.DataFrame)
    
    # Check that at least some columns with 'age' and 'reach' exist
    age_columns = [col for col in df.columns if 'age' in col]
    reach_columns = [col for col in df.columns if 'reach' in col]
    
    assert len(age_columns) > 0, "Should have columns containing 'age'"
    assert len(reach_columns) > 0, "Should have columns containing 'reach'"


def test_create_training_data_with_exclude_patterns(db_conn):
    """Test create_training_data with exclude_patterns."""
    # Test excluding opponent stats
    exclude_patterns = {'_opp'}
    ctd = CreateTrainingData(db_conn, exclude_patterns=exclude_patterns)
    df = ctd.create_training_data()
    
    # Check that we got a DataFrame
    assert isinstance(df, pd.DataFrame)
    
    # Check that no columns contain '_opp'
    opp_columns = [col for col in df.columns if '_opp' in col]
    assert len(opp_columns) == 0, "Should not have columns containing '_opp'"


def test_create_training_data_with_required_features(db_conn):
    """Test create_training_data with required_features."""
    # Test with a required feature that might be excluded by patterns
    include_patterns = {'age', 'reach'}
    exclude_patterns = {'win'}  # Would normally exclude 'win' columns
    required_features = {'win_avg'}  # But we require 'win_avg'
    
    ctd = CreateTrainingData(
        db_conn,
        include_patterns=include_patterns,
        exclude_patterns=exclude_patterns,
        required_features=required_features
    )
    
    df = ctd.create_training_data()
    
    # Check that the required feature is present despite exclude pattern
    assert 'win_avg' in df.columns, "Required feature 'win_avg' should be present"


@pytest.mark.parametrize(
    "include_patterns,exclude_patterns,expected_present,expected_absent",
    [
        (
            # Test with realistic training features
            {'age', 'reach', 'days_since_last_fight', 'ufcage', 'dec_adjperf_dec_avg', 'opp_dec_avg'},
            {'_sdev', '_ratio'},
            ['age', 'reach', 'days_since_last_fight'],  # Patterns that should be present
            ['_sdev', '_ratio']  # Patterns that should be absent
        ),
        (
            # Test with minimal features
            {'age', 'reach'},
            {'_avg', '_mad'},
            ['age', 'reach'],
            ['_avg', '_mad']
        )
    ]
)
def test_complex_filter_combinations(db_conn, include_patterns, exclude_patterns, expected_present, expected_absent):
    """Test create_training_data with different combinations of filters."""
    ctd = CreateTrainingData(
        db_conn,
        include_patterns=include_patterns,
        exclude_patterns=exclude_patterns
    )
    
    df = ctd.create_training_data()
    
    # Check that we got a DataFrame with the right columns
    assert isinstance(df, pd.DataFrame)
    assert len(df) > 0
    
    # Check for expected present patterns
    for pattern in expected_present:
        matching_columns = [col for col in df.columns if pattern in col]
        assert len(matching_columns) > 0, f"Should have columns containing '{pattern}'"
    
    # Check for expected absent patterns
    for pattern in expected_absent:
        excluded_columns = [col for col in df.columns if pattern in col]
        assert len(excluded_columns) == 0, f"Should not have columns containing '{pattern}'"


def test_dataframe_content(db_conn):
    """Test that the created DataFrame contains valid data."""
    include_patterns = {'age', 'reach'}
    ctd = CreateTrainingData(db_conn, include_patterns=include_patterns)
    df = ctd.create_training_data()
    
    # Check for non-null values in key columns
    assert df['fight_id'].isnull().sum() == 0, "Should have no null values in fight_id"
    assert df['fighter_id'].isnull().sum() == 0, "Should have no null values in fighter_id"
    assert df['event_id'].isnull().sum() == 0, "Should have no null values in event_id"
    
    # Check numeric columns have reasonable values
    for col in df.columns:
        if df[col].dtype in ['int64', 'float64'] and col not in ['fight_id', 'fighter_id', 'event_id']:
            # Check for reasonable range of values
            assert not ((df[col] > 1e6).any() and (df[col] < -1e6).any()), f"Column {col} has unreasonable values" 

def test_data_integrity_after_chunking(db_conn):
    """Test that data integrity is maintained during chunking and merging."""
    # Use specific include patterns to keep the test manageable
    include_patterns = {'age', 'reach', 'ufcage', 'days_since_last_fight'}
    ctd = CreateTrainingData(db_conn, include_patterns=include_patterns)
    
    # Override the _get_feature_dataframe method to add debugging
    original_method = ctd._get_feature_dataframe
    
    def wrapped_method(*args, **kwargs):
        # Call original method to get the result dataframe
        result_df = original_method(*args, **kwargs)
        
        # Test data integrity for known fighters
        test_fighters = ['alexander volkanovski', 'max holloway', 'ian garry', 'shavkat rakhmonov']
        
        for fighter in test_fighters:
            # Get fighter data
            fighter_data = result_df[result_df['fighter_name'].str.lower() == fighter.lower()]
            
            if len(fighter_data) > 0:
                # Check for NaN values in key columns
                nan_count = fighter_data[['age', 'reach', 'ufcage', 'days_since_last_fight']].isna().sum()
                
                # Verify NaN counts
                print(f"\nNaN counts for {fighter}:")
                print(nan_count)
                
                # Check for reasonable event dates
                print(f"Event dates for {fighter}:")
                dates = fighter_data['event_date'].tolist()
                print(dates)
                
                # Verify dates are after 2000 (UFC modern era)
                for date in dates:
                    assert pd.Timestamp(date).year >= 2000, f"Found unreasonable date {date} for {fighter}"
                
                # Check data consistency
                assert nan_count.sum() == 0, f"Found NaN values for {fighter}: {nan_count}"
            else:
                print(f"No data found for {fighter}")
        
        return result_df
    
    # Replace method with wrapped version
    ctd._get_feature_dataframe = wrapped_method
    
    # Execute the method
    df = ctd.create_training_data()
    
    # Verify expected rows and columns
    assert len(df) > 0, "DataFrame should have rows"
    assert set(['age', 'reach', 'ufcage', 'days_since_last_fight']).issubset(df.columns), "Key columns missing"


def test_concat_vs_merge_method(db_conn):
    """Test to compare concat vs merge method for combining dataframes."""
    # Setup simple include patterns
    include_patterns = {'age', 'reach'}
    ctd = CreateTrainingData(db_conn, include_patterns=include_patterns)
    
    # Get feature tables and build columns for two "chunks"
    tables = ctd._get_feature_tables()
    tables = [t for t in tables if 'age' in t or 'reach' in t]
    
    # Get base query directly
    base_query = """
        SELECT 
            fsd.fight_id,
            fsd.fighter_id,
            fsd.event_id,
            e.event_date,
            fm.method,
            fm.result,
            f.fighter_name,
            f.fighter_dob,
            fm.fighter1_id,
            fm.fighter2_id
        FROM features.fight_stats_derived fsd
        LEFT JOIN features.event_mapping e ON fsd.event_id = e.event_id
        LEFT JOIN features.fight_mapping fm ON fsd.fight_id = fm.fight_id
        LEFT JOIN features.fighter_mapping f ON fsd.fighter_id = f.fighter_id;
    """
    base_df = pd.read_sql_query(base_query, db_conn)
    
    # Create two test chunks with different columns
    chunk1_query = """
        SELECT 
            fsd.fight_id,
            fsd.fighter_id,
            fsd.event_id,
            a.age
        FROM features.fight_stats_derived fsd
        JOIN features.age a 
        ON fsd.fight_id = a.fight_id 
        AND fsd.fighter_id = a.fighter_id;
    """
    
    chunk2_query = """
        SELECT 
            fsd.fight_id,
            fsd.fighter_id,
            fsd.event_id,
            r.reach
        FROM features.fight_stats_derived fsd
        JOIN features.reach r 
        ON fsd.fight_id = r.fight_id 
        AND fsd.fighter_id = r.fighter_id;
    """
    
    chunk1_df = pd.read_sql_query(chunk1_query, db_conn)
    chunk2_df = pd.read_sql_query(chunk2_query, db_conn)
    
    # Method 1: Current concat approach
    # Merge chunk1 with base
    result1_df = pd.merge(
        base_df, chunk1_df,
        on=['fight_id', 'fighter_id', 'event_id'],
        how='left'
    )
    
    # Merge chunk2 with base
    temp1_df = pd.merge(
        base_df, chunk2_df,
        on=['fight_id', 'fighter_id', 'event_id'],
        how='left'
    )
    
    # Extract feature columns and concat
    feature_cols = [col for col in temp1_df.columns if col not in base_df.columns]
    concat_result = pd.concat([
        result1_df, 
        temp1_df[feature_cols]
    ], axis=1)
    
    # Method 2: Alternative merge approach
    # Merge chunk1 with base
    result2_df = pd.merge(
        base_df, chunk1_df,
        on=['fight_id', 'fighter_id', 'event_id'],
        how='left'
    )
    
    # Merge chunk2 with result1
    merge_result = pd.merge(
        result2_df, chunk2_df,
        on=['fight_id', 'fighter_id', 'event_id'],
        how='left'
    )
    
    # Compare the two results for specific fighters
    test_fighters = ['alexander volkanovski', 'max holloway', 'ian garry', 'shavkat rakhmonov']
    
    for fighter in test_fighters:
        print(f"\nTesting data for {fighter}:")
        
        # Find the fighter in both result sets
        concat_data = concat_result[concat_result['fighter_name'].str.lower() == fighter.lower()]
        merge_data = merge_result[merge_result['fighter_name'].str.lower() == fighter.lower()]
        
        # Count rows
        print(f"  Concat method: {len(concat_data)} rows")
        print(f"  Merge method: {len(merge_data)} rows")
        
        # Check NaN counts
        concat_nans = concat_data[['age', 'reach']].isna().sum()
        merge_nans = merge_data[['age', 'reach']].isna().sum()
        
        print(f"  Concat NaNs: {concat_nans.sum()}")
        print(f"  Merge NaNs: {merge_nans.sum()}")
        
        # Check first few rows to see differences
        if len(concat_data) > 0 and len(merge_data) > 0:
            concat_sample = concat_data[['event_date', 'age', 'reach']].head(3)
            merge_sample = merge_data[['event_date', 'age', 'reach']].head(3)
            print("  Concat sample:")
            print(concat_sample)
            print("  Merge sample:")
            print(merge_sample)
            
            # Verify data consistency
            assert merge_nans.sum() <= concat_nans.sum(), f"Merge method should have equal or fewer NaNs than concat"
            
            # Check for specific issues with dates
            for date in merge_data['event_date']:
                assert pd.Timestamp(date).year >= 2000, f"Found unreasonable date {date} for {fighter}"


def test_event_date_accuracy(db_conn):
    """Test that event dates are correctly pulled from database."""
    # First query the database directly to get fighter and event info
    fighter_query = """
    SELECT 
        f.fighter_name, 
        e.event_date,
        fm.fight_id
    FROM features.fighter_mapping f
    JOIN features.fight_mapping fm ON f.fighter_id = fm.fighter1_id OR f.fighter_id = fm.fighter2_id
    JOIN features.event_mapping e ON fm.event_id = e.event_id
    WHERE LOWER(f.fighter_name) IN ('ian garry', 'shavkat rakhmonov')
    ORDER BY e.event_date DESC
    LIMIT 10;
    """
    
    db_dates = pd.read_sql_query(fighter_query, db_conn)
    print("Event dates directly from database:")
    print(db_dates)
    
    # Now create training data with minimal columns
    include_patterns = {'age'}
    ctd = CreateTrainingData(db_conn, include_patterns=include_patterns)
    df = ctd.create_training_data()
    
    # Filter for the fighters we're interested in
    fighters_to_check = ['ian garry', 'shavkat rakhmonov']
    filtered_df = df[df['fighter_name'].str.lower().isin(fighters_to_check)]
    
    print("\nEvent dates from createtrainingdata:")
    print(filtered_df[['fighter_name', 'event_date']].sort_values('event_date', ascending=False))
    
    # Compare the dates
    for fighter in fighters_to_check:
        db_fighter_dates = db_dates[db_dates['fighter_name'].str.lower() == fighter.lower()]['event_date']
        train_fighter_dates = filtered_df[filtered_df['fighter_name'].str.lower() == fighter.lower()]['event_date']
        
        print(f"\nChecking dates for {fighter}:")
        print(f"  Database: {db_fighter_dates.tolist()}")
        print(f"  Training: {train_fighter_dates.tolist()}")
        
        # Verify all training dates are in the database dates
        for date in train_fighter_dates:
            date_found = any((pd.Timestamp(date) - pd.Timestamp(db_date)).total_seconds() < 86400 
                            for db_date in db_fighter_dates)
            assert date_found, f"Date {date} for {fighter} not found in database dates"
            assert pd.Timestamp(date).year >= 2000, f"Unreasonable date {date} for {fighter}"


def test_missing_feature_preservation(db_conn):
    """Test that fighter data is preserved even when some features are missing (e.g., odds)."""
    # Create a dataframe with specific features for a known fight 
    # (Volkanovski vs Rodriguez, fight_id=71, which we know has no odds data)
    include_patterns = {'age', 'reach', 'ufcage', 'days_since_last_fight', 'odds'}
    ctd = CreateTrainingData(db_conn, include_patterns=include_patterns)
    df = ctd.create_training_data()
    
    # Filter for the specific fight
    fight_data = df[df['fight_id'] == 71]
    
    # Check that the fight exists in the result
    assert len(fight_data) == 2, f"Expected 2 rows for fight_id=71, got {len(fight_data)}"
    
    # Verify the fighter names
    fighter_names = sorted(fight_data['fighter_name'].str.lower().tolist())
    expected_names = ['alexander volkanovski', 'yair rodriguez']
    assert fighter_names == expected_names, f"Expected {expected_names}, got {fighter_names}"
    
    # Check that non-odds features are present and not NaN
    for _, row in fight_data.iterrows():
        for feature in ['age', 'reach', 'ufcage', 'days_since_last_fight']:
            assert not pd.isna(row[feature]), f"Feature {feature} is NaN for {row['fighter_name']}"
    
    # Confirm that the LEFT JOIN approach preserves rows even with missing features
    # by checking if we have the expected number of rows
    row_count_query = """
    SELECT COUNT(*) FROM features.fight_stats_derived;
    """
    expected_total_rows = pd.read_sql(text(row_count_query), db_conn).iloc[0, 0]
    assert len(df) == expected_total_rows, f"Expected {expected_total_rows} rows, got {len(df)}"
    
    print(f"Verified that fighter data is preserved with LEFT JOIN even when some features (odds) are missing")


def test_volkanovski_data_integrity(db_conn):
    """Test specifically for Alexander Volkanovski's data."""
    # First, get Volkanovski's data directly from the database
    volk_query = """
    WITH volk_fighter AS (
        SELECT fighter_id FROM features.fighter_mapping
        WHERE LOWER(fighter_name) LIKE '%volkanovski%'
    )
    SELECT 
        fsd.fight_id,
        e.event_date,
        a.age,
        r.reach,
        u.ufcage,
        d.days_since_last_fight
    FROM volk_fighter v
    JOIN features.fight_stats_derived fsd ON v.fighter_id = fsd.fighter_id
    JOIN features.event_mapping e ON fsd.event_id = e.event_id
    JOIN features.age a ON fsd.fight_id = a.fight_id AND fsd.fighter_id = a.fighter_id
    JOIN features.reach r ON fsd.fight_id = r.fight_id AND fsd.fighter_id = r.fighter_id
    JOIN features.ufcage u ON fsd.fight_id = u.fight_id AND fsd.fighter_id = u.fighter_id
    JOIN features.days_since_last_fight d ON fsd.fight_id = d.fight_id AND fsd.fighter_id = d.fighter_id
    ORDER BY e.event_date;
    """
    
    db_volk = pd.read_sql_query(text(volk_query), db_conn)
    print("Volkanovski data directly from database:")
    print(db_volk)
    
    # Get a count of all Volkanovski fights in the main table
    volk_count_query = """
    SELECT COUNT(*) 
    FROM features.fight_stats_derived fsd
    JOIN features.fighter_mapping f ON fsd.fighter_id = f.fighter_id
    WHERE LOWER(f.fighter_name) LIKE '%volkanovski%';
    """
    total_volk_fights = pd.read_sql(text(volk_count_query), db_conn).iloc[0, 0]
    print(f"Total Volkanovski fights in database: {total_volk_fights}")
    
    # Now create training data with the same features
    include_patterns = {'age', 'reach', 'ufcage', 'days_since_last_fight'}
    ctd = CreateTrainingData(db_conn, include_patterns=include_patterns)
    
    # Get data through CreateTrainingData
    df = ctd.create_training_data()
    
    # Filter for Volkanovski
    volk_df = df[df['fighter_name'].str.lower().str.contains('volkanovski')]
    
    # Verify we have the expected number of rows
    assert len(volk_df) == total_volk_fights, f"Expected {total_volk_fights} Volkanovski fights, got {len(volk_df)}"
    
    print("\nVolkanovski data from CreateTrainingData:")
    print(volk_df[['fight_id', 'event_date', 'age', 'reach', 'ufcage', 'days_since_last_fight']].head())
    
    # Check specifically for fight_id 71 (vs Rodriguez)
    volk_rodriguez = volk_df[volk_df['fight_id'] == 71]
    if len(volk_rodriguez) > 0:
        print("\nVerifying Volkanovski vs Rodriguez fight (id=71):")
        print(volk_rodriguez[['event_date', 'age', 'reach', 'ufcage', 'days_since_last_fight']])
        assert not volk_rodriguez[['age', 'reach', 'ufcage', 'days_since_last_fight']].isna().any().any(), "Found NaN values in Volkanovski vs Rodriguez fight"
    
    # Check for NaN values in key columns
    nan_counts = volk_df[['age', 'reach', 'ufcage', 'days_since_last_fight']].isna().sum()
    print(f"\nNaN counts: {nan_counts}")
    assert nan_counts.sum() == 0, f"Found {nan_counts.sum()} NaN values in Volkanovski data"
    
    # Compare data values with direct database query
    for i, row in db_volk.iterrows():
        # Find matching fight_id in volk_df
        match_rows = volk_df[volk_df['fight_id'] == row['fight_id']]
        assert len(match_rows) == 1, f"Expected 1 row for fight_id {row['fight_id']}, got {len(match_rows)}"
        
        match_row = match_rows.iloc[0]
        for col in ['age', 'reach', 'ufcage', 'days_since_last_fight']:
            db_val = row[col]
            df_val = match_row[col]
            
            # Verify values match (allowing small float differences)
            assert not pd.isna(df_val), f"Feature {col} is NaN for fight_id {row['fight_id']}"
            assert abs(df_val - db_val) < 0.01, f"Value mismatch for {col}: {df_val} vs {db_val} for fight_id {row['fight_id']}"
    
    print("\nAll Volkanovski data integrity checks passed!")
