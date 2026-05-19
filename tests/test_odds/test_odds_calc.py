import pytest
from datetime import datetime, timedelta, date
import pandas as pd
from sqlalchemy import create_engine, text
from unittest.mock import MagicMock, patch, call, ANY
import time
from collections import defaultdict
import logging
import re # Import re for patching query

# Ensure the logger is configured to avoid warnings/errors during tests
logging.basicConfig(level=logging.INFO)

from libs.feature_store.calculators.odds_calc import OddsCalculator

@pytest.fixture
def mock_main_conn():
    """Mock database connection for the main mma-ai database."""
    conn = MagicMock(name="MainDBConnection")
    # Default fetchall for load_fight_mappings (can be overridden in tests)
    conn.execute.return_value.fetchall.return_value = []
    # Mock rowcount for bulk_normalize_vigless_odds
    conn.execute.return_value.rowcount = 100
    # Mock context manager behavior
    conn.__enter__.return_value = conn
    conn.__exit__.return_value = None
    return conn

@pytest.fixture
def mock_bfo_engine():
    """Create in-memory SQLite engine for the BFO odds database."""
    # Use SQLite :memory: database for BFO tests
    engine = create_engine('sqlite:///:memory:')
    
    # Create the bfo table needed by load_bfo_odds (no schema in SQLite)
    with engine.connect() as conn:
        conn.execute(text('''
            CREATE TABLE bfo (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                fighter VARCHAR(255) NOT NULL,
                opponent VARCHAR(255) NOT NULL,
                timestamp TIMESTAMP NOT NULL,
                odds FLOAT NOT NULL
            )
        '''))
        # Add necessary functions for the query if using complex SQL
        # DATE_TRUNC and ROW_NUMBER are specific to PG/other DBs
        # SQLite needs workarounds or simpler test data/queries
        # For simplicity here, we assume the test data/query works or is patched.
        conn.commit()
    return engine

@pytest.fixture
def odds_calculator(mock_main_conn, mock_bfo_engine):
    """Create OddsCalculator instance with mocked main connection and real BFO engine."""
    # Patch create_engine used within OddsCalculator.__init__
    with patch('libs.feature_store.calculators.odds_calc.create_engine', return_value=mock_bfo_engine):
        calculator = OddsCalculator(mock_main_conn, bfo_db_url='sqlite:///:memory:')
        # Assign the mock engine directly for BFO operations within the calculator
        # This ensures tests use the in-memory DB for BFO operations
        calculator.bfo_engine = mock_bfo_engine
        calculator.logger = MagicMock(spec=logging.Logger)
        return calculator

def insert_bfo_test_data(engine, data):
    """Helper to insert test data into the in-memory BFO database (table 'bfo')."""
    with engine.connect() as conn:
        for entry in data:
            # Ensure timestamp is in a format SQLite understands if necessary
            # Default ISO format usually works
            conn.execute(
                text("""
                    INSERT INTO bfo (fighter, opponent, timestamp, odds)
                    VALUES (:fighter, :opponent, :timestamp, :odds)
                """),
                entry
            )
        conn.commit()

def create_fight_mapping_mock_result(mappings):
    """Creates a mock result set suitable for mock_main_conn.execute().fetchall()."""
    mock_rows = []
    for m in mappings:
        mock_rows.append((
            m['fight_id'],
            m['fighter1_id'],
            m['fighter2_id'],
            m['event_id'],
            m['fighter1_name'],
            m['fighter2_name'],
            m['event_date']
        ))
    return mock_rows

def structure_fight_mappings_for_test(raw_mappings_list):
    """Simulates the output dict structure of load_fight_mappings for testing match_fighter_odds."""
    date_to_fights = defaultdict(list)
    for row_dict in raw_mappings_list:
        event_date = row_dict['event_date']
        for offset in range(-3, 4):
            date_key = event_date + timedelta(days=offset)
            date_to_fights[date_key].append(row_dict)
    return date_to_fights

class TestOddsCalculator:

    def test_create_odds_table(self, odds_calculator, mock_main_conn):
        """Test creation and modification of the features.odds table."""
        odds_calculator.create_odds_table()

        # Expected calls: 1 for CREATE TABLE, 1 for ADD COLUMNS, 1 for modifications = 3
        assert mock_main_conn.execute.call_count == 3
        assert mock_main_conn.commit.call_count == 3

        # Check CREATE TABLE call (Call 1)
        create_call_args = mock_main_conn.execute.call_args_list[0][0][0]
        sql_create = str(create_call_args).lower()
        assert "create table if not exists features.odds" in sql_create
        assert 'sevenday_opening_odds float' in sql_create # Check renamed column
        assert 'sevenday_ip_opening_odds float' in sql_create # Check renamed column
        assert 'sevenday_vigless_ip_opening_odds float' in sql_create # Check renamed column
        assert "primary key (fight_id, fighter_id, event_id)" in sql_create
        assert "constraint fk_fight" in sql_create
        assert "constraint fk_fighter" in sql_create
        assert "constraint fk_event" in sql_create
        
        # Check ADD COLUMNS call (Call 2)
        add_cols_args = mock_main_conn.execute.call_args_list[1][0][0]
        sql_add_cols = str(add_cols_args).lower()
        assert "alter table features.odds" in sql_add_cols
        assert "add column if not exists sevenday_opening_odds" in sql_add_cols
        assert "add column if not exists sevenday_ip_opening_odds" in sql_add_cols
        assert "add column if not exists sevenday_vigless_ip_opening_odds" in sql_add_cols

        # Check Modifications call (Call 3) (DO blocks, INDEX, RENAME checks)
        modify_call_args = mock_main_conn.execute.call_args_list[2][0][0]
        sql_modify = str(modify_call_args).lower()
        assert "do $$" in sql_modify # Check for DO blocks
        assert "pg_constraint" in sql_modify # PK check
        assert "alter table features.odds add primary key" in sql_modify
        assert "create index if not exists idx_odds_fight" in sql_modify
        assert "create index if not exists idx_odds_fighter" in sql_modify
        assert "create index if not exists idx_odds_event" in sql_modify
        assert "information_schema.columns" in sql_modify # Rename check
        assert "alter table features.odds rename column" in sql_modify

        # Check logger calls
        odds_calculator.logger.info.assert_any_call("Creating features.odds table if it doesn't exist")
        odds_calculator.logger.info.assert_any_call("Table features.odds ensured to exist and columns added. Applying modifications...")
        odds_calculator.logger.info.assert_any_call("features.odds table modifications complete.")

    def test_load_bfo_odds(self, odds_calculator, mock_bfo_engine):
        """Test loading and formatting BFO odds data, patching query for SQLite compatibility."""
        # Populate test data
        bfo_data = [
            # Fight 1: Normal case
            {"fighter": "fighter_a", "opponent": "opponent_x", "timestamp": "2023-01-10 10:00:00", "odds": 1.5}, # Jan Opening
            {"fighter": "fighter_a", "opponent": "opponent_x", "timestamp": "2023-01-10 15:00:00", "odds": 1.7}, # Jan Closing
            # Fight 1: Opponent's odds (same month)
            {"fighter": "opponent_x", "opponent": "fighter_a", "timestamp": "2023-01-10 11:00:00", "odds": 2.8}, # Jan Opening
            {"fighter": "opponent_x", "opponent": "fighter_a", "timestamp": "2023-01-10 16:00:00", "odds": 2.5}, # Jan Closing
            # Fight 2: Rematch in different month
            {"fighter": "fighter_a", "opponent": "opponent_x", "timestamp": "2023-03-15 09:00:00", "odds": 1.9}, # Mar Opening
            {"fighter": "fighter_a", "opponent": "opponent_x", "timestamp": "2023-03-15 14:00:00", "odds": 2.1}, # Mar Closing
            # Fight 3: Only one odds entry
            {"fighter": "fighter_b", "opponent": "opponent_y", "timestamp": "2023-02-05 12:00:00", "odds": 3.0}, # Feb
            # Fight 4: Middle odds entry (should be ignored)
            {"fighter": "fighter_c", "opponent": "opponent_z", "timestamp": "2023-04-01 10:00:00", "odds": 1.4}, # Apr Opening
            {"fighter": "fighter_c", "opponent": "opponent_z", "timestamp": "2023-04-01 11:00:00", "odds": 1.45}, # Middle
            {"fighter": "fighter_c", "opponent": "opponent_z", "timestamp": "2023-04-01 12:00:00", "odds": 1.5}, # Apr Closing
        ]
        insert_bfo_test_data(mock_bfo_engine, bfo_data)

        # Patch the text() function within the scope of load_bfo_odds
        # to modify the query for SQLite compatibility
        original_text = text
        def patched_text(query_str, *args, **kwargs):
            # Replace PG-specific date function with SQLite equivalent
            query_str_sqlite = query_str.replace("to_char(timestamp, 'YYYY-MM')", "strftime('%Y-%m', timestamp)")
            # Remove schema prefix
            query_str_sqlite = query_str_sqlite.replace("bestfightodds.bfo", "bfo")
            # Use original_text to execute the modified query
            return original_text(query_str_sqlite, *args, **kwargs)

        with patch('libs.feature_store.calculators.odds_calc.text', patched_text):
            result = odds_calculator.load_bfo_odds()

        # Assertions (focus on Python processing given simplified SQL mock)
        # There are 5 distinct fighter/opponent/month groups in the test data
        assert len(result) == 5 # Updated assertion
        key1_jan = ('fighter_a', 'opponent_x', '2023-01')
        assert key1_jan in result
        assert result[key1_jan]['opening']['odds'] == 1.5
        assert result[key1_jan]['closing']['odds'] == 1.7
        # Check sevenday (should be opening in this simple case as it's the only one > 7 days before closing)
        # Closing: 2023-01-10 15:00, 7 days prior target: 2023-01-03 15:00
        # Closest entry <= closing: 2023-01-10 10:00 (opening)
        assert result[key1_jan]['sevenday_opening']['odds'] == 1.5

        key1_opp_jan = ('opponent_x', 'fighter_a', '2023-01')
        assert key1_opp_jan in result
        assert result[key1_opp_jan]['opening']['odds'] == 2.8
        assert result[key1_opp_jan]['closing']['odds'] == 2.5
        # Closing: 2023-01-10 16:00, 7 days prior target: 2023-01-03 16:00
        # Closest entry <= closing: 2023-01-10 11:00 (opening)
        assert result[key1_opp_jan]['sevenday_opening']['odds'] == 2.8

        key2_feb = ('fighter_b', 'opponent_y', '2023-02') # Single entry
        assert key2_feb in result
        assert result[key2_feb]['opening']['odds'] == 3.0
        assert result[key2_feb]['closing']['odds'] == 3.0
        assert result[key2_feb]['sevenday_opening']['odds'] == 3.0 # Falls back to opening/closing
        
        key3_mar = ('fighter_a', 'opponent_x', '2023-03')
        assert key3_mar in result
        assert result[key3_mar]['opening']['odds'] == 1.9
        assert result[key3_mar]['closing']['odds'] == 2.1
        assert result[key3_mar]['sevenday_opening']['odds'] == 1.9 # Falls back
        
        key4_apr = ('fighter_c', 'opponent_z', '2023-04')
        assert key4_apr in result
        assert result[key4_apr]['opening']['odds'] == 1.4
        assert result[key4_apr]['closing']['odds'] == 1.5
        assert result[key4_apr]['sevenday_opening']['odds'] == 1.4 # Falls back

    def test_load_fight_mappings(self, odds_calculator, mock_main_conn):
        """Test loading fight mappings into a date-indexed dictionary with date window."""
        # Mock result for the fight mappings query
        raw_mappings = [
            {'fight_id': 1, 'fighter1_id': 101, 'fighter2_id': 102, 'event_id': 201, 'fighter1_name': 'fighter_a', 'fighter2_name': 'opponent_x', 'event_date': date(2023, 1, 10)},
            {'fight_id': 2, 'fighter1_id': 103, 'fighter2_id': 104, 'event_id': 202, 'fighter1_name': 'fighter_b', 'fighter2_name': 'opponent_y', 'event_date': date(2023, 2, 5)},
            {'fight_id': 3, 'fighter1_id': 101, 'fighter2_id': 102, 'event_id': 203, 'fighter1_name': 'fighter_a', 'fighter2_name': 'opponent_x', 'event_date': date(2023, 3, 15)}, # Rematch
        ]
        mock_main_conn.execute.return_value.fetchall.return_value = create_fight_mapping_mock_result(raw_mappings)

        # Call the method
        fight_mappings = odds_calculator.load_fight_mappings()

        # Assertions
        target_date = date(2023, 1, 11) # Should include fight 1 (Jan 10)
        assert target_date in fight_mappings
        assert len(fight_mappings[target_date]) == 1
        assert fight_mappings[target_date][0]['fight_id'] == 1
        assert fight_mappings[target_date][0]['fighter1_name'] == 'fighter_a'

        target_date_outside = date(2023, 1, 15) # Should not include fight 1 (Jan 10)
        assert target_date_outside not in fight_mappings

        target_date_rematch = date(2023, 3, 14) # Should include fight 3 (Mar 15)
        assert target_date_rematch in fight_mappings
        assert len(fight_mappings[target_date_rematch]) == 1
        assert fight_mappings[target_date_rematch][0]['fight_id'] == 3

        # Verify the SQL query was executed
        mock_main_conn.execute.assert_called_once()
        sql_executed = str(mock_main_conn.execute.call_args[0][0]).lower()
        assert "select" in sql_executed
        assert "features.fight_mapping" in sql_executed
        assert "features.fighter_mapping" in sql_executed
        assert "features.event_mapping" in sql_executed
        assert "lower(fm1.fighter_name)" in sql_executed
        assert "lower(fm2.fighter_name)" in sql_executed

    @pytest.mark.parametrize("scenario, bfo_odds_in, raw_fight_map_list, expected_records_out", [
        # Scenario 1: Direct match, opponent odds present
        (
            "Direct Match, Opponent Odds Found",
            { # CORRECTED STRUCTURE: dict as value
                ('fighter_a', 'opponent_x', '2023-01'): {'opening': {'timestamp': datetime(2023, 1, 10, 10, 0), 'odds': 1.5}, 'closing': {'timestamp': datetime(2023, 1, 10, 15, 0), 'odds': 1.7}, 'sevenday_opening': {'timestamp': datetime(2023, 1, 10, 10, 0), 'odds': 1.5}}, # Using opening as sevenday fallback
                ('opponent_x', 'fighter_a', '2023-01'): {'opening': {'timestamp': datetime(2023, 1, 10, 11, 0), 'odds': 2.8}, 'closing': {'timestamp': datetime(2023, 1, 10, 16, 0), 'odds': 2.5}, 'sevenday_opening': {'timestamp': datetime(2023, 1, 10, 11, 0), 'odds': 2.8}}
            },
            [{'fight_id': 1, 'fighter1_id': 101, 'fighter2_id': 102, 'event_id': 201, 'fighter1_name': 'fighter_a', 'fighter2_name': 'opponent_x', 'event_date': date(2023, 1, 10)}],
            [ # Expected records with sevenday odds
                {'fight_id': 1, 'fighter_id': 101, 'event_id': 201, 'fighter_name': 'fighter_a', 'opponent_name': 'opponent_x', 'opening_odds': 1.5, 'closing_odds': 1.7, 'sevenday_opening_odds': 1.5},
                {'fight_id': 1, 'fighter_id': 102, 'event_id': 201, 'fighter_name': 'opponent_x', 'opponent_name': 'fighter_a', 'opening_odds': 2.8, 'closing_odds': 2.5, 'sevenday_opening_odds': 2.8}
            ]
        ),
        # Scenario 2: Reversed match, opponent odds present (Same data as Scen 1, just testing order)
        (
            "Reversed Match, Opponent Odds Found",
             { # BFO has opponent listed first
                ('opponent_x', 'fighter_a', '2023-01'): {'opening': {'timestamp': datetime(2023, 1, 10, 11, 0), 'odds': 2.8}, 'closing': {'timestamp': datetime(2023, 1, 10, 16, 0), 'odds': 2.5}, 'sevenday_opening': {'timestamp': datetime(2023, 1, 10, 11, 0), 'odds': 2.8}},
                ('fighter_a', 'opponent_x', '2023-01'): {'opening': {'timestamp': datetime(2023, 1, 10, 10, 0), 'odds': 1.5}, 'closing': {'timestamp': datetime(2023, 1, 10, 15, 0), 'odds': 1.7}, 'sevenday_opening': {'timestamp': datetime(2023, 1, 10, 10, 0), 'odds': 1.5}}
             },
            [{'fight_id': 1, 'fighter1_id': 101, 'fighter2_id': 102, 'event_id': 201, 'fighter1_name': 'fighter_a', 'fighter2_name': 'opponent_x', 'event_date': date(2023, 1, 10)}], # DB has fighter_a first
            [ # Output should still map correctly, including sevenday
                {'fight_id': 1, 'fighter_id': 102, 'event_id': 201, 'fighter_name': 'opponent_x', 'opponent_name': 'fighter_a', 'opening_odds': 2.8, 'closing_odds': 2.5, 'sevenday_opening_odds': 2.8},
                {'fight_id': 1, 'fighter_id': 101, 'event_id': 201, 'fighter_name': 'fighter_a', 'opponent_name': 'opponent_x', 'opening_odds': 1.5, 'closing_odds': 1.7, 'sevenday_opening_odds': 1.5}
            ]
        ),
        # Scenario 3: Direct match, opponent odds MISSING (should be calculated)
        (
            "Direct Match, Opponent Odds Missing",
            {('fighter_a', 'opponent_x', '2023-01'): {'opening': {'timestamp': datetime(2023, 1, 10, 10, 0), 'odds': 1.5}, 'closing': {'timestamp': datetime(2023, 1, 10, 15, 0), 'odds': 2.0}, 'sevenday_opening': {'timestamp': datetime(2023, 1, 10, 10, 0), 'odds': 1.5}}} # No entry for ('opponent_x', 'fighter_a', ...)
            ,
            [{'fight_id': 1, 'fighter1_id': 101, 'fighter2_id': 102, 'event_id': 201, 'fighter1_name': 'fighter_a', 'fighter2_name': 'opponent_x', 'event_date': date(2023, 1, 10)}],
            [ # Expected with calculated opponent odds, including sevenday
                {'fight_id': 1, 'fighter_id': 101, 'event_id': 201, 'fighter_name': 'fighter_a', 'opponent_name': 'opponent_x', 'opening_odds': 1.5, 'closing_odds': 2.0, 'sevenday_opening_odds': 1.5},
                {'fight_id': 1, 'fighter_id': 102, 'event_id': 201, 'fighter_name': 'opponent_x', 'opponent_name': 'fighter_a', 'opening_odds': 3.0, 'closing_odds': 2.0, 'sevenday_opening_odds': 3.0} # Calculated: 1+(1/(1.5-1))=3.0, 1+(1/(2.0-1))=2.0. Sevenday uses opening fallback 1.5->3.0
            ]
        ),
        # Scenario 4: Date mismatch within window
        (
            "Date Mismatch within Window",
            {('fighter_a', 'opponent_x', '2023-01'): {'opening': {'timestamp': datetime(2023, 1, 10, 10, 0), 'odds': 1.5}, 'closing': {'timestamp': datetime(2023, 1, 12, 15, 0), 'odds': 1.7}, 'sevenday_opening': {'timestamp': datetime(2023, 1, 10, 10, 0), 'odds': 1.5}}} # Closing odds timestamp is Jan 12
            ,
            [{'fight_id': 1, 'fighter1_id': 101, 'fighter2_id': 102, 'event_id': 201, 'fighter1_name': 'fighter_a', 'fighter2_name': 'opponent_x', 'event_date': date(2023, 1, 10)}], # Fight date is Jan 10
            [ # Should still match based on closing date Jan 12 being in window of Jan 10
                {'fight_id': 1, 'fighter_id': 101, 'event_id': 201, 'fighter_name': 'fighter_a', 'opponent_name': 'opponent_x', 'opening_odds': 1.5, 'closing_odds': 1.7, 'sevenday_opening_odds': 1.5},
                {'fight_id': 1, 'fighter_id': 102, 'event_id': 201, 'fighter_name': 'opponent_x', 'opponent_name': 'fighter_a', 'opening_odds': pytest.approx(3.0), 'closing_odds': pytest.approx(1.0 + (1.0 / (1.7 - 1.0))), 'sevenday_opening_odds': pytest.approx(3.0)} # Calculated, sevenday uses opening fallback
            ]
        ),
        # Scenario 5: Rematch handled correctly (different event_month)
        (
            "Rematch",
            { # Two distinct fights based on event_month
                ('fighter_a', 'opponent_x', '2023-01'): {'opening': {'timestamp': datetime(2023, 1, 10, 10, 0), 'odds': 1.5}, 'closing': {'timestamp': datetime(2023, 1, 10, 15, 0), 'odds': 1.7}, 'sevenday_opening': {'timestamp': datetime(2023, 1, 10, 10, 0), 'odds': 1.5}},
                ('fighter_a', 'opponent_x', '2023-03'): {'opening': {'timestamp': datetime(2023, 3, 15, 9, 0), 'odds': 1.9}, 'closing': {'timestamp': datetime(2023, 3, 15, 14, 0), 'odds': 2.1}, 'sevenday_opening': {'timestamp': datetime(2023, 3, 15, 9, 0), 'odds': 1.9}}
            },
            [ # Fight mappings for both dates
                {'fight_id': 1, 'fighter1_id': 101, 'fighter2_id': 102, 'event_id': 201, 'fighter1_name': 'fighter_a', 'fighter2_name': 'opponent_x', 'event_date': date(2023, 1, 10)},
                {'fight_id': 3, 'fighter1_id': 101, 'fighter2_id': 102, 'event_id': 203, 'fighter1_name': 'fighter_a', 'fighter2_name': 'opponent_x', 'event_date': date(2023, 3, 15)}
            ],
            [ # Expect records for both fights
                {'fight_id': 1, 'fighter_id': 101, 'event_id': 201, 'fighter_name': 'fighter_a', 'opponent_name': 'opponent_x', 'opening_odds': 1.5, 'closing_odds': 1.7, 'sevenday_opening_odds': 1.5},
                {'fight_id': 1, 'fighter_id': 102, 'event_id': 201, 'fighter_name': 'opponent_x', 'opponent_name': 'fighter_a', 'opening_odds': pytest.approx(3.0), 'closing_odds': pytest.approx(1.0 + (1.0 / (1.7 - 1.0))), 'sevenday_opening_odds': pytest.approx(3.0)},
                {'fight_id': 3, 'fighter_id': 101, 'event_id': 203, 'fighter_name': 'fighter_a', 'opponent_name': 'opponent_x', 'opening_odds': 1.9, 'closing_odds': 2.1, 'sevenday_opening_odds': 1.9},
                {'fight_id': 3, 'fighter_id': 102, 'event_id': 203, 'fighter_name': 'opponent_x', 'opponent_name': 'fighter_a', 'opening_odds': pytest.approx(1.0 + (1.0 / (1.9 - 1.0))), 'closing_odds': pytest.approx(1.0 + (1.0 / (2.1 - 1.0))), 'sevenday_opening_odds': pytest.approx(1.0 + (1.0 / (1.9 - 1.0)))}
            ]
        ),
        # Scenario 6: No matching fight found for odds data
        (
            "No Matching Fight",
            {('nonexistent', 'fighter', '2023-01'): {'opening': {'timestamp': datetime(2023, 1, 10, 10, 0), 'odds': 1.5}, 'closing': {'timestamp': datetime(2023, 1, 10, 15, 0), 'odds': 1.7}, 'sevenday_opening': {'timestamp': datetime(2023, 1, 10, 10, 0), 'odds': 1.5}}},
            [{'fight_id': 1, 'fighter1_id': 101, 'fighter2_id': 102, 'event_id': 201, 'fighter1_name': 'fighter_a', 'fighter2_name': 'opponent_x', 'event_date': date(2023, 1, 10)}],
            [] # Expect no records
        ),
        # Scenario 7: Odds data missing opening or closing entry
        (
            "Missing Opening/Closing Odds",
            {('fighter_a', 'opponent_x', '2023-01'): {'closing': {'timestamp': datetime(2023, 1, 10, 15, 0), 'odds': 1.7}}} # Missing opening
            ,
            [{'fight_id': 1, 'fighter1_id': 101, 'fighter2_id': 102, 'event_id': 201, 'fighter1_name': 'fighter_a', 'fighter2_name': 'opponent_x', 'event_date': date(2023, 1, 10)}],
            [] # Expect no records
        ),
    ])
    def test_match_fighter_odds(self, odds_calculator, scenario, bfo_odds_in, raw_fight_map_list, expected_records_out):
        """Test matching BFO odds to fights across various scenarios."""
        # Structure the fight map list correctly for the test
        structured_fight_map = structure_fight_mappings_for_test(raw_fight_map_list)

        processed_odds = odds_calculator.match_fighter_odds(bfo_odds_in, structured_fight_map)

        # Check number of records
        assert len(processed_odds) == len(expected_records_out), f"Scenario '{scenario}' failed: Expected {len(expected_records_out)} records, got {len(processed_odds)}"

        # Check content
        processed_simplified = [
            {k: v for k, v in record.items() if k in ['fight_id', 'fighter_id', 'event_id', 'fighter_name', 'opponent_name', 'opening_odds', 'closing_odds', 'sevenday_opening_odds']}
            for record in processed_odds
        ]
        # Sort both lists for consistent comparison
        processed_simplified.sort(key=lambda x: (x['fight_id'], x['fighter_id']))
        expected_records_out.sort(key=lambda x: (x['fight_id'], x['fighter_id']))

        for i, expected in enumerate(expected_records_out):
             actual = processed_simplified[i]
             assert actual['fight_id'] == expected['fight_id'], f"Scenario '{scenario}' failed: fight_id mismatch at index {i}"
             assert actual['fighter_id'] == expected['fighter_id'], f"Scenario '{scenario}' failed: fighter_id mismatch at index {i}"
             assert actual['event_id'] == expected['event_id'], f"Scenario '{scenario}' failed: event_id mismatch at index {i}"
             assert actual['fighter_name'] == expected['fighter_name'], f"Scenario '{scenario}' failed: fighter_name mismatch at index {i}"
             assert actual['opponent_name'] == expected['opponent_name'], f"Scenario '{scenario}' failed: opponent_name mismatch at index {i}"
             # Use approx for odds comparisons due to potential float calculation
             assert actual['opening_odds'] == pytest.approx(expected['opening_odds']), f"Scenario '{scenario}' failed: opening_odds mismatch at index {i}"
             assert actual['closing_odds'] == pytest.approx(expected['closing_odds']), f"Scenario '{scenario}' failed: closing_odds mismatch at index {i}"
             # Check sevenday odds (allow approx for calculated opponent odds)
             assert actual['sevenday_opening_odds'] == pytest.approx(expected['sevenday_opening_odds']), f"Scenario '{scenario}' failed: sevenday_opening_odds mismatch at index {i}"
             # Check IPs are present (including sevenday)
             assert processed_odds[i]['ip_opening_odds'] is not None
             assert processed_odds[i]['ip_closing_odds'] is not None
             assert processed_odds[i]['vigless_ip_opening_odds'] is not None
             assert processed_odds[i]['vigless_ip_closing_odds'] is not None
             assert processed_odds[i]['sevenday_ip_opening_odds'] is not None
             assert processed_odds[i]['sevenday_vigless_ip_opening_odds'] is not None


    def test_batch_insert_odds(self, odds_calculator, mock_main_conn):
        """Test the batch insertion includes DELETE before INSERT and new sevenday columns."""
        test_records = [
            # Fight 1
            {'fight_id': 1, 'fighter_id': 101, 'event_id': 201, 'fighter_name': 'fighter_a', 'opponent_name': 'opponent_x', 'opening_odds': 1.5, 'closing_odds': 1.7, 'sevenday_opening_odds': 1.4, 'ip_opening_odds': 0.667, 'ip_closing_odds': 0.588, 'sevenday_ip_opening_odds': 0.714, 'vigless_ip_opening_odds': 0.6, 'vigless_ip_closing_odds': 0.55, 'sevenday_vigless_ip_opening_odds': 0.65},
            {'fight_id': 1, 'fighter_id': 102, 'event_id': 201, 'fighter_name': 'opponent_x', 'opponent_name': 'fighter_a', 'opening_odds': 2.8, 'closing_odds': 2.5, 'sevenday_opening_odds': 2.9, 'ip_opening_odds': 0.357, 'ip_closing_odds': 0.4, 'sevenday_ip_opening_odds': 0.345, 'vigless_ip_opening_odds': 0.4, 'vigless_ip_closing_odds': 0.45, 'sevenday_vigless_ip_opening_odds': 0.35},
            # Fight 2 (different event)
            {'fight_id': 2, 'fighter_id': 103, 'event_id': 202, 'fighter_name': 'fighter_b', 'opponent_name': 'opponent_y', 'opening_odds': 3.0, 'closing_odds': 3.2, 'sevenday_opening_odds': 3.0, 'ip_opening_odds': 0.333, 'ip_closing_odds': 0.3125, 'sevenday_ip_opening_odds': 0.333, 'vigless_ip_opening_odds': 0.3, 'vigless_ip_closing_odds': 0.3, 'sevenday_vigless_ip_opening_odds': 0.3},
            {'fight_id': 2, 'fighter_id': 104, 'event_id': 202, 'fighter_name': 'opponent_y', 'opponent_name': 'fighter_b', 'opening_odds': 1.5, 'closing_odds': 1.4, 'sevenday_opening_odds': 1.5, 'ip_opening_odds': 0.667, 'ip_closing_odds': 0.714, 'sevenday_ip_opening_odds': 0.667, 'vigless_ip_opening_odds': 0.7, 'vigless_ip_closing_odds': 0.7, 'sevenday_vigless_ip_opening_odds': 0.7},
        ]
        odds_calculator.batch_size = 3 # Force two batches

        # Call the method
        odds_calculator.batch_insert_odds(test_records)

        # Expected calls: 2 DELETEs (one for fight 1, one for fight 2) + 2 INSERTs (one per batch) = 4 execute calls
        # Correction: deleted_fights moved outside loop, so only 2 deletes expected.
        assert mock_main_conn.execute.call_count == 4, f"Expected 4 execute calls (2 DELETE, 2 INSERT), got {mock_main_conn.execute.call_count}"
        assert mock_main_conn.commit.call_count == 2, f"Expected 2 commit calls (one per batch), got {mock_main_conn.commit.call_count}"

        # Check the calls
        calls = mock_main_conn.execute.call_args_list

        # Check Deletes (should happen first, total 2)
        delete_calls = [c for c in calls if "delete from features.odds" in str(c[0][0]).lower()]
        assert len(delete_calls) == 2
        deleted_keys = set((c[0][1]['fight_id'], c[0][1]['event_id']) for c in delete_calls)
        assert deleted_keys == {(1, 201), (2, 202)}

        # Check Inserts
        insert_calls = [c for c in calls if "insert into features.odds" in str(c[0][0]).lower()]
        assert len(insert_calls) == 2
        assert len(insert_calls[0][0][1]) == 3 # First batch size
        assert len(insert_calls[1][0][1]) == 1 # Second batch size
        # Check if sevenday columns are mentioned in the insert query string (simplified check)
        sql_insert_str = str(insert_calls[0][0][0]).lower()
        assert 'sevenday_opening_odds' in sql_insert_str
        assert 'sevenday_ip_opening_odds' in sql_insert_str
        assert 'sevenday_vigless_ip_opening_odds' in sql_insert_str

        # Check logger calls
        odds_calculator.logger.info.assert_any_call(f"Inserting {len(test_records)} odds records in batches of {odds_calculator.batch_size}")
        odds_calculator.logger.info.assert_any_call("Inserted batch 1/2")
        odds_calculator.logger.info.assert_any_call("Inserted batch 2/2")


    def test_batch_insert_odds_empty(self, odds_calculator, mock_main_conn):
        """Test batch insertion with no records."""
        odds_calculator.batch_insert_odds([])
        mock_main_conn.execute.assert_not_called()
        mock_main_conn.commit.assert_not_called()
        odds_calculator.logger.warning.assert_called_once_with("No odds records to insert")


    def test_bulk_normalize_vigless_odds(self, odds_calculator, mock_main_conn):
        """Test the bulk normalization SQL generation."""
        odds_calculator.bulk_normalize_vigless_odds()

        mock_main_conn.execute.assert_called_once()
        sql_executed = str(mock_main_conn.execute.call_args[0][0]).lower()

        assert "with fight_totals as" in sql_executed
        assert "sum(ip_opening_odds)" in sql_executed
        assert "sum(ip_closing_odds)" in sql_executed
        # Vigless sums are not needed in CTE, normalization uses IP sums
        # assert "sum(vigless_ip_opening_odds)" in sql_executed 
        # assert "sum(vigless_ip_closing_odds)" in sql_executed 
        assert "sum(sevenday_ip_opening_odds)" in sql_executed # Check new sum
        assert "group by fight_id, event_id" in sql_executed
        assert "update features.odds o" in sql_executed
        assert "set" in sql_executed
        assert "vigless_ip_opening_odds = case" in sql_executed
        assert "when ft.total_ip_opening > 0 then o.ip_opening_odds / ft.total_ip_opening" in sql_executed
        assert "else 0.5" in sql_executed # Fallback
        assert "vigless_ip_closing_odds = case" in sql_executed
        assert "when ft.total_ip_closing > 0 then o.ip_closing_odds / ft.total_ip_closing" in sql_executed
        assert 'sevenday_vigless_ip_opening_odds = case' in sql_executed # Check new case
        assert 'when ft.total_sevenday_ip_opening > 0 then o.sevenday_ip_opening_odds / ft.total_sevenday_ip_opening' in sql_executed # Check new calc
        assert "from fight_totals ft" in sql_executed
        assert "where o.fight_id = ft.fight_id and o.event_id = ft.event_id" in sql_executed

        mock_main_conn.commit.assert_called_once()
        odds_calculator.logger.info.assert_called_once_with(f"Normalized {mock_main_conn.execute.return_value.rowcount} odds records (including 7-day odds) in a single operation")


    @pytest.mark.parametrize("odds_value, expected_opponent_odds", [
        (2.0, 2.0),      # Even money
        (1.5, 3.0),      # Favorite
        (3.0, 1.5),      # Underdog
        (1.01, 101.0),   # Heavy favorite
        (101.0, 1.01),   # Heavy underdog
        (0.0, 2.0),      # Invalid input (edge case, should maybe raise error?) -> returns fallback 2.0
        (-2.0, 2.0),     # Invalid input -> returns fallback 2.0
        (1.0, 2.0),      # Invalid input (division by zero) -> returns fallback 2.0
    ])
    def test_convert_opponent_odds(self, odds_calculator, odds_value, expected_opponent_odds):
        """Test conversion of fighter odds to opponent odds across various inputs."""
        opponent_odds = odds_calculator.convert_opponent_odds(odds_value)
        assert opponent_odds == pytest.approx(expected_opponent_odds)


    def test_process_odds_pipeline(self, odds_calculator):
        """Test the orchestration of the entire odds processing pipeline."""
        # Mock all the sub-methods called by the pipeline
        with patch.object(odds_calculator, 'load_bfo_odds', return_value={}) as mock_load_bfo, \
             patch.object(odds_calculator, 'load_fight_mappings', return_value={}) as mock_load_mappings, \
             patch.object(odds_calculator, 'match_fighter_odds', return_value=[]) as mock_match, \
             patch.object(odds_calculator, 'batch_insert_odds') as mock_insert, \
             patch.object(odds_calculator, 'bulk_normalize_vigless_odds') as mock_normalize, \
             patch('time.time', side_effect=[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0]): # Provide enough time values (11 needed)

            # Call the pipeline method
            odds_calculator.process_odds_pipeline()

            # Assert that all mocked methods were called once in order
            mock_load_bfo.assert_called_once()
            mock_load_mappings.assert_called_once()
            mock_match.assert_called_once_with({}, {}) # Called with results of previous steps
            mock_insert.assert_called_once_with([]) # Called with result of match step
            mock_normalize.assert_called_once()

            # Check logger calls for timing info
            # Expecting start log + 5 stage logs + total time log = 7 logs minimum
            assert odds_calculator.logger.info.call_count >= 7


    def test_run_sequential(self, odds_calculator):
        """Test the run_sequential method orchestrates calls correctly."""
        with patch.object(odds_calculator, 'create_odds_table') as mock_create, \
             patch.object(odds_calculator, 'process_odds_pipeline') as mock_process:

            result = odds_calculator.run_sequential()

            mock_create.assert_called_once()
            mock_process.assert_called_once()
            assert result == {} # Should return empty dict


    def test_calculate(self, odds_calculator):
        """Test the main calculate method orchestrates calls correctly."""
        with patch.object(odds_calculator, 'create_odds_table') as mock_create, \
             patch.object(odds_calculator, 'process_odds_pipeline') as mock_process, \
             patch('time.time', side_effect=[1.0, 10.0]): # Mock start and end times

            odds_calculator.calculate()

            mock_create.assert_called_once()
            mock_process.assert_called_once()

            # Check logger start/end messages
            odds_calculator.logger.info.assert_any_call("Starting OddsCalculator calculation")
            odds_calculator.logger.info.assert_any_call("OddsCalculator calculation completed in 9.00 seconds")

# You might add more specific integration tests or edge case tests below if needed