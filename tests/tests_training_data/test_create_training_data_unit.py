import re

import pandas as pd
import pytest

from libs.feature_store.create_training_data import CreateTrainingData


KEY_COLUMNS = ["fight_id", "fighter_id", "event_id"]


def _base_dataframe():
    return pd.DataFrame(
        {
            "fight_id": [1, 2],
            "fighter_id": [11, 22],
            "event_id": [101, 102],
            "event_date": ["2025-01-01", "2025-01-02"],
            "fighter_name": ["one", "two"],
        }
    )


def _read_sql_recorder(monkeypatch, base_df=None):
    queries = []
    base_df = _base_dataframe() if base_df is None else base_df

    def read_sql_query(query, _conn):
        query = str(query)
        queries.append(query)
        if len(queries) == 1:
            return base_df.copy()

        selected_columns = re.findall(r"\bas\s+\"?([A-Za-z_]\w*)\"?", query, re.IGNORECASE)
        feature_values = {column: list(range(len(base_df))) for column in selected_columns}
        return pd.concat(
            [base_df[KEY_COLUMNS].reset_index(drop=True), pd.DataFrame(feature_values)],
            axis=1,
        )

    monkeypatch.setattr(pd, "read_sql_query", read_sql_query)
    return queries


def test_default_query_bounds_limit_selected_columns_and_joined_tables(monkeypatch):
    queries = _read_sql_recorder(monkeypatch)
    creator = CreateTrainingData(object())
    tables = [f"stats_{index}" for index in range(8)]
    monkeypatch.setattr(creator, "_get_feature_tables", lambda: tables)
    monkeypatch.setattr(
        creator,
        "_get_table_columns",
        lambda table: [f"{table}_feature_{index}" for index in range(70)],
    )

    creator.create_training_data()

    feature_queries = queries[1:]
    assert len(feature_queries) > 1
    for query in feature_queries:
        selected_columns = re.findall(r"\bas\s+\"?([A-Za-z_]\w*)\"?", query, re.IGNORECASE)
        joined_tables = set(re.findall(r"LEFT JOIN features\.([A-Za-z_]\w*)", query))
        assert len(selected_columns) <= 200
        assert len(joined_tables) <= 6


def _configure_feature_catalog(monkeypatch, creator, table_count=5, columns_per_table=2):
    tables = [f"stats_{index}" for index in range(table_count)]
    monkeypatch.setattr(creator, "_get_feature_tables", lambda: tables)
    monkeypatch.setattr(
        creator,
        "_get_table_columns",
        lambda table: [f"{table}_feature_{index}" for index in range(columns_per_table)],
    )


def test_constructor_configuration_controls_query_bounds(monkeypatch):
    queries = _read_sql_recorder(monkeypatch)
    creator = CreateTrainingData(
        object(),
        max_feature_columns_per_query=3,
        max_feature_tables_per_query=2,
    )
    _configure_feature_catalog(monkeypatch, creator)

    creator.create_training_data()

    for query in queries[1:]:
        selected_columns = re.findall(r"\bas\s+\"?([A-Za-z_]\w*)\"?", query, re.IGNORECASE)
        joined_tables = set(re.findall(r"LEFT JOIN features\.([A-Za-z_]\w*)", query))
        assert len(selected_columns) <= 3
        assert len(joined_tables) <= 2


def test_environment_configuration_controls_query_bounds(monkeypatch):
    monkeypatch.setenv("MMA_AI_MAX_FEATURE_COLUMNS_PER_QUERY", "4")
    monkeypatch.setenv("MMA_AI_MAX_FEATURE_TABLES_PER_QUERY", "1")
    queries = _read_sql_recorder(monkeypatch)
    creator = CreateTrainingData(object())
    _configure_feature_catalog(monkeypatch, creator)

    creator.create_training_data()

    for query in queries[1:]:
        selected_columns = re.findall(r"\bas\s+\"?([A-Za-z_]\w*)\"?", query, re.IGNORECASE)
        joined_tables = set(re.findall(r"LEFT JOIN features\.([A-Za-z_]\w*)", query))
        assert len(selected_columns) <= 4
        assert len(joined_tables) <= 1


@pytest.mark.parametrize("value", [0, -1, 1.5, "2", True])
def test_constructor_configuration_rejects_invalid_bounds_before_query(monkeypatch, value):
    monkeypatch.setattr(pd, "read_sql_query", lambda *_args: pytest.fail("query ran"))

    with pytest.raises(ValueError, match="positive integer"):
        CreateTrainingData(object(), max_feature_columns_per_query=value)


@pytest.mark.parametrize("value", ["0", "-1", "1.5", "many", ""])
def test_environment_configuration_rejects_invalid_bounds_before_query(monkeypatch, value):
    monkeypatch.setenv("MMA_AI_MAX_FEATURE_TABLES_PER_QUERY", value)
    monkeypatch.setattr(pd, "read_sql_query", lambda *_args: pytest.fail("query ran"))

    with pytest.raises(ValueError, match="MMA_AI_MAX_FEATURE_TABLES_PER_QUERY.*positive integer"):
        CreateTrainingData(object())


def test_integrity_rejects_duplicate_base_keys(monkeypatch):
    base_df = _base_dataframe()
    base_df.loc[1, KEY_COLUMNS] = base_df.loc[0, KEY_COLUMNS]
    queries = _read_sql_recorder(monkeypatch, base_df)
    creator = CreateTrainingData(object())
    _configure_feature_catalog(monkeypatch, creator, table_count=1, columns_per_table=1)

    with pytest.raises(ValueError, match="base dataframe contains duplicate key rows"):
        creator.create_training_data()

    assert len(queries) == 1


def test_integrity_rejects_duplicate_chunk_keys(monkeypatch):
    base_df = _base_dataframe()
    calls = 0

    def read_sql_query(_query, _conn):
        nonlocal calls
        calls += 1
        if calls == 1:
            return base_df.copy()
        return pd.DataFrame(
            {
                "fight_id": [1, 1],
                "fighter_id": [11, 11],
                "event_id": [101, 101],
                "stats_0_feature_0": [10, 20],
            }
        )

    monkeypatch.setattr(pd, "read_sql_query", read_sql_query)
    creator = CreateTrainingData(object())
    _configure_feature_catalog(monkeypatch, creator, table_count=1, columns_per_table=1)

    with pytest.raises(ValueError, match="feature query chunk 1 contains duplicate key rows"):
        creator.create_training_data()


def test_integrity_rejects_a_row_count_changing_merge(monkeypatch):
    _read_sql_recorder(monkeypatch)
    original_merge = pd.merge

    def drop_row_after_merge(*args, **kwargs):
        return original_merge(*args, **kwargs).iloc[:-1]

    monkeypatch.setattr(pd, "merge", drop_row_after_merge)
    creator = CreateTrainingData(object())
    _configure_feature_catalog(monkeypatch, creator, table_count=1, columns_per_table=1)

    with pytest.raises(ValueError, match="feature query chunk 1 changed row count from 2 to 1"):
        creator.create_training_data()


def test_integrity_preserves_base_rows_across_valid_one_to_one_merges(monkeypatch):
    base_df = _base_dataframe()
    _read_sql_recorder(monkeypatch, base_df)
    creator = CreateTrainingData(object(), max_feature_columns_per_query=1)
    _configure_feature_catalog(monkeypatch, creator, table_count=2, columns_per_table=1)

    result = creator.create_training_data()

    assert len(result) == len(base_df)
    pd.testing.assert_frame_equal(
        result[KEY_COLUMNS].reset_index(drop=True),
        base_df[KEY_COLUMNS].reset_index(drop=True),
    )
    assert {"stats_0_feature_0", "stats_1_feature_0"}.issubset(result.columns)
