import re

import pandas as pd

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
