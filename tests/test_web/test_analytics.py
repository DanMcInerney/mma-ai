import json
import sys
from types import ModuleType, SimpleNamespace

import pytest
import pandas as pd

from libs.web.analytics import _execute_read_only_query, database_context, is_read_only_sql, parse_llm_json, run_analytics


def clear_llm_env(monkeypatch):
    for name in [
        "LLM_PROVIDER",
        "LLM_MODEL",
        "LLM_API_KEY",
        "LLM_BASE_URL",
        "OPENAI_API_KEY",
        "ANTHROPIC_API_KEY",
        "XAI_API_KEY",
        "GROK_API_KEY",
        "OPENROUTER_API_KEY",
        "DEEPSEEK_API_KEY",
        "MISTRAL_API_KEY",
        "TOGETHER_API_KEY",
        "PERPLEXITY_API_KEY",
        "GEMINI_API_KEY",
        "GOOGLE_API_KEY",
    ]:
        monkeypatch.delenv(name, raising=False)


def install_fake_gemini(monkeypatch, response_text: str) -> dict:
    captured = {}
    google_module = ModuleType("google")
    genai_module = ModuleType("google.generativeai")

    def configure(api_key):
        captured["api_key"] = api_key

    class FakeModel:
        def __init__(self, model_name):
            captured["model_name"] = model_name

        def generate_content(self, prompt):
            captured["prompt"] = prompt
            return SimpleNamespace(text=response_text)

    genai_module.configure = configure
    genai_module.GenerativeModel = FakeModel
    google_module.generativeai = genai_module
    monkeypatch.setitem(sys.modules, "google", google_module)
    monkeypatch.setitem(sys.modules, "google.generativeai", genai_module)
    return captured


@pytest.mark.parametrize(
    "sql",
    [
        "select * from features.fight_stats_derived",
        "WITH recent AS (SELECT 1 AS value) SELECT * FROM recent",
        " select fighter_name from features.fighter_mapping limit 5; ",
    ],
)
def test_is_read_only_sql_accepts_selects(sql):
    assert is_read_only_sql(sql)


@pytest.mark.parametrize(
    "sql",
    [
        "",
        "update features.fight_stats_fe set win = 1",
        "select * from x; drop table x",
        "delete from features.fight_stats_fe",
        "vacuum",
    ],
)
def test_is_read_only_sql_rejects_mutations(sql):
    assert not is_read_only_sql(sql)


def test_run_analytics_without_sql_returns_context(monkeypatch, tmp_path):
    clear_llm_env(monkeypatch)
    monkeypatch.setenv("MMA_AI_DATA_DIR", str(tmp_path))
    result = run_analytics("What data is available?")
    assert result["sql"] is None
    assert "schema_context" in result
    assert "GOOGLE_API_KEY" in result["answer"]


def test_run_analytics_executes_against_finalized_csv_fallback(monkeypatch, tmp_path):
    clear_llm_env(monkeypatch)
    monkeypatch.setenv("MMA_AI_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("DATABASE_URL", "postgresql://postgres:postgres@127.0.0.1:1/missing")
    pd.DataFrame(
        [
            {"fighter1_name": "a", "fighter2_name": "b", "y_true": 1, "feature_diff": 0.4},
            {"fighter1_name": "c", "fighter2_name": "d", "y_true": 0, "feature_diff": -0.2},
        ]
    ).to_csv(tmp_path / "training_data.csv", index=False)

    result = run_analytics(
        "Show winners",
        sql="select fighter1_name, y_true, feature_diff from training_data order by feature_diff desc",
        max_rows=1,
    )

    assert result["rows"] == [{"fighter1_name": "a", "y_true": 1, "feature_diff": 0.4}]
    assert result["chart"] is not None


def test_database_analytics_runs_inside_postgres_read_only_transaction(monkeypatch):
    executed = []

    class FakeTransaction:
        def __enter__(self):
            executed.append("BEGIN")
            return self

        def __exit__(self, *_args):
            executed.append("END")
            return False

    class FakeConnection:
        dialect = SimpleNamespace(name="postgresql")

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def begin(self):
            return FakeTransaction()

        def execute(self, statement, params=None):
            executed.append((str(statement), params))

    class FakeEngine:
        def __init__(self):
            self.disposed = False

        def connect(self):
            return FakeConnection()

        def dispose(self):
            self.disposed = True
            executed.append("DISPOSE")

    fake_engine = FakeEngine()

    def fake_read_sql(statement, connection, params):
        executed.append(("READ", str(statement), params, connection.dialect.name))
        return pd.DataFrame([{"rows": 1}])

    monkeypatch.setattr("libs.web.analytics.database_url", lambda: "postgresql://postgres@localhost:5432/mma-ai")
    monkeypatch.setattr("libs.web.analytics.create_engine", lambda _url: fake_engine)
    monkeypatch.setattr("libs.web.analytics.pd.read_sql", fake_read_sql)

    result = _execute_read_only_query("select count(*) as rows from features.fight_mapping", max_rows=5)

    assert result.to_dict(orient="records") == [{"rows": 1}]
    assert executed[:3] == [
        "BEGIN",
        ("SET TRANSACTION READ ONLY", None),
        ("SET LOCAL statement_timeout = 30000", None),
    ]
    assert executed[3] == (
        "READ",
        "SELECT * FROM (select count(*) as rows from features.fight_mapping) AS analytics_query LIMIT :max_rows",
        {"max_rows": 5},
        "postgresql",
    )
    assert executed[-2:] == ["END", "DISPOSE"]


def test_run_analytics_uses_google_api_key_alias(monkeypatch, tmp_path):
    clear_llm_env(monkeypatch)
    monkeypatch.setenv("GOOGLE_API_KEY", "google-key")
    monkeypatch.setenv("MMA_AI_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("DATABASE_URL", "postgresql://postgres:postgres@127.0.0.1:1/missing")
    pd.DataFrame([{"fighter1_name": "a", "wins": 3}]).to_csv(tmp_path / "training_data.csv", index=False)
    captured = install_fake_gemini(
        monkeypatch,
        json.dumps(
            {
                "answer": "AI summary",
                "sql": "select fighter1_name, wins from training_data",
                "chart": {"type": "bar", "x": "fighter1_name", "y": "wins"},
            }
        ),
    )

    result = run_analytics("Show wins")

    assert captured["api_key"] == "google-key"
    assert captured["model_name"] == "gemini-1.5-pro"
    assert result["answer"] == "AI summary"
    assert result["rows"] == [{"fighter1_name": "a", "wins": 3}]
    assert result["chart"] is not None


def test_run_analytics_uses_openai_compatible_llm(monkeypatch, tmp_path):
    clear_llm_env(monkeypatch)
    monkeypatch.setenv("OPENAI_API_KEY", "openai-key")
    monkeypatch.setenv("LLM_MODEL", "analytics-model")
    monkeypatch.setenv("LLM_BASE_URL", "https://llm.example/v1")
    monkeypatch.setenv("MMA_AI_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("DATABASE_URL", "postgresql://postgres:postgres@127.0.0.1:1/missing")
    pd.DataFrame([{"fighter1_name": "a", "wins": 3}]).to_csv(tmp_path / "training_data.csv", index=False)
    captured = {}

    class FakeResponse:
        def raise_for_status(self):
            return None

        def json(self):
            return {
                "choices": [
                    {
                        "message": {
                            "content": json.dumps(
                                {
                                    "answer": "AI summary",
                                    "sql": "select fighter1_name, wins from training_data",
                                    "chart": {"type": "bar", "x": "fighter1_name", "y": "wins"},
                                }
                            )
                        }
                    }
                ]
            }

    def fake_post(url, headers, json, timeout):
        captured.update({"url": url, "headers": headers, "json": json, "timeout": timeout})
        return FakeResponse()

    monkeypatch.setattr("libs.web.llm.requests.post", fake_post)

    result = run_analytics("Show wins")

    assert captured["url"] == "https://llm.example/v1/chat/completions"
    assert captured["headers"]["Authorization"] == "Bearer openai-key"
    assert captured["json"]["model"] == "analytics-model"
    assert captured["json"]["response_format"] == {"type": "json_object"}
    assert result["answer"] == "AI summary"
    assert result["rows"] == [{"fighter1_name": "a", "wins": 3}]


def test_run_analytics_accepts_llm_plotly_figure_spec(monkeypatch, tmp_path):
    clear_llm_env(monkeypatch)
    monkeypatch.setenv("OPENAI_API_KEY", "openai-key")
    monkeypatch.setenv("MMA_AI_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("DATABASE_URL", "postgresql://postgres:postgres@127.0.0.1:1/missing")
    pd.DataFrame([{"fighter1_name": "a", "wins": 3}]).to_csv(tmp_path / "training_data.csv", index=False)

    class FakeResponse:
        def raise_for_status(self):
            return None

        def json(self):
            return {
                "choices": [
                    {
                        "message": {
                            "content": json.dumps(
                                {
                                    "answer": "AI chart",
                                    "sql": "select fighter1_name, wins from training_data",
                                    "chart": {
                                        "data": [{"type": "bar", "x": ["a"], "y": [3], "name": "Wins"}],
                                        "layout": {"title": "Wins by fighter"},
                                    },
                                }
                            )
                        }
                    }
                ]
            }

    monkeypatch.setattr("libs.web.llm.requests.post", lambda *_args, **_kwargs: FakeResponse())

    result = run_analytics("Show wins")

    assert result["chart"]["data"] == [{"type": "bar", "x": ["a"], "y": [3], "name": "Wins"}]
    assert result["chart"]["layout"]["title"] == "Wins by fighter"
    assert result["chart"]["layout"]["template"] == "plotly_white"


def test_run_analytics_falls_back_when_llm_chart_spec_is_not_an_object(monkeypatch, tmp_path):
    clear_llm_env(monkeypatch)
    monkeypatch.setenv("GOOGLE_API_KEY", "google-key")
    monkeypatch.setenv("MMA_AI_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("DATABASE_URL", "postgresql://postgres:postgres@127.0.0.1:1/missing")
    pd.DataFrame([{"fighter1_name": "a", "wins": 3}]).to_csv(tmp_path / "training_data.csv", index=False)
    install_fake_gemini(
        monkeypatch,
        json.dumps(
            {
                "answer": "AI summary",
                "sql": "select fighter1_name, wins from training_data",
                "chart": "not a chart object",
            }
        ),
    )

    result = run_analytics("Show wins")

    assert result["chart"] is not None
    assert result["chart"]["data"][0]["type"] == "bar"


def test_database_context_lists_finalized_csvs(monkeypatch, tmp_path):
    monkeypatch.setenv("MMA_AI_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("DATABASE_URL", "postgresql://postgres:postgres@127.0.0.1:1/missing")
    pd.DataFrame([{"fighter1_name": "a", "y_true": 1}]).to_csv(tmp_path / "training_data.csv", index=False)

    context = database_context()

    assert context["source"] == "finalized_csvs"
    assert context["tables"][0]["table"] == "training_data"


def test_parse_llm_json_accepts_markdown_fence():
    parsed = parse_llm_json(
        """```json
        {"answer": "ok", "sql": "select * from training_data", "chart": {"type": "bar", "x": "fighter1_name", "y": "y_true"}}
        ```"""
    )

    assert parsed["sql"] == "select * from training_data"


def test_parse_llm_json_accepts_wrapped_json_object():
    parsed = parse_llm_json(
        'Here is the analysis request: {"answer": "ok", "sql": "select count(*) as fights from training_data"}'
    )

    assert parsed == {"answer": "ok", "sql": "select count(*) as fights from training_data"}
