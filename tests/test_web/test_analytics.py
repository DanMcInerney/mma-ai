import json
import sys
from types import ModuleType, SimpleNamespace

import pytest
import pandas as pd

from libs.web.analytics import (
    _execute_read_only_query,
    _run_python_analysis,
    _validate_python_analysis_code,
    database_context,
    is_read_only_sql,
    parse_llm_json,
    run_analytics,
)
from libs.web.analytics_prompt import analytics_system_prompt


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
    assert "chart" not in result
    assert result["charts"]


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


def test_python_analytics_runner_uses_guarded_postgres_helper(monkeypatch):
    captured = {}

    def fake_database_query(sql, max_rows):
        captured["sql"] = sql
        captured["max_rows"] = max_rows
        return pd.DataFrame(
            [
                {"weightclass": "Lightweight", "fights": 28},
                {"weightclass": "Welterweight", "fights": 31},
            ]
        )

    monkeypatch.setattr("libs.web.analytics._execute_database_only_read_only_query", fake_database_query)

    result = _run_python_analysis(
        "\n".join(
            [
                "df = run_sql('select weightclass, fights from features.fight_mapping', limit=50)",
                "set_answer('Weight class summary')",
                "set_rows(df)",
                "fig = px.bar(df, x='weightclass', y='fights', title='Fights by weight class')",
                "add_chart(fig)",
            ]
        ),
        max_rows=100,
    )

    assert captured == {"sql": "select weightclass, fights from features.fight_mapping", "max_rows": 50}
    assert result["answer"] == "Weight class summary"
    assert result["sql"] == "select weightclass, fights from features.fight_mapping"
    assert result["rows"][0]["weightclass"] == "Lightweight"
    assert result["columns"] == ["weightclass", "fights"]
    assert len(result["charts"]) == 1
    assert result["charts"][0]["layout"]["title"]["text"] == "Fights by weight class"


def test_python_analytics_runner_normalizes_escaped_llm_newlines(monkeypatch):
    monkeypatch.setattr(
        "libs.web.analytics._execute_database_only_read_only_query",
        lambda _sql, _max_rows: pd.DataFrame([{"weightclass": "Lightweight", "fights": 28}]),
    )

    result = _run_python_analysis(
        (
            "df = run_sql('select weightclass, fights from features.fight_mapping', limit=50)\\n"
            "set_answer('Escaped newlines work')\\n"
            "set_rows(df)\\n"
            "add_chart(px.bar(df, x='weightclass', y='fights', title='Escaped chart'))"
        ),
        max_rows=100,
    )

    assert result["answer"] == "Escaped newlines work"
    assert result["rows"] == [{"weightclass": "Lightweight", "fights": 28}]
    assert result["charts"][0]["layout"]["title"]["text"] == "Escaped chart"


def test_python_analytics_runner_rejects_unsafe_code():
    with pytest.raises(ValueError, match="import"):
        _validate_python_analysis_code("import os\nset_answer('bad')")

    with pytest.raises(ValueError, match="open"):
        _validate_python_analysis_code("open('secret.txt').read()")

    with pytest.raises(ValueError, match="read_csv"):
        _validate_python_analysis_code("pd.read_csv('training_data.csv')")


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
                "python": (
                    "df = run_sql('select fighter1_name, wins from features.win_summary', limit=100)\n"
                    "set_answer('AI summary')\n"
                    "set_rows(df)\n"
                    "add_chart(px.bar(df, x='fighter1_name', y='wins', title='Wins by fighter'))"
                ),
            }
        ),
    )
    monkeypatch.setattr(
        "libs.web.analytics._execute_database_only_read_only_query",
        lambda _sql, _max_rows: pd.DataFrame([{"fighter1_name": "a", "wins": 3}]),
    )

    result = run_analytics("Show wins")

    assert captured["api_key"] == "google-key"
    assert captured["model_name"] == "gemini-1.5-pro"
    assert result["answer"] == "AI summary"
    assert result["rows"] == [{"fighter1_name": "a", "wins": 3}]
    assert "chart" not in result
    assert result["charts"]


def test_analytics_llm_prompt_includes_copyable_system_prompt(monkeypatch, tmp_path):
    clear_llm_env(monkeypatch)
    monkeypatch.setenv("GOOGLE_API_KEY", "google-key")
    monkeypatch.setenv("MMA_AI_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("DATABASE_URL", "postgresql://postgres:postgres@127.0.0.1:1/missing")
    pd.DataFrame([{"fighter1_name": "a", "wins": 3}]).to_csv(tmp_path / "training_data.csv", index=False)
    captured = install_fake_gemini(
        monkeypatch,
        json.dumps(
            {
                "python": (
                    "df = run_sql('select fighter1_name, wins from features.win_summary', limit=100)\n"
                    "set_answer('AI summary')\n"
                    "set_rows(df)"
                )
            }
        ),
    )
    monkeypatch.setattr(
        "libs.web.analytics._execute_database_only_read_only_query",
        lambda _sql, _max_rows: pd.DataFrame([{"fighter1_name": "a", "wins": 3}]),
    )

    run_analytics("Show wins")

    payload = json.loads(captured["prompt"])
    assert payload["system_prompt"] == analytics_system_prompt()
    assert "_adjperf" in payload["system_prompt"]
    assert "features.odds" in payload["system_prompt"]


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
                                    "python": (
                                        "df = run_sql('select fighter1_name, wins from features.win_summary', limit=100)\n"
                                        "set_answer('AI summary')\n"
                                        "set_rows(df)\n"
                                        "add_chart(px.bar(df, x='fighter1_name', y='wins', title='Wins by fighter'))"
                                    ),
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
    monkeypatch.setattr(
        "libs.web.analytics._execute_database_only_read_only_query",
        lambda _sql, _max_rows: pd.DataFrame([{"fighter1_name": "a", "wins": 3}]),
    )

    result = run_analytics("Show wins")

    assert captured["url"] == "https://llm.example/v1/chat/completions"
    assert captured["headers"]["Authorization"] == "Bearer openai-key"
    assert captured["json"]["model"] == "analytics-model"
    assert captured["json"]["response_format"] == {"type": "json_object"}
    assert result["answer"] == "AI summary"
    assert result["rows"] == [{"fighter1_name": "a", "wins": 3}]


def test_run_analytics_accepts_llm_plotly_figure_specs(monkeypatch, tmp_path):
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
                                    "python": (
                                        "df = run_sql('select fighter1_name, wins from features.win_summary', limit=100)\n"
                                        "set_answer('AI chart')\n"
                                        "set_rows(df)\n"
                                        "add_chart({'data': [{'type': 'bar', 'x': ['a'], 'y': [3], 'name': 'Wins'}], "
                                        "'layout': {'title': 'Wins by fighter'}})"
                                    ),
                                }
                            )
                        }
                    }
                ]
            }

    monkeypatch.setattr("libs.web.llm.requests.post", lambda *_args, **_kwargs: FakeResponse())
    monkeypatch.setattr(
        "libs.web.analytics._execute_database_only_read_only_query",
        lambda _sql, _max_rows: pd.DataFrame([{"fighter1_name": "a", "wins": 3}]),
    )

    result = run_analytics("Show wins")

    assert "chart" not in result
    assert result["charts"][0]["data"] == [{"type": "bar", "x": ["a"], "y": [3], "name": "Wins"}]
    assert result["charts"][0]["layout"]["title"] == "Wins by fighter"
    assert result["charts"][0]["layout"]["template"] == "plotly_white"


def test_run_analytics_accepts_llm_charts_array(monkeypatch, tmp_path):
    clear_llm_env(monkeypatch)
    monkeypatch.setenv("OPENAI_API_KEY", "openai-key")
    monkeypatch.setenv("MMA_AI_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("DATABASE_URL", "postgresql://postgres:postgres@127.0.0.1:1/missing")
    pd.DataFrame(
        [
            {"weightclass": "Lightweight", "avg_sig_str_acc": 0.48, "fight_count": 28},
            {"weightclass": "Welterweight", "avg_sig_str_acc": 0.44, "fight_count": 31},
        ]
    ).to_csv(tmp_path / "training_data.csv", index=False)

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
                                    "python": (
                                        "df = run_sql('select weightclass, avg_sig_str_acc, fight_count "
                                        "from features.sig_str_summary', limit=100)\n"
                                        "set_answer('Two coordinated charts')\n"
                                        "set_rows(df)\n"
                                        "add_chart(px.bar(df, x='weightclass', y='avg_sig_str_acc', "
                                        "title='Significant strike accuracy by weight class'))\n"
                                        "add_chart(px.scatter(df, x='fight_count', y='avg_sig_str_acc', "
                                        "title='Volume versus accuracy'))"
                                    ),
                                }
                            )
                        }
                    }
                ]
            }

    monkeypatch.setattr("libs.web.llm.requests.post", lambda *_args, **_kwargs: FakeResponse())
    monkeypatch.setattr(
        "libs.web.analytics._execute_database_only_read_only_query",
        lambda _sql, _max_rows: pd.DataFrame(
            [
                {"weightclass": "Lightweight", "avg_sig_str_acc": 0.48, "fight_count": 28},
                {"weightclass": "Welterweight", "avg_sig_str_acc": 0.44, "fight_count": 31},
            ]
        ),
    )

    result = run_analytics("Compare striking accuracy and volume")

    assert result["answer"] == "Two coordinated charts"
    assert "chart" not in result
    assert len(result["charts"]) == 2
    assert result["charts"][0]["layout"]["title"]["text"] == "Significant strike accuracy by weight class"
    assert result["charts"][1]["layout"]["title"]["text"] == "Volume versus accuracy"
    assert result["charts"][1]["layout"]["paper_bgcolor"] == "#ffffff"


def test_run_analytics_requires_llm_python_script(monkeypatch, tmp_path):
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
            }
        ),
    )

    with pytest.raises(ValueError, match="python script"):
        run_analytics("Show wins")


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
        {"answer": "ok", "sql": "select * from training_data", "charts": [{"type": "bar", "x": "fighter1_name", "y": "y_true"}]}
        ```"""
    )

    assert parsed["sql"] == "select * from training_data"


def test_parse_llm_json_accepts_wrapped_json_object():
    parsed = parse_llm_json(
        'Here is the analysis request: {"answer": "ok", "sql": "select count(*) as fights from training_data"}'
    )

    assert parsed == {"answer": "ok", "sql": "select count(*) as fights from training_data"}
