"""Read-only AI analytics helpers for finalized MMA data."""

from __future__ import annotations

import json
import re
import sqlite3
from typing import Any

import pandas as pd
import plotly.express as px
from sqlalchemy import create_engine, inspect, text

from libs.paths import PROJECT_ROOT, data_file, database_url
from libs.web.llm import llm_config, llm_config_hint, llm_generate_text


FORBIDDEN_SQL = re.compile(
    r"\b(alter|analyze|attach|checkpoint|comment|copy|create|delete|detach|drop|execute|grant|insert|merge|refresh|reindex|replace|reset|revoke|set|truncate|update|vacuum)\b",
    re.IGNORECASE,
)


def is_read_only_sql(sql: str) -> bool:
    stripped = sql.strip().rstrip(";")
    if not stripped:
        return False
    if ";" in stripped:
        return False
    if not re.match(r"^(select|with)\b", stripped, re.IGNORECASE):
        return False
    return FORBIDDEN_SQL.search(stripped) is None


def database_context(max_columns_per_table: int = 80) -> dict[str, Any]:
    """Return compact schema context for prompts and diagnostics."""
    try:
        engine = create_engine(database_url())
        inspector = inspect(engine)
        schemas = [schema for schema in inspector.get_schema_names() if schema in {"features", "model_data", "public"}]
        tables = []
        for schema in schemas:
            for table_name in inspector.get_table_names(schema=schema):
                columns = inspector.get_columns(table_name, schema=schema)[:max_columns_per_table]
                tables.append(
                    {
                        "schema": schema,
                        "table": table_name,
                        "columns": [{"name": column["name"], "type": str(column["type"])} for column in columns],
                    }
                )
        return {"source": "database", "tables": tables}
    except Exception as exc:
        tables = _csv_table_context()
        if not tables:
            return {"source": "unavailable", "warning": str(exc), "tables": []}
        return {
            "source": "finalized_csvs",
            "warning": str(exc),
            "tables": tables,
        }


def run_analytics(question: str, sql: str | None = None, max_rows: int = 100) -> dict[str, Any]:
    generated = None
    config = llm_config()
    if sql is None and config and config.is_configured:
        generated = _ask_llm(question)
        sql = generated.get("sql")

    if not sql:
        context = database_context()
        return {
            "answer": f"{llm_config_hint()} Or provide a read-only SQL query to execute analytics from the dashboard.",
            "schema_context": context,
            "sql": None,
            "rows": [],
            "chart": None,
        }

    if not is_read_only_sql(sql):
        raise ValueError("Only a single read-only SELECT or WITH query is allowed.")

    df = _execute_read_only_query(sql, max_rows)
    raw_chart_spec = (generated or {}).get("chart")
    chart_spec = raw_chart_spec if isinstance(raw_chart_spec, dict) else _default_chart_spec(df)
    chart = _build_chart(df, chart_spec)
    return {
        "answer": (generated or {}).get("answer", "Query executed."),
        "sql": sql,
        "rows": df.to_dict(orient="records"),
        "columns": list(df.columns),
        "chart": chart,
    }


def _ask_llm(question: str) -> dict[str, Any]:
    prompt = {
        "task": "Return strict JSON with answer, sql, and optional chart keys for read-only MMA analytics.",
        "question": question,
        "schema_context": database_context(),
        "project_guidance": _agent_guidance_excerpt(),
        "constraints": [
            "SQL must be a single SELECT or WITH query.",
            "Prefer features schema tables and finalized model_data/training outputs when Postgres is available.",
            "When schema_context.source is finalized_csvs, query table names such as training_data, training_data_dec, or prediction_data directly.",
            "Never mutate data.",
            "For charts, either return {type, x, y} using result column names, or a Plotly JSON object with data and layout.",
            "Return JSON only, with no Markdown fences.",
        ],
    }
    return parse_llm_json(llm_generate_text(prompt, json_mode=True))


def parse_llm_json(text_response: str) -> dict[str, Any]:
    """Parse JSON from LLM output, tolerating common Markdown wrappers."""
    stripped = text_response.strip()
    fenced = re.match(r"^```(?:json)?\s*(.*?)\s*```$", stripped, flags=re.IGNORECASE | re.DOTALL)
    if fenced:
        stripped = fenced.group(1).strip()

    try:
        parsed = json.loads(stripped)
    except json.JSONDecodeError:
        start = stripped.find("{")
        end = stripped.rfind("}")
        if start < 0 or end <= start:
            raise
        parsed = json.loads(stripped[start : end + 1])

    if not isinstance(parsed, dict):
        raise ValueError("Analytics LLM response must be a JSON object.")
    return parsed


def _execute_read_only_query(sql: str, max_rows: int) -> pd.DataFrame:
    stripped_sql = sql.strip().rstrip(";")
    try:
        engine = create_engine(database_url())
        bounded_sql = f"SELECT * FROM ({stripped_sql}) AS analytics_query LIMIT :max_rows"
        return pd.read_sql(text(bounded_sql), engine, params={"max_rows": max_rows})
    except Exception as db_exc:
        csv_tables = _load_csv_tables()
        if not csv_tables:
            raise RuntimeError(f"Database query failed and no finalized CSV fallback is available: {db_exc}") from db_exc

        with sqlite3.connect(":memory:") as conn:
            for table_name, df in csv_tables.items():
                df.to_sql(table_name, conn, index=False, if_exists="replace")
            try:
                return pd.read_sql_query(
                    f"SELECT * FROM ({stripped_sql}) AS analytics_query LIMIT ?",
                    conn,
                    params=(max_rows,),
                )
            except Exception as csv_exc:
                table_names = ", ".join(sorted(csv_tables))
                raise RuntimeError(
                    f"Database query failed, and CSV fallback query failed. Available CSV tables: {table_names}. CSV error: {csv_exc}"
                ) from csv_exc


def _load_csv_tables() -> dict[str, pd.DataFrame]:
    tables = {}
    for table_name, filename in _finalized_csv_sources().items():
        path = data_file(filename)
        if path.exists() and path.stat().st_size > 0:
            tables[table_name] = pd.read_csv(path)
    return tables


def _csv_table_context() -> list[dict[str, Any]]:
    tables = []
    for table_name, filename in _finalized_csv_sources().items():
        path = data_file(filename)
        if not path.exists() or path.stat().st_size == 0:
            continue
        df = pd.read_csv(path, nrows=5)
        tables.append(
            {
                "schema": "csv",
                "table": table_name,
                "path": str(path),
                "columns": [{"name": column, "type": str(dtype)} for column, dtype in df.dtypes.items()],
            }
        )
    return tables


def _finalized_csv_sources() -> dict[str, str]:
    return {
        "training_data": "training_data.csv",
        "training_data_dec": "training_data_dec.csv",
        "prediction_data": "prediction_data.csv",
    }


def _agent_guidance_excerpt() -> str:
    excerpts = []
    for filename in ("AGENTS.md", "CLAUDE.md"):
        path = PROJECT_ROOT / filename
        if path.exists():
            excerpts.append(path.read_text(encoding="utf-8", errors="replace")[:6000])
    return "\n\n".join(excerpts)


def _default_chart_spec(df: pd.DataFrame) -> dict[str, str] | None:
    if df.empty or len(df.columns) < 2:
        return None
    numeric = [column for column in df.columns if pd.api.types.is_numeric_dtype(df[column])]
    if not numeric:
        return None
    x_column = next((column for column in df.columns if column not in numeric), df.columns[0])
    return {"type": "bar", "x": x_column, "y": numeric[0]}


def _build_chart(df: pd.DataFrame, spec: dict[str, Any] | None) -> dict[str, Any] | None:
    if not spec:
        return None
    if isinstance(spec.get("data"), list):
        return {
            "data": spec["data"],
            "layout": {
                "template": "plotly_white",
                "margin": {"l": 36, "r": 16, "t": 28, "b": 36},
                **(spec.get("layout") if isinstance(spec.get("layout"), dict) else {}),
            },
        }
    chart_type = spec.get("type", "bar")
    x_column = spec.get("x")
    y_column = spec.get("y")
    if x_column not in df.columns or (y_column and y_column not in df.columns):
        return None

    if chart_type == "line" and y_column:
        fig = px.line(df, x=x_column, y=y_column)
    elif chart_type == "scatter" and y_column:
        fig = px.scatter(df, x=x_column, y=y_column)
    elif chart_type == "histogram":
        fig = px.histogram(df, x=x_column)
    elif y_column:
        fig = px.bar(df, x=x_column, y=y_column)
    else:
        return None

    fig.update_layout(template="plotly_white", margin={"l": 36, "r": 16, "t": 28, "b": 36})
    return json.loads(fig.to_json())
