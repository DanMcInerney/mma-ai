"""Read-only AI analytics helpers for finalized MMA data."""

from __future__ import annotations

import ast
import json
import re
import sqlite3
from typing import Any

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sqlalchemy import create_engine, inspect, text

from libs.paths import PROJECT_ROOT, data_file, database_url
from libs.web.analytics_prompt import analytics_system_prompt
from libs.web.llm import llm_config, llm_config_hint, llm_generate_text


FORBIDDEN_SQL = re.compile(
    r"\b(alter|analyze|attach|checkpoint|comment|copy|create|delete|detach|drop|execute|grant|insert|merge|refresh|reindex|replace|reset|revoke|set|truncate|update|vacuum)\b",
    re.IGNORECASE,
)
POSTGRES_ANALYTICS_STATEMENT_TIMEOUT_MS = 30_000
MAX_ANALYTICS_CHARTS = 4
MAX_ANALYTICS_CODE_CHARS = 12_000
MAX_ANALYTICS_DISPLAY_ROWS = 200
FORBIDDEN_PYTHON_NAMES = {
    "__import__",
    "breakpoint",
    "compile",
    "dir",
    "eval",
    "exec",
    "getattr",
    "globals",
    "help",
    "input",
    "locals",
    "open",
    "setattr",
    "vars",
}
FORBIDDEN_PYTHON_REFERENCES = {
    "builtins",
    "importlib",
    "io",
    "os",
    "pathlib",
    "requests",
    "shutil",
    "socket",
    "subprocess",
    "sys",
}
FORBIDDEN_PYTHON_ATTRIBUTES = {
    "connect",
    "dispose",
    "execute",
    "read_clipboard",
    "read_csv",
    "read_excel",
    "read_feather",
    "read_fwf",
    "read_gbq",
    "read_hdf",
    "read_html",
    "read_json",
    "read_orc",
    "read_parquet",
    "read_pickle",
    "read_sql",
    "read_sql_query",
    "read_spss",
    "read_stata",
    "read_table",
    "read_xml",
    "remove",
    "rmdir",
    "run",
    "system",
    "to_csv",
    "to_excel",
    "to_feather",
    "to_json",
    "to_parquet",
    "to_pickle",
    "to_sql",
    "unlink",
    "write_html",
    "write_image",
    "write_json",
}
SAFE_BUILTINS = {
    "abs": abs,
    "all": all,
    "any": any,
    "bool": bool,
    "dict": dict,
    "enumerate": enumerate,
    "float": float,
    "int": int,
    "len": len,
    "list": list,
    "max": max,
    "min": min,
    "pow": pow,
    "range": range,
    "round": round,
    "set": set,
    "sorted": sorted,
    "str": str,
    "sum": sum,
    "tuple": tuple,
    "zip": zip,
}


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
        python_code = generated.get("python")
        if not isinstance(python_code, str) or not python_code.strip():
            raise ValueError("Analytics LLM response must include a python script.")
        return _run_python_analysis(python_code, max_rows)

    if not sql:
        context = database_context()
        return {
            "answer": f"{llm_config_hint()} Or provide a read-only SQL query to execute analytics from the dashboard.",
            "schema_context": context,
            "sql": None,
            "rows": [],
            "charts": [],
        }

    if not is_read_only_sql(sql):
        raise ValueError("Only a single read-only SELECT or WITH query is allowed.")

    df = _execute_read_only_query(sql, max_rows)
    chart_specs = _chart_specs_from_generated(generated, df)
    charts = [_build_chart(df, chart_spec) for chart_spec in chart_specs]
    charts = [chart for chart in charts if chart is not None]
    return {
        "answer": (generated or {}).get("answer", "Query executed."),
        "sql": sql,
        "rows": df.to_dict(orient="records"),
        "columns": list(df.columns),
        "charts": charts,
    }


def _ask_llm(question: str) -> dict[str, Any]:
    prompt = {
        "system_prompt": analytics_system_prompt(),
        "task": "Return strict JSON with one python key for a guarded database analytics script.",
        "question": question,
        "schema_context": database_context(),
        "project_guidance": _agent_guidance_excerpt(),
        "constraints": [
            "Return JSON only with a python string. Do not return answer/sql/rows/charts directly.",
            "The python script runs in a restricted environment with pd, np, px, go, run_sql, set_answer, set_rows, and add_chart already available.",
            "Do not import modules, read files, write files, use network calls, or connect to the database directly.",
            "All data access must use run_sql(sql, limit=...). run_sql only allows one read-only SELECT or WITH query and executes it against Postgres.",
            "The script must call set_answer(text), set_rows(dataframe_or_records), and add_chart(fig_or_plotly_dict) for each useful chart.",
            "Prefer features schema tables and finalized model_data tables when Postgres is available.",
            "Never mutate data.",
            "Return JSON only, with no Markdown fences.",
        ],
    }
    return parse_llm_json(llm_generate_text(prompt, json_mode=True))


def _run_python_analysis(code: str, max_rows: int) -> dict[str, Any]:
    """Run a guarded LLM-authored analytics script against Postgres."""
    code = _normalize_python_analysis_code(code)
    _validate_python_analysis_code(code)
    output: dict[str, Any] = {
        "answer": "",
        "sql": None,
        "rows": [],
        "columns": [],
        "charts": [],
    }
    state: dict[str, Any] = {"last_df": None}

    def run_sql(sql: str, limit: int | None = None) -> pd.DataFrame:
        bounded_limit = max(1, min(int(limit or max_rows), max_rows))
        df = _execute_database_only_read_only_query(sql, bounded_limit)
        state["last_df"] = df
        output["sql"] = sql.strip().rstrip(";")
        return df

    def set_answer(value: Any) -> None:
        output["answer"] = str(value)

    def set_rows(value: Any) -> None:
        df = _value_to_dataframe(value)
        output["rows"] = df.head(MAX_ANALYTICS_DISPLAY_ROWS).to_dict(orient="records")
        output["columns"] = list(df.columns)

    def add_chart(value: Any) -> None:
        if len(output["charts"]) >= MAX_ANALYTICS_CHARTS:
            return
        chart = _plotly_value_to_chart(value)
        if chart:
            output["charts"].append(chart)

    globals_dict: dict[str, Any] = {
        "__builtins__": SAFE_BUILTINS,
        "add_chart": add_chart,
        "go": go,
        "np": np,
        "pd": pd,
        "px": px,
        "run_sql": run_sql,
        "set_answer": set_answer,
        "set_rows": set_rows,
    }
    exec(compile(code, "<analytics-llm-python>", "exec"), globals_dict, {})

    if not output["rows"] and isinstance(state.get("last_df"), pd.DataFrame):
        set_rows(state["last_df"])
    if not output["answer"]:
        output["answer"] = "Analysis executed."
    return output


def _validate_python_analysis_code(code: str) -> None:
    code = _normalize_python_analysis_code(code)
    if len(code) > MAX_ANALYTICS_CODE_CHARS:
        raise ValueError(f"Analytics Python script is too long; limit is {MAX_ANALYTICS_CODE_CHARS} characters.")
    try:
        tree = ast.parse(code)
    except SyntaxError as exc:
        raise ValueError(f"Analytics Python script has invalid syntax: {exc}") from exc

    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            raise ValueError("Analytics Python scripts may not import modules.")
        if isinstance(node, (ast.AsyncFunctionDef, ast.ClassDef, ast.Delete, ast.Global, ast.Nonlocal, ast.With, ast.AsyncWith)):
            raise ValueError(f"Analytics Python scripts may not use {type(node).__name__}.")
        if isinstance(node, ast.Name):
            if node.id.startswith("__") or node.id in FORBIDDEN_PYTHON_REFERENCES:
                raise ValueError(f"Analytics Python scripts may not reference {node.id}.")
        if isinstance(node, ast.Attribute):
            if node.attr.startswith("__") or node.attr in FORBIDDEN_PYTHON_ATTRIBUTES:
                raise ValueError(f"Analytics Python scripts may not access attribute {node.attr}.")
        if isinstance(node, ast.Call):
            call_name = _call_name(node.func)
            if call_name in FORBIDDEN_PYTHON_NAMES or call_name in FORBIDDEN_PYTHON_REFERENCES:
                raise ValueError(f"Analytics Python scripts may not call {call_name}.")


def _normalize_python_analysis_code(code: str) -> str:
    """Tolerate common provider formatting around generated Python strings."""
    stripped = code.strip()
    fenced = re.match(r"^```(?:python|py)?\s*(.*?)\s*```$", stripped, flags=re.IGNORECASE | re.DOTALL)
    if fenced:
        stripped = fenced.group(1).strip()
    if len(stripped) >= 2 and stripped[0] == stripped[-1] and stripped[0] in {'"', "'"}:
        try:
            decoded = json.loads(stripped)
            if isinstance(decoded, str):
                stripped = decoded.strip()
        except json.JSONDecodeError:
            pass
    if "\\n" in stripped and "\n" not in stripped:
        stripped = stripped.replace("\\r\\n", "\n").replace("\\n", "\n").replace("\\t", "    ")
    return stripped


def _call_name(func: ast.AST) -> str | None:
    if isinstance(func, ast.Name):
        return func.id
    if isinstance(func, ast.Attribute):
        return func.attr
    return None


def _value_to_dataframe(value: Any) -> pd.DataFrame:
    if isinstance(value, pd.DataFrame):
        return value
    if isinstance(value, pd.Series):
        return value.to_frame()
    return pd.DataFrame(value)


def _plotly_value_to_chart(value: Any) -> dict[str, Any] | None:
    if hasattr(value, "to_json") and callable(value.to_json):
        chart = json.loads(value.to_json())
    elif isinstance(value, dict) and isinstance(value.get("data"), list):
        chart = {"data": value["data"], "layout": value.get("layout") if isinstance(value.get("layout"), dict) else {}}
    else:
        return None
    chart["layout"] = _polished_layout(chart.get("layout"))
    return chart


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
    bounded_sql = f"SELECT * FROM ({stripped_sql}) AS analytics_query LIMIT :max_rows"
    try:
        engine = create_engine(database_url())
        try:
            return _execute_database_read_only_query(engine, bounded_sql, max_rows)
        finally:
            engine.dispose()
    except Exception as db_exc:
        csv_tables = _load_csv_tables()
        if not csv_tables:
            raise RuntimeError(f"Database query failed and no finalized CSV fallback is available: {db_exc}") from db_exc

        with sqlite3.connect(":memory:") as conn:
            for table_name, df in csv_tables.items():
                df.to_sql(table_name, conn, index=False, if_exists="replace")
            conn.execute("PRAGMA query_only = ON")
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


def _execute_database_only_read_only_query(sql: str, max_rows: int) -> pd.DataFrame:
    stripped_sql = sql.strip().rstrip(";")
    if not is_read_only_sql(stripped_sql):
        raise ValueError("Only a single read-only SELECT or WITH query is allowed.")
    bounded_sql = f"SELECT * FROM ({stripped_sql}) AS analytics_query LIMIT :max_rows"
    engine = create_engine(database_url())
    try:
        return _execute_database_read_only_query(engine, bounded_sql, max_rows)
    finally:
        engine.dispose()


def _execute_database_read_only_query(engine: Any, bounded_sql: str, max_rows: int) -> pd.DataFrame:
    """Execute analytics SQL inside a database-enforced read-only transaction."""
    with engine.connect() as connection:
        with connection.begin():
            if connection.dialect.name == "postgresql":
                connection.execute(text("SET TRANSACTION READ ONLY"))
                connection.execute(text(f"SET LOCAL statement_timeout = {POSTGRES_ANALYTICS_STATEMENT_TIMEOUT_MS}"))
            return pd.read_sql(text(bounded_sql), connection, params={"max_rows": max_rows})


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


def _chart_specs_from_generated(generated: dict[str, Any] | None, df: pd.DataFrame) -> list[dict[str, Any]]:
    if isinstance(generated, dict) and isinstance(generated.get("charts"), list):
        chart_specs = [chart for chart in generated["charts"] if isinstance(chart, dict)]
        return chart_specs[:MAX_ANALYTICS_CHARTS]

    default_chart = _default_chart_spec(df)
    return [default_chart] if default_chart else []


def _build_chart(df: pd.DataFrame, spec: dict[str, Any] | None) -> dict[str, Any] | None:
    if not spec:
        return None
    if isinstance(spec.get("data"), list):
        return {
            "data": spec["data"],
            "layout": _polished_layout(spec.get("layout")),
        }
    chart_type = spec.get("type", "bar")
    x_column = spec.get("x")
    y_column = spec.get("y")
    if x_column not in df.columns or (y_column and y_column not in df.columns):
        return None

    labels = {
        column: label
        for column, label in ((x_column, spec.get("x_label")), (y_column, spec.get("y_label")))
        if column and label
    }
    if chart_type == "line" and y_column:
        fig = px.line(df, x=x_column, y=y_column, labels=labels)
    elif chart_type == "scatter" and y_column:
        fig = px.scatter(df, x=x_column, y=y_column, labels=labels)
    elif chart_type == "histogram":
        fig = px.histogram(df, x=x_column, labels=labels)
    elif chart_type == "area" and y_column:
        fig = px.area(df, x=x_column, y=y_column, labels=labels)
    elif chart_type == "box":
        fig = px.box(df, x=x_column, y=y_column, labels=labels)
    elif y_column:
        fig = px.bar(df, x=x_column, y=y_column, labels=labels)
    else:
        return None

    fig.update_layout(_polished_layout({"title": spec.get("title")} if spec.get("title") else None))
    return json.loads(fig.to_json())


def _polished_layout(layout: Any | None) -> dict[str, Any]:
    provided = layout if isinstance(layout, dict) else {}
    return {
        "template": "plotly_white",
        "paper_bgcolor": "#ffffff",
        "plot_bgcolor": "#ffffff",
        "font": {"family": "Inter, system-ui, sans-serif", "color": "#16201f"},
        "colorway": ["#0f766e", "#b42318", "#2563eb", "#92400e", "#7c3aed", "#475569"],
        "margin": {"l": 44, "r": 20, "t": 44, "b": 44},
        "hovermode": "closest",
        **provided,
    }
