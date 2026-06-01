# Data Tab LLM Analytics System Prompt Spec

## Goal

Make the Data tab feel like a careful MMA analytics workbench: a user asks a
natural-language question, the app supplies a rich MMA feature-store system
prompt plus live schema context, the LLM returns a safe read-only query and a
guarded Python analysis script, and the dashboard displays a polished report
with narrative, SQL provenance, row preview, and Plotly visuals.

This spec also defines the copyable prompt exposed in the dashboard through the
`Copy Analytics Agent System Prompt` button so a user can paste the same
context into a coding agent.

## Research Basis

The design follows these current agent and analytics-agent practices:

- [OpenAI practical guide to building agents](https://openai.com/business/guides-and-resources/a-practical-guide-to-building-ai-agents/): use clear orchestration patterns, typed application code, guardrails, and tracing rather than relying on one free-form prompt.
- [OpenAI structured outputs](https://developers.openai.com/api/docs/guides/structured-outputs): prefer schema-shaped output for machine-consumed results; JSON mode alone does not guarantee schema adherence.
- [OpenAI prompt engineering](https://developers.openai.com/api/docs/guides/prompt-engineering): use role/context separation, explicit success criteria, examples, and structured output expectations.
- [Anthropic multi-agent research system](https://www.anthropic.com/engineering/multi-agent-research-system): subagents help when work can be decomposed into independent branches; each delegated task needs an objective, source/tool guidance, boundaries, and output format.
- [Anthropic building effective agents](https://www.anthropic.com/engineering/building-effective-agents): start simple, evaluate thoroughly, keep the agent-computer interface explicit, and add multi-step agentic complexity only when it earns its keep.
- [Microsoft Fabric data-agent best practices](https://learn.microsoft.com/en-us/fabric/data-science/data-agent-configuration-best-practices): narrow the domain, define business terms, prioritize sources, document joins, and include representative query logic.
- [Snowflake Cortex Analyst semantic-view best practices](https://www.snowflake.com/en/developers/guides/best-practices-semantic-views-cortex-analyst/): use business-focused descriptions, clear relationships, metrics/filters, verified queries, and iterative test sets.
- [Plotly Python graph objects](https://plotly.com/python/graph-objects/): let analysis scripts create real Plotly figures, then serialize those figures to portable JSON for browser rendering.
- [Text-to-SQL security guidance](https://www.dpriver.com/blog/text-to-sql-security-10-risks-before-production-deployment/): validate generated SQL deterministically before execution; read-only credentials are helpful but not sufficient by themselves.

## Product Shape

User flow:

1. User opens the Data tab.
2. User optionally clicks `Copy Analytics Agent System Prompt` and pastes it
   into a coding agent or external LLM.
3. User asks a question in the Analytics box.
4. Backend sends the canonical analytics system prompt, live schema context,
   user question, and output constraints to the configured LLM.
5. LLM returns strict JSON with a `python` script string.
6. Backend validates the script, runs it in a restricted Python environment, and
   lets the script fetch Postgres data only through guarded `run_sql()`.
7. Frontend displays an organized analytics report: explanation, metrics, SQL
   disclosure, row preview, and one or more chart cards.

Current app hooks:

- Prompt source: `libs/web/analytics_prompt.py`.
- Prompt copy endpoint: `GET /api/data/analytics/system-prompt`.
- LLM prompt assembly: `libs/web/analytics.py`.
- Data-tab copy button: `libs/web/static/index.html` and `libs/web/static/app.js`.
- Data-tab report renderer: `renderAnalyticsReport()` and
  `renderPlotlyCharts()` in `libs/web/static/app.js`.

## Subagent Pattern

Do not require subagents for the first dashboard implementation. Use a single
LLM call with deterministic Python and SQL validation for the normal path.
Subagents become useful when a host coding agent, research agent, or future
dashboard backend can parallelize bounded checks:

- Schema scout: inspect available tables and columns, then return only relevant
  schema snippets.
- Python analyst: draft a short analysis script using `run_sql`, pandas, and
  Plotly.
- SQL verifier: check table names, joins, temporal leakage, aggregation grain,
  row explosion risk, and forbidden statement types.
- Chart designer: choose the most useful Plotly chart for the analysis output.
- Results critic: flag missing data, weak sample size, null-heavy features, and
  whether the answer overclaims.

The coordinator should pass each subagent a narrow objective, explicit output
format, and clear stop condition. The coordinator owns the final response and
must not blindly concatenate subagent outputs.

## Prompt Contract

The canonical prompt should be treated as a system or developer message when a
provider supports that. In the current dashboard, it is also included inside the
JSON user payload under `system_prompt` so every configured provider receives
the same context.

The LLM must return strict JSON only:

```json
{
  "python": "df = run_sql(\"SELECT weightclass, COUNT(*) AS fights FROM features.fight_mapping GROUP BY weightclass ORDER BY fights DESC\", limit=100)\\nset_answer(\"Lightweight and welterweight have the largest fight counts in this database slice.\")\\nset_rows(df)\\nfig = px.bar(df, x=\"weightclass\", y=\"fights\", title=\"Fight count by weight class\")\\nfig.update_layout(xaxis_title=\"Weight class\", yaxis_title=\"Fights\")\\nadd_chart(fig)"
}
```

The backend executes the script only in a restricted analytics runner. The script
cannot import modules, read/write files, call network APIs, or connect to the
database directly. All data must come from Postgres through `run_sql()`, which
enforces one read-only `SELECT` or `WITH` statement in a read-only transaction.
The script may call `set_answer()`, `set_rows()`, and `add_chart()` to create the
final UI report.

## Canonical System Prompt

The full canonical prompt lives in `libs/web/analytics_prompt.py` and is exposed
verbatim by `GET /api/data/analytics/system-prompt`. It is kept in code so the
dashboard copy button and the live LLM request always share one source of truth.

The critical output contract is:

```json
{
  "python": "df = run_sql(\"SELECT ...\", limit=100)\\nset_answer(\"...\")\\nset_rows(df)\\nadd_chart(px.bar(df, x=\"...\", y=\"...\"))"
}
```

The prompt tells the LLM about the feature store, feature suffixes, odds fields,
opponent-adjusted performance, and the restricted runner helpers: `pd`, `np`,
`px`, `go`, `run_sql`, `set_answer`, `set_rows`, and `add_chart`.

## Implementation Notes

- Keep the prompt source in code so the copy button and LLM path share one
  canonical string.
- Keep live schema context dynamic. The prompt teaches the conceptual schema;
  `database_context()` supplies the actual tables and columns currently present.
- Continue validating generated SQL with `is_read_only_sql()` and database
  read-only transactions. The LLM Python runner must use Postgres through
  `run_sql()` only; any manual SQL fallback path should remain separate from
  generated LLM analysis.
- Add structured-output retries later if providers support JSON schema, not just
  JSON object mode.
- Add a verified-question suite with 10 to 20 examples covering feature drift,
  sparse features, matchup feature deltas, odds/EV, calibration buckets, and
  raw scrape quality.

## Acceptance Criteria

- Data tab has a self-explanatory copy button labeled
  `Copy Analytics Agent System Prompt`.
- Clicking the button copies the canonical prompt from
  `/api/data/analytics/system-prompt`.
- LLM analytics prompt assembly includes the same canonical prompt.
- SQL execution remains read-only and bounded.
- Tests cover the API endpoint, UI wiring, local icon shim, and prompt inclusion
  in LLM requests.
