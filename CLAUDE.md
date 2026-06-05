# MMA AI Claude Guide

This repo is becoming a public, Dockerized MMA AI app that combines the old
`UFCScraper` raw-data scraper with the `mma-ai-db` feature store, training, and
prediction system. Keep changes simple, tested, and aligned with the Data,
and Predict dashboard. Training remains a CLI workflow, not a dashboard tab.

## Main Entry Points

- Web app: `libs.web.app:app`
- First-time setup: `setup.ps1` on Windows or `./setup.sh` on macOS/Linux
- Web command: `uv run mma-web`
- Docker: `docker compose up --build`
- Docker Postgres init: `docker/postgres-init/01-create-odds.sql` creates the
  auxiliary `odds` database for `ODDS_DATABASE_URL`
- Local chart asset: `/vendor/plotly.min.js` served from the Python `plotly`
  package
- Local icon asset: `/static/icons.js`; do not reintroduce CDN icon dependencies
- Scrape raw UFCStats CSVs: `uv run mma-scrape-ufcstats`
- Rebuild database and finalized CSVs: `uv run mma-rebuild-db --reset-db`
- Train model from the CLI: `uv run mma-train`
- Summarize model evaluation artifacts: `uv run mma-evaluate`
- Predict upcoming event: `uv run mma-predict`

## Dashboard Tabs

- Data: scrape `competitions.csv` and `individuals.csv`, rebuild the Postgres
  feature store, recalculate odds features from the imported Hugging Face
  `odds` database, create `prediction_data.csv`, `training_data.csv`, and
  `training_data_dec.csv`, then support read-only analytics. Live
  BestFightOdds refresh is opt-in rather than part of the default dashboard
  update.
- Predict: select a model, automatically load upcoming UFC events from
  Wikipedia into an event-name dropdown, predict the selected event, and run
  manual fighter matchups through the same inference path. Prediction-time
  live/manual odds are enabled by default but only calculate EV, market
  probability, and pick edge reporting. Historical odds columns can exist in
  generated CSVs and model artifacts; inspect `feats.txt` before claiming
  whether a specific model used odds. Manual matchup per-fighter odds controls
  are not exposed in the dashboard. The advanced Flaresolverr proxy toggle is
  only for BestFightOdds blocking normal odds scraping.

LLM-assisted analytics use `LLM_PROVIDER`, `LLM_MODEL`, `LLM_API_KEY`, and
optional `LLM_BASE_URL` first. The setup scripts can configure OpenAI,
Codex/OpenAI-compatible, Anthropic Claude, Google Gemini, xAI Grok, OpenRouter,
DeepSeek, Mistral, Together AI, Perplexity Sonar, a local OpenAI-compatible
server such as Ollama or LM Studio, or a custom OpenAI-compatible endpoint.
Legacy `GEMINI_API_KEY` and `GOOGLE_API_KEY` still work as Google aliases.

Background jobs write stdout, stderr, subprocess command lines, and tracebacks
to `data/logs/jobs`; the dashboard reads full logs from `/api/jobs/{job_id}/log`.

The setup scripts download from
`https://huggingface.co/datasets/DanMcInerney/mma-ai`, verify checksums, restore
the main and odds dumps into Docker Postgres, copy processed CSVs into `data/`,
extract `ag-20260304_110750-win-extreme` into `AutogluonModels/`, optionally
write the LLM provider/model/API key configuration, and start the dashboard.

## Data And Database

Default paths:

- Raw UFCStats CSVs: `data/raw/ufcstats`
- Finalized CSVs: `data`
- Models: `AutogluonModels`
- Picks and graphics: `pics`

Important environment variables:

- `DATABASE_URL`
- `ODDS_DATABASE_URL`
- `MMA_AI_DATA_DIR`
- `MMA_AI_UFCSTATS_DIR`
- `MMA_AI_MODELS_DIR`
- `MMA_AI_PICKS_DIR`
- `LLM_PROVIDER`
- `LLM_MODEL`
- `LLM_API_KEY`
- `LLM_BASE_URL`
- `OPENAI_API_KEY`
- `ANTHROPIC_API_KEY`
- `XAI_API_KEY`
- `OPENROUTER_API_KEY`
- `DEEPSEEK_API_KEY`
- `MISTRAL_API_KEY`
- `TOGETHER_API_KEY`
- `PERPLEXITY_API_KEY`
- `GEMINI_API_KEY`
- `GOOGLE_API_KEY`
- `THE_ODDS_API_KEY`

The release repo tracks the seed raw UFCStats files
`data/raw/ufcstats/competitions.csv` and `data/raw/ufcstats/individuals.csv`.
Generated finalized CSVs, model artifacts, logs, and database dumps are ignored.
The UFCStats scraper is incremental by default: it skips existing fighter URLs
and event URLs, merges only new rows, and requires `--force-full` for a raw CSV
rebuild from scratch. After the initial Hugging Face database import, the normal
update command is `uv run mma-rebuild-db --scrape --reset-db --odds-features`.
That recalculates `features.odds` from the configured imported
`ODDS_DATABASE_URL` without scraping BestFightOdds; add `--odds` only when you
explicitly want a live BestFightOdds refresh before odds feature calculation.

Core tables live in the `features` schema:

- `fight_stats_fe`: raw UFCStats rows.
- `fight_stats_derived`: smoothed and derived fight rows.
- `fighter_mapping`, `event_mapping`, `fight_mapping`: stable IDs and joins.
- Feature tables such as `age`, `reach`, `height`, `ufc_age`,
  `days_since_last_fight`, `sig_str`, `strikes`, `td`, `ctrl`, `head`, `body`,
  `leg`, `distance`, `clinch`, `ground`, `ko`, `decision`, `win`, and `odds`.

Feature suffixes:

- `_rd1`: first round.
- `_smooth`: Bayesian smoothing.
- `_total`: cumulative total.
- `_acc`, `_def`, `_per_min`, `_ratio`, `_per`: derived rate features.
- `_avg`, `_dec_avg`: historical and time-decayed averages.
- `_mad`, `_sdev`: variability features.
- `_opp_*`: opponent features.
- `_adjperf`, `_dec_adjperf`: opponent-adjusted performance.
- `_diff`: fighter1 minus fighter2 for model input.

For deeper analytics, consult `docs/ANALYTICS_SCHEMA.md` before writing
nontrivial queries. It documents feature-family meanings, leakage status,
`_adjperf` interpretation, and the difference between historical decimal odds
in `features.odds` and live/manual American odds used during prediction.

## Training Defaults

Match `libs/modeling/train.py` unless the user asks otherwise:

- target `win`
- preset `extreme`
- time limit `3000`
- split `timeseries_split`
- walk-forward windows `4`
- walk-forward initial year `2021`
- start date `2014-01-01`
- minimum fights `2`
- normalization `robust`
- recency weights enabled with decay `0.15`
- feature importance enabled
- refit full enabled
- refit all data disabled
- custom feature list/include/exclude/required filters unset by default

## Analytics Rules

Analytics must be read-only. Only execute one `SELECT` or `WITH` statement and
reject mutation keywords. Prefer finalized model data, then feature-specific
tables, then `fight_stats_derived`, then raw `fight_stats_fe`.
Postgres analytics run inside a read-only transaction with a statement timeout;
CSV fallback analytics use SQLite query-only mode after loading finalized CSVs.

When Postgres is unavailable, analytics can query finalized CSV fallbacks as
read-only tables: `training_data`, `training_data_dec`, and `prediction_data`.

Respect time ordering. A feature for a fight must not use fights after that
fight's `event_date`.

## Prediction Rules

Manual matchups use `predict.py --fighter1 ... --fighter2 ...` with a fight
date. The dashboard defaults that date to today and does not expose matchup odds
inputs. Web jobs must use `--no-manual-odds` for BFO lookups so prediction never waits for terminal input.
Keep event and manual prediction on the same `InferenceDataBuilder` path.
Advanced dashboard prediction CSV path overrides must stay under
`MMA_AI_DATA_DIR`, prediction output directory overrides must stay under
`MMA_AI_DATA_DIR`, and manual matchup dates must use `YYYY-MM-DD`. Event
prediction can ingest user-supplied American odds through the API/UI
`manual_odds` mapping, which is passed to `predict.py --manual-odds-json`.

## Testing Rules

Add tests with behavior changes. Web app import and API tests must not import
AutoGluon, start Scrapy, call Wikipedia, or contact external LLMs. Use
monkeypatching for heavy training and prediction paths.
