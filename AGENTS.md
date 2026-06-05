# MMA AI Agent Guide

This file is the root instruction surface for coding agents working in this
repository. Keep it high-signal and repo-specific. The detailed human-facing
reference lives in `README.md`; use this file for rules that should change how
an agent edits, tests, or reasons about the code.

## Project Snapshot

MMA AI is a public, Dockerized UFC analytics and prediction app. It combines the
historical UFCStats scraper from `UFCScraper` with the PostgreSQL feature store,
training, and prediction pipeline from `mma-ai-db`.

The release surface is:

- A FastAPI dashboard for data refresh, read-only analytics, and prediction.
- CLI training and evaluation for advanced users.
- Docker Postgres 18.1 with two databases: `mma-ai` and `odds`.

Do not add dashboard training controls or a training navigation surface.
Training remains a CLI workflow.

## Repository Map

- `libs.web.app:app`: web app entry point.
- `libs/web`: FastAPI routes, services, jobs, analytics, evaluation summaries,
  path safety, and static UI.
- `libs/web/static`: dashboard frontend. Charts load from `/vendor/plotly.min.js`
  and icons load from `libs/web/static/icons.js` through a local Lucide shim.
  Do not add public CDN dependencies for Plotly or icons.
- `libs/scraping/ufcstats.py`: in-repo UFCStats scraper adapter.
- `libs/feature_store`: PostgreSQL schema, calculators, feature table assembly,
  training-data creation, and inference feature builders.
- `libs/modeling`: training, walk-forward utilities, evaluation, calibration,
  profit reporting, and portable artifacts.
- `scripts`: CLI adapters and release/dev utilities.
- `docs/ANALYTICS_SCHEMA.md`: plain-English analytics schema reference for
  feature meanings, leakage status, odds units, and query patterns.
- `data/raw/ufcstats`: tracked seed `competitions.csv` and `individuals.csv`.
- `docker/postgres-init/01-create-odds.sql`: creates the auxiliary `odds`
  database used by `ODDS_DATABASE_URL`.

Generated finalized CSVs, model artifacts, prediction outputs, logs, screenshots,
and database dumps stay out of git.

## Commands

Use `uv` as the Python command runner.

- First-time setup: `setup.ps1` on Windows or `./setup.sh` on macOS/Linux.
- Local web app: `uv run mma-web`.
- Docker app: `docker compose up --build`.
- Scrape raw UFCStats CSVs: `uv run mma-scrape-ufcstats`.
- Rebuild generated schemas and finalized CSVs: `uv run mma-rebuild-db --reset-db`.
- Normal public data update after initial Hugging Face import:
  `uv run mma-rebuild-db --scrape --reset-db --odds-features`.
- Train model from CLI: `uv run mma-train`.
- Evaluate model artifacts: `uv run mma-evaluate --write-report --format text`.
- Predict from CLI: `uv run mma-predict`.
- Release audit: `uv run mma-release-audit`.
- Docker smoke: `uv run mma-docker-smoke`.

For tests, prefer the narrowest relevant command first. Common examples:

- Web/API/docs: `uv run pytest tests/test_web -q`.
- Release docs only: `uv run pytest tests/test_web/test_release_docs.py -q`.
- Feature calculators: `uv run pytest tests/tests_layer1 tests/tests_layer2 tests/tests_layer3 -q`.
- Inference: `uv run pytest tests/test_inference -q`.
- Full suite: `uv run pytest`.

## Runtime Rules

- Importing the web app must be light and side-effect free. Do not import
  AutoGluon, start Scrapy, connect to Postgres, call Wikipedia, call LLMs, or
  hit external APIs at import time.
- Long-running Data and Predict actions must run as background jobs. Preserve
  stdout, stderr, subprocess command lines, tracebacks, and script output in
  `data/logs/jobs`; full logs are exposed at `/api/jobs/{job_id}/log`.
- Dashboard jobs are serialized. Do not add concurrent writers for model/data
  artifacts without explicit design work.
- Docker Compose must keep Postgres at `postgres:18.1` and mount
  `docker/postgres-init/01-create-odds.sql` so the `odds` database exists.
- The setup scripts download from the Hugging Face dataset, verify checksums,
  restore both database dumps, copy processed CSVs, extract the starter
  `ag-20260304_110750-win-extreme` model, optionally write LLM configuration,
  and start the dashboard.

## Dashboard Surface

Data tab:

- Incrementally scrape `competitions.csv` and `individuals.csv`.
- Rebuild the PostgreSQL feature store.
- Recalculate odds features from the imported Hugging Face `odds` database.
- Write `prediction_data.csv`, `training_data.csv`, and
  `training_data_dec.csv`.
- Run read-only analytics. Live BestFightOdds refresh is opt-in and not part of
  the default dashboard update.

Predict tab:

- Select a model from `MMA_AI_MODELS_DIR`, defaulting to `AutogluonModels`.
- Load upcoming events from Wikipedia into an event-name dropdown.
- Predict events and manual fighter matchups through the same `predict.py` and
  `InferenceDataBuilder` path.
- Prediction-time live/manual odds are enabled by default but are used for
  market probability, expected value, and pick edge reporting. Historical odds
  columns may exist in generated CSVs and model artifacts; inspect `feats.txt`
  before claiming whether a specific model used odds.
- Manual matchup odds controls are not exposed in the dashboard.
- The advanced Flaresolverr proxy toggle is only for BestFightOdds blocking
  normal odds scraping.

LLM-assisted analytics use `LLM_PROVIDER`, `LLM_MODEL`, `LLM_API_KEY`, and
optional `LLM_BASE_URL`. Supported setup choices include OpenAI,
Codex/OpenAI-compatible, Anthropic Claude, Google Gemini, xAI Grok, OpenRouter,
DeepSeek, Mistral, Together AI, Perplexity Sonar, local OpenAI-compatible
servers, and custom OpenAI-compatible endpoints.

## Data Pipeline

Raw UFCStats data lives in `MMA_AI_UFCSTATS_DIR`, defaulting to
`data/raw/ufcstats`.

- `competitions.csv`: one row per completed fight with event metadata, fighter
  URLs, result, method, round, time, time format, referee, details, and round
  statistics for both fighters.
- `individuals.csv`: fighter profile data with name, nickname, URL, date of
  birth, weight, reach, height, and stance.

The UFCStats scraper is incremental by default. It skips fighter URLs and event
URLs already present in the CSVs, then merges new rows. `--force-full` is the
explicit destructive raw-CSV rebuild path.

`main.py --reset-db` recreates generated schemas and finalized CSVs from the raw
CSVs. Use `--odds-features` to recalculate `features.odds` from the imported
`ODDS_DATABASE_URL` without scraping BestFightOdds. Use `--odds` only when live
BestFightOdds refresh is explicitly requested.

Normal generated outputs:

- `data/prediction_data.csv`: feature rows used for inference and
  upcoming-fight construction.
- `data/training_data.csv`: finalized win/loss model training data.
- `data/training_data_dec.csv`: finalized decision/no-decision training data.

## Database And Feature Semantics

PostgreSQL is the authoritative feature store. `DATABASE_URL` controls the main
database and `ODDS_DATABASE_URL` controls the imported odds database.

Primary schemas:

- `features`: raw, derived, and feature-specific tables.
- `model_data`: finalized model-ready outputs when present.
- `public`: infrastructure or ad hoc tables only.

Core tables:

- `features.fight_stats_fe`: raw-ish UFCStats rows after base calculators add
  fight duration, totals, outcomes, and static attributes.
- `features.fight_stats_derived`: selected copy of `fight_stats_fe` where
  smoothing and first-order derived layers are built.
- `features.fighter_mapping`: fighter ID, name, DOB, stance, reach, height.
- `features.event_mapping`: event ID, date, and location.
- `features.fight_mapping`: fight ID, both fighter IDs, event ID, weight class,
  method, end time, and result.

Feature-family tables are split from `fight_stats_derived`. Common families are
`age`, `reach`, `ape`, `ufcage`, `days_since_last_fight`, `sig_str`, `strikes`,
`td`, `sub`, `ctrl`, `head`, `body`, `leg`, `distance`, `clinch`, `ground`,
`ko`, `decision`, `win`, `time_sec`, and `odds`.

For complex analytics, consult `docs/ANALYTICS_SCHEMA.md` before writing queries.
It documents plain-English feature meanings, known-before-fight vs post-fight
status, and the difference between historical decimal odds in `features.odds`
and live/manual American odds used during prediction.

Feature suffixes:

- `_rd1`: first-round value.
- `_smooth`: temporary Bayesian-smoothed value before replacement.
- `_raw`: temporary observed value used during accuracy smoothing.
- `_total`: cumulative fighter total through the completed row.
- `_acc`: landed divided by attempted, with Bayesian smoothing.
- `_def`: opponent accuracy against the fighter; lower is better defense.
- `_per_min`: rate per fight minute, using `time_sec / 60`.
- `_ratio`: fighter share of fighter-plus-opponent value in the same fight.
- `_per_`: custom domain ratio such as `td_land_per_ctrl`.
- `_opp`: opponent value in the same fight.
- `_avg`: rolling average through the completed row.
- `_dec_avg`: time-decayed rolling average through the completed row.
- `_mad`: rolling median absolute deviation.
- `_adjperf`: opponent-adjusted performance score.
- `_dec_adjperf`: time-decayed opponent-adjusted performance score.
- `_prev`: previous-fight shifted value in cleaned training data.
- `_diff`: fighter1 minus fighter2 in finalized model data.

Important leakage note: feature-family tables are post-fight artifacts for
completed fights. `_avg`, `_dec_avg`, `_total`, and `_mad` include the current
completed row. Predictive model data shifts non-static features by one fight
before creating `_diff` columns.

## CLI Training Defaults

`uv run mma-train` should match `libs/modeling/train.py` unless the user asks
otherwise:

- target: `win`
- preset: `extreme`
- time limit: `3000`
- split strategy: `timeseries_split`
- walk-forward windows: `4`
- walk-forward initial year: `2021`
- start date: `2014-01-01`
- minimum prior fights: `2`
- normalization: `robust`
- recency weights enabled with decay `0.15`
- feature importance enabled
- refit full enabled
- refit all data disabled
- default families: `TABICL`, `MITRA`, `TABM`, `GBM_PREP`, `CAT`, `GBM`,
  `REALTABPFN-V2`
- custom feature list/include/exclude/required filters unset by default

## Prediction Rules

Upcoming event prediction is driven by `libs/wikipedia_scraper.py`,
`libs/upcoming_fights.py`, `predict.py`, and
`libs/feature_store/inference/*`.

- A valid model directory usually includes `feats.txt`; single models also need
  `scaler.pkl`, while walk-forward ensembles scale internally.
- Advanced prediction CSV path overrides must stay under `MMA_AI_DATA_DIR`.
- Prediction output directory overrides must stay under `MMA_AI_DATA_DIR`.
- Event predictions write under `data/predictions/latest` by default, including
  `fight_predictions.csv` on success.
- Manual matchups use `predict.py --fighter1 ... --fighter2 ...` and optional
  `--fight-date`; the dashboard date input must be valid `YYYY-MM-DD`.
- Web-triggered prediction jobs must pass `--no-manual-odds` when fetching BFO
  odds so jobs never block waiting for terminal input.
- When event odds are missing, accept API/UI American odds through the
  `manual_odds` mapping and pass them to `predict.py --manual-odds-json`.
- Do not create a separate feature formula for manual matchups.

## Analytics Rules

Analytics must be read-only:

- Execute only one `SELECT` or `WITH` query.
- Reject mutation keywords such as `insert`, `update`, `delete`, `drop`,
  `create`, `alter`, `copy`, `truncate`, and `vacuum`.
- The dashboard wraps Postgres analytics in a database-enforced read-only
  transaction with a statement timeout.
- CSV fallback analytics load finalized CSVs into SQLite query-only mode.

Prefer analytics sources in this order:

1. Finalized model tables or CSVs for model-facing analytics.
2. Feature-specific tables for understanding a feature family.
3. `fight_stats_derived` for row-level engineered features.
4. `fight_stats_fe` only when investigating raw scrape quality.

When Postgres is unavailable, analytics can query finalized CSV fallbacks as
read-only tables: `training_data`, `training_data_dec`, and `prediction_data`.

Avoid future leakage. Any historical aggregate used for a fight must be based
only on rows before that fight's `event_date`, unless the task is explicitly
descriptive analytics over completed fights.

## Testing Expectations

Add tests with every behavior change.

- Web app import and API tests must not run scrapers, train models, import
  AutoGluon, connect to Postgres, call Wikipedia, call LLMs, or contact external
  services.
- Service tests should use temporary `MMA_AI_*` paths.
- Analytics tests should cover SQL guardrails and fallback query-only mode.
- Prediction integration tests should monkeypatch heavy functions unless
  explicitly marked slow.
- Evaluation tests should use fixture model directories with saved artifacts
  rather than training real AutoGluon models.
- Release docs and setup changes should keep
  `tests/test_web/test_release_docs.py` and `uv run mma-release-audit` passing.

## Change Discipline

- Keep dashboard behavior aligned with the Data tab and Predict tab described
  above.
- Keep path validation strict around `MMA_AI_DATA_DIR`.
- Do not commit secrets, generated data, models, DB dumps, logs, screenshots, or
  local `.env` files.
- If you change public setup, Docker, dashboard assets, or artifact restore
  behavior, update `README.md`, `docs/RELEASE_READINESS.md`, and
  `docs/HUGGINGFACE_DATASET.md` as needed.
- If you change feature semantics, update `README.md`,
  `docs/ANALYTICS_SCHEMA.md`, and tests so analytics agents can craft correct
  queries.
