# MMA AI Agent Guide

This repository combines the historical UFCStats scraper from `UFCScraper` with
the feature store, training, and prediction code from `mma-ai-db`. The intended
release artifact is a Dockerized app with a small web dashboard for data refresh,
model training, analytics, and fight prediction.

## Current App Surface

- Web app entry point: `libs.web.app:app`
- First-time setup: `setup.ps1` on Windows or `./setup.sh` on macOS/Linux. The
  scripts download Hugging Face artifacts, restore both Docker Postgres
  databases, copy processed CSVs, extract the starter model, optionally
  configure `LLM_PROVIDER`/`LLM_MODEL` plus provider API keys, and start the
  dashboard.
- Local web command: `uv run mma-web`
- Docker command: `docker compose up --build`
- Docker Postgres 18.1 initializes both `mma-ai` and `odds`; keep
  `docker/postgres-init/01-create-odds.sql` mounted in Compose while
  `ODDS_DATABASE_URL` points at the `odds` database.
- Browser charts use the local `/vendor/plotly.min.js` route backed by the
  installed Python `plotly` package, so chart rendering does not depend on the
  public Plotly CDN.
- Dashboard icons use `libs/web/static/icons.js`, a local `window.lucide`
  compatibility shim for the icons used in the UI; do not reintroduce CDN icon
  dependencies.
- Dashboard tabs:
  - Data tab: scrape raw UFCStats CSVs, rebuild PostgreSQL feature tables,
    recalculate odds features from the imported Hugging Face `odds` database,
    create finalized CSVs, and run read-only analytics. Live BestFightOdds
    refresh is opt-in, not part of the default dashboard update.
  - Train tab: launch AutoGluon training with defaults from `libs/modeling/train.py`; advanced knobs stay collapsed.
    The Evaluations subtab summarizes saved artifacts such as `evals.txt`,
    `model_stats.txt`, `test_predictions.csv`, `all_predictions.csv`, and
    `calibration_curve.png`; completed dashboard training jobs also write
    `dashboard_evaluation_summary.json` and `dashboard_evaluation.md`, and the
    `mma-evaluate --write-report --format text` script produces the same report
    artifacts for automation.
  - Predict tab: choose a model, automatically load upcoming events from
    Wikipedia into an event-name dropdown, run event prediction, and validate
    manual fighter matchups. Odds are not model inputs; they are only for market
    probability and EV calculations.

LLM setup choices are OpenAI, Codex/OpenAI-compatible, Anthropic, Google Gemini,
xAI Grok, OpenRouter, DeepSeek, Mistral, Together AI, Perplexity Sonar, local
OpenAI-compatible servers, and custom OpenAI-compatible endpoints. LLMs are
used by Data-tab analytics.

Heavy workflows are intentionally lazy. Importing the web app must not import
AutoGluon, start Scrapy, connect to Postgres, or call external APIs.
Background jobs capture stdout, stderr, command lines, tracebacks, and script
output in `data/logs/jobs` and expose full logs at `/api/jobs/{job_id}/log`.

## Data Pipeline

Raw UFCStats data lives in `MMA_AI_UFCSTATS_DIR`, defaulting to
`data/raw/ufcstats`. The release repo intentionally tracks the seed
`competitions.csv` and `individuals.csv` files in that directory; generated
model CSVs, models, logs, and DB dumps remain ignored.

- `competitions.csv`: one row per completed fight with event metadata, fighter
  URLs, result, method, round, time, time format, referee, details, and round
  statistics for both fighters.
- `individuals.csv`: fighter profile data with name, nickname, URL, date of
  birth, weight, reach, height, and stance.

The in-repo scraper adapter is `libs/scraping/ufcstats.py`. It preserves the
standalone UFCScraper field order and incremental merge behavior, and is
exposed through `scripts/scrape_ufcstats.py` and the dashboard Data tab. Default
scrapes skip fighter URLs and event URLs already present in the CSVs, then merge
new rows; `--force-full` is the explicit destructive raw-CSV rebuild path.

`main.py --reset-db` recreates generated schemas and finalized CSVs from the raw
CSVs. The normal public update path after the initial Hugging Face DB import is
`uv run mma-rebuild-db --scrape --reset-db --odds-features`, which recalculates
`features.odds` from the configured imported `ODDS_DATABASE_URL` without
scraping BestFightOdds. Use `--odds` only when you explicitly want to refresh
live BestFightOdds data before calculating odds features. Its normal outputs
are:

- `data/prediction_data.csv`: side-by-side fighter feature rows used for
  inference and upcoming-fight feature construction.
- `data/training_data.csv`: finalized win/loss model training data.
- `data/training_data_dec.csv`: finalized decision/no-decision model training
  data.

## Database Schemas And Tables

PostgreSQL is the authoritative feature store. The default URL is controlled by
`DATABASE_URL`.

Primary schemas:

- `features`: raw, derived, and feature-specific tables.
- `model_data`: finalized model-ready outputs when present.
- `public`: infrastructure or ad hoc tables only.

Core `features` tables:

- `features.fight_stats_fe`: raw fight rows loaded from UFCStats with round 1 columns,
  total-fight columns, result metadata, fighter IDs, and event IDs.
- `features.fight_stats_derived`: enhanced copy of `features.fight_stats_fe` after base
  calculators add derived fields and smoothing replaces sparse raw values.
- `features.fighter_mapping`: fighter ID, normalized fighter name, and date of birth.
- `features.event_mapping`: event ID and event date.
- `features.fight_mapping`: fight ID, both fighter IDs, event ID, and weight class.

Feature-specific tables are created from the derived table and then layered with
historical calculations. Common examples include `age`, `reach`, `height`,
`ufc_age`, `days_since_last_fight`, `sig_str`, `strikes`, `td`, `sub_att`,
`ctrl`, `head`, `body`, `leg`, `distance`, `clinch`, `ground`, `ko`, `decision`,
`win`, `odds`, and style or opponent-adjusted variants.

## Feature Naming Guide

The system uses suffixes to describe feature meaning:

- `_rd1`: first-round value.
- `_smooth`: Bayesian-smoothed value before raw replacement.
- `_total`: cumulative total.
- `_acc`: landed divided by attempted.
- `_def`: defensive rate against opponent attempts.
- `_per_min`: rate per fight minute.
- `_ratio` or `_per`: normalized rate or conversion metric.
- `_avg`: historical simple average.
- `_dec_avg`: time-decayed historical average.
- `_mad`: median absolute deviation.
- `_sdev`: standard deviation.
- `_opp_*`: opponent historical performance.
- `_adjperf`: opponent-adjusted performance.
- `_dec_adjperf`: time-decayed opponent-adjusted performance.
- `_diff`: fighter1 minus fighter2, used by model training and inference.

Static or metadata fields that frequently matter:

- `fighter_name`, `fighter1_name`, `fighter2_name`
- `event_date`, `fight_id`, `event_id`
- `weightclass` and `weightclass_encoded`
- `age`, `reach`, `height`, `ape`, `ufcage`
- `odds`, implied market probability, and expected value fields

## Training Defaults

The dashboard mirrors `libs/modeling/train.py` defaults:

- target: `win`
- preset: `extreme`
- time limit: `3000` seconds
- split strategy: `timeseries_split`
- walk-forward windows: `4`
- walk-forward initial year: `2021`
- start date: `2014-01-01`
- minimum prior fights: `2`
- normalization: `robust`
- recency weights: enabled
- recency decay: `0.15`
- feature importance: enabled
- refit full: enabled
- refit all data: disabled
- default model families: `TABICL`, `MITRA`, `TABM`, `GBM_PREP`, `CAT`, `GBM`, `REALTABPFN-V2`
- custom feature list/include/exclude/required feature filters: unset by default

Keep advanced training controls collapsed in the UI. The default path should be
safe for a user who just wants to train the current best model.

## Prediction Workflow

Upcoming event prediction is driven by:

- `libs/wikipedia_scraper.py`
- `libs/upcoming_fights.py`
- `predict.py`
- `libs/feature_store/inference/*`

The dashboard can list model directories from `MMA_AI_MODELS_DIR`, defaulting to
`AutogluonModels`. A valid model directory usually includes `feats.txt`; single
models also need `scaler.pkl`, while walk-forward ensembles scale internally.
The collapsed advanced prediction controls can override the finalized
`prediction_data.csv` and `training_data.csv` paths; server-side path validation
must keep those files under `MMA_AI_DATA_DIR`. Prediction output directory
overrides are also allowed, but only under `MMA_AI_DATA_DIR`.

Event predictions write artifacts under `data/predictions/latest` by default,
including `fight_predictions.csv` when prediction succeeds. The UI should render
AI probability, market probability, AI odds, and positive EV status in a compact
result graphic.

Manual fighter matchup prediction uses the same `predict.py` pipeline as event
prediction. The CLI accepts `--fighter1`, `--fighter2`, optional `--fight-date`,
and optional `--fighter1-odds` / `--fighter2-odds`; this skips Wikipedia event
lookup while still routing through `InferenceDataBuilder`, model feature
filtering, scaling, visualization, CSV output, and positive-EV calculation.
The dashboard exposes the optional fight date as a `YYYY-MM-DD` date input and
must validate it before starting a background job.
Web-triggered prediction jobs must pass `--no-manual-odds` when fetching BFO
odds so background jobs never block waiting for terminal input. Do not create a
separate feature formula for manual matchups; that would drift from the
production event path.
When event odds are missing, accept user-provided American odds through the
dashboard/API `manual_odds` mapping and pass it to `predict.py` with
`--manual-odds-json`. The local CLI may still use the interactive
`get_manual_fighter_odds()` prompts.

## Analytics Guidance For AI Agents

Analytics queries must be read-only. Only run a single `SELECT` or `WITH` query.
Reject mutation keywords such as `insert`, `update`, `delete`, `drop`, `create`,
`alter`, `copy`, `truncate`, or `vacuum`.
The dashboard also wraps Postgres analytics in a database-enforced read-only
transaction with a statement timeout, and CSV fallback analytics use SQLite
query-only mode after loading the finalized CSVs.

Prefer these sources in order:

1. Finalized model tables or CSVs for model-facing analytics.
2. Feature-specific tables for understanding a feature family.
3. `fight_stats_derived` for row-level engineered features.
4. `fight_stats_fe` only when investigating raw scrape quality.

When Postgres is unavailable, the dashboard analytics helper exposes finalized
CSVs through in-memory read-only table names:

- `training_data` for `data/training_data.csv`.
- `training_data_dec` for `data/training_data_dec.csv`.
- `prediction_data` for `data/prediction_data.csv`.

Good analytics tasks:

- Compare feature drift by year or event.
- Find sparse feature families and first-time-fighter sensitivity.
- Inspect calibration or confidence buckets from prediction outputs.
- Compare AI probability to market probability and EV.
- Summarize top positive and negative feature differences for a matchup.

Avoid leaking future data. Any historical aggregate used for a fight must be
based only on rows before that fight's `event_date`.

## Testing Expectations

Add tests with every behavior change. For the web layer:

- App import should be light and side-effect free.
- API tests should not run scrapers, train models, or call external services.
- Service tests should use temporary `MMA_AI_*` paths.
- Analytics tests should cover SQL guardrails.
- Prediction and training integration tests should monkeypatch heavy functions
  unless explicitly marked as slow.
- Evaluation tests should use fixture model directories with saved artifact
  files rather than training real AutoGluon models.
