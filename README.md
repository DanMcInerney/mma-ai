# MMA AI

Dockerized UFC fight data, feature engineering, analytics, and fight prediction.
This repository combines the historical UFCStats scraping workflow from
`UFCScraper` with the PostgreSQL feature store, modeling code, and prediction
pipeline from `mma-ai-db`.

The public app is a small FastAPI dashboard for data refresh, read-only
analytics, and fight prediction. Training remains a CLI workflow for advanced
users.

## Contents

1. [Quick Start](#quick-start)
2. [Dashboard](#dashboard)
3. [Data Model Overview](#data-model-overview)
4. [Core Tables](#core-tables)
5. [Feature Engineering Pipeline](#feature-engineering-pipeline)
6. [Layer Reference](#layer-reference)
7. [Query Patterns](#query-patterns)
8. [Analytics Schema Reference](#analytics-schema-reference)
9. [Manual Development Setup](#manual-development-setup)

## Quick Start

For a first-time local install with predictions ready, run the bootstrap script.
It downloads database dumps, processed prediction/training CSVs, and the starter
AutoGluon model from `https://huggingface.co/datasets/DanMcInerney/mma-ai`,
imports the dumps into Docker Postgres, optionally configures an analytics LLM,
starts the dashboard, and opens it in your browser.

Windows PowerShell:

```powershell
powershell -ExecutionPolicy Bypass -File .\setup.ps1
```

macOS/Linux:

```bash
./setup.sh
```

Run `powershell -ExecutionPolicy Bypass -File .\setup.ps1 -Help` or
`./setup.sh --help` to see all non-interactive setup options before Docker,
downloads, or database imports begin.

Open the dashboard at http://localhost:8000, or at the alternate port printed by
setup if port 8000 is already in use. The top bar shows a `Ready` badge when the
imported database tables, processed CSVs, and starter model are visible to the
web app. Readiness also validates required CSV headers, so a partial or malformed
artifact reports the missing columns before prediction begins.

The bootstrap download is about 2.5 GB. Docker is required. Optional: copy
`.env.example` to `.env` yourself if you want to provide keys or non-default
paths before running setup.

Setup is safe to rerun after an interrupted install. It reuses verified cached
artifacts, re-extracts incomplete starter models, and skips the large database
restore when the existing Docker databases already contain the expected
`features.fight_mapping` and `bestfightodds.bfo` tables. Use `-ForceImport` or
`--force-import` when you intentionally want to restore the Hugging Face dumps
again. Use `-SkipDownload` or `--skip-download` only when the artifact cache
already exists; setup validates the cache before copying CSVs, extracting the
model, or importing dumps. Use `-ForceDownload` or `--force-download` to repair a
corrupt cache.

If readiness still times out, inspect the stack logs:

```bash
docker compose logs --tail 120 web db
```

Missing database-table readiness errors usually mean rerun setup with
`-ForceImport` or `--force-import`; missing CSV/model errors usually mean rerun
without skip-download or with force-download.

If your machine already has Postgres on `localhost:5432`, setup automatically
chooses another free host port for Docker Postgres and writes it to
`MMA_AI_POSTGRES_PORT` in `.env`. It also updates the local `DATABASE_URL` and
`ODDS_DATABASE_URL` entries to that selected host port, while the Docker web app
still reaches Postgres through Docker's internal `db:5432` address. To force a
specific host port:

```powershell
powershell -ExecutionPolicy Bypass -File .\setup.ps1 -PostgresPort 55432
```

```bash
./setup.sh --postgres-port 55432
```

Setup also writes `MMA_AI_WEB_PORT` to `.env`. If another local web server is
already using port 8000, setup chooses a free port from 18000 upward and prints
the dashboard URL after `/api/readiness` confirms the databases, all processed
CSVs, and the Hugging Face starter model are visible to the web app. To force a
specific dashboard port:

```powershell
powershell -ExecutionPolicy Bypass -File .\setup.ps1 -WebPort 18000
```

```bash
./setup.sh --web-port 18000
```

If you already bootstrapped artifacts and only want to start the app:

```bash
docker compose up --build db web
```

After the stack is deployed, run the deployed readiness smoke check against the
actual dashboard URL:

```bash
uv run mma-docker-smoke --deployed-url http://localhost:8000
```

If setup chose another web port, use that URL instead. This check verifies
`/api/health`, requires `/api/readiness` to be ready, and confirms at least one
compatible `win` model is discoverable before prediction.

The Compose stack starts PostgreSQL 18.1 and initializes both the main `mma-ai`
database and the auxiliary `odds` database used by odds-related workflows. The
setup scripts restore the Hugging Face dumps into those databases. The web
service waits for the PostgreSQL healthcheck before starting, so first-run
readiness checks do not race a cold database container.

Compose also maps `host.docker.internal` to Docker's host gateway, so Linux,
macOS, and Windows users can point the web container at host-side services such
as a local Postgres instance, Ollama, or LM Studio with the same URL shape.

During setup you can choose OpenAI, Codex/OpenAI-compatible, Anthropic Claude,
Google Gemini, xAI Grok, OpenRouter, DeepSeek, Mistral, Together AI,
Perplexity Sonar, a local OpenAI-compatible server such as Ollama or LM Studio,
or a custom OpenAI-compatible endpoint for Data-tab analytics. These choices are
saved in `.env` as `LLM_PROVIDER`, `LLM_MODEL`, `LLM_API_KEY`, and optional
`LLM_BASE_URL`. Non-interactive installs can pass values directly:

```powershell
powershell -ExecutionPolicy Bypass -File .\setup.ps1 `
  -LlmProvider anthropic `
  -LlmModel claude-3-5-sonnet-latest `
  -LlmApiKey "<token>"
```

```bash
./setup.sh --llm-provider local --llm-model llama3.1 --llm-base-url http://host.docker.internal:11434/v1
```

For local development without Docker:

```bash
uv sync
uv run mma-web
```

Local Python commands automatically load the repo `.env` file without overriding
values already exported in your shell, so the URLs written by setup work for
`uv run mma-web`, `uv run mma-predict`, and the other CLI entrypoints. The local
web command uses `MMA_AI_PORT` first, then the setup-selected
`MMA_AI_WEB_PORT`, and also accepts explicit overrides:

```bash
uv run mma-web --host 127.0.0.1 --port 18000
```

To run the Docker web app against an existing Postgres instance on your host,
copy `.env.example` to `.env` and set the Compose-specific URLs:

```env
MMA_AI_COMPOSE_DATABASE_URL=postgresql://postgres:postgres@host.docker.internal:5432/mma-ai
MMA_AI_COMPOSE_ODDS_DATABASE_URL=postgresql://postgres:postgres@host.docker.internal:5432/odds
```

Then run only the web service without Compose dependencies, so Compose does not
also start its bundled Postgres service on port 5432:

```bash
docker compose up --no-deps --build web
```

If you want a fully isolated Docker database instead, leave those Compose URLs
unset and import the Hugging Face dumps into the Compose Postgres volume.

Before publishing a release, run the tracked-file hygiene audit:

```bash
uv run mma-release-audit
```

## Dashboard

- Data: update the shipped raw UFCStats CSVs incrementally, rebuild the
  PostgreSQL feature store, recalculate odds features from the imported
  Hugging Face `odds` database, write finalized CSVs, and run read-only AI
  analytics over Postgres or finalized CSV fallbacks. Completed refreshes report
  before and after row deltas so new fights/fighters are visible without
  opening CSVs. The Analytics panel shows whether an LLM provider is ready or
  the dashboard is currently in SQL-only analytics mode.
- Predict: choose a model, automatically load upcoming UFC events from
  Wikipedia into an event-name dropdown, predict a selected event, or run a
  manual fighter-vs-fighter matchup. Prediction-time live/manual odds are
  enabled by default for market probability, expected value, and pick-edge
  reporting, rather than being passed into the predictor. Event prediction
  accepts manual fighter odds in the dashboard so web jobs never need to block
  on terminal prompts. Manual matchup prediction defaults the fight date to
  today and does not expose per-fighter odds controls. Use the advanced
  Flaresolverr proxy toggle only when BestFightOdds is blocking normal odds
  scraping.

Each long-running Data or Predict job streams stdout/stderr into a debug log
under `data/logs/jobs` and exposes it through the dashboard and
`/api/jobs/{job_id}/log`. Dashboard jobs run one at a time so model/data writes
and captured debug logs remain deterministic.

Dashboard charts use the local `/vendor/plotly.min.js` route backed by the
installed Python `plotly` package. Icons use `libs/web/static/icons.js`, a local
Lucide-compatible shim. The Docker smoke command checks `/api/health`, verifies
that the dashboard HTML plus local Plotly/icon assets are served, and confirms
the runtime image does not include test tooling.

## Commands

```bash
uv run mma-scrape-ufcstats
uv run mma-rebuild-db
uv run mma-rebuild-db --scrape --reset-db --odds-features
uv run mma-train
uv run mma-evaluate --write-report --format text
uv run python scripts/evaluate_model.py --write-report --format text
uv run mma-predict --help
uv run mma-web --help
uv run pytest
docker compose up --build
docker compose build web
uv run mma-docker-smoke
uv run mma-release-audit
uv run mma-docker-smoke --deployed-url http://localhost:8000
```

Optional real-browser Predict tab e2e, useful before release when Chrome is
available:

```powershell
$env:MMA_AI_RUN_BROWSER_E2E='1'; uv run pytest tests/e2e/test_predict_tab_next_event.py::test_predict_tab_browser_predicts_next_ufc_event -q
```

```bash
MMA_AI_RUN_BROWSER_E2E=1 uv run pytest tests/e2e/test_predict_tab_next_event.py::test_predict_tab_browser_predicts_next_ufc_event -q
```

The dashboard uses the same command paths in background jobs so the UI does not
fork a separate feature or prediction implementation.

## Project Files

- `AGENTS.md` and `CLAUDE.md`: compact agent guidance for safe analytics,
  training, prediction, feature semantics, and test expectations.
- `docs/ANALYTICS_SCHEMA.md`: deeper plain-English analytics schema reference
  for feature meanings, odds units, leakage status, and query patterns.
- `Dockerfile` and `docker-compose.yml`: public release runtime with Postgres
  and the FastAPI dashboard. After building the web image, `uv run
  mma-docker-smoke` runs the container, checks `/api/health`, verifies the
  dashboard HTML plus local Plotly/icon assets are served, and confirms the
  runtime image does not include test tooling. After deploying a bootstrapped
  Compose stack, `uv run mma-docker-smoke --deployed-url http://localhost:8000`
  checks readiness and `win` model discovery against the running dashboard.
- `libs/web`: FastAPI app, background jobs, web service adapters, analytics,
  evaluation summaries, and static UI.
- `libs/scraping/ufcstats.py`: in-repo UFCStats scraper adapter.
- `libs/feature_store`: PostgreSQL schemas, feature calculators, training-data
  assembly, and inference feature builders.
- `libs/modeling`: training, evaluation, calibration, profit reporting, and
  portable model artifact helpers.
- `data/raw/ufcstats`: tracked seed raw CSVs. Generated model CSVs, predictions,
  DB dumps, logs, and models stay out of git.

## Data Model Overview

The database is designed around one central grain:

```text
one row = one fighter in one completed fight
primary key = (fight_id, fighter_id)
fight_id joins the two fighter rows for the bout
event_id joins the bout to its event date
```

Most analytics queries start in the `features` schema. The important schemas are:

- `features`: authoritative feature store. It contains raw scraped stats,
  derived stats, mapping tables, feature-family tables, and odds features.
- `model_data`: finalized model-ready outputs when a run materializes them in
  the database.
- `public`: infrastructure or ad hoc tables only.

The normal public update after the first Hugging Face DB import is:

```bash
uv run mma-rebuild-db --scrape --reset-db --odds-features
```

That command incrementally refreshes the raw UFCStats CSVs, recreates generated
feature schemas from those CSVs, recalculates `features.odds` from the imported
`ODDS_DATABASE_URL`, and writes finalized CSVs. Use `--odds` only when you
explicitly want to refresh live BestFightOdds data before calculating odds
features.

## Raw Data

The repo tracks current seed UFCStats files:

- `data/raw/ufcstats/competitions.csv`
- `data/raw/ufcstats/individuals.csv`

`competitions.csv` has one row per completed fight. It includes result,
fighter names/URLs, weight class, method, end round, end time, time format,
referee, event metadata, and per-round stats for both fighters.

`individuals.csv` has one row per fighter profile. It includes name, nickname,
UFCStats URL, date of birth, weight, reach, height, and stance.

The scraper is incremental by default. It skips fighter URLs and event URLs
already present in the tracked CSVs, merges new rows, and preserves existing
rows. Use `uv run mma-scrape-ufcstats --force-full` only when you intentionally
want a destructive raw-CSV rebuild.

## Core Tables

### `features.fighter_mapping`

One row per fighter known to the scraper.

Key columns:

- `fighter_id`: internal integer identifier.
- `fighter_url`: UFCStats fighter URL, unique.
- `fighter_name`, `fighter_nickname`, `fighter_stance`.
- `fighter_weight`: UFCStats profile weight string.
- `fighter_dob`: date of birth when available.
- `fighter_height`, `fighter_reach`: inches, with missing values left null at
  this mapping layer.

### `features.event_mapping`

One row per UFCStats event.

Key columns:

- `event_id`: internal integer identifier.
- `event_url`: UFCStats event URL, unique.
- `event_date`: event date.
- `event_location`: event location text.

### `features.fight_mapping`

One row per fight.

Key columns:

- `fight_id`: internal integer identifier.
- `event_id`: joins to `features.event_mapping`.
- `fighter1_id`, `fighter2_id`: the two fighter rows for the bout.
- `weightclass`, `weightclass_encoded`.
- `method`, `details`.
- `end_round`, `end_time`, `time_format`.
- `result`: `1` means `fighter1_id` won, `0` means `fighter2_id` won. Draws,
  no contests, and other non-binary outcomes are not treated as wins by the
  outcome calculators.

### `features.fight_stats_core`

The normalized scrape output. It has one row per fighter per fight and stores
round-by-round count columns from UFCStats, such as `sig_str_land_rd1`,
`sig_str_att_rd1`, `td_land_rd3`, `ctrl_rd2`, `distance_att_rd5`, and so on.

This table is close to raw UFCStats semantics and is mainly an ingestion
checkpoint.

### `features.fight_stats_fe`

The first engineered table. It begins as a copy of `fight_stats_core`, then the
base calculators add totals, outcomes, time, and static attributes.

Use this table when you need to inspect raw-ish fight values before smoothing
and higher-order layers.

### `features.fight_stats_derived`

The main derived staging table. It receives selected fields from
`fight_stats_fe`, applies smoothing, and builds first-order derived columns such
as `_total`, `_acc`, `_def`, `_per_min`, `_ratio`, and `_pressure`.

After smoothing, the original observed count columns are temporarily renamed to
`_raw`, the smoothed columns take the original names, and `_raw` columns are
dropped after accuracy/defense calculations no longer need them. As a result,
plain columns such as `sig_str_land` in `fight_stats_derived` are smoothed
values, not the untouched scrape counts.

## Feature-Family Tables

After `fight_stats_derived` is prepared, the pipeline splits it into smaller
feature-family tables. Each table keeps the same grain and primary key:
`fight_id`, `fighter_id`, `event_id`.

Common feature-family tables include:

| Table | Meaning |
| --- | --- |
| `features.sig_str` and `features.sig_str_rd1` | Significant strikes, attempts, accuracy, defense, rates, ratios |
| `features.strikes` and `features.strikes_rd1` | Total strikes |
| `features.head`, `features.body`, `features.leg` | Significant strikes by target |
| `features.distance`, `features.clinch`, `features.ground` | Significant strikes by position/range |
| `features.td` and `features.td_rd1` | Takedowns landed/attempted and related rates |
| `features.sub` and `features.sub_rd1` | Submission attempts and submission-win indicators |
| `features.ctrl` and `features.ctrl_rd1` | Control time in seconds and derived control rates |
| `features.rev` and `features.rev_rd1` | Reversals |
| `features.kd` and `features.kd_rd1` | Knockdowns |
| `features.ko`, `features.decision`, `features.win` | Outcome indicators and finish-derived features |
| `features.time_sec` and `features.time_sec_rd1` | Fight duration features |
| `features.age`, `features.reach`, `features.ape`, `features.ufcage` | Static or pre-fight biographical features |
| `features.days_since_last_fight` | Layoff feature |
| `features.odds` | Matched BestFightOdds prices and implied probabilities |

Some generated databases may also contain auxiliary prior tables such as
`features.sig_str_wc_mean`, `features.sig_str_wc_mad`,
`features.sig_str_minimum_mad`, or first-time-fighter statistic tables. These
are support tables for adjusted performance and are usually not the first place
to query for analytics.

Discover available feature tables:

```sql
SELECT table_name
FROM information_schema.tables
WHERE table_schema = 'features'
  AND table_type = 'BASE TABLE'
ORDER BY table_name;
```

Discover columns inside one feature family:

```sql
SELECT column_name, data_type
FROM information_schema.columns
WHERE table_schema = 'features'
  AND table_name = 'sig_str'
ORDER BY ordinal_position;
```

## Base Stat Glossary

The scraper and feature store use compact stat names. Most are available in
total-fight form and, for many stats, first-round form with `_rd1`.

| Base stat | What it means |
| --- | --- |
| `kd` | Knockdowns scored by the fighter |
| `sig_str_land`, `sig_str_att` | Significant strikes landed and attempted |
| `strikes_land`, `strikes_att` | Total strikes landed and attempted |
| `head_land`, `head_att` | Significant strikes to the head |
| `body_land`, `body_att` | Significant strikes to the body |
| `leg_land`, `leg_att` | Significant strikes to the legs |
| `distance_land`, `distance_att` | Significant strikes at distance |
| `clinch_land`, `clinch_att` | Significant strikes in the clinch |
| `ground_land`, `ground_att` | Significant strikes on the ground |
| `td_land`, `td_att` | Takedowns landed and attempted |
| `sub_att` | Submission attempts |
| `rev` | Reversals |
| `ctrl` | Control time, stored in seconds |
| `time_sec` | Fight duration in seconds |
| `win` | `1` for the winning fighter, `0` otherwise |
| `ko` | `1` for the fighter who won by KO/TKO, `0` otherwise |
| `decision` | `1` for the fighter who won by decision, `0` otherwise |
| `sub_land` | `1` for the fighter who won by submission, `0` otherwise |
| `age` | Fighter age in years at the event date |
| `days_since_last_fight` | Days since that fighter's previous UFCStats fight |
| `reach` | Reach in inches, imputed by weight-class average when missing |
| `height` | Height in inches in `fight_stats_fe`; imputed by weight-class average when missing |
| `ape` | Reach divided by height |
| `ufcage` | Years since the fighter's first UFCStats fight |

Round suffixes identify round-specific values:

- `_rd1` is first-round value. Example: `sig_str_land_rd1`.
- Some outcome calculators create round-specific outcome columns such as
  `ko_rd1` or `win_rd3`.
- The public modeling pipeline primarily carries first-round stat families into
  `fight_stats_derived`; full round 2-5 columns stay closer to the raw/core
  tables.

Outcome features are fighter-row indicators. If a fight ends by KO, only the
winner's row has `ko = 1`; the losing row has `ko = 0`.

## Feature Engineering Pipeline

The pipeline is intentionally layered. Later suffixes often depend on earlier
suffixes, so column names encode the calculation path.

```text
raw CSVs
  -> features.fight_stats_core
  -> features.fight_stats_fe
  -> base derived stats
  -> features.fight_stats_derived
  -> smoothing
  -> totals, accuracy, defense, per-minute rates, ratios
  -> feature-family tables
  -> custom per features
  -> opponent values
  -> weight-class priors and MAD floors
  -> rolling averages and time-decayed averages
  -> adjusted performance
  -> prediction_data.csv and training_data.csv
```

The most important rule: if you are doing predictive analytics, be explicit
about time. Feature-family tables are row-level historical artifacts for
completed fights. The final training CSV shifts non-static fighter stats by one
fight before modeling. If you query raw feature-family tables directly for a
pre-fight question, filter or shift so the current fight does not leak into the
answer.

## Layer Reference

### Smoothing: `_smooth` becomes the plain column

Two Bayesian smoothing calculators run on `fight_stats_derived` before most
rate features are created.

Beta-Binomial smoothing handles binary or bounded outcome-style stats:

- `win`, `win_rd1`
- `ko`, `ko_rd1`
- `decision`
- `sub_land`, `sub_land_rd1`
- `ctrl`, `ctrl_rd1`

For a binary stat, the posterior mean is:

```text
p_smoothed = (prior_rate * tau + observed_successes) / (tau + attempts)
```

Attempts depend on the stat. A win or KO has one opportunity per fight;
`sub_land` uses `sub_att`; control time is modeled against fight duration.
Control output is converted back to smoothed seconds.

Poisson-Gamma smoothing handles count stats:

- `_land` and `_att` striking/takedown/submission count columns
- `kd`
- `rev`

For a count stat:

```text
prior_rate = historical count / exposure minutes
posterior_rate = (prior_rate * tau + observed_count) / (tau + exposure_minutes)
smoothed_count = exposure_minutes * posterior_rate
```

Round-one stats use capped round-one exposure; total stats use full fight
duration. Priors are weight-class aware with global fallback parameters loaded
from `config/optimized_parameters.json`.

Implementation detail: the database first writes `<stat>_smooth`, then renames
the original observed column to `<stat>_raw`, renames `<stat>_smooth` to
`<stat>`, and eventually deletes `_raw`. So a later column like
`sig_str_land_per_min` is based on smoothed `sig_str_land`.

### Totals: `_total`

`TotalCalculator` computes cumulative career totals per fighter:

```text
<stat>_total =
  SUM(<stat>) OVER (
    PARTITION BY fighter_id
    ORDER BY event_date, fight_id
    ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
  )
```

Example: `td_land_total` is the fighter's cumulative takedowns landed through
that row. In the feature tables this includes the current completed fight; the
training cleaner shifts dynamic totals to previous-fight values before model
training.

### Accuracy: `_acc`

`AccuracyCalculator` creates smoothed landed-over-attempted rates for `_land`
and `_att` pairs:

```text
<family>_acc = (prior_accuracy * tau + land_raw) / (tau + att_raw)
```

Examples:

- `sig_str_acc`
- `head_acc`
- `td_acc`
- `distance_rd1_acc`

Accuracy uses the temporary `_raw` landed/attempted columns so it can smooth the
ratio directly from observed attempts while still benefiting from weight-class
priors. Values are clipped to `[0, 1]`. If attempts are zero, the calculator
falls back to a weight-class or global prior rate.

### Defense: `_def`

`DefenseCalculator` copies the opponent's accuracy into the fighter row under a
defensive suffix:

```text
fighter.<family>_def = opponent.<family>_acc
```

Despite the name, this is best read as "opponent landing accuracy against this
fighter in this fight." Lower values are better defensive outcomes. For example,
if a fighter's opponent landed 35 percent of attempted significant strikes, the
fighter's `sig_str_def` is `0.35`.

This convention is important when writing analytics. To ask "who limited
opponents to low head-strike accuracy," sort `head_def` ascending, not
descending.

### Per-Minute Rates: `_per_min`

`PerMinCalculator` divides count-like stats by fight duration in minutes:

```text
<stat>_per_min = <stat> / (time_sec / 60.0)
<stat>_rd1_per_min = <stat>_rd1 / (time_sec_rd1 / 60.0)
```

The code's canonical suffix is `_per_min`. If you are thinking in per-second
terms, convert manually with `<stat>_per_min / 60.0`; the current feature store
does not generate a separate `_per_sec` suffix.

Examples:

- `sig_str_land_per_min`
- `td_att_per_min`
- `kd_rd1_per_min`
- `ctrl_per_min`

### Within-Fight Shares: `_ratio`

`RatioCalculator` compares a fighter's value to the opponent's value in the
same fight:

```text
<stat>_ratio = fighter_stat / (fighter_stat + opponent_stat)
```

This produces a bounded fight-share feature. A value near `1.0` means the
fighter accounted for almost all of that stat in the fight; a value near `0.0`
means the opponent did. When both values are zero, the Python fallback uses
`0.5`; the SQL template fallback uses `0`. Check your generated database if
zero-zero treatment matters for a query.

Examples:

- `sig_str_land_ratio`
- `td_land_ratio`
- `kd_ratio`
- `win_ratio`

### First-Round Pressure: `_pressure`

`PressureCalculator` currently creates one pressure feature:

```text
sig_str_land_pressure = sig_str_land_rd1 / sig_str_land
```

It answers: "what share of this fighter's significant-strike output came in
round 1?" If total significant strikes are zero, the value is `0`.

### Custom Per Features: `_per_`

`PerCalculator` creates domain-specific ratios that do not fit the generic
`_per_min` or `_ratio` patterns. These are stored in the most relevant
feature-family table.

Examples:

| Feature | Formula |
| --- | --- |
| `ko_per_sig_str_land` | `ko / sig_str_land` |
| `sig_str_per_str_att` | `sig_str_land / strikes_att` |
| `distance_per_sig_str_land` | `distance_land / sig_str_land` |
| `clinch_per_sig_str_land` | `clinch_land / sig_str_land` |
| `ground_per_sig_str_land` | `ground_land / sig_str_land` |
| `head_per_sig_str_land` | `head_land / sig_str_land` |
| `body_leg_per_sig_str_land` | `(body_land + leg_land) / sig_str_land` |
| `td_per_sig_str_att` | `td_att / sig_str_att` |
| `td_land_per_ctrl` | `td_land / ctrl` |
| `ground_land_per_ctrl` | `ground_land / ctrl` |
| `ground_land_per_td_land` | `ground_land / td_land` |
| `sub_att_per_ctrl` | `sub_att / ctrl` |
| `rev_per_ctrlopp` | `rev / opponent_ctrl` |
| `ko_sub_per_win` | `(ko + sub_land) / win` |
| `ko_sub_rd1_per_win` | `(ko_rd1 + sub_land_rd1) / win` |
| `sub_per_all_ctrl` | `sub_att / (fighter_ctrl + opponent_ctrl)` |

Zero denominators become `0.0`.

### Opponent Values: `_opp`

`OpponentCalculator` joins each fighter row to the other fighter row in the same
fight and copies selected columns:

```text
<stat>_opp = opponent.<stat>
```

Examples:

- `sig_str_land_opp`
- `td_att_per_min_opp`
- `head_acc_opp`
- `ko_opp`

Use `_opp` when you want the opponent's realized performance in that fight.
Use adjusted performance layers when you want to compare a fighter's realized
performance against what that opponent historically tends to allow.

### Weight-Class Priors: `_wc_mean`, `_wc_mad`, `_minimum_mad`

Several support tables store robust weight-class baselines:

- `features.<table>_wc_mean`: mean value by weight class.
- `features.<table>_wc_mad`: median absolute deviation by weight class.
- `features.<table>_minimum_mad`: small positive MAD floor by weight class.

The mean and MAD prior tables are computed from a historical calibration window
beginning at `2014-01-01` and ending at `2023-01-01`, with a minimum sample
size per weight class. The MAD floor is a low percentile of existing rolling
MAD values and prevents adjusted performance from exploding when a denominator
is too small.

These support tables are most useful for understanding how `_adjperf` is
shrunk. For ordinary analytics, query the main feature-family tables first.

### Rolling Averages: `_avg`

`AverageCalculator` computes fighter-level rolling means:

```text
<stat>_avg =
  AVG(<stat>) OVER (
    PARTITION BY fighter_id
    ORDER BY event_date, fight_id
    ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
  )
```

Examples:

- `sig_str_land_avg`
- `td_acc_avg`
- `ko_per_sig_str_land_avg`
- `sig_str_land_adjperf_avg`

In feature-family tables, `_avg` includes the current completed fight. In the
final win/loss training CSV, dynamic columns are shifted to previous-fight
values and receive `_prev` before differences are calculated.

### Time-Decayed Averages: `_dec_avg`

`TimedecAvgCalculator` computes exponentially weighted fighter history:

```text
weight = exp(-ln(2) * years_since_fight / half_life_years)
<stat>_dec_avg = sum(stat * weight) / sum(weight)
```

The half-life comes from `config/decay.py`: environment override first, then
`data/comprehensive_tuning/optimized_decay.json`, then the default. A
three-year half-life means a fight three years old receives half the weight of
a fight at the current event date.

Examples:

- `sig_str_land_dec_avg`
- `td_land_per_min_dec_avg`
- `head_def_dec_avg`
- `distance_acc_dec_adjperf_dec_avg`

Like `_avg`, the feature-table implementation is rolling through the current
completed fight. Treat `_dec_avg` as a post-fight historical feature unless you
are using the cleaned training output, the inference builder, or a custom
past-only query.

### Median Absolute Deviation: `_mad`

`MedianAbsoluteDeviationCalculator` measures a fighter's rolling variability:

```text
median = median(fighter history for stat)
<stat>_mad = median(abs(value - median))
```

For a fighter's first fight, the calculator falls back to precomputed
weight-class first-time MAD statistics when available. MAD is more robust than
standard deviation for sparse and outlier-heavy fight stats.

### Adjusted Performance: `_adjperf`

Adjusted performance asks:

```text
How much better or worse was this fighter's observed stat than what this
opponent historically allows, after shrinking sparse opponent history toward
weight-class priors?
```

For each eligible stat, `AdjustedPerformanceCalculator(decay=False)` creates:

```text
<stat>_adjperf
```

The simplified formula is:

```text
n = opponent-history sample size
w_mean = n / (n + K_mean)
w_mad = n / (n + K_mad)

mu_shrunk =
  w_mean * opponent_allowed_mean
  + (1 - w_mean) * weightclass_mean

mad_shrunk =
  max(
    w_mad * opponent_allowed_mad
    + (1 - w_mad) * weightclass_mad,
    mad_floor
  )

adjperf = clip((observed - mu_shrunk) / mad_shrunk, -7, 7)
```

The default shrinkage parameters are `K_mean = 4.0` and `K_mad = 4.0`.

The opponent history is not "the opponent's own stats." It is what previous
opponents achieved against that opponent. For example, for
`sig_str_land_per_min_adjperf`, the expected value is based on how many
significant strikes per minute previous fighters landed against today's
opponent, with sparse history shrunk toward the opponent's weight-class
baseline.

Eligible adjusted-performance inputs include:

- Accuracy, defense, ratio, pressure, and per-minute features.
- Custom `_per_` domain features.
- `win`, `decision`, and `time_sec`.
- Selected finishing/grappling ratios.

Totals are excluded from adjusted performance to avoid compounding volume,
career length, and opponent adjustment into a single unstable scale.

Interpretation:

- Positive `_adjperf`: the fighter exceeded expectation against that opponent.
- Negative `_adjperf`: the fighter underperformed relative to expectation.
- Values are robust z-score-like units clipped to `[-7, 7]`.

### Time-Decayed Adjusted Performance: `_dec_adjperf`

`AdjustedPerformanceCalculator(decay=True)` creates:

```text
<stat>_dec_adjperf
```

It uses the same adjusted-performance formula, but opponent history uses
time-decayed weights and a Kish effective sample size:

```text
n_effective = (sum(weights) ^ 2) / sum(weights ^ 2)
```

This gives recent fights more influence while still shrinking heavily when the
opponent has little effective history.

### Adjusted-Performance Aggregates

After `_adjperf` and `_dec_adjperf` are created, the pipeline runs historical
aggregation again on adjusted-performance columns:

- `<stat>_adjperf_avg`
- `<stat>_adjperf_dec_avg`
- `<stat>_dec_adjperf_avg`
- `<stat>_dec_adjperf_dec_avg`

Long names are normal. Read them from left to right:

```text
distance_acc_dec_adjperf_dec_avg
```

means:

1. Start with distance-striking accuracy.
2. Compare the fight value to a time-decayed opponent-allowed expectation.
3. Convert it to adjusted performance.
4. Take the fighter's time-decayed average of that adjusted-performance score.

### Odds Features

`features.odds` is populated from the imported BestFightOdds database, not from
UFCStats. It has the same `(fight_id, fighter_id, event_id)` grain.

Important columns:

- `opening_odds`, `closing_odds`: decimal odds from BestFightOdds.
- `ip_opening_odds`, `ip_closing_odds`: implied probabilities from decimal
  odds, computed as `1 / decimal_odds`.
- `vigless_ip_opening_odds`, `vigless_ip_closing_odds`: no-vig normalized
  implied probabilities across the two fighters.
- `sevenday_opening_odds`, `sevenday_ip_opening_odds`,
  `sevenday_vigless_ip_opening_odds`: odds closest to seven days before close.

Prediction-time live or manual odds use American odds in the UI/CLI and are used
for market probability, expected value, and pick-edge reporting. Historical odds
columns can still appear in generated CSVs, profit analysis, and model feature
lists depending on the artifact. Inspect a model's `feats.txt` before claiming
whether odds were or were not model inputs.

## Reading Feature Names

Feature names compose suffixes. A few examples:

| Feature | How to read it |
| --- | --- |
| `head_acc` | Smoothed head significant-strike accuracy in this fight |
| `head_acc_opp` | The opponent's `head_acc` in this fight |
| `head_acc_avg` | Fighter's rolling average of `head_acc` through this completed fight |
| `head_acc_dec_avg` | Fighter's time-decayed average of `head_acc` through this completed fight |
| `head_acc_adjperf` | How far `head_acc` exceeded what this opponent usually allows |
| `head_acc_dec_adjperf` | Same, but opponent history is time-decayed |
| `head_acc_dec_adjperf_dec_avg` | Fighter's time-decayed average of time-decayed adjusted performance |
| `td_land_per_ctrl_avg` | Rolling average of takedowns landed per control second |
| `sig_str_land_ratio_dec_adjperf_dec_avg` | Time-decayed average of adjusted within-fight significant-strike share |
| `sig_str_land_dec_avg_diff` | Fighter1 minus fighter2 for the selected historical feature in final training data |

Useful suffixes:

| Suffix | Meaning |
| --- | --- |
| `_rd1` | First-round value |
| `_smooth` | Temporary smoothed column before it is renamed to the base name |
| `_raw` | Temporary observed column kept briefly for accuracy calculations |
| `_total` | Cumulative fighter total through the row |
| `_acc` | Landed divided by attempted, with Bayesian smoothing |
| `_def` | Opponent accuracy against the fighter |
| `_per_min` | Rate per fight minute; derived from `time_sec` |
| `_ratio` | Fighter share of fighter-plus-opponent value in the same fight |
| `_pressure` | First-round share of total significant-strike output |
| `_per_` | Domain-specific custom ratio |
| `_opp` | Opponent's value in the same fight |
| `_wc_mean` | Weight-class mean prior in support tables |
| `_wc_mad` | Weight-class MAD prior in support tables |
| `_minimum_mad` | Weight-class MAD floor support table |
| `_mad` | Rolling median absolute deviation |
| `_avg` | Rolling average through the row |
| `_dec_avg` | Time-decayed rolling average through the row |
| `_adjperf` | Opponent-adjusted performance score |
| `_dec_adjperf` | Time-decayed opponent-adjusted performance score |
| `_prev` | Previous-fight shifted feature in cleaned training data |
| `_diff` | Fighter1 value minus fighter2 value in finalized model data |

## Training And Prediction Outputs

`CreateTrainingData` builds a wide, fighter-row dataframe by joining
feature-family tables back to `fight_stats_derived`. This intermediate output is
written as:

- `data/prediction_data.csv`

Despite the name, this file is a reusable feature matrix for inference and
upcoming-fight construction. It keeps both fighter rows and feature columns.

`CleanTrainingData` then prepares model training data:

1. Splits static features from dynamic fight stats.
2. Shifts dynamic features by one fight within each fighter history.
3. Rejoins fighter1 and fighter2 onto one row per fight.
4. Creates fighter1 absolute columns and fighter1-minus-fighter2 `_diff`
   columns.
5. Creates `y_true` from fighter1's result.

The main finalized outputs are:

- `data/training_data.csv`: win/loss model training data.
- `data/training_data_dec.csv`: decision/no-decision model training data.

Static features are not shifted because they are known pre-fight: `age`,
`days_since_last_fight`, `reach`, `height`, `ufcage`, historical odds fields
when included, and `weightclass_encoded`.

A valid starter model directory such as `ag-20260304_110750-win-extreme`
usually includes `feats.txt`. Single models also need `scaler.pkl`; walk-forward
ensembles scale internally.

## Query Patterns

Join feature-family tables to fighter and event metadata:

```sql
SELECT
  em.event_date,
  fm.weightclass,
  fmap.fighter_name,
  s.sig_str_land_per_min,
  s.sig_str_land_per_min_dec_avg,
  s.sig_str_land_per_min_dec_adjperf
FROM features.sig_str s
JOIN features.event_mapping em ON em.event_id = s.event_id
JOIN features.fight_mapping fm ON fm.fight_id = s.fight_id
JOIN features.fighter_mapping fmap ON fmap.fighter_id = s.fighter_id
ORDER BY em.event_date DESC, s.fight_id DESC
LIMIT 50;
```

Compare both fighters in one fight:

```sql
SELECT
  em.event_date,
  f1.fighter_name AS fighter1_name,
  f2.fighter_name AS fighter2_name,
  s1.sig_str_land_per_min_dec_avg AS f1_sig_str_pm,
  s2.sig_str_land_per_min_dec_avg AS f2_sig_str_pm,
  s1.sig_str_land_per_min_dec_avg - s2.sig_str_land_per_min_dec_avg AS diff
FROM features.fight_mapping fm
JOIN features.event_mapping em ON em.event_id = fm.event_id
JOIN features.fighter_mapping f1 ON f1.fighter_id = fm.fighter1_id
JOIN features.fighter_mapping f2 ON f2.fighter_id = fm.fighter2_id
JOIN features.sig_str s1
  ON s1.fight_id = fm.fight_id
 AND s1.fighter_id = fm.fighter1_id
JOIN features.sig_str s2
  ON s2.fight_id = fm.fight_id
 AND s2.fighter_id = fm.fighter2_id
ORDER BY em.event_date DESC
LIMIT 25;
```

Find fighters who recently suppressed opponent head accuracy:

```sql
SELECT
  fmap.fighter_name,
  COUNT(*) AS fights,
  AVG(h.head_def) AS avg_opponent_head_accuracy
FROM features.head h
JOIN features.event_mapping em ON em.event_id = h.event_id
JOIN features.fighter_mapping fmap ON fmap.fighter_id = h.fighter_id
WHERE em.event_date >= DATE '2023-01-01'
GROUP BY fmap.fighter_name
HAVING COUNT(*) >= 3
ORDER BY avg_opponent_head_accuracy ASC
LIMIT 20;
```

Find positive adjusted performance outliers for ground striking:

```sql
SELECT
  em.event_date,
  fmap.fighter_name,
  g.ground_land_per_min_dec_adjperf,
  g.ground_land_per_min,
  g.ground_land_per_min_opp
FROM features.ground g
JOIN features.event_mapping em ON em.event_id = g.event_id
JOIN features.fighter_mapping fmap ON fmap.fighter_id = g.fighter_id
WHERE g.ground_land_per_min_dec_adjperf IS NOT NULL
ORDER BY g.ground_land_per_min_dec_adjperf DESC
LIMIT 25;
```

Inspect model-facing finalized CSVs through dashboard analytics fallback table
names when Postgres is unavailable:

- `training_data` for `data/training_data.csv`.
- `training_data_dec` for `data/training_data_dec.csv`.
- `prediction_data` for `data/prediction_data.csv`.

For Postgres analytics, prefer sources in this order:

1. Finalized model tables or CSVs for model-facing analytics.
2. Feature-specific tables for understanding a feature family.
3. `features.fight_stats_derived` for row-level engineered features.
4. `features.fight_stats_fe` only when investigating raw scrape quality.

## Analytics Schema Reference

For deeper analytics work, see `docs/ANALYTICS_SCHEMA.md`. It documents the
fighter-fight grain, table families, feature suffixes, plain-English
interpretations such as `_adjperf`, historical-vs-predictive leakage rules,
`ODDS_DATABASE_URL` / `bestfightodds.bfo`, and the decimal-vs-American odds
boundary.

## Analytics Safety

Dashboard analytics are read-only by design. Only a single `SELECT` or `WITH`
query is allowed. Mutation keywords such as `insert`, `update`, `delete`,
`drop`, `create`, `alter`, `copy`, `truncate`, and `vacuum` are rejected. The
Postgres analytics path also runs inside a database-enforced read-only
transaction with a statement timeout, and CSV fallback analytics use SQLite
query-only mode.

To avoid future leakage in your own analytics:

- For descriptive analytics over completed fights, feature-family tables are
  appropriate.
- For predictive analytics, use `training_data.csv`, `prediction_data.csv`, or
  explicitly restrict aggregates to rows before the fight's `event_date`.
- Remember that `_avg`, `_dec_avg`, `_total`, and `_mad` in feature-family
  tables are rolling-through-current by implementation.
- Odds are known only when market data exists. Treat missing odds as missing
  market data, not as a neutral price.

## Manual Development Setup

For installation and first-time use, prefer the Quick Start above. Most users should use the repository bootstrap scripts
from the Quick Start: `setup.ps1` on Windows or `./setup.sh` on macOS/Linux. This section is retained as a
low-level development reference for contributors who explicitly want to restore
Hugging Face artifacts by hand, point at their own PostgreSQL instance, or run
CLI entrypoints outside Docker.

### Prerequisites

- Python 3.10-3.12. Python 3.12.4 is recommended.
- PostgreSQL database.
- Docker, if using the public release stack.
- `uv` package manager.
- Optional GPU for faster model training.

### Manual Installation Without Bootstrap Scripts

1. Clone and install:

   ```bash
   git clone <repository-url>
   cd mma-ai
   uv python install 3.12.4
   uv sync
   ```

2. Configure local environment:

   ```bash
   cp .env.example .env
   ```

   The example URLs match Docker Compose's localhost Postgres defaults. Edit
   them if your PostgreSQL setup uses different credentials, host, or database
   names.

3. Restore shared database artifacts:

   ```bash
   git lfs install
   mkdir -p artifacts
   git clone https://huggingface.co/datasets/DanMcInerney/mma-ai artifacts/mma-ai-dataset
   ```

   PowerShell equivalent:

   ```powershell
   git lfs install
   New-Item -ItemType Directory -Force artifacts | Out-Null
   git clone https://huggingface.co/datasets/DanMcInerney/mma-ai artifacts/mma-ai-dataset
   ```

   Restore the two PostgreSQL databases:

   ```bash
   createdb -U postgres mma-ai
   createdb -U postgres odds

   pg_restore --clean --if-exists --no-owner --jobs 4 \
     --dbname "postgresql://postgres:postgres@localhost:5432/mma-ai" \
     artifacts/mma-ai-dataset/dumps/mma-ai.postgres-custom

   pg_restore --clean --if-exists --no-owner --jobs 4 \
     --dbname "postgresql://postgres:postgres@localhost:5432/odds" \
     artifacts/mma-ai-dataset/dumps/odds.postgres-custom
   ```

   PowerShell restore:

   ```powershell
   createdb -U postgres mma-ai
   createdb -U postgres odds

   pg_restore --clean --if-exists --no-owner --jobs 4 `
     --dbname "postgresql://postgres:postgres@localhost:5432/mma-ai" `
     artifacts\mma-ai-dataset\dumps\mma-ai.postgres-custom

   pg_restore --clean --if-exists --no-owner --jobs 4 `
     --dbname "postgresql://postgres:postgres@localhost:5432/odds" `
     artifacts\mma-ai-dataset\dumps\odds.postgres-custom
   ```

4. Copy convenience CSVs and extract the pretrained win model:

   ```bash
   mkdir -p data AutogluonModels
   cp artifacts/mma-ai-dataset/processed/training_data.csv data/training_data.csv
   cp artifacts/mma-ai-dataset/processed/training_data_dec.csv data/training_data_dec.csv
   cp artifacts/mma-ai-dataset/processed/prediction_data.csv data/prediction_data.csv
   tar -xzf artifacts/mma-ai-dataset/models/ag-20260304_110750-win-extreme.tar.gz -C AutogluonModels
   ```

   PowerShell equivalent:

   ```powershell
   New-Item -ItemType Directory -Force data, AutogluonModels | Out-Null
   Copy-Item artifacts\mma-ai-dataset\processed\training_data.csv data\training_data.csv
   Copy-Item artifacts\mma-ai-dataset\processed\training_data_dec.csv data\training_data_dec.csv
   Copy-Item artifacts\mma-ai-dataset\processed\prediction_data.csv data\prediction_data.csv
   tar -xzf artifacts\mma-ai-dataset\models\ag-20260304_110750-win-extreme.tar.gz -C AutogluonModels
   ```

   With those copied, you can run predictions immediately:

   ```bash
   uv run python predict.py \
     --model-path AutogluonModels/ag-20260304_110750-win-extreme \
     --prediction-data-csv data/prediction_data.csv \
     --training-data-csv data/training_data.csv \
     --no-shap
   ```

5. Scrape UFCStats from this repo:

   ```bash
   uv run python -m scripts.scrape_ufcstats
   ```

6. Recreate generated schemas and finalized CSVs:

   ```bash
   uv run python main.py --reset-db
   ```

   You can combine incremental scrape, schema recreation, and odds feature
   recalculation:

   ```bash
   uv run python main.py --scrape --reset-db --odds-features
   ```

7. Train a model:

   ```bash
   uv run python -m libs.modeling.train --model-type win
   ```

8. Run predictions:

   ```bash
   uv run python predict.py --model-type win --no-shap
   ```

Generated raw scrape CSVs default to `data/raw/ufcstats/`. Training outputs
default to `data/`. You can override paths with `MMA_AI_UFCSTATS_DIR`,
`MMA_AI_DATA_DIR`, `MMA_AI_MODELS_DIR`, and `MMA_AI_PICKS_DIR`.

## Configuration

Root `.env` example:

```env
DATABASE_URL=postgresql://postgres:postgres@localhost:5432/mma-ai
ODDS_DATABASE_URL=postgresql://postgres:postgres@localhost:5432/odds
MMA_AI_COMPOSE_DATABASE_URL=
MMA_AI_COMPOSE_ODDS_DATABASE_URL=
MMA_AI_POSTGRES_PORT=5432
MMA_AI_WEB_PORT=8000
MMA_AI_DATA_DIR=data
MMA_AI_MODELS_DIR=AutogluonModels
MMA_AI_UFCSTATS_DIR=data/raw/ufcstats
THE_ODDS_API_KEY=
LLM_PROVIDER=
LLM_MODEL=
LLM_API_KEY=
LLM_BASE_URL=
OPENAI_API_KEY=
ANTHROPIC_API_KEY=
XAI_API_KEY=
OPENROUTER_API_KEY=
DEEPSEEK_API_KEY=
MISTRAL_API_KEY=
TOGETHER_API_KEY=
PERPLEXITY_API_KEY=
GOOGLE_API_KEY=
GEMINI_API_KEY=
```

## Troubleshooting

### Database not found

```text
psycopg2.OperationalError: database "mma-ai" does not exist
```

Create the database or rerun setup:

```bash
createdb -U postgres mma-ai
```

### Setup incomplete

If the dashboard reports setup incomplete, use `/api/readiness` or the top-bar
readiness details to identify whether the missing piece is database tables,
processed CSVs, or model artifacts. For Docker installs, check:

```bash
docker compose logs --tail 120 web db
```

### Optimized parameters missing

First full feature rebuilds can generate or load smoothing parameters. To force
re-optimization:

```bash
FORCE_REOPTIMIZE=1 uv run python main.py --reset-db
```

### Feature appears to include current fight

Many feature-family table columns are post-fight rolling values. That is useful
for descriptive analytics, but predictive analytics should use the cleaned
training data, the inference builder, or an explicit past-only query.

## Quick Reference

```bash
# First-time public setup
powershell -ExecutionPolicy Bypass -File .\setup.ps1
./setup.sh

# Start already bootstrapped stack
docker compose up --build db web

# Local app
uv run mma-web

# Incremental public data update
uv run mma-rebuild-db --scrape --reset-db --odds-features

# Scrape raw UFCStats data only
uv run mma-scrape-ufcstats

# Train through CLI
uv run mma-train

# Predict
uv run mma-predict --help

# Evaluate
uv run mma-evaluate --write-report --format text

# Release checks
uv run pytest
uv run mma-docker-smoke
uv run mma-release-audit
```
