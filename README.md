# MMA AI

Dockerized UFC fight data, model training, analytics, and prediction dashboard.
This repository combines the raw UFCStats scraping workflow from `UFCScraper`
with the feature store, training, and prediction system from `mma-ai-db`.

## Quick Start

For a first-time local install with predictions ready, run the bootstrap script.
It downloads the database dumps, processed prediction/training CSVs, and starter
AutoGluon model from `https://huggingface.co/datasets/DanMcInerney/mma-ai`,
imports the dumps into Docker Postgres, optionally configures your preferred
analytics LLM provider/model/API key, starts the dashboard, and opens it in your
browser.

Windows PowerShell:

```powershell
powershell -ExecutionPolicy Bypass -File .\setup.ps1
```

macOS/Linux:

```bash
bash setup.sh
```

Open the dashboard at http://127.0.0.1:8000.

The bootstrap download is about 2.5 GB. Docker is required. Optional: copy
`.env.example` to `.env` yourself if you want to provide keys or non-default
paths before running setup.

If your machine already has Postgres on `localhost:5432`, setup automatically
chooses another free host port for Docker Postgres and writes it to
`MMA_AI_POSTGRES_PORT` in `.env`. The web app still reaches Postgres through
Docker's internal `db:5432` address. To force a specific host port:

```powershell
powershell -ExecutionPolicy Bypass -File .\setup.ps1 -PostgresPort 55432
```

```bash
bash setup.sh --postgres-port 55432
```

If you already bootstrapped artifacts and only want to start the app:

```bash
docker compose up --build
```

The Compose stack starts PostgreSQL 17 and initializes both the main `mma-ai`
database and the auxiliary `odds` database used by odds-related workflows. The
setup scripts restore the Hugging Face dumps into those databases.

During setup you can choose OpenAI, Codex/OpenAI-compatible, Anthropic Claude,
Google Gemini, xAI Grok, a local OpenAI-compatible server such as Ollama or LM
Studio, or a custom endpoint for Data-tab analytics and Train-tab chat. The
choices are saved in `.env` as `LLM_PROVIDER`, `LLM_MODEL`, `LLM_API_KEY`, and
optional `LLM_BASE_URL`. Non-interactive installs can pass values directly:

```powershell
powershell -ExecutionPolicy Bypass -File .\setup.ps1 `
  -LlmProvider anthropic `
  -LlmModel claude-3-5-sonnet-latest `
  -LlmApiKey "<token>"
```

```bash
bash setup.sh --llm-provider local --llm-model llama3.1 --llm-base-url http://host.docker.internal:11434/v1
```

For local development without Docker:

```bash
uv sync
uv run mma-web
```

## Seed Data And Database Bootstrap

This repo ships with current seed UFCStats CSVs at
`data/raw/ufcstats/competitions.csv` and `data/raw/ufcstats/individuals.csv`.
Generated training/prediction CSVs, trained models, screenshots, logs, and
database dumps stay out of git. Docker Compose bind-mounts local `data/` to
`/app/data`, so the checked-in seed CSVs are available to the web app and can be
updated in place.

Large PostgreSQL dumps for the main `mma-ai` database and the auxiliary `odds`
database are distributed through the companion Hugging Face Dataset:
`https://huggingface.co/datasets/DanMcInerney/mma-ai`. Import those once for the
fast bootstrap path. `setup.ps1` and `setup.sh` perform that import and extract
the pretrained `ag-20260304_110750-win-extreme` model as the initial Predict tab
model. After that, normal data updates are:

```bash
uv run mma-rebuild-db --scrape --reset-db
```

The UFCStats scraper is incremental by default. It reads the existing CSVs,
skips fighter URLs and event URLs already present, merges only newly discovered
fighters/fights, and preserves the existing rows. Use `uv run
mma-scrape-ufcstats --force-full` only when you intentionally want to rebuild
the raw CSVs from scratch.

To run the dashboard directly against your current host PostgreSQL databases:

```powershell
$env:DATABASE_URL = "postgresql://postgres:postgres@localhost:5432/mma-ai"
$env:ODDS_DATABASE_URL = "postgresql://postgres:postgres@localhost:5432/odds"
uv run mma-web
```

To run the Docker web app against an existing Postgres instance on your host,
copy `.env.example` to `.env` and set the Compose-specific URLs:

```env
MMA_AI_COMPOSE_DATABASE_URL=postgresql://postgres:postgres@host.docker.internal:5432/mma-ai
MMA_AI_COMPOSE_ODDS_DATABASE_URL=postgresql://postgres:postgres@host.docker.internal:5432/odds
```

Then run only the web service so Compose does not also start its bundled
Postgres service on port 5432:

```bash
docker compose up --build web
```

If you want a fully isolated Docker database instead, leave those Compose URLs
unset and import the Hugging Face dumps into the Compose Postgres volume. The
Data tab defaults to incremental UFCStats scrape plus generated-schema
recreation, so future raw CSV updates rebuild the local feature store without
requiring another dump import. Train/Predict also need generated CSVs and model
artifacts in the mounted `data/` and `AutoGluonModels/` folders, or you need to
run the Data and Train tabs to create them.

## Dashboard

- Data: refresh raw UFCStats CSVs, rebuild the PostgreSQL feature store, write
  finalized CSVs, and run read-only AI analytics over Postgres or finalized CSV
  fallbacks. The collapsed pipeline options include an opt-in BFO odds refresh
  that scrapes odds and recalculates odds features only when enabled.
- Train: launch model training with the existing `libs/modeling/train.py`
  defaults, keep advanced knobs collapsed, chat about training/features, and
  inspect saved evaluation artifacts.
- Predict: choose a model, load upcoming UFC events from Wikipedia, predict a
  selected event, or run a manual fighter-vs-fighter matchup with positive-EV
  output cards. Event prediction accepts manual fighter odds in the dashboard so
  web jobs never need to block on terminal prompts.

Each long-running Data, Train, or Predict job writes a debug log under
`data/logs/jobs` and exposes it through the dashboard and `/api/jobs/{job_id}/log`.

## Commands

```bash
uv run mma-scrape-ufcstats
uv run mma-rebuild-db
uv run mma-train
uv run mma-predict
uv run pytest
```

The dashboard uses the same command paths in background jobs so the UI does not
fork a separate feature or prediction implementation.

## Project Files

- `AGENTS.md` and `CLAUDE.md`: agent guidance for safe analytics, training,
  prediction, feature semantics, and test expectations.
- `Dockerfile` and `docker-compose.yml`: public release runtime with Postgres
  and the FastAPI dashboard.
- `libs/web`: FastAPI app, background jobs, web service adapters, analytics,
  training chat, evaluation summaries, and static UI.
- `data`: finalized CSV outputs such as `prediction_data.csv`,
  `training_data.csv`, and `training_data_dec.csv`.

## Architecture Reference

The remainder of this README preserves the technical feature-store guide from
the original `mma-ai-db` project.

**Last Updated:** 2026-01-01
**System Version:** Production (Post Tau & Decay Optimization)

---

## Table of Contents

1. [System Overview](#system-overview)
2. [High-Level Architecture](#high-level-architecture)
3. [Data Ingestion & Schema](#data-ingestion--schema)
4. [Feature Engineering Pipeline](#feature-engineering-pipeline)
5. [Calculator Execution Order & Dependencies](#calculator-execution-order--dependencies)
6. [Data Leakage Prevention](#data-leakage-prevention)
7. [Configuration System](#configuration-system)
8. [Parameter Optimization](#parameter-optimization)
9. [Training Data Creation](#training-data-creation)
10. [Critical Design Decisions](#critical-design-decisions)
11. [Installation & Setup](#installation--setup)
12. [Troubleshooting & FAQ](#troubleshooting--faq)

---

## System Overview

### Purpose

This system transforms raw UFC fight statistics into high-quality machine learning features for predicting fight outcomes. It implements:

- **Bayesian smoothing** to handle small sample sizes
- **Time-decayed averages** to weight recent performance more heavily
- **Opponent-adjusted performance** to account for strength of schedule
- **Temporal validation** to prevent data leakage
- **Automated parameter optimization** for smoothing and decay rates

### Key Components

1. **Data Ingestion** (`CoreFeatureStore`): Scrapes UFC Stats, loads into PostgreSQL
2. **Feature Engineering** (`main.py` + 45+ calculators): Transforms raw stats into 1000+ derived features
3. **Parameter Optimization** (`tuning/`): Optimizes tau (smoothing) and decay half-life values
4. **Training Data** (`CreateTrainingData` + `CleanTrainingData`): Builds model-ready datasets
5. **Configuration** (`config/`): Centralized parameters and decay rates

### Technology Stack

- **Database:** PostgreSQL (features schema)
- **Language:** Python 3.10-3.12
- **Key Libraries:** SQLAlchemy, pandas, numpy, scipy
- **Optimization:** Time-series cross-validation with Beta-Binomial/Negative Binomial likelihood

---

## High-Level Architecture

```
┌─────────────────┐
│  UFCStats.com   │  Raw fight data (scraper: libs/feature_store/core.py)
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ fight_stats_fe  │  Raw stats table (round 1 + totals)
└────────┬────────┘
         │ Lines 591-626: Basic derived features
         │ (ko, win, decision, sub_land, age, days_since_last_fight, etc.)
         │
         ▼
┌─────────────────┐
│fight_stats_     │  Copy of fight_stats_fe
│   derived       │  Lines 627-628: copy_to_derived(conn)
└────────┬────────┘
         │
         │ Lines 630-645: PARAMETER OPTIMIZATION (if needed)
         │ ├─ Check if config/optimized_parameters.json exists
         │ ├─ If missing: Run comprehensive_likelihood_tuner.py (30-60 min)
         │ └─ If exists: Load optimized tau values (<1 sec)
         │
         ▼
┌─────────────────┐
│   SMOOTHING     │  Bayesian shrinkage to handle small samples
│  (Lines 647-657)│  ├─ BetaBinomialCalculator: Binary outcomes (win, ko, decision)
│                 │  └─ PoissonGammaCalculator: Count data (sig_str_land, td_land)
└────────┬────────┘
         │ Lines 658: Rename smoothed_columns → Replace raw with smoothed
         │
         ▼
┌─────────────────┐
│  RATE/RATIO     │  Derived statistics
│  FEATURES       │  Lines 660-676:
│  (Lines 660-676)│  ├─ TotalCalculator: Cumulative sums
│                 │  ├─ AccuracyCalculator: landed / attempted
│                 │  ├─ DefenseCalculator: 1 - opponent_accuracy
│                 │  ├─ PerMinCalculator: stat / fight_minutes
│                 │  ├─ RatioCalculator: stat / total_fights
│                 │  └─ PressureCalculator: rd1_stat / total_stat
│                 │  └─ Delete _raw columns (line 668)
└────────┬────────┘
         │ Lines 679-680: Populate 45 feature-specific tables
         │
         ▼
┌─────────────────┐
│ Feature Tables  │  One table per stat category (body, head, td, ctrl, etc.)
│ (45 tables)     │  Lines 684-685: PerCalculator (ko_per_sig_str_land, etc.)
└────────┬────────┘
         │ Lines 687-688: OpponentCalculator (what opponents achieved)
         │ **CRITICAL:** Must run BEFORE historical aggregations
         │
         ▼
┌─────────────────┐
│  PRIORS &       │  Weightclass baselines for Bayesian models
│  BASELINES      │  Lines 695-710:
│  (Lines 695-710)│  ├─ WeightclassMeanCalculator: WC averages
│                 │  ├─ WeightclassMadCalculator: WC variability (MAD)
│                 │  ├─ FirstTimeMadCalculator: First-fighter MADs
│                 │  └─ MinimumMadCalculator: MAD floors (prevent /0)
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  HISTORICAL     │  Aggregations across fighter history
│  AGGREGATIONS   │  Lines 719-760:
│  (Lines 719-760)│  ├─ MedianAbsoluteDeviationCalculator: _mad (variability)
│                 │  ├─ AverageCalculator: _avg (simple mean)
│                 │  ├─ TimedecAvgCalculator: _dec_avg (time-weighted mean)
│                 │  │   └─ Uses decay half-life from config/decay.py
│                 │  │   └─ Runtime output: data/comprehensive_tuning/optimized_decay.json
│                 │  ├─ MinimumMadCalculator: Minimum MAD for adjperf
│                 │  └─ AdjustedPerformanceCalculator: _adjperf, _dec_adjperf
│                 │      └─ Opponent-adjusted z-scores with reliability weighting
└────────┬────────┘
         │ Lines 772-786: Scrape BFO odds & calculate odds features
         │
         ▼
┌─────────────────┐
│  FINAL TABLES   │  Feature tables with all layers (_avg, _dec_avg, _adjperf)
└────────┬────────┘
         │ Lines 795-825: CreateTrainingData (pattern filtering + merging)
         │
         ▼
┌─────────────────┐
│prediction_data  │  Wide-format DataFrame (fight_id, f1_*, f2_*, all features)
│    .csv         │  Used for inference (upcoming fights)
└────────┬────────┘
         │ Lines 838-848: CleanTrainingData (shifting + diffing + balancing)
         │
         ▼
┌─────────────────┐
│training_data    │  Model-ready DataFrame (feature_diffs, target, balanced)
│    .csv         │  Fed into XGBoost/LightGBM for training
└─────────────────┘
```

---

## Data Ingestion & Schema

### Data Source

**Source:** UFCStats.com (official UFC statistics)
**Scraper:** `libs/feature_store/core.py` → `CoreFeatureStore`
**Entry Point:** `main.py` lines 548-589

### Core Tables

#### `features.fight_stats_fe`
**Purpose:** Raw scraped fight statistics
**Created:** Lines 551-595 in `main.py`
**Key Columns:**
- `fight_id`, `fighter_id`, `event_id` (primary keys)
- **Round 1 stats:** `kd_rd1`, `sig_str_land_rd1`, `sig_str_att_rd1`, `td_land_rd1`, `td_att_rd1`, `sub_att_rd1`, `rev_rd1`, `ctrl_rd1`
- **Strike locations (round 1):** `head_land_rd1`, `head_att_rd1`, `body_land_rd1`, `body_att_rd1`, `leg_land_rd1`, `leg_att_rd1`, `distance_land_rd1`, `distance_att_rd1`, `clinch_land_rd1`, `clinch_att_rd1`, `ground_land_rd1`, `ground_att_rd1`
- **Total stats (aggregated):** Same as round 1 without `_rd1` suffix
- **Metadata:** `result`, `method`, `time_format`, `weightclass`, `fighter_dob`, `fighter_name`

#### `features.fight_stats_derived`
**Purpose:** Enhanced version with smoothed stats + derived features
**Created:** Line 628 (`copy_to_derived(conn)`)
**Additions:**
- Smoothed stats: `win_smooth`, `ko_smooth`, `sig_str_land_smooth`, etc.
- Derived features: `time_sec`, `age`, `days_since_last_fight`, `reach`, `ape`, `ufcage`
- After line 658: Smoothed columns replace raw (`rename_smoothed_columns()`)
- After line 668: Raw columns deleted (`delete_raw_columns()`)

#### Mapping Tables
- **`features.fighter_mapping`**: `fighter_id` ↔ `fighter_name`, `fighter_dob`
- **`features.event_mapping`**: `event_id` ↔ `event_date`
- **`features.fight_mapping`**: `fight_id`, `fighter1_id`, `fighter2_id`, `event_id`, `weightclass`

#### Feature-Specific Tables (45 tables)
**Created:** Lines 679-680 (`populate_feature_tables()`)
**Examples:**
- `features.age` – Age-related features
- `features.body` – Body strike features
- `features.head` – Head strike features
- `features.td` – Takedown features
- `features.ctrl` – Control time features
- `features.odds` – Betting odds

**Purpose:** Isolate features by category for layered calculations (`_avg`, `_dec_avg`, `_adjperf`)

---

## Feature Engineering Pipeline

### Execution Flow (main.py lines 598-770)

The pipeline executes calculators in strict order to satisfy dependencies:

```python
# 1. Basic Derived Features (Lines 598-626)
TimeSecCalculator(context).run()              # time_sec, time_sec_rd1
KOCalculator(context).run()                   # ko, ko_rd1
DecisionCalculator(context).run()             # decision
SubmissionslandCalculator(context).run()      # sub_land, sub_land_rd1
WinCalculator(context).run()                  # win, win_rd1
FullFightStatsCalculator(context).run()       # Aggregate round stats
AgeCalculator(context).run()                  # age (at fight date)
DaysSinceLastFightCalculator(context).run()   # days_since_last_fight
ReachCalculator(context).run()                # reach
HeightCalculator(context).run()               # height
ApeCalculator(context).run()                  # ape (reach - height)
UfcAgeCalculator(context).run()               # ufcage (years in UFC)

# 2. Copy to Derived Table (Line 628)
copy_to_derived(conn)  # fight_stats_fe → fight_stats_derived

# 3. Parameter Optimization Check (Lines 630-645)
optimized_params_path = Path('config/optimized_parameters.json')
if not optimized_params_path.exists():
    # Run tau optimization (30-60 minutes)
    from tuning.comprehensive_likelihood_tuner import main as run_tau_optimizer
    run_tau_optimizer()
else:
    # Use existing optimized parameters (<1 second)
    param_loader = get_default_parameter_loader()

# 4. Smoothing (Lines 647-657)
BetaBinomialCalculator(conn, param_loader).run()   # win, ko, decision, sub_land, ctrl
PoissonGammaCalculator(conn, param_loader).run()   # sig_str_land, td_land, kd, rev, etc.

# 5. Rename Smoothed Columns (Line 658)
rename_smoothed_columns(conn)  # _smooth → original name, original → _raw

# 6. Rate & Ratio Features (Lines 660-676)
TotalCalculator(conn).run()       # _total (cumulative sums)
AccuracyCalculator(conn).run()    # _acc (landed / attempted)
DefenseCalculator(conn).run()     # _def (1 - opponent_acc)
delete_raw_columns(conn)          # Delete _raw columns
PerMinCalculator(conn).run()      # _per_min (stat / minutes)
RatioCalculator(conn).run()       # _ratio (stat / total_fights)
PressureCalculator(conn).run()    # _pressure (rd1 / total)

# 7. Populate Feature Tables (Lines 679-680)
feature_groups = create_feature_specific_tables(conn)
populate_feature_tables(conn, feature_groups)

# 8. Per Features (Lines 684-685)
PerCalculator(conn).run()  # ko_per_sig_str_land, td_per_sig_str_att, etc.

# 9. Opponent Stats (Lines 687-688)
OpponentCalculator(conn).run()  # _opp (what opponents achieved)

# 10. Prior Distributions (Lines 695-710)
WeightclassMeanCalculator(conn).run()         # _wc_mean
WeightclassMadCalculator(conn).run()          # _wc_mad
FirstTimeMadCalculator(conn).run()            # First-fighter MADs
MinimumMadCalculator(conn, decay=False).run() # _minimum_mad

# 11. Historical Aggregations (Lines 719-760)
MedianAbsoluteDeviationCalculator(conn).run()  # _mad
AverageCalculator(conn).run()                  # _avg
TimedecAvgCalculator(conn, decay_rate_years).run()  # _dec_avg
MinimumMadCalculator(conn, decay=False).run()       # Minimum MAD for adjperf

# 12. Adjusted Performance (Lines 743-760)
AdjustedPerformanceCalculator(conn, decay=False).run()  # _adjperf
AdjustedPerformanceCalculator(conn, decay=True).run()   # _dec_adjperf
TimedecAvgCalculator(conn, decay_rate_years, include_patterns={'_adjperf'}).run()  # _adjperf_dec_avg
AverageCalculator(conn, include_patterns={'_adjperf', '_per_'}).run()  # _adjperf_avg

# 13. Odds Data (Lines 772-786)
BFOScraper(conn).scrape_all_fighters()
OddsCalculator(conn).run()

# 14. Training Data Creation (Lines 795-866)
CreateTrainingData(conn, include_patterns, exclude_patterns).create_training_data()
CleanTrainingData(df, include_patterns, exclude_patterns).clean_training_data()
```

---

## Calculator Execution Order & Dependencies

### Why Order Matters

**THE ORDER IS CRITICAL.** Each calculator depends on previous ones. Running out of order causes:
- **Missing columns** → SQL errors
- **Data leakage** → Invalid features
- **Incorrect calculations** → Wrong statistics

### Critical Ordering Rules

1. **Smoothing BEFORE rate features**: `BetaBinomialCalculator` and `PoissonGammaCalculator` MUST run before `AccuracyCalculator`, `DefenseCalculator`, `PerMinCalculator`, `RatioCalculator`
   - **Why:** Accuracy = smoothed(landed) / smoothed(attempted), NOT smoothed(landed / attempted)
   - **Impact:** Preserves Bayesian conjugate prior structure

2. **OpponentCalculator BEFORE historical aggregations**: `OpponentCalculator` MUST run before `AverageCalculator`, `TimedecAvgCalculator`, `MedianAbsoluteDeviationCalculator`
   - **Why:** Creates new `_opp` columns that need to be aggregated
   - **Impact:** Without this, `sig_str_land_opp_avg` won't exist for opponent strength calculations

3. **Priors BEFORE adjusted performance**: `WeightclassMeanCalculator`, `WeightclassMadCalculator`, `MinimumMadCalculator` MUST run before `AdjustedPerformanceCalculator`
   - **Why:** Adjperf uses weightclass priors for shrinkage
   - **Impact:** Missing priors → NaN adjperf values

4. **Historical aggregations BEFORE adjusted performance aggregations**: `AverageCalculator`, `TimedecAvgCalculator` on base stats MUST run before running them on `_adjperf` columns
   - **Why:** Can't aggregate what doesn't exist yet
   - **Impact:** Missing `_adjperf_dec_avg` columns

---

## Data Leakage Prevention

**DATA LEAKAGE = Using information from the future to predict the past**

This system has **7 CRITICAL ANTI-LEAKAGE MEASURES**:

### 1. Time-Decayed Averages: Strict Past-Only Filter

**Location:** `libs/feature_store/calculators/time_dec_avg_calc.py`

**The Problem:**
If we include the current fight in `_dec_avg` calculations, we leak the outcome into the predictor.

**The Solution:**
```sql
-- CRITICAL FILTER (time_dec_avg_calc.py line ~121)
WHERE past.event_date < current.event_date  -- STRICT inequality
```

**Example:**
```
Fighter A's fights:
├─ 2020-01-15: 50 strikes landed
├─ 2021-06-10: 75 strikes landed
└─ 2022-09-20: 100 strikes landed  ← CURRENT FIGHT

For 2022-09-20 prediction:
✓ CORRECT: Use only 2020-01-15 and 2021-06-10 (past < current)
✗ WRONG: Include 2022-09-20 (current fight)

Result: sig_str_land_dec_avg calculated from 2020 & 2021 only
```

---

### 2. Adjusted Performance: Historical Opponent Stats Only

**Location:** `libs/feature_store/calculators/adj_perf_calc.py`

**The Problem:**
Adjperf compares "what you did" vs "what opponents usually allow". If we include the current fight's opponent data, we leak information.

**The Solution:**
```sql
-- Step 1: Get opponent's HISTORICAL allowed stats (adj_perf_calc.py line ~280)
WITH opponent_history AS (
    SELECT opp_stats
    FROM past_fights
    WHERE opponent_id = current_opponent_id
      AND event_date < current_event_date  -- STRICT past only
)

-- Step 2: Calculate expected performance
expected = reliability_weighted_average(opponent_history)

-- Step 3: Compare to observed
adjperf = (observed - expected) / MAD
```

---

### 3. Training Data: Temporal Shifting

**Location:** `libs/feature_store/clean_training_data.py`

**The Problem:**
If we use Fight T's stats to predict Fight T's outcome, we leak the outcome.

**The Solution (Lines 150-180):**
```python
# Identify dynamic vs static features
static_columns = []  # age, reach, odds, _dec_avg (already past-only)
stat_columns = []    # All dynamic stats that need shifting

# CRITICAL: Shift dynamic stats by 1 fight
for col in stat_columns:
    if 'fighter1' in col:
        stats_df[col] = stats_df.groupby('fighter1_id')[col].shift(1)
    elif 'fighter2' in col:
        stats_df[col] = stats_df.groupby('fighter2_id')[col].shift(1)
```

**Example:**
```
Fighter A's fight history:
├─ Fight 100 (2020-01): 50 strikes, 1 KO
├─ Fight 200 (2021-06): 75 strikes, 0 KO
└─ Fight 300 (2022-09): 100 strikes, 1 KO

Training data WITHOUT shifting:
Row for Fight 300: sig_str_land=100, ko=1, target=win
                  ↑ LEAKAGE! We're using Fight 300's stats to predict Fight 300

Training data WITH shifting:
Row for Fight 300: sig_str_land_prev=75, ko_prev=0, target=win
                  ↑ CORRECT! We're using Fight 200's stats to predict Fight 300
```

**Why NOT Shift Static Features:**
- `age`, `reach`, `odds` are known PRE-FIGHT → No leakage
- `_dec_avg` features are calculated with `event_date < current_date` → Already exclude current fight → No leakage

---

### 4. Parameter Optimization: Time-Series Cross-Validation

**Location:** `tuning/comprehensive_likelihood_tuner.py`

**The Problem:**
If we optimize parameters (tau, decay half-life) using random cross-validation, we mix past and future data.

**The Solution (Lines 586-650):**
```python
from sklearn.model_selection import TimeSeriesSplit

# Time-series CV with gap
cv_configs = [
    TimeSeriesSplit(n_splits=3, gap=30),   # 30-fight gap prevents correlation
    TimeSeriesSplit(n_splits=3, gap=45),   # Different gap for stability
    TimeSeriesSplit(n_splits=4, gap=30),   # Different splits for robustness
]

# For each CV fold:
# ├─ Training: 2014-2023 (fixed)
# ├─ Gap: 30-45 fights (excluded)
# └─ Validation: 2024-2026 (future only)
```

---

### 5. Decay Optimization: Strict Train/Val Split

**Location:** `tuning/decay_rate_optimizer.py`

**The Problem:**
If we use the same data for both computing statistics AND evaluating decay rates, we overfit.

**The Solution (Lines 88-120):**
```python
class DecayRateOptimizer:
    def __init__(
        self,
        train_start: str = '2014-01-01',
        train_end: str = '2024-01-02',    # HARD CUT
        val_start: str = '2024-01-02',    # No overlap
        val_end: str = '2026-01-01'
    ):
        # Statistics computed on 2014-2024
        # Parameters evaluated on 2024-2026
        # ZERO overlap
```

---

## Configuration System

### Overview

**Location:** `config/` directory
**Purpose:** Centralized configuration for decay rates, tau parameters, and optimization settings

### Files

#### `config/decay.py`
**Purpose:** Time decay configuration for exponential weighting

**Key Function:**
```python
def get_decay_half_life_years() -> float:
    """
    Get time decay half-life in years.

    Priority:
    1. Environment variable DECAY_HALF_LIFE_YEARS (highest)
    2. Optimized runtime output from data/comprehensive_tuning/optimized_decay.json
    3. Default: 2.0 years (lowest)
    """
```

**Usage:**
```python
from config.decay import DECAY_HALF_LIFE_YEARS, DECAY_RATE

# In calculators:
decay_rate = DECAY_RATE  # ln(2) / half_life
weight = EXP(-decay_rate * days_diff / 365.25)
```

**Optimization:**
```bash
# Run decay optimization (finds optimal half-life)
uv run python tuning/decay_rate_optimizer.py

# Output: data/comprehensive_tuning/optimized_decay.json (ignored by git)
{
  "decay_half_life_years": 3.0,
  "nll": 3234.12,
  "improvement_pct": 1.16,
  "optimization_metadata": {
    "training_period": "2014-01-01 to 2024-01-01",
    "evaluation_period": "2024-01-02 to 2026-01-01"
  }
}
```

---

#### `config/optimized_parameters.json` (AUTO-GENERATED)
**Purpose:** Optimized tau values for Bayesian smoothing

**Generated By:** `tuning/comprehensive_likelihood_tuner.py` (lines 630-645 in main.py)

**Structure:**
```json
{
  "metadata": {
    "training_period": "2014-01-01 to 2024-01-01",
    "n_stats_tuned": 49,
    "optimized_at": "2026-01-01T12:00:00"
  },
  "beta_binomial": {
    "global": {
      "ko": 7.29,
      "win": 43.11,
      "decision": 60.0,
      "sub_land": 9.0,
      "ctrl": 2.0
    },
    "per_weightclass": {
      "featherweight": {
        "sub_land": 3.0
      }
    }
  },
  "poisson_gamma": {
    "global": {
      "sig_str": 0.98,
      "head": 0.98,
      "kd": 20.0,
      "td": 7.5
    }
  }
}
```

**Lifecycle:**
1. **First run:** `main.py` detects file missing → Runs optimization (30-60 min) → Saves file
2. **Subsequent runs:** Loads file (<1 sec)
3. **Re-optimization:** Delete file or `FORCE_REOPTIMIZE=1 uv run python main.py`

---

## Parameter Optimization

### Tau (Smoothing) Optimization

**Script:** `tuning/comprehensive_likelihood_tuner.py`
**Runtime:** 10-30 minutes
**Output:** `config/optimized_parameters.json`

**What It Does:**
Finds optimal tau values for:
1. **Beta-Binomial smoothing** (binary outcomes: ko, win, decision, sub_land, ctrl)
2. **Poisson-Gamma smoothing** (count data: sig_str_land, td_land, kd, rev)
3. **Accuracy smoothing** (ratios: sig_str_acc, td_acc, etc.)

**Methodology:**
```
For each stat:
1. Load raw data from fight_stats_fe (2014-2024)
2. Time-series CV split (3 folds, gap=30 fights)
3. Grid search tau values (25-30 candidates)
4. Compute negative log-likelihood (NLL) on validation
5. Select tau with lowest NLL
6. Test stability across CV configs
7. Compare per-weightclass vs global
8. Accept per-weightclass only if:
   ├─ Improvement ≥ 0.5%
   ├─ Stable across CV (variation < 20%)
   └─ Not on boundary of search range
```

---

### Decay Half-Life Optimization

**Script:** `tuning/decay_rate_optimizer.py`
**Runtime:** 2-4 hours
**Output:** `data/comprehensive_tuning/optimized_decay.json` (ignored by git)

**What It Does:**
Finds optimal time decay half-life for `_dec_avg` features.

**Methodology:**
```
1. Load fight data (2014-2024 train, 2024-2026 eval)
2. Extract base stats from vSeven_testing2 feature list (25 stats)
3. For each decay half-life candidate [2.0, 2.2, 2.4, 2.6, 2.8, 3.0]:
   a. Apply exponential decay weights in Python:
      weight = EXP(-ln(2) * days_diff / (half_life * 365.25))
   b. Calculate decayed averages for all stats
   c. Compute Mean Squared Error (MSE) on evaluation set
4. Select half-life with lowest MSE
```

---

## Training Data Creation

### CreateTrainingData (`libs/feature_store/create_training_data.py`)

**Purpose:** Build wide-format DataFrame with all features for both fighters

**Input:**
- Feature-specific tables (body, head, td, ctrl, age, odds, etc.)
- Fight mapping (fighter1_id, fighter2_id, event_id)

**Process (Lines 795-830 in main.py):**
```python
ctd = CreateTrainingData(
    conn,
    include_patterns={'dec_avg', 'age', 'reach', 'ufcage', 'odds',
                     'days_since_last_fight', 'time_sec', 'weightclass_encoded'},
    exclude_patterns=set(),
    required_features=set()
)
training_df = ctd.create_training_data()
```

**Output:** `data/prediction_data.csv` (for inference)

---

### CleanTrainingData (`libs/feature_store/clean_training_data.py`)

**Purpose:** Transform fight-level data into model-ready format

**Steps:**

1. **Split Static vs Dynamic Features**
2. **Shift Dynamic Features** (prevent data leakage)
3. **Create Differences** (f1 - f2)
4. **Balance Fighters** (prevent model bias)
5. **Create Target Variable**

**Output:** `data/training_data.csv`

---

## Critical Design Decisions

### 1. Smoothing Before Feature Engineering

**Decision:** Apply Beta-Binomial and Poisson-Gamma smoothing BEFORE calculating accuracy, defense, per-minute rates

**Rationale:**
- Preserves Bayesian conjugate prior
- Mathematically principled
- Better calibration (predictions match outcomes)

---

### 2. Time-Decayed Average Half-Life: 3.0 Years

**Decision:** Use 3.0 year half-life (optimized from 2.0 year default)

**Evidence:** generated optimization output at `data/comprehensive_tuning/optimized_decay.json` (not tracked)
```json
{
  "decay_half_life_years": 3.0,
  "nll": 3234.12,
  "baseline_nll": 3272.23,
  "improvement_pct": 1.16
}
```

**Impact:**
- Recent fights weighted heavily (12-month = 82% weight)
- Older fights still contribute (36-month = 50% weight)
- Better for established veterans with long histories

---

### 3. Train/Val Temporal Split (2014-2024 train, 2024-2026 eval)

**Decision:** Hard temporal split, no cross-validation for final evaluation

**Rationale:**
- User insight: "We're finding optimal scalar parameters, not training a complex model"
- No overfitting risk with simple scalars (tau=7.5, decay=3.0)
- Real test is when XGBoost model (using these features) predicts future fights
- Simpler = faster = easier to understand

**Impact:**
- 2-4 hour optimization (vs 10-20 hours with CV)
- Clear interpretation (performance on unseen future data)
- Matches production scenario (always predicting future)

---

## Installation & Setup

### Prerequisites

- **Python 3.10-3.12** (Python 3.12.4 recommended)
- PostgreSQL database
- **uv package manager**
- Optional GPU for faster model training

### Standard Installation

1. **Clone and install:**
   ```bash
   git clone <repository-url>
   cd mma-ai
   uv python install 3.12.4
   uv sync
   ```

2. **Configure local environment:**
   ```bash
   cp .env.example .env
   # The example URLs match Docker Compose's localhost Postgres defaults.
   # Edit them if your PostgreSQL setup uses different credentials, host, or DB names.
   ```

3. **Restore shared database artifacts (fast path):**
   Download the Hugging Face dataset artifacts:

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

   Copy convenience CSVs and extract the pretrained win model:

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

4. **Scrape UFCStats from this repo (incremental raw CSV update):**
   ```bash
   uv run python -m scripts.scrape_ufcstats
   ```

5. **Recreate the database schemas and training CSVs from the CSVs:**
   ```bash
   uv run python main.py --reset-db
   ```

   You can combine the incremental scrape and database recreation in one command:
   ```bash
   uv run python main.py --scrape --reset-db
   ```

6. **Train a model:**
   ```bash
   uv run python -m libs.modeling.train --model-type win
   ```

7. **Run predictions:**
   ```bash
   uv run python predict.py --model-type win --no-shap
   ```

### Configuration

Root `.env` example:
```env
DATABASE_URL=postgresql://postgres:postgres@localhost:5432/mma-ai
ODDS_DATABASE_URL=postgresql://postgres:postgres@localhost:5432/odds
THE_ODDS_API_KEY=
LLM_PROVIDER=
LLM_MODEL=
LLM_API_KEY=
LLM_BASE_URL=
OPENAI_API_KEY=
ANTHROPIC_API_KEY=
XAI_API_KEY=
GOOGLE_API_KEY=
GEMINI_API_KEY=
```

Generated raw scrape CSVs default to `data/raw/ufcstats/`. Training outputs default to `data/`. You can override paths with `MMA_AI_UFCSTATS_DIR`, `MMA_AI_DATA_DIR`, `MMA_AI_MODELS_DIR`, and `MMA_AI_PICKS_DIR`.

---

## Troubleshooting & FAQ

### Common Issues

#### Issue: "Database not found"
```
psycopg2.OperationalError: database "mma-ai" does not exist
```

**Solution:**
```bash
createdb -U postgres mma-ai
```

---

#### Issue: "optimized_parameters.json not found"

**Expected Behavior:** First run will trigger optimization (30-60 minutes)

**To Force Re-Optimization:**
```bash
FORCE_REOPTIMIZE=1 uv run python main.py
```

---

### FAQ

**Q: Why do some features have `_prev` suffix?**
A: Shifted features. `f1_prev_sig_str_land` means Fighter 1's `sig_str_land` from their previous fight (T-1), preventing data leakage.

**Q: What's the difference between `_avg` and `_dec_avg`?**
A: `_avg` = simple mean, `_dec_avg` = time-weighted mean with exponential decay (3.0yr half-life).

**Q: How do I add a new feature?**
A:
1. Create calculator in `libs/feature_store/calculators/`
2. Inherit from `BaseCalculator`
3. Implement `calculate_for_table()` or `run()`
4. Add to pipeline in `main.py` (respect execution order!)
5. Rebuild database: `uv run python main.py --reset-db`

---

## Quick Reference

### Key Commands

```bash
# Scrape raw UFCStats data
uv run python -m scripts.scrape_ufcstats

# Rebuild feature database and CSVs
uv run python main.py --reset-db

# Scrape, reset, and rebuild in one command
uv run python main.py --scrape --reset-db

# Train a win model
uv run python -m libs.modeling.train --model-type win

# Predict next event with latest win model
uv run python predict.py --model-type win --no-shap

# Optimize tau parameters (30-60 minutes)
uv run python tuning/comprehensive_likelihood_tuner.py

# Optimize decay half-life (2-4 hours)
uv run python tuning/decay_rate_optimizer.py

# Override decay rate temporarily
DECAY_HALF_LIFE_YEARS=2.5 uv run python main.py

# Force re-optimize parameters
FORCE_REOPTIMIZE=1 uv run python main.py
```

---

## Document History

| Version | Date | Changes |
|---------|------|---------|
| 1.0 | 2025-11-22 | Initial comprehensive documentation |
| 2.0 | 2026-01-01 | Updated with tau & decay optimization details, data leakage prevention, configuration system |

---

**For questions, issues, or contributions:** File an issue at the repository or contact the data science team.
