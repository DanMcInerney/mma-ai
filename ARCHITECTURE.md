# MMA AI Pipeline Architecture

**Complete technical documentation of the UFC fight prediction pipeline from raw data to inference.**

---

## Table of Contents

1. [Overview](#overview)
2. [Data Source & Ingestion](#data-source--ingestion)
3. [Database Schema](#database-schema)
4. [Feature Engineering Pipeline](#feature-engineering-pipeline)
5. [Calculator Reference](#calculator-reference)
6. [Training Data Creation](#training-data-creation)
7. [Inference Pipeline](#inference-pipeline)
8. [Key Design Decisions](#key-design-decisions)

---

## Overview

### High-Level Architecture

```
┌─────────────────┐
│   UFCStats.com  │ Raw fight data
└────────┬────────┘
         │ Scraping (CoreFeatureStore)
         ▼
┌─────────────────┐
│  fight_stats_fe │ Features extracted from raw data
└────────┬────────┘
         │ Copy to derived
         ▼
┌─────────────────┐
│fight_stats_     │ Smoothed stats + derived features
│   derived       │
└────────┬────────┘
         │ Populate feature-specific tables
         ▼
┌─────────────────┐
│ Feature Tables  │ age, body, head, td, ctrl, etc. (45 tables)
│ (features.*)    │
└────────┬────────┘
         │ Layer calculations (avg, dec_avg, adjperf, etc.)
         ▼
┌─────────────────┐
│  Layered Feats  │ age_dec_avg, body_land_dec_adjperf, etc.
└────────┬────────┘
         │ CreateTrainingData
         ▼
┌─────────────────┐
│ Training Data   │ Unshifted, fight-level features
└────────┬────────┘
         │ CleanTrainingData (shifting + diffing)
         ▼
┌─────────────────┐
│  Final Model    │ training_data.csv (fighter1 vs fighter2)
│    Training     │
└─────────────────┘

┌─────────────────┐
│ New Fight       │ Upcoming fight to predict
└────────┬────────┘
         │ CreateInferenceData
         ▼
┌─────────────────┐
│   Inference     │ Recalculated features for both fighters  
│   Features      │
└────────┬────────┘
         │ Model prediction
         ▼
┌─────────────────┐
│  Win Prob       │ P(Fighter 1 wins)
└─────────────────┘
```

---

## Data Source & Ingestion

### 1. Raw Data Source

**Source:** UFCStats.com  
**Scraper:** `CoreFeatureStore` ([libs/feature_store/core.py](libs/feature_store/core.py))

**Scraped Data:**
- Fighter information (name, DOB, height, reach, stance)
- Event information (date, location)
- Fight-level stats:
  - **Round 1 stats**: KD, strikes, takedowns, submissions, control time
  - **Total fight stats**: All round 1 metrics aggregated
  - **Strike locations**: Head, body, leg, distance, clinch, ground
  - **Fight outcome**: Win/loss, method (KO/TKO, Submission, Decision), time

### 2. Database Ingestion

**Entry Point:** `main.py` → `CoreFeatureStore.run()`

**Tables Created:**
1. `features.fight_stats_fe` – Raw scraped data with basic feature engineering
2. `features.fighter_mapping` – Fighter ID → Name mapping
3. `features.event_mapping` – Event ID → Date mapping  
4. `features.fight_mapping` – Fight ID → Fighter IDs, Event ID

**Key Fields in `fight_stats_fe`:**
```sql
fight_id, fighter_id, event_id,
-- Round 1 stats
kd_rd1, sig_str_land_rd1, sig_str_att_rd1, td_land_rd1, td_att_rd1,
sub_att_rd1, rev_rd1, ctrl_rd1, head_land_rd1, body_land_rd1, leg_land_rd1,
distance_land_rd1, clinch_land_rd1, ground_land_rd1,
-- Total stats  
kd, sig_str_land, sig_str_att, td_land, td_att,
sub_att, rev, ctrl, head_land, body_land, leg_land,
distance_land, clinch_land, ground_land,
-- Metadata
result, method, time_format, fighter_dob, fighter_name, weightclass
```

---

## Database Schema

### Core Tables

#### `features.fight_stats_fe`
Raw scraped data with basic feature extraction.

#### `features.fight_stats_derived`
Enhanced version with:
- Smoothed statistics (Poisson-Gamma, Beta-Binomial)
- Derived features (age, days_since_last_fight, ko, decision, win)
- Copied from `fight_stats_fe` via `copy_to_derived()`

#### Feature-Specific Tables (45 tables)
One table per stat category, created via `create_feature_specific_tables()`:

**Examples:**
- `features.age` – Age-related features
- `features.body` – Body strike features (land, att, acc, def, ratio, per_min, etc.)
- `features.head` – Head strike features
- `features.td` – Takedown features
- `features.ctrl` – Control time features
- `features.odds` – Betting odds features

**Purpose:** Isolate features by category for layered calculations (avg, dec_avg, adjperf).

### Mapping Tables

- `features.fighter_mapping` – `fighter_id` ↔ `fighter_name`
- `features.event_mapping` – `event_id` ↔ `event_date`
- `features.fight_mapping` – `fight_id`, `fighter1_id`, `fighter2_id`, `event_id`, `weightclass`

---

## Feature Engineering Pipeline

### Execution Order (main.py lines 590-770)

```
1. Basic Derived Features
   ├─ TimeSecCalculator         → time_sec, time_sec_rd1
   ├─ KOCalculator               → ko, ko_rd1
   ├─ DecisionCalculator         → decision
   ├─ SubmissionslandCalculator  → sub_land, sub_land_rd1
   ├─ WinCalculator              → win, win_rd1
   ├─ FullFightStatsCalculator   → Aggregate round stats
   ├─ AgeCalculator              → age (at fight date)
   ├─ DaysSinceLastFightCalculator → days_since_last_fight
   ├─ ReachCalculator            → reach
   ├─ HeightCalculator           → height
   ├─ ApeCalculator              → ape (reach - height)
   └─ UfcAgeCalculator           → ufcage (years in UFC)

2. Copy to Derived
   └─ copy_to_derived()          → Populate fight_stats_derived

3. Smoothing (Address Small Sample Sizes)
   ├─ BetaBinomialCalculator     → Smooth binary outcomes (win, ko, sub_land)
   └─ PoissonGammaCalculator     → Smooth count stats (sig_str_land, td_land)

4. Rename Smoothed Columns
   └─ rename_smoothed_columns()  → Replace raw with smoothed

5. Rate & Ratio Features
   ├─ TotalCalculator            → _total columns (cumulative sums)
   ├─ AccuracyCalculator         → _acc (landed / attempted)
   ├─ DefenseCalculator          → _def (1 - opponent_landing_rate)
   ├─ PerMinCalculator           → _per_min (stat / minutes_fought)
   ├─ RatioCalculator            → _ratio (stat / total_fights)
   └─ PressureCalculator         → _pressure (rd1 / total)

6. Populate Feature Tables
   └─ populate_feature_tables()  → Split into feature-specific tables

7. Custom Features
   └─ PerCalculator              → Custom per-stat ratios (ko_per_sig_str_land, etc.)

8. Opponent Stats
   └─ OpponentCalculator         → _opp (what opponents achieved)

9. Prior Distributions
   ├─ WeightclassMeanCalculator  → _wc_mean (weightclass averages)
   ├─ WeightclassMadCalculator   → _wc_mad (weightclass MAD)
   ├─ FirstTimeMadCalculator     → First-time fighter MADs
   └─ MinimumMadCalculator       → _minimum_mad (floor for MAD)

10. Historical Aggregations
    ├─ MedianAbsoluteDeviationCalculator → _mad (median absolute deviation)
    ├─ AverageCalculator          → _avg (mean of past fights)
    └─ TimedecAvgCalculator       → _dec_avg (time-decayed average, 1.0yr half-life)

11. Adjusted Performance
    └─ AdjustedPerformanceCalculator → _adjperf, _dec_adjperf (opponent-adjusted z-scores)

12. Odds Data
    ├─ BFOScraper                 → Scrape BestFightOdds.com
    └─ OddsCalculator             → Process betting odds
```

---

## Calculator Reference

### Base Calculators

All calculators inherit from `BaseCalculator` which provides:
- Database connection management
- SQL execution utilities
- Pattern-based feature filtering
- Layer update mechanism

### Category 1: Basic Derived Features

#### **TimeSecCalculator** ([time_sec_calc.py](libs/feature_store/calculators/time_sec_calc.py))
**Purpose:** Calculate fight duration in seconds  
**Formula:**
```python
time_sec = round * 300 + (minute * 60 + second)
time_sec_rd1 = minute_rd1 * 60 + second_rd1
```
**Output:** `time_sec`, `time_sec_rd1`

#### **KOCalculator** ([ko_calc.py](libs/feature_store/calculators/ko_calc.py))
**Purpose:** Binary indicator for KO/TKO finishes  
**Formula:**
```python
ko = 1 if method in ['KO/TKO', 'TKO - Doctor\'s Stoppage'] else 0
ko_rd1 = 1 if ko == 1 and time_sec < 300 else 0
```
**Output:** `ko`, `ko_rd1`

#### **DecisionCalculator** ([decision_calc.py](libs/feature_store/calculators/decision_calc.py))
**Purpose:** Binary indicator for decision outcomes  
**Formula:**
```python
decision = 1 if 'Decision' in method else 0
```
**Output:** `decision`

#### **SubmissionslandCalculator** ([sub_land_calc.py](libs/feature_store/calculators/sub_land_calc.py))
**Purpose:** Binary indicator for successful submissions  
**Formula:**
```python
sub_land = 1 if method == 'Submission' and result == 'win' else 0
sub_land_rd1 = 1 if sub_land == 1 and time_sec < 300 else 0
```
**Output:** `sub_land`, `sub_land_rd1`

#### **WinCalculator** ([win_calc.py](libs/feature_store/calculators/win_calc.py))
**Purpose:** Binary indicator for wins  
**Formula:**
```python
win = 1 if result == 'win' else 0
win_rd1 = 1 if win == 1 and time_sec < 300 else 0
```
**Output:** `win`, `win_rd1`

#### **AgeCalculator** ([age_calc.py](libs/feature_store/calculators/age_calc.py))
**Purpose:** Fighter's age at time of fight  
**Formula:**
```sql
age = EXTRACT(YEAR FROM AGE(event_date, fighter_dob))
```
**Output:** `age` (years)

#### **DaysSinceLastFightCalculator** ([dslf_calc.py](libs/feature_store/calculators/dslf_calc.py))
**Purpose:** Layoff duration since previous fight  
**Formula:**
```sql
days_since_last_fight = current_event_date - LAG(event_date) OVER (PARTITION BY fighter_id ORDER BY event_date)
```
**Output:** `days_since_last_fight` (days)

#### **UfcAgeCalculator** ([ufc_age_calc.py](libs/feature_store/calculators/ufc_age_calc.py))
**Purpose:** Years of experience in UFC  
**Formula:**
```sql
ufcage = EXTRACT(YEAR FROM AGE(current_event_date, first_ufc_fight_date))
```
**Output:** `ufcage` (years)

### Category 2: Smoothing (Bayesian Shrinkage)

#### **BetaBinomialCalculator** ([beta_binomial_calc.py](libs/feature_store/calculators/beta_binomial_calc.py))
**Purpose:** Smooth binary outcomes (win/loss, ko, submission) using Beta-Binomial conjugate prior  
**Why:** Prevent overfitting for fighters with few fights (e.g., 1-0 record → 100% win rate)

**Formula:**
```python
smoothed_rate = (successes + α) / (attempts + α + β)
```
Where:
- α, β = Beta prior hyperparameters (tuned per stat)
- Shrinks towards league-wide average for new fighters
- Converges to true rate with more data

**Process:**
1. Rename original columns to `{col}_raw`
2. Calculate smoothed versions as `{col}_smooth`
3. Later renamed to replace originals

**Applies To:** `win`, `ko`, `sub_land`, `decision`

#### **PoissonGammaCalculator** ([poisson_gamma_smoothing_calc.py](libs/feature_store/calculators/poisson_gamma_smoothing_calc.py))
**Purpose:** Smooth count statistics using Poisson-Gamma conjugate prior  
**Why:** Prevent noise from small samples (e.g., 1 fight with 10 strikes landed)

**Formula:**
```python
smoothed_count = (total_count + α) / (total_exposure + β)
```
Where:
- α, β = Gamma prior hyperparameters  
- exposure = minutes fought
- Shrinks towards league-wide rate for new fighters

**Applies To:** `sig_str_land`, `sig_str_att`, `td_land`, `td_att`, `sub_att`, `kd`, `head_land`, `body_land`, `leg_land`, etc.

### Category 3: Rate & Ratio Features

#### **TotalCalculator** ([total_calc.py](libs/feature_store/calculators/total_calc.py))
**Purpose:** Cumulative sums of stats  
**Formula:**
```sql
{stat}_total = SUM({stat}) OVER (PARTITION BY fighter_id ORDER BY event_date ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW)
```
**Output:** `sig_str_land_total`, `td_land_total`, etc.

#### **AccuracyCalculator** ([acc_calc.py](libs/feature_store/calculators/acc_calc.py))
**Purpose:** Striking/takedown accuracy  
**Formula:**
```sql
{stat}_acc = {stat}_land / NULLIF({stat}_att, 0)
```
**Output:** `sig_str_acc`, `td_acc`, `head_acc`, etc.

#### **DefenseCalculator** ([def_calc.py](libs/feature_store/calculators/def_calc.py))
**Purpose:** Defensive ability (1 - opponent's success rate)  
**Formula:**
```sql
{stat}_def = 1 - (opponent_{stat}_land / NULLIF(our_{stat}_att, 0))
```
**Output:** `sig_str_def`, `td_def`, etc.

#### **PerMinCalculator** ([per_min_calc.py](libs/feature_store/calculators/per_min_calc.py))
**Purpose:** Per-minute rates (normalize for fight duration)  
**Formula:**
```sql
{stat}_per_min = {stat} / (time_sec / 60.0)
```
**Output:** `sig_str_land_per_min`, `td_land_per_min`, etc.

#### **RatioCalculator** ([ratio_calc.py](libs/feature_store/calculators/ratio_calc.py))
**Purpose:** Per-fight averages  
**Formula:**
```sql
{stat}_ratio = {stat}_total / num_fights
```
**Output:** `sig_str_land_ratio`, `win_ratio`, etc.

#### **PressureCalculator** ([pressure_calc.py](libs/feature_store/calculators/pressure_calc.py))
**Purpose:** Round 1 intensity  
**Formula:**
```sql
{stat}_pressure = {stat}_rd1 / NULLIF({stat}, 0)
```
**Output:** `sig_str_land_pressure`

### Category 4: Opponent-Relative Features

#### **OpponentCalculator** ([opp_calc.py](libs/feature_store/calculators/opp_calc.py))
**Purpose:** What did opponents achieve against this fighter?  
**Why:** "Allowed" stats reveal defensive weaknesses

**Formula:**
```sql
{stat}_opp = opponent's {stat} when fighting this fighter
```

**Process:**
1. For each fight, identify opponent
2. Get opponent's performance in that fight
3. Aggregate as historical "allowed" stat

**Output:** `sig_str_land_opp`, `td_land_opp`, `ko_opp`, `sub_land_opp`, etc.

### Category 5: Prior Distributions

#### **WeightclassMeanCalculator** ([weightclass_mean_calc.py](libs/feature_store/calculators/weightclass_mean_calc.py))
**Purpose:** League-wide averages per weight class  
**Usage:** Shrinkage target for adjusted performance

**Formula:**
```sql
SELECT weightclass, AVG({stat}) as {stat}_wc_mean
FROM features.{table}
GROUP BY weightclass
```

**Output Tables:** `{table}_wc_mean` for each feature table

#### **WeightclassMadCalculator** ([weightclass_mad_calc.py](libs/feature_store/calculators/weightclass_mad_calc.py))
**Purpose:** League-wide variability (MAD) per weight class  
**Usage:** Scale for adjusted performance z-scores

**Formula:**
```sql
SELECT weightclass, 
       PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY ABS({stat} - median)) as {stat}_wc_mad
FROM features.{table}
GROUP BY weightclass
```

**Output Tables:** `{table}_wc_mad` for each feature table

#### **MinimumMadCalculator** ([minimum_mad_calc.py](libs/feature_store/calculators/minimum_mad_calc.py))
**Purpose:** Floor for MAD to prevent division by zero  
**Formula:**
```sql
{stat}_minimum_mad = MAX(0.001, global_mad * 0.1)
```
**Output Tables:** `{table}_minimum_mad`

### Category 6: Historical Aggregations

#### **AverageCalculator** ([avg_calc.py](libs/feature_store/calculators/avg_calc.py))
**Purpose:** Simple mean of past fights  
**Formula:**
```sql
{stat}_avg = AVG({stat}) 
OVER (PARTITION BY fighter_id ORDER BY event_date 
      ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW)
```
**Output:** `sig_str_land_avg`, `td_land_avg`, etc.

#### **MedianAbsoluteDeviationCalculator** ([mad_calc.py](libs/feature_store/calculators/mad_calc.py))
**Purpose:** Measure of variability (robust to outliers)  
**Formula:**
```sql
mad = MEDIAN(ABS(x - MEDIAN(x)))
```
**Output:** `sig_str_land_mad`, `td_land_mad`, etc.

#### **TimedecAvgCalculator** ([time_dec_avg_calc.py](libs/feature_store/calculators/time_dec_avg_calc.py))
**Purpose:** Time-weighted average (recent fights matter more)  
**Why:** Fighter skill/physical condition evolves over time

**Formula:**
```sql
weight = EXP(-ln(2) * time_gap_years / 1.0)  -- 1.0 year half-life
{stat}_dec_avg = SUM({stat} * weight) / SUM(weight)
```

**Key Details:**
- **Half-life:** 1.0 year (12-month-old fight = 50% weight)
- **Temporal filter:** Uses strictly past data (`event_date < current_date`)
- **Decay reference:** Relative to current fight date

**Output:** `age_dec_avg`, `sig_str_land_dec_avg`, `td_land_dec_avg`, etc.

### Category 7: Adjusted Performance

#### **AdjustedPerformanceCalculator** ([adj_perf_calc.py](libs/feature_store/calculators/adj_perf_calc.py))
**Purpose:** Opponent-adjusted z-scores (strength of schedule adjustment)  
**Why:** Landing 50 strikes vs. elite defense ≠ landing 50 vs. weak defense

**Formula:**
```python
# 1. Opponent history: What did opponent's past opponents achieve?
opp_mean = reliability_weighted_average(opponent_allowed_stats)
opp_mad = reliability_weighted_MAD(opponent_allowed_stats)

# 2. Shrink towards weightclass prior
n_fights = opponent_history_count
w_mean = n_fights / (n_fights + K_mean)  # K_mean = 4.0
w_mad = n_fights / (n_fights + K_mad)    # K_mad = 4.0

mu_shrunk = w_mean * opp_mean + (1 - w_mean) * wc_mean
mad_shrunk = max(w_mad * opp_mad + (1 - w_mad) * wc_mad, mad_floor)

# 3. Z-score with winsorization
adjperf = clip((observed - mu_shrunk) / mad_shrunk, -7, +7)
```

**Modes:**
- **Non-decayed (`_adjperf`)**: Equal weight to all opponent history
- **Time-decayed (`_dec_adjperf`)**: 1.0yr half-life on opponent history (weight = EXP(-0.6931 * years))

**Output:** `sig_str_land_adjperf`, `sig_str_land_dec_adjperf`, etc.

**Key Insight:** Positive adjperf = performed better than expected given opponent quality

### Category 8: Betting Odds

#### **BFOScraper** ([libs/bfo_scraper.py](libs/bfo_scraper.py))
**Purpose:** Scrape historical betting odds from BestFightOdds.com  
**Data:** Opening odds, closing odds, 7-day prior odds

#### **OddsCalculator** ([odds_calc.py](libs/feature_store/calculators/odds_calc.py))
**Purpose:** Process and normalize betting odds  
**Output:**
- `ip_opening_odds` – Implied probability from opening line
- `ip_closing_odds` – Implied probability from closing line  
- `sevenday_ip_opening_odds` – Odds from 7 days before fight
- `sevenday_vigless_ip_opening_odds` – Vig-removed odds

**Normalization:**
```python
# Remove bookmaker vig (overround)
true_prob_A = decimal_odds_A / (decimal_odds_A + decimal_odds_B)
true_prob_B = decimal_odds_B / (decimal_odds_A + decimal_odds_B)
```

---

## Training Data Creation

### Step 1: CreateTrainingData ([create_training_data.py](libs/feature_store/create_training_data.py))

**Purpose:** Build wide-format training dataframe with all features  

**Process:**
```python
# 1. Load fight mapping
fights = SELECT fight_id, fighter1_id, fighter2_id, event_id, weightclass

# 2. For each feature table (age, body, head, td, etc.):
#    - Load features for all fighters
#    - Merge onto fights dataframe
#    - Create f1_{feature} and f2_{feature} columns

# 3. Encode weightclass
weightclass_encoded = one_hot_encode(weightclass)

# 4. Filter features by patterns
if include_patterns:
    features = [col for col in df.columns if any(pattern in col for pattern in include_patterns)]

# 5. Save as prediction_data.csv (for inference)
```

**Output DataFrame Schema:**
```
fight_id | fighter1_id | fighter2_id | event_id | event_date |
f1_age | f2_age | f1_sig_str_land_dec_avg | f2_sig_str_land_dec_avg |
f1_td_land_dec_adjperf | f2_td_land_dec_adjperf | ... |
result | fighter_dob | weightclass_encoded
```

**Key:** All features are **unshifted** (row T contains data calculated at time T)

### Step 2: CleanTrainingData ([clean_training_data.py](libs/feature_store/clean_training_data.py))

**Purpose:** Transform fight-level data into model-ready format  

**Process:**

#### 2.1 Load and Split Data
```python
# Load fights
df = training_df

# Split into static vs dynamic features
static_columns = []  # age, reach, odds, weightclass_encoded, _dec_avg
stat_columns = []    # All dynamic stats

for col in df.columns:
    if any(static_feat in col for static_feat in BASE_STATIC_FEATS):
        static_columns.append(col)
    elif col.endswith('_dec_avg'):  # Time-decayed avgs are static
        static_columns.append(col)
    else:
        stat_columns.append(col)

static_df = df[static_columns]
stats_df = df[stat_columns]
```

#### 2.2 Shift Dynamic Features
```python
# CRITICAL: Shift dynamic stats by 1 row to prevent leakage
# Row T prediction uses T-1 data

for col in stat_columns:
    if 'fighter1' in col:
        stats_df[col] = stats_df.groupby('fighter1_id')[col].shift(1)
    elif 'fighter2' in col:
        stats_df[col] = stats_df.groupby('fighter2_id')[col].shift(1)

# Rename shifted columns
stats_df.columns = [col.replace('f1_', 'f1_prev_').replace('f2_', 'f2_prev_') for col in stats_df.columns]
```

**Why Shift?**
- **Without shifting:** Row T uses Fight T's stats → Leakage (model sees outcome stats)
- **With shifting:** Row T uses Fight T-1's stats → No leakage

**Why NOT Shift Static Features?**
- `age`, `reach`, `odds` are known pre-fight → No leakage
- `_dec_avg` features are calculated using strictly past data (`< event_date`) → Already exclude current fight → No leakage

#### 2.3 Create Differences
```python
# Fighter1 - Fighter2 differences
for stat in stats:
    df[f'{stat}_diff'] = df[f'f1_prev_{stat}'] - df[f'f2_prev_{stat}']

for static in static_stats:
    df[f'{static}_diff'] = df[f'f1_{static}'] - df[f'f2_{static}']
```

#### 2.4 Balance Fighters
```python
# Flip 50% of fights so fighter2 is on left
# Prevents model from learning "fighter1 always wins"
balanced_df = FighterBalancer().balance(df)
```

#### 2.5 Create Target Variable
```python
if target_type == 'win':
    y = (balanced_df['result'] == 'win').astype(int)
elif target_type == 'decision':
    y = (balanced_df['decision'] == 1).astype(int)
```

**Output:** `training_data.csv`

**Final Schema:**
```
fight_id | fighter1_id | fighter2_id |
age_diff | reach_diff | sig_str_land_dec_avg_diff | 
sig_str_land_dec_adjperf_diff | td_land_dec_adjperf_diff | ... |
result | target
```

---

## Inference Pipeline

### Overview

**Goal:** Predict outcome of an upcoming fight between Fighter A vs Fighter B

**Challenge:** New fights don't exist in database yet → Must construct features programmatically

### Step 1: CreateInferenceData ([create_inference_data.py](libs/feature_store/create_inference_data.py))

**Input:**
- `prediction_data.csv` – Historical training data  
- `upcoming_fight` – (fighter1_name, fighter2_name, fight_date)

**Process:**

#### 1.1 Load Historical Data
```python
# Load prediction_data.csv
historical_df = pd.read_csv('prediction_data.csv')

# Filter for upcoming fighters
fighter1_history = historical_df[historical_df['fighter1_id'] == fighter1_id | 
                                historical_df['fighter2_id'] == fighter1_id]
fighter2_history = historical_df[historical_df['fighter1_id'] == fighter2_id | 
                                historical_df['fighter2_id'] == fighter2_id]
```

#### 1.2 Create Placeholder Row for Upcoming Fight
```python
# For each fighter:
# 1. Take last row of historical data
# 2. Duplicate it
# 3. Set event_date = upcoming_fight_date

last_row = fighter1_history.iloc[-1].copy()
upcoming_row = last_row.copy()
upcoming_row['event_date'] = upcoming_fight_date

fighter1_df = pd.concat([fighter1_history, upcoming_row])
```

#### 1.3 Recalculate Static Features
```python
# Update age for upcoming fight
age = (upcoming_date - fighter_dob).total_seconds() / (365.25 * 24 * 60 * 60)

# Update days_since_last_fight  
days_since_last = (upcoming_date - last_fight_date).days

# Update ufcage
ufcage = (upcoming_date - first_ufc_fight_date).total_seconds() / (365.25 * 24 * 60 * 60)
```

#### 1.4 Recalculate Time-Decayed Averages
```python
# StatUpdater.update_dec_avgs()
decay_rate = math.log(2) / 1.0  # 1.0 year half-life

# Include ALL fights (including current)
all_fights = fighter_df.copy()

# Calculate time differences from upcoming fight
all_fights['time_diff_years'] = (upcoming_date - all_fights['event_date']).dt.total_seconds() / (365.25 * 24 * 60 * 60)

# Calculate decay weights
all_fights['weight'] = all_fights['time_diff_years'].apply(lambda x: math.exp(-decay_rate * x))

# For each stat (age, reach, days_since_last_fight):
for stat in BASE_STATIC_FEATS:
    weighted_sum = (all_fights[stat] * all_fights['weight']).sum()
    sum_weights = all_fights['weight'].sum()
    
    upcoming_row[f'{stat}_dec_avg'] = weighted_sum / sum_weights
```

**Key Detail:** Current fight has `time_diff = 0` → `weight = exp(0) = 1.0` → Maximum weight to current values (consistent with training)

#### 1.5 Merge Both Fighters
```python
# Construct final prediction row
prediction_row = {
    'f1_age': fighter1_age,
    'f2_age': fighter2_age,
    'f1_age_dec_avg': fighter1_age_dec_avg,
    'f2_age_dec_avg': fighter2_age_dec_avg,
    'f1_sig_str_land_dec_avg': fighter1_sig_str_dec_avg,
    'f2_sig_str_land_dec_avg': fighter2_sig_str_dec_avg,
    # ... all features
}
```

### Step 2: Feature Transformation

**Apply Same Transformations as Training:**

```python
# 1. Create differences
for stat in feature_columns:
    prediction_row[f'{stat}_diff'] = prediction_row[f'f1_{stat}'] - prediction_row[f'f2_{stat}']

# 2. Select same features as training
model_features = [col for col in prediction_row.keys() if col.endswith('_diff')]
X = prediction_row[model_features]
```

### Step 3: Model Prediction

```python
# Load trained model
model = load_model('trained_model.pkl')

# Predict
prob_fighter1_wins = model.predict_proba(X)[0][1]
```

---

## Key Design Decisions

### 1. Time-Decayed Average Half-Life

**Decision:** 1.0 year half-life (changed from 1.5 years)

**Rationale:**
- UFC fighters compete 2-3x/year
- 1.0yr half-life → 12-month fight gets 50% weight
- Performance correlation drops significantly after 12-18 months
- More responsive to recent skill/physical changes

**Impact:**
- Recent fights weighted more heavily
- Better for young fighters (rapid improvement)
- Better for veterans (faster decline detection)

### 2. Static Feature Handling

**Decision:** Do NOT shift `_dec_avg` columns

**Rationale:**
- Time-decayed averages are calculated using `event_date < current_date` (strictly past)
- Current fight is already excluded from calculation
- Double-shifting would create staleness (using T-2 data)

**Exception:** Dynamic stats (`sig_str_land`, `td_land`) without `_dec_avg` suffix ARE shifted

### 3. Inference Static Feature Inclusion

**Decision:** Include current fight in `_dec_avg` calculation for static features

**Rationale:**
- Static features (`age`, `reach`) are known pre-fight
- Including current values (weight = 1.0) maximizes fidelity
- Matches training behavior after fixes

### 4. Smoothing Before Feature Engineering

**Decision:** Apply Poisson-Gamma and Beta-Binomial smoothing EARLY (before avg, dec_avg, adjperf)

**Rationale:**
- Small sample bias compounds through layers
- Example: 1 fight with 50 strikes → 50 avg → overfits
- Smoothing prevents propagation of noise

### 5. Adjusted Performance Design

**Decision:** Use opponent-allowed stats with reliability-weighted shrinkage

**Rationale:**
- Opponent quality varies dramatically
- Shrinkage prevents overfitting on fighters with weak opponents
- Converges to true skill as sample size grows

**Alternative Considered:** Elo ratings  
**Why Not:** Elo assumes transitive property (A > B, B > C → A > C), doesn't hold in MMA due to style matchups

### 6. Feature Table Separation

**Decision:** Split `fight_stats_derived` into 45 feature-specific tables

**Rationale:**
- Cleaner namespacing for layered features
- Easier to add new stats without table bloat
- Parallelizable feature calculations

---

## Appendix: Complete Calculator List

### Basic Derived (10 calculators)
1. `TimeSecCalculator` – Fight duration
2. `KOCalculator` – KO outcomes
3. `DecisionCalculator` – Decision outcomes
4. `SubmissionslandCalculator` – Submission successes
5. `WinCalculator` – Win indicators
6. `AgeCalculator` – Age at fight
7. `DaysSinceLastFightCalculator` – Layoff duration
8. `ReachCalculator` – Reach
9. `HeightCalculator` – Height
10. `ApeCalculator` – Ape index
11. `UfcAgeCalculator` – UFC experience

### Smoothing (2 calculators)
12. `Beta BinomialCalculator` – Binary outcome smoothing
13. `PoissonGammaCalculator` – Count stat smoothing

### Rate Features (6 calculators)
14. `TotalCalculator` – Cumulative sums
15. `AccuracyCalculator` – Accuracy rates
16. `DefenseCalculator` – Defense rates
17. `PerMinCalculator` – Per-minute rates
18. `RatioCalculator` – Per-fight ratios
19. `PressureCalculator` – Round 1 intensity

### Custom Features (2 calculators)
20. `PerCalculator` – Custom per-stat ratios
21. `OpponentCalculator` – Opponent-allowed stats

### Priors (4 calculators)
22. `WeightclassMeanCalculator` – Weightclass averages
23. `WeightclassMadCalculator` – Weightclass variability
24. `FirstTimeMadCalculator` – First-fighter MADs
25. `MinimumMadCalculator` – MAD floors

### Aggregations (3 calculators)
26. `AverageCalculator` – Simple averages
27. `MedianAbsoluteDeviationCalculator` – Variability
28. `TimedecAvgCalculator` – Time-weighted averages

### Adjusted Performance (1 calculator)
29. `AdjustedPerformanceCalculator` – Opponent-adjusted z-scores

### Odds (2 components)
30. `BFOScraper` – Scrape betting odds
31. `OddsCalculator` – Process odds

---

## Questions & Troubleshooting

**Q: Why do some features have `_prev` suffix?**  
A: Shifted features. `f1_prev_sig_str_land` means Fighter 1's `sig_str_land` from their previous fight (T-1).

**Q: What's the difference between `_avg` and `_dec_avg`?**  
A: `_avg` = simple mean, `_dec_avg` = time-weighted mean (1.0yr half-life).

**Q: Why are there both `_adjperf` and `_dec_adjperf`?**  
A: `_adjperf` uses all opponent history equally, `_dec_adjperf` uses 1.0yr time-decay on opponent history.

**Q: How do I add a new feature?**  
A:
1. Create calculator in `libs/feature_store/calculators/`
2. Inherit from `BaseCalculator`
3. Implement `calculate_for_table()` or `run()`
4. Add to pipeline in `main.py`
5. Rebuild database

**Q: How do I change the time decay half-life?**  
A: Update `decay_rate = math.log(2) / X` in:
- `time_dec_avg_calc.py` (documentation)
- `adj_perf_calc.py` (SQL constant)
- `create_inference_data.py`
- `stat_updater.py`

Then rebuild features.

---

**Document Version:** 1.0  
**Last Updated:** 2025-11-22  
**Pipeline Version:** Post half-life optimization (1.0yr)
