# Analytics Schema Reference

Use this document when writing analytics against MMA AI data. It is the
plain-English companion to the feature-store code and the live database schema.
When there is any mismatch, inspect the current database with
`information_schema` before making a claim.

## Core Grain

Most `features` tables use one central grain:

```text
one row = one fighter in one completed fight
primary key = fight_id, fighter_id, event_id
```

`features.fight_mapping` is one row per fight and stores both fighter IDs.
`features.event_mapping` gives the fight date. `features.fighter_mapping` gives
names and profile fields. For fighter-vs-fighter comparisons, join a feature
table twice: once for `fighter1_id` and once for `fighter2_id`.

## Source Priority

Prefer analytics sources in this order:

1. Finalized `model_data` tables or CSVs for model-facing questions.
2. Feature-family tables such as `features.sig_str`, `features.td`,
   `features.ctrl`, `features.odds`, or `features.style`.
3. `features.fight_stats_derived` for row-level engineered features.
4. `features.fight_stats_fe` only for raw scrape or calculator-quality checks.

For predictive analytics, avoid leakage. Feature-family tables are post-fight
artifacts for completed fights: `_avg`, `_dec_avg`, `_total`, and `_mad`
include the current completed row. Use `training_data.csv`, inference-builder
outputs, `_prev` columns, or an explicit past-only query when answering a
pre-fight question.

## Table Families

| Table family | Plain-English meaning | Analytics notes |
| --- | --- | --- |
| `age` | Fighter age at the event date | Known before the fight. |
| `days_since_last_fight` | Layoff/rest before the fight | Known before the fight. |
| `reach`, `height` | Reach and height in inches | Missing values may be imputed by weight class. |
| `ape` | Reach divided by height | Ratio, not reach-minus-height. |
| `ufcage` | Years since first UFCStats fight | Known before the fight. |
| `sig_str` | Significant striking volume, accuracy, defense, rates, shares | Smoothed values after derived staging. |
| `strikes` | Total strike volume, accuracy, defense, rates, shares | Broader than significant strikes. |
| `head`, `body`, `leg` | Significant strikes by target | Use for targeting and defensive accuracy questions. |
| `distance`, `clinch`, `ground` | Significant strikes by position/range | Use for range mix and range-specific efficiency. |
| `td` | Takedowns landed/attempted, accuracy, defense, pace | `td_def` is opponent takedown accuracy against the fighter. |
| `ctrl` | Control time in seconds and rates/shares | `ctrl_per_min` is control seconds per fight minute. |
| `sub` | Submission attempts and submission-win indicators | `sub_land` means completed submission win when present. |
| `rev` | Reversals | Often sparse; report sample size. |
| `kd` | Knockdowns | Often sparse; prefer rates or adjusted summaries for broad claims. |
| `ko`, `decision`, `win` | Outcome indicators and finish-derived features | Post-fight outcomes; do not use as pre-fight evidence unless shifted. |
| `time_sec` | Fight duration | Post-fight outcome-like duration. |
| `odds` | Historical BestFightOdds-derived market prices and probabilities | See odds section below; missing odds means missing market data. |
| `style` | Composite style metrics from lower-level features | Good for exploratory style summaries; inspect definitions before causal claims. |

Discover the tables and columns that exist in a restored database:

```sql
SELECT table_name
FROM information_schema.tables
WHERE table_schema = 'features'
  AND table_type = 'BASE TABLE'
ORDER BY table_name;
```

```sql
SELECT column_name, data_type
FROM information_schema.columns
WHERE table_schema = 'features'
  AND table_name = 'sig_str'
ORDER BY ordinal_position;
```

## Suffix Semantics

| Suffix | Plain-English meaning |
| --- | --- |
| `_rd1` | First-round value. |
| `_smooth` | Temporary Bayesian-smoothed column before it is renamed to the base name. |
| `_raw` | Temporary observed value kept briefly for accuracy calculations. |
| `_total` | Fighter cumulative total through the completed row. |
| `_acc` | Landed divided by attempted, with Bayesian smoothing. |
| `_def` | Opponent accuracy against this fighter in the same fight. Lower is better defense. |
| `_per_min` | Rate per fight minute, using `time_sec / 60`. |
| `_ratio` | Fighter share of fighter-plus-opponent value in the same fight. |
| `_pressure` | First-round share of total output for that stat. |
| `_per_` | Domain-specific conversion ratio, such as `td_land_per_ctrl`. |
| `_opp` | The opponent's realized same-fight value copied onto this fighter row. |
| `_wc_mean` | Weight-class prior mean in support tables. |
| `_wc_mad` | Weight-class prior median absolute deviation in support tables. |
| `_minimum_mad` | MAD floor used to keep adjusted performance stable. |
| `_avg` | Fighter rolling average through the completed row. |
| `_dec_avg` | Fighter time-decayed rolling average through the completed row. |
| `_mad` | Fighter rolling median absolute deviation. |
| `_adjperf` | Opponent-adjusted performance score. |
| `_dec_adjperf` | Opponent-adjusted performance score using time-decayed opponent history. |
| `_prev` | Previous-fight shifted feature in cleaned training data. |
| `_diff` | Fighter1 value minus fighter2 value in finalized fight-row data. |

## Adjusted Performance

`_adjperf` asks:

```text
How much better or worse was this fighter's observed stat than what this
opponent historically allows, after shrinking sparse opponent history toward
weight-class priors?
```

For example, `sig_str_land_per_min_adjperf` compares the fighter's significant
strikes landed per minute in this fight to what previous opponents usually
landed per minute against today's opponent. Sparse opponent histories are
shrunk toward weight-class baselines.

The score is z-score-like and clipped. Positive values mean the fighter exceeded
expectation against that opponent. Negative values mean the fighter
underperformed relative to expectation. `_dec_adjperf` uses the same idea but
weights the opponent's historical allowance by recency.

Read long names left to right:

```text
distance_acc_dec_adjperf_dec_avg
```

means distance-striking accuracy, compared against a time-decayed
opponent-allowed expectation, converted into adjusted performance, then averaged
over the fighter's own history with time decay.

## Odds Databases

There are two odds-related storage layers.

`ODDS_DATABASE_URL` points to the auxiliary odds database. Its important table is
`bestfightodds.bfo`:

| Column | Meaning |
| --- | --- |
| `id` | Surrogate row ID. |
| `fighter` | BestFightOdds fighter name. |
| `fighter_url` | BestFightOdds fighter URL when available. |
| `opponent` | BestFightOdds opponent name. |
| `timestamp` | Timestamp for the price point. |
| `odds` | Decimal odds from BestFightOdds, not American odds. |

Rows are unique by `(fighter, opponent, timestamp)`. This table is the imported
or scraped market-data source. It is not joined directly for most analytics.

`features.odds` lives in the main MMA database and is generated from
`bestfightodds.bfo`. It uses the normal fighter-fight grain:

| Column | Meaning |
| --- | --- |
| `opening_odds` | First decimal odds point found for the matched fighter/opponent group. |
| `closing_odds` | Last decimal odds point found for the matched group. |
| `sevenday_opening_odds` | Decimal odds closest to seven days before close, falling back when needed. |
| `ip_opening_odds` | Implied probability from `opening_odds`, computed as `1 / decimal_odds`. |
| `ip_closing_odds` | Implied probability from `closing_odds`. |
| `sevenday_ip_opening_odds` | Implied probability from `sevenday_opening_odds`. |
| `vigless_ip_opening_odds` | Opening implied probability normalized across both fighters. |
| `vigless_ip_closing_odds` | Closing implied probability normalized across both fighters. |
| `sevenday_vigless_ip_opening_odds` | Seven-day implied probability normalized across both fighters. |

Odds matching is name-based and date-tolerant. The calculator groups BFO rows by
fighter, opponent, and event month, then matches to UFCStats fights by lower-case
fighter names near the event date. Rematches, missing opponent rows, and name
aliases are the main failure modes. Always report odds coverage and null rates.

Prediction-time live/manual odds use American odds in the UI/CLI, then convert
to implied probabilities or vig-free American odds for reporting. Do not confuse
those live American odds with historical `features.odds` decimal columns.

Dashboard live odds are used for market probability, AI odds, expected value,
and pick-edge reporting. Historical odds columns can still appear in
`prediction_data.csv`, `training_data.csv`, profit reports, and model feature
lists depending on the artifact. Inspect a model's `feats.txt` before claiming
whether odds were or were not model inputs.

## Model Outputs

`data/prediction_data.csv` is a wide fighter-row feature matrix used by the
inference builder. Despite the name, it is not only predictions.

`data/training_data.csv` and `data/training_data_dec.csv` are finalized
fight-row model data:

1. Dynamic fighter stats are shifted to the previous fight and receive `_prev`.
2. Fighter1 and fighter2 rows are joined into one fight row.
3. `_diff` columns are fighter1 minus fighter2.
4. Static/pre-fight features such as `age`, `days_since_last_fight`, `reach`,
   `height`, `ufcage`, historical odds fields, and `weightclass_encoded` are not
   shifted.

For model explanations, use the exact columns in the model artifact's
`feats.txt`. For descriptive fight-history analytics, feature-family tables are
usually easier to read.

## Query Patterns

Join a feature table to names and dates:

```sql
SELECT
  em.event_date,
  fmap.fighter_name,
  s.sig_str_land_per_min_dec_avg
FROM features.sig_str s
JOIN features.event_mapping em ON em.event_id = s.event_id
JOIN features.fighter_mapping fmap ON fmap.fighter_id = s.fighter_id
ORDER BY em.event_date DESC
LIMIT 50;
```

Compare both fighters in a fight:

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
 AND s2.fighter_id = fm.fighter2_id;
```

Check odds coverage:

```sql
SELECT
  COUNT(*) AS fighter_rows,
  COUNT(o.closing_odds) AS rows_with_closing_odds,
  AVG(o.vigless_ip_closing_odds) AS avg_vigless_close_ip
FROM features.fight_mapping fm
JOIN features.event_mapping em ON em.event_id = fm.event_id
LEFT JOIN features.odds o ON o.fight_id = fm.fight_id;
```

When a question depends on time ordering, add a CTE that restricts each
fighter's history to rows before the fight date being analyzed. Do not rely on
feature-family `_avg` or `_dec_avg` columns as pre-fight values.
