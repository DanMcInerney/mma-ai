# MMA AI Database - CLAUDE.md

**Keep this file under 150 lines.**

## Project Overview

This repo contains MMA fight prediction models and outputs. The main workflow is generating predictions, then publishing them to the website.

## Key Paths

- **Picks folder:** `pics/picks/` — contains event subfolders with `fight_predictions.csv`
- **Website repo:** configure per machine, commonly a sibling checkout such as `../mmaai-flask/`
- **Events JSON:** configure per machine, commonly `../mmaai-flask/data/eventsv6.json`

## Updating the Website with New Picks

### Step 1: Read the predictions CSV

The user will tell you which folder in `pics/picks/` to read from. Open `fight_predictions.csv` in that folder.

CSV columns: `Fighter1, Fighter2, Fighter1_Odds, Fighter2_Odds, Fighter1_AI_Prob, Fighter2_AI_Prob, Fighter1_Market_Prob, Fighter2_Market_Prob, AI_Pick, Confidence, AI_Odds, EV`

### Step 2: Determine result status for each fight

For each fight, set the `result` field based on these rules:

- **"Pending"** — ONLY if BOTH conditions are true:
  1. `EV` = 1 (positive expected value)
  2. The AI pick is the **Vegas odds favorite** (the picked fighter's odds are negative)
  - Limit to top 1-2 fights (user specifies how many). Rank by confidence.
- **"Pending - no bet"** — all other fights

### Step 3: Build the new event entry

Add a new entry to the `events` array in `eventsv6.json`:

```json
{
  "name": "UFC Fight Night",
  "date": "YYYY-M-DD",
  "predictions": [
    {
      "fight": "fighter1 vs fighter2",
      "prediction": "ai_pick_name",
      "ai_win_pct": 73.6,
      "ai_odds": -278,
      "vegas_odds": -116.0,
      "result": "Pending"
    }
  ]
}
```

Field mapping from CSV:
- `fight`: lowercase `"Fighter1 vs Fighter2"`
- `prediction`: lowercase `AI_Pick`
- `ai_win_pct`: `Confidence` value
- `ai_odds`: `AI_Odds` value (integer)
- `vegas_odds`: The picked fighter's odds as a float (Fighter1_Odds if AI picked Fighter1, else Fighter2_Odds)
- `result`: See Step 2

Sort predictions by confidence (highest first). Append the new event to the end of the `events` array.

### Step 4: Resolve the previous event's results

Find the most recent prior event in `eventsv6.json` that still has "Pending" or "Pending - no bet" fights. For each unresolved fight:

1. **Search the internet** for the fight result (who won)
2. Compare the actual winner to the `prediction` field
3. Update `result`:

| Old Status | AI Correct? | New Status |
|---|---|---|
| Pending | Yes | Hit! |
| Pending | No | Miss |
| Pending - no bet | Yes | Hit - no bet |
| Pending - no bet | No | Miss - no bet |

Note: If a fight was a draw or no-contest, ask the user how to handle it.

### Step 5: Save and verify

Save `eventsv6.json` with the updates. Verify the JSON is valid.

## Important Notes

- Fighter names should be **lowercase** in the JSON
- Date format in JSON is `YYYY-M-DD` (no leading zeros on month/day, e.g., `2025-3-15`)
- Vegas odds: negative = favorite, positive = underdog
- The `events` array is ordered chronologically; always append new events at the end
- When searching for fight results, confirm the correct event/date to avoid confusion with rematches
