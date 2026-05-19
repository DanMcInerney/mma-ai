# Release Readiness Notes

Status: clean-history release candidate.

This snapshot is intended for the new public repository. It was exported without
the old `.git` directory, scanned, initialized as a fresh repository, and pushed
as a single root commit.

The core workflow now has repo-local commands for the important path:

```bash
uv sync
uv run python -m scripts.scrape_ufcstats
uv run python main.py --reset-db
uv run python -m libs.modeling.train --model-type win
uv run python predict.py --model-type win --no-shap
```

What changed:

- UFCStats scraping is integrated into this repo in `libs/scraping/ufcstats.py` and `scripts/scrape_ufcstats.py`.
- The rebuild/train/predict path now uses repo-relative defaults and environment-variable overrides instead of sibling checkout paths.
- Generated analysis, screenshots, notebooks, ad hoc query notes, and runtime data outputs were removed from source control. Large reusable data/model artifacts live in the Hugging Face dataset instead.
- Hardcoded third-party API key strings were removed from current code. Use `.env` values for `THE_ODDS_API_KEY`, `GOOGLE_API_KEY`, or `GEMINI_API_KEY`.
- `.claude/settings.local.json` was removed from the tracked tree.
- Personal Windows paths were scrubbed from docs, logs, old query notes, and script defaults.

Remaining caveats:

- Rotate any API keys that were ever committed to older repositories or remotes. This clean snapshot does not include the old history, but rotation is still the right safety move.
- Runtime outputs are intentionally ignored. Recreate them locally by scraping/restoring data, retraining, or downloading the prepared artifacts from Hugging Face.
- Some legacy scripts still default to local PostgreSQL URLs. Core commands now use `DATABASE_URL`, but those scripts should be updated before treating them as public entry points.

Suggested scans before publishing:

```bash
rg -n --hidden --glob '!**/.git/**' --glob '!**/.venv/**' --glob '!uv.lock' "C:/Users|C:\\Users|@gmail\.com|api[_-]?key|secret|token" .
git log --all -S "genai.configure(api_key=" -- libs/llm_odds.py
git log --all -S "self.api_key =" -- libs/odds.py libs/feature_store/calculators/odds_calc_oddsapi_backup.py
git ls-files | rg '\.(csv|png|jpg|jpeg|gif|html|ipynb)$|(^pics/|^data/|^visualizations/|^blogs/|^queries/)'
```
