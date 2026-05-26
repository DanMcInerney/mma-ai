# Release Readiness Notes

Status: dashboard release candidate.

This repository is intended to be the public, Dockerized home for the combined
UFCStats scraper, feature-store rebuild, model training, analytics, and
prediction workflows.

## Primary Smoke Path

```bash
uv sync
uv run mma-scrape-ufcstats --help
uv run mma-rebuild-db --help
uv run mma-train --help
uv run mma-evaluate --help
uv run mma-predict --help
uv run mma-web
```

Docker smoke path:

```bash
powershell -ExecutionPolicy Bypass -File .\setup.ps1
bash setup.sh
docker compose config --quiet
docker compose up --build
```

After the web service starts, verify `http://localhost:8000/api/health` returns
`{"status":"ok"}` and open the dashboard at `http://localhost:8000`, or the
alternate port printed by setup.

## Release Surface

- Data tab: refresh UFCStats CSVs, rebuild PostgreSQL feature tables, write
  `prediction_data.csv`, `training_data.csv`, and `training_data_dec.csv`, run
  read-only analytics. The default data run incrementally merges new UFCStats
  rows into the shipped seed CSVs and recreates generated schemas from those
  CSVs.
- Train tab: run `libs/modeling/train.py` defaults through a compact UI, keep
  advanced knobs collapsed, and summarize saved model evaluation artifacts with
  metrics and charts. `mma-evaluate` emits the same evaluation summary as JSON
  for automation.
- Predict tab: list models, automatically load upcoming UFC events from
  Wikipedia into an event-name dropdown, predict a selected event, run manual
  fighter matchups, and show AI probability, market probability, AI odds, and
  positive EV status. Odds are not model inputs; event prediction can ingest
  manual American odds through the dashboard/API instead of waiting on terminal
  input.
- Job logs: Data, Train, and Predict jobs persist stdout, stderr, command lines,
  and tracebacks under `data/logs/jobs` and expose them at
  `/api/jobs/{job_id}/log`.
- Bootstrap scripts: `setup.ps1` and `setup.sh` download the Hugging Face
  dataset artifacts, verify checksums, restore both Postgres dumps into Docker,
  copy processed CSVs, extract the starter model, optionally configure
  `LLM_PROVIDER`, `LLM_MODEL`, `LLM_API_KEY`, and provider aliases for analytics
  chat, auto-select `MMA_AI_POSTGRES_PORT` when host port 5432 is occupied,
  start the web service, and open the dashboard.
- Docker stack: `postgres:17` plus the FastAPI web service. Compose initializes
  the auxiliary `odds` database with `docker/postgres-init/01-create-odds.sql`.
  `MMA_AI_COMPOSE_DATABASE_URL` and `MMA_AI_COMPOSE_ODDS_DATABASE_URL` let the
  Docker web service use an existing host PostgreSQL instance instead; in that
  mode, start `docker compose up --build web` so the bundled database service
  does not claim port 5432.
- Local static assets: Plotly is served from `/vendor/plotly.min.js`; dashboard
  icons are served from `/static/icons.js`.

## Configuration

Public configuration lives in `.env.example`.

- `DATABASE_URL`
- `ODDS_DATABASE_URL`
- `MMA_AI_COMPOSE_DATABASE_URL`
- `MMA_AI_COMPOSE_ODDS_DATABASE_URL`
- `MMA_AI_DATA_DIR`
- `MMA_AI_UFCSTATS_DIR`
- `MMA_AI_MODELS_DIR`
- `MMA_AI_PICKS_DIR`
- `THE_ODDS_API_KEY`
- `LLM_PROVIDER`
- `LLM_MODEL`
- `LLM_API_KEY`
- `LLM_BASE_URL`
- `OPENAI_API_KEY`
- `ANTHROPIC_API_KEY`
- `XAI_API_KEY`
- `GEMINI_API_KEY`
- `GOOGLE_API_KEY`

Do not commit real credentials, local personal paths, trained model artifacts,
raw data dumps, screenshots, notebooks, or generated prediction outputs.

## Verification Checklist

```bash
uv lock --check
uv run pytest -q
docker compose config --quiet
docker compose build web
```

Security and hygiene scans:

```bash
rg -n --hidden --glob '!**/.git/**' --glob '!**/.venv/**' --glob '!uv.lock' --glob '!docs/RELEASE_READINESS.md' "C:/Users|C:\\Users|@gmail\\.com|api[_-]?key|secret|token" .
git ls-files | rg '(^pics/|^data/|^visualizations/|^blogs/|^queries/|\\.(csv|png|jpg|jpeg|gif|ipynb)$)'
```

## Remaining Caveats

- Rotate any API keys that were ever committed to older repositories or remotes.
  This repo should not contain real keys, but rotation is still prudent for
  previously exposed credentials.
- Runtime outputs are intentionally ignored except for
  `data/raw/ufcstats/competitions.csv` and `data/raw/ufcstats/individuals.csv`,
  which are tracked seed data. Recreate generated outputs by scraping/restoring
  data, retraining, or downloading prepared artifacts from the companion dataset.
- Legacy scripts that are not part of the dashboard path may still assume local
  PostgreSQL defaults. Treat `mma-scrape-ufcstats`, `mma-rebuild-db`,
  `mma-train`, `mma-evaluate`, `mma-predict`, and `mma-web` as the public entry
  points.
