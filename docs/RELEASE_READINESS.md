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
uv run mma-web --help
```

Docker smoke path:

```bash
powershell -ExecutionPolicy Bypass -File .\setup.ps1
./setup.sh
docker compose config --quiet
docker compose up --build
```

For a bootstrapped local app start, use `docker compose up --build db web` so
the API and the bundled PostgreSQL service come up together. The web service is
gated on the PostgreSQL healthcheck in Compose. Use
`docker compose up --no-deps --build web` only when
`MMA_AI_COMPOSE_DATABASE_URL` and `MMA_AI_COMPOSE_ODDS_DATABASE_URL` point to an
external PostgreSQL instance.

After the web service starts, verify `http://localhost:8000/api/health` returns
`{"status":"ok"}` for liveness and `http://localhost:8000/api/readiness`
returns a ready payload before relying on predictions. Readiness requires both
databases to contain their imported tables, all processed CSVs, and the Hugging
Face starter model. CSV readiness checks validate the minimal headers needed by
scraping, training, and prediction, and report `missing_columns` for malformed
artifacts. Then open the
dashboard at `http://localhost:8000`, or the alternate port printed by setup.
The dashboard top bar mirrors this state with a `Ready` or `Setup incomplete`
badge.

For deployed-stack verification, run the readiness smoke check against the
actual dashboard URL:

```bash
uv run mma-docker-smoke --deployed-url http://localhost:8000
```

Use the alternate web port printed by setup when port 8000 is unavailable. This
check fails if `/api/readiness` is incomplete or if `/api/predict/models` cannot
discover a compatible `win` model, so it catches the same conditions that block
the Predict tab.

## Release Surface

- Data tab: refresh UFCStats CSVs, rebuild PostgreSQL feature tables, write
  `prediction_data.csv`, `training_data.csv`, and `training_data_dec.csv`, run
  read-only analytics. The default data run incrementally merges new UFCStats
  rows into the shipped seed CSVs, recreates generated schemas from those CSVs,
  and recalculates odds features from the imported Hugging Face `odds` database
  without refreshing BestFightOdds. Completed data jobs return before/after row
  deltas so users can verify how many raw and finalized rows changed during the
  refresh. Analytics SQL is
  constrained to one read-only query, runs inside a read-only Postgres
  transaction with a statement timeout, and uses SQLite query-only mode for
  finalized CSV fallbacks. The Analytics panel exposes a non-secret LLM status
  line so users know whether Ask is in LLM-backed or SQL-only mode.
- Predict tab: list models, automatically load upcoming UFC events from
  Wikipedia into an event-name dropdown, predict a selected event, run manual
  fighter matchups, and show book odds, AI odds, positive EV status, and pick
  edge in compact result cards. Prediction-time live/manual odds are enabled by
  default but are used for reporting rather than passed into the predictor;
  event prediction can ingest manual American odds through the dashboard/API
  instead of waiting on terminal input. Manual matchup prediction defaults the
  fight date to today and does not expose per-fighter odds controls.
  The advanced Flaresolverr proxy toggle is only for BestFightOdds blocking
  normal odds scraping.
- Job logs: Data and Predict jobs stream and persist stdout, stderr,
  command lines, exit codes, and tracebacks under `data/logs/jobs` and expose them at
  `/api/jobs/{job_id}/log`. Dashboard jobs are serialized so long-running data,
  and prediction workflows do not interleave process-wide stdout/stderr
  captures or write shared artifacts at the same time.
- Bootstrap scripts: `setup.ps1` and `setup.sh` download the Hugging Face
  dataset artifacts, verify checksums, resume complete database imports by
  checking `features.fight_mapping` and `bestfightodds.bfo`, restore both
  Postgres dumps into Docker when needed, copy processed CSVs, extract the
  starter model, optionally configure
  `LLM_PROVIDER`, `LLM_MODEL`, `LLM_API_KEY`, and provider aliases for
  analytics, including hosted OpenAI-compatible providers such as OpenRouter,
  DeepSeek, Mistral, Together AI, and Perplexity Sonar, auto-select
  `MMA_AI_POSTGRES_PORT` when host port 5432 is occupied,
  write matching host-side `DATABASE_URL` and `ODDS_DATABASE_URL` values for
  local CLI commands, start the web service, wait for `/api/readiness`, and open
  the dashboard.
- Docker stack: `postgres:18.1` plus the FastAPI web service. Compose initializes
  the auxiliary `odds` database with `docker/postgres-init/01-create-odds.sql`.
  `MMA_AI_COMPOSE_DATABASE_URL` and `MMA_AI_COMPOSE_ODDS_DATABASE_URL` let the
  Docker web service use an existing host PostgreSQL instance instead; in that
  mode, start only `docker compose up --no-deps --build web` so the bundled
  database service does not claim port 5432. The web service also maps
  `host.docker.internal:host-gateway` so Linux users can reach host-side
  Postgres and local OpenAI-compatible LLM servers through the same URLs
  documented for Docker Desktop.
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
- `OPENROUTER_API_KEY`
- `DEEPSEEK_API_KEY`
- `MISTRAL_API_KEY`
- `TOGETHER_API_KEY`
- `PERPLEXITY_API_KEY`
- `GEMINI_API_KEY`
- `GOOGLE_API_KEY`

Do not commit real credentials, local personal paths, trained model artifacts,
raw data dumps, screenshots, notebooks, or generated prediction outputs.
Local Python commands load the root `.env` file without overriding already
exported shell variables. Docker builds ignore `.env`, so credentials used for a
local setup do not enter the image build context.

## Verification Checklist

```bash
uv lock --check
uv run pytest -q
uv run mma-release-audit
docker compose config --quiet
docker compose build web
uv run mma-docker-smoke
uv run mma-docker-smoke --deployed-url http://localhost:8000
```

When Chrome is available, run the real-browser Predict tab e2e before release:

```bash
MMA_AI_RUN_BROWSER_E2E=1 uv run pytest tests/e2e/test_predict_tab_next_event.py::test_predict_tab_browser_predicts_next_ufc_event -q
```

The default Docker smoke command starts the built web image, checks
`/api/health`, fetches the dashboard HTML, `/vendor/plotly.min.js`, and
`/static/icons.js` from inside the container, then verifies the runtime image
does not include test tooling. The deployed URL mode checks the running Compose
stack's health, readiness, and `win` model discovery.

Security and hygiene scans:

```bash
uv run mma-release-audit --json
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
