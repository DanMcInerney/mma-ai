# Hugging Face Dataset Artifacts

Dataset URL: `https://huggingface.co/datasets/DanMcInerney/mma-ai`

This repo tracks only the small raw UFCStats seed CSVs needed to update from
source: `data/raw/ufcstats/competitions.csv` and
`data/raw/ufcstats/individuals.csv`. Large database dumps, finalized model CSVs,
and trained model artifacts live in the Hugging Face Dataset repository.

For first-time app setup, prefer the code repo bootstrap scripts:

```bash
./setup.sh
```

or on Windows:

```powershell
powershell -ExecutionPolicy Bypass -File .\setup.ps1
```

The scripts download these artifacts, verify checksums, restore the dumps into
the Docker Postgres service, copy processed CSVs into `data/`, extract the
starter model into `AutogluonModels/`, optionally configure LLM analytics, and
start the web dashboard. If host port `5432` is already occupied, setup writes a
free `MMA_AI_POSTGRES_PORT` value plus matching local `DATABASE_URL` and
`ODDS_DATABASE_URL` values to `.env`; pass `-PostgresPort 55432` or
`--postgres-port 55432` to force a port. Local `uv run ...` commands load this
`.env` file automatically without overriding values already exported in your
shell.
Setup starts the bundled local stack and waits for the dashboard readiness
check. To restart the already bootstrapped stack later, run
`docker compose up --build db web`.
Rerunning setup reuses complete imported databases when `features.fight_mapping`
and `bestfightodds.bfo` are present; pass `-ForceImport` or `--force-import` to
restore the dumps again.

LLM analytics setup supports OpenAI, Codex/OpenAI-compatible, Anthropic Claude,
Google Gemini, xAI Grok, OpenRouter, DeepSeek, Mistral, Together AI,
Perplexity Sonar, local OpenAI-compatible servers, and custom endpoints.
The scripts save `LLM_PROVIDER`, `LLM_MODEL`, `LLM_API_KEY`, and optional
`LLM_BASE_URL` to `.env`.

## Required Files

| File | Purpose |
| --- | --- |
| `dumps/mma-ai.postgres-custom` | Main `mma-ai` PostgreSQL database dump. |
| `dumps/odds.postgres-custom` | Separate `odds` PostgreSQL database dump. |

## Convenience Files

| File | Purpose |
| --- | --- |
| `processed/training_data.csv` | Generated win-model training CSV. |
| `processed/training_data_dec.csv` | Generated decision-model training CSV. |
| `processed/prediction_data.csv` | Generated prediction feature CSV. |
| `models/ag-20260304_110750-win-extreme.tar.gz` | Pretrained AutoGluon win model. |

The dumps use PostgreSQL custom archive format with gzip compression. Restore
them with the repo's PostgreSQL 18.1 Docker service or a compatible local
PostgreSQL version.

## Restore

Download the dataset artifacts:

```bash
git lfs install
mkdir -p artifacts
git clone https://huggingface.co/datasets/DanMcInerney/mma-ai artifacts/mma-ai-dataset
```

PowerShell:

```powershell
git lfs install
New-Item -ItemType Directory -Force artifacts | Out-Null
git clone https://huggingface.co/datasets/DanMcInerney/mma-ai artifacts/mma-ai-dataset
```

From the code repo root, restore the databases:

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

Use your own username, password, host, and port in the connection strings if
your local Postgres setup differs.

After the first import, you do not need to restore the dumps again for routine
UFCStats updates. Run the incremental scraper, then recreate the generated
schemas and finalized CSVs from the merged raw CSVs:

```bash
uv run mma-rebuild-db --scrape --reset-db --odds-features
```

The scraper skips fighter URLs and event URLs already present in the checked-in
raw CSVs and appends newly discovered fighters/fights.

## Pretrained Model

```bash
mkdir -p AutogluonModels
tar -xzf artifacts/mma-ai-dataset/models/ag-20260304_110750-win-extreme.tar.gz -C AutogluonModels
mkdir -p data
cp artifacts/mma-ai-dataset/processed/training_data.csv data/training_data.csv
cp artifacts/mma-ai-dataset/processed/training_data_dec.csv data/training_data_dec.csv
cp artifacts/mma-ai-dataset/processed/prediction_data.csv data/prediction_data.csv
```

Then run:

```bash
uv run python predict.py \
  --model-path AutogluonModels/ag-20260304_110750-win-extreme \
  --prediction-data-csv data/prediction_data.csv \
  --training-data-csv data/training_data.csv \
  --no-shap
```
