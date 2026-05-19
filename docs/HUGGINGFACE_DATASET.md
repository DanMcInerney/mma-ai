# Hugging Face Dataset Artifacts

Dataset URL: `https://huggingface.co/datasets/DanMcInerney/mma-ai`

This repo is code-only. Large database, CSV, and model artifacts live in the
Hugging Face Dataset repository.

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
  --dbname "postgresql://postgres@localhost:5432/mma-ai" \
  artifacts/mma-ai-dataset/dumps/mma-ai.postgres-custom

pg_restore --clean --if-exists --no-owner --jobs 4 \
  --dbname "postgresql://postgres@localhost:5432/odds" \
  artifacts/mma-ai-dataset/dumps/odds.postgres-custom
```

Use your own username, password, host, and port in the connection strings if
your local Postgres setup differs.

## Pretrained Model

```bash
mkdir -p AutogluonModels
tar -xzf artifacts/mma-ai-dataset/models/ag-20260304_110750-win-extreme.tar.gz -C AutogluonModels
mkdir -p data
cp artifacts/mma-ai-dataset/processed/training_data.csv data/training_data.csv
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
