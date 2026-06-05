import csv
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def read_text(path: str) -> str:
    return (ROOT / path).read_text(encoding="utf-8")


def git_ls_files() -> list[str]:
    result = subprocess.run(["git", "ls-files"], cwd=ROOT, check=True, capture_output=True, text=True)
    return [line.strip().replace("\\", "/") for line in result.stdout.splitlines() if line.strip()]


def test_public_release_docs_cover_runtime_and_dashboard_surface():
    readme = read_text("README.md")
    agents = read_text("AGENTS.md")
    claude = read_text("CLAUDE.md")
    release_notes = read_text("docs/RELEASE_READINESS.md")
    huggingface_docs = read_text("docs/HUGGINGFACE_DATASET.md")
    compose = read_text("docker-compose.yml")
    dockerignore = read_text(".dockerignore")
    gitignore = read_text(".gitignore")
    postgres_init = read_text("docker/postgres-init/01-create-odds.sql")

    assert "docker compose up --build" in readme
    assert "setup.ps1" in readme
    assert "setup.sh" in readme
    assert "./setup.sh" in readme
    assert "ag-20260304_110750-win-extreme" in readme
    assert "auxiliary `odds` database" in readme
    assert "Python 3.10-3.12" in readme
    assert "Data: update the shipped raw UFCStats CSVs" in readme
    assert "recalculate odds features from the imported" in readme
    assert "SQL-only analytics mode" in readme
    assert "Training remains a CLI workflow" in readme or "training new models is a\nCLI workflow" in readme
    assert "Predict: choose a model" in readme
    assert "uv run mma-evaluate" in readme
    assert "uv run mma-docker-smoke" in readme
    assert "MMA_AI_RUN_BROWSER_E2E=1" in readme
    assert "test_predict_tab_browser_predicts_next_ufc_event" in readme
    assert "uv run mma-release-audit" in readme
    assert "dashboard HTML plus local Plotly/icon assets are served" in readme
    assert "docker compose up --build db web" in readme
    assert "PostgreSQL 18.1" in readme
    assert "uv run mma-rebuild-db --scrape --reset-db --odds-features" in readme
    assert "MMA_AI_COMPOSE_DATABASE_URL" in readme
    assert "MMA_AI_POSTGRES_PORT" in readme
    assert "--postgres-port 55432" in readme
    assert "updates the local `DATABASE_URL` and" in readme
    assert "Local Python commands automatically load the repo `.env` file" in readme
    assert "Setup is safe to rerun after an interrupted install" in readme
    assert "/api/readiness" in readme
    assert "top bar shows a `Ready` badge" in readme
    assert "imported database tables" in readme
    assert "[Manual Development Setup](#manual-development-setup)" in readme
    assert "For installation and first-time use, prefer" in readme
    assert "Most users should use the repository bootstrap scripts" in readme
    assert "Manual Installation Without Bootstrap Scripts" in readme
    assert "## Installation & Setup" not in readme
    assert "### Standard Installation" not in readme
    assert "--skip-download" in readme
    assert "--force-download" in readme
    assert "-ForceImport" in readme
    assert "--force-import" in readme
    assert "docker compose logs --tail 120 web db" in readme
    assert "Dashboard jobs run one at a time" in readme
    assert "AGENTS.md" in readme
    assert "CLAUDE.md" in readme
    dockerfile = read_text("Dockerfile")
    assert "FROM python:3.12-slim AS runtime" in dockerfile
    assert "/api/health" in dockerfile
    assert 'CMD ["/app/.venv/bin/uvicorn"' in dockerfile
    assert 'CMD ["uv", "run"' not in dockerfile

    assert "Data tab" in agents
    assert "./setup.sh" in agents
    assert "Train tab" not in agents
    assert "CLI Training Defaults" in agents
    assert "Predict tab" in agents
    assert "01-create-odds.sql" in agents
    assert "/vendor/plotly.min.js" in agents
    assert "static/icons.js" in agents
    assert "features.fight_stats_fe" in agents
    assert "database-enforced read-only" in agents
    assert "query-only mode" in agents
    assert "MMA_AI_DATA_DIR" in agents
    assert "output directory" in agents
    assert "YYYY-MM-DD" in agents
    assert "--odds-features" in agents
    assert "BestFightOdds" in agents
    assert "prediction_data.csv" in claude
    assert "./setup.sh" in claude
    assert "training_data.csv" in claude
    assert "MMA_AI_DATA_DIR" in claude
    assert "01-create-odds.sql" in claude
    assert "/vendor/plotly.min.js" in claude
    assert "/static/icons.js" in claude
    assert "output directory" in claude
    assert "YYYY-MM-DD" in claude
    assert "read-only transaction" in claude
    assert "SQLite query-only mode" in claude
    assert "--odds-features" in claude
    assert "without scraping BestFightOdds" in claude

    assert "dashboard release candidate" in release_notes
    assert "setup.ps1" in release_notes
    assert "setup.sh" in release_notes
    assert "./setup.sh" in release_notes
    assert "uv run mma-web --help" in release_notes
    assert "docker compose up --build" in release_notes
    assert "Data tab" in release_notes
    assert "recalculates odds features from the imported" in release_notes
    assert "read-only Postgres" in release_notes
    assert "transaction with a statement timeout" in release_notes
    assert "SQLite query-only mode" in release_notes
    assert "SQL-only mode" in release_notes
    assert "Train tab" not in release_notes
    assert "Predict tab" in release_notes
    assert "mma-evaluate" in release_notes
    assert "Prediction-time live/manual odds are enabled by\n  default" in release_notes
    assert "does not expose per-fighter odds controls" in release_notes
    assert "Flaresolverr proxy toggle" in release_notes
    assert "wait for `/api/readiness`" in release_notes
    assert "databases to contain their imported tables" in release_notes
    assert "write matching host-side `DATABASE_URL` and `ODDS_DATABASE_URL`" in release_notes
    assert "Docker builds ignore `.env`" in release_notes
    assert "The dashboard top bar mirrors this state" in release_notes
    assert "`Setup incomplete`" in release_notes
    assert "features.fight_mapping" in release_notes
    assert "bestfightodds.bfo" in release_notes
    assert "Dashboard jobs are serialized" in release_notes
    assert "docker compose up --build db web" in release_notes
    assert "docker compose up --no-deps --build web" in readme
    assert "docker compose up --no-deps --build web" in release_notes
    assert "host.docker.internal" in readme
    assert "host-gateway" in release_notes
    assert "uv run mma-docker-smoke" in release_notes
    assert "uv run pytest -q" in release_notes
    assert "uv run mma-release-audit" in release_notes
    assert "MMA_AI_RUN_BROWSER_E2E=1" in release_notes
    assert "test_predict_tab_browser_predicts_next_ufc_event" in release_notes
    assert "/vendor/plotly.min.js" in release_notes
    assert "/static/icons.js" in release_notes

    assert "postgres:18.1" in compose
    assert '"127.0.0.1:${MMA_AI_WEB_PORT:-8000}:8000"' in compose
    assert "MMA_AI_DATA_DIR: /app/data" in compose
    assert "postgres-data:/var/lib/postgresql" in compose
    assert "postgres-data:/var/lib/postgresql/data" not in compose
    assert "./docker/postgres-init:/docker-entrypoint-initdb.d:ro" in compose
    assert "MMA_AI_COMPOSE_DATABASE_URL:-postgresql://postgres:postgres@db:5432/mma-ai" in compose
    assert "MMA_AI_COMPOSE_ODDS_DATABASE_URL:-postgresql://postgres:postgres@db:5432/odds" in compose
    assert "host.docker.internal:host-gateway" in compose
    assert "MMA_AI_POSTGRES_PORT:-5432" in compose
    assert "LLM_PROVIDER: ${LLM_PROVIDER:-}" in compose
    assert "LLM_MODEL: ${LLM_MODEL:-}" in compose
    assert "ANTHROPIC_API_KEY: ${ANTHROPIC_API_KEY:-}" in compose
    assert "OPENROUTER_API_KEY: ${OPENROUTER_API_KEY:-}" in compose
    assert "DEEPSEEK_API_KEY: ${DEEPSEEK_API_KEY:-}" in compose
    assert "MISTRAL_API_KEY: ${MISTRAL_API_KEY:-}" in compose
    assert "TOGETHER_API_KEY: ${TOGETHER_API_KEY:-}" in compose
    assert "PERPLEXITY_API_KEY: ${PERPLEXITY_API_KEY:-}" in compose
    assert "depends_on:" in compose
    assert "condition: service_healthy" in compose
    assert "artifacts" in dockerignore
    assert ".cursor" in dockerignore
    assert ".env" in dockerignore
    assert ".env.local" in dockerignore
    assert ".webapp*.log" in dockerignore
    assert "tests" in dockerignore
    assert "*.log" in dockerignore
    assert "*.csv" in dockerignore
    assert "*.html" in dockerignore
    assert "logs" in dockerignore
    assert "AutogluonModels" in dockerignore
    assert "AutoGluonModels" in dockerignore
    assert "!libs/web/static/index.html" in dockerignore
    assert "AutogluonModels/" in gitignore
    assert "AutoGluonModels/" in gitignore
    assert postgres_init.strip() == "CREATE DATABASE odds;"

    assert "matching local `DATABASE_URL` and" in huggingface_docs
    assert "./setup.sh" in huggingface_docs
    assert "Setup starts the bundled local stack and waits for the dashboard readiness" in huggingface_docs
    assert "To restart the already bootstrapped stack later" in huggingface_docs
    assert "Local `uv run ...` commands load this" in huggingface_docs
    assert "--force-import" in huggingface_docs
    assert "processed/training_data_dec.csv" in huggingface_docs
    assert "PostgreSQL 18.1 Docker service" in huggingface_docs
    assert "uv run mma-rebuild-db --scrape --reset-db --odds-features" in huggingface_docs
    assert "postgresql://postgres:postgres@localhost:5432/mma-ai" in huggingface_docs
    assert "postgresql://postgres:postgres@localhost:5432/odds" in huggingface_docs


def test_env_example_lists_public_configuration_without_real_secrets():
    env_example = read_text(".env.example")

    for key in [
        "DATABASE_URL",
        "ODDS_DATABASE_URL",
        "MMA_AI_COMPOSE_DATABASE_URL",
        "MMA_AI_COMPOSE_ODDS_DATABASE_URL",
        "MMA_AI_DATA_DIR",
        "MMA_AI_MODELS_DIR",
        "MMA_AI_UFCSTATS_DIR",
        "MMA_AI_POSTGRES_PORT",
        "MMA_AI_WEB_PORT",
        "LLM_PROVIDER",
        "LLM_MODEL",
        "LLM_API_KEY",
        "LLM_BASE_URL",
        "OPENAI_API_KEY",
        "ANTHROPIC_API_KEY",
        "XAI_API_KEY",
        "OPENROUTER_API_KEY",
        "DEEPSEEK_API_KEY",
        "MISTRAL_API_KEY",
        "TOGETHER_API_KEY",
        "PERPLEXITY_API_KEY",
        "GEMINI_API_KEY",
        "GOOGLE_API_KEY",
        "THE_ODDS_API_KEY",
    ]:
        assert f"{key}=" in env_example

    assert "DATABASE_URL=postgresql://postgres:postgres@localhost:5432/mma-ai" in env_example
    assert "ODDS_DATABASE_URL=postgresql://postgres:postgres@localhost:5432/odds" in env_example
    assert "host.docker.internal" in env_example
    assert "host.docker.internal to the host gateway" in env_example
    assert "secret" not in env_example.lower()


def test_public_repo_tracks_seed_raw_csvs_and_no_heavy_generated_artifacts():
    tracked = set(git_ls_files())
    seed_paths = {
        "data/raw/ufcstats/competitions.csv",
        "data/raw/ufcstats/individuals.csv",
    }

    assert seed_paths.issubset(tracked)
    assert "libs/web/static/index.html" in tracked
    for seed_path in seed_paths:
        path = ROOT / seed_path
        assert path.exists()
        with path.open(newline="", encoding="utf-8", errors="replace") as handle:
            reader = csv.reader(handle)
            header = next(reader)
            row_count = sum(1 for _row in reader)
        assert row_count > 1000
        assert len(header) > 5

    forbidden_prefixes = (".cursor/", "AutoGluonModels/", "AutogluonModels/", "artifacts/", "pics/", "data/predictions/")
    forbidden_suffixes = (".png", ".jpg", ".jpeg", ".gif", ".ipynb")
    generated_data_files = {
        "data/prediction_data.csv",
        "data/training_data.csv",
        "data/training_data_dec.csv",
    }
    forbidden = [
        path
        for path in tracked
        if path not in seed_paths
        and (
            path in generated_data_files
            or path.startswith(forbidden_prefixes)
            or path.lower().endswith(forbidden_suffixes)
        )
    ]

    assert forbidden == []


def test_removed_training_chat_surface_stays_removed():
    searchable_paths = [
        "README.md",
        "AGENTS.md",
        "CLAUDE.md",
        "docs/RELEASE_READINESS.md",
        "setup.ps1",
        "setup.sh",
        "libs/web/app.py",
        "libs/web/models.py",
        "libs/web/static/index.html",
        "libs/web/static/app.js",
    ]
    combined = "\n".join(read_text(path) for path in searchable_paths)

    assert "training chat" not in combined.lower()
    assert "/api/train/chat" not in combined
    assert "TrainingChatRequest" not in combined
    assert not (ROOT / "libs/web/training_chat.py").exists()
