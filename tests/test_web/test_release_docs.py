from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def read_text(path: str) -> str:
    return (ROOT / path).read_text(encoding="utf-8")


def test_public_release_docs_cover_runtime_and_dashboard_surface():
    readme = read_text("README.md")
    agents = read_text("AGENTS.md")
    claude = read_text("CLAUDE.md")
    release_notes = read_text("docs/RELEASE_READINESS.md")
    compose = read_text("docker-compose.yml")
    dockerignore = read_text(".dockerignore")
    postgres_init = read_text("docker/postgres-init/01-create-odds.sql")

    assert "docker compose up --build" in readme
    assert "setup.ps1" in readme
    assert "setup.sh" in readme
    assert "ag-20260304_110750-win-extreme" in readme
    assert "auxiliary `odds` database" in readme
    assert "Python 3.10-3.12" in readme
    assert "Data: refresh raw UFCStats CSVs" in readme
    assert "Train: launch model training" in readme
    assert "Predict: choose a model" in readme
    assert "docker compose up --build web" in readme
    assert "MMA_AI_POSTGRES_PORT" in readme
    assert "--postgres-port 55432" in readme
    assert "AGENTS.md" in readme
    assert "CLAUDE.md" in readme
    assert "/api/health" in read_text("Dockerfile")

    assert "Data tab" in agents
    assert "Train tab" in agents
    assert "Predict tab" in agents
    assert "01-create-odds.sql" in agents
    assert "/vendor/plotly.min.js" in agents
    assert "static/icons.js" in agents
    assert "features.fight_stats_fe" in agents
    assert "MMA_AI_DATA_DIR" in agents
    assert "output directory" in agents
    assert "YYYY-MM-DD" in agents
    assert "prediction_data.csv" in claude
    assert "training_data.csv" in claude
    assert "MMA_AI_DATA_DIR" in claude
    assert "01-create-odds.sql" in claude
    assert "/vendor/plotly.min.js" in claude
    assert "/static/icons.js" in claude
    assert "output directory" in claude
    assert "YYYY-MM-DD" in claude

    assert "dashboard release candidate" in release_notes
    assert "setup.ps1" in release_notes
    assert "setup.sh" in release_notes
    assert "uv run mma-web" in release_notes
    assert "docker compose up --build" in release_notes
    assert "Data tab" in release_notes
    assert "Train tab" in release_notes
    assert "Predict tab" in release_notes
    assert "BFO odds/features" in release_notes
    assert "uv run pytest -q" in release_notes

    assert "postgres:17" in compose
    assert '"8000:8000"' in compose
    assert "MMA_AI_DATA_DIR: /app/data" in compose
    assert "./docker/postgres-init:/docker-entrypoint-initdb.d:ro" in compose
    assert "MMA_AI_COMPOSE_DATABASE_URL:-postgresql://postgres:postgres@db:5432/mma-ai" in compose
    assert "MMA_AI_COMPOSE_ODDS_DATABASE_URL:-postgresql://postgres:postgres@db:5432/odds" in compose
    assert "MMA_AI_POSTGRES_PORT:-5432" in compose
    assert "depends_on" not in compose
    assert "artifacts" in dockerignore
    assert postgres_init.strip() == "CREATE DATABASE odds;"


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
        "GEMINI_API_KEY",
        "GOOGLE_API_KEY",
        "THE_ODDS_API_KEY",
    ]:
        assert f"{key}=" in env_example

    assert "DATABASE_URL=postgresql://postgres:postgres@localhost:5432/mma-ai" in env_example
    assert "ODDS_DATABASE_URL=postgresql://postgres:postgres@localhost:5432/odds" in env_example
    assert "host.docker.internal" in env_example
    assert "secret" not in env_example.lower()
