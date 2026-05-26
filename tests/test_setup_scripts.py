from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def read_text(path: str) -> str:
    return (ROOT / path).read_text(encoding="utf-8")


def test_setup_scripts_download_restore_configure_and_start_dashboard():
    powershell = read_text("setup.ps1")
    bash = read_text("setup.sh")

    for script in (powershell, bash):
        assert "https://huggingface.co/datasets/DanMcInerney/mma-ai/resolve/main" in script
        assert "dumps/mma-ai.postgres-custom" in script
        assert "dumps/odds.postgres-custom" in script
        assert "processed/prediction_data.csv" in script
        assert "processed/training_data.csv" in script
        assert "processed/training_data_dec.csv" in script
        assert "models/ag-20260304_110750-win-extreme.tar.gz" in script
        assert "248511976D55895BE2C167F2F8FA8C4013E635B39A9BAB0D5F28C0916B5AAD74" in script
        assert "pg_restore" in script
        assert "--clean" in script
        assert "--if-exists" in script
        assert "--no-owner" in script
        assert "GEMINI_API_KEY" in script
        assert "MMA_AI_POSTGRES_PORT" in script
        assert "docker compose up" in script
        assert "http://127.0.0.1:8000" in script


def test_setup_scripts_pin_compose_database_and_starter_model():
    powershell = read_text("setup.ps1")
    bash = read_text("setup.sh")

    for script in (powershell, bash):
        assert "MMA_AI_COMPOSE_DATABASE_URL" in script
        assert "postgresql://postgres:postgres@db:5432/mma-ai" in script
        assert "MMA_AI_COMPOSE_ODDS_DATABASE_URL" in script
        assert "postgresql://postgres:postgres@db:5432/odds" in script
        assert "55432" in script
        assert "ag-20260304_110750-win-extreme" in script
        assert "AutogluonModels" in script
