import shutil
import subprocess
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]


def read_text(path: str) -> str:
    return (ROOT / path).read_text(encoding="utf-8")


def test_powershell_setup_script_has_valid_syntax():
    shell = shutil.which("powershell") or shutil.which("pwsh")
    if not shell:
        pytest.skip("PowerShell is not available")

    args = [shell, "-NoProfile"]
    if Path(shell).name.lower().startswith("powershell"):
        args.extend(["-ExecutionPolicy", "Bypass"])
    args.extend(
        [
            "-Command",
            (
                "$tokens = $null; $errors = $null; "
                "$path = (Resolve-Path './setup.ps1').Path; "
                "[System.Management.Automation.Language.Parser]::ParseFile($path, [ref]$tokens, [ref]$errors) | Out-Null; "
                "if ($errors.Count -gt 0) { $errors | ForEach-Object { Write-Error $_.Message }; exit 1 }"
            ),
        ]
    )
    result = subprocess.run(args, cwd=ROOT, text=True, capture_output=True, check=False)

    assert result.returncode == 0, result.stdout + result.stderr


def test_bash_setup_script_has_valid_syntax():
    bash = shutil.which("bash")
    if not bash:
        pytest.skip("bash is not available")

    result = subprocess.run([bash, "-n", "setup.sh"], cwd=ROOT, text=True, capture_output=True, check=False)
    combined = result.stdout + result.stderr
    if result.returncode != 0 and "Windows Subsystem for Linux has no installed distributions" in combined:
        pytest.skip("bash is present but WSL is not configured")

    assert result.returncode == 0, combined


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
        assert "LLM_PROVIDER" in script
        assert "LLM_MODEL" in script
        assert "LLM_API_KEY" in script
        assert "LLM_BASE_URL" in script
        assert "Anthropic Claude" in script
        assert "xAI Grok" in script
        assert "Local model" in script
        assert "MMA_AI_POSTGRES_PORT" in script
        assert "MMA_AI_WEB_PORT" in script
        assert "docker compose up" in script
        assert "db" in script
        assert "web" in script
        assert "recreating the setup database volume" in script
        assert "setup-complete" in script
        assert "extracting" in script
        assert "http://localhost:" in script
        assert "/api/readiness" in script
        assert "Waiting for MMA AI web dashboard readiness check" in script
        assert "Validating setup artifact cache" in script
        assert "Required setup artifact cache is incomplete or corrupt" in script
        assert "feats.txt" in script
        assert "predictor.pkl" in script
        assert "ensemble_info.txt" in script
        assert "final_model" in script
        assert "window_*" in script
        assert "Starter model extraction did not create a usable model directory" in script


def test_setup_scripts_detect_existing_postgres_host_port():
    powershell = read_text("setup.ps1")
    bash = read_text("setup.sh")

    assert "Test-DockerPublishedPortInUse" in powershell
    assert "Invoke-DockerComposeOptional" in powershell
    assert "Get-NetTCPConnection" in powershell
    assert "docker ps --format" in powershell
    assert "Remove-SetupDirectory" in powershell
    assert "Assert-ArtifactCache" in powershell
    assert "Test-LocalhostPortOwnedByNonDocker" in powershell
    assert "docker_published_port_in_use" in bash
    assert "docker ps --format" in bash
    assert "safe_remove_setup_dir" in bash
    assert "assert_artifact_cache" in bash


def test_setup_scripts_validate_starter_model_before_resuming():
    powershell = read_text("setup.ps1")
    bash = read_text("setup.sh")

    assert "function Test-StarterModelComplete" in powershell
    assert "Test-StarterModelComplete $modelDir" in powershell
    assert "Starter model is missing required files; re-extracting" in powershell
    assert "Remove-Item -LiteralPath $markerPath -Force" in powershell

    assert "starter_model_complete()" in bash
    assert 'starter_model_complete "$model_dir"' in bash
    assert "Starter model is missing required files; re-extracting" in bash
    assert 'rm -f "$marker_path"' in bash


def test_setup_scripts_pin_compose_database_and_starter_model():
    powershell = read_text("setup.ps1")
    bash = read_text("setup.sh")

    for script in (powershell, bash):
        assert "MMA_AI_COMPOSE_DATABASE_URL" in script
        assert "postgresql://postgres:postgres@db:5432/mma-ai" in script
        assert "MMA_AI_COMPOSE_ODDS_DATABASE_URL" in script
        assert "postgresql://postgres:postgres@db:5432/odds" in script
        assert "DATABASE_URL" in script
        assert "postgresql://postgres:postgres@localhost:" in script
        assert "ODDS_DATABASE_URL" in script
        assert "55432" in script
        assert "18000" in script
        assert "ag-20260304_110750-win-extreme" in script
        assert "AutogluonModels" in script


def test_setup_scripts_start_database_and_web_together():
    powershell = read_text("setup.ps1")
    bash = read_text("setup.sh")

    assert 'Invoke-DockerCompose @("up", "-d", "--build", "db", "web")' in powershell
    assert "docker compose up -d --build db web" in powershell
    assert "docker compose up -d --build db web" in bash


def test_setup_scripts_wait_for_web_health_before_opening():
    powershell = read_text("setup.ps1")
    bash = read_text("setup.sh")

    assert "function Format-WebReadinessDetail" in powershell
    assert "function Get-WebReadinessStatus" in powershell
    assert "function Test-WebReady" in powershell
    assert "function Wait-ForWeb" in powershell
    assert 'Invoke-WebRequest -Uri "$WebUrl/api/readiness"' in powershell
    assert "Get-WebReadinessStatus $WebUrl" in powershell
    assert "Last readiness response:" in powershell
    assert "Wait-ForWeb $webUrl" in powershell
    assert powershell.index("Wait-ForWeb $webUrl") < powershell.index("Start-Process $webUrl")

    assert "readiness_response()" in bash
    assert "web_ready()" in bash
    assert "wait_for_web()" in bash
    assert 'curl -sS -w' in bash
    assert "Last readiness response:" in bash
    assert 'wait_for_web "$WEB_URL"' in bash
    assert bash.index('wait_for_web "$WEB_URL"') < bash.index('xdg-open "$WEB_URL"')
