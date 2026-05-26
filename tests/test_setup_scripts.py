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

    result = subprocess.run(
        [bash, "-n", "scripts/verify_hf_manifest.sh"],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stdout + result.stderr


def test_bash_setup_script_is_tracked_executable():
    result = subprocess.run(
        ["git", "ls-files", "--stage", "setup.sh"],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=True,
    )

    assert result.stdout.startswith("100755 ")


def test_bash_setup_scripts_are_pinned_to_lf_line_endings():
    result = subprocess.run(
        ["git", "check-attr", "eol", "--", "setup.sh", "scripts/verify_hf_manifest.sh"],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=True,
    )

    assert "setup.sh: eol: lf" in result.stdout
    assert "scripts/verify_hf_manifest.sh: eol: lf" in result.stdout
    assert b"\r\n" not in (ROOT / "setup.sh").read_bytes()
    assert b"\r\n" not in (ROOT / "scripts" / "verify_hf_manifest.sh").read_bytes()


def test_setup_scripts_help_exits_before_install_work():
    shell = shutil.which("powershell") or shutil.which("pwsh")
    if shell:
        args = [shell, "-NoProfile"]
        if Path(shell).name.lower().startswith("powershell"):
            args.extend(["-ExecutionPolicy", "Bypass"])
        args.extend(["-File", ".\\setup.ps1", "-Help"])
        result = subprocess.run(args, cwd=ROOT, text=True, capture_output=True, check=False)

        assert result.returncode == 0, result.stdout + result.stderr
        assert "MMA AI setup" in result.stdout
        assert "-ForceImport" in result.stdout
        assert "docker compose" not in result.stderr.lower()

    bash = shutil.which("bash")
    if bash:
        result = subprocess.run([bash, "setup.sh", "--help"], cwd=ROOT, text=True, capture_output=True, check=False)
        combined = result.stdout + result.stderr
        if result.returncode != 0 and "Windows Subsystem for Linux has no installed distributions" in combined:
            pytest.skip("bash is present but WSL is not configured")

        assert result.returncode == 0, combined
        assert "MMA AI setup" in result.stdout
        assert "./setup.sh" in result.stdout
        assert "--force-import" in result.stdout
        assert "docker compose" not in result.stderr.lower()

    if not shell and not bash:
        pytest.skip("Neither PowerShell nor bash is available")


def test_setup_scripts_reject_invalid_arguments_before_install_work():
    shell = shutil.which("powershell") or shutil.which("pwsh")
    if shell:
        args = [shell, "-NoProfile"]
        if Path(shell).name.lower().startswith("powershell"):
            args.extend(["-ExecutionPolicy", "Bypass"])
        args.extend(["-File", ".\\setup.ps1", "-PostgresPort", "70000", "-Help"])
        result = subprocess.run(args, cwd=ROOT, text=True, capture_output=True, check=False)

        assert result.returncode != 0
        assert "docker compose" not in result.stderr.lower()

    bash = shutil.which("bash")
    if bash:
        cases = [
            (["setup.sh", "--llm-provider", "--no-start"], "requires a value"),
            (["setup.sh", "--web-port", "not-a-port"], "TCP port number"),
            (["setup.sh", "--postgres-port", "70000"], "TCP port number"),
        ]
        for args, expected in cases:
            result = subprocess.run([bash, *args], cwd=ROOT, text=True, capture_output=True, check=False)
            combined = result.stdout + result.stderr
            if result.returncode != 0 and "Windows Subsystem for Linux has no installed distributions" in combined:
                pytest.skip("bash is present but WSL is not configured")

            assert result.returncode == 2, combined
            assert expected in combined
            assert "docker compose" not in combined.lower()

    if not shell and not bash:
        pytest.skip("Neither PowerShell nor bash is available")


def test_setup_scripts_download_restore_configure_and_start_dashboard():
    powershell = read_text("setup.ps1")
    bash = read_text("setup.sh")
    bash_manifest_helper = read_text("scripts/verify_hf_manifest.sh")

    for script in (powershell, bash):
        assert "MMA AI setup" in script
        assert "Show this help and exit before Docker or downloads" in script
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
        assert "OpenRouter" in script
        assert "DeepSeek" in script
        assert "Mistral" in script
        assert "Together AI" in script
        assert "Perplexity Sonar" in script
        assert "Local model" in script
        assert "MMA_AI_POSTGRES_PORT" in script
        assert "MMA_AI_WEB_PORT" in script
        assert "force-import" in script.lower() or "ForceImport" in script
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
        assert ".db-import-complete" in script
        assert "features.fight_mapping" in script
        assert "bestfightodds.bfo" in script
        assert "Using existing imported Postgres databases" in script
        assert "Database import finished but required tables were not found." in script
    assert "Hugging Face manifest entry" in powershell
    assert "Update setup artifact checksums before downloading large artifacts." in powershell
    assert "Hugging Face manifest entry" in bash_manifest_helper
    assert "Update setup artifact checksums before downloading large artifacts." in bash_manifest_helper


def test_setup_scripts_validate_huggingface_manifest_before_large_downloads():
    powershell = read_text("setup.ps1")
    bash = read_text("setup.sh")
    bash_manifest_helper = read_text("scripts/verify_hf_manifest.sh")

    assert "function Test-ManifestArtifactPins" in powershell
    assert "ConvertFrom-Json" in powershell
    assert 'Where-Object { $_.Path -ne "manifest.json" }' in powershell
    manifest_validation = powershell.index("Test-ManifestArtifactPins\n\n    foreach ($artifact")
    large_download_loop = powershell.index('foreach ($artifact in ($Artifacts | Where-Object { $_.Path -ne "manifest.json" }))')
    assert manifest_validation < large_download_loop
    assert "Test-ManifestArtifactPins\nAssert-ArtifactCache" in powershell

    assert "validate_manifest_artifact_pins()" in bash
    assert 'pins+=("$relative=$expected")' in bash
    assert 'bash "$ROOT/scripts/verify_hf_manifest.sh" "$manifest" "${pins[@]}"' in bash
    assert "manifest_sha_for_path()" in bash_manifest_helper
    assert 'current_path == wanted && current_sha != ""' in bash_manifest_helper
    assert "does not match the setup pin" in bash_manifest_helper
    assert 'download_file "manifest.json" ""\n  validate_manifest_artifact_pins' in bash
    assert '[[ "$relative" == "manifest.json" ]] && continue' in bash
    assert "validate_manifest_artifact_pins\nassert_artifact_cache" in bash


def test_bash_setup_declares_local_tool_prerequisites():
    bash = read_text("setup.sh")
    bash_manifest_helper = read_text("scripts/verify_hf_manifest.sh")

    for command in ("docker", "curl", "tar", "awk", "grep", "mktemp"):
        assert f"require_command {command}" in bash

    assert "require_any_command()" in bash
    assert 'require_any_command "sha256sum or shasum" sha256sum shasum' in bash
    assert "Required command 'sha256sum or shasum' was not found" in bash
    assert (
        "else\n"
        "    echo \"Required command 'sha256sum or shasum' was not found. Install one and rerun setup.\" >&2\n"
        "    exit 1\n"
        "  fi"
    ) in bash
    assert "command -v awk" in bash_manifest_helper
    assert "Required command 'awk' was not found" in bash_manifest_helper


def test_bash_huggingface_manifest_validation_pairs_paths_and_hashes():
    bash = shutil.which("bash")
    if not bash:
        pytest.skip("bash is not available")

    manifest = ROOT / ".pytest_cache" / "manifest-test.json"
    manifest.parent.mkdir(exist_ok=True)
    manifest.write_text(
        """
{
  "files": [
    {
      "path": "dumps/mma-ai.postgres-custom",
      "sha256": "AAA111"
    },
    {
      "path": "dumps/odds.postgres-custom",
      "sha256": "BBB222"
    }
  ]
}
""",
        encoding="utf-8",
    )

    passing = subprocess.run(
        [
            bash,
            "scripts/verify_hf_manifest.sh",
            ".pytest_cache/manifest-test.json",
            "dumps/mma-ai.postgres-custom=AAA111",
            "dumps/odds.postgres-custom=BBB222",
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert passing.returncode == 0, passing.stdout + passing.stderr

    failing = subprocess.run(
        [
            bash,
            "scripts/verify_hf_manifest.sh",
            ".pytest_cache/manifest-test.json",
            "dumps/mma-ai.postgres-custom=BBB222",
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    combined = failing.stdout + failing.stderr
    if failing.returncode != 0 and "Windows Subsystem for Linux has no installed distributions" in combined:
        pytest.skip("bash is present but WSL is not configured")

    assert failing.returncode == 1, combined
    assert "dumps/mma-ai.postgres-custom" in combined
    assert "does not match the setup pin" in combined


def test_setup_scripts_clear_stale_llm_key_for_keyless_custom_and_local_endpoints():
    powershell = read_text("setup.ps1")
    bash = read_text("setup.sh")

    assert '$normalizedProvider -in @("local", "custom")' in powershell
    assert 'Set-EnvValue "LLM_API_KEY" ""' in powershell
    assert '[[ "$provider" == "local" || "$provider" == "custom" ]]' in bash
    assert 'set_env_value "LLM_API_KEY" ""' in bash


def test_setup_scripts_include_hosted_openai_compatible_llm_providers():
    powershell = read_text("setup.ps1")
    bash = read_text("setup.sh")

    for script in (powershell, bash):
        assert "openrouter" in script
        assert "https://openrouter.ai/api/v1" in script
        assert "OPENROUTER_API_KEY" in script
        assert "deepseek" in script
        assert "https://api.deepseek.com" in script
        assert "DEEPSEEK_API_KEY" in script
        assert "mistral" in script
        assert "https://api.mistral.ai/v1" in script
        assert "MISTRAL_API_KEY" in script
        assert "together" in script
        assert "https://api.together.ai/v1" in script
        assert "TOGETHER_API_KEY" in script
        assert "perplexity" in script
        assert "https://api.perplexity.ai" in script
        assert "PERPLEXITY_API_KEY" in script
        assert "openai-compatible" in script


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


def test_setup_scripts_validate_database_import_before_resuming():
    powershell = read_text("setup.ps1")
    bash = read_text("setup.sh")

    assert "[switch]$ForceImport" in powershell
    assert "function Test-DatabaseTableExists" in powershell
    assert "function Test-DatabaseImportComplete" in powershell
    assert "function Mark-DatabaseImportComplete" in powershell
    assert "function Clear-DatabaseImportMarker" in powershell
    assert "to_regclass('$QualifiedTable')" in powershell
    assert "features.fight_mapping" in powershell
    assert "bestfightodds.bfo" in powershell
    assert "-not $ForceImport -and (Test-DatabaseImportComplete)" in powershell
    assert "Start-PostgresForImport\n\n    if (-not $ForceImport -and (Test-DatabaseImportComplete))" in powershell

    assert "--force-import" in bash
    assert "database_table_exists()" in bash
    assert "database_import_complete()" in bash
    assert "mark_database_import_complete()" in bash
    assert "clear_database_import_marker()" in bash
    assert "to_regclass('$qualified_table')" in bash
    assert "features.fight_mapping" in bash
    assert "bestfightodds.bfo" in bash
    assert '[[ "$FORCE_IMPORT" -eq 0 ]] && database_import_complete' in bash
    assert 'start_postgres_for_import\n\n  if [[ "$FORCE_IMPORT" -eq 0 ]] && database_import_complete' in bash


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
    assert "function Get-ReadinessRecoveryHint" in powershell
    assert "function Get-WebReadinessStatus" in powershell
    assert "function Test-WebReady" in powershell
    assert "function Wait-ForWeb" in powershell
    assert 'Invoke-WebRequest -Uri "$WebUrl/api/readiness"' in powershell
    assert "-TimeoutSec 30" in powershell
    assert "Get-WebReadinessStatus $WebUrl" in powershell
    assert "Last readiness response:" in powershell
    assert "docker compose logs --tail 120 web db" in powershell
    assert "-ForceImport" in powershell
    assert "-ForceDownload" in powershell
    assert "Wait-ForWeb $webUrl" in powershell
    assert powershell.index("Wait-ForWeb $webUrl") < powershell.index("Start-Process $webUrl")

    assert "readiness_response()" in bash
    assert "readiness_recovery_hint()" in bash
    assert "web_ready()" in bash
    assert "wait_for_web()" in bash
    assert "curl -sS --max-time 30 -w" in bash
    assert "Last readiness response:" in bash
    assert "docker compose logs --tail 120 web db" in bash
    assert "--force-import" in bash
    assert "--force-download" in bash
    assert 'wait_for_web "$WEB_URL"' in bash
    assert bash.index('wait_for_web "$WEB_URL"') < bash.index('xdg-open "$WEB_URL"')
