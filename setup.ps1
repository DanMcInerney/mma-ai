param(
    [switch]$Help,
    [switch]$SkipDownload,
    [switch]$SkipImport,
    [switch]$ForceImport,
    [switch]$NoStart,
    [switch]$NoOpen,
    [switch]$ForceDownload,
    [switch]$SkipLlmPrompt,
    [string]$GeminiApiKey,
    [string]$LlmProvider,
    [string]$LlmModel,
    [string]$LlmApiKey,
    [string]$LlmBaseUrl,
    [ValidateRange(0, 65535)]
    [int]$PostgresPort = 0,
    [ValidateRange(0, 65535)]
    [int]$WebPort = 0
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $Root

$DatasetBaseUrl = "https://huggingface.co/datasets/DanMcInerney/mma-ai/resolve/main"
$ArtifactsRoot = Join-Path $Root "artifacts\mma-ai-dataset"
$ModelName = "ag-20260304_110750-win-extreme"

$Artifacts = @(
    [pscustomobject]@{ Path = "manifest.json"; Sha256 = "" },
    [pscustomobject]@{ Path = "dumps/mma-ai.postgres-custom"; Sha256 = "0EB0D2CBDECC55B6EA625F70A12914F72BD0FDCF67B91BCDFC0146393E1A7B7A" },
    [pscustomobject]@{ Path = "dumps/odds.postgres-custom"; Sha256 = "767AFB5C2642DD8D450B6F043F333CD5FE8589B4D8574E41831E8BBC2614F352" },
    [pscustomobject]@{ Path = "processed/training_data.csv"; Sha256 = "FFBF161D6F6E307132EB8150B5978728DED93AA9B4D3282F892C725503BA654E" },
    [pscustomobject]@{ Path = "processed/training_data_dec.csv"; Sha256 = "91D6918DFCE10C5C5C788721C58FB98AB42AC51D9FB854BA935E6CB54701EFFB" },
    [pscustomobject]@{ Path = "processed/prediction_data.csv"; Sha256 = "1C28D3B04DA412980777D38032E95A5B695C4B53BEA0014192D4D6C07413F754" },
    [pscustomobject]@{ Path = "models/ag-20260304_110750-win-extreme.tar.gz"; Sha256 = "248511976D55895BE2C167F2F8FA8C4013E635B39A9BAB0D5F28C0916B5AAD74" }
)

function Show-SetupHelp {
    Write-Host @'
MMA AI setup

Usage:
  powershell -ExecutionPolicy Bypass -File .\setup.ps1 [options]

First-time setup downloads verified Hugging Face artifacts, restores the main
and odds databases into Docker Postgres, extracts the starter model, optionally
configures LLM analytics, starts the dashboard, and waits for /api/readiness.

Options:
  -Help                 Show this help and exit before Docker or downloads.
  -SkipDownload         Reuse the existing artifact cache after validating it.
  -ForceDownload        Re-download artifacts and verify checksums.
  -SkipImport           Do not restore database dumps into Docker Postgres.
  -ForceImport          Restore database dumps even if required tables exist.
  -NoStart              Prepare files/imports but do not start the dashboard.
  -NoOpen               Start the dashboard but do not open a browser.
  -SkipLlmPrompt        Do not prompt for analytics LLM configuration.
  -PostgresPort <port>  Force the Docker Postgres host port.
  -WebPort <port>       Force the dashboard host port.
  -LlmProvider <name>   Configure analytics LLM provider non-interactively.
  -LlmModel <name>      Configure analytics LLM model.
  -LlmApiKey <token>    Configure analytics LLM API key or token.
  -LlmBaseUrl <url>     Configure custom/OpenAI-compatible API base URL.

Examples:
  powershell -ExecutionPolicy Bypass -File .\setup.ps1
  powershell -ExecutionPolicy Bypass -File .\setup.ps1 -ForceImport
  powershell -ExecutionPolicy Bypass -File .\setup.ps1 -PostgresPort 55432 -WebPort 18000
  powershell -ExecutionPolicy Bypass -File .\setup.ps1 -SkipLlmPrompt
'@
}

if ($Help) {
    Show-SetupHelp
    return
}

function Require-Command {
    param([string]$Name)
    if (-not (Get-Command $Name -ErrorAction SilentlyContinue)) {
        throw "Required command '$Name' was not found. Install it and rerun setup."
    }
}

function Join-ArtifactPath {
    param([string]$RelativePath)
    $target = $ArtifactsRoot
    foreach ($part in ($RelativePath -split "/")) {
        $target = Join-Path $target $part
    }
    return $target
}

function Test-ExpectedHash {
    param([string]$Path, [string]$ExpectedHash)
    if ([string]::IsNullOrWhiteSpace($ExpectedHash)) {
        return Test-Path -LiteralPath $Path
    }
    if (-not (Test-Path -LiteralPath $Path)) {
        return $false
    }
    $actual = (Get-FileHash -Algorithm SHA256 -LiteralPath $Path).Hash.ToUpperInvariant()
    return $actual -eq $ExpectedHash.ToUpperInvariant()
}

function Assert-ArtifactCache {
    $missingOrInvalid = @()
    foreach ($artifact in $Artifacts) {
        $target = Join-ArtifactPath $artifact.Path
        if (-not (Test-ExpectedHash $target $artifact.Sha256)) {
            $missingOrInvalid += $artifact.Path
        }
    }

    if ($missingOrInvalid.Count -gt 0) {
        $advice = if ($SkipDownload) {
            "Rerun setup without -SkipDownload, or pass -ForceDownload to refresh the cache."
        } else {
            "Rerun setup with -ForceDownload to refresh the cache."
        }
        throw "Required setup artifact cache is incomplete or corrupt: $($missingOrInvalid -join ', '). $advice"
    }
}

function Test-ManifestArtifactPins {
    $manifestPath = Join-ArtifactPath "manifest.json"
    if (-not (Test-Path -LiteralPath $manifestPath)) {
        throw "Hugging Face manifest is missing from the setup artifact cache."
    }

    $manifest = Get-Content -LiteralPath $manifestPath -Raw | ConvertFrom-Json
    $manifestFiles = @{}
    foreach ($file in @($manifest.files)) {
        $manifestFiles[$file.path] = ($file.sha256.ToUpperInvariant())
    }

    foreach ($artifact in $Artifacts) {
        if ($artifact.Path -eq "manifest.json" -or [string]::IsNullOrWhiteSpace($artifact.Sha256)) {
            continue
        }
        if (-not $manifestFiles.ContainsKey($artifact.Path) -or $manifestFiles[$artifact.Path] -ne $artifact.Sha256.ToUpperInvariant()) {
            throw "Hugging Face manifest entry for $($artifact.Path) does not match the setup pin. Update setup artifact checksums before downloading large artifacts."
        }
    }
}

function Remove-SetupDirectory {
    param([string]$Path, [string]$Parent)

    if (-not (Test-Path -LiteralPath $Path)) {
        return
    }

    $parentFullPath = [System.IO.Path]::GetFullPath($Parent).TrimEnd([System.IO.Path]::DirectorySeparatorChar, [System.IO.Path]::AltDirectorySeparatorChar)
    $targetFullPath = [System.IO.Path]::GetFullPath($Path)
    $expectedPrefix = "$parentFullPath$([System.IO.Path]::DirectorySeparatorChar)"
    if (-not $targetFullPath.StartsWith($expectedPrefix, [System.StringComparison]::OrdinalIgnoreCase)) {
        throw "Refusing to remove setup directory outside $parentFullPath`: $targetFullPath"
    }

    Remove-Item -LiteralPath $Path -Recurse -Force
}

function Download-File {
    param([string]$Url, [string]$Target)
    $parent = Split-Path -Parent $Target
    New-Item -ItemType Directory -Force $parent | Out-Null
    $tmp = "$Target.download"
    if (Test-Path -LiteralPath $tmp) {
        Remove-Item -LiteralPath $tmp -Force
    }

    $curl = Get-Command curl.exe -CommandType Application -ErrorAction SilentlyContinue
    if (-not $curl) {
        $curl = Get-Command curl -CommandType Application -ErrorAction SilentlyContinue
    }

    if ($curl) {
        & $curl.Source -L --fail --retry 3 --output $tmp $Url
        if ($LASTEXITCODE -ne 0) {
            throw "Download failed: $Url"
        }
    } else {
        Invoke-WebRequest -Uri $Url -OutFile $tmp
    }

    Move-Item -LiteralPath $tmp -Destination $Target -Force
}

function Ensure-EnvFile {
    $envPath = Join-Path $Root ".env"
    if (-not (Test-Path -LiteralPath $envPath)) {
        Copy-Item -LiteralPath (Join-Path $Root ".env.example") -Destination $envPath
    }
}

function Set-EnvValue {
    param([string]$Key, [string]$Value)
    Ensure-EnvFile
    $envPath = Join-Path $Root ".env"
    $escapedKey = [regex]::Escape($Key)
    $replacement = "$Key=$Value"
    $matched = $false
    $lines = Get-Content -LiteralPath $envPath
    $updated = foreach ($line in $lines) {
        if ($line -match "^\s*#?\s*$escapedKey=") {
            $matched = $true
            $replacement
        } else {
            $line
        }
    }
    if (-not $matched) {
        $updated += $replacement
    }
    Set-Content -LiteralPath $envPath -Value $updated -Encoding utf8
}

function Normalize-LlmProvider {
    param([string]$Provider)
    if ([string]::IsNullOrWhiteSpace($Provider)) {
        return ""
    }
    $value = $Provider.Trim().ToLowerInvariant()
    switch ($value) {
        "gemini" { return "google" }
        "google" { return "google" }
        "openai" { return "openai" }
        "codex" { return "codex" }
        "anthropic" { return "anthropic" }
        "claude" { return "anthropic" }
        "grok" { return "grok" }
        "xai" { return "grok" }
        "openrouter" { return "openrouter" }
        "open-router" { return "openrouter" }
        "deepseek" { return "deepseek" }
        "deep-seek" { return "deepseek" }
        "mistral" { return "mistral" }
        "together" { return "together" }
        "togetherai" { return "together" }
        "together-ai" { return "together" }
        "perplexity" { return "perplexity" }
        "sonar" { return "perplexity" }
        "local" { return "local" }
        "ollama" { return "local" }
        "lmstudio" { return "local" }
        "lm-studio" { return "local" }
        "custom" { return "custom" }
        "openai-compatible" { return "custom" }
        default { return $value }
    }
}

function Get-LlmDefaultModel {
    param([string]$Provider)
    switch ($Provider) {
        "google" { return "gemini-1.5-pro" }
        "openai" { return "gpt-4o-mini" }
        "codex" { return "gpt-5-codex" }
        "anthropic" { return "claude-3-5-sonnet-latest" }
        "grok" { return "grok-2-latest" }
        "openrouter" { return "~openai/gpt-latest" }
        "deepseek" { return "deepseek-chat" }
        "mistral" { return "mistral-large-latest" }
        "together" { return "meta-llama/Llama-3.3-70B-Instruct-Turbo" }
        "perplexity" { return "sonar-pro" }
        "local" { return "llama3.1" }
        default { return "gpt-4o-mini" }
    }
}

function Get-LlmDefaultBaseUrl {
    param([string]$Provider)
    switch ($Provider) {
        "openai" { return "https://api.openai.com/v1" }
        "codex" { return "https://api.openai.com/v1" }
        "grok" { return "https://api.x.ai/v1" }
        "openrouter" { return "https://openrouter.ai/api/v1" }
        "deepseek" { return "https://api.deepseek.com" }
        "mistral" { return "https://api.mistral.ai/v1" }
        "together" { return "https://api.together.ai/v1" }
        "perplexity" { return "https://api.perplexity.ai" }
        "local" { return "http://host.docker.internal:11434/v1" }
        "custom" { return "http://host.docker.internal:11434/v1" }
        default { return "" }
    }
}

function Read-SecretPlainText {
    param([string]$Prompt)
    $secureValue = Read-Host $Prompt -AsSecureString
    $ptr = [Runtime.InteropServices.Marshal]::SecureStringToBSTR($secureValue)
    try {
        return [Runtime.InteropServices.Marshal]::PtrToStringBSTR($ptr)
    } finally {
        [Runtime.InteropServices.Marshal]::ZeroFreeBSTR($ptr)
    }
}

function Set-LlmConfiguration {
    param(
        [string]$Provider,
        [string]$Model,
        [string]$ApiKey,
        [string]$BaseUrl
    )

    $normalizedProvider = Normalize-LlmProvider $Provider
    if ([string]::IsNullOrWhiteSpace($normalizedProvider)) {
        throw "LLM provider is required when configuring analytics."
    }

    if ([string]::IsNullOrWhiteSpace($Model)) {
        $Model = Get-LlmDefaultModel $normalizedProvider
    }

    if ([string]::IsNullOrWhiteSpace($BaseUrl)) {
        $BaseUrl = Get-LlmDefaultBaseUrl $normalizedProvider
    }

    Set-EnvValue "LLM_PROVIDER" $normalizedProvider
    Set-EnvValue "LLM_MODEL" $Model
    Set-EnvValue "LLM_BASE_URL" $BaseUrl

    if (-not [string]::IsNullOrWhiteSpace($ApiKey)) {
        Set-EnvValue "LLM_API_KEY" $ApiKey
        switch ($normalizedProvider) {
            "google" {
                Set-EnvValue "GEMINI_API_KEY" $ApiKey
                Set-EnvValue "GOOGLE_API_KEY" $ApiKey
            }
            "openai" { Set-EnvValue "OPENAI_API_KEY" $ApiKey }
            "codex" { Set-EnvValue "OPENAI_API_KEY" $ApiKey }
            "anthropic" { Set-EnvValue "ANTHROPIC_API_KEY" $ApiKey }
            "grok" {
                Set-EnvValue "XAI_API_KEY" $ApiKey
                Set-EnvValue "GROK_API_KEY" $ApiKey
            }
            "openrouter" { Set-EnvValue "OPENROUTER_API_KEY" $ApiKey }
            "deepseek" { Set-EnvValue "DEEPSEEK_API_KEY" $ApiKey }
            "mistral" { Set-EnvValue "MISTRAL_API_KEY" $ApiKey }
            "together" { Set-EnvValue "TOGETHER_API_KEY" $ApiKey }
            "perplexity" { Set-EnvValue "PERPLEXITY_API_KEY" $ApiKey }
        }
    } elseif ($normalizedProvider -in @("local", "custom")) {
        Set-EnvValue "LLM_API_KEY" ""
    }

    Write-Host "Configured LLM analytics: provider=$normalizedProvider model=$Model"
}

function Resolve-LlmProviderChoice {
    param([string]$Choice)
    $value = $Choice.Trim().ToLowerInvariant()
    switch ($value) {
        "" { return "openai" }
        "1" { return "openai" }
        "2" { return "codex" }
        "3" { return "anthropic" }
        "4" { return "google" }
        "5" { return "grok" }
        "6" { return "openrouter" }
        "7" { return "deepseek" }
        "8" { return "mistral" }
        "9" { return "together" }
        "10" { return "perplexity" }
        "11" { return "local" }
        "12" { return "custom" }
        default { return (Normalize-LlmProvider $value) }
    }
}

function Configure-LlmAnalytics {
    if ($GeminiApiKey) {
        Set-LlmConfiguration -Provider "google" -Model $LlmModel -ApiKey $GeminiApiKey -BaseUrl $LlmBaseUrl
        return
    }

    if ($LlmProvider -or $LlmModel -or $LlmApiKey -or $LlmBaseUrl) {
        $provider = if ($LlmProvider) { $LlmProvider } else { "custom" }
        Set-LlmConfiguration -Provider $provider -Model $LlmModel -ApiKey $LlmApiKey -BaseUrl $LlmBaseUrl
        return
    }

    if ($SkipLlmPrompt) {
        return
    }

    $answer = Read-Host "Set up LLM analytics now? [y/N]"
    if ($answer -notmatch "^(y|yes)$") {
        return
    }

    Write-Host "Choose your LLM provider:"
    Write-Host "  1) OpenAI"
    Write-Host "  2) Codex / OpenAI-compatible"
    Write-Host "  3) Anthropic Claude"
    Write-Host "  4) Google Gemini"
    Write-Host "  5) xAI Grok"
    Write-Host "  6) OpenRouter"
    Write-Host "  7) DeepSeek"
    Write-Host "  8) Mistral"
    Write-Host "  9) Together AI"
    Write-Host "  10) Perplexity Sonar"
    Write-Host "  11) Local model (Ollama or LM Studio)"
    Write-Host "  12) Custom OpenAI-compatible endpoint"
    $provider = Resolve-LlmProviderChoice (Read-Host "Provider [1]")

    $defaultModel = Get-LlmDefaultModel $provider
    $model = Read-Host "Model name [$defaultModel]"
    if ([string]::IsNullOrWhiteSpace($model)) {
        $model = $defaultModel
    }

    $baseUrl = Get-LlmDefaultBaseUrl $provider
    if ($provider -in @("openai", "codex", "grok", "openrouter", "deepseek", "mistral", "together", "perplexity")) {
        $override = Read-Host "API base URL [$baseUrl]"
        if (-not [string]::IsNullOrWhiteSpace($override)) {
            $baseUrl = $override
        }
    } elseif ($provider -in @("local", "custom")) {
        $override = Read-Host "OpenAI-compatible base URL [$baseUrl]"
        if (-not [string]::IsNullOrWhiteSpace($override)) {
            $baseUrl = $override
        }
    }

    $apiKey = ""
    if ($provider -eq "local") {
        $apiKey = Read-SecretPlainText "API key/token if required by your local server; otherwise press Enter"
    } elseif ($provider -eq "custom") {
        $apiKey = Read-SecretPlainText "API key/token if required by your endpoint; otherwise press Enter"
    } else {
        $apiKey = Read-SecretPlainText "API key/token"
        if ([string]::IsNullOrWhiteSpace($apiKey)) {
            Write-Host "No API key entered; skipping LLM analytics configuration."
            return
        }
    }

    Set-LlmConfiguration -Provider $provider -Model $model -ApiKey $apiKey -BaseUrl $baseUrl
}

function Invoke-DockerCompose {
    param([string[]]$ComposeArgs)
    & docker compose @ComposeArgs
    if ($LASTEXITCODE -ne 0) {
        throw "docker compose $($ComposeArgs -join ' ') failed"
    }
}

function Invoke-DockerComposeOptional {
    param([string[]]$ComposeArgs)
    $previousErrorActionPreference = $ErrorActionPreference
    try {
        $ErrorActionPreference = "Continue"
        & docker compose @ComposeArgs *> $null
    } finally {
        $ErrorActionPreference = $previousErrorActionPreference
    }
}

function Get-ComposeDbPort {
    return Get-ComposeServicePort "db" 5432
}

function Get-ComposeWebPort {
    return Get-ComposeServicePort "web" 8000
}

function Get-ComposeServicePort {
    param([string]$Service, [int]$ContainerPort)
    $previousErrorActionPreference = $ErrorActionPreference
    try {
        $ErrorActionPreference = "Continue"
        $output = & docker compose port $Service $ContainerPort 2>$null
        $exitCode = $LASTEXITCODE
    } finally {
        $ErrorActionPreference = $previousErrorActionPreference
    }

    if ($exitCode -eq 0 -and -not [string]::IsNullOrWhiteSpace($output)) {
        $lastLine = @($output)[-1].Trim()
        $portText = ($lastLine -split ":")[-1]
        $parsed = 0
        if ([int]::TryParse($portText, [ref]$parsed)) {
            return $parsed
        }
    }
    return $null
}

function Test-TcpPortAvailable {
    param([int]$Port)

    if (Test-DockerPublishedPortInUse $Port) {
        return $false
    }

    if (Test-LocalTcpListenerInUse $Port) {
        return $false
    }

    $listener = $null
    try {
        $listener = [System.Net.Sockets.TcpListener]::new([System.Net.IPAddress]::Any, $Port)
        $listener.Start()
        return $true
    } catch {
        return $false
    } finally {
        if ($listener) {
            $listener.Stop()
        }
    }
}

function Test-DockerPublishedPortInUse {
    param([int]$Port)

    $previousErrorActionPreference = $ErrorActionPreference
    try {
        $ErrorActionPreference = "Continue"
        $output = & docker ps --format "{{.Ports}}" 2>$null
        $exitCode = $LASTEXITCODE
    } finally {
        $ErrorActionPreference = $previousErrorActionPreference
    }

    if ($exitCode -ne 0) {
        return $false
    }

    foreach ($line in @($output)) {
        foreach ($publishedPort in ($line -split ",")) {
            $portMapping = $publishedPort.Trim()
            if ($portMapping -match "(^|[^0-9])$Port->") {
                return $true
            }
        }
    }

    return $false
}

function Test-LocalTcpListenerInUse {
    param([int]$Port)

    if (Get-Command Get-NetTCPConnection -ErrorAction SilentlyContinue) {
        $listeners = @(Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue)
        return $listeners.Count -gt 0
    }

    return $false
}

function Test-LocalhostPortOwnedByNonDocker {
    param([int]$Port)

    if (-not (Get-Command Get-NetTCPConnection -ErrorAction SilentlyContinue)) {
        return $false
    }

    $listeners = @(Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue)
    foreach ($listener in $listeners) {
        if ($listener.LocalAddress -notin @("127.0.0.1", "0.0.0.0")) {
            continue
        }

        $processName = (Get-Process -Id $listener.OwningProcess -ErrorAction SilentlyContinue).ProcessName
        if ($processName -and $processName -notin @("com.docker.backend", "docker-proxy", "wslrelay")) {
            return $true
        }
    }

    return $false
}

function Get-SetupPostgresPort {
    if ($PostgresPort -gt 0) {
        return $PostgresPort
    }

    $existingPort = Get-ComposeDbPort
    if ($existingPort) {
        return $existingPort
    }

    if (Test-TcpPortAvailable 5432) {
        return 5432
    }

    for ($candidate = 55432; $candidate -le 55532; $candidate++) {
        if (Test-TcpPortAvailable $candidate) {
            return $candidate
        }
    }

    throw "Could not find an available host port for PostgreSQL. Pass -PostgresPort <port> to choose one."
}

function Get-SetupWebPort {
    if ($WebPort -gt 0) {
        return $WebPort
    }

    $existingPort = Get-ComposeWebPort
    if ($existingPort -and -not (Test-LocalhostPortOwnedByNonDocker $existingPort)) {
        return $existingPort
    }

    if (Test-TcpPortAvailable 8000) {
        return 8000
    }

    for ($candidate = 18000; $candidate -le 18100; $candidate++) {
        if (Test-TcpPortAvailable $candidate) {
            return $candidate
        }
    }

    throw "Could not find an available host port for the web dashboard. Pass -WebPort <port> to choose one."
}

function Test-PostgresReady {
    $previousErrorActionPreference = $ErrorActionPreference
    try {
        $ErrorActionPreference = "Continue"
        & docker compose exec -T db pg_isready -U postgres -d postgres *> $null
        return $LASTEXITCODE -eq 0
    } finally {
        $ErrorActionPreference = $previousErrorActionPreference
    }
}

function Wait-ForPostgres {
    for ($i = 0; $i -lt 90; $i++) {
        if (Test-PostgresReady) {
            return
        }
        Start-Sleep -Seconds 2
    }
    throw "Postgres did not become ready in time."
}

function Get-DatabaseImportMarker {
    return (Join-Path $ArtifactsRoot ".db-import-complete")
}

function Test-DatabaseTableExists {
    param([string]$Database, [string]$QualifiedTable)

    $previousErrorActionPreference = $ErrorActionPreference
    try {
        $ErrorActionPreference = "Continue"
        $query = "SELECT to_regclass('$QualifiedTable') IS NOT NULL;"
        $output = & docker compose exec -T db psql -U postgres -d $Database -tAc $query 2>$null
        $exitCode = $LASTEXITCODE
    } finally {
        $ErrorActionPreference = $previousErrorActionPreference
    }

    $normalized = ((@($output) -join "") -replace "\s+", "")
    return $exitCode -eq 0 -and $normalized -eq "t"
}

function Test-DatabaseImportComplete {
    return (
        (Test-DatabaseTableExists -Database "mma-ai" -QualifiedTable "features.fight_mapping") -and
        (Test-DatabaseTableExists -Database "odds" -QualifiedTable "bestfightodds.bfo")
    )
}

function Mark-DatabaseImportComplete {
    New-Item -ItemType Directory -Force $ArtifactsRoot | Out-Null
    New-Item -ItemType File -Force (Get-DatabaseImportMarker) | Out-Null
}

function Clear-DatabaseImportMarker {
    $markerPath = Get-DatabaseImportMarker
    if (Test-Path -LiteralPath $markerPath) {
        Remove-Item -LiteralPath $markerPath -Force
    }
}

function Format-WebReadinessDetail {
    param([string]$Detail)

    if ([string]::IsNullOrWhiteSpace($Detail)) {
        return "No readiness detail returned."
    }
    return (($Detail -replace "\s+", " ").Trim())
}

function Get-ReadinessRecoveryHint {
    return "Review Docker logs with: docker compose logs --tail 120 web db. If readiness reports missing database tables, rerun setup with -ForceImport. If it reports missing CSVs or model files, rerun setup without -SkipDownload or with -ForceDownload."
}

function Get-WebReadinessStatus {
    param([string]$WebUrl)

    $previousErrorActionPreference = $ErrorActionPreference
    try {
        $ErrorActionPreference = "Stop"
        $response = Invoke-WebRequest -Uri "$WebUrl/api/readiness" -UseBasicParsing -TimeoutSec 30
        return [pscustomobject]@{
            Ready = ($response.StatusCode -ge 200 -and $response.StatusCode -lt 300)
            Detail = (Format-WebReadinessDetail $response.Content)
        }
    } catch {
        $detail = $_.Exception.Message
        if ($_.ErrorDetails -and $_.ErrorDetails.Message) {
            $detail = $_.ErrorDetails.Message
        }
        return [pscustomobject]@{
            Ready = $false
            Detail = (Format-WebReadinessDetail $detail)
        }
    } finally {
        $ErrorActionPreference = $previousErrorActionPreference
    }
}

function Test-WebReady {
    param([string]$WebUrl)

    return (Get-WebReadinessStatus $WebUrl).Ready
}

function Wait-ForWeb {
    param([string]$WebUrl)

    $lastDetail = "No readiness detail returned."
    for ($i = 0; $i -lt 90; $i++) {
        $status = Get-WebReadinessStatus $WebUrl
        if ($status.Ready) {
            return
        }
        $lastDetail = $status.Detail
        Start-Sleep -Seconds 2
    }
    throw "Web dashboard did not become ready at $WebUrl in time. Last readiness response: $lastDetail. $(Get-ReadinessRecoveryHint)"
}

function Test-StarterModelComplete {
    param([string]$ModelDir)

    if (-not (Test-Path -LiteralPath $ModelDir -PathType Container)) {
        return $false
    }
    if (-not (Test-Path -LiteralPath (Join-Path $ModelDir "feats.txt") -PathType Leaf)) {
        return $false
    }

    if (Test-Path -LiteralPath (Join-Path $ModelDir "predictor.pkl") -PathType Leaf) {
        return $true
    }
    if (-not (Test-Path -LiteralPath (Join-Path $ModelDir "ensemble_info.txt") -PathType Leaf)) {
        return $false
    }
    if (Test-Path -LiteralPath (Join-Path $ModelDir "final_model") -PathType Container) {
        return $true
    }
    $windowDirs = @(Get-ChildItem -LiteralPath $ModelDir -Directory -Filter "window_*" -ErrorAction SilentlyContinue)
    if ($windowDirs.Count -gt 0) {
        return $true
    }
    return $false
}

function Start-PostgresForImport {
    Write-Host "Starting Docker Postgres"
    try {
        Invoke-DockerCompose @("up", "-d", "db")
        Wait-ForPostgres
    } catch {
        Write-Host "Postgres did not start cleanly; recreating the setup database volume and retrying."
        Invoke-DockerCompose @("down", "--volumes", "--remove-orphans")
        Invoke-DockerCompose @("up", "-d", "db")
        Wait-ForPostgres
    }
}

function Ensure-StarterModel {
    $modelsRoot = Join-Path $Root "AutogluonModels"
    $modelDir = Join-Path $modelsRoot $ModelName
    $markerPath = Join-Path $modelsRoot ".$ModelName.setup-complete"
    $extractDir = Join-Path $modelsRoot ".$ModelName.extracting"

    New-Item -ItemType Directory -Force $modelsRoot | Out-Null

    if ((Test-StarterModelComplete $modelDir) -and (Test-Path -LiteralPath $markerPath -PathType Leaf)) {
        Write-Host "Using existing starter model $ModelName"
        return
    }

    if (Test-Path -LiteralPath $modelDir) {
        Write-Host "Starter model is missing required files; re-extracting $ModelName"
        Remove-SetupDirectory $modelDir $modelsRoot
    } else {
        Write-Host "Extracting starter model $ModelName"
    }
    if (Test-Path -LiteralPath $markerPath) {
        Remove-Item -LiteralPath $markerPath -Force
    }

    Remove-SetupDirectory $extractDir $modelsRoot
    New-Item -ItemType Directory -Force $extractDir | Out-Null

    & tar -xzf (Join-ArtifactPath "models/$ModelName.tar.gz") -C $extractDir
    if ($LASTEXITCODE -ne 0) {
        Remove-SetupDirectory $extractDir $modelsRoot
        throw "Model extraction failed."
    }

    $nestedModelDir = Join-Path $extractDir $ModelName
    if (Test-Path -LiteralPath $nestedModelDir) {
        Move-Item -LiteralPath $nestedModelDir -Destination $modelDir
    } else {
        New-Item -ItemType Directory -Force $modelDir | Out-Null
        Get-ChildItem -LiteralPath $extractDir -Force | Move-Item -Destination $modelDir
    }

    if (-not (Test-StarterModelComplete $modelDir)) {
        Remove-SetupDirectory $modelDir $modelsRoot
        Remove-SetupDirectory $extractDir $modelsRoot
        throw "Starter model extraction did not create a usable model directory."
    }

    New-Item -ItemType File -Force $markerPath | Out-Null
    Remove-SetupDirectory $extractDir $modelsRoot
}

Require-Command "docker"
Require-Command "tar"
& docker compose version *> $null
if ($LASTEXITCODE -ne 0) {
    throw "Docker Compose v2 is required. Install Docker Desktop or the Docker Compose plugin."
}

Ensure-EnvFile
Set-EnvValue "MMA_AI_COMPOSE_DATABASE_URL" "postgresql://postgres:postgres@db:5432/mma-ai"
Set-EnvValue "MMA_AI_COMPOSE_ODDS_DATABASE_URL" "postgresql://postgres:postgres@db:5432/odds"
$selectedPostgresPort = Get-SetupPostgresPort
Set-EnvValue "MMA_AI_POSTGRES_PORT" "$selectedPostgresPort"
Set-EnvValue "DATABASE_URL" "postgresql://postgres:postgres@localhost:$selectedPostgresPort/mma-ai"
Set-EnvValue "ODDS_DATABASE_URL" "postgresql://postgres:postgres@localhost:$selectedPostgresPort/odds"
if ($selectedPostgresPort -ne 5432) {
    Write-Host "Host port 5432 is unavailable; Docker Postgres will use localhost:$selectedPostgresPort."
}
$selectedWebPort = Get-SetupWebPort
Set-EnvValue "MMA_AI_WEB_PORT" "$selectedWebPort"
if ($selectedWebPort -ne 8000) {
    Write-Host "Host port 8000 is unavailable or ambiguous; the dashboard will use http://localhost:$selectedWebPort."
}

if (-not $SkipDownload) {
    $manifestArtifact = $Artifacts | Where-Object { $_.Path -eq "manifest.json" } | Select-Object -First 1
    $manifestTarget = Join-ArtifactPath $manifestArtifact.Path
    if (-not $ForceDownload -and (Test-ExpectedHash $manifestTarget $manifestArtifact.Sha256)) {
        Write-Host "Using cached $($manifestArtifact.Path)"
    } else {
        Write-Host "Downloading $($manifestArtifact.Path)"
        Download-File "$DatasetBaseUrl/$($manifestArtifact.Path)" $manifestTarget
        if (-not (Test-ExpectedHash $manifestTarget $manifestArtifact.Sha256)) {
            throw "Checksum verification failed for $($manifestArtifact.Path)"
        }
    }
    Test-ManifestArtifactPins

    foreach ($artifact in ($Artifacts | Where-Object { $_.Path -ne "manifest.json" })) {
        $target = Join-ArtifactPath $artifact.Path
        if (-not $ForceDownload -and (Test-ExpectedHash $target $artifact.Sha256)) {
            Write-Host "Using cached $($artifact.Path)"
            continue
        }

        Write-Host "Downloading $($artifact.Path)"
        Download-File "$DatasetBaseUrl/$($artifact.Path)" $target
        if (-not (Test-ExpectedHash $target $artifact.Sha256)) {
            throw "Checksum verification failed for $($artifact.Path)"
        }
    }
}

Write-Host "Validating setup artifact cache"
Test-ManifestArtifactPins
Assert-ArtifactCache

New-Item -ItemType Directory -Force (Join-Path $Root "data") | Out-Null
New-Item -ItemType Directory -Force (Join-Path $Root "AutogluonModels") | Out-Null
Copy-Item -LiteralPath (Join-ArtifactPath "processed/prediction_data.csv") -Destination (Join-Path $Root "data\prediction_data.csv") -Force
Copy-Item -LiteralPath (Join-ArtifactPath "processed/training_data.csv") -Destination (Join-Path $Root "data\training_data.csv") -Force
Copy-Item -LiteralPath (Join-ArtifactPath "processed/training_data_dec.csv") -Destination (Join-Path $Root "data\training_data_dec.csv") -Force

Ensure-StarterModel

if (-not $SkipImport) {
    Start-PostgresForImport

    if (-not $ForceImport -and (Test-DatabaseImportComplete)) {
        Write-Host "Using existing imported Postgres databases"
        Mark-DatabaseImportComplete
    } else {
        Clear-DatabaseImportMarker

        Invoke-DockerComposeOptional @("exec", "-T", "db", "createdb", "-U", "postgres", "mma-ai")
        Invoke-DockerComposeOptional @("exec", "-T", "db", "createdb", "-U", "postgres", "odds")

        Write-Host "Copying database dumps into the Postgres container"
        Invoke-DockerCompose @("cp", (Join-ArtifactPath "dumps/mma-ai.postgres-custom"), "db:/tmp/mma-ai.postgres-custom")
        Invoke-DockerCompose @("cp", (Join-ArtifactPath "dumps/odds.postgres-custom"), "db:/tmp/odds.postgres-custom")

        Write-Host "Restoring mma-ai database"
        Invoke-DockerCompose @("exec", "-T", "db", "pg_restore", "--clean", "--if-exists", "--no-owner", "--jobs", "4", "-U", "postgres", "-d", "mma-ai", "/tmp/mma-ai.postgres-custom")

        Write-Host "Restoring odds database"
        Invoke-DockerCompose @("exec", "-T", "db", "pg_restore", "--clean", "--if-exists", "--no-owner", "--jobs", "4", "-U", "postgres", "-d", "odds", "/tmp/odds.postgres-custom")

        Invoke-DockerComposeOptional @("exec", "-T", "db", "rm", "-f", "/tmp/mma-ai.postgres-custom", "/tmp/odds.postgres-custom")

        if (-not (Test-DatabaseImportComplete)) {
            throw "Database import finished but required tables were not found."
        }
        Mark-DatabaseImportComplete
    }
}

Configure-LlmAnalytics

if (-not $NoStart) {
    Write-Host "Starting MMA AI web dashboard"
    Invoke-DockerCompose @("up", "-d", "--build", "db", "web")
    $webUrl = "http://localhost:$selectedWebPort"
    Write-Host "Waiting for MMA AI web dashboard readiness check"
    Wait-ForWeb $webUrl
    Write-Host "MMA AI is ready: $webUrl"
    if (-not $NoOpen) {
        Start-Process $webUrl
    }
} else {
    Write-Host "Setup complete. Start the dashboard with: docker compose up -d --build db web"
}
