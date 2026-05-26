param(
    [switch]$SkipDownload,
    [switch]$SkipImport,
    [switch]$NoStart,
    [switch]$NoOpen,
    [switch]$ForceDownload,
    [switch]$SkipLlmPrompt,
    [string]$GeminiApiKey,
    [int]$PostgresPort = 0
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

function Invoke-DockerCompose {
    param([string[]]$ComposeArgs)
    & docker compose @ComposeArgs
    if ($LASTEXITCODE -ne 0) {
        throw "docker compose $($ComposeArgs -join ' ') failed"
    }
}

function Get-ComposeDbPort {
    $output = & docker compose port db 5432 2>$null
    if ($LASTEXITCODE -eq 0 -and -not [string]::IsNullOrWhiteSpace($output)) {
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

function Wait-ForPostgres {
    for ($i = 0; $i -lt 90; $i++) {
        & docker compose exec -T db pg_isready -U postgres -d postgres *> $null
        if ($LASTEXITCODE -eq 0) {
            return
        }
        Start-Sleep -Seconds 2
    }
    throw "Postgres did not become ready in time."
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
if ($selectedPostgresPort -ne 5432) {
    Write-Host "Host port 5432 is unavailable; Docker Postgres will use localhost:$selectedPostgresPort."
}

if (-not $SkipDownload) {
    foreach ($artifact in $Artifacts) {
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

New-Item -ItemType Directory -Force (Join-Path $Root "data") | Out-Null
New-Item -ItemType Directory -Force (Join-Path $Root "AutogluonModels") | Out-Null
Copy-Item -LiteralPath (Join-ArtifactPath "processed/prediction_data.csv") -Destination (Join-Path $Root "data\prediction_data.csv") -Force
Copy-Item -LiteralPath (Join-ArtifactPath "processed/training_data.csv") -Destination (Join-Path $Root "data\training_data.csv") -Force
Copy-Item -LiteralPath (Join-ArtifactPath "processed/training_data_dec.csv") -Destination (Join-Path $Root "data\training_data_dec.csv") -Force

$modelDir = Join-Path $Root "AutogluonModels\$ModelName"
if (-not (Test-Path -LiteralPath $modelDir)) {
    Write-Host "Extracting starter model $ModelName"
    & tar -xzf (Join-ArtifactPath "models/$ModelName.tar.gz") -C (Join-Path $Root "AutogluonModels")
    if ($LASTEXITCODE -ne 0) {
        throw "Model extraction failed."
    }
} else {
    Write-Host "Using existing starter model $ModelName"
}

if (-not $SkipImport) {
    Write-Host "Starting Docker Postgres"
    Invoke-DockerCompose @("up", "-d", "db")
    Wait-ForPostgres

    & docker compose exec -T db createdb -U postgres "mma-ai" 2>$null
    & docker compose exec -T db createdb -U postgres "odds" 2>$null

    Write-Host "Copying database dumps into the Postgres container"
    Invoke-DockerCompose @("cp", (Join-ArtifactPath "dumps/mma-ai.postgres-custom"), "db:/tmp/mma-ai.postgres-custom")
    Invoke-DockerCompose @("cp", (Join-ArtifactPath "dumps/odds.postgres-custom"), "db:/tmp/odds.postgres-custom")

    Write-Host "Restoring mma-ai database"
    Invoke-DockerCompose @("exec", "-T", "db", "pg_restore", "--clean", "--if-exists", "--no-owner", "--jobs", "4", "-U", "postgres", "-d", "mma-ai", "/tmp/mma-ai.postgres-custom")

    Write-Host "Restoring odds database"
    Invoke-DockerCompose @("exec", "-T", "db", "pg_restore", "--clean", "--if-exists", "--no-owner", "--jobs", "4", "-U", "postgres", "-d", "odds", "/tmp/odds.postgres-custom")

    & docker compose exec -T db rm -f /tmp/mma-ai.postgres-custom /tmp/odds.postgres-custom *> $null
}

if ($GeminiApiKey) {
    Set-EnvValue "GEMINI_API_KEY" $GeminiApiKey
} elseif (-not $SkipLlmPrompt) {
    $answer = Read-Host "Set up LLM analytics now with a Gemini/Google API key? [y/N]"
    if ($answer -match "^(y|yes)$") {
        $secureKey = Read-Host "Enter Gemini API key" -AsSecureString
        $ptr = [Runtime.InteropServices.Marshal]::SecureStringToBSTR($secureKey)
        try {
            $plainKey = [Runtime.InteropServices.Marshal]::PtrToStringBSTR($ptr)
        } finally {
            [Runtime.InteropServices.Marshal]::ZeroFreeBSTR($ptr)
        }
        if (-not [string]::IsNullOrWhiteSpace($plainKey)) {
            Set-EnvValue "GEMINI_API_KEY" $plainKey
        }
    }
}

if (-not $NoStart) {
    Write-Host "Starting MMA AI web dashboard"
    Invoke-DockerCompose @("up", "-d", "--build", "web")
    Write-Host "MMA AI is ready: http://127.0.0.1:8000"
    if (-not $NoOpen) {
        Start-Process "http://127.0.0.1:8000"
    }
} else {
    Write-Host "Setup complete. Start the dashboard with: docker compose up -d --build web"
}
