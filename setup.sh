#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

DATASET_BASE_URL="https://huggingface.co/datasets/DanMcInerney/mma-ai/resolve/main"
ARTIFACTS_ROOT="$ROOT/artifacts/mma-ai-dataset"
MODEL_NAME="ag-20260304_110750-win-extreme"

SKIP_DOWNLOAD=0
SKIP_IMPORT=0
FORCE_IMPORT=0
NO_START=0
NO_OPEN=0
FORCE_DOWNLOAD=0
SKIP_LLM_PROMPT=0
HELP=0
GEMINI_API_KEY_VALUE=""
LLM_PROVIDER_VALUE=""
LLM_MODEL_VALUE=""
LLM_API_KEY_VALUE=""
LLM_BASE_URL_VALUE=""
POSTGRES_PORT=0
WEB_PORT=0

require_option_value() {
  local option="$1"
  local value="${2:-}"
  if [[ -z "$value" || "$value" == -* ]]; then
    echo "Option $option requires a value." >&2
    exit 2
  fi
}

validate_port_value() {
  local option="$1"
  local value="$2"
  if [[ ! "$value" =~ ^[0-9]+$ ]] || (( value < 1 || value > 65535 )); then
    echo "Option $option must be a TCP port number from 1 to 65535." >&2
    exit 2
  fi
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    -h|--help) HELP=1 ;;
    --skip-download) SKIP_DOWNLOAD=1 ;;
    --skip-import) SKIP_IMPORT=1 ;;
    --force-import) FORCE_IMPORT=1 ;;
    --no-start) NO_START=1 ;;
    --no-open) NO_OPEN=1 ;;
    --force-download) FORCE_DOWNLOAD=1 ;;
    --skip-llm-prompt) SKIP_LLM_PROMPT=1 ;;
    --gemini-api-key)
      require_option_value "$1" "${2:-}"
      shift
      GEMINI_API_KEY_VALUE="${1:-}"
      ;;
    --llm-provider)
      require_option_value "$1" "${2:-}"
      shift
      LLM_PROVIDER_VALUE="${1:-}"
      ;;
    --llm-model)
      require_option_value "$1" "${2:-}"
      shift
      LLM_MODEL_VALUE="${1:-}"
      ;;
    --llm-api-key)
      require_option_value "$1" "${2:-}"
      shift
      LLM_API_KEY_VALUE="${1:-}"
      ;;
    --llm-base-url)
      require_option_value "$1" "${2:-}"
      shift
      LLM_BASE_URL_VALUE="${1:-}"
      ;;
    --postgres-port)
      require_option_value "$1" "${2:-}"
      validate_port_value "$1" "$2"
      shift
      POSTGRES_PORT="${1:-0}"
      ;;
    --web-port)
      require_option_value "$1" "${2:-}"
      validate_port_value "$1" "$2"
      shift
      WEB_PORT="${1:-0}"
      ;;
    *)
      echo "Unknown argument: $1" >&2
      exit 2
      ;;
  esac
  shift
done

show_usage() {
  cat <<'EOF'
MMA AI setup

Usage:
  ./setup.sh [options]

First-time setup downloads verified Hugging Face artifacts, restores the main
and odds databases into Docker Postgres, extracts the starter model, optionally
configures LLM analytics, starts the dashboard, and waits for /api/readiness.

Options:
  -h, --help              Show this help and exit before Docker or downloads.
  --skip-download         Reuse the existing artifact cache after validating it.
  --force-download        Re-download artifacts and verify checksums.
  --skip-import           Do not restore database dumps into Docker Postgres.
  --force-import          Restore database dumps even if required tables exist.
  --no-start              Prepare files/imports but do not start the dashboard.
  --no-open               Start the dashboard but do not open a browser.
  --skip-llm-prompt       Do not prompt for analytics LLM configuration.
  --postgres-port <port>  Force the Docker Postgres host port.
  --web-port <port>       Force the dashboard host port.
  --llm-provider <name>   Configure analytics LLM provider non-interactively.
  --llm-model <name>      Configure analytics LLM model.
  --llm-api-key <token>   Configure analytics LLM API key or token.
  --llm-base-url <url>    Configure custom/OpenAI-compatible API base URL.

Examples:
  ./setup.sh
  ./setup.sh --force-import
  ./setup.sh --postgres-port 55432 --web-port 18000
  ./setup.sh --skip-llm-prompt
EOF
}

if [[ "$HELP" -eq 1 ]]; then
  show_usage
  exit 0
fi

ARTIFACTS=(
  "manifest.json|"
  "dumps/mma-ai.postgres-custom|0EB0D2CBDECC55B6EA625F70A12914F72BD0FDCF67B91BCDFC0146393E1A7B7A"
  "dumps/odds.postgres-custom|767AFB5C2642DD8D450B6F043F333CD5FE8589B4D8574E41831E8BBC2614F352"
  "processed/training_data.csv|FFBF161D6F6E307132EB8150B5978728DED93AA9B4D3282F892C725503BA654E"
  "processed/training_data_dec.csv|91D6918DFCE10C5C5C788721C58FB98AB42AC51D9FB854BA935E6CB54701EFFB"
  "processed/prediction_data.csv|1C28D3B04DA412980777D38032E95A5B695C4B53BEA0014192D4D6C07413F754"
  "models/ag-20260304_110750-win-extreme.tar.gz|248511976D55895BE2C167F2F8FA8C4013E635B39A9BAB0D5F28C0916B5AAD74"
)

require_command() {
  if ! command -v "$1" >/dev/null 2>&1; then
    echo "Required command '$1' was not found. Install it and rerun setup." >&2
    exit 1
  fi
}

require_any_command() {
  local label="$1"
  shift
  local command_name
  for command_name in "$@"; do
    if command -v "$command_name" >/dev/null 2>&1; then
      return 0
    fi
  done
  echo "Required command '$label' was not found. Install one and rerun setup." >&2
  exit 1
}

hash_file() {
  if command -v sha256sum >/dev/null 2>&1; then
    sha256sum "$1" | awk '{ print toupper($1) }'
  elif command -v shasum >/dev/null 2>&1; then
    shasum -a 256 "$1" | awk '{ print toupper($1) }'
  else
    echo "Required command 'sha256sum or shasum' was not found. Install one and rerun setup." >&2
    exit 1
  fi
}

hash_matches() {
  local path="$1"
  local expected="$2"
  [[ -z "$expected" && -f "$path" ]] && return 0
  [[ -f "$path" ]] || return 1
  local actual
  actual="$(hash_file "$path")"
  [[ -n "$actual" && "$actual" == "$expected" ]]
}

assert_artifact_cache() {
  local missing=()
  local artifact relative expected target
  for artifact in "${ARTIFACTS[@]}"; do
    relative="${artifact%%|*}"
    expected="${artifact#*|}"
    target="$ARTIFACTS_ROOT/$relative"
    if ! hash_matches "$target" "$expected"; then
      missing+=("$relative")
    fi
  done

  if [[ "${#missing[@]}" -gt 0 ]]; then
    printf 'Required setup artifact cache is incomplete or corrupt: %s. ' "${missing[*]}" >&2
    if [[ "$SKIP_DOWNLOAD" -eq 1 ]]; then
      echo "Rerun setup without --skip-download, or pass --force-download to refresh the cache." >&2
    else
      echo "Rerun setup with --force-download to refresh the cache." >&2
    fi
    exit 1
  fi
}

validate_manifest_artifact_pins() {
  local manifest="$ARTIFACTS_ROOT/manifest.json"
  [[ -f "$manifest" ]] || {
    echo "Hugging Face manifest is missing from the setup artifact cache." >&2
    exit 1
  }

  local pins=()
  local artifact relative expected
  for artifact in "${ARTIFACTS[@]}"; do
    relative="${artifact%%|*}"
    expected="${artifact#*|}"
    [[ "$relative" == "manifest.json" || -z "$expected" ]] && continue
    pins+=("$relative=$expected")
  done

  bash "$ROOT/scripts/verify_hf_manifest.sh" "$manifest" "${pins[@]}"
}

safe_remove_setup_dir() {
  local target="$1"
  local parent="$2"
  [[ -e "$target" ]] || return 0

  local parent_abs
  local target_abs
  parent_abs="$(cd "$parent" && pwd -P)"
  target_abs="$(cd "$(dirname "$target")" && pwd -P)/$(basename "$target")"

  case "$target_abs" in
    "$parent_abs"/*) rm -rf "$target_abs" ;;
    *)
      echo "Refusing to remove setup directory outside $parent_abs: $target_abs" >&2
      exit 1
      ;;
  esac
}

download_file() {
  local relative="$1"
  local expected="$2"
  local target="$ARTIFACTS_ROOT/$relative"
  local tmp="$target.download"
  mkdir -p "$(dirname "$target")"

  if [[ "$FORCE_DOWNLOAD" -eq 0 ]] && hash_matches "$target" "$expected"; then
    echo "Using cached $relative"
    return
  fi

  echo "Downloading $relative"
  rm -f "$tmp"
  curl -L --fail --retry 3 --output "$tmp" "$DATASET_BASE_URL/$relative"
  if [[ -n "$expected" ]] && ! hash_matches "$tmp" "$expected"; then
    rm -f "$tmp"
    echo "Checksum verification failed for $relative" >&2
    exit 1
  fi
  mv "$tmp" "$target"
}

ensure_env_file() {
  if [[ ! -f "$ROOT/.env" ]]; then
    cp "$ROOT/.env.example" "$ROOT/.env"
  fi
}

set_env_value() {
  local key="$1"
  local value="$2"
  local tmp
  ensure_env_file
  tmp="$(mktemp)"
  if grep -Eq "^[[:space:]]*#?[[:space:]]*$key=" "$ROOT/.env"; then
    awk -v key="$key" -v value="$value" '
      $0 ~ "^[[:space:]]*#?[[:space:]]*" key "=" { print key "=" value; next }
      { print }
    ' "$ROOT/.env" > "$tmp"
  else
    cat "$ROOT/.env" > "$tmp"
    printf "\n%s=%s\n" "$key" "$value" >> "$tmp"
  fi
  mv "$tmp" "$ROOT/.env"
}

normalize_llm_provider() {
  local provider
  provider="$(printf '%s' "${1:-}" | tr '[:upper:]' '[:lower:]')"
  case "$provider" in
    gemini|google) echo "google" ;;
    openai) echo "openai" ;;
    codex) echo "codex" ;;
    anthropic|claude) echo "anthropic" ;;
    grok|xai) echo "grok" ;;
    openrouter|open-router) echo "openrouter" ;;
    deepseek|deep-seek) echo "deepseek" ;;
    mistral) echo "mistral" ;;
    together|togetherai|together-ai) echo "together" ;;
    perplexity|sonar) echo "perplexity" ;;
    local|ollama|lmstudio|lm-studio) echo "local" ;;
    custom|openai-compatible) echo "custom" ;;
    *) echo "$provider" ;;
  esac
}

llm_default_model() {
  case "$1" in
    google) echo "gemini-1.5-pro" ;;
    openai) echo "gpt-4o-mini" ;;
    codex) echo "gpt-5-codex" ;;
    anthropic) echo "claude-3-5-sonnet-latest" ;;
    grok) echo "grok-2-latest" ;;
    openrouter) echo "~openai/gpt-latest" ;;
    deepseek) echo "deepseek-chat" ;;
    mistral) echo "mistral-large-latest" ;;
    together) echo "meta-llama/Llama-3.3-70B-Instruct-Turbo" ;;
    perplexity) echo "sonar-pro" ;;
    local) echo "llama3.1" ;;
    *) echo "gpt-4o-mini" ;;
  esac
}

llm_default_base_url() {
  case "$1" in
    openai|codex) echo "https://api.openai.com/v1" ;;
    grok) echo "https://api.x.ai/v1" ;;
    openrouter) echo "https://openrouter.ai/api/v1" ;;
    deepseek) echo "https://api.deepseek.com" ;;
    mistral) echo "https://api.mistral.ai/v1" ;;
    together) echo "https://api.together.ai/v1" ;;
    perplexity) echo "https://api.perplexity.ai" ;;
    local|custom) echo "http://host.docker.internal:11434/v1" ;;
    *) echo "" ;;
  esac
}

set_llm_config() {
  local provider
  local model
  local api_key
  local base_url
  provider="$(normalize_llm_provider "${1:-}")"
  model="${2:-}"
  api_key="${3:-}"
  base_url="${4:-}"

  if [[ -z "$provider" ]]; then
    echo "LLM provider is required when configuring analytics." >&2
    exit 1
  fi

  [[ -n "$model" ]] || model="$(llm_default_model "$provider")"
  [[ -n "$base_url" ]] || base_url="$(llm_default_base_url "$provider")"

  set_env_value "LLM_PROVIDER" "$provider"
  set_env_value "LLM_MODEL" "$model"
  set_env_value "LLM_BASE_URL" "$base_url"

  if [[ -n "$api_key" ]]; then
    set_env_value "LLM_API_KEY" "$api_key"
    case "$provider" in
      google)
        set_env_value "GEMINI_API_KEY" "$api_key"
        set_env_value "GOOGLE_API_KEY" "$api_key"
        ;;
      openai|codex)
        set_env_value "OPENAI_API_KEY" "$api_key"
        ;;
      anthropic)
        set_env_value "ANTHROPIC_API_KEY" "$api_key"
        ;;
      grok)
        set_env_value "XAI_API_KEY" "$api_key"
        set_env_value "GROK_API_KEY" "$api_key"
        ;;
      openrouter)
        set_env_value "OPENROUTER_API_KEY" "$api_key"
        ;;
      deepseek)
        set_env_value "DEEPSEEK_API_KEY" "$api_key"
        ;;
      mistral)
        set_env_value "MISTRAL_API_KEY" "$api_key"
        ;;
      together)
        set_env_value "TOGETHER_API_KEY" "$api_key"
        ;;
      perplexity)
        set_env_value "PERPLEXITY_API_KEY" "$api_key"
        ;;
    esac
  elif [[ "$provider" == "local" || "$provider" == "custom" ]]; then
    set_env_value "LLM_API_KEY" ""
  fi

  echo "Configured LLM analytics: provider=$provider model=$model"
}

resolve_llm_provider_choice() {
  local choice
  choice="$(printf '%s' "${1:-}" | tr '[:upper:]' '[:lower:]')"
  case "$choice" in
    ""|1) echo "openai" ;;
    2) echo "codex" ;;
    3) echo "anthropic" ;;
    4) echo "google" ;;
    5) echo "grok" ;;
    6) echo "openrouter" ;;
    7) echo "deepseek" ;;
    8) echo "mistral" ;;
    9) echo "together" ;;
    10) echo "perplexity" ;;
    11) echo "local" ;;
    12) echo "custom" ;;
    *) normalize_llm_provider "$choice" ;;
  esac
}

configure_llm_analytics() {
  if [[ -n "$GEMINI_API_KEY_VALUE" ]]; then
    set_llm_config "google" "$LLM_MODEL_VALUE" "$GEMINI_API_KEY_VALUE" "$LLM_BASE_URL_VALUE"
    return
  fi

  if [[ -n "$LLM_PROVIDER_VALUE" || -n "$LLM_MODEL_VALUE" || -n "$LLM_API_KEY_VALUE" || -n "$LLM_BASE_URL_VALUE" ]]; then
    local provider="${LLM_PROVIDER_VALUE:-custom}"
    set_llm_config "$provider" "$LLM_MODEL_VALUE" "$LLM_API_KEY_VALUE" "$LLM_BASE_URL_VALUE"
    return
  fi

  [[ "$SKIP_LLM_PROMPT" -eq 0 ]] || return

  local answer
  read -r -p "Set up LLM analytics now? [y/N] " answer
  case "$answer" in
    y|Y|yes|YES) ;;
    *) return ;;
  esac

  echo "Choose your LLM provider:"
  echo "  1) OpenAI"
  echo "  2) Codex / OpenAI-compatible"
  echo "  3) Anthropic Claude"
  echo "  4) Google Gemini"
  echo "  5) xAI Grok"
  echo "  6) OpenRouter"
  echo "  7) DeepSeek"
  echo "  8) Mistral"
  echo "  9) Together AI"
  echo "  10) Perplexity Sonar"
  echo "  11) Local model (Ollama or LM Studio)"
  echo "  12) Custom OpenAI-compatible endpoint"

  local choice provider default_model model base_url override api_key
  read -r -p "Provider [1] " choice
  provider="$(resolve_llm_provider_choice "$choice")"
  default_model="$(llm_default_model "$provider")"
  read -r -p "Model name [$default_model] " model
  model="${model:-$default_model}"

  base_url="$(llm_default_base_url "$provider")"
  case "$provider" in
    openai|codex|grok|openrouter|deepseek|mistral|together|perplexity)
      read -r -p "API base URL [$base_url] " override
      base_url="${override:-$base_url}"
      ;;
    local|custom)
      read -r -p "OpenAI-compatible base URL [$base_url] " override
      base_url="${override:-$base_url}"
      ;;
  esac

  api_key=""
  if [[ "$provider" == "local" ]]; then
    read -r -s -p "API key/token if required by your local server; otherwise press Enter: " api_key
    echo
  elif [[ "$provider" == "custom" ]]; then
    read -r -s -p "API key/token if required by your endpoint; otherwise press Enter: " api_key
    echo
  else
    read -r -s -p "API key/token: " api_key
    echo
    if [[ -z "$api_key" ]]; then
      echo "No API key entered; skipping LLM analytics configuration."
      return
    fi
  fi

  set_llm_config "$provider" "$model" "$api_key" "$base_url"
}

wait_for_postgres() {
  for _ in $(seq 1 90); do
    if docker compose exec -T db pg_isready -U postgres -d postgres >/dev/null 2>&1; then
      return 0
    fi
    sleep 2
  done
  echo "Postgres did not become ready in time." >&2
  return 1
}

readiness_recovery_hint() {
  printf '%s' "Review Docker logs with: docker compose logs --tail 120 web db. If readiness reports missing database tables, rerun setup with --force-import. If it reports missing CSVs or model files, rerun setup without --skip-download or with --force-download."
}

database_import_marker() {
  printf '%s/.db-import-complete' "$ARTIFACTS_ROOT"
}

database_table_exists() {
  local database="$1"
  local qualified_table="$2"
  local result
  result="$(docker compose exec -T db psql -U postgres -d "$database" -tAc "SELECT to_regclass('$qualified_table') IS NOT NULL;" 2>/dev/null | tr -d '[:space:]')" || return 1
  [[ "$result" == "t" ]]
}

database_import_complete() {
  database_table_exists "mma-ai" "features.fight_mapping" \
    && database_table_exists "odds" "bestfightodds.bfo"
}

mark_database_import_complete() {
  mkdir -p "$ARTIFACTS_ROOT"
  touch "$(database_import_marker)"
}

clear_database_import_marker() {
  rm -f "$(database_import_marker)"
}

readiness_response() {
  local web_url="$1"
  local response status body
  if ! response="$(curl -sS --max-time 30 -w $'\n%{http_code}' "$web_url/api/readiness" 2>&1)"; then
    printf '%s' "$response"
    return 1
  fi
  status="${response##*$'\n'}"
  body="${response%$'\n'$status}"
  printf '%s' "$body"
  [[ "$status" =~ ^2 ]]
}

web_ready() {
  local web_url="$1"
  readiness_response "$web_url" >/dev/null
}

wait_for_web() {
  local web_url="$1"
  local last_detail="No readiness detail returned."
  for _ in $(seq 1 90); do
    if last_detail="$(readiness_response "$web_url")"; then
      return 0
    fi
    [[ -n "$last_detail" ]] || last_detail="No readiness detail returned."
    sleep 2
  done
  echo "Web dashboard did not become ready at $web_url in time. Last readiness response: $last_detail. $(readiness_recovery_hint)" >&2
  return 1
}

starter_model_complete() {
  local model_dir="$1"
  [[ -d "$model_dir" ]] || return 1
  [[ -f "$model_dir/feats.txt" ]] || return 1

  [[ -f "$model_dir/predictor.pkl" ]] && return 0
  [[ -f "$model_dir/ensemble_info.txt" ]] || return 1
  [[ -d "$model_dir/final_model" ]] && return 0

  local window_dir
  for window_dir in "$model_dir"/window_*; do
    [[ -d "$window_dir" ]] && return 0
  done
  return 1
}

compose_db_port() {
  docker compose port db 5432 2>/dev/null | awk -F: 'NF { print $NF; exit }'
}

compose_web_port() {
  docker compose port web 8000 2>/dev/null | awk -F: 'NF { print $NF; exit }'
}

port_available() {
  local port="$1"
  if docker_published_port_in_use "$port"; then
    return 1
  fi

  if command -v nc >/dev/null 2>&1; then
    ! nc -z 127.0.0.1 "$port" >/dev/null 2>&1
  else
    ! (echo >"/dev/tcp/127.0.0.1/$port") >/dev/null 2>&1
  fi
}

docker_published_port_in_use() {
  local port="$1"
  docker ps --format '{{.Ports}}' 2>/dev/null \
    | tr ',' '\n' \
    | grep -Eq "(^|[^0-9])${port}->"
}

setup_postgres_port() {
  if [[ "$POSTGRES_PORT" != "0" ]]; then
    echo "$POSTGRES_PORT"
    return
  fi

  local existing
  existing="$(compose_db_port || true)"
  if [[ -n "$existing" ]]; then
    echo "$existing"
    return
  fi

  if port_available 5432; then
    echo "5432"
    return
  fi

  for candidate in $(seq 55432 55532); do
    if port_available "$candidate"; then
      echo "$candidate"
      return
    fi
  done

  echo "Could not find an available host port for PostgreSQL. Pass --postgres-port <port> to choose one." >&2
  exit 1
}

setup_web_port() {
  if [[ "$WEB_PORT" != "0" ]]; then
    echo "$WEB_PORT"
    return
  fi

  local existing
  existing="$(compose_web_port || true)"
  if [[ -n "$existing" ]]; then
    echo "$existing"
    return
  fi

  if port_available 8000; then
    echo "8000"
    return
  fi

  for candidate in $(seq 18000 18100); do
    if port_available "$candidate"; then
      echo "$candidate"
      return
    fi
  done

  echo "Could not find an available host port for the web dashboard. Pass --web-port <port> to choose one." >&2
  exit 1
}

start_postgres_for_import() {
  echo "Starting Docker Postgres"
  if docker compose up -d db && wait_for_postgres; then
    return 0
  fi

  echo "Postgres did not start cleanly; recreating the setup database volume and retrying."
  docker compose down --volumes --remove-orphans
  docker compose up -d db
  wait_for_postgres
}

ensure_starter_model() {
  local models_root="$ROOT/AutogluonModels"
  local model_dir="$models_root/$MODEL_NAME"
  local marker_path="$models_root/.$MODEL_NAME.setup-complete"
  local extract_dir="$models_root/.$MODEL_NAME.extracting"

  mkdir -p "$models_root"

  if starter_model_complete "$model_dir" && [[ -f "$marker_path" ]]; then
    echo "Using existing starter model $MODEL_NAME"
    return
  fi

  if [[ -d "$model_dir" ]]; then
    echo "Starter model is missing required files; re-extracting $MODEL_NAME"
    safe_remove_setup_dir "$model_dir" "$models_root"
  else
    echo "Extracting starter model $MODEL_NAME"
  fi
  rm -f "$marker_path"

  safe_remove_setup_dir "$extract_dir" "$models_root"
  mkdir -p "$extract_dir"

  if ! tar -xzf "$ARTIFACTS_ROOT/models/$MODEL_NAME.tar.gz" -C "$extract_dir"; then
    safe_remove_setup_dir "$extract_dir" "$models_root"
    echo "Model extraction failed." >&2
    exit 1
  fi

  if [[ -d "$extract_dir/$MODEL_NAME" ]]; then
    mv "$extract_dir/$MODEL_NAME" "$model_dir"
  else
    mkdir -p "$model_dir"
    shopt -s dotglob nullglob
    mv "$extract_dir"/* "$model_dir"
    shopt -u dotglob nullglob
  fi

  if ! starter_model_complete "$model_dir"; then
    safe_remove_setup_dir "$model_dir" "$models_root"
    safe_remove_setup_dir "$extract_dir" "$models_root"
    echo "Starter model extraction did not create a usable model directory." >&2
    exit 1
  fi

  touch "$marker_path"
  safe_remove_setup_dir "$extract_dir" "$models_root"
}

require_command docker
require_command curl
require_command tar
require_command awk
require_command grep
require_command mktemp
require_any_command "sha256sum or shasum" sha256sum shasum
docker compose version >/dev/null

ensure_env_file
set_env_value "MMA_AI_COMPOSE_DATABASE_URL" "postgresql://postgres:postgres@db:5432/mma-ai"
set_env_value "MMA_AI_COMPOSE_ODDS_DATABASE_URL" "postgresql://postgres:postgres@db:5432/odds"
SELECTED_POSTGRES_PORT="$(setup_postgres_port)"
set_env_value "MMA_AI_POSTGRES_PORT" "$SELECTED_POSTGRES_PORT"
set_env_value "DATABASE_URL" "postgresql://postgres:postgres@localhost:$SELECTED_POSTGRES_PORT/mma-ai"
set_env_value "ODDS_DATABASE_URL" "postgresql://postgres:postgres@localhost:$SELECTED_POSTGRES_PORT/odds"
if [[ "$SELECTED_POSTGRES_PORT" != "5432" ]]; then
  echo "Host port 5432 is unavailable; Docker Postgres will use localhost:$SELECTED_POSTGRES_PORT."
fi
SELECTED_WEB_PORT="$(setup_web_port)"
set_env_value "MMA_AI_WEB_PORT" "$SELECTED_WEB_PORT"
if [[ "$SELECTED_WEB_PORT" != "8000" ]]; then
  echo "Host port 8000 is unavailable; the dashboard will use http://localhost:$SELECTED_WEB_PORT."
fi

if [[ "$SKIP_DOWNLOAD" -eq 0 ]]; then
  download_file "manifest.json" ""
  validate_manifest_artifact_pins
  for artifact in "${ARTIFACTS[@]}"; do
    relative="${artifact%%|*}"
    expected="${artifact#*|}"
    [[ "$relative" == "manifest.json" ]] && continue
    download_file "$relative" "$expected"
  done
fi

echo "Validating setup artifact cache"
validate_manifest_artifact_pins
assert_artifact_cache

mkdir -p "$ROOT/data" "$ROOT/AutogluonModels"
cp -f "$ARTIFACTS_ROOT/processed/prediction_data.csv" "$ROOT/data/prediction_data.csv"
cp -f "$ARTIFACTS_ROOT/processed/training_data.csv" "$ROOT/data/training_data.csv"
cp -f "$ARTIFACTS_ROOT/processed/training_data_dec.csv" "$ROOT/data/training_data_dec.csv"

ensure_starter_model

if [[ "$SKIP_IMPORT" -eq 0 ]]; then
  start_postgres_for_import

  if [[ "$FORCE_IMPORT" -eq 0 ]] && database_import_complete; then
    echo "Using existing imported Postgres databases"
    mark_database_import_complete
  else
    clear_database_import_marker

    docker compose exec -T db createdb -U postgres "mma-ai" >/dev/null 2>&1 || true
    docker compose exec -T db createdb -U postgres "odds" >/dev/null 2>&1 || true

    echo "Copying database dumps into the Postgres container"
    docker compose cp "$ARTIFACTS_ROOT/dumps/mma-ai.postgres-custom" "db:/tmp/mma-ai.postgres-custom"
    docker compose cp "$ARTIFACTS_ROOT/dumps/odds.postgres-custom" "db:/tmp/odds.postgres-custom"

    echo "Restoring mma-ai database"
    docker compose exec -T db pg_restore --clean --if-exists --no-owner --jobs 4 -U postgres -d "mma-ai" /tmp/mma-ai.postgres-custom

    echo "Restoring odds database"
    docker compose exec -T db pg_restore --clean --if-exists --no-owner --jobs 4 -U postgres -d "odds" /tmp/odds.postgres-custom

    docker compose exec -T db rm -f /tmp/mma-ai.postgres-custom /tmp/odds.postgres-custom >/dev/null 2>&1 || true

    if ! database_import_complete; then
      echo "Database import finished but required tables were not found." >&2
      exit 1
    fi
    mark_database_import_complete
  fi
fi

configure_llm_analytics

if [[ "$NO_START" -eq 0 ]]; then
  echo "Starting MMA AI web dashboard"
  docker compose up -d --build db web
  WEB_URL="http://localhost:$SELECTED_WEB_PORT"
  echo "Waiting for MMA AI web dashboard readiness check"
  wait_for_web "$WEB_URL"
  echo "MMA AI is ready: $WEB_URL"
  if [[ "$NO_OPEN" -eq 0 ]]; then
    if command -v xdg-open >/dev/null 2>&1; then
      xdg-open "$WEB_URL" >/dev/null 2>&1 || true
    elif command -v open >/dev/null 2>&1; then
      open "$WEB_URL" >/dev/null 2>&1 || true
    fi
  fi
else
  echo "Setup complete. Start the dashboard with: docker compose up -d --build db web"
fi
