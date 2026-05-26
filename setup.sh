#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

DATASET_BASE_URL="https://huggingface.co/datasets/DanMcInerney/mma-ai/resolve/main"
ARTIFACTS_ROOT="$ROOT/artifacts/mma-ai-dataset"
MODEL_NAME="ag-20260304_110750-win-extreme"

SKIP_DOWNLOAD=0
SKIP_IMPORT=0
NO_START=0
NO_OPEN=0
FORCE_DOWNLOAD=0
SKIP_LLM_PROMPT=0
GEMINI_API_KEY_VALUE=""
POSTGRES_PORT=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --skip-download) SKIP_DOWNLOAD=1 ;;
    --skip-import) SKIP_IMPORT=1 ;;
    --no-start) NO_START=1 ;;
    --no-open) NO_OPEN=1 ;;
    --force-download) FORCE_DOWNLOAD=1 ;;
    --skip-llm-prompt) SKIP_LLM_PROMPT=1 ;;
    --gemini-api-key)
      shift
      GEMINI_API_KEY_VALUE="${1:-}"
      ;;
    --postgres-port)
      shift
      POSTGRES_PORT="${1:-0}"
      ;;
    *)
      echo "Unknown argument: $1" >&2
      exit 2
      ;;
  esac
  shift
done

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

hash_file() {
  if command -v sha256sum >/dev/null 2>&1; then
    sha256sum "$1" | awk '{ print toupper($1) }'
  elif command -v shasum >/dev/null 2>&1; then
    shasum -a 256 "$1" | awk '{ print toupper($1) }'
  else
    echo ""
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

wait_for_postgres() {
  for _ in $(seq 1 90); do
    if docker compose exec -T db pg_isready -U postgres -d postgres >/dev/null 2>&1; then
      return 0
    fi
    sleep 2
  done
  echo "Postgres did not become ready in time." >&2
  exit 1
}

compose_db_port() {
  docker compose port db 5432 2>/dev/null | awk -F: 'NF { print $NF; exit }'
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

require_command docker
require_command curl
require_command tar
docker compose version >/dev/null

ensure_env_file
set_env_value "MMA_AI_COMPOSE_DATABASE_URL" "postgresql://postgres:postgres@db:5432/mma-ai"
set_env_value "MMA_AI_COMPOSE_ODDS_DATABASE_URL" "postgresql://postgres:postgres@db:5432/odds"
SELECTED_POSTGRES_PORT="$(setup_postgres_port)"
set_env_value "MMA_AI_POSTGRES_PORT" "$SELECTED_POSTGRES_PORT"
if [[ "$SELECTED_POSTGRES_PORT" != "5432" ]]; then
  echo "Host port 5432 is unavailable; Docker Postgres will use localhost:$SELECTED_POSTGRES_PORT."
fi

if [[ "$SKIP_DOWNLOAD" -eq 0 ]]; then
  for artifact in "${ARTIFACTS[@]}"; do
    relative="${artifact%%|*}"
    expected="${artifact#*|}"
    download_file "$relative" "$expected"
  done
fi

mkdir -p "$ROOT/data" "$ROOT/AutogluonModels"
cp -f "$ARTIFACTS_ROOT/processed/prediction_data.csv" "$ROOT/data/prediction_data.csv"
cp -f "$ARTIFACTS_ROOT/processed/training_data.csv" "$ROOT/data/training_data.csv"
cp -f "$ARTIFACTS_ROOT/processed/training_data_dec.csv" "$ROOT/data/training_data_dec.csv"

if [[ ! -d "$ROOT/AutogluonModels/$MODEL_NAME" ]]; then
  echo "Extracting starter model $MODEL_NAME"
  tar -xzf "$ARTIFACTS_ROOT/models/$MODEL_NAME.tar.gz" -C "$ROOT/AutogluonModels"
else
  echo "Using existing starter model $MODEL_NAME"
fi

if [[ "$SKIP_IMPORT" -eq 0 ]]; then
  echo "Starting Docker Postgres"
  docker compose up -d db
  wait_for_postgres

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
fi

if [[ -n "$GEMINI_API_KEY_VALUE" ]]; then
  set_env_value "GEMINI_API_KEY" "$GEMINI_API_KEY_VALUE"
elif [[ "$SKIP_LLM_PROMPT" -eq 0 ]]; then
  read -r -p "Set up LLM analytics now with a Gemini/Google API key? [y/N] " answer
  case "$answer" in
    y|Y|yes|YES)
      read -r -s -p "Enter Gemini API key: " api_key
      echo
      if [[ -n "$api_key" ]]; then
        set_env_value "GEMINI_API_KEY" "$api_key"
      fi
      ;;
  esac
fi

if [[ "$NO_START" -eq 0 ]]; then
  echo "Starting MMA AI web dashboard"
  docker compose up -d --build web
  echo "MMA AI is ready: http://127.0.0.1:8000"
  if [[ "$NO_OPEN" -eq 0 ]]; then
    if command -v xdg-open >/dev/null 2>&1; then
      xdg-open "http://127.0.0.1:8000" >/dev/null 2>&1 || true
    elif command -v open >/dev/null 2>&1; then
      open "http://127.0.0.1:8000" >/dev/null 2>&1 || true
    fi
  fi
else
  echo "Setup complete. Start the dashboard with: docker compose up -d --build web"
fi
