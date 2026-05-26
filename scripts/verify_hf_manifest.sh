#!/usr/bin/env bash
set -euo pipefail

if ! command -v awk >/dev/null 2>&1; then
  echo "Required command 'awk' was not found. Install it and rerun setup." >&2
  exit 1
fi

if [[ $# -lt 2 ]]; then
  echo "Usage: verify_hf_manifest.sh <manifest.json> <relative-path=SHA256>..." >&2
  exit 2
fi

MANIFEST="$1"
shift

if [[ ! -f "$MANIFEST" ]]; then
  echo "Hugging Face manifest is missing from the setup artifact cache." >&2
  exit 1
fi

manifest_sha_for_path() {
  local manifest="$1"
  local wanted_path="$2"
  awk -v wanted="$wanted_path" '
    function json_string_value(line, value) {
      value = line
      sub(/^[^:]*:[[:space:]]*"/, "", value)
      sub(/",[[:space:]]*$/, "", value)
      sub(/"[[:space:]]*$/, "", value)
      return value
    }

    /"path"[[:space:]]*:/ {
      current_path = json_string_value($0)
      current_sha = ""
    }

    current_path != "" && /"sha256"[[:space:]]*:/ {
      current_sha = json_string_value($0)
    }

    /}/ {
      if (current_path == wanted && current_sha != "") {
        print current_sha
        found = 1
        exit
      }
      current_path = ""
      current_sha = ""
    }

    END {
      if (!found) {
        exit 1
      }
    }
  ' "$manifest"
}

for pin in "$@"; do
  relative="${pin%%=*}"
  expected="${pin#*=}"
  if [[ -z "$relative" || -z "$expected" || "$relative" == "$pin" ]]; then
    echo "Invalid manifest pin argument: $pin" >&2
    exit 2
  fi

  actual="$(manifest_sha_for_path "$MANIFEST" "$relative" || true)"
  if [[ "$actual" != "$expected" ]]; then
    echo "Hugging Face manifest entry for $relative does not match the setup pin. Update setup artifact checksums before downloading large artifacts." >&2
    exit 1
  fi
done
