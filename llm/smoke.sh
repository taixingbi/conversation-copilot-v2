#!/usr/bin/env bash
# Smoke-test Bedrock Function URL + Q/A reconstruction.
#   bash transcribe/llm/smoke.sh
#   bash run.sh --smoke-llm
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PY="$ROOT/venv/bin/python"
if [[ ! -x "$PY" ]]; then
  echo "error: missing $PY — run bash run.sh once to create the venv" >&2
  exit 1
fi
cd "$ROOT"
exec "$PY" -m llm.smoke
