#!/usr/bin/env bash
# Run from anywhere:  bash run.sh   or   ./run.sh
# Stop with Ctrl+C  (not Ctrl+Z — that only suspends)
set -euo pipefail

APP="$(cd "$(dirname "$0")" && pwd)"
VENV="$APP/venv"
PY="$VENV/bin/python"
LOG_DIR="$APP/log"

# Prefer a modern interpreter; Apple CLT python3 is often 3.9.
resolve_python() {
  local c
  for c in python3.13 python3.12 python3.11 python3.10 python3; do
    if command -v "$c" >/dev/null 2>&1; then
      echo "$c"
      return 0
    fi
  done
  return 1
}

if [[ ! -x "$PY" ]]; then
  if [[ -d "$VENV" ]]; then
    echo "Removing broken venv at $VENV ..."
    rm -rf "$VENV"
  fi
  HOST_PY="$(resolve_python)" || { echo "Error: no python3 found"; exit 1; }
  echo "Creating venv at $VENV (with $HOST_PY) ..."
  "$HOST_PY" -m venv "$VENV"
fi

SITE="$(echo "$VENV"/lib/python*/site-packages)"

need_deps=0
"$PY" -c "import numpy, sounddevice, sherpa_onnx" 2>/dev/null || need_deps=1
if [[ "$need_deps" -eq 1 ]]; then
  echo "Installing Python deps..."
  "$PY" -m pip install --upgrade pip
  "$PY" -m pip install -r "$APP/requirements.txt"
fi

# pip leaves @rpath pointing at a deleted temp build dir; point it at site-packages.
fix_pywhisper_rpath() {
  local so
  so="$(echo "$SITE"/_pywhispercpp*.so)"
  [[ -f "$so" ]] || return 0
  while IFS= read -r old_rpath; do
    [[ -z "$old_rpath" ]] && continue
    [[ "$old_rpath" == "$SITE" ]] && continue
    install_name_tool -rpath "$old_rpath" "$SITE" "$so" 2>/dev/null \
      || install_name_tool -add_rpath "$SITE" "$so" 2>/dev/null \
      || true
  done < <(otool -l "$so" | awk '/LC_RPATH/{getline; getline; sub(/^ *path /,""); sub(/ \(offset.*/,""); print}')
  otool -l "$so" | grep -q "path $SITE " || install_name_tool -add_rpath "$SITE" "$so" 2>/dev/null || true
}

# pywhispercpp uses PEP604 unions (3.10+); Apple CLT is 3.9 — postpone annotation eval.
fix_pywhisper_py39() {
  local utils="$SITE/pywhispercpp/utils.py"
  [[ -f "$utils" ]] || return 0
  "$PY" -c "import sys; raise SystemExit(0 if sys.version_info >= (3, 10) else 1)" && return 0
  grep -q 'from __future__ import annotations' "$utils" 2>/dev/null && return 0
  "$PY" - "$utils" <<'PY'
import sys
from pathlib import Path
p = Path(sys.argv[1])
text = p.read_text()
if "from __future__ import annotations" in text:
    raise SystemExit(0)
lines = text.splitlines(True)
if lines and lines[0].startswith("#!"):
    lines.insert(1, "from __future__ import annotations\n")
else:
    lines.insert(0, "from __future__ import annotations\n")
p.write_text("".join(lines))
PY
}

fix_pywhisper_rpath
fix_pywhisper_py39
export DYLD_FALLBACK_LIBRARY_PATH="$SITE${DYLD_FALLBACK_LIBRARY_PATH:+:$DYLD_FALLBACK_LIBRARY_PATH}"

if ! "$PY" -c "from pywhispercpp.model import Model" 2>/dev/null; then
  echo "Building pywhispercpp with Metal (first time may take a few minutes)..."
  CMAKE_ARGS="-DGGML_METAL=ON" \
    "$PY" -m pip install "git+https://github.com/absadiki/pywhispercpp" \
    --no-binary=pywhispercpp --no-cache-dir --force-reinstall
  SITE="$(echo "$VENV"/lib/python*/site-packages)"
  fix_pywhisper_rpath
  fix_pywhisper_py39
  export DYLD_FALLBACK_LIBRARY_PATH="$SITE${DYLD_FALLBACK_LIBRARY_PATH:+:$DYLD_FALLBACK_LIBRARY_PATH}"
fi

mkdir -p "$LOG_DIR"
export TRANSCRIBE_LOG_DIR="$LOG_DIR"
cd "$APP"

if [[ "${1:-}" == "--smoke-llm" ]]; then
  exec "$PY" -m llm.smoke
fi

ELECTRON_BIN="$APP/node_modules/.bin/electron"
if [[ ! -x "$ELECTRON_BIN" ]] && command -v npm >/dev/null 2>&1 && [[ "${1:-}" != "--list-devices" ]]; then
  echo "Installing Electron overlay..."
  (cd "$APP" && npm install --no-fund --no-audit)
fi

exec "$PY" "$APP/main.py" "$@"
