#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python3}"
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8000}"
RELOAD="${RELOAD:-1}"

CMD=(
  "${PYTHON_BIN}" -m uvicorn src.api.app:app
  --host "${HOST}"
  --port "${PORT}"
)

if [[ "${RELOAD}" == "1" ]]; then
  CMD+=(--reload)
fi

exec "${CMD[@]}"
