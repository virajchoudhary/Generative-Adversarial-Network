#!/usr/bin/env bash
set -euo pipefail

PORT="${PORT:-10000}"

for python_bin in ".venv/bin/python" "../.venv/bin/python" "/opt/render/project/src/.venv/bin/python"; do
  if [ -x "$python_bin" ]; then
    exec "$python_bin" -m uvicorn main:app --host 0.0.0.0 --port "$PORT"
  fi
done

exec uvicorn main:app --host 0.0.0.0 --port "$PORT"
