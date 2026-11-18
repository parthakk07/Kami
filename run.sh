#!/usr/bin/env bash
# Start Paper Highlighter. Usage: ./run.sh [port]
set -euo pipefail
cd "$(dirname "$0")"
PORT="${1:-${PORT:-8765}}"
if [ -f .env ]; then set -a; . ./.env; set +a; fi
exec .venv/bin/uvicorn app:app --host 0.0.0.0 --port "$PORT"
