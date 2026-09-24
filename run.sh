#!/usr/bin/env sh
# Start the web app on http://localhost:8000
set -e
cd "$(dirname "$0")"
exec uvicorn floorplan.server:app --host "${HOST:-0.0.0.0}" --port "${PORT:-8000}"
