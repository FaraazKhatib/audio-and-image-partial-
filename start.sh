#!/usr/bin/env bash
# Starts the VeriTrust API and frontend. Assumes setup.sh has already run.

set -euo pipefail

cd "$(dirname "$0")/backend"

if [ ! -f venv/bin/activate ]; then
  echo "No virtual environment found at backend/venv."
  echo "Run ./setup.sh first."
  exit 1
fi

# shellcheck disable=SC1091
source venv/bin/activate

if ! python -c "import fastapi" >/dev/null 2>&1; then
  echo "Dependencies are missing. Run ./setup.sh, or inside backend with the venv active:"
  echo "    pip install -r requirements.txt"
  exit 1
fi

echo "Starting VeriTrust on http://localhost:8000"
echo "Press Ctrl+C to stop."
echo
exec uvicorn veritrust.main:app --port 8000
