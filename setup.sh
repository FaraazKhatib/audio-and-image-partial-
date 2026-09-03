#!/usr/bin/env bash
# One time setup for a fresh clone on macOS or Linux. Creates backend/venv, installs dependencies,
# then fetches the YuNet face detector and the Hub checkpoints.
#
# Pass a CUDA tag to get a GPU build, for example:  ./setup.sh cu124
# Without one you get the default PyPI wheel, which is CPU only on Windows and on macOS.

set -euo pipefail

cd "$(dirname "$0")/backend"

if ! command -v python3 >/dev/null 2>&1; then
  echo "python3 was not found on PATH. Install Python 3.10 or newer and try again."
  exit 1
fi

if [ ! -f venv/bin/activate ]; then
  echo "Creating virtual environment in backend/venv"
  python3 -m venv venv
fi

# shellcheck disable=SC1091
source venv/bin/activate

python -m pip install --upgrade pip

# torchvision pins an exact torch version, so both come from the CUDA index in one command.
# Installing torchvision from PyPI afterwards can silently replace a CUDA torch with a CPU one.
if [ "${1:-}" != "" ]; then
  echo "Installing torch and torchvision from the $1 index"
  python -m pip install torch torchvision --index-url "https://download.pytorch.org/whl/$1"
fi

python -m pip install -r requirements.txt

echo
echo "Fetching the face detector and the checkpoints. This is the slow part on a cold cache."
# Neither of the next two steps is fatal. A missing checkpoint degrades the ensemble and is
# reported at runtime, and verify_models exits nonzero when nothing loaded, which is a finding
# about the download rather than a reason to abandon a working environment.
python -m scripts.download_models || echo "[warn] Prefetch did not complete. See the messages above."

echo
echo "Checking what actually loaded."
python -m scripts.verify_models || echo "[warn] Verification reported a problem. See the messages above."

echo
echo "Setup done. Run ./start.sh to serve on http://localhost:8000"
