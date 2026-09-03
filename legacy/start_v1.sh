#!/bin/bash
# setup_backend.sh  –  run once to prepare the backend

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  VeriTrust Backend Setup"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# 1. Virtual environment
if [ ! -d "$SCRIPT_DIR/venv" ]; then
  echo "[1/4] Creating virtual environment..."
  python3 -m venv "$SCRIPT_DIR/venv"
fi
source "$SCRIPT_DIR/venv/bin/activate"

# 2. Install dependencies
echo "[2/4] Installing Python packages..."
pip install --upgrade pip -q
pip install -r "$SCRIPT_DIR/requirements.txt" -q

# 3. NLTK data
echo "[3/4] Downloading NLTK data..."
python3 -c "
import nltk
for pkg in ['stopwords','wordnet','punkt_tab']:
    try:
        nltk.download(pkg, quiet=True)
    except:
        pass
"

# 4. Check models
echo "[4/4] Verifying model files..."
for f in fake_news_model.pkl tfidf_vectorizer.pkl efficientnetv2b2_deepfake_final.hdf5; do
  if [ ! -f "$SCRIPT_DIR/models/$f" ]; then
    echo "  ⚠  Missing: models/$f"
    echo "     Copy it into backend/models/ and re-run."
    exit 1
  fi
  echo "  ✅ $f"
done

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  Setup complete! Starting server..."
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
python3 "$SCRIPT_DIR/app.py"