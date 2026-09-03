@echo off
REM Starts the VeriTrust API and frontend. Assumes setup is already done; see README.md.

cd /d "%~dp0backend"

if not exist "venv\Scripts\activate.bat" (
  echo No virtual environment found at backend\venv.
  echo Follow the setup steps in README.md first.
  exit /b 1
)

call "venv\Scripts\activate.bat"

python -c "import fastapi" >nul 2>&1
if errorlevel 1 (
  echo Dependencies are missing. Run this inside backend, with the venv active:
  echo     pip install -r requirements.txt
  exit /b 1
)

echo Starting VeriTrust on http://localhost:8000
echo Press Ctrl+C to stop.
echo.
uvicorn veritrust.main:app --port 8000
