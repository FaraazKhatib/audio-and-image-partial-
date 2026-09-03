@echo off
REM One time setup for a fresh clone on Windows. Creates backend\venv, installs dependencies,
REM then fetches the YuNet face detector and the Hub checkpoints.
REM
REM Pass a CUDA tag to get a GPU build, for example:  setup.bat cu124
REM Without one you get the default PyPI wheel, which on Windows is CPU only.

setlocal
cd /d "%~dp0backend"

where python >nul 2>&1
if errorlevel 1 (
  echo Python was not found on PATH. Install Python 3.10 or newer and try again.
  exit /b 1
)

if not exist "venv\Scripts\activate.bat" (
  echo Creating virtual environment in backend\venv
  python -m venv venv
  if errorlevel 1 exit /b 1
)

call "venv\Scripts\activate.bat"

python -m pip install --upgrade pip
if errorlevel 1 exit /b 1

REM torchvision pins an exact torch version, so both come from the CUDA index in one command.
REM Installing torchvision from PyPI afterwards can silently replace a CUDA torch with a CPU one.
if not "%~1"=="" (
  echo Installing torch and torchvision from the %~1 index
  python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/%~1
  if errorlevel 1 exit /b 1
)

python -m pip install -r requirements.txt
if errorlevel 1 exit /b 1

echo.
echo Fetching the face detector and the checkpoints. This is the slow part on a cold cache.
python -m scripts.download_models

echo.
echo Checking what actually loaded.
python -m scripts.verify_models

echo.
echo Setup done. Run start.bat to serve on http://localhost:8000
endlocal
