@echo off
rem One-time local setup on Windows (Python 3.11 + NVIDIA GPU optional). Creates .venv with everything.
cd /d "%~dp0"
if not exist .venv ( py -3.11 -m venv .venv || python -m venv .venv )
.venv\Scripts\python.exe -m pip install --upgrade pip
.venv\Scripts\python.exe -m pip install --extra-index-url https://download.pytorch.org/whl/cu124 "torch==2.5.1+cu124" "torchvision==0.20.1+cu124" -r requirements-dev.txt
if not exist .env copy .env.example .env
echo.
echo Done. Put your Gemini key in .env (PI_GEMINI_API_KEY=...) and run run_local.bat
