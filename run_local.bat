@echo off
rem Start the app locally: http://127.0.0.1:8080  (settings come from .env)
cd /d "%~dp0"
if not exist .venv\Scripts\python.exe ( echo Run setup_local.bat first & exit /b 1 )
rem Local models + GPU programs: models\ and bin\ in this folder (settings defaults).
rem Depth Anything 3 weights are bundled in models\hf (no download needed).
if exist models\hf\hub if not defined HF_HOME set "HF_HOME=%~dp0models\hf"
set PYTHONUNBUFFERED=1
rem Same server as the container (hypercorn: HTTP/1.1 + HTTP/2 cleartext)
.venv\Scripts\python.exe -c "import hypercorn" 2>nul || .venv\Scripts\python.exe -m pip install -q "hypercorn==0.18.0"
.venv\Scripts\python.exe -m hypercorn backend.main:app --bind 127.0.0.1:8080 --keep-alive 75
