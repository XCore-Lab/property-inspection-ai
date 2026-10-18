#!/usr/bin/env bash
# Linux / macOS / Cloud Shell: one-time setup + start on http://127.0.0.1:8080 (settings from .env)
set -e
cd "$(dirname "$0")"
if [ ! -x .venv/bin/python ]; then
  python3.11 -m venv .venv
  .venv/bin/pip install --upgrade pip
  .venv/bin/pip install --extra-index-url https://download.pytorch.org/whl/cu124 \
    "torch==2.5.1+cu124" "torchvision==0.20.1+cu124" -r requirements-dev.txt
  [ -f .env ] || cp .env.example .env
fi
.venv/bin/python -c "import hypercorn" 2>/dev/null || .venv/bin/pip install -q "hypercorn==0.18.0"
exec .venv/bin/python -m hypercorn backend.main:app --bind "127.0.0.1:${PORT:-8080}" --keep-alive 75
