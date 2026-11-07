#!/bin/sh
# Start in MOCK MODE: fake songs, no YouTube or AI models used.
cd "$(dirname "$0")" || exit 1
if [ ! -x .venv/bin/python ]; then
  echo "No .venv found. Run setup first:  zsh setup.sh"; exit 1
fi
export KARAOKE_MOCK=1
export KARAOKE_DATA="$(pwd)/data-mock"
ip="$(ipconfig getifaddr en0 2>/dev/null || ipconfig getifaddr en1 2>/dev/null || hostname -I 2>/dev/null | cut -d' ' -f1)"
echo "MOCK MODE - fake songs, no YouTube or AI models used."
echo "Open http://localhost:8000 and paste any YouTube link   (phones: http://${ip:-YOUR-IP}:8000)"
exec .venv/bin/python -m uvicorn app.server:app --host 0.0.0.0 --port 8000
