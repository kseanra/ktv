#!/bin/sh
# Start the karaoke server (macOS / Linux). Run setup.sh once first.
cd "$(dirname "$0")" || exit 1
if [ ! -x .venv/bin/python ]; then
  echo "No .venv found. Run setup first:  zsh setup.sh"; exit 1
fi
# Whisper (lyrics) on the CPU, as run.bat does on Windows. A Mac has no NVIDIA
# GPU, so this is what happens anyway; set KARAOKE_DEVICE yourself to override.
export KARAOKE_DEVICE="${KARAOKE_DEVICE:-cpu}"
ip="$(ipconfig getifaddr en0 2>/dev/null || ipconfig getifaddr en1 2>/dev/null || hostname -I 2>/dev/null | cut -d' ' -f1)"
echo "Open http://localhost:8000   (phones: http://${ip:-YOUR-IP}:8000)"
exec .venv/bin/python -m uvicorn app.server:app --host 0.0.0.0 --port 8000
