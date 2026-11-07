#!/usr/bin/env zsh
# Karaoke prototype: one-time setup for macOS (zsh).
#   cd karaoke-prototype && zsh setup.sh
# Creates .venv and installs everything. Safe to run again.

# Re-run under zsh if started with sh/bash.
if [ -z "$ZSH_VERSION" ]; then exec zsh "$0" "$@"; fi

set -euo pipefail
cd "${0:A:h}"

bold() { print -P "%B$*%b"; }
info() { print -P "%F{cyan}==>%f $*"; }
warn() { print -P "%F{yellow}warning:%f $*"; }
fail() { print -P "%F{red}error:%f $*"; exit 1; }

[[ "$(uname -s)" == "Darwin" ]] || warn "this script is written for macOS; on Linux use run.sh after a manual venv setup."

# ---------------------------------------------------------------- Python
# Apple Silicon: 3.12 / 3.13 (3.11 also works). Intel Macs: 3.11 / 3.12 only,
# because PyTorch stopped publishing Intel-Mac builds after Python 3.12.
find_python() {
  local v c dir
  for v in 3.12 3.13 3.11; do
    for c in "python$v" /opt/homebrew/bin/python$v /usr/local/bin/python$v \
             /Library/Frameworks/Python.framework/Versions/$v/bin/python3; do
      if command -v "$c" >/dev/null 2>&1; then
        # On an Intel Mac, skip 3.13 (no PyTorch build for it).
        if [[ "$v" == 3.13 && "$("$c" -c 'import platform; print(platform.machine())')" == x86_64 ]]; then
          continue
        fi
        command -v "$c"; return 0
      fi
    done
  done
  return 1
}

PY="$(find_python || true)"
if [[ -z "$PY" ]]; then
  warn "no suitable Python found (need 3.11, 3.12 or 3.13)."
  if command -v brew >/dev/null 2>&1; then
    if read -q "?Install Python 3.12 with Homebrew now? [y/N] "; then
      print; brew install python@3.12
      PY="$(find_python || true)"
    else
      print
    fi
  fi
  [[ -n "$PY" ]] || fail "install Python 3.12 (https://www.python.org/downloads/macos/ or 'brew install python@3.12'), then run this script again."
fi

PYVER="$("$PY" -c 'import sys; print("%d.%d" % sys.version_info[:2])')"
ARCH="$("$PY" -c 'import platform; print(platform.machine())')"
info "using Python $PYVER ($ARCH): $PY"

# Apple Silicon running an Intel (Rosetta) Python would get the slow, older Intel packages.
if [[ "$ARCH" == x86_64 && "$(sysctl -n sysctl.proc_translated 2>/dev/null || echo 0)" == 1 ]] || \
   [[ "$ARCH" == x86_64 && "$(sysctl -n hw.optional.arm64 2>/dev/null || echo 0)" == 1 ]]; then
  warn "this Mac has Apple Silicon but the Python found is an Intel build (Rosetta)."
  warn "for full speed install a native Python (python.org installer, or Homebrew in /opt/homebrew)."
fi

# ---------------------------------------------------------------- venv
if [[ -d .venv && ! -x .venv/bin/python ]]; then
  # e.g. a .venv copied over from Windows (it has Scripts\ instead of bin/)
  backup=".venv.unused-$(date +%Y%m%d-%H%M%S)"
  warn ".venv is not a macOS environment; moving it to $backup"
  mv .venv "$backup"
fi
if [[ -x .venv/bin/python ]]; then
  VENVVER="$(.venv/bin/python -c 'import sys; print("%d.%d" % sys.version_info[:2])' 2>/dev/null || echo broken)"
  if [[ "$VENVVER" != "$PYVER" ]]; then
    backup=".venv.unused-$(date +%Y%m%d-%H%M%S)"
    warn ".venv uses Python $VENVVER, not $PYVER; moving it to $backup"
    mv .venv "$backup"
  fi
fi
if [[ ! -x .venv/bin/python ]]; then
  info "creating virtual environment in .venv"
  "$PY" -m venv .venv
fi
VPY=.venv/bin/python

# ---------------------------------------------------------------- packages
info "installing packages (first time downloads about 1 GB, takes a few minutes)"
"$VPY" -m pip install --upgrade --quiet pip

if [[ "$ARCH" == arm64 ]]; then
  "$VPY" -m pip install -r requirements.txt
else
  # Intel Mac: Demucs 4.1 needs a library with no Intel-Mac build, and the last
  # Intel-Mac PyTorch (2.2.2) needs NumPy 1.x. MDX-Net (the default) is unaffected.
  warn "Intel Mac: installing without Demucs (KARAOKE_SEPARATOR=demucs won't be available)."
  tmpreq="$(mktemp "${TMPDIR:-/tmp}/karaoke-req.XXXXXX")"
  grep -v '^demucs' requirements.txt > "$tmpreq"
  "$VPY" -m pip install -r "$tmpreq" "torch==2.2.2" "numpy<2"
  rm -f "$tmpreq"
fi

# ---------------------------------------------------------------- checks
info "checking the installation"
"$VPY" - <<'EOF'
import importlib, subprocess, sys
mods = ["fastapi", "uvicorn", "yt_dlp", "faster_whisper", "openvino", "torch", "soundfile", "imageio_ffmpeg"]
bad = []
for m in mods:
    try:
        importlib.import_module(m)
    except Exception as e:  # noqa: BLE001
        bad.append(f"{m}: {e}")
import numpy, torch
try:
    torch.from_numpy(numpy.zeros(4, dtype=numpy.float32))      # PyTorch <-> NumPy must work
except Exception as e:  # noqa: BLE001
    bad.append(f"torch/numpy: {e}")
import imageio_ffmpeg
exe = imageio_ffmpeg.get_ffmpeg_exe()
r = subprocess.run([exe, "-version"], capture_output=True, text=True)
if r.returncode != 0:
    bad.append(f"ffmpeg at {exe} does not run")
import openvino as ov
print(f"   torch {torch.__version__}, numpy {numpy.__version__}, openvino {ov.__version__} devices {ov.Core().available_devices}")
print(f"   ffmpeg: {exe}")
if bad:
    print("\n".join("   PROBLEM: " + b for b in bad)); sys.exit(1)
EOF

# yt-dlp needs a JavaScript runtime for YouTube.
if ! command -v deno >/dev/null 2>&1; then
  warn "Deno not found. yt-dlp needs it to download from YouTube."
  if command -v brew >/dev/null 2>&1 && read -q "?Install Deno with Homebrew now? [y/N] "; then
    print; brew install deno
  else
    print
    print "   Install it later with:  brew install deno"
    print "   or:                     curl -fsSL https://deno.land/install.sh | sh"
  fi
fi

chmod +x run.sh run-mock.sh 2>/dev/null || true

ip="$(ipconfig getifaddr en0 2>/dev/null || ipconfig getifaddr en1 2>/dev/null || echo YOUR-MAC-IP)"
print
bold "Setup complete."
print "  Try the UI with fake songs:   ./run-mock.sh"
print "  Run for real:                 ./run.sh"
print "  TV / this Mac:                http://localhost:8000"
print "  Phones on the same Wi-Fi:     http://$ip:8000"
print "  (Allow incoming connections for Python if macOS asks.)"
