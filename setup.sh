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
  # Intel Mac: Demucs 4.1 needs a library with no Intel-Mac build, so it is
  # left out, and with it PyTorch. That is deliberate: on Intel Macs PyTorch
  # and Whisper each bundle Intel's OpenMP runtime, and loading both crashes
  # the server (OMP Error #15 / segmentation fault). MDX-Net, the default
  # vocal remover, doesn't need PyTorch.
  warn "Intel Mac: installing without Demucs/PyTorch (KARAOKE_SEPARATOR=demucs won't be available)."
  tmpreq="$(mktemp "${TMPDIR:-/tmp}/karaoke-req.XXXXXX")"
  grep -v '^demucs' requirements.txt > "$tmpreq"
  "$VPY" -m pip install -r "$tmpreq" "numpy>=2"
  rm -f "$tmpreq"
  # An earlier version of this script installed PyTorch here; remove it.
  if "$VPY" -m pip show torch >/dev/null 2>&1; then
    info "removing PyTorch installed by an earlier setup (not needed on Intel Macs)"
    "$VPY" -m pip uninstall -y --quiet torch
  fi
fi

# ---------------------------------------------------------------- checks
info "checking the installation"
# Each library is imported in its own process: on Intel Macs, PyTorch and
# Whisper must never be loaded together (see above).
mods=(fastapi uvicorn yt_dlp faster_whisper openvino soundfile imageio_ffmpeg truststore certifi)
[[ "$ARCH" == arm64 ]] && mods+=(torch demucs)
problems=0
for m in $mods; do
  if ! err="$("$VPY" -c "import $m" 2>&1)"; then
    print "   PROBLEM: $m: ${err##*$'\n'}"; problems=1
  fi
done
# Run as "python -c" rather than feeding the script on stdin: importing
# OpenVINO starts a side process (its telemetry), and on macOS a side process
# re-runs the main script from its file, which fails for "<stdin>".
check_code="$(cat <<'EOF'
import app.config, subprocess, numpy, imageio_ffmpeg, openvino as ov
exe = imageio_ffmpeg.get_ffmpeg_exe()
if subprocess.run([exe, "-version"], capture_output=True).returncode != 0:
    raise SystemExit(f"   PROBLEM: ffmpeg at {exe} does not run")
numpy.fft.rfft(numpy.zeros(16, numpy.float32))
print(f"   numpy {numpy.__version__}, openvino {ov.__version__} devices {ov.Core().available_devices}")
print(f"   ffmpeg: {exe}")
EOF
)"
"$VPY" -c "$check_code" || problems=1
(( problems == 0 )) || fail "some packages did not install correctly (see PROBLEM lines above)."

# HTTPS must work (python.org builds have no root certificates; the app uses
# the macOS Keychain via truststore, this checks that it really works).
if ! "$VPY" -c 'import app.config, urllib.request; urllib.request.urlopen("https://www.youtube.com", timeout=15)' 2>/dev/null; then
  warn "HTTPS check to youtube.com failed (no internet, or a certificate problem)."
  print "   If downloads fail with CERTIFICATE_VERIFY_FAILED, run the 'Install Certificates.command'"
  print "   in Applications > Python $PYVER, then start the server again."
else
  info "HTTPS to youtube.com works"
fi

# yt-dlp needs a JavaScript runtime for YouTube: Deno (preferred), Node.js,
# QuickJS or Bun. Deno is installed with its official installer, which
# downloads a ready-made program (macOS 12+). Homebrew is not used for it:
# on older macOS versions Homebrew tries to compile Deno and demands Xcode.
if command -v deno >/dev/null 2>&1 || [[ -x "$HOME/.deno/bin/deno" ]]; then
  info "Deno found"
elif command -v node >/dev/null 2>&1 || command -v bun >/dev/null 2>&1 || command -v qjs >/dev/null 2>&1; then
  info "Node.js / Bun / QuickJS found; yt-dlp will use it (Deno is optional)"
else
  warn "no JavaScript runtime found. yt-dlp needs one to download from YouTube."
  if read -q "?Install Deno now (official installer, into ~/.deno)? [y/N] "; then
    print
    curl -fsSL https://deno.land/install.sh | sh -s -- -y --no-modify-path \
      && info "Deno installed in ~/.deno/bin (the app finds it there)" \
      || warn "Deno install failed; install it later with: curl -fsSL https://deno.land/install.sh | sh"
  else
    print
    print "   Install it later with:  curl -fsSL https://deno.land/install.sh | sh"
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
