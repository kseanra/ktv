"""Settings, read from environment variables so nothing needs editing in code."""
import os
from pathlib import Path

# HTTPS certificates. Python from python.org on macOS ships without trusted root
# certificates, so downloads fail with CERTIFICATE_VERIFY_FAILED. Verify against
# the operating system's certificate store instead (macOS Keychain, Windows
# store), as pip does; fall back to certifi's bundle. An SSL_CERT_FILE you set
# yourself (e.g. a company proxy certificate) is left alone.
if not os.environ.get("SSL_CERT_FILE"):
    try:
        import truststore
        truststore.inject_into_ssl()
    except Exception:  # noqa: BLE001
        try:
            import certifi
            os.environ["SSL_CERT_FILE"] = certifi.where()
        except ImportError:
            pass

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = Path(os.environ.get("KARAOKE_DATA", ROOT / "data"))
STATIC_DIR = ROOT / "static"

# How many songs may download at the same time (network-bound).
DOWNLOAD_WORKERS = int(os.environ.get("KARAOKE_DOWNLOAD_WORKERS", "2"))
# How many songs may run AI models at the same time. 1 is right for most GPUs
# and for CPU; raise to 2-3 on a GPU with plenty of memory.
MODEL_WORKERS = int(os.environ.get("KARAOKE_MODEL_WORKERS", "1"))

# faster-whisper model: tiny / base / small / medium / large-v3.
# "small" is a reasonable CPU default; use "large-v3" on a GPU for Cantonese.
WHISPER_MODEL = os.environ.get("KARAOKE_WHISPER_MODEL", "small")
# Vocal separation engine:
#   "mdx"    UVR MDX-Net vocal model on Intel OpenVINO (default). About 2x faster
#            than Demucs on CPU, and can use an Intel Arc / integrated GPU.
#   "demucs" Meta HTDemucs on PyTorch. Best choice with an NVIDIA GPU.
SEPARATOR = os.environ.get("KARAOKE_SEPARATOR", "mdx")
MDX_MODEL = os.environ.get("KARAOKE_MDX_MODEL", "Kim_Vocal_2")   # or UVR-MDX-NET-Voc_FT
# OpenVINO device for MDX: "auto" tries the GPU (checked against CPU output
# at startup) and falls back to CPU. Or force "GPU" / "CPU".
MDX_DEVICE = os.environ.get("KARAOKE_MDX_DEVICE", "auto")
MODELS_DIR = DATA_DIR / "models"
# Demucs model used when SEPARATOR=demucs.
DEMUCS_MODEL = os.environ.get("KARAOKE_DEMUCS_MODEL", "htdemucs")
DEMUCS_OVERLAP = float(os.environ.get("KARAOKE_DEMUCS_OVERLAP", "0.1"))
# PyTorch device for Demucs and Whisper: "auto" picks CUDA when available, otherwise CPU.
DEVICE = os.environ.get("KARAOKE_DEVICE", "auto")

# Refuse videos longer than this (seconds) so one request can't hog the workers.
MAX_DURATION = int(os.environ.get("KARAOKE_MAX_DURATION", "600"))
# Video height to download. 720 keeps files small; 1080 also works.
MAX_HEIGHT = int(os.environ.get("KARAOKE_MAX_HEIGHT", "720"))

# KARAOKE_MOCK=1 replaces download + models with fake but realistic stages,
# so the queue and player can be tried without YouTube access or model weights.
MOCK = os.environ.get("KARAOKE_MOCK", "0") == "1"
