"""Benchmark vocal-removal options on a song you already downloaded.

    python tools/bench_separation.py data/<video_id>/video.mp4 --seconds 60

For each option it prints the time taken, speed (x real time) and how closely
its vocals match the current setup (dB; higher = closer; above ~15 dB the
difference is hard to hear). Outputs are written to bench_out/ so you can
listen and compare.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.pipeline import run_ffmpeg  # noqa: E402
from app.separators import DemucsSeparator, MDXSeparator  # noqa: E402


def sdr(ref, est):
    return 10 * np.log10((ref ** 2).sum() / (((ref - est) ** 2).sum() + 1e-10) + 1e-10)


class DemucsApply:
    """Exactly what the app did before: demucs.apply.apply_model, overlap 0.25."""
    def __init__(self, device="cpu"):
        from demucs.pretrained import get_model
        self.m = get_model("htdemucs").to(device).eval()
        self.device = device
        self.name = f"demucs:htdemucs apply_model overlap=0.25 on {device} (current)"

    def separate(self, mix, progress=None):
        import torch
        from demucs.apply import apply_model
        x = torch.from_numpy(mix)
        ref = x.mean(0); mean, std = ref.mean(), ref.std() + 1e-8
        with torch.no_grad():
            s = apply_model(self.m, ((x - mean) / std)[None], device=self.device, split=True, overlap=0.25, progress=False)[0]
        s = s * std + mean
        v = s[self.m.sources.index("vocals")]
        return v.numpy(), (s.sum(0) - v).numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("input", help="video.mp4 or a wav file")
    ap.add_argument("--start", type=float, default=30, help="excerpt start (s)")
    ap.add_argument("--seconds", type=float, default=60, help="excerpt length (s); 0 = whole song")
    ap.add_argument("--only", default="", help="comma-separated option numbers to run, e.g. 1,4,5")
    ap.add_argument("--threads", type=int, default=0, help="CPU threads (0 = library default)")
    args = ap.parse_args()

    out_dir = ROOT / "bench_out"
    out_dir.mkdir(exist_ok=True)
    wav = out_dir / "excerpt.wav"
    cut = ["-ss", str(args.start), "-t", str(args.seconds)] if args.seconds else []
    run_ffmpeg(*cut, "-i", args.input, "-vn", "-ac", "2", "-ar", "44100", str(wav))
    mix, _ = sf.read(wav, dtype="float32", always_2d=True)
    mix = np.ascontiguousarray(mix.T)
    dur = mix.shape[1] / 44100

    import torch
    has_xpu = hasattr(torch, "xpu") and torch.xpu.is_available()
    has_cuda = torch.cuda.is_available()
    try:
        import openvino as ov
        ov_devices = ov.Core().available_devices
    except Exception:  # noqa: BLE001
        ov_devices = []
    print(f"excerpt: {dur:.0f}s | torch {torch.__version__} threads={torch.get_num_threads()} "
          f"cuda={has_cuda} xpu={has_xpu} | openvino devices={ov_devices}\n")

    t = args.threads
    md = ROOT / "data" / "models"
    options = [
        ("1", lambda: DemucsApply("cpu")),
        ("2", lambda: DemucsSeparator(overlap=0.1, batch=1, threads=t)),
        ("3", lambda: DemucsSeparator(overlap=0.1, batch=4, threads=t)),
        ("4", lambda: MDXSeparator("Kim_Vocal_2", md, backend="onnxruntime", threads=t)),
        ("5", lambda: MDXSeparator("Kim_Vocal_2", md, backend="openvino", device="CPU", threads=t)),
    ]
    if "GPU" in ov_devices:
        options.append(("6", lambda: MDXSeparator("Kim_Vocal_2", md, backend="openvino", device="GPU", batch=2)))
    if has_xpu:
        options.append(("7", lambda: DemucsSeparator(device="xpu", overlap=0.1, batch=4)))
    if has_cuda:
        options.append(("8", lambda: DemucsSeparator(device="cuda", overlap=0.1, batch=4)))
    only = set(filter(None, args.only.split(",")))

    ref = None
    rows = []
    for key, make in options:
        if only and key not in only:
            continue
        try:
            t0 = time.time(); sep = make(); load = time.time() - t0
            sep.separate(mix[:, : 44100 * 8])                  # warm-up (first run compiles kernels)
            t0 = time.time(); v, inst = sep.separate(mix); took = time.time() - t0
        except Exception as e:  # noqa: BLE001
            print(f"[{key}] failed: {e}")
            continue
        if ref is None:
            ref = v
        agree = sdr(ref, v)
        sf.write(out_dir / f"{key}_vocals.wav", v.T, 44100)
        sf.write(out_dir / f"{key}_instrumental.wav", inst.T, 44100)
        rows.append((key, sep.name, took, dur / took, agree, load))
        print(f"[{key}] {sep.name}\n     {took:6.1f}s  = {dur / took:4.2f}x real time | "
              f"match vs first: {agree:5.1f} dB | load {load:.1f}s", flush=True)

    if rows:
        base = rows[0][2]
        print("\nsummary (full 4-min song estimate on this machine):")
        for key, name, took, rt, agree, _ in rows:
            print(f"  [{key}] {240 / rt:6.0f}s  {base / took:4.1f}x faster than [{rows[0][0]}]  {name}")
    print(f"\nlisten: {out_dir}")


if __name__ == "__main__":
    main()
