"""Reference melody for singing scores.

The original singer's pitch, taken from the kept 16 kHz vocals
(vocals16k.flac) with the same pitch tracker as the duet voice tags.
Saved as melody.json next to lyrics.json:

    {"version": 1, "hop": 0.02, "midi10": [0, 0, 693, 694, ...]}

midi10 = MIDI note number x 10 (69.3 -> 693) every 20 ms; 0 = no singing.
Takes about 10-30 s per song on the CPU and is made once per song.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np

log = logging.getLogger("karaoke.melody")

VERSION = 1
HOP_SEC = 0.02
CHUNK_SEC = 60          # processed in pieces so memory stays flat


def path(out: Path) -> Path:
    return out / "melody.json"


def has_vocals(out: Path) -> bool:
    f = out / "vocals16k.flac"
    return f.exists() and f.stat().st_size > 0


def exists(out: Path) -> bool:
    p = path(out)
    if not p.exists():
        return False
    try:
        return json.loads(p.read_text(encoding="utf-8")).get("version") == VERSION
    except (OSError, ValueError):
        return False


def track(audio: np.ndarray) -> np.ndarray:
    """16 kHz mono vocals -> MIDI pitch every 10 ms (0 = not singing)."""
    from . import voices as v
    sr, hop = v.SR, v.HOP
    n_frames = len(audio) // hop + 1
    f0 = np.zeros(n_frames, np.float32)
    strength = np.zeros(n_frames, np.float32)
    step = CHUNK_SEC * sr
    for a in range(0, len(audio), step):
        part = audio[a:a + step]
        if len(part) < v.N_FFT:
            break
        s = v._spectrum(part)
        sal = v._salience(s, v._W)
        c = v._main_pitch(s, sal)
        t0 = a // hop
        n = min(len(c), n_frames - t0)
        f0[t0:t0 + n] = v.CANDS[c[:n]]
        strength[t0:t0 + n] = sal[c[:n], np.arange(n)]
        del s, sal

    # loudness per 10 ms frame (25 ms window)
    pad = np.pad(audio.astype(np.float32), (200, 200))
    fr = np.lib.stride_tricks.sliding_window_view(pad, 400)[::hop][:n_frames]
    db = 20 * np.log10(np.sqrt((fr ** 2).mean(1)) + 1e-9)
    if len(db) < n_frames:
        db = np.pad(db, (0, n_frames - len(db)), constant_values=-120)
    loud = db > max(np.percentile(db, 98) - 30, -55)
    clear = strength > 0.25 * (np.percentile(strength[loud], 90) if loud.any() else 1)
    voiced = loud & clear & (f0 > 0)

    midi = np.where(voiced, 69 + 12 * np.log2(np.maximum(f0, 1) / 440.0), 0).astype(np.float32)
    # smooth single-frame octave jumps and drop specks shorter than 80 ms
    smooth = midi.copy()
    for t in np.nonzero(voiced)[0]:
        w = midi[max(0, t - 2):t + 3]
        w = w[w > 0]
        smooth[t] = np.median(w)
    out = np.zeros_like(smooth)
    t, T = 0, len(smooth)
    while t < T:
        if smooth[t] <= 0:
            t += 1
            continue
        u = t
        while u < T and smooth[u] > 0:
            u += 1
        if u - t >= 8:
            out[t:u] = smooth[t:u]
        t = u
    return out


def build(out: Path) -> dict:
    """Make melody.json for one song folder. Raises FileNotFoundError when the
    song has no kept vocals."""
    import soundfile as sf
    flac = out / "vocals16k.flac"
    if not flac.exists():
        raise FileNotFoundError("no vocals")
    if flac.stat().st_size == 0:
        midi10 = _mock_melody(out)              # mock mode: no real vocals
    else:
        audio, sr = sf.read(flac, dtype="float32", always_2d=False)
        if audio.ndim > 1:
            audio = audio.mean(1)
        if sr != 16000:
            raise RuntimeError(f"expected 16 kHz vocals, got {sr}")
        m = track(audio)
        m = m[: len(m) // 2 * 2].reshape(-1, 2)          # 10 ms -> 20 ms
        m = np.where((m > 0).all(1), m.mean(1), m.max(1))
        midi10 = [int(round(x * 10)) for x in m]
    data = {"version": VERSION, "hop": HOP_SEC, "midi10": midi10}
    path(out).write_text(json.dumps(data, separators=(",", ":")), encoding="utf-8")
    sung = sum(1 for x in midi10 if x)
    log.info("melody for %s: %.0f s sung of %.0f s", out.name, sung * HOP_SEC, len(midi10) * HOP_SEC)
    return data


def _mock_melody(out: Path) -> list[int]:
    """Mock songs: A4 (the mock 440 Hz "vocal") during every lyric line."""
    try:
        lines = json.loads((out / "lyrics.json").read_text(encoding="utf-8")).get("lines") or []
    except (OSError, ValueError):
        lines = []
    n = int(30 / HOP_SEC)
    midi10 = [0] * n
    for ln in lines:
        for i in range(int(ln["start"] / HOP_SEC), min(n, int(ln["end"] / HOP_SEC))):
            midi10[i] = 690
    return midi10
