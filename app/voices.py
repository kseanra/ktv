"""Male / female tags for duet lyrics, from pitch alone.

Works on the separated vocals (16 kHz mono). For each 10 ms frame:
  * a harmonic-salience pitch analysis: every candidate note is scored by how
    well its harmonic stack (1x, 2x, 3x ... the note) matches the spectrum;
  * male evidence  = strongest note in the male range, counting only its ODD
    harmonics (a woman an octave higher can't produce those);
  * female evidence = strongest note in the female range after removing the
    male candidate's harmonics (so a man's harmonics don't count as her notes).

The male and female ranges are calibrated per song: the typical pitch of every
lyric line is split into a low and a high group (the two singers). Each line
is then tagged with whichever evidence is stronger, relative to the song.

Known limits (measured on a duet with known answers: about 70-80% of solo
lines right): a man's high chorus notes or a woman's low verse notes can be
tagged as the other singer; lines sung together get one of the two tags.
"""
from __future__ import annotations

import logging

import numpy as np

log = logging.getLogger("karaoke.voices")

SR = 16000
N_FFT = 2048            # 128 ms window, 7.8 Hz per bin
HOP = 160               # 10 ms frames
FMIN, FMAX = 70.0, 1100.0
STEP_CENTS = 20
NH = 10                 # harmonics per candidate
FCEIL = 5000.0          # harmonics above this are ignored
DF = SR / N_FFT
NBINS = int(FCEIL * 1.1 / DF)   # spectrum bins kept (saves memory)
MIN_VOICED = 30         # frames (0.3 s) of singing needed to tag a line

CANDS = FMIN * 2 ** (np.arange(0, 1200 * np.log2(FMAX / FMIN), STEP_CENTS) / 1200)
_H = np.arange(1, NH + 1)
_HF = CANDS[:, None] * _H[None, :]
_IDX = np.clip(np.round(_HF / DF).astype(int), 0, NBINS - 1)
_W = (0.84 ** (_H - 1) * (_HF < FCEIL)).astype(np.float32)
_W_ODD = _W * (_H % 2 == 1)[None, :]
_W_EVEN = _W * (_H % 2 == 0)[None, :]
_OCTAVE = int(round(1200 / STEP_CENTS))


# ----------------------------------------------------------------- spectrum
def _spectrum(audio: np.ndarray) -> np.ndarray:
    """Whitened, log-compressed magnitude spectrogram, shape (NBINS, frames)."""
    win = np.hanning(N_FFT).astype(np.float32)
    pad = np.pad(audio.astype(np.float32), (N_FFT // 2, N_FFT // 2))
    frames = np.lib.stride_tricks.sliding_window_view(pad, N_FFT)[::HOP]
    out = np.empty((NBINS, len(frames)), np.float32)
    for t0 in range(0, len(frames), 2000):                # in blocks to limit memory
        m = np.abs(np.fft.rfft(frames[t0:t0 + 2000] * win, axis=-1))[:, :NBINS]
        out[:, t0:t0 + 2000] = m.T
    ref = np.percentile(out, 99.5) + 1e-9
    s = np.log1p(1000 * out / ref)
    # remove the slowly varying floor so only harmonic peaks remain
    k = np.ones(15, np.float32) / 15
    floor = np.apply_along_axis(lambda c: np.convolve(c, k, mode="same"), 0, s)
    return np.maximum(s - floor, 0).astype(np.float32)


def _salience(s: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """(NBINS, T) -> (candidates, T): weighted sum over each candidate's harmonics
    (taking the max of neighbouring bins, to tolerate slightly sharp/flat notes)."""
    sm = s.copy()
    sm[1:] = np.maximum(sm[1:], s[:-1])
    sm[:-1] = np.maximum(sm[:-1], s[1:])
    T = s.shape[1]
    out = np.empty((len(CANDS), T), np.float32)
    for t0 in range(0, T, 1500):
        out[:, t0:t0 + 1500] = np.einsum("cht,ch->ct", sm[_IDX, t0:t0 + 1500], weights)
    return out


def _main_pitch(s: np.ndarray, sal: np.ndarray) -> np.ndarray:
    """Candidate index of the main pitch per frame, with octave correction:
    a candidate whose odd harmonics are missing is really an octave too low."""
    so, se = _salience(s, _W_ODD), _salience(s, _W_EVEN)
    T = sal.shape[1]
    tt = np.arange(T)
    c = sal.argmax(0)
    for _ in range(2):
        up = (so[c, tt] / (se[c, tt] + 1e-6) < 0.5) & (c + _OCTAVE < len(CANDS))
        for t in np.nonzero(up)[0]:
            lo, hi = c[t] + _OCTAVE - 5, min(c[t] + _OCTAVE + 6, len(CANDS))
            c[t] = lo + sal[lo:hi, t].argmax()
    return c


def _cancel(s: np.ndarray, c: np.ndarray, rel=0.035, min_bins=2) -> np.ndarray:
    """Remove the harmonics of candidate c (per frame). The removed band widens
    with harmonic number: vibrato of a few % moves high harmonics a lot."""
    r = s.copy()
    F, T = s.shape
    f0 = CANDS[c]
    tt = np.arange(T)
    for h in range(1, 80):
        fc = h * f0
        ok = fc < FCEIL * 1.05
        if not ok.any():
            break
        half = np.maximum(min_bins, np.round(rel * fc / DF)).astype(int)
        b = np.round(fc / DF).astype(int)
        for d in range(-int(half[ok].max()), int(half[ok].max()) + 1):
            sel = ok & (np.abs(d) <= half)
            r[np.clip(b + d, 0, F - 1)[sel], tt[sel]] = 0
    return r


def _medfilt(x: np.ndarray, k: int) -> np.ndarray:
    xp = np.pad(x, k // 2, mode="edge")
    return np.median(np.lib.stride_tricks.sliding_window_view(xp, k), axis=-1)


def _two_groups(values: np.ndarray) -> tuple[float, float]:
    """Split pitches (Hz) into a low and a high group (1-D k-means on log pitch)."""
    v = np.log2(values)
    lo, hi = np.percentile(v, 25), np.percentile(v, 75)
    for _ in range(100):
        th = (lo + hi) / 2
        a, b = v[v < th], v[v >= th]
        if len(a) == 0 or len(b) == 0:
            break
        lo, hi = a.mean(), b.mean()
    return float(2 ** lo), float(2 ** hi)


# ----------------------------------------------------------------- public
def tag_lines(audio: np.ndarray, lines: list[dict]) -> dict:
    """Add "voice": "M" / "F" / None to each lyric line (in place).
    audio: 16 kHz mono float32 vocals. Returns calibration info."""
    s = _spectrum(audio)
    T = s.shape[1]
    # singing frames: within 30 dB of the loud parts
    pad = np.pad(audio, (N_FFT // 2, N_FFT // 2))
    fr = np.lib.stride_tricks.sliding_window_view(pad, 400)[::HOP][:T]
    db = 20 * np.log10(np.sqrt((fr.astype(np.float32) ** 2).mean(1)) + 1e-9)
    voiced = db > np.percentile(db, 98) - 30

    sal = _salience(s, _W)
    f1 = CANDS[_main_pitch(s, sal)]
    del sal

    spans = []
    for ln in lines:
        a = max(0, int(ln["start"] * 100)); b = min(T, int(ln["end"] * 100) + 1)
        spans.append((a, b) if b - a > 0 and voiced[a:b].sum() >= MIN_VOICED else None)
    medians = np.array([np.median(f1[a:b][voiced[a:b]]) for sp in spans if sp for a, b in [sp]])
    for ln in lines:
        ln["voice"] = None
    if len(medians) < 4:
        return {"tagged": 0, "reason": "too few sung lines"}
    lo, hi = _two_groups(medians)
    if 1200 * np.log2(hi / lo) < 300:          # less than 3 semitones apart
        return {"tagged": 0, "reason": "could not find two separate voice ranges", "low_hz": lo, "high_hz": hi}
    th = float(np.sqrt(lo * hi))

    male_band = (CANDS >= lo / 1.5) & (CANDS <= th * 1.06)
    fem_band = (CANDS >= th / 1.06) & (CANDS <= hi * 1.5)
    so = _salience(s, _W_ODD)
    cm = np.where(male_band[:, None], so, -1).argmax(0)
    male_ev = so[cm, np.arange(T)]
    del so
    sres = _salience(_cancel(s, cm), _W)
    fem_ev = np.where(fem_band[:, None], sres, -1).max(0)
    del sres
    male_ev, fem_ev = _medfilt(male_ev, 15), _medfilt(fem_ev, 15)
    nm, nf = np.median(male_ev[voiced]) + 1e-6, np.median(fem_ev[voiced]) + 1e-6

    n = 0
    for ln, sp in zip(lines, spans):
        if not sp:
            continue
        a, b = sp
        v = voiced[a:b]
        m = male_ev[a:b][v].mean() / nm
        f = fem_ev[a:b][v].mean() / nf
        ln["voice"] = "F" if f > m else "M"
        n += 1
    return {"tagged": n, "low_hz": round(lo, 1), "high_hz": round(hi, 1), "boundary_hz": round(th, 1)}
