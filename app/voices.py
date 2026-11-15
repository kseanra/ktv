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

Lines whose main pitch sits almost entirely on one side of the boundary
(over 85% above, or under 15%) follow the pitch directly; the evidence
comparison decides the in-between lines. (High female lines can otherwise
read as male: a note at 1/3 of her pitch is in the male range and its 3rd
harmonic is her note.)

Known limits (measured on a duet with known answers: about 70-80% of solo
lines right): a man's high chorus notes or a woman's low verse notes can be
tagged as the other singer; lines sung together get one of the two tags.
"""
from __future__ import annotations

import logging
import re

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


# ----------------------------------------------------------------- pitch tagger
def pitch_tag_lines(audio: np.ndarray, lines: list[dict]) -> dict:
    """Add "voice": "M" / "F" / None to each lyric line (in place), from pitch.
    audio: 16 kHz mono float32 vocals. Returns calibration info, including
    each line's typical pitch ("medians", Hz or None)."""
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
    line_hz = [float(np.median(f1[a:b][voiced[a:b]])) if sp else None for sp in spans for a, b in [sp or (0, 0)]]
    for ln in lines:
        ln["voice"] = None
    if len(medians) < 4:
        return {"tagged": 0, "reason": "too few sung lines", "medians": line_hz}
    lo, hi = _two_groups(medians)
    if 1200 * np.log2(hi / lo) < 300:          # less than 3 semitones apart
        return {"tagged": 0, "reason": "could not find two separate voice ranges", "low_hz": lo, "high_hz": hi,
                "medians": line_hz}
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
    evidence = [None] * len(lines)
    for li, (ln, sp) in enumerate(zip(lines, spans)):
        if not sp:
            continue
        a, b = sp
        v = voiced[a:b]
        m = male_ev[a:b][v].mean() / nm
        f = fem_ev[a:b][v].mean() / nf
        share = np.mean(f1[a:b][v] >= th)      # main voice above the boundary
        evidence[li] = (float(m), float(f), float(share))
        if share > 0.85:
            ln["voice"] = "F"                   # clearly high: follow the pitch
        elif share < 0.15:
            ln["voice"] = "M"                   # clearly low
        else:
            ln["voice"] = "F" if f > m else "M"
        n += 1
    return {"tagged": n, "low_hz": round(lo, 1), "high_hz": round(hi, 1), "boundary_hz": round(th, 1),
            "medians": line_hz, "evidence": evidence}


# ----------------------------------------------------------------- voice identity (speaker embeddings)
# A speaker-embedding model turns a stretch of voice into a "voice fingerprint"
# (timbre), which, unlike pitch, doesn't change when a man sings high or a
# woman sings low. Lines are grouped into the song's two voices by fingerprint;
# only then is each group named M or F, by its overall pitch (or by the lyric
# marks 男/女 when some lines have them). Model: WeSpeaker ResNet34 (VoxCeleb,
# CC-BY-4.0), ONNX, run with ONNX Runtime on the CPU.
EMB_MODEL = "wespeaker_en_voxceleb_resnet34_LM.onnx"
EMB_URL = ("https://github.com/k2-fsa/sherpa-onnx/releases/download/"
           "speaker-recongition-models/" + EMB_MODEL)
MIN_EMB_SEC = 1.0          # a line needs this much singing for a fingerprint
_emb_sess = None


def _mel_bank(n_fft=512, n_mels=80, sr=16000, fmin=20.0, fmax=8000.0):
    mel = lambda f: 1127.0 * np.log(1.0 + f / 700.0)   # Kaldi's mel scale
    bins = np.arange(n_fft // 2 + 1) * sr / n_fft
    m = mel(bins)
    pts = np.linspace(mel(fmin), mel(fmax), n_mels + 2)
    fb = np.zeros((n_mels, len(bins)), np.float32)
    for i in range(n_mels):
        l, c, r = pts[i], pts[i + 1], pts[i + 2]
        up, down = (m - l) / (c - l), (r - m) / (r - c)
        fb[i] = np.maximum(0, np.minimum(up, down))
    return fb


_MEL = None


def fbank(audio: np.ndarray, scale: float = 32768.0) -> np.ndarray:
    """80-band log-mel filterbank like Kaldi's (25 ms frames, 10 ms hop, Povey
    window, pre-emphasis), mean-normalised: the input WeSpeaker models expect."""
    global _MEL
    if _MEL is None:
        _MEL = _mel_bank()
    x = audio.astype(np.float32) * scale
    if len(x) < 400:
        x = np.pad(x, (0, 400 - len(x)))
    fr = np.lib.stride_tricks.sliding_window_view(x, 400)[::160].copy()
    fr -= fr.mean(1, keepdims=True)
    fr[:, 1:] -= 0.97 * fr[:, :-1]
    fr[:, 0] *= 1 - 0.97
    win = (0.5 - 0.5 * np.cos(2 * np.pi * np.arange(400) / 399)) ** 0.85
    spec = np.abs(np.fft.rfft(fr * win, n=512)) ** 2
    feats = np.log(np.maximum(spec @ _MEL.T, np.finfo(np.float32).eps))
    return (feats - feats.mean(0, keepdims=True)).astype(np.float32)


def _embedder(models_dir):
    global _emb_sess
    if _emb_sess is None:
        import onnxruntime as ort
        path = models_dir / EMB_MODEL
        if not path.exists():
            import urllib.request
            models_dir.mkdir(parents=True, exist_ok=True)
            log.info("downloading the voice fingerprint model (%s, 26 MB) ...", EMB_MODEL)
            tmp = path.with_suffix(".part")
            urllib.request.urlretrieve(EMB_URL, tmp)
            tmp.replace(path)
        so = ort.SessionOptions()
        so.log_severity_level = 3
        _emb_sess = ort.InferenceSession(str(path), so, providers=["CPUExecutionProvider"])
    return _emb_sess


def embed(sess, audio: np.ndarray) -> np.ndarray:
    f = fbank(audio)[None]
    e = sess.run(None, {sess.get_inputs()[0].name: f})[0][0]
    return e / (np.linalg.norm(e) + 1e-9)


def _two_clusters(E: np.ndarray, iters: int = 30):
    """Split unit vectors into two groups (spherical 2-means, started from the
    two most different lines). Returns labels (0/1) and the two centres."""
    S = E @ E.T
    i, j = np.unravel_index(np.argmin(S), S.shape)
    C = E[[i, j]].copy()
    lab = np.zeros(len(E), int)
    for _ in range(iters):
        new = (E @ C.T).argmax(1)
        if (new == lab).all() and _ > 0:
            break
        lab = new
        for k in range(2):
            if (lab == k).any():
                c = E[lab == k].mean(0)
                C[k] = c / (np.linalg.norm(c) + 1e-9)
    return lab, C


def tag_lines(audio: np.ndarray, lines: list[dict], models_dir=None, method: str = "auto") -> dict:
    """Add "voice" M / F to each lyric line (in place). Lines with a lyric mark
    (line["mark"] = "M" / "F" / "B", from 男：/女：/合： in the lyrics) keep it.

    method "auto": voice fingerprints, falling back to pitch if the model
    can't be loaded or the song doesn't split into two clear voices;
    "pitch": the pitch tagger only."""
    marks = {i: ln["mark"] for i, ln in enumerate(lines) if ln.get("mark")}
    info = pitch_tag_lines(audio, lines)
    hz = info.pop("medians", None) or [None] * len(lines)
    evidence = info.pop("evidence", None) or [None] * len(lines)
    pitch_voice = [ln.get("voice") for ln in lines]
    info["method"] = "pitch"
    if method != "pitch" and models_dir is not None:
        try:
            info.update(_fingerprint_tags(audio, lines, models_dir, hz, marks, evidence))
        except Exception as e:  # noqa: BLE001
            log.warning("voice fingerprints failed (%s); using pitch tags", e)
            for ln, v in zip(lines, pitch_voice):
                ln["voice"] = v
            info["method"] = "pitch"
            info["fingerprint_error"] = str(e)
    for i, m in marks.items():                    # lyric marks always win
        lines[i]["voice"] = m
    info["marked"] = len(marks)
    return info


def _fingerprint_tags(audio, lines, models_dir, hz, marks, evidence=None) -> dict:
    sess = _embedder(models_dir)
    idx, embs = [], []
    for i, ln in enumerate(lines):
        if marks.get(i) == "B":                    # sung together: a mix of both voices
            continue
        a, b = int(max(0, ln["start"]) * SR), int(min(len(audio) / SR, ln["end"]) * SR)
        seg = audio[a:b]
        if len(seg) < MIN_EMB_SEC * SR or np.sqrt(np.mean(seg ** 2)) < 1e-3:
            continue
        idx.append(i)
        embs.append(embed(sess, seg))
    if len(idx) < 4:
        return {"method": "pitch", "fingerprint": "too few lines with enough singing"}
    E = np.stack(embs)
    lab, C = _two_clusters(E)
    sep = float(C[0] @ C[1])                       # similarity of the two voices (1 = same)
    sizes = [int((lab == 0).sum()), int((lab == 1).sum())]
    if min(sizes) < 2 or sep > 0.85:
        return {"method": "pitch", "fingerprint": f"no two clear voices (similarity {sep:.2f})"}
    # Name the groups: lyric marks if there are any, else overall pitch.
    votes = [0.0, 0.0]                             # evidence that group k is female
    for k_line, i in enumerate(idx):
        if marks.get(i) in ("M", "F"):
            votes[lab[k_line]] += 1 if marks[i] == "F" else -1
    if votes[0] != votes[1]:
        fem = 0 if votes[0] > votes[1] else 1
        named_by = "lyric marks"
    else:
        med = []
        for k in range(2):
            h = [hz[i] for k_line, i in enumerate(idx) if lab[k_line] == k and hz[i]]
            med.append(float(np.median(h)) if h else 0.0)
        fem = 0 if med[0] > med[1] else 1
        named_by = f"pitch ({med[0]:.0f} vs {med[1]:.0f} Hz)"
    sims = E @ C.T
    margin = np.abs(sims[:, 0] - sims[:, 1])
    voice = ["F" if l == fem else "M" for l in lab]
    # Unsure lines (nearly as close to both voices) follow their neighbours
    # when both neighbours agree: singers usually trade whole lines/verses.
    for k in range(1, len(voice) - 1):
        if margin[k] < 0.04 and voice[k - 1] == voice[k + 1] != voice[k]:
            voice[k] = voice[k - 1]
    # Sung together ("B"): the line's fingerprint is a mix, so it is fairly
    # close to BOTH voices (its similarity to the farther voice is high for
    # this song), and the pitch analysis finds a male and a female voice at
    # the same time. A part in brackets ("... (echo)") is usually the other
    # singer joining in, which counts as extra evidence.
    near_both = sims.min(1)
    thr = max(0.48, float(np.median(near_both)) + 0.08)
    both = 0
    evidence = evidence or [None] * len(lines)
    for k, i in enumerate(idx):
        ev = evidence[i]
        two_pitches = bool(ev) and ev[0] >= 0.85 and ev[1] >= 1.0
        bracket = bool(re.search(r"[\(（][^\)）]{2,}[\)）]", lines[i].get("text") or ""))
        if (near_both[k] >= thr and two_pitches) or (bracket and near_both[k] >= thr - 0.05):
            voice[k] = "B"
            both += 1
    by_line = dict(zip(idx, voice))
    for i, ln in enumerate(lines):
        if i in by_line:
            ln["voice"] = by_line[i]
    return {"method": "fingerprint", "fingerprinted": len(idx), "voice_similarity": round(sep, 3),
            "groups": sizes, "named_by": named_by, "together": both}
