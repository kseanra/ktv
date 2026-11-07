"""The three processing stages for one song.

    download    yt-dlp fetches the video (mp4); ffmpeg pulls a WAV from that same
                file, so the separated audio lines up exactly with the video.
    separate    Demucs splits the WAV into vocals + instrumental.
    transcribe  faster-whisper listens to the *isolated vocals* (much more
                accurate than the full mix) and returns per-word timestamps.

Each method is called from a worker thread with (job, out_dir) and writes its
results into out_dir. Models are loaded once and kept in memory.
"""
from __future__ import annotations

import json
import logging
import re
import shutil
import subprocess
import threading
import time
from pathlib import Path

from . import config

log = logging.getLogger("karaoke.pipeline")

_YT_ID = re.compile(r"(?:v=|/shorts/|/embed/|/live/|youtu\.be/)([A-Za-z0-9_-]{11})")


def extract_video_id(url: str) -> str | None:
    url = url.strip()
    if re.fullmatch(r"[A-Za-z0-9_-]{11}", url):
        return url
    m = _YT_ID.search(url)
    return m.group(1) if m else None


_FFMPEG: str | None = None


def ffmpeg_exe() -> str:
    """ffmpeg from PATH if installed, otherwise the copy bundled with the
    imageio-ffmpeg package, so no separate install or PATH setup is needed."""
    global _FFMPEG
    if _FFMPEG is None:
        candidates = []
        try:
            import imageio_ffmpeg
            candidates.append(("bundled (imageio-ffmpeg)", imageio_ffmpeg.get_ffmpeg_exe()))
        except Exception as e:  # noqa: BLE001
            log.warning("imageio-ffmpeg unavailable: %s", e)
        if shutil.which("ffmpeg"):
            candidates.append(("PATH", shutil.which("ffmpeg")))
        for source, exe in candidates:
            # Only accept a binary that actually runs: a broken shim on PATH
            # would otherwise make yt-dlp report "ffmpeg is not installed".
            try:
                r = subprocess.run([exe, "-version"], capture_output=True, text=True, timeout=30)
                first = (r.stdout or "").splitlines()[0] if r.stdout else ""
                if r.returncode == 0 and first.startswith("ffmpeg version"):
                    _FFMPEG = exe
                    log.info("using ffmpeg from %s: %s (%s)", source, exe, first[:40])
                    break
                log.warning("ffmpeg from %s does not work: %s", source, exe)
            except Exception as e:  # noqa: BLE001
                log.warning("ffmpeg from %s failed to start (%s): %s", source, exe, e)
        if _FFMPEG is None:
            raise RuntimeError("no working ffmpeg: run 'pip install imageio-ffmpeg' inside .venv")
    return _FFMPEG


def check_ffmpeg_for_ytdlp() -> str:
    """Startup self-test: can yt-dlp merge video + audio with our ffmpeg?"""
    exe = ffmpeg_exe()
    import yt_dlp
    from yt_dlp.postprocessor.ffmpeg import FFmpegMergerPP
    with yt_dlp.YoutubeDL({"ffmpeg_location": exe, "quiet": True}) as ydl:
        ok = FFmpegMergerPP(ydl).available
    if not ok:
        raise RuntimeError(f"yt-dlp cannot use ffmpeg at {exe}")
    return exe


def run_ffmpeg(*args: str):
    cmd = [ffmpeg_exe(), "-hide_banner", "-loglevel", "error", "-y", *args]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"ffmpeg failed: {r.stderr.strip()[-400:]}")


def split_lines(segments: list[dict], max_words: int = 8, max_chars: int = 16, gap: float = 0.7) -> list[dict]:
    """Whisper segments can be long. Break them into karaoke-sized lines at
    pauses, or when a line gets too long (chars for CJK, words otherwise)."""
    lines = []
    for seg in segments:
        cur: list[dict] = []
        for w in seg["words"]:
            if cur:
                text = "".join(x["w"] for x in cur)
                is_cjk = bool(re.search(r"[぀-ヿ㐀-鿿가-힯]", text))
                too_long = len(text.strip()) >= max_chars if is_cjk else len(cur) >= max_words
                if w["s"] - cur[-1]["e"] > gap or too_long:
                    lines.append(cur)
                    cur = []
            cur.append(w)
        if cur:
            lines.append(cur)
    out = []
    for ws in lines:
        out.append({
            "start": ws[0]["s"],
            "end": ws[-1]["e"],
            "text": "".join(w["w"] for w in ws).strip(),
            "words": ws,
        })
    return out


SR_TEST = 44100


class RealPipeline:
    def __init__(self):
        self._lock = threading.Lock()
        self._demucs = None
        self._whisper = None
        self._device = None

    # ---------- model loading (once, shared by all worker threads) ----------
    def device(self) -> str:
        if self._device is None:
            if config.DEVICE != "auto":
                self._device = config.DEVICE
            else:
                import torch
                self._device = "cuda" if torch.cuda.is_available() else "cpu"
            log.info("using device: %s", self._device)
        return self._device

    def separator(self):
        with self._lock:
            if self._demucs is None:
                self._demucs = self._make_separator()
                log.info("vocal separator: %s", self._demucs.name)
        return self._demucs

    def _make_separator(self):
        from .separators import DemucsSeparator, MDXSeparator

        if config.SEPARATOR == "demucs":
            dev = self.device()
            return DemucsSeparator(config.DEMUCS_MODEL, dev, overlap=config.DEMUCS_OVERLAP,
                                   batch=1 if dev == "cpu" else 4)
        cpu = MDXSeparator(config.MDX_MODEL, config.MODELS_DIR, backend="openvino", device="CPU")
        want = config.MDX_DEVICE.upper()
        if want == "CPU":
            return cpu
        try:
            import openvino as ov
            if "GPU" not in ov.Core().available_devices:
                if want == "GPU":
                    log.warning("OpenVINO found no GPU, using CPU")
                return cpu
            log.info("compiling MDX model for the GPU (first time takes a minute) ...")
            gpu = MDXSeparator(config.MDX_MODEL, config.MODELS_DIR, backend="openvino", device="GPU", batch=2)
            # Check the GPU (fp16) gives the same answer as the CPU on a test signal.
            import numpy as np
            rng = np.random.default_rng(0)
            t = np.arange(SR_TEST * 6) / SR_TEST
            tone = sum(np.sin(2 * np.pi * f * t) / (k + 1) for k, f in enumerate([220, 330, 440, 660, 880]))
            test = np.stack([tone, tone * 0.8]).astype(np.float32) * 0.2
            test += rng.normal(0, 0.02, test.shape).astype(np.float32)
            a, _ = cpu.separate(test)
            b, _ = gpu.separate(test)
            err = 10 * np.log10((a ** 2).sum() / (((a - b) ** 2).sum() + 1e-10) + 1e-10)
            if not np.isfinite(b).all() or err < 20:
                log.warning("GPU output differs from CPU (%.1f dB), using CPU", err)
                return cpu
            log.info("GPU check OK (%.1f dB match with CPU)", err)
            return gpu
        except Exception as e:  # noqa: BLE001
            log.warning("GPU not usable (%s), using CPU", e)
            return cpu

    def whisper(self):
        with self._lock:
            if self._whisper is None:
                from faster_whisper import WhisperModel
                dev = self.device()
                compute = "float16" if dev == "cuda" else "int8"
                log.info("loading Whisper model %s (%s, %s) ...", config.WHISPER_MODEL, dev, compute)
                self._whisper = WhisperModel(config.WHISPER_MODEL, device=dev, compute_type=compute)
        return self._whisper

    # ---------- stage 1: download ----------
    def download(self, job, out: Path):
        import yt_dlp

        def hook(d):
            if d.get("status") == "downloading":
                total = d.get("total_bytes") or d.get("total_bytes_estimate") or 0
                if total:
                    job.progress = min(d.get("downloaded_bytes", 0) / total, 1.0)

        video = out / "video.mp4"
        info_file = out / "info.json"
        if video.exists():
            # Already downloaded earlier (e.g. reprocessing): don't fetch it again.
            if info_file.exists():
                d = json.loads(info_file.read_text(encoding="utf-8"))
                job.title, job.duration = d.get("title", ""), d.get("duration", 0)
            else:
                with yt_dlp.YoutubeDL({"quiet": True, "no_warnings": True, "noplaylist": True}) as ydl:
                    info = ydl.extract_info(job.url, download=False)
                job.title, job.duration = info.get("title", ""), info.get("duration") or 0
            job.progress = 1.0
            run_ffmpeg("-i", str(video), "-vn", "-ac", "2", "-ar", "44100", str(out / "audio.wav"))
            info_file.write_text(json.dumps({"title": job.title, "duration": job.duration}, ensure_ascii=False),
                                 encoding="utf-8")
            return

        h = config.MAX_HEIGHT
        opts = {
            # Prefer mp4 video + m4a audio so the merged file plays in any browser.
            "format": f"bv*[height<={h}][ext=mp4]+ba[ext=m4a]/b[height<={h}][ext=mp4]/bv*[height<={h}]+ba/b",
            "merge_output_format": "mp4",
            "ffmpeg_location": ffmpeg_exe(),   # needed to merge video + audio streams
            "outtmpl": str(out / "video.%(ext)s"),
            "noplaylist": True,
            "quiet": True,
            "no_warnings": True,
            "progress_hooks": [hook],
        }
        with yt_dlp.YoutubeDL(opts) as ydl:
            info = ydl.extract_info(job.url, download=False)
            if info.get("is_live"):
                raise RuntimeError("live streams are not supported")
            dur = info.get("duration") or 0
            if dur > config.MAX_DURATION:
                raise RuntimeError(f"video is {dur}s, limit is {config.MAX_DURATION}s")
            job.title = info.get("title", "")
            job.duration = dur
            ydl.process_ie_result(info, download=True)

        video = out / "video.mp4"
        if not video.exists():
            # yt-dlp fell back to another container; remux to mp4 for the browser.
            others = [p for p in out.glob("video.*") if p.suffix not in (".part", ".ytdl")]
            if not others:
                raise RuntimeError("download produced no video file")
            run_ffmpeg("-i", str(others[0]), "-c", "copy", "-movflags", "+faststart", str(video))
            others[0].unlink()

        # Audio for the models, taken from the same file the browser will play.
        run_ffmpeg("-i", str(video), "-vn", "-ac", "2", "-ar", "44100", str(out / "audio.wav"))
        (out / "info.json").write_text(json.dumps({"title": job.title, "duration": job.duration}, ensure_ascii=False),
                                       encoding="utf-8")

    # ---------- stage 2: vocal separation ----------
    def separate(self, job, out: Path):
        import numpy as np
        import soundfile as sf

        sep = self.separator()
        audio, sr = sf.read(out / "audio.wav", dtype="float32", always_2d=True)  # (samples, ch)
        if sr != 44100:
            raise RuntimeError(f"expected 44100 Hz audio, got {sr}")
        mix = np.ascontiguousarray(audio.T)                     # (ch, samples)

        def progress(p):
            job.progress = 0.95 * p

        # The song is processed in short overlapping chunks, so memory stays
        # flat however long the song is.
        vocals, instrumental = sep.separate(mix, progress)

        def save(arr, name):
            sf.write(out / f"{name}.wav", np.clip(arr.T, -1.0, 1.0), sr)

        save(vocals, "vocals")
        save(instrumental, "instrumental")
        # The browser streams a compressed copy of the instrumental.
        run_ffmpeg("-i", str(out / "instrumental.wav"), "-b:a", "192k", str(out / "instrumental.mp3"))
        (out / "instrumental.wav").unlink(missing_ok=True)
        (out / "audio.wav").unlink(missing_ok=True)

    # ---------- stage 3: lyrics ----------
    def transcribe(self, job, out: Path):
        import soundfile as sf

        model = self.whisper()
        # Decode with our own ffmpeg to 16 kHz mono and pass the samples in
        # directly. (faster-whisper's built-in decoder breaks with PyAV >= 16.)
        run_ffmpeg("-i", str(out / "vocals.wav"), "-ac", "1", "-ar", "16000", str(out / "vocals16k.wav"))
        audio, _ = sf.read(out / "vocals16k.wav", dtype="float32")
        (out / "vocals16k.wav").unlink(missing_ok=True)
        language = job.language
        if language == "yue" and model.hf_tokenizer.token_to_id("<|yue|>") is None:
            # Only large-v3 models know Cantonese. Smaller ones have no "yue"
            # token: asking for it silently breaks decoding (junk English like
            # "The JackINS", and very slow). Cantonese is written in Chinese
            # characters, so ask for Chinese instead.
            log.info("model %s has no Cantonese; transcribing %s as Chinese", config.WHISPER_MODEL, job.video_id)
            language = "zh"
        segments, info = model.transcribe(
            audio,
            language=language,              # None = auto-detect; "yue" = Cantonese (large-v3 only)
            word_timestamps=True,           # needed for word-by-word fill
            vad_filter=True,                # skip instrumental gaps (fewer made-up words)
            condition_on_previous_text=False,
            beam_size=5,
        )
        dur = job.duration or info.duration or 1
        segs = []
        for s in segments:                  # generator: decoding happens as we iterate
            job.progress = min(s.end / dur, 1.0)
            # No extra no_speech_prob filter here: Whisper rates *singing* as
            # "no speech" very often, which dropped every line. faster-whisper
            # already skips segments that are both low-confidence and no-speech.
            if not s.words:
                continue
            segs.append({"words": [{"w": w.word, "s": round(w.start, 3), "e": round(w.end, 3)} for w in s.words]})
        if not job.language:
            job.language = info.language
        lyrics = {"language": info.language, "lines": split_lines(segs)}
        (out / "lyrics.json").write_text(json.dumps(lyrics, ensure_ascii=False), encoding="utf-8")
        (out / "vocals.wav").unlink(missing_ok=True)


class MockPipeline:
    """Same interface, no network or models. Generates a test video, a fake
    instrumental and fake lyrics, with sleeps that mimic real stage times."""

    WORDS = "la la la sing along with me under city lights tonight we never stop the music".split()

    def _sleep(self, job, seconds):
        steps = 20
        for i in range(steps):
            time.sleep(seconds / steps)
            job.progress = (i + 1) / steps

    def download(self, job, out: Path):
        ffmpeg_exe()   # fail early with a clear message if ffmpeg is missing
        job.title = f"Mock song {job.video_id}"
        job.duration = 30
        self._sleep(job, 2)
        # 30 s colour-bar video with a 440 Hz "vocal" tone + 110 Hz "band" tone.
        run_ffmpeg(
            "-f", "lavfi", "-i", "testsrc2=size=1280x720:rate=25:duration=30",
            "-f", "lavfi", "-i", "sine=frequency=440:duration=30",
            "-f", "lavfi", "-i", "sine=frequency=110:duration=30",
            "-filter_complex", "[1:a][2:a]amix=inputs=2[a]", "-map", "0:v", "-map", "[a]",
            "-c:v", "libx264", "-preset", "veryfast", "-pix_fmt", "yuv420p",
            "-c:a", "aac", "-shortest", "-movflags", "+faststart", str(out / "video.mp4"),
        )

    def separate(self, job, out: Path):
        self._sleep(job, 4)
        run_ffmpeg("-f", "lavfi", "-i", "sine=frequency=110:duration=30", "-ac", "2",
                   "-b:a", "128k", str(out / "instrumental.mp3"))
        run_ffmpeg("-f", "lavfi", "-i", "sine=frequency=440:duration=30", "-ac", "2", str(out / "vocals.wav"))

    def transcribe(self, job, out: Path):
        self._sleep(job, 3)
        words, t = [], 2.0
        for i in range(48):
            w = self.WORDS[i % len(self.WORDS)]
            words.append({"w": " " + w, "s": round(t, 2), "e": round(t + 0.4, 2)})
            t += 0.5 + (0.9 if i % 8 == 7 else 0)   # pause after every 8 words
        lyrics = {"language": "en", "lines": split_lines([{"words": words}], max_words=8)}
        (out / "lyrics.json").write_text(json.dumps(lyrics), encoding="utf-8")
        (out / "vocals.wav").unlink(missing_ok=True)


def make_pipeline():
    return MockPipeline() if config.MOCK else RealPipeline()
