"""Job queue: how the server processes many songs at once.

Each song goes through three stages, served by two priority queues with their
own worker pools, so the work overlaps like an assembly line:

    download queue --(DOWNLOAD_WORKERS threads, network)--> model queue
    model queue    --(MODEL_WORKERS threads, GPU/CPU)-----> separate, then
                                                            transcribe, then done

Separation and transcription share one queue because they compete for the
same GPU: one model worker runs whichever waiting job has the best priority.
While songs A and B download, song C can be on the GPU. Heavy work runs in
threads; PyTorch, OpenVINO, CTranslate2 and ffmpeg release Python's GIL, so the
threads really do run in parallel.

Other rules:
  * Dedup / cache: a song is keyed by its YouTube video ID. If it was already
    processed (files on disk) or is already in progress, a new request just
    attaches to it instead of starting again.
  * Priority: lower number = sooner. The song about to play gets priority 0,
    so it jumps ahead of songs further down the playlist.
  * Skip lyrics: a song can stop after vocal removal. Its vocals are kept, so
    lyrics can still be added later by requesting the song with lyrics.
  * Resume: stages whose output is already on disk are not redone.
"""
from __future__ import annotations

import asyncio
import itertools
import json
import logging
import shutil
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

log = logging.getLogger("karaoke.jobs")

STAGES = ["download", "separate", "transcribe", "voices"]   # "voices" only for duets


def _read_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


@dataclass
class Job:
    video_id: str
    url: str
    language: str | None = None
    priority: int = 10
    skip_lyrics: bool = False
    # waiting:<stage> | running:<stage> | done | error
    status: str = "waiting:download"
    progress: float = 0.0          # 0..1 within the current stage
    title: str = ""
    duration: float = 0.0
    error: str = ""
    has_lyrics: bool = False       # lyrics.json exists and has lines
    duet: bool = False             # tag lyric lines male / female
    has_voices: bool = False       # lyrics carry male / female tags
    created: float = field(default_factory=time.time)
    timings: dict = field(default_factory=dict)

    def public(self) -> dict:
        return {
            "video_id": self.video_id,
            "url": self.url,
            "title": self.title,
            "duration": self.duration,
            "language": self.language,
            "priority": self.priority,
            "status": self.status,
            "progress": round(self.progress, 3),
            "error": self.error,
            "timings": self.timings,
            "skip_lyrics": self.skip_lyrics,
            "has_lyrics": self.has_lyrics,
            "duet": self.duet,
            "has_voices": self.has_voices,
            "ready": self.status == "done",
        }


class JobManager:
    def __init__(self, pipeline, data_dir: Path, download_workers: int, model_workers: int):
        self.pipeline = pipeline
        self.data_dir = data_dir
        self.jobs: dict[str, Job] = {}
        self.queues = {"download": asyncio.PriorityQueue(), "model": asyncio.PriorityQueue()}
        self._seq = itertools.count()   # tie-breaker so equal priorities stay FIFO
        # One thread per worker, so a job handed to a pool starts immediately
        # (it never waits in the pool's own FIFO and loses its priority).
        self.download_pool = ThreadPoolExecutor(download_workers, thread_name_prefix="download")
        self.model_pool = ThreadPoolExecutor(model_workers, thread_name_prefix="model")
        self.n_download = download_workers
        self.n_model = model_workers
        self._tasks: list[asyncio.Task] = []

    # ---------- lifecycle ----------
    async def start(self):
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self._load_cache()
        for i in range(self.n_download):
            self._tasks.append(asyncio.create_task(self._worker("download", self.download_pool), name=f"download-{i}"))
        for i in range(self.n_model):
            self._tasks.append(asyncio.create_task(self._worker("model", self.model_pool), name=f"model-{i}"))
        log.info("workers started: %d download, %d model", self.n_download, self.n_model)

    async def stop(self):
        for t in self._tasks:
            t.cancel()
        self.download_pool.shutdown(wait=False, cancel_futures=True)
        self.model_pool.shutdown(wait=False, cancel_futures=True)

    def _load_cache(self):
        """Songs finished in an earlier run are served straight from disk."""
        for d in self.data_dir.iterdir():
            if not d.is_dir():
                continue
            meta, lyrics = _read_json(d / "meta.json"), _read_json(d / "lyrics.json")
            if not meta or not (d / "instrumental.mp3").exists():
                continue
            has_lyrics = bool(lyrics.get("lines"))
            skipped = bool(lyrics.get("skipped"))
            if not has_lyrics and not skipped:
                continue   # empty lyrics (old bug): reprocess when requested again
            self.jobs[d.name] = Job(
                video_id=d.name, url=meta.get("url", ""), title=meta.get("title", ""),
                duration=meta.get("duration", 0), language=meta.get("language"),
                status="done", progress=1.0, timings=meta.get("timings", {}),
                skip_lyrics=skipped, has_lyrics=has_lyrics,
                duet=bool(lyrics.get("duet")), has_voices=bool(lyrics.get("duet")),
            )
        log.info("cache: %d songs already processed", len(self.jobs))

    # ---------- public API ----------
    def submit(self, video_id: str, url: str, language: str | None, priority: int,
               skip_lyrics: bool = False, duet: bool = False) -> Job:
        duet = duet and not skip_lyrics          # voice tags need lyrics
        job = self.jobs.get(video_id)
        if job and job.status != "error":
            # Already cached or in progress: attach to it, maybe raise its priority.
            if language and not job.has_lyrics:
                job.language = language
            more_work = False
            if not skip_lyrics and job.skip_lyrics:
                job.skip_lyrics = False          # lyrics were skipped before but are wanted now
                more_work = True
            if duet:
                # Tag (or re-tag: it only takes a couple of seconds, and picks up
                # improvements to the tagger). An in-progress job picks this up
                # after its lyrics step.
                job.duet = True
                more_work = True
            if more_work and job.status == "done":
                job.priority = priority
                self._enqueue(job, self._resume_stage(video_id, want_voices=job.duet))
                return job
            self.set_priority(video_id, priority)
            return job
        job = Job(video_id=video_id, url=url, language=language or None, priority=priority,
                  skip_lyrics=skip_lyrics, duet=duet)
        self.jobs[video_id] = job
        info = _read_json(self.data_dir / video_id / "info.json")
        job.title, job.duration = info.get("title", ""), info.get("duration", 0)
        stage = self._resume_stage(video_id, want_voices=duet)
        if stage != "download":
            log.info("resuming %s at %s", video_id, stage)
        self._enqueue(job, stage)
        return job

    def _resume_stage(self, video_id: str, want_voices: bool = False) -> str:
        """First stage whose output isn't on disk yet."""
        out = self.data_dir / video_id
        if (want_voices and (out / "vocals16k.flac").exists()
                and _read_json(out / "lyrics.json").get("lines")):
            return "voices"        # lyrics done; only the male/female tags are missing
        if (out / "vocals.wav").exists() and (out / "instrumental.mp3").exists():
            return "transcribe"
        if (out / "video.mp4").exists() and (out / "audio.wav").exists():
            return "separate"
        return "download"   # reuses video.mp4 if it is already there

    def set_priority(self, video_id: str, priority: int):
        job = self.jobs.get(video_id)
        if not job or priority >= job.priority:
            return
        job.priority = priority
        # asyncio.PriorityQueue can't reorder an item in place, so push a second
        # entry with the better priority. Whichever copy a worker pops second is
        # skipped because the job is no longer waiting for that stage.
        if job.status.startswith("waiting:"):
            self._enqueue(job, job.status.split(":", 1)[1])

    def queue_depths(self) -> dict:
        waiting = {s: 0 for s in STAGES}
        running = {s: 0 for s in STAGES}
        for j in self.jobs.values():
            kind, _, stage = j.status.partition(":")
            if kind == "waiting":
                waiting[stage] += 1
            elif kind == "running":
                running[stage] += 1
        return {"waiting": waiting, "running": running}

    def history(self) -> list[dict]:
        """Every song with a downloaded video on disk, most recently played first."""
        items = []
        for d in self.data_dir.iterdir():
            if not d.is_dir() or not (d / "video.mp4").exists():
                continue
            meta, info, lyrics = _read_json(d / "meta.json"), _read_json(d / "info.json"), _read_json(d / "lyrics.json")
            job = self.jobs.get(d.name)
            items.append({
                "video_id": d.name,
                "title": (job and job.title) or meta.get("title") or info.get("title") or "",
                "duration": meta.get("duration") or info.get("duration") or 0,
                "has_audio": (d / "instrumental.mp3").exists(),
                "has_lyrics": bool(lyrics.get("lines")),
                "lyrics_skipped": bool(lyrics.get("skipped")),
                "duet": bool(lyrics.get("duet")),
                "status": job.status if job else ("done" if (d / "instrumental.mp3").exists() else "partial"),
                "last_played": meta.get("last_played", 0),
                "added": (d / "video.mp4").stat().st_mtime,
            })
        items.sort(key=lambda x: (x["last_played"], x["added"]), reverse=True)
        return items

    def delete(self, video_id: str):
        """Delete a song's files for good. Raises RuntimeError if it is being
        processed right now, OSError if its files stay locked (e.g. still playing)."""
        job = self.jobs.get(video_id)
        if job and job.status.startswith("running:"):
            raise RuntimeError("being processed")
        # A job still waiting in a queue is dropped: its queue entry is skipped later.
        self.jobs.pop(video_id, None)
        folder = self.data_dir / video_id
        for attempt in range(10):
            try:
                if folder.exists():
                    shutil.rmtree(folder)
                log.info("deleted %s", video_id)
                return
            except OSError:
                # On Windows a file can't be deleted while it is being streamed;
                # the TV lets go within a second once the song leaves the playlist.
                if attempt == 9:
                    raise
                time.sleep(0.5)

    def mark_played(self, video_id: str):
        meta_path = self.data_dir / video_id / "meta.json"
        meta = _read_json(meta_path)
        if meta:
            meta["last_played"] = time.time()
            meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")

    # ---------- internals ----------
    def _enqueue(self, job: Job, stage: str):
        job.status = f"waiting:{stage}"
        job.progress = 0.0
        q = self.queues["download" if stage == "download" else "model"]
        q.put_nowait((job.priority, next(self._seq), job.video_id, stage))

    async def _worker(self, queue_name: str, pool: ThreadPoolExecutor):
        loop = asyncio.get_running_loop()
        q = self.queues[queue_name]
        while True:
            _prio, _seq, vid, stage = await q.get()
            job = self.jobs.get(vid)
            if not job or job.status != f"waiting:{stage}":
                continue   # stale duplicate entry (see set_priority)
            fn = getattr(self.pipeline, stage)
            job.status = f"running:{stage}"
            out_dir = self.data_dir / vid
            out_dir.mkdir(parents=True, exist_ok=True)
            t0 = time.time()
            try:
                # The heavy work runs in a thread so the web server stays responsive.
                await loop.run_in_executor(pool, fn, job, out_dir)
            except Exception as e:  # noqa: BLE001
                log.error("%s failed for %s: %s", stage, vid, e)
                log.debug(traceback.format_exc())
                job.status = "error"
                job.error = f"{stage}: {e}"
                continue
            job.timings[stage] = round(time.time() - t0, 1)
            if stage == "separate" and job.skip_lyrics:
                # Stop here. vocals.wav stays on disk so lyrics can be added later.
                (out_dir / "lyrics.json").write_text(json.dumps({"lines": [], "skipped": True}), encoding="utf-8")
                job.has_lyrics = False
                self._finish(job, out_dir)
            elif stage == "transcribe":
                job.has_lyrics = bool(_read_json(out_dir / "lyrics.json").get("lines"))
                if job.duet and job.has_lyrics:
                    self._enqueue(job, "voices")
                else:
                    self._finish(job, out_dir)
            elif stage == "voices":
                job.has_voices = bool(_read_json(out_dir / "lyrics.json").get("duet"))
                self._finish(job, out_dir)
            else:
                self._enqueue(job, STAGES[STAGES.index(stage) + 1])

    def _finish(self, job: Job, out_dir: Path):
        job.status = "done"
        job.progress = 1.0
        old = _read_json(out_dir / "meta.json")
        meta = {k: v for k, v in job.public().items() if k in ("url", "title", "duration", "language", "duet")}
        meta["timings"] = {**old.get("timings", {}), **job.timings}
        if "last_played" in old:
            meta["last_played"] = old["last_played"]
        (out_dir / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")
        log.info("done %s (%s)%s %s", job.video_id, job.title, " [no lyrics]" if job.skip_lyrics else "", job.timings)
