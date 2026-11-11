"""Web server: one shared room with a playlist, plus the processing API.

Open http://<this-pc>:8000 on the TV/PC (player) and
http://<this-pc>:8000/?remote on phones (add songs only).
"""
from __future__ import annotations

import asyncio
import faulthandler
import logging
import io
import re
import socket
import sys
import threading
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from . import config
from . import lyrics_sync
from .jobs import JobManager, _read_json
from .pipeline import check_ffmpeg_for_ytdlp, check_js_runtime, extract_video_id, make_pipeline, search_youtube

# If a native library ever crashes the process (e.g. "segmentation fault"),
# print where it happened instead of exiting silently. Not on Windows: there it
# also prints errors that libraries trigger and handle on purpose, which would
# look like crashes but aren't.
if sys.platform != "win32":
    faulthandler.enable()

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("karaoke")

manager = JobManager(make_pipeline(), config.DATA_DIR, config.DOWNLOAD_WORKERS, config.MODEL_WORKERS)
playlist: list[dict] = []      # [{"video_id", "singer"}]; playlist[0] is now playing
# Player settings shared by the TV and every phone. "rev" goes up on each change
# so a page can ignore a stale poll that was sent before its own change landed.
player = {"mode": "inst", "show_lyrics": True, "fullscreen": False, "paused": False,
          "lyrics_scale": 1.2,
          "started": False,         # TV player started (Start on the TV, or from a phone)
          "sound_blocked": False,   # TV browser blocks sound until someone clicks on the TV
          "rev": 0}   # mode: "inst" (backing) | "orig"


def reprioritize():
    """Position in the playlist = processing priority. The song playing now
    (and the next one) are always processed first."""
    for i, item in enumerate(playlist):
        manager.set_priority(item["video_id"], i)


@asynccontextmanager
async def lifespan(app: FastAPI):
    log.info("karaoke prototype v3 | mode: %s | data: %s", "MOCK" if config.MOCK else "REAL", config.DATA_DIR)
    try:
        if config.MOCK:
            raise StopIteration
        log.info("ffmpeg check OK, yt-dlp can merge video+audio using: %s", check_ffmpeg_for_ytdlp())
    except StopIteration:
        pass
    except Exception as e:  # noqa: BLE001
        log.error("ffmpeg check FAILED: %s  (downloads will fail until this is fixed)", e)
    if not config.MOCK:
        # Separate from the ffmpeg check, so a problem here is reported as itself.
        try:
            rt = check_js_runtime()
            if rt:
                log.info("JavaScript runtime for YouTube: %s", rt)
            else:
                log.warning("no JavaScript runtime found (Deno, Node.js, QuickJS or Bun): YouTube downloads "
                            "may fail or miss formats. macOS/Linux: curl -fsSL https://deno.land/install.sh | sh  "
                            "Windows: winget install DenoLand.Deno")
        except Exception as e:  # noqa: BLE001
            log.warning("could not check for a JavaScript runtime: %s", e)
    await manager.start()
    if not config.MOCK:
        # Load the vocal separator in the background so the first song doesn't wait for it.
        async def preload():
            try:
                await asyncio.get_running_loop().run_in_executor(manager.model_pool, manager.pipeline.separator)
            except Exception as e:  # noqa: BLE001
                log.error("could not load vocal separator: %s", e)
        asyncio.create_task(preload())
    yield
    await manager.stop()


app = FastAPI(title="Karaoke prototype", lifespan=lifespan)
config.DATA_DIR.mkdir(parents=True, exist_ok=True)
# Serves video.mp4 / instrumental.mp3 / lyrics.json (with Range support for seeking).
app.mount("/media", StaticFiles(directory=config.DATA_DIR), name="media")


class AddSong(BaseModel):
    url: str
    language: str | None = None   # e.g. "yue", "zh", "en", "ja", "ko"; empty = auto
    singer: str | None = None
    skip_lyrics: bool = False     # True = only remove vocals (faster), no lyrics
    duet: bool = False            # male + female duet: tag lyric lines M / F
    lyrics_query: str | None = None  # song name (+ singer) to look the lyrics up by


class PlayerUpdate(BaseModel):
    mode: str | None = None          # "inst" = backing track, "orig" = original vocals
    show_lyrics: bool | None = None
    fullscreen: bool | None = None   # TV video fills the whole screen
    paused: bool | None = None       # pause / resume the song on the TV
    lyrics_scale: float | None = None  # lyrics size on the TV, 1.0 = standard (0.5 - 3.0)
    started: bool | None = None        # start the TV player (or, from the TV, reset on load)
    sound_blocked: bool | None = None  # reported by the TV


class Process(BaseModel):
    url: str
    language: str | None = None
    priority: int = 10
    skip_lyrics: bool = False
    duet: bool = False


def lan_ip() -> str:
    """This computer's address on the local network (the interface used for
    the default route). Connecting a UDP socket sends no packets."""
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect(("10.255.255.255", 1))
            return s.getsockname()[0]
    except OSError:
        try:
            return socket.gethostbyname(socket.gethostname())
        except OSError:
            return "127.0.0.1"


def join_url(request: Request) -> str:
    """Address phones should open. The TV page is often opened as localhost,
    which only works on this computer, so use the network address instead."""
    if config.PUBLIC_URL:
        return config.PUBLIC_URL
    host, port = request.url.hostname or "localhost", request.url.port
    if host in ("localhost", "127.0.0.1", "::1", "0.0.0.0"):
        host = lan_ip()
    return f"{request.url.scheme}://{host}" + (f":{port}" if port else "")


@app.get("/api/join")
def join(request: Request):
    # The QR code adds ?start=1: a phone that scans it also starts the TV player.
    url = join_url(request)
    return {"url": url, "qr_url": url + "/?start=1"}


@app.get("/qr.svg")
def qr_code(request: Request, start: int = 1):
    """QR code of the join address, drawn here so it works without internet.
    start=1 (the TV's code) also starts the TV player; start=0 (a phone's
    "Invite" code) only joins."""
    try:
        import segno
    except ImportError:
        raise HTTPException(404, "QR codes need the 'segno' package: pip install segno")
    buf = io.BytesIO()
    segno.make(join_url(request) + ("/?start=1" if start else "/"), error="m").save(
        buf, kind="svg", scale=10, border=2, dark="#000000", light="#ffffff", xmldecl=False)
    return Response(buf.getvalue(), media_type="image/svg+xml", headers={"Cache-Control": "no-store"})


@app.get("/")
def index():
    # no-cache: browsers must check for a newer page on every load, so an
    # updated page is never served from a stale cache.
    return FileResponse(config.STATIC_DIR / "index.html", headers={"Cache-Control": "no-cache"})


@app.post("/api/queue")
def add_to_playlist(body: AddSong):
    vid = extract_video_id(body.url)
    if not vid:
        raise HTTPException(400, "not a YouTube link")
    query = (body.lyrics_query or "").strip()
    job = manager.submit(vid, f"https://www.youtube.com/watch?v={vid}", body.language,
                         priority=len(playlist), skip_lyrics=body.skip_lyrics, duet=body.duet,
                         lyrics_query=query)
    if query and job.status == "done":
        # Processed before: look the lyrics up now with the new name, in the
        # background (only replaces the AI's own lyrics).
        d = config.DATA_DIR / vid
        if _read_json(d / "lyrics.json").get("source", "ai") == "ai" and lyrics_sync.whisper_lyrics(d):
            _save_query(d, query)
            threading.Thread(target=lyrics_sync.auto_online, args=(d, job.title, job.duration, query),
                             daemon=True).start()
    playlist.append({"video_id": vid, "singer": (body.singer or "").strip()[:20]})
    reprioritize()
    return {"added": vid, "position": len(playlist) - 1, "job": job.public()}


@app.post("/api/next")
def next_song():
    if playlist:
        manager.mark_played(playlist.pop(0)["video_id"])
    if player["paused"]:
        player["paused"] = False      # the next song starts playing
        player["rev"] += 1
    reprioritize()
    return {"ok": True}


@app.post("/api/player")
def update_player(body: PlayerUpdate):
    """Change the TV's audio mode, lyrics visibility or fullscreen or pause, from the TV or a phone."""
    if body.mode is not None:
        if body.mode not in ("inst", "orig"):
            raise HTTPException(400, "mode must be 'inst' or 'orig'")
        player["mode"] = body.mode
    if body.show_lyrics is not None:
        player["show_lyrics"] = body.show_lyrics
    if body.fullscreen is not None:
        player["fullscreen"] = body.fullscreen
    if body.paused is not None:
        player["paused"] = body.paused
    if body.lyrics_scale is not None:
        player["lyrics_scale"] = round(min(3.0, max(0.5, body.lyrics_scale)), 2)
    if body.started is not None:
        player["started"] = body.started
    if body.sound_blocked is not None:
        player["sound_blocked"] = body.sound_blocked
    player["rev"] += 1
    return player


@app.post("/api/playlist/{index}/remove")
def remove(index: int, video_id: str | None = None):
    """Take a song off the playlist (its files stay; it remains in History).
    video_id guards against the list having changed since the page drew it."""
    if not 0 <= index < len(playlist):
        raise HTTPException(404)
    if video_id and playlist[index]["video_id"] != video_id:
        raise HTTPException(409, "the playlist changed, try again")
    playlist.pop(index)
    reprioritize()
    return {"ok": True}


@app.post("/api/playlist/{index}/up")
def move_up(index: int):
    if not 1 <= index < len(playlist):
        raise HTTPException(404)
    playlist[index - 1], playlist[index] = playlist[index], playlist[index - 1]
    reprioritize()
    return {"ok": True}


@app.post("/api/playlist/{index}/top")
def move_top(index: int, video_id: str | None = None):
    """Play a song next: move it to just below the song playing now (#0)."""
    if not 1 <= index < len(playlist):
        raise HTTPException(404)
    if video_id and playlist[index]["video_id"] != video_id:
        raise HTTPException(409, "the playlist changed, try again")
    playlist.insert(1, playlist.pop(index))
    reprioritize()
    return {"ok": True}


@app.get("/api/state")
def state():
    items = []
    for item in playlist:
        job = manager.jobs.get(item["video_id"])
        lj = config.DATA_DIR / item["video_id"] / "lyrics.json"
        items.append({**item, **(job.public() if job else {}),
                      # changes when the lyrics are replaced, so the TV reloads them
                      "lyrics_rev": lj.stat().st_mtime if lj.exists() else 0})
    return {
        "playlist": items,
        "workers": {"download": config.DOWNLOAD_WORKERS, "model": config.MODEL_WORKERS},
        "queues": manager.queue_depths(),
        "player": player,
        "mock": config.MOCK,
        "voice_labels": config.VOICE_LABELS,
    }


# ---- direct processing API (no playlist), handy for scripts and load tests ----
@app.post("/api/process")
def process(body: Process):
    vid = extract_video_id(body.url)
    if not vid:
        raise HTTPException(400, "not a YouTube link")
    return manager.submit(vid, f"https://www.youtube.com/watch?v={vid}", body.language, body.priority,
                          skip_lyrics=body.skip_lyrics, duet=body.duet).public()


@app.delete("/api/songs/{video_id}")
async def delete_song(video_id: str):
    """Delete a downloaded song for good (video, backing track, lyrics)."""
    if not re.fullmatch(r"[A-Za-z0-9_-]{11}", video_id):
        raise HTTPException(404)
    job = manager.jobs.get(video_id)
    if job and job.status.startswith("running:"):
        raise HTTPException(409, "This song is being processed right now. Delete it when it has finished.")
    if not job and not (config.DATA_DIR / video_id).exists():
        raise HTTPException(404, "song not found")
    # Take it off the playlist first, so the TV stops playing it and releases its files.
    before = len(playlist)
    playlist[:] = [p for p in playlist if p["video_id"] != video_id]
    if len(playlist) != before:
        reprioritize()
    try:
        await asyncio.get_running_loop().run_in_executor(None, manager.delete, video_id)
    except RuntimeError:
        raise HTTPException(409, "This song is being processed right now. Delete it when it has finished.")
    except OSError:
        raise HTTPException(409, "Its files are still in use. Try again in a moment.")
    return {"deleted": video_id}


@app.get("/api/search")
async def search(q: str, limit: int = 30):
    """Search YouTube by song name. Each result says whether that song is
    already processed (ready to sing) or already in the playlist."""
    q = q.strip()
    if not q:
        return []
    try:
        results = await asyncio.get_running_loop().run_in_executor(
            None, search_youtube, q, max(1, min(limit, 50)))
    except Exception as e:  # noqa: BLE001
        log.warning("search failed for %r: %s", q, e)
        raise HTTPException(502, "YouTube search failed. Check the internet connection, or update yt-dlp.")
    in_playlist = {item["video_id"] for item in playlist}
    out = []
    for r in results:
        job = manager.jobs.get(r["video_id"])
        ready = (job.status == "done") if job else (config.DATA_DIR / r["video_id"] / "instrumental.mp3").exists()
        out.append({**r, "ready": ready, "in_playlist": r["video_id"] in in_playlist})
    return out


# ---- lyrics: real lyric text (LRCLIB or pasted), timed with Whisper ----
def _song_dir(video_id: str):
    if not re.fullmatch(r"[A-Za-z0-9_-]{11}", video_id):
        raise HTTPException(404)
    d = config.DATA_DIR / video_id
    if not (d / "video.mp4").exists():
        raise HTTPException(404, "song not found")
    return d


def _song_title(video_id: str, d) -> tuple[str, float]:
    job = manager.jobs.get(video_id)
    info, meta = _read_json(d / "info.json"), _read_json(d / "meta.json")
    title = (job and job.title) or meta.get("title") or info.get("title") or ""
    duration = (job and job.duration) or meta.get("duration") or info.get("duration") or None
    return title, duration


def _save_query(d, query: str):
    """Remember the name a song's lyrics were looked up by (meta.json)."""
    meta = _read_json(d / "meta.json")
    if meta and query and meta.get("lyrics_query") != query:
        import json
        meta["lyrics_query"] = query[:100]
        (d / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")
    job = manager.jobs.get(d.name)
    if job and query:
        job.lyrics_query = query[:100]


@app.get("/api/lyrics/{video_id}")
def get_lyrics(video_id: str):
    """Current lyrics as text (one line per row) and where they came from."""
    d = _song_dir(video_id)
    title, _ = _song_title(video_id, d)
    lyr = _read_json(d / "lyrics.json")
    return {
        "video_id": video_id, "title": title,
        "source": lyr.get("source", "ai") if lyr.get("lines") else ("skipped" if lyr.get("skipped") else "none"),
        "source_info": lyr.get("source_info") or {},
        "text": "\n".join(ln.get("text", "") for ln in lyr.get("lines", [])),
        "has_ai": lyrics_sync.whisper_lyrics(d) is not None,
        "query": _read_json(d / "meta.json").get("lyrics_query")
                 or (manager.jobs.get(video_id) and manager.jobs[video_id].lyrics_query)
                 or (lyrics_sync.guess_queries(title) or [title])[0],
    }


@app.get("/api/lyrics/{video_id}/search")
async def search_lyrics(video_id: str, q: str = ""):
    """Look up lyrics on LRCLIB. Best matches for this song first."""
    d = _song_dir(video_id)
    title, duration = _song_title(video_id, d)
    q = q.strip()
    loop = asyncio.get_running_loop()
    found, seen = [], set()
    try:
        # A typed name: plain search plus song/singer searches. Otherwise guesses from the title.
        batches = [lambda: lyrics_sync.search_user(q)] if q else \
            [(lambda g=g: lyrics_sync.search(g)) for g in lyrics_sync.guess_queries(title)[:3]]
        for fetch in batches:
            for c in await loop.run_in_executor(None, fetch):
                if c["id"] not in seen and not c["instrumental"] and (c["synced"] or c["plain"]):
                    seen.add(c["id"])
                    found.append(c)
    except Exception as e:  # noqa: BLE001
        log.warning("lyrics search failed: %s", e)
        raise HTTPException(502, "Could not reach the lyrics site (lrclib.net). Check the internet connection.")
    title_n = lyrics_sync.norm(f"{title} {q}")
    for c in found:
        c["score"] = lyrics_sync._score(c, title_n, duration) or 0
    found.sort(key=lambda c: -c["score"])
    return [{"id": c["id"], "track": c["track"], "artist": c["artist"], "album": c["album"],
             "duration": c["duration"], "synced": bool(c["synced"]), "likely": c["score"] >= 4,
             "text": lyrics_sync.candidate_text(c)} for c in found[:15]]


class LyricsUpdate(BaseModel):
    text: str | None = None        # pasted lyrics (plain, or LRC with [mm:ss.xx] times)
    lrclib_id: int | None = None   # use this LRCLIB entry
    use_ai: bool = False           # switch back to the AI's own lyrics
    query: str | None = None       # the name searched for (remembered for next time)


@app.post("/api/lyrics/{video_id}")
async def set_lyrics(video_id: str, body: LyricsUpdate):
    """Replace a song's lyrics. The text is timed against what the AI heard."""
    d = _song_dir(video_id)
    job = manager.jobs.get(video_id)
    if job and job.status.startswith("running:"):
        raise HTTPException(409, "This song is being processed right now. Try again when it has finished.")
    title, duration = _song_title(video_id, d)
    was_duet = bool(_read_json(d / "lyrics.json").get("duet") or _read_json(d / "meta.json").get("duet")
                    or (job and job.duet))
    loop = asyncio.get_running_loop()

    def work():
        if body.use_ai:
            if not lyrics_sync.use_ai(d):
                raise ValueError("This song has no AI lyrics (it was added with Skip lyrics).")
            return {}
        if body.lrclib_id is not None:
            c = lyrics_sync.get_by_id(body.lrclib_id)
            return lyrics_sync.apply_text(d, lyrics_sync.candidate_text(c), duration,
                                          {"kind": "lrclib", "id": c["id"], "track": c["track"],
                                           "artist": c["artist"], "synced": bool(c["synced"])})
        text = (body.text or "").strip()[:20000]
        if not text:
            raise ValueError("Paste some lyrics first.")
        return lyrics_sync.apply_text(d, text, duration, {"kind": "pasted"})

    try:
        stats = await loop.run_in_executor(None, work)
    except ValueError as e:
        raise HTTPException(400, str(e))
    except Exception as e:  # noqa: BLE001
        log.warning("lyrics update failed for %s: %s", video_id, e)
        raise HTTPException(502, f"Could not update the lyrics: {e}")
    if body.query and body.lrclib_id is not None:
        _save_query(d, body.query.strip())
    lyr = _read_json(d / "lyrics.json")
    if job:
        job.has_lyrics = bool(lyr.get("lines"))
        job.has_voices = False
    # A duet song gets its M/F tags again for the new lines.
    if lyr.get("lines") and was_duet and (d / "vocals16k.flac").exists():
        manager.submit(video_id, f"https://www.youtube.com/watch?v={video_id}", None, priority=0, duet=True)
    return {"ok": True, "source": lyr.get("source", "ai"), "lines": len(lyr.get("lines", [])), **(stats or {})}


@app.get("/api/history")
def history():
    """All songs downloaded before (finished or not), most recently played first."""
    in_playlist = {item["video_id"] for item in playlist}
    return [{**h, "in_playlist": h["video_id"] in in_playlist} for h in manager.history()]


@app.get("/api/jobs")
def jobs():
    return sorted((j.public() for j in manager.jobs.values()), key=lambda j: j["priority"])


@app.get("/api/jobs/{video_id}")
def job(video_id: str):
    j = manager.jobs.get(video_id)
    if not j:
        raise HTTPException(404)
    return j.public()
