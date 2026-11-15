"""Chinese singer and song catalogue, from MusicBrainz (see decision.md).

* Singers: every artist from Hong Kong, Taiwan, mainland China and Macau,
  downloaded from the MusicBrainz web service by "Update" and stored locally
  in data/catalog.sqlite3 (works offline afterwards).
* Songs: fetched for one singer the first time that singer is opened, then
  stored locally too. "Update" also refreshes the songs already stored.
* Popularity: listener counts from ListenBrainz (same IDs) put the best-known
  singers first, and each singer's most-played songs first.

MusicBrainz asks for at most about one request per second and a User-Agent
naming the application; both are respected here. Its core data is CC0.
"""
from __future__ import annotations

import json
import logging
import random
import re
import sqlite3
import threading
import time
import urllib.error
import urllib.parse
import urllib.request

from . import config

log = logging.getLogger("karaoke.catalog")

API = "https://musicbrainz.org/ws/2"
# Popularity (how many people listen) comes from ListenBrainz, MusicBrainz's
# sister project: same artist / recording IDs, free, no key.
LB_API = "https://api.listenbrainz.org/1"
LB_BATCH = 100                                     # artists per popularity request
USER_AGENT = "KaraokePrototype/1.0 ( https://github.com/kseanra/ktv )"
REGIONS = {"HK": "Hong Kong", "TW": "Taiwan", "CN": "Mainland China", "MO": "Macau",
           "SG": "Singapore", "MY": "Malaysia"}
# Singapore and Malaysia have many Mandopop singers (JJ Lin, Stefanie Sun,
# Fish Leong ...), but also Malay, Tamil and English ones: from these, only
# artists with a Chinese name are kept.
CHINESE_NAME_ONLY = {"SG", "MY"}
KEEP_TYPES = {"Person", "Group", "Choir", None}   # leave out orchestras, characters, "Other"
MAX_RECORDINGS = 2500                              # per singer (25 requests)
DB_PATH = config.DATA_DIR / "catalog.sqlite3"

_CJK = re.compile(r"[㐀-鿿豈-﫿]")
_LATIN = re.compile(r"[A-Za-z]")


# ----------------------------------------------------------------- helpers
_cc = None


def _trad(s: str) -> str:
    global _cc
    if _cc is None:
        try:
            import opencc
            _cc = opencc.OpenCC("s2t")
        except Exception:  # noqa: BLE001
            _cc = False
    return _cc.convert(s) if _cc else s


def search_key(s: str) -> str:
    """For searching: Traditional characters, lower case, no spaces/punctuation."""
    return re.sub(r"[\s\-_.,'·・/&()（）]+", "", _trad(s or "").lower())


_BRACKETS = re.compile(r"\s*[\(\[（【「〈<][^\)\]）】」〉>]*[\)\]）】」〉>]")
_SUFFIX = re.compile(r"\s+[-–—]\s+.*$")


def title_key(title: str) -> str:
    """Songs that only differ by a note like "(Live)", "(粵語版)" or
    " - 2015 Remaster" count as one song."""
    t = _SUFFIX.sub("", _BRACKETS.sub("", title or "")).strip() or (title or "")
    return search_key(t)


# ----------------------------------------------------------------- MusicBrainz
_rate_lock = threading.Lock()
_last_call = [0.0]


def _get(path: str, params: dict, tries: int = 4) -> dict:
    """One MusicBrainz request, at most about one per second."""
    url = f"{API}/{path}?" + urllib.parse.urlencode({**params, "fmt": "json"})
    for attempt in range(tries):
        with _rate_lock:
            wait = 1.1 - (time.time() - _last_call[0])
            if wait > 0:
                time.sleep(wait)
            _last_call[0] = time.time()
        try:
            req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT, "Accept": "application/json"})
            with urllib.request.urlopen(req, timeout=30) as r:
                return json.loads(r.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            if e.code in (429, 503) and attempt < tries - 1:     # too fast / busy: back off
                time.sleep(2 + 3 * attempt)
                continue
            raise
        except (urllib.error.URLError, TimeoutError):
            if attempt < tries - 1:
                time.sleep(2 + 3 * attempt)
                continue
            raise
    raise RuntimeError("MusicBrainz did not answer")


def _names(a: dict) -> tuple[str, str]:
    """(Chinese name, English/Latin name) of an artist, from its name and aliases."""
    name = a.get("name") or ""
    aliases = a.get("aliases") or []

    def pick(test, locales):
        for al in sorted(aliases, key=lambda x: (not x.get("primary"), x.get("type") != "Artist name")):
            if (al.get("locale") or "").split("_")[0] in locales and test(al.get("name", "")):
                return al["name"]
        for al in aliases:
            if test(al.get("name", "")):
                return al["name"]
        return ""
    zh = name if _CJK.search(name) else pick(lambda s: bool(_CJK.search(s)), {"zh", "yue"})
    en = name if (_LATIN.search(name) and not _CJK.search(name)) else pick(
        lambda s: bool(_LATIN.search(s)) and not _CJK.search(s), {"en"})
    return zh, en


def _singer_row(a: dict, region: str) -> tuple | None:
    if a.get("type") not in KEEP_TYPES:
        return None
    zh, en = _names(a)
    aliases = [al.get("name", "") for al in (a.get("aliases") or [])]
    # A rough "how well known" score: well-known artists have more aliases and tags.
    fame = len(aliases) * 2 + sum(int(t.get("count") or 0) for t in (a.get("tags") or []))
    keys = " ".join(dict.fromkeys(search_key(x) for x in [a.get("name", ""), a.get("sort-name", ""), *aliases] if x))
    return (a["id"], a.get("name") or zh or en, a.get("sort-name") or "", zh, en, region,
            a.get("type") or "", a.get("disambiguation") or "", fame, keys)


# ----------------------------------------------------------------- database
_db_lock = threading.Lock()


def _db() -> sqlite3.Connection:
    con = sqlite3.connect(DB_PATH, timeout=30, check_same_thread=False)
    con.row_factory = sqlite3.Row
    con.executescript("""
        CREATE TABLE IF NOT EXISTS singers (
            mbid TEXT PRIMARY KEY, name TEXT, sort_name TEXT, zh TEXT, en TEXT, region TEXT,
            type TEXT, disambiguation TEXT, fame INTEGER, keys TEXT);
        CREATE INDEX IF NOT EXISTS singers_region ON singers(region);
        CREATE TABLE IF NOT EXISTS songs (
            singer TEXT, title TEXT, key TEXT, versions INTEGER, length_ms INTEGER, first_release TEXT,
            PRIMARY KEY (singer, key));
        CREATE TABLE IF NOT EXISTS fetched (singer TEXT PRIMARY KEY, at REAL, recordings INTEGER);
        CREATE TABLE IF NOT EXISTS meta (k TEXT PRIMARY KEY, v TEXT);
        CREATE TABLE IF NOT EXISTS popularity (mbid TEXT PRIMARY KEY, listeners INTEGER, listens INTEGER);
        CREATE TABLE IF NOT EXISTS added (mbid TEXT PRIMARY KEY, row TEXT);  -- singers added by hand (kept through updates)
        CREATE TABLE IF NOT EXISTS lists (id INTEGER PRIMARY KEY AUTOINCREMENT, name TEXT UNIQUE, created REAL);
        CREATE TABLE IF NOT EXISTS list_members (list_id INTEGER, singer TEXT, added REAL, PRIMARY KEY (list_id, singer));
    """)
    try:                                   # added later: plays per song
        con.execute("ALTER TABLE songs ADD COLUMN listens INTEGER")
    except sqlite3.OperationalError:
        pass
    return con


def _meta(con, k, v=None):
    if v is None:
        r = con.execute("SELECT v FROM meta WHERE k=?", (k,)).fetchone()
        return r["v"] if r else None
    con.execute("INSERT OR REPLACE INTO meta VALUES (?, ?)", (k, str(v)))


# ----------------------------------------------------------------- public: read
def list_singers(q: str = "", region: str = "", offset: int = 0, limit: int = 60, list_id: int | None = None) -> dict:
    with _db_lock:
        con = _db()
        where, args = [], []
        if list_id is not None:                 # a singer list made by the user
            where.append("s.mbid IN (SELECT singer FROM list_members WHERE list_id = ?)")
            args.append(list_id)
        if region in REGIONS:
            where.append("s.region = ?")
            args.append(region)
        k = search_key(q)
        order_args = []
        match = ""
        if k:
            where.append("s.keys LIKE ?")
            args.append(f"%{k}%")
            # Best matches first: a whole name or alias equal to the search,
            # then one starting with it, then anything containing it.
            match = ("CASE WHEN ' ' || s.keys || ' ' LIKE ? THEN 0 "
                     "WHEN ' ' || s.keys LIKE ? THEN 1 ELSE 2 END, ")
            order_args = [f"% {k} %", f"% {k}%"]
        w = ("WHERE " + " AND ".join(where)) if where else ""
        total = con.execute(f"SELECT COUNT(*) FROM singers s {w}", args).fetchone()[0]
        rows = con.execute(
            f"""SELECT s.*, f.at AS fetched_at, p.listeners, p.listens,
                       (SELECT COUNT(*) FROM songs WHERE singer = s.mbid) AS songs
                FROM singers s LEFT JOIN fetched f ON f.singer = s.mbid LEFT JOIN popularity p ON p.mbid = s.mbid {w}
                ORDER BY {match}COALESCE(p.listeners, -1) DESC, COALESCE(p.listens, -1) DESC, s.fame DESC, s.sort_name
                LIMIT ? OFFSET ?""",
            [*args, *order_args, max(1, min(limit, 200)), max(0, offset)]).fetchall()
        updated = _meta(con, "singers_updated")
        con.close()
    return {"total": total, "updated": float(updated) if updated else None,
            "items": [{"mbid": r["mbid"], "name": r["name"], "zh": r["zh"], "en": r["en"], "region": r["region"],
                       "type": r["type"], "disambiguation": r["disambiguation"],
                       "listeners": r["listeners"],
                       "songs": r["songs"] if r["fetched_at"] else None} for r in rows]}


def get_singer(mbid: str) -> dict | None:
    with _db_lock:
        con = _db()
        r = con.execute("SELECT * FROM singers WHERE mbid=?", (mbid,)).fetchone()
        con.close()
    return dict(r) if r else None


def songs(mbid: str, refresh: bool = False) -> dict:
    """A singer's songs: from the local store, or fetched now if not stored yet."""
    singer = get_singer(mbid)
    if not singer:
        raise KeyError(mbid)
    if is_local(mbid):                      # added by hand, not on MusicBrainz: no song list
        return {"singer": {k: singer[k] for k in ("mbid", "name", "zh", "en", "region", "type", "disambiguation")},
                "fetched": None, "local": True, "songs": []}
    with _db_lock:
        con = _db()
        f = con.execute("SELECT at FROM fetched WHERE singer=?", (mbid,)).fetchone()
        con.close()
    if refresh or not f:
        fetch_songs(mbid)
    with _db_lock:
        con = _db()
        rows = con.execute("SELECT * FROM songs WHERE singer=? "
                           "ORDER BY COALESCE(listens, -1) DESC, versions DESC, title", (mbid,)).fetchall()
        f = con.execute("SELECT at FROM fetched WHERE singer=?", (mbid,)).fetchone()
        con.close()
    return {"singer": {k: singer[k] for k in ("mbid", "name", "zh", "en", "region", "type", "disambiguation")},
            "fetched": f["at"] if f else None,
            "songs": [{"title": r["title"], "versions": r["versions"], "listens": r["listens"],
                       "length": round(r["length_ms"] / 1000) if r["length_ms"] else None,
                       "year": (r["first_release"] or "")[:4] or None} for r in rows]}


# ----------------------------------------------------------------- public: fetch
def fetch_songs(mbid: str) -> int:
    """Download one singer's recordings from MusicBrainz and store them as songs
    (versions of the same song merged). Returns the number of songs."""
    recs = []
    if config.MOCK:
        rnd = random.Random(mbid)
        base = ["月光", "Goodbye", "海闊天空", "朋友", "Shine", "下一站", "情歌", "Tonight", "雨天", "Home",
                "愛的故事", "Forever", "星空", "Hello", "時光"]
        for k in range(rnd.randint(8, 15)):
            t = base[k % len(base)]
            for v in range(rnd.randint(1, 4)):
                recs.append({"title": t + (" (Live)" if v else ""), "length": 200000 + 1000 * k,
                             "first-release-date": f"{2000 + k}-01-01"})
        time.sleep(0.5)
    else:
        offset, total = 0, None
        while total is None or (offset < total and offset < MAX_RECORDINGS):
            d = _get("recording", {"artist": mbid, "limit": 100, "offset": offset})
            total = d.get("recording-count", 0)
            batch = d.get("recordings") or []
            if not batch:
                break
            recs += batch
            offset += len(batch)
    groups: dict[str, dict] = {}
    for r in recs:
        if r.get("video"):
            continue
        title = (r.get("title") or "").strip()
        key = title_key(title)
        if not key:
            continue
        g = groups.setdefault(key, {"titles": {}, "n": 0, "length": None, "first": None})
        g["n"] += 1
        g["titles"][title] = g["titles"].get(title, 0) + 1
        if r.get("length") and not g["length"]:
            g["length"] = r["length"]
        frd = r.get("first-release-date") or ""
        if frd and (not g["first"] or frd < g["first"]):
            g["first"] = frd
    plays = _song_plays(mbid)
    rows = []
    for key, g in groups.items():
        # show the plainest spelling: without "(Live)"-style notes if any has none,
        # then the most frequent, then the shortest
        plain = [t for t in g["titles"] if not _BRACKETS.search(t) and not _SUFFIX.search(t)]
        title = sorted(plain or g["titles"], key=lambda t: (-g["titles"][t], len(t)))[0]
        rows.append((mbid, title, key, g["n"], g["length"], g["first"], plays.get(key)))
    with _db_lock:
        con = _db()
        con.execute("DELETE FROM songs WHERE singer=?", (mbid,))
        con.executemany("INSERT OR REPLACE INTO songs (singer, title, key, versions, length_ms, first_release, listens) "
                        "VALUES (?,?,?,?,?,?,?)", rows)
        con.execute("INSERT OR REPLACE INTO fetched VALUES (?,?,?)", (mbid, time.time(), len(recs)))
        con.commit()
        con.close()
    log.info("songs for %s: %d songs from %d recordings", mbid, len(rows), len(recs))
    return len(rows)


# ----------------------------------------------------------------- online search
_online_cache: dict[str, dict] = {}     # mbid -> artist JSON from the last online searches


def search_online(q: str) -> list[dict]:
    """Look a singer up on MusicBrainz by name (any country), for singers that
    are not in the stored list. Results are not stored until one is opened."""
    q = q.strip()[:100]
    if not q:
        return []
    if config.MOCK:
        arts = [{"id": "00000000-0000-0000-0001-000000000001", "name": "林俊傑", "sort-name": "Lin, JJ",
                 "type": "Person", "country": "SG", "aliases": [{"name": "JJ Lin", "locale": "en"}]}]
        if not any(w in search_key(q) for w in ("林", "jj", "lin")):
            arts = []
    else:
        # quoted phrase first (exact name), then the words as typed
        e = re.sub(r'([+\-!(){}\[\]^"~*?:\\/&|])', r"\\\1", q)    # escape search syntax
        arts = _get("artist", {"query": f'artist:"{e}" OR alias:"{e}" OR {e}', "limit": 25}).get("artists") or []
    out = []
    with _db_lock:
        con = _db()
        have = {r["mbid"] for r in con.execute(
            f"SELECT mbid FROM singers WHERE mbid IN ({','.join('?' * len(arts)) or 'NULL'})", [a['id'] for a in arts])}
        con.close()
    for a in arts:
        if a.get("type") not in KEEP_TYPES:
            continue
        _online_cache[a["id"]] = a
        zh, en = _names(a)
        out.append({"mbid": a["id"], "name": a.get("name"), "zh": zh, "en": en, "region": a.get("country") or "",
                    "type": a.get("type") or "", "disambiguation": a.get("disambiguation") or "",
                    "in_list": a["id"] in have})
    if len(_online_cache) > 500:
        _online_cache.clear()
    return out


def add_singer(mbid: str) -> dict:
    """Add a singer found by search_online to the stored list (kept through updates)."""
    if get_singer(mbid):
        return get_singer(mbid)
    a = _online_cache.get(mbid)
    if not a:
        a = _get(f"artist/{mbid}", {"inc": "aliases+tags"})
    row = _singer_row(a, a.get("country") or "")
    if not row:
        raise ValueError("not a singer or band")
    with _db_lock:
        con = _db()
        con.execute("INSERT OR REPLACE INTO singers VALUES (?,?,?,?,?,?,?,?,?,?)", row)
        con.execute("INSERT OR REPLACE INTO added VALUES (?, ?)", (mbid, json.dumps(row, ensure_ascii=False)))
        con.commit()
        con.close()
    try:
        update_popularity([mbid])
    except Exception:  # noqa: BLE001
        pass
    return get_singer(mbid)


# ----------------------------------------------------------------- singer lists made by the user
def is_local(mbid: str) -> bool:
    """Singers added by hand that MusicBrainz doesn't have get an ID "local-…"."""
    return mbid.startswith("local-")


def get_lists() -> list[dict]:
    with _db_lock:
        con = _db()
        rows = con.execute("""SELECT l.id, l.name, COUNT(m.singer) AS n FROM lists l
                              LEFT JOIN list_members m ON m.list_id = l.id
                              GROUP BY l.id ORDER BY l.created""").fetchall()
        con.close()
    return [{"id": r["id"], "name": r["name"], "count": r["n"]} for r in rows]


def create_list(name: str) -> dict:
    name = " ".join((name or "").split())[:40]
    if not name:
        raise ValueError("Give the list a name.")
    with _db_lock:
        con = _db()
        try:
            cur = con.execute("INSERT INTO lists (name, created) VALUES (?, ?)", (name, time.time()))
        except sqlite3.IntegrityError:
            con.close()
            raise ValueError(f"There is already a list called “{name}”.")
        con.commit()
        lid = cur.lastrowid
        con.close()
    return {"id": lid, "name": name, "count": 0}


def delete_list(list_id: int):
    with _db_lock:
        con = _db()
        con.execute("DELETE FROM list_members WHERE list_id=?", (list_id,))
        con.execute("DELETE FROM lists WHERE id=?", (list_id,))
        con.commit()
        con.close()


def _keep(con, mbid: str):
    """Remember a singer's row so it survives "Update list" (which rebuilds the table)."""
    r = con.execute("SELECT * FROM singers WHERE mbid=?", (mbid,)).fetchone()
    if r:
        con.execute("INSERT OR REPLACE INTO added VALUES (?, ?)",
                    (mbid, json.dumps([r[k] for k in r.keys()], ensure_ascii=False)))


def add_to_list(list_id: int, mbid: str) -> dict:
    """Add a singer to a list: one already stored, or one just found on MusicBrainz."""
    if not get_singer(mbid):
        add_singer(mbid)                    # from MusicBrainz into the local store
    with _db_lock:
        con = _db()
        if not con.execute("SELECT 1 FROM lists WHERE id=?", (list_id,)).fetchone():
            con.close()
            raise KeyError(list_id)
        con.execute("INSERT OR IGNORE INTO list_members VALUES (?, ?, ?)", (list_id, mbid, time.time()))
        _keep(con, mbid)
        con.commit()
        con.close()
    return get_singer(mbid)


def add_local_to_list(list_id: int, name: str) -> dict:
    """Add a singer MusicBrainz doesn't know (the user confirmed): stored with
    just the name; no song list, but YouTube can still be searched by name."""
    import uuid
    name = " ".join((name or "").split())[:60]
    if not name:
        raise ValueError("Type the singer's name.")
    with _db_lock:
        con = _db()
        if not con.execute("SELECT 1 FROM lists WHERE id=?", (list_id,)).fetchone():
            con.close()
            raise KeyError(list_id)
        # the same hand-added name again: reuse it
        r = con.execute("SELECT mbid FROM singers WHERE mbid LIKE 'local-%' AND keys = ?", (search_key(name),)).fetchone()
        mbid = r["mbid"] if r else "local-" + uuid.uuid4().hex[:30]
        if not r:
            zh = name if _CJK.search(name) else ""
            en = name if not zh else ""
            con.execute("INSERT INTO singers VALUES (?,?,?,?,?,?,?,?,?,?)",
                        (mbid, name, name, zh, en, "", "", "added by hand", 0, search_key(name)))
        con.execute("INSERT OR IGNORE INTO list_members VALUES (?, ?, ?)", (list_id, mbid, time.time()))
        _keep(con, mbid)
        con.commit()
        con.close()
    return get_singer(mbid)


def remove_from_list(list_id: int, mbid: str):
    with _db_lock:
        con = _db()
        con.execute("DELETE FROM list_members WHERE list_id=? AND singer=?", (list_id, mbid))
        con.commit()
        con.close()


# ----------------------------------------------------------------- ListenBrainz popularity
_lb_lock = threading.Lock()


def _lb(path: str, body: dict | None = None, tries: int = 4):
    """One ListenBrainz request (GET, or POST with a JSON body), politely paced."""
    url = f"{LB_API}/{path}"
    data = json.dumps(body).encode() if body is not None else None
    for attempt in range(tries):
        with _lb_lock:
            time.sleep(0.3)
            try:
                req = urllib.request.Request(url, data=data, headers={
                    "User-Agent": USER_AGENT, "Content-Type": "application/json", "Accept": "application/json"})
                with urllib.request.urlopen(req, timeout=30) as r:
                    return json.loads(r.read().decode("utf-8") or "null")
            except urllib.error.HTTPError as e:
                if e.code == 429 and attempt < tries - 1:          # rate limited: wait as told
                    time.sleep(min(60, float(e.headers.get("X-RateLimit-Reset-In") or 5) + 1))
                    continue
                if e.code in (500, 502, 503) and attempt < tries - 1:
                    time.sleep(2 + 3 * attempt)
                    continue
                raise
            except (urllib.error.URLError, TimeoutError):
                if attempt < tries - 1:
                    time.sleep(2 + 3 * attempt)
                    continue
                raise
    return None


def update_popularity(mbids: list[str]):
    """Listener counts for these singers, 100 per request (part of Update)."""
    for i in range(0, len(mbids), LB_BATCH):
        chunk = mbids[i:i + LB_BATCH]
        if config.MOCK:
            res = [{"artist_mbid": m, "total_user_count": (int(m[-4:]) * 37) % 1000 + 50,
                    "total_listen_count": (int(m[-4:]) * 911) % 50000} for m in chunk]
        else:
            res = _lb("popularity/artist", {"artist_mbids": chunk}) or []
        rows = [(x["artist_mbid"], x.get("total_user_count"), x.get("total_listen_count"))
                for x in res if x and x.get("artist_mbid")]
        with _db_lock:
            con = _db()
            con.executemany("INSERT OR REPLACE INTO popularity VALUES (?,?,?)", rows)
            con.commit()
            con.close()
        if status["running"] and status["stage"] == "popularity":
            status.update(done=status["done"] + 1)


def _song_plays(mbid: str) -> dict[str, int]:
    """A singer's most-played songs on ListenBrainz: {title key: plays}. Never raises."""
    if config.MOCK:
        rnd = random.Random(mbid + "plays")
        return {title_key(t): rnd.randint(0, 5000) for t in ("月光", "海闊天空", "朋友", "Shine", "下一站", "Home")}
    try:
        res = _lb(f"popularity/top-recordings-for-artist/{mbid}") or []
    except Exception as e:  # noqa: BLE001
        log.info("no song popularity for %s: %s", mbid, e)
        return {}
    plays: dict[str, int] = {}
    for r in res:
        k = title_key(r.get("recording_name") or "")
        if k:
            plays[k] = plays.get(k, 0) + int(r.get("total_listen_count") or 0)
    return plays


# ----------------------------------------------------------------- public: update
status = {"running": False, "stage": "", "done": 0, "total": 0, "message": "", "error": None, "finished": None}


def start_update() -> bool:
    """Refresh the singer list, then the songs already stored. In the background."""
    if status["running"]:
        return False
    status.update(running=True, stage="singers", done=0, total=0, message="Starting…", error=None)
    threading.Thread(target=_update, daemon=True, name="catalog-update").start()
    return True


def _update():
    try:
        rows = _download_singers()
        with _db_lock:
            con = _db()
            con.execute("DELETE FROM singers")
            con.executemany("INSERT OR REPLACE INTO singers VALUES (?,?,?,?,?,?,?,?,?,?)", rows)
            # singers added by hand from an online search stay in the list
            con.executemany("INSERT OR IGNORE INTO singers VALUES (?,?,?,?,?,?,?,?,?,?)",
                            [tuple(json.loads(r["row"])) for r in con.execute("SELECT row FROM added")])
            _meta(con, "singers_updated", time.time())
            con.commit()
            stored = [r["singer"] for r in con.execute("SELECT singer FROM fetched")]
            con.close()
        status.update(stage="popularity", done=0, total=(len(rows) + LB_BATCH - 1) // LB_BATCH,
                      message="Getting popularity from ListenBrainz…")
        try:
            update_popularity([r[0] for r in rows])
        except Exception as e:  # noqa: BLE001
            log.warning("popularity update failed (singers keep their previous order): %s", e)
        status.update(stage="songs", done=0, total=len(stored),
                      message=f"{len(rows)} singers saved. Refreshing songs of opened singers…")
        for i, mbid in enumerate(stored):
            if get_singer(mbid):
                try:
                    fetch_songs(mbid)
                except Exception as e:  # noqa: BLE001
                    log.info("songs refresh failed for %s: %s", mbid, e)
            status.update(done=i + 1)
        status.update(message=f"Done: {len(rows)} singers, songs refreshed for {len(stored)}.")
    except Exception as e:  # noqa: BLE001
        log.warning("singer list update failed: %s", e)
        status.update(error=f"Update failed: {e}. The previous list is kept.",
                      message="Update failed. The previous list is kept.")
    finally:
        status.update(running=False, finished=time.time())


def _download_singers() -> list[tuple]:
    """All artists of each region from MusicBrainz's search, 100 per request."""
    if config.MOCK:
        fake = [("陳奕迅", "Eason Chan", "HK"), ("容祖兒", "Joey Yung", "HK"), ("周杰倫", "Jay Chou", "TW"),
                ("鄧紫棋", "G.E.M.", "HK"), ("王菲", "Faye Wong", "CN"), ("蔡依林", "Jolin Tsai", "TW"),
                ("張學友", "Jacky Cheung", "HK"), ("五月天", "Mayday", "TW"), ("李克勤", "Hacken Lee", "HK"),
                ("林家謙", "Terence Lam", "HK"), ("毛不易", "Mao Buyi", "CN"), ("梅艷芳", "Anita Mui", "HK")]
        status.update(total=len(fake))
        out = []
        for i, (zh, en, reg) in enumerate(fake):
            time.sleep(0.15)
            a = {"id": f"00000000-0000-0000-0000-{i:012d}", "name": zh, "sort-name": en, "type": "Person",
                 "aliases": [{"name": en, "locale": "en", "primary": True}], "tags": [{"count": 12 - i}]}
            out.append(_singer_row(a, reg))
            status.update(done=i + 1, message=f"{zh}…")
        return out
    # MusicBrainz's search only pages through the first few hundred results
    # (the first live update stopped at about 500 artists per region), so
    # artists are listed by *area* instead ("browse", which pages to the end):
    # each region's own area, its subdivisions (provinces, districts) and their
    # cities. The search is kept as an extra pass for anything missed.
    out, seen = [], set()

    def add(artists, code):
        n = 0
        for a in artists:
            if a.get("id") and a["id"] not in seen:
                row = _singer_row(a, code)
                if row and code in CHINESE_NAME_ONLY and not row[3]:
                    continue                    # no Chinese name: not a Chinese singer
                if row:
                    seen.add(a["id"])
                    out.append(row)
                    n += 1
        return n

    for code, label in REGIONS.items():
        try:
            areas = _region_areas(code)
        except Exception as e:  # noqa: BLE001
            log.warning("could not list the areas of %s (%s); using search only", label, e)
            areas = []
        status.update(total=status["total"] + len(areas) + 1)
        for k, area in enumerate(areas):
            offset, count = 0, None
            while count is None or offset < count:
                d = _get("artist", {"area": area, "limit": 100, "offset": offset, "inc": "aliases+tags"})
                count = d.get("artist-count", 0)
                batch = d.get("artists") or []
                if not batch:
                    break
                add(batch, code)
                offset += len(batch)
            status.update(done=status["done"] + 1,
                          message=f"{label}: area {k + 1} of {len(areas)}, {len(out)} singers so far")
        # extra pass: search (first pages only; catches artists without an area link)
        offset, count = 0, None
        while (count is None or offset < count) and offset < 1000:
            try:
                d = _get("artist", {"query": f"country:{code}", "limit": 100, "offset": offset})
            except urllib.error.HTTPError as e:
                log.info("search for %s stopped at %d: %s", code, offset, e)
                break
            count = d.get("count", 0)
            batch = d.get("artists") or []
            if not batch:
                break
            add(batch, code)
            offset += len(batch)
        status.update(done=status["done"] + 1, message=f"{label}: done, {len(out)} singers so far")
    return out


MAX_AREAS = 700   # per region; enough for China's provinces and their cities


def _region_areas(code: str) -> list[str]:
    """The MusicBrainz area IDs of a region: the country/territory itself and
    its subdivisions (e.g. Taipei City, Hong Kong's districts); for mainland
    China also the cities inside each province (Guangzhou, Chengdu ...)."""
    d = _get("area", {"query": f"iso1:{code}", "limit": 10})
    top = next((a["id"] for a in d.get("areas") or [] if code in (a.get("iso-3166-1-codes") or [])), None)
    if not top:
        raise RuntimeError(f"no MusicBrainz area for {code}")
    found, level = [top], [top]
    for _ in range(2 if code == "CN" else 1):
        nxt = []
        for area in level:
            if len(found) >= MAX_AREAS:
                break
            for child in _child_areas(area):
                if child not in found:
                    found.append(child)
                    nxt.append(child)
        level = nxt
    return found[:MAX_AREAS]


def _child_areas(area: str) -> list[str]:
    d = _get(f"area/{area}", {"inc": "area-rels"})
    return [r["area"]["id"] for r in d.get("relations") or []
            if r.get("type") == "part of" and r.get("direction") == "backward" and (r.get("area") or {}).get("id")]
