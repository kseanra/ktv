"""Correct lyrics: use the real lyric text, and take only the TIMING from Whisper.

Whisper often mishears sung Cantonese (similar-sounding characters), but it
is good at *when* each syllable is sung. So:

  1. Lyric text comes from LRCLIB (https://lrclib.net, a free, open lyrics
     database, no API key), or is pasted by a user.
  2. That text is lined up, character by character, with what Whisper heard.
     Characters match when they are the same (Simplified and Traditional
     count as the same) or when they SOUND the same in Cantonese (Jyutping),
     so most of Whisper's homophone mistakes still give exact timing.
  3. Matched characters take Whisper's time; the rest are spread evenly
     between them. Line timestamps from LRCLIB (if any) help place lines that
     Whisper missed completely, after shifting them to this video's timing.

Whisper's own result is kept in whisper.json, so lyrics can be changed again,
or switched back to Whisper's, at any time without re-running Whisper.
"""
from __future__ import annotations

import json
import logging
import re
import statistics
import urllib.parse
import urllib.request
from pathlib import Path

from . import config

log = logging.getLogger("karaoke.lyrics")

LRCLIB = "https://lrclib.net/api"
USER_AGENT = "karaoke-prototype/1.0 (https://github.com/kseanra/ktv)"

# A "unit" is one sung syllable: one Chinese character, one kana/hangul
# character, or one word of English (letters, digits, apostrophes).
_UNIT = re.compile(r"[㐀-鿿豈-﫿぀-ヿ가-힯]|[A-Za-z0-9']+")
_CJK = re.compile(r"[㐀-鿿豈-﫿]")
CJK_DUR = 0.35      # typical seconds per sung character, used to fill gaps
WORD_DUR = 0.40
MAX_LINE_UNITS = 18  # longer lines are split in two (the TV shows short lines)
MIN_MATCH = 0.20     # online lyrics matching less than this are the wrong song

_CREDIT = re.compile(
    r"^\s*(作詞|作词|作曲|編曲|编曲|監製|监制|填詞|填词|詞|词|曲|主唱|演唱|原唱|和音|製作|制作|混音|"
    r"lyrics?|lyricist|composer|composed|arranger|arranged|producer|produced)\s*(by)?\s*[:：]", re.I)


# ----------------------------------------------------------------- helpers
_cc = None
_jp = None


def _to_trad(s: str) -> str:
    global _cc
    if _cc is None:
        try:
            import opencc
            _cc = opencc.OpenCC("s2t")
        except Exception:  # noqa: BLE001
            _cc = False
    return _cc.convert(s) if _cc else s


def _sound(ch: str) -> str | None:
    """Cantonese pronunciation of one character without its tone, e.g. 愛 -> oi."""
    global _jp
    if _jp is None:
        try:
            import ToJyutping
            _jp = ToJyutping
        except Exception:  # noqa: BLE001
            _jp = False
    if not _jp or not _CJK.match(ch):
        return None
    try:
        r = _jp.get_jyutping_list(ch)
        return re.sub(r"\d", "", r[0][1]) if r and r[0][1] else None
    except Exception:  # noqa: BLE001
        return None


def _key(u: str) -> str:
    return _to_trad(u) if _CJK.match(u) else u.lower()


def norm(s: str) -> str:
    """For comparing names: Traditional, lower case, letters/digits/CJK only."""
    return re.sub(r"[^0-9a-z㐀-鿿豈-﫿぀-ヿ가-힯]", "", _to_trad(s or "").lower())


# ----------------------------------------------------------------- reference text
def parse_lyrics_text(text: str) -> list[dict]:
    """Pasted or downloaded lyrics -> [{"text", "t"}]. Understands plain text
    (one line per row) and LRC ("[01:23.45] line"). Credit lines are dropped."""
    out = []
    for raw in (text or "").splitlines():
        tags = re.findall(r"\[(\d+):(\d+(?:[.:]\d+)?)\]", raw)
        body = re.sub(r"\[[^\]]*\]", "", raw).strip()
        if not body or _CREDIT.match(body):
            continue
        if not _UNIT.search(body):
            continue
        times = [int(m) * 60 + float(s.replace(":", ".")) for m, s in tags] or [None]
        for t in times:
            out.append({"text": body, "t": t})
    if any(x["t"] is not None for x in out):
        out.sort(key=lambda x: (x["t"] is None, x["t"] or 0))
    return out


def _split_long(lines: list[dict]) -> list[dict]:
    out = []
    for ln in lines:
        units = list(_UNIT.finditer(ln["text"]))
        if len(units) <= MAX_LINE_UNITS:
            out.append(ln)
            continue
        mid = len(units) // 2
        # prefer a space or punctuation near the middle
        best, cut = None, units[mid].start()
        for k in range(max(1, mid - 4), min(len(units), mid + 5)):
            between = ln["text"][units[k - 1].end():units[k].start()]
            if between.strip(" ") != between or re.search(r"[\s,，、。!！?？]", between):
                if best is None or abs(k - mid) < abs(best - mid):
                    best, cut = k, units[k].start()
        a, b = ln["text"][:cut].strip(), ln["text"][cut:].strip()
        out += _split_long([{"text": a, "t": ln["t"]}]) + _split_long([{"text": b, "t": None}])
    return out


def _ref_units(lines: list[dict]):
    """Units of the reference lyrics, each with the text shown for it
    (the unit plus any spaces/punctuation after it)."""
    units = []   # (line_index, display, key, sound)
    for li, ln in enumerate(lines):
        ms = list(_UNIT.finditer(ln["text"]))
        for k, m in enumerate(ms):
            start = 0 if k == 0 else m.start()
            end = ms[k + 1].start() if k + 1 < len(ms) else len(ln["text"])
            u = m.group()
            units.append((li, ln["text"][start:end], _key(u), _sound(u)))
    return units


def _hyp_units(words: list[dict]):
    """Whisper's words -> units with times (a multi-character word is split evenly)."""
    out = []   # (key, sound, s, e)
    for w in words:
        us = _UNIT.findall(w["w"])
        if not us:
            continue
        s, e = float(w["s"]), float(w["e"])
        d = max(e - s, 0.05) / len(us)
        for k, u in enumerate(us):
            out.append((_key(u), _sound(u), s + k * d, s + (k + 1) * d))
    return out


# ----------------------------------------------------------------- alignment
def _align(ref, hyp):
    """Global alignment (Needleman-Wunsch, free gaps at both ends of Whisper's
    side). Returns [(ref_index, hyp_index, kind)] with kind "same" / "sound" / "sub"."""
    n, m = len(ref), len(hyp)
    GAP = -1
    prev = [0] * (m + 1)                       # free leading gap in Whisper's text
    back = []
    for i in range(1, n + 1):
        rk, rs = ref[i - 1][2], ref[i - 1][3]
        cur = [prev[0] + GAP] + [0] * m
        bt = bytearray(m + 1)
        bt[0] = 1
        for j in range(1, m + 1):
            hk, hs = hyp[j - 1][0], hyp[j - 1][1]
            sc = 3 if rk == hk else 2 if (rs and rs == hs) else -1
            d, u, l = prev[j - 1] + sc, prev[j] + GAP, cur[j - 1] + GAP
            if d >= u and d >= l:
                cur[j] = d                  # 0 = diagonal
            elif u >= l:
                cur[j], bt[j] = u, 1        # 1 = skip a reference unit
            else:
                cur[j], bt[j] = l, 2        # 2 = skip a Whisper unit
        back.append(bt)
        prev = cur
    # free trailing gap in Whisper's text: best end column on the last row
    j = max(range(m + 1), key=lambda c: prev[c])
    i, pairs = n, []
    while i > 0 and j > 0:
        b = back[i - 1][j]
        if b == 0:
            rk, rs = ref[i - 1][2], ref[i - 1][3]
            hk, hs = hyp[j - 1][0], hyp[j - 1][1]
            kind = "same" if rk == hk else "sound" if (rs and rs == hs) else "sub"
            pairs.append((i - 1, j - 1, kind))
            i, j = i - 1, j - 1
        elif b == 1:
            i -= 1
        else:
            j -= 1
    pairs.reverse()
    return pairs


def align(ref_lines: list[dict], words: list[dict], duration: float | None = None):
    """Time the reference lyrics using Whisper's words.
    Returns (lines in lyrics.json format, stats) or (None, stats) if impossible."""
    ref_lines = _split_long(ref_lines)
    ref = _ref_units(ref_lines)
    hyp = _hyp_units(words)
    n = len(ref)
    if not n:
        return None, {"reason": "no lyrics text"}
    s = [None] * n
    e = [None] * n
    matched = 0
    if hyp:
        pairs = _align(ref, hyp)
        good = [i for i, _, k in pairs if k != "sub"]
        matched = len(good)
        for i, j, k in pairs:
            if k != "sub":
                s[i], e[i] = hyp[j][2], hyp[j][3]
        # A mismatch between two matches (within 3 units each side) is very
        # likely the same syllable misheard: use its time too.
        import bisect
        for i, j, k in pairs:
            if k == "sub":
                p = bisect.bisect_left(good, i)
                before = good[p - 1] if p > 0 else None
                after = good[p] if p < len(good) else None
                if before is not None and after is not None and i - before <= 3 and after - i <= 3:
                    s[i], e[i] = hyp[j][2], hyp[j][3]
    stats = {"units": n, "matched": matched, "match": round(matched / n, 3)}
    if stats["match"] < 0.1 and any(ln.get("t") is not None for ln in ref_lines):
        # Almost nothing matched (Whisper heard junk): trust the LRC timestamps alone.
        s = [None] * n
        e = [None] * n
        hyp = []

    # Line timestamps from an LRC file, shifted to this video. The shift is
    # the median difference between Whisper's time and the LRC time on lines
    # whose first character matched (videos often have a longer intro).
    first = {}
    for i, u in enumerate(ref):
        first.setdefault(u[0], i)
    lrc = [ln.get("t") for ln in ref_lines]
    diffs = [s[first[li]] - t for li, t in enumerate(lrc) if t is not None and li in first and s[first[li]] is not None]
    shift = statistics.median(diffs) if len(diffs) >= 3 else 0.0
    stats["lrc_shift"] = round(shift, 2) if any(t is not None for t in lrc) else None
    for li, t in enumerate(lrc):
        if t is None or li not in first:
            continue
        idx = [i for i, u in enumerate(ref) if u[0] == li]
        have = sum(1 for i in idx if s[i] is not None)
        ts = t + shift
        nxt = next((lrc[k] + shift for k in range(li + 1, len(lrc)) if lrc[k] is not None), None)
        # drop matches far outside the line's LRC window (likely wrong)
        for i in idx:
            if s[i] is not None and (s[i] < ts - 4 or (nxt is not None and s[i] > nxt + 4)):
                s[i] = e[i] = None
        if s[idx[0]] is None and (have < len(idx) * 0.5 or not hyp):
            s[idx[0]] = max(ts, 0.0)
            e[idx[0]] = s[idx[0]] + _dur(ref[idx[0]][1])

    known = [i for i in range(n) if s[i] is not None]
    if not known:
        reason = ("None of these words match what is sung. Check that the lyrics belong to this song."
                  if hyp else "These lyrics can't be timed: the AI heard no words in this song (or its lyrics "
                  "were skipped), and the lyrics have no timestamps. Lyrics found online marked \"timed\" work.")
        return None, {**stats, "reason": reason}
    # keep times in order (drop anything that goes backwards)
    last, keep = -1.0, []
    for i in known:
        if s[i] >= last:
            keep.append(i)
            last = s[i]
        else:
            s[i] = e[i] = None
    known = keep
    _fill(ref, s, e, known, duration)

    lines = []
    for li, ln in enumerate(ref_lines):
        idx = [i for i, u in enumerate(ref) if u[0] == li]
        if not idx:
            continue
        ws = [{"w": ref[i][1], "s": round(s[i], 3), "e": round(e[i], 3)} for i in idx]
        lines.append({"start": ws[0]["s"], "end": ws[-1]["e"], "text": ln["text"], "words": ws})
    return lines, stats


def _dur(display: str) -> float:
    return CJK_DUR if _CJK.search(display) else WORD_DUR


def _fill(ref, s, e, known, duration):
    """Give every unit without a time one, between its timed neighbours."""
    n = len(ref)
    line = [u[0] for u in ref]
    dur = [_dur(u[1]) for u in ref]
    # before the first known unit: backwards from it
    k0 = known[0]
    t = s[k0]
    for i in range(k0 - 1, -1, -1):
        t -= dur[i]
        s[i], e[i] = max(t, 0.0), max(t, 0.0) + dur[i]
    # after the last known unit: forwards
    k1 = known[-1]
    t = e[k1] if e[k1] is not None else s[k1] + dur[k1]
    for i in range(k1 + 1, n):
        s[i], e[i] = t, t + dur[i]
        t += dur[i]
    # gaps between known units
    for a, b in zip(known, known[1:]):
        gap = list(range(a + 1, b))
        if not gap:
            continue
        ta = e[a] if e[a] is not None else s[a] + dur[a]
        ta = min(ta, s[b])
        tb = s[b]
        need = sum(dur[i] for i in gap)
        if tb - ta <= need * 1.6:
            step = (tb - ta) / len(gap)
            for k, i in enumerate(gap):
                s[i] = ta + k * step
                e[i] = s[i] + max(step, 0.05)
            continue
        # A long gap (an instrumental break?): units on a's line follow a,
        # units on b's line lead into b, whole lines in between are spread out.
        head = [i for i in gap if line[i] == line[a]]
        tail = [i for i in gap if line[i] == line[b] and i not in head]
        mid = [i for i in gap if i not in head and i not in tail]
        t = ta
        for i in head:
            s[i], e[i] = t, t + dur[i]
            t += dur[i]
        lo = t
        t = tb
        for i in reversed(tail):
            t -= dur[i]
            s[i], e[i] = t, t + dur[i]
        hi = max(t, lo)
        if mid:
            groups = []
            for i in mid:
                if groups and line[groups[-1][-1]] == line[i]:
                    groups[-1].append(i)
                else:
                    groups.append([i])
            slot = (hi - lo) / len(groups)
            for g_i, g in enumerate(groups):
                d = min(CJK_DUR * 1.2, slot / max(len(g), 1))
                start = lo + g_i * slot + max(0.0, (slot - d * len(g)) / 2)
                for k, i in enumerate(g):
                    s[i], e[i] = start + k * d, start + (k + 1) * d
    # end times: never past the next unit's start, never longer than 2 s
    for i in range(n):
        nxt = s[i + 1] if i + 1 < n else (duration or s[i] + 2)
        if e[i] is None or e[i] <= s[i]:
            e[i] = s[i] + dur[i]
        e[i] = max(s[i] + 0.05, min(e[i], nxt if nxt > s[i] else e[i], s[i] + 2.0))


# ----------------------------------------------------------------- LRCLIB
def _get(path: str, params: dict | None = None, timeout: float = 10):
    url = f"{LRCLIB}/{path}" + ("?" + urllib.parse.urlencode(params) if params else "")
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8"))


def _candidate(d: dict) -> dict:
    return {"id": d.get("id"), "track": d.get("trackName") or "", "artist": d.get("artistName") or "",
            "album": d.get("albumName") or "", "duration": d.get("duration"),
            "instrumental": bool(d.get("instrumental")),
            "synced": d.get("syncedLyrics") or "", "plain": d.get("plainLyrics") or ""}


_NOISE = re.compile(
    r"(?<![a-z])(official|music\s*video|lyrics?\s*video|lyrics?|karaoke|ktv|mv|hd|hq|4k|1080p|720p|60fps|upscale|"
    r"audio|live|version|ver|cover|off\s*main\s*vocal)(?![a-z])|"
    r"官方|完整版|高清|修復版?|修复版?|清晰版|歌詞版?|歌词版?|動態歌詞|动态歌词|字幕|伴奏|伴唱|原版|純音樂|纯音乐|"
    r"卡拉ok版?|合唱版?|合唱|純享版?|纯享版?|演唱會|演唱会|音樂會|音乐会|現場|现场|版", re.I)


def _clean(t: str) -> str:
    t = re.sub(r"[\[\(（【〔][^\]\)）】〕]*[\]\)）】〕]", " ", t)     # (Official Music Video), [伴奏] ...
    t = re.split(r"[|｜#@]", t)[0]
    t = _NOISE.sub(" ", t)
    t = re.sub(r"(?<![A-Za-z])[x×](?![A-Za-z])", " ", t)            # "KELLY x EKIN"
    t = re.sub(r"[-–—─/&,，、:：.．!！?？~《》「」『』\[\]()（）【】]", " ", t)
    return re.sub(r"\s+", " ", t).strip()


def guess_queries(title: str) -> list[str]:
    """Search texts to try for a YouTube title such as
    '陳奕迅 Eason Chan - 富士山下 KTV (4K修復)' or 'KELLY x EKIN《甘心替代你》Live @ ...'."""
    t = re.sub(r"[\[\(（【〔][^\]\)）】〕]*[\]\)）】〕]", " ", title or "")
    queries = []
    m = re.search(r"[《「『]([^》」』]+)[》」』]", t)
    if m:                                    # 《song》: the song name is certain
        track = m.group(1).strip()
        queries += [f"{_clean(t[:m.start()])} {track}".strip(), track]
    c = _clean(t)
    if c:
        queries.append(c)
        runs = re.findall(r"[\u3400-\u9fff\uf900-\ufaff]{2,}", c)
        if len(runs) >= 2:
            queries.append(" ".join(runs))           # Chinese names only
        queries += runs                              # each name alone (one is the song)
    # Titles are often in Simplified while LRCLIB has Traditional (or the
    # reverse): try a Traditional version of each search too.
    queries = [v for q in queries for v in (q, _to_trad(q))]
    seen, out = set(), []
    for q in queries:
        k = q.lower()
        if q and k not in seen:
            seen.add(k)
            out.append(q)
    return out[:8]


def search(query: str, limit: int = 15, params: dict | None = None) -> list[dict]:
    """LRCLIB search -> candidates (with lyric text). `params` replaces the
    plain text search, e.g. {"track_name": ..., "artist_name": ...}."""
    if config.MOCK:
        return _mock_search(query)
    res = _get("search", params or {"q": query})
    return [_candidate(d) for d in res[:limit]]


def user_searches(query: str) -> list[dict]:
    """Searches for a name typed by a user, e.g. "富士山下 陳奕迅" or "富士山下".
    Besides the plain text search, try it as song name + singer (both ways
    round), and in Traditional characters."""
    out = []
    for q in dict.fromkeys([query, _to_trad(query)]):     # unique, in order
        parts = q.split()
        out.append({"q": q})
        if len(parts) == 1:
            out.append({"track_name": q})
        elif len(parts) == 2:
            out += [{"track_name": parts[0], "artist_name": parts[1]},
                    {"track_name": parts[1], "artist_name": parts[0]}]
    return out


def search_user(query: str) -> list[dict]:
    """All results for a typed name (see user_searches), without duplicates."""
    found, seen = [], set()
    for params in user_searches(query.strip()):
        for c in search(query, params=params):
            if c["id"] not in seen:
                seen.add(c["id"])
                found.append(c)
        if config.MOCK:
            break
    return found


def get_by_id(lrclib_id: int) -> dict:
    if config.MOCK:
        return next(c for c in _mock_search("") if c["id"] == lrclib_id)
    return _candidate(_get(f"get/{int(lrclib_id)}"))


def _score(c: dict, title_n: str, duration: float | None, strict: bool = False) -> float | None:
    """How well an LRCLIB entry fits a YouTube title; None = not this song.
    strict (automatic use): the singer must be in the title too, or the song
    name must be 3+ characters with nearly the same length."""
    if c["instrumental"] or not (c["synced"] or c["plain"]):
        return None
    tn = norm(c["track"])
    if len(tn) < 2 or tn not in title_n:
        return None
    sc = 2.0 + min(len(tn), 6) * 0.1
    an = [norm(a) for a in re.split(r"[,&/、×]| x | feat\.? | and ", c["artist"], flags=re.I)]
    artist_ok = any(len(a) >= 2 and a in title_n for a in an)
    if artist_ok:
        sc += 2
    close = bool(duration and c["duration"] and abs(duration - c["duration"]) <= 10)
    if strict and not (artist_ok or (close and len(tn) >= 3)):
        return None
    if c["synced"]:
        sc += 1
    if duration and c["duration"]:
        d = abs(duration - c["duration"])
        sc += 1.5 if d <= 5 else 1 if d <= 15 else 0.3 if d <= 40 else -1
    return sc


def find_online(title: str, duration: float | None, query: str | None = None) -> dict | None:
    """Best LRCLIB entry for a YouTube video, or None. `query` is a song name
    (and singer) typed by the user: searched first, and trusted, so the entry
    only has to match it (not the video title)."""
    query = (query or "").strip()
    title_n = norm(f"{title} {query}")
    best, best_sc, seen = None, None, set()
    searches = [(query, p, False) for p in user_searches(query)] if query else []
    searches += [(q, None, True) for q in guess_queries(title)]
    for q, params, strict in searches:
        try:
            cands = search(q, params=params)
        except Exception as e:  # noqa: BLE001
            log.info("LRCLIB search failed (%s): %s", params or q, e)
            return best
        for c in cands:
            if c["id"] in seen:
                continue
            seen.add(c["id"])
            sc = _score(c, norm(query) if not strict else title_n, duration, strict=strict)
            if sc is not None and (best_sc is None or sc > best_sc):
                best, best_sc = c, sc
        if best_sc is not None and best_sc >= 5:       # name + artist + synced: good enough
            break
    return best


def candidate_text(c: dict) -> str:
    return c["synced"] or c["plain"]


# ----------------------------------------------------------------- song files
def whisper_lyrics(out: Path) -> dict | None:
    """Whisper's own result for a song (kept in whisper.json)."""
    w = out / "whisper.json"
    if w.exists():
        return json.loads(w.read_text(encoding="utf-8"))
    lj = out / "lyrics.json"
    if lj.exists():
        d = json.loads(lj.read_text(encoding="utf-8"))
        if d.get("source", "ai") == "ai" and not d.get("skipped"):
            w.write_text(json.dumps(d, ensure_ascii=False), encoding="utf-8")   # keep it before replacing
            return d
    return None


def apply_text(out: Path, text: str, duration: float | None, source: dict,
               min_match: float | None = None) -> dict:
    """Time `text` against Whisper's words and write lyrics.json.
    Returns stats; raises ValueError if it can't be done (lyrics.json untouched)."""
    ref = parse_lyrics_text(text)
    if not ref:
        raise ValueError("No lyric lines found in the text.")
    ai = whisper_lyrics(out) or {}
    words = [w for ln in ai.get("lines", []) for w in ln.get("words", [])]
    lines, stats = align(ref, words, duration)
    if lines is None:
        raise ValueError(stats.get("reason", "Could not time these lyrics."))
    if min_match is not None and words and stats["match"] < min_match:
        raise ValueError(f"only {stats['match']:.0%} matched what is sung")
    lyrics = {"language": ai.get("language"), "lines": lines, "source": source.get("kind", "pasted"),
              "source_info": {**source, **stats},
              "source_text": text}          # kept, so it can be re-timed after "Redo AI lyrics"
    (out / "lyrics.json").write_text(json.dumps(lyrics, ensure_ascii=False), encoding="utf-8")
    log.info("lyrics for %s from %s: %d lines, %.0f%% matched", out.name, lyrics["source"], len(lines), stats["match"] * 100)
    return stats


def use_ai(out: Path) -> bool:
    """Switch back to Whisper's own lyrics."""
    ai = whisper_lyrics(out)
    if not ai:
        return False
    (out / "lyrics.json").write_text(json.dumps(ai, ensure_ascii=False), encoding="utf-8")
    return True


def after_ai(out: Path, title: str, duration: float | None, query: str | None, previous: dict,
             fresh: bool = False):
    """Run after Whisper. Lyrics chosen before (online or pasted) keep their
    text and get re-timed with the new AI words; otherwise look online.
    fresh: a new lyrics name was given, so look online with it first (the
    previous lyrics are only kept if nothing is found)."""
    if fresh and query and auto_online(out, title, duration, query):
        return True
    if previous.get("source") in ("lrclib", "pasted") and previous.get("lines"):
        text = previous.get("source_text") or "\n".join(ln.get("text", "") for ln in previous["lines"])
        src = {k: v for k, v in (previous.get("source_info") or {}).items()
               if k in ("kind", "id", "track", "artist", "synced")}
        src.setdefault("kind", previous["source"])
        try:
            return apply_text(out, text, duration, src)
        except Exception as e:  # noqa: BLE001
            log.info("could not re-time the previous lyrics of %s: %s", out.name, e)
    return auto_online(out, title, duration, query)


def auto_online(out: Path, title: str, duration: float | None, query: str | None = None) -> dict | None:
    """After Whisper: look the song up on LRCLIB and use those lyrics if they
    clearly belong to this song. Never raises."""
    if not config.LYRICS_ONLINE:
        return None
    try:
        if not title and (out / "info.json").exists():
            title = json.loads((out / "info.json").read_text(encoding="utf-8")).get("title", "")
        c = find_online(title, duration, query)
        if not c:
            log.info("no online lyrics found for %r", query or title)
            return None
        return apply_text(out, candidate_text(c), duration,
                          {"kind": "lrclib", "id": c["id"], "track": c["track"], "artist": c["artist"],
                           "synced": bool(c["synced"])}, min_match=MIN_MATCH)
    except Exception as e:  # noqa: BLE001
        log.info("online lyrics not used for %r: %s", title, e)
        return None


# ----------------------------------------------------------------- mock
def _mock_search(query: str) -> list[dict]:
    words = "la la la sing along with me under city lights tonight we never stop the music".split()
    lrc, t = [], 2.0
    for k in range(6):
        line = " ".join(words[(k * 8 + i) % len(words)] for i in range(8))
        lrc.append(f"[{int(t // 60):02d}:{t % 60:05.2f}] {line.capitalize()}")
        t += 4.0 + 0.9
    return [{"id": 9001, "track": f"{query or 'Mock song'}", "artist": "Mock singer", "album": "Mock album",
             "duration": 30, "instrumental": False, "synced": "\n".join(lrc), "plain": ""},
            {"id": 9002, "track": "Another mock song", "artist": "Someone", "album": "", "duration": 200,
             "instrumental": False, "synced": "", "plain": "First line of words\nSecond line of words"}]
