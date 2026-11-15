# Decisions

## 2026-11-14 · Chinese singer and song list from MusicBrainz

**Status:** decided, built (`app/catalog.py`, *Singers* tab).

### What we wanted

A list of Chinese singers in the app. Tapping a singer shows their songs;
tapping a song searches YouTube for "song name + singer", so it can be added
to the playlist. The lists must be updatable.

### Source: MusicBrainz

MusicBrainz is a free, open music database (artists, releases, recordings),
run by the MetaBrainz Foundation. Its core data (artists, songs, releases) is
CC0 (free for any use); tags, genres, ratings and annotations are
CC BY-NC-SA 3.0 (non-commercial use, with credit).

Other free sources were looked at and not chosen: Wikidata (good for singer
names, no song lists), Last.fm API (popularity, needs a key), KKBOX Open API
(good Taiwan/HK charts, developer account and terms), unofficial QQ Music /
NetEase APIs (against their terms, often break). AI models were rejected for
song lists: they invent titles and miss many.

### Options considered

| Option | Download | Notes |
|---|---|---|
| Offline JSON dumps (artist + release) | artist.tar.xz about 2 GB; release.tar.xz about 23 GB compressed (songs are only in the release dump; the recording dump has standalone recordings only) | Fully offline, but the dumps are not split by country: everything must be downloaded and filtered. Hours of processing per update; too heavy for the laptop. |
| PostgreSQL dumps | several GB, needs PostgreSQL or `mbdata` | Same problem: whole database, not just Chinese artists. |
| Hybrid: offline singer file + songs from the web service | 2 GB once per update | Still downloads all artists to keep the Chinese ones. |
| **MusicBrainz web service, Chinese only (chosen)** | only Chinese data | Fetch only what is needed, store it locally. |

### Decision

Use the MusicBrainz web service (`https://musicbrainz.org/ws/2`) and store the
results in a local SQLite file, `data/catalog.sqlite3`, so the app works
offline after fetching.

- **Chinese singers only, no "Other" list** (the user's choice). Regions:
  Hong Kong, Taiwan, mainland China, Macau, plus Chinese-named singers from
  Singapore and Malaysia (added 2026-11-15).
- **Singer list:** "Update list" lists the artists of each region (Hong Kong,
  Taiwan, mainland China, Macau), 100 per request (by area; see the update
  below), and stores every person, group or choir (orchestras, fictional
  characters and "Other" types are left out). Chinese and English names come
  from the artist name and its aliases. Ordered by popularity (see the
  update below). Search works in Traditional or
  Simplified characters and English.
- **Songs:** fetched for one singer the first time that singer is opened
  (`/ws/2/recording?artist=<id>`, 100 per request, up to 2,500 recordings),
  then stored. Versions of the same song ("(Live)", "(粵語版)",
  " - Remaster", etc.) are merged into one entry; the number of versions is
  used as a rough popularity order ("Popular"), with A–Z and Newest as other
  orders. Music videos are left out.
- **Update:** "Update list" downloads the singer list again (the old list stays
  usable until the new one is complete) and then refreshes the songs of
  every singer already opened.
- **Tapping a song** fills the Playlist search box with "song singer" and runs
  the existing YouTube search; the same text fills the Add popup's lyrics name,
  which helps the online lyrics lookup.

### Update (2026-11-14): popularity ranking from ListenBrainz

MusicBrainz has no popularity figure, so the first version ordered singers by
a rough guess (number of aliases and tags). To put the top singers first:

- **Singers** are ordered by **listener count from ListenBrainz**
  (`POST https://api.listenbrainz.org/1/popularity/artist`, 100 artists per
  request). ListenBrainz is MusicBrainz's sister project (MetaBrainz): it uses
  the same artist IDs, so no name matching is needed; it is free and needs no
  key. Fetched as part of "Update list". Singers without ListenBrainz data
  follow, in the old order.
- **Songs** of a singer are ordered by **plays** from ListenBrainz's
  most-played recordings of that artist
  (`GET /1/popularity/top-recordings-for-artist/<id>`, one request when the
  singer's songs are fetched), then by the number of versions.
- Listener and play counts are shown in the lists.

Considered and not chosen: Last.fm listener counts (needs an API key and name
matching), KKBOX charts (developer account; charts, not a full ranking), our
own play history (too little data at first; could be added later as a boost).

Limits: ListenBrainz users skew towards Western listeners, so counts for Chinese
singers are small and favour singers with international audiences. The order
is a good "best known first", not an official chart.

### Update (2026-11-14): singers listed by area, not by search

The first live "Update list" saved only about 500 artists per region
(Hong Kong 495, Taiwan 496, mainland China 489, Macau 37): MusicBrainz's
search pages through only the first few hundred results. The update now lists
artists by **area** with the browse API (`/ws/2/artist?area=<id>`), which pages
to the end:

- each region's own area (found by ISO code, `iso1:HK` etc.),
- its subdivisions (Hong Kong's districts, Taiwan's cities and counties,
  China's provinces), via `area/<id>?inc=area-rels` ("part of"),
- for mainland China also the cities inside each province,
- plus the old search as an extra pass (first 1,000 results) for artists
  without an area link.

This means more requests (one or more per area, several hundred for China), so
an update takes roughly 10-20 minutes in the background.

### Update (2026-11-15): better singer search

Searching the stored list sometimes showed the wrong singer first or nothing:

- **Ranking:** results are now ordered by how well they match (a whole name or
  alias equal to the search, then names starting with it, then names
  containing it), and only then by popularity.
- **Region filter:** if the chosen region has no match, all regions are
  searched and the list says so.
- **Singapore and Malaysia** are added as regions (JJ Lin, Stefanie Sun,
  Fish Leong ... were missing); from these two only artists with a Chinese
  name are kept.
- **Online fallback:** "Not here? Search MusicBrainz" looks the name up on
  MusicBrainz (any country). Tapping a result adds that singer to the list
  (kept through later updates) and opens their songs.

### Update (2026-11-15): singer lists made by the user

Users can make their own singer lists (e.g. "我的最愛", "Party") next to the
full Chinese list:

- **New list** (popup): a name; lists are stored in `data/catalog.sqlite3`
  (`lists`, `list_members`).
- **Add singer** (popup): the typed name is checked on MusicBrainz first.
  If it is found, the user picks the right artist and it is stored locally
  (and its popularity fetched). If MusicBrainz has no such singer (or can't be
  reached), the user is asked to confirm, and the singer is added by hand with
  just the name: ID `local-…`, no song list, but "Search YouTube for <name>"
  still works.
- Singers in a list are kept through "Update list" (stored in `added`).
- A list can be shown instead of the full list (remembered per phone), searched,
  and singers removed (✕); a list can be deleted (tap twice).

### Rules we follow (MusicBrainz terms)

- At most about one request per second (`app/catalog.py` waits 1.1 s between
  requests and backs off on HTTP 503/429).
- A User-Agent naming the app and a contact:
  `KaraokePrototype/1.0 ( https://github.com/kseanra/ktv )`.

### Consequences

- A singer-list download takes roughly 10-20 minutes (hundreds of requests at
  one per second); it runs in the background with progress shown.
- Internet is needed to update and to open a singer for the first time; after
  that, browsing works offline.
- Downloading every song of every Chinese singer up front would take many hours
  at one request per second, so songs are fetched per singer instead.
- MusicBrainz is crowd-edited: big Cantopop/Mandopop names are well covered,
  smaller singers can be incomplete, and some entries are composers or
  lyricists rather than singers.
- If very large result sets stop paging (a search-service limit), the update
  keeps what it got and logs a warning.


## 2026-11-15 · Male / female duet tags: voice fingerprints instead of pitch

**Status:** decided, built (`app/voices.py`, `app/lyrics_sync.py`, Lyrics window).

### Problem

Duet lines were tagged M / F from pitch only. A man singing high or a woman
singing low got the wrong tag; measured on known-answer test duets, pitch got
about 68-72% of lines right.

### Decision

Three parts, used together:

1. **Lyric marks first.** Duet lyrics (online or pasted) often say who sings:
   `男：`, `女：`, `合：`, `(男)`, `【女】`, `M:`, `F:`, `Both:`. A mark applies to
   its line and the following lines until the next mark. Marked lines get
   their tag straight away (M, F, or B for 合), with no audio analysis, and
   the duet step keeps them.
2. **Voice fingerprints (speaker embeddings)** for the other lines. Each line's
   separated vocals go through **WeSpeaker ResNet34** (trained on VoxCeleb,
   CC-BY-4.0, 26 MB ONNX, run with ONNX Runtime on the CPU; no PyTorch, so it
   also works on the Mac). This gives a timbre "fingerprint" that doesn't
   change with how high a note is. The lines are split into the song's two
   voices (spherical 2-means); only then is each group named M or F, by lyric
   marks if any, else by the group's overall pitch (reliable when averaged
   over many lines). Lines nearly as close to both voices follow their
   neighbours. If the song doesn't split into two clear voices, or the model
   can't load, the pitch tags are used. The model is downloaded once from the
   sherpa-onnx GitHub release (Hugging Face is blocked on this network).
   `KARAOKE_VOICE_METHOD=pitch` switches back to pitch only.
3. **Fix by hand.** The Lyrics window has *Singer tags*: tap a line's tag to
   cycle M → F → B → none, *Swap M ↔ F*, *Save tags*. Saved tags are kept as
   marks, so re-tagging doesn't undo them.

### Update (2026-11-15): lines sung together ("B", green)

Lines where both sing were still given one singer. They are now detected and
shown as **B** with a **green** highlight (labels: M blue, F pink, B green):

- the line's fingerprint is a mix, so it is fairly close to *both* voices: its
  similarity to the farther voice is high for this song (above the song's
  median + 0.08, at least 0.48), **and**
- the pitch analysis finds a male and a female voice at the same time
  (male evidence >= 0.85 and female evidence >= 1.0 of the song's typical level);
- or the lyric line has a part in brackets ("... (echo)"), usually the other
  singer joining in, and the line is nearly as close to both voices;
- lyric marks `合：` (and hand fixes) still win.

On *A Whole New World* this tagged 10 lines B; 8-9 of them are the together or
echo lines (one solo high line was a false B), and some together lines were
missed. On *祝君好* (Kelly x Chilam) both "雖不可親口細訴" lines and the final
line came out B. Wrong ones can be fixed in *Singer tags*.

### Results (test run on this PC's library)

- Known-answer test duets built from a male and a female solo song from the
  library (6 pairs, 246 lines): pitch **69.5%**, fingerprints **99.6%**. These
  tests are easier than real duets (the two voices come from different
  recordings), so real duets will score lower.
- Real duet, *A Whole New World* (Peabo Bryson / Regina Belle): in the solo
  verses, fingerprints matched the known singer on most lines where pitch
  failed (e.g. the female verse that pitch tagged M).
- Takes about 8 s per song extra on the CPU.

### Considered and not chosen

- A per-line gender classifier (e.g. an ECAPA voice-gender model, 98.7% on
  speech): needs PyTorch (removed for the Intel Mac), and judges each line on
  its own; adds little over fingerprints plus group naming.
- Using the video (faces): the user doesn't want to use the video.

### Limits

- Lines sung together are found from the audio only roughly (see the B
  update above); lyric marks (合) are exact.
- The model was trained on speech, not singing; very similar voices, heavy
  effects or backing vocals can still confuse it.
