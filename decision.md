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
