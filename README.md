# Karaoke prototype

A minimal working version of the DimSing-style pipeline:

paste a YouTube link → download the video → AI vocal removal → AI lyrics with
per-word timing → play the video with a backing-track toggle and word-by-word
lyrics. Several songs are processed at once through a priority job queue.

```
app/
  config.py    settings (all overridable with environment variables)
  jobs.py      job queue: dedup cache, priorities, worker pools  <- the concurrency part
  pipeline.py  the stages: yt-dlp download, vocal separation, faster-whisper lyrics
  separators.py vocal separators: MDX-Net on OpenVINO (default) and Demucs
tools/
  bench_separation.py  compare separation speed/quality on a downloaded song
  server.py    FastAPI: shared playlist + processing API + media files
static/
  index.html   player (TV/PC) and phone remote (phones get it automatically)
data/          one folder per YouTube video ID = the cache
```

## Requirements (Windows)

1. **Python 3.11–3.14** (64-bit) from python.org. Tick "Add python.exe to PATH".
2. **ffmpeg**: nothing to do. A copy is installed with the Python packages (`imageio-ffmpeg`).
   If you have your own ffmpeg on PATH, that one is used instead.
3. **Deno** (recent yt-dlp versions need a JavaScript runtime to read YouTube):
   `winget install DenoLand.Deno`.
4. Disk: about 3 GB for PyTorch plus models. The models download automatically
   on first use (MDX-Net vocal model about 65 MB, Whisper "small" about 500 MB, "large-v3" about 3 GB).

## Setup and run (Windows)

```bat
cd D:\sandbox\karaoke-prototype
setup.bat          :: creates .venv and installs everything (takes a while)
run-mock.bat       :: try the UI first: fake songs, no YouTube or models needed
run.bat            :: the real thing
```

Open **http://localhost:8000**, click *Start player*, then paste a YouTube link.
On a phone on the same Wi-Fi, open `http://<your-PC-IP>:8000` to add
songs (allow Python through Windows Firewall when asked; `ipconfig` shows the IP).

The first real song is slower because the models download. Vocal removal uses
an MDX-Net model on Intel OpenVINO, which is about 2× faster than Demucs on CPU
and also runs on an Intel Arc / integrated GPU when one is found (the start-up
log says which device it picked).

### Measuring vocal-removal speed on your machine

```bat
.venv\Scripts\activate
python tools\bench_separation.py data\<video_id>\video.mp4 --seconds 60
```

It times each option on a 60-second excerpt (the old Demucs setup, faster
Demucs, MDX-Net on CPU, and on your GPU if available), and writes each result
to `bench_out\` so you can listen to them. Add `--threads 6` to try only the
fast cores of a hybrid Intel CPU.

## Setup and run (macOS)

Needs **Python 3.11–3.13** on Apple Silicon (M1–M4), or **3.11–3.12** on an
Intel Mac: the python.org installer or `brew install python@3.12`. ffmpeg is
included. YouTube downloads need a JavaScript runtime: setup offers to install
**Deno** with its official installer (`curl -fsSL https://deno.land/install.sh | sh`,
a ready-made program for macOS 12 or newer, no Xcode needed). Don't use
`brew install deno` on macOS 12: Homebrew no longer supports it and tries to
compile Deno, which demands Xcode 15. Node.js, QuickJS or Bun also work if you
already have one.

```zsh
cd karaoke-prototype
zsh setup.sh       # creates .venv and installs everything (safe to re-run)
./run-mock.sh      # try the UI with fake songs
./run.sh           # the real thing
```

Open http://localhost:8000; phones use `http://<your-Mac-IP>:8000` (setup
prints it). Allow incoming connections for Python if macOS asks.

- Apple Silicon gets the same packages as Windows. Vocal removal and lyrics
  run on the CPU (OpenVINO and Whisper have no Apple GPU support here).
- Intel Macs install without Demucs (a library it needs has no Intel-Mac
  build) and without PyTorch. The default MDX-Net separator doesn't need
  PyTorch, and keeping it out matters: PyTorch and Whisper each bundle Intel's
  OpenMP runtime, and having both in the server crashed it ("OMP: Error #15",
  then a segmentation fault). Re-running `zsh setup.sh` removes a PyTorch an
  older setup installed.
- A `.venv` copied over from Windows is set aside and a Mac one is created.
- HTTPS certificates: Python from python.org has no root certificates of its
  own, which made downloads fail with `CERTIFICATE_VERIFY_FAILED`. The app now
  verifies against the macOS Keychain (the `truststore` package). If you still
  see that error, run *Install Certificates.command* in Applications > Python 3.x.

### NVIDIA GPU (optional, much faster)

After `setup.bat` (check pytorch.org/get-started for the CUDA version that matches your driver):

```bat
.venv\Scripts\activate
pip install torch --index-url https://download.pytorch.org/whl/cu128 --force-reinstall
set KARAOKE_WHISPER_MODEL=large-v3
set KARAOKE_SEPARATOR=demucs
run.bat
```

faster-whisper on GPU also needs the cuBLAS/cuDNN libraries; if it complains, run
`pip install nvidia-cublas-cu12 nvidia-cudnn-cu12==9.*`, or set `KARAOKE_DEVICE=cpu`.

## Using it

- **Adding a song**: *Add to playlist* (or *+ Add* in History) opens a popup
  with two options, then *Add* adds it:
  - **Skip lyrics**: only remove the vocals (saves the lyrics step). The vocals are kept, so adding the song again with
  the box unticked extracts the lyrics later, without redoing anything else.
  - **Male + female duet**: each lyric line gets
  an **M** or **F** tag, with male lines filling blue and female lines pink.
  It works from pitch alone (see `app/voices.py`): the song's two voice
  ranges are found automatically, then each line goes to whichever singer's
  evidence is stronger. Expect roughly 6-8 of 10 lines right: a man's high
  chorus lines and a woman's low verse lines can get the other tag, and lines
  sung together get one of the two. Adding a finished song again from History
  with the box ticked only runs the tagging (about 2 s), because a small copy
  of the vocals (`vocals16k.flac`, ~4 MB) is kept. Labels can be changed, e.g.
  `set KARAOKE_VOICE_LABELS=男,女`.
- **History** tab: every song downloaded so far, newest played first, with a
  search box. *+ Add* puts it back in the playlist; finished songs are ready
  instantly.
- **Removing / deleting**: ✕ on a playlist song takes it off the playlist (it
  stays in History; the song playing now needs a second tap). **Delete** in
  History removes the song's files from `data\` for good, after a second tap
  to confirm, and takes it off the playlist. A song that is being processed
  right now can't be deleted until it finishes.
- **Lyrics size**: **A− / A+** on the TV's player bar or the phone card, 50% to
  300% in 10% steps (default 120%). Shared like the other TV settings.
- **Lyrics: on/off** (player bar) hides the lyrics; it shows *No lyrics* for
  songs without them.
- **Starting the TV player**: browsers only play sound after a click on the
  page, so the TV shows *Start player*. You can also start it from a phone
  (*▶ Start TV player* on the phone card). If the TV browser blocks sound, a
  phone start plays muted and the TV shows "click or press a key on the TV
  once" to turn sound on. To skip all that, start the TV browser with
  `--autoplay-policy=no-user-gesture-required` (Chrome and Edge; see the
  kiosk commands below): the TV then starts by itself, with sound.
- **Joining from a phone**: the TV's start screen and idle screen show a QR
  code (and the address) for phones to scan. Scanning it also starts the TV
  player (the code's address ends in `?start=1`); typing the address shown
  under it just joins. On a phone, **Invite friends → Show QR** (Dashboard) shows a big
  QR code for others to scan; that one only joins, it doesn't start the TV. It uses this computer's network
  address, not `localhost`. If it picks the wrong one (e.g. with a VPN), set
  `KARAOKE_PUBLIC_URL=http://192.168.x.x:8000`.
- **Phone layout**: three tabs at the bottom. **Dashboard**: what's on the TV,
  Start, Pause, Original / Backing track, Lyrics on/off and size, TV
  fullscreen, Skip, and *Invite friends* (QR code). **Playlist**: add a song
  (link, then the Add popup) and the queue. **History**: search and re-add or
  delete songs. The phone remembers the last tab.
- **Phones and tablets always get the phone view** (add songs + remote control),
  never the player. Add `?tv` to the address to use a tablet as the screen, or
  `?remote` to force the phone view on a PC.
- The phone's *On the TV* card has Pause / Resume, Original / Backing track,
  Lyrics on/off, TV fullscreen and Skip (tap twice); it is always shown, even with nothing playing. These settings are
  shared by the TV and all phones: change them anywhere and the TV follows
  within a second.
- **TV fullscreen**: browsers only allow real fullscreen from a click on the TV
  page itself, so from a phone the TV switches to a full-window view (video
  fills the browser, panels hidden). Put the TV browser in fullscreen once with
  **F11**, or start it in kiosk mode (`msedge --kiosk http://localhost:8000
  --edge-kiosk-type=fullscreen --autoplay-policy=no-user-gesture-required`, or
  `chrome --kiosk --autoplay-policy=no-user-gesture-required http://localhost:8000`),
  and it looks exactly like fullscreen. On the TV: **F** toggles, **Esc**
  exits, or double-click the video.

## Settings (environment variables)

| Variable | Default | Meaning |
|---|---|---|
| `KARAOKE_DOWNLOAD_WORKERS` | 2 | songs downloading at the same time |
| `KARAOKE_MODEL_WORKERS` | 1 | songs on the AI models at the same time (2–3 on a big GPU) |
| `KARAOKE_WHISPER_MODEL` | small | tiny / base / small / medium / large-v3 |
| `KARAOKE_SEPARATOR` | mdx | mdx (OpenVINO, Intel CPU/GPU) or demucs (PyTorch, best with NVIDIA) |
| `KARAOKE_MDX_MODEL` | Kim_Vocal_2 | or UVR-MDX-NET-Voc_FT |
| `KARAOKE_MDX_DEVICE` | auto | auto (GPU if it passes a check, else CPU) / GPU / CPU |
| `KARAOKE_DEMUCS_MODEL` | htdemucs | used when `KARAOKE_SEPARATOR=demucs` |
| `KARAOKE_DEMUCS_OVERLAP` | 0.1 | chunk overlap for Demucs (old setting was 0.25) |
| `KARAOKE_DEVICE` | auto | PyTorch device for Whisper and Demucs: auto / cuda / cpu |
| `KARAOKE_MAX_DURATION` | 600 | reject videos longer than this (seconds) |
| `KARAOKE_MAX_HEIGHT` | 720 | video resolution to download |
| `KARAOKE_MOCK` | 0 | 1 = fake stages for testing |

Pick the language in the form when you know it. Cantonese (`yue`) works properly
only with `large-v3`. Smaller models have no Cantonese setting, so the app
transcribes Cantonese songs as Chinese with them (Chinese characters, with
more wrong characters than `large-v3`). On CPU, `large-v3-turbo` is a good middle ground: much
faster than `large-v3` and far better than `small`.

## How it works

**Processing several songs at once (`app/jobs.py`).**
- *Cache and dedup:* each song is keyed by its YouTube video ID. Finished songs
  stay in `data/<id>/` and are reused instantly, even after a restart. A request
  for a song that is already in progress attaches to the existing job.
- *Two queues, two worker pools:* downloads (network-bound) and model work
  (GPU/CPU-bound) have separate pools, so songs overlap like an assembly line.
  While one song is on the GPU, the next ones are downloading.
- *Priority:* a song's position in the playlist is its priority. The song about
  to play is always processed first, and moving a song up re-prioritises it.
- Models are loaded once and shared by the worker threads. The heavy libraries
  release Python's GIL, so the threads really run in parallel.

**Keeping audio in sync.** The model audio is extracted from the *same* mp4
the browser plays, so the instrumental lines up sample-for-sample. In
backing-track mode the video is muted and the instrumental plays alongside,
with its position corrected whenever it drifts more than 80 ms from the video.

**Lyrics.** Whisper runs on the *isolated vocals* (far more accurate than on
the full mix) with word timestamps and voice-activity filtering. Long segments
are split into karaoke-sized lines at pauses. Use the − / + buttons if the
lyrics run early or late.

## API

| Method | Path | |
|---|---|---|
| POST | `/api/queue` | `{url, language?, singer?, skip_lyrics?, duet?}` add to playlist (`url` may be a bare video ID) |
| POST | `/api/next` | finish current song |
| POST | `/api/player` | `{mode?: "inst"\|"orig", show_lyrics?, fullscreen?, paused?, lyrics_scale?, started?}` TV audio mode / lyrics / fullscreen / pause / lyrics size |
| POST | `/api/playlist/{i}/remove`, `/up` | edit playlist |
| GET | `/api/state` | playlist + queue status (the page polls this every second) |
| POST | `/api/process` | `{url, language?, priority?, skip_lyrics?}` process without the playlist |
| GET | `/api/join`, `/qr.svg?start=1\|0` | address phones should open, and its QR code (start=1 also starts the TV) |
| GET | `/api/history` | every downloaded song, most recently played first |
| DELETE | `/api/songs/{id}` | delete a song's files for good |
| GET | `/api/jobs`, `/api/jobs/{id}` | job status |
| GET | `/media/{id}/video.mp4`, `instrumental.mp3`, `lyrics.json` | results |

## Not included (next steps)

- Rooms with QR codes; this prototype has one shared room.
- Pitch line and scoring: run a pitch tracker (torchcrepe / librosa.pyin) over
  the vocals in the model stage, and compare it with the singer's microphone.
- Redis-backed queue and separate worker machines for scale-out.
- WebSockets instead of polling.

## Legal note

Downloading from YouTube is against YouTube's Terms of Service, and songs,
lyrics and videos are copyrighted. Keep this to personal experiments.
