# Radio Stream Archiver + Transcriber

Records an internet radio stream in hourly segments and transcribes the
**speech** in each hour to searchable text — skipping music so the transcriber
doesn't hallucinate over songs — alongside per-hour quality checks and a
best-effort schedule tag, all stored in one JSON sidecar per hour.

The station is **not hardcoded** — the stream URL, schedule URL, and related
settings live in `config.json` (see [Configuration](#configuration)). The
repo ships `config.example.json` with generic placeholder URLs; copy it to
`config.json` and edit it to point at your station. The author runs this
against [UCLA Radio](https://uclaradio.com/), but any station that publishes
an Icecast-style MP3 stream should work.

| Script | Job |
|---|---|
| `config.py` | Loads `config.json` (station-specific settings). |
| `archive.py` | Records the stream into hourly `.mp3` files. Recording only. |
| `transcribe.py` | Per-hour processing: transcript + quality checks + schedule hint, written to one `.txt` + `.json` sidecar. |
| `quality.py` | Shared audio quality checks (size / silence / volume / decode errors) + ffmpeg/ffprobe location + `is_off_air` helper. |
| `schedule_archive.py` | Archives the published show schedule daily and parses it into structured snapshots. |
| `merge_archives.py` | Reconciles two parallel-recorder archives into one canonical tree (winner-takes-all per hour, plus cross-fill splicing for partially-bad hours). |
| `purge_silent.py` | Tombstone-style cleanup: deletes the `.mp3` for sidecars marked `is_off_air`, keeps `.json` and `.txt`. Default dry-run. |

---

## 1. Requirements

- **OS: Windows, Linux, or macOS** (incl. Apple Silicon). The code is
  cross-platform; per-OS specifics are in [§10](#10-platform-notes).
- **Python 3.11–3.13** (developed on 3.13).
- **ffmpeg + ffprobe** — used by `archive.py` for recording, and by
  `transcribe.py`/`quality.py` for the silence check and the full
  null-muxer decode pass that catches mid-stream frame errors.
  (Whisper itself decodes audio via the bundled PyAV/`av` library, so if
  ffmpeg is missing transcription still works — quality just records a
  `tool_error`.)
- **~250 MB disk** for the Whisper model (downloaded automatically on first run).
- **No GPU required.** Transcription runs on CPU. An NVIDIA GPU can optionally
  accelerate it; Intel/AMD integrated GPUs are *not* usable by faster-whisper.
  See [§6 Design decisions & tradeoffs](#6-design-decisions--tradeoffs).

### Installing ffmpeg

| OS | Command |
|---|---|
| Windows | `winget install Gyan.FFmpeg` |
| macOS | `brew install ffmpeg` |
| Debian/Ubuntu | `sudo apt install ffmpeg` |

If `ffmpeg`/`ffprobe` aren't on `PATH`, they're auto-discovered from the usual
install dirs — winget/chocolatey/scoop on Windows, `/opt/homebrew/bin`,
`/usr/local/bin`, `/usr/bin`, `/snap/bin` on Linux/macOS — or you can point at
them with the `FFMPEG`/`FFPROBE` env vars (see `_find_tool` in `quality.py` and
[§10](#10-platform-notes)).

---

## 2. Installation

Uses [uv](https://docs.astral.sh/uv/) for environment + package management.
Install uv first if you don't have it (`pipx install uv`, `brew install uv`, or
the standalone installer at <https://docs.astral.sh/uv/getting-started/installation/>).

```bash
# from the project directory
uv venv                          # creates .venv/

# activate the venv
#   Windows (PowerShell):  .\.venv\Scripts\Activate.ps1
#   macOS/Linux:           source .venv/bin/activate

uv pip install faster-whisper
```

`faster-whisper` pulls in everything the transcriber needs, including
`onnxruntime` (for the bundled Silero VAD) and `av` (for audio decoding). No
PyTorch or TensorFlow required.

> **venv binary path.** Once activated, just use `python`. When calling the venv
> directly (e.g. in a scheduled job), the interpreter lives at
> `.venv\Scripts\python.exe` on Windows and `.venv/bin/python` on Linux/macOS.
> Examples below use the activated `python`; the service templates in
> [§10](#10-platform-notes) use the full path.

**Versions this was built/tested against** (as of 2026-06):

```
faster-whisper 1.2.1
ctranslate2    4.7.2
onnxruntime    1.26.0
av             17.0.1
```

To reproduce exactly on another machine, pin these in a `requirements.txt`:

```
faster-whisper==1.2.1
```

(That single pin drags in compatible `ctranslate2` / `onnxruntime` / `av`.)

> **First-run model download:** the first transcription downloads the model
> (`small.en`, ~250 MB) from Hugging Face into `~/.cache/huggingface`. After
> that it runs fully offline. On Windows you may see a harmless symlink warning
> — caching still works, it just uses a bit more disk.

### Configuration

Station-specific settings live in **`config.json`** at the project root — the
code references config keys only, never a specific station. Copy the template
and edit it:

```bash
cp config.example.json config.json   # then edit the URLs
```

This is exactly what `config.example.json` contains:

```json
{
  "label": "My Radio Station",
  "stream":   { "url": "https://stream.example.com/listen/my_station/radio.mp3", "bitrate_kbps": 192 },
  "schedule": { "url": "https://example.com/shows/", "source_type": "r34ics" },
  "paths":    { "archive_dir": "archive", "schedule_dir": "schedule" }
}
```

| Key | Meaning |
|---|---|
| `label` | Display name, used only in log lines. |
| `stream.url` | The MP3 stream to record. (This `/listen/<station>/` mount is the **AzuraCast** convention; any Icecast-style MP3 URL works — recording is framework-agnostic.) |
| `stream.bitrate_kbps` | Stream bitrate, used by the size quality-check to estimate the expected file size. |
| `schedule.url` | The page that publishes the show schedule. |
| `schedule.source_type` | Which schedule adapter to use. Only **`r34ics`** is implemented — it scrapes the [ICS Calendar / r34ics](https://wordpress.org/plugins/ics-calendar/) WordPress plugin's AJAX-rendered week grid. Other systems would add a new adapter (see [§9](#9-show-schedule-archiving)). |
| `paths.archive_dir` / `paths.schedule_dir` | Output directories. |

`stream.url` and `schedule.url` are required; everything else has a default. The
config path can be overridden with the `RADIO_CONFIG` env var (handy for running
multiple stations from one checkout). Algorithm settings that aren't
station-specific (Whisper model, VAD/silence thresholds, segment length) stay as
code defaults / env vars — see [§8](#8-configuration-reference).

---

## 3. Usage

> If you want transcription, **run from the venv** (where faster-whisper lives).
> Plain `python archive.py` still records fine without it.

### Recommended: two decoupled processes

Run the recorder and the transcriber as **separate long-running processes**. The
recorder only records; the transcriber watches for completed hours. This keeps a
live capture from ever needing a restart when you change transcription, and lets
you stop/upgrade transcription freely.

```bash
python archive.py              # process 1: record only (never needs restarting)
python transcribe.py --watch   # process 2: transcribe each hour as it completes
```

`--watch` scans `archive/` periodically, transcribing any completed segment that
lacks a transcript, and **leaves the in-progress hour alone** until it's done.
A file is treated as in-progress when its mtime is recent *and* its size is
below the expected full-hour size; the check is applied to every candidate
so `.partN` rotation siblings don't fool it. Ctrl+C stops it cleanly after
the current file.

### Recording options

```bash
python archive.py              # record forever, hourly segments (recording only)
python archive.py --test       # record one 60-second sample, then exit
python archive.py --transcribe # single-process alternative: record AND transcribe
                               #   in one process (couples their lifecycles — prefer
                               #   the two-process setup above)
```

In `--test` mode the output is a single fixed-duration capture (`ffmpeg -t 60`)
that lands in `archive/test-YYYY-MM-DD_HH-MM-SS.mp3` at the top of the archive
directory — so test files don't mix with the hourly `archive/YYYY/MM/` tree
and successive runs don't overwrite each other. The startup log line "Output:"
prints the exact path it'll use.

Output lands in `archive/YYYY/MM/YYYY-MM-DD_HH-00.mp3` — files are grouped
by year and month so a multi-year archive stays browsable. The `.txt` and
`.json` sidecars sit next to each `.mp3` in the same subdirectory.
Recording logs go to `archiver.log`. Quality results are stored per hour in
the transcript JSON (see [§4](#4-output-format)), not in a separate log — to
(re-)compute quality for files lacking a transcript, just transcribe them.

> **Legacy flat archives still work.** `transcribe.py`, `archive.py
> --transcribe`, and `merge_archives.py` find files at any depth under the
> archive directory (recursive glob), so an existing flat
> `archive/YYYY-MM-DD_HH-00.mp3` layout is read transparently alongside the
> new nested layout. Migrating old files into `archive/YYYY/MM/` is
> optional and can be done with a one-shot shell script when convenient.

### Restart resilience: `.partN.mp3` siblings

If `archive.py` (or the underlying ffmpeg) restarts mid-hour — script
upgrade, reboot, stream drop, anything — the existing in-progress
`YYYY-MM-DD_HH-00.mp3` is renamed to `YYYY-MM-DD_HH-00.partN.mp3` before
the new ffmpeg invocation, so the new ffmpeg writes the canonical name
fresh instead of truncating the old one. `N` is the smallest free integer,
so multiple restarts within the same hour stack as `.part0`, `.part1`, …

```
archive/2026/06/
  2026-06-04_01-00.part0.mp3   # first run, 01:31–01:45
  2026-06-04_01-00.part1.mp3   # second run, 01:45–01:52
  2026-06-04_01-00.mp3         # third run, 01:55–02:00 (the live one)
  2026-06-04_02-00.mp3         # third run continued past the hour rollover
```

`transcribe.py` recognises the `.partN` suffix:
- `schedule_hint` still gets the right hour (the suffix is stripped before
  the timestamp is parsed).
- `quality.analyze` is called with `partial=True`, which keeps the
  measured size/silence/decode values but sets `size.ok = None` so the
  short-file check doesn't flag it as a stream outage.
- The `.txt` header shows `quality: PARTIAL HOUR (NN.N MB)` instead of
  `SMALL SEGMENT`.

Each `.partN.mp3` gets its own pair of sidecars, so a restart-y hour
produces multiple transcripts — no automatic merge yet; if you want one
combined file per hour, `ffmpeg -f concat -c copy` can splice them after
the fact (this is what `merge_archives.py` does for redundant-recorder
splices).

### Transcribing (standalone / backfill)

```bash
python transcribe.py FILE.mp3 [FILE2.mp3 ...]   # specific files
python transcribe.py --all                      # every archive/*.mp3 lacking a transcript
python transcribe.py --all --force              # re-transcribe everything
python transcribe.py --watch                    # run continuously (see above)
```

`--all` and `--watch` both skip the in-progress hour and any file that already
has a `.json` sidecar, so they're **idempotent and resumable** — safe to run
anytime (recorder running or not); rerun and they continue where they left off.
Only `--force` re-transcribes existing files.

### Requesting a self-restart (`/tmp/RADIO_ARCH_RESTART_REQUIRED`)

When the services are under systemd with `Restart=always`, you can ask them
to swap in newly-deployed code **without sudo** by touching a sentinel file:

```bash
touch /tmp/RADIO_ARCH_RESTART_REQUIRED
```

What happens:

- **`transcribe.py --watch`** notices within ~1 second, finishes the current
  file (or exits immediately if idle), and exits. systemd brings it back up
  within a few seconds with whatever code is now on disk.
- **`archive.py`** notices within ~60 seconds, schedules its exit for the
  **next clock-hour boundary + 5 seconds** so the in-progress hour is fully
  captured. ffmpeg gets the same graceful-shutdown signal as a manual
  `systemctl stop`. On restart, the brief next-hour file (5 seconds of
  audio) is rotated to `.part0.mp3` and the new ffmpeg writes the canonical
  filename fresh — exactly the existing `.partN` recovery flow.

Worst-case wait for the recorder is just under one hour (touch at HH:00:01).
Best case is seconds (touch at HH:59:59). If you need an immediate restart,
`sudo systemctl restart radio-archive` still works.

**Mtime-based detection** means the sentinel left over on disk between
touches doesn't keep re-triggering: each service baselines the file's mtime
at startup and reacts only to a strictly newer mtime, so a touch only
triggers one restart cycle. The **recorder is responsible for unlinking the
sentinel** when it finally exits — by then the (much faster) transcriber has
already responded, so there's no race where the sentinel disappears before
the recorder sees it.

### Tuning via environment variables

| Variable | Default | Effect |
|---|---|---|
| `WHISPER_MODEL` | `small.en` | Model size. `base.en` = faster/rougher; `medium.en` = slower/more accurate. Drop `.en` for non-English. |
| `WHISPER_COMPUTE` | `int8` | Compute type — passed through to CTranslate2. Common values: `int8` (default, fast on CPU), `int8_float32`, `float16` (CUDA), `float32` (slower, slightly more accurate). |

```bash
# example: higher accuracy
WHISPER_MODEL=medium.en python transcribe.py --all      # macOS/Linux
$env:WHISPER_MODEL="medium.en"; python transcribe.py --all   # PowerShell
```

---

## 4. Output format

For each `archive/2026/05/2026-05-24_12-00.mp3`, the transcriber writes two
sidecars next to it (same subdirectory):

**`2026-05-24_12-00.txt`** — human-readable:

```
# Transcript: 2026-05-24_12-00.mp3
# model=small.en  faster-whisper=1.2.1  talk=96%  segments=49  generated=2026-05-25T22:10:08+00:00
# quality: OK
# schedule hint (per published schedule, may differ): Some Show with DJ X (12:00-13:00)
# Full quality/settings/versions/machine/schedule recorded in the .json sidecar.

[0:00:00 -> 0:00:08]  were sort of their militants was kind of a fallout of...
[0:00:08 -> 0:00:13]  I found a New York Times article from 1976 that was...
```

**`2026-05-24_12-00.json`** — structured, for search/programmatic use, with a
full **provenance** block so a future reader can decide whether re-transcribing
(newer model, stronger machine, different params) is worthwhile:

```json
{
  "schema_version": 3,
  "file": "2026-05-24_12-00.mp3",
  "generated_utc": "2026-05-25T22:18:39+00:00",
  "audio_seconds": 3599.9,
  "speech_seconds": 412.3,
  "talk_ratio": 0.115,
  "language": "en",
  "language_probability": 1.0,
  "transcribe_seconds": 77.2,
  "realtime_factor": 46.6,
  "segment_count": 87,
  "quality": {
    "expected_seconds": 3600,
    "partial": false,
    "size": { "actual_mb": 86.4, "expected_mb": 86.4, "ratio": 1.0, "ok": true },
    "silence_periods": [
      { "start": 0.0, "end": 219.7, "duration": 219.7 }
    ],
    "max_volume_db": 0.0,
    "mean_volume_db": -10.7,
    "decode_errors": null,
    "decode_exit_code": 0,
    "tool_error": null,
    "thresholds": { "size_ratio_warn": 0.8, "silence_threshold": "-40dB", "silence_min_secs": 10, "stream_bitrate_kbps": 192, "off_air_peak_db": -30, "off_air_min_audio_s": 3300 },
    "is_off_air": false,
    "ok": false
  },
  "schedule_hint": {
    "source": "https://example.com/shows/",
    "note": "Best-effort match from the published weekly schedule; the actual broadcast may differ and is not guaranteed.",
    "snapshot_date": "2026-05-26",
    "snapshot_after_recording": true,
    "day_of_week": "Sunday",
    "listed_shows": [
      { "start": "12:00", "end": "13:00", "title": "Some Show with DJ X", "status": "confirmed" }
    ]
  },
  "provenance": {
    "settings": {
      "model": "small.en", "device": "cpu", "compute_type": "int8",
      "vad_filter": true, "vad_parameters": { "threshold": 0.6, "...": "..." },
      "beam_size": 5, "condition_on_previous_text": false
    },
    "versions": {
      "python": "3.13.13", "faster_whisper": "1.2.1",
      "ctranslate2": "4.7.2", "onnxruntime": "1.26.0", "av": "17.0.1"
    },
    "machine": {
      "platform": "Windows-11-10.0.26200-SP0",
      "processor": "Intel64 Family 6 ...", "cpu_count": 14
    }
  },
  "segments": [
    { "start": 12.34, "end": 18.9, "text": "..." }
  ]
}
```

Field notes for future evaluation:
- `realtime_factor` = audio seconds ÷ wall-clock seconds (higher = faster). Lets
  you estimate the cost of re-running with a heavier model or on another machine.
- `provenance.settings` = the model + decode/VAD parameters used; compare against
  what a newer/bigger model would offer.
- `provenance.versions` / `provenance.machine` = the exact libraries and hardware,
  so you can tell whether a result predates a model/library upgrade.
- `provenance.backfilled: true` marks sidecars whose provenance was reconstructed
  after the fact (settings/versions were unchanged, so the values are accurate;
  the flag just means they weren't emitted at original run time).
- `schema_version` guards the layout — bumped if the JSON shape changes
  (1 = transcript + provenance; 2 = added `schedule_hint`; 3 = added `quality`).
- `quality` = the per-hour audio checks (computed by `quality.py`): `size`
  (actual vs expected bytes — catches stream outages), `silence_periods` (dead
  air ≥ threshold), `max_volume_db` / `mean_volume_db` (peak / mean amplitude
  in dBFS, from `volumedetect`), and `decode_errors` (full ffmpeg null-muxer
  decode pass — every frame is decoded so mid-stream corruption is caught,
  not only unopenable containers; `decode_exit_code` records ffmpeg's return
  code). `ok` is true only if all pass; `tool_error` is set instead if
  ffmpeg/ffprobe was unavailable. `thresholds` records the limits used, so a
  future reader knows how it was judged.
- `is_off_air` = true only when the file is positively identified as
  broadcast-side carrier noise: `max_volume_db` below the
  `off_air_peak_db` threshold (−30 dB by default, so inaudible to a
  listener), `speech_seconds == 0` (Silero VAD found nothing), audio
  duration ≥ `off_air_min_audio_s` (a near-full hour, not a `.partN`
  partial), and no `tool_error`. The intent is **strict**: an
  instrumental music hour with peaks at 0 dB is *not* off-air; even one
  VAD frame keeps the file. See [§12](#12-off-air-detection-and-purging)
  for the purge workflow.
- `schedule_hint` = a **non-authoritative** guess at the show for that hour, from
  the archived schedule (see [§9](#9-show-schedule-archiving)). Names are chosen
  to avoid implying authority: `listed_shows` = "what the published grid listed",
  not what aired. `snapshot_after_recording: true` warns the only schedule
  snapshot available postdates the recording (so it may not reflect that week).
  Any lookup problem appears as `schedule_hint.error` rather than failing the run.

A music-only hour writes valid sidecars with `segment_count: 0` and a
`(no speech detected)` note, so it isn't re-processed on the next pass.

> **Time formats differ between the sidecars.** The `.txt` header uses
> `H:MM:SS` (e.g. `[0:00:08 -> 0:00:13]`) for human readability; the `.json`
> stores `start` / `end` as raw seconds (rounded to 2 decimals) so downstream
> tooling doesn't have to parse a clock string.

---

## 5. Architecture

```
   process 1: archive.py                  process 2: transcribe.py --watch
  ┌───────────────────────────┐          ┌──────────────────────────────────────┐
  │ ffmpeg segment muxer       │          │ poll archive/ for a COMPLETED hour     │
  │ -c copy, clock-aligned 1h, │  .mp3    │ lacking a .json sidecar                │
  │ auto-reconnect             │ ───────▶ │   │                                    │
  │ archive/YYYY/MM/...mp3     │  files   │   ├─ Silero VAD ─▶ speech ─▶ faster-   │
  └───────────────────────────┘          │   │  (drops music/silence)  whisper    │
     records only; never needs           │   ├─ quality.py: size/silence/errors    │
     restarting for transcription        │   └─ schedule_hint (schedule_archive)   │
                                          │            │                           │
                                          │            ▼   .txt + .json (one/hour)  │
                                          └──────────────────────────────────────┘
```

### `archive.py` (records only)
- Uses **ffmpeg's segment muxer** with `-segment_atclocktime` so files break on
  the clock hour, and `-c copy` so the MP3 is saved without re-encoding (cheap,
  lossless passthrough).
- `-reconnect` flags make ffmpeg recover from stream drops automatically; the
  outer loop also restarts ffmpeg if it exits.
- It does no per-hour processing — that's `transcribe.py`'s job. (The optional
  `--transcribe` flag runs that processing inline via a watcher thread for a
  single-process setup, but the recommended setup keeps them separate.)

### `transcribe.py` (per-hour processing → one JSON sidecar)
- **Silero VAD first** (bundled with faster-whisper, runs via onnxruntime). It
  isolates speech and discards music/silence *before* anything reaches Whisper.
  This is the key step that prevents hallucinated lyrics/garbage over songs.
- **faster-whisper** transcribes only the speech regions and remaps timestamps
  back onto the original hour's timeline.
- **Quality checks** (`quality.py`) and a **`schedule_hint`** are computed for the
  same hour and folded into the one JSON sidecar — so each hour has a single
  record, written by a single process (no write races), and skipped on restart
  if the `.json` already exists (the checks are never redundantly re-run).
- The model is a lazily-loaded singleton, so a `--all`/`--watch` run loads it
  once and reuses it.
- **`--watch`** decouples this from recording: it polls `archive/` for
  completed-but-untranscribed segments (skipping the in-progress hour), so the
  recorder never has to restart when you change transcription. Recommended
  runtime; `archive.py --transcribe` remains as a single-process alternative.

### `quality.py`
- Shared, pure functions: `analyze(path)` returns a structured dict (size /
  silence / decode-error results) and never raises — a missing ffmpeg/ffprobe
  is reported as `tool_error`. Decode errors come from a full ffmpeg
  null-muxer pass (`-v error -i FILE -f null -`), so mid-stream frame errors
  are caught, not only unopenable containers. Also houses the winget-aware
  ffmpeg/ffprobe locator used by `archive.py` for recording.

### `config.py`
- Loads `config.json` once at import (raising a clear error if it's missing or
  if a required URL is absent), so the other modules can read station-specific
  values without re-parsing.

### `schedule_archive.py`
- The shows page renders its calendar client-side via the `r34ics` "ICS
  Calendar" plugin, so the static HTML is an empty container. The script GETs
  the page (for a fresh nonce + args), then POSTs to `admin-ajax.php` to get the
  rendered week grid — exactly what the browser does.
- It saves the **raw page HTML, raw calendar HTML, and a parsed JSON snapshot**
  per day. Raw HTML is always kept, so a future format change can be re-parsed
  retroactively (`--reparse`).
- Standard-library only (`urllib` + `html.parser`) — no extra dependencies.

---

## 6. Design decisions & tradeoffs

**Why faster-whisper (not openai-whisper or whisper.cpp).**
CTranslate2-backed, so it's fast on CPU, quantizes to int8, and needs neither
PyTorch nor TensorFlow. Lightest path to a working CPU pipeline.

**Why CPU + int8.**
The workload is batch — one hour of audio per hour of wall-clock — so CPU is
plenty even without a discrete GPU (the dev box runs talk-heavy audio at
~10× realtime and music hours near-instantly, since VAD discards them
before transcription). faster-whisper can't use Intel Arc / AMD integrated
GPUs, so on a machine without NVIDIA hardware CPU+int8 is the only path.
*On a machine with an NVIDIA GPU,* set `device="cuda"` and
`compute_type="float16"` in `transcribe.py` for a large speedup.

**Why Silero VAD instead of a speech/music classifier.**
The purpose-built option, **inaSpeechSegmenter** (true speech-vs-music
labels), pulls in TensorFlow, which we wanted to avoid to keep the install
lightweight. Silero VAD (bundled with faster-whisper, zero extra deps)
solves the core problem: instrumental music and silence are dropped cleanly.
- **Limitation:** VAD detects *voice activity*, not music specifically. Songs
  with prominent **sung vocals** can occasionally be sent to Whisper, producing
  messy lyric-ish lines. The VAD `threshold` is set conservatively (0.6) to
  reduce this. If clean speech/music separation matters, add inaSpeechSegmenter
  (needs a TensorFlow-compatible Python) and gate Whisper on its "speech"
  regions.

**Model choice (`small.en`).**
English-only is faster and more accurate on English speech than the
multilingual model, and avoids language-detection wobble on music. `base.en` is
faster but rougher; `medium.en` is notably more accurate but slower. Override
with `WHISPER_MODEL`.

**Hallucination mitigations** (in `transcribe.py`):
- `vad_filter=True` — the big one; no audio without detected speech reaches Whisper.
- `condition_on_previous_text=False` — stops runaway repetition loops.
- Conservative VAD threshold and minimum speech/silence durations.

**No music identification.**
This pipeline transcribes *talk*; it does not identify songs. That needs audio
fingerprinting — open-source **Chromaprint/AcoustID** (weak indie/college-radio
coverage) or commercial **ACRCloud / AudD** APIs (built for broadcast
monitoring). Not implemented here.

---

## 7. Running on a new machine — checklist

1. Install Python 3.11–3.13, ffmpeg, and [uv](https://docs.astral.sh/uv/).
2. `uv venv` and activate it.
3. `uv pip install faster-whisper` (or `uv pip install -r requirements.txt`).
4. Copy the example config and edit the URLs (required — the scripts won't
   import without `config.json`):
   - macOS / Linux: `cp config.example.json config.json`
   - Windows (PowerShell): `Copy-Item config.example.json config.json`
   Then set your `stream.url` / `schedule.url`.
5. `python archive.py --test` — confirms ffmpeg works and records a 60s sample.
6. `python transcribe.py <that-test-file>.mp3` — first run downloads the model;
   confirms transcription works.
7. For ongoing capture + transcription, run two processes: `python archive.py`
   (record only) and `python transcribe.py --watch` (transcribe as hours
   complete). See [§3](#3-usage).
8. (Optional) `python schedule_archive.py` to grab a schedule snapshot, then add
   the daily scheduler task ([§9](#9-show-schedule-archiving)).
9. (Optional, NVIDIA GPU) edit `transcribe.py`: `device="cuda"`,
   `compute_type="float16"`.
10. (Optional) if you are running a second redundant recorder, see
    [§11](#11-merging-two-redundant-recorders) for the merge tool. Run it from
    the venv — it imports `transcribe.py` and so needs faster-whisper.

---

## 8. Configuration reference

**Station-specific settings** live in `config.json` — see
[Configuration](#configuration). That's where the stream URL, schedule URL,
bitrate, paths, and label come from.

**Algorithm defaults** (not station-specific) are editable constants at the top
of each file:

**`quality.py`** — `DEFAULT_BITRATE_KBPS` (192 fallback; the real value comes
from `config.json`), `SIZE_RATIO_WARN` (0.80), `SILENCE_THRESHOLD` (`-40dB`),
`SILENCE_MIN_SECS` (10), `EXPECTED_SEGMENT_SECONDS` (3600).

**`transcribe.py`** — `MODEL_SIZE` (also `WHISPER_MODEL` env), `DEVICE`,
`COMPUTE_TYPE` (also `WHISPER_COMPUTE` env), and `VAD_PARAMETERS`.

**`archive.py`** — `SEGMENT_SECONDS` (3600), `RECONNECT_DELAY`.

---

## 9. Show-schedule archiving

The published schedule at the configured `schedule.url` is archived and parsed
by `schedule_archive.py`, then used to tag transcripts (the `schedule_hint`
field — see [§4](#4-output-format)). The default `r34ics` source adapter targets
the ICS Calendar WordPress plugin; pick whatever URL your station publishes its
schedule at and put it in `config.json`.

> **Not authoritative.** Shows are sometimes changed without the schedule being
> updated, and the page's format has changed many times. Treat every schedule
> field as *"what was listed"*, never *"what aired"*. Field names reflect this
> (`schedule_hint`, `listed_shows`) and a `note` repeats the caveat in the data.

### Commands

```bash
python schedule_archive.py              # fetch + archive today's snapshot
python schedule_archive.py --force      # re-fetch even if today's exists
python schedule_archive.py --reparse    # re-parse today's saved calendar HTML
python schedule_archive.py --reparse-all # re-parse EVERY archived day (see below)
python transcribe.py --retag            # refresh schedule_hint on existing
                                        #   transcripts (no re-transcription)
```

> **`--force` is undo-safe.** Before refetching it rotates the existing
> `YYYY-MM-DD.page.html` / `.calendar.html` / `.json` trio to `*.bak`. If
> the fetch fails, the previous trio is restored from `.bak` — so a bad
> re-fetch never destroys a good snapshot. On success the `.bak` files are
> left behind for a one-deep manual undo.

### Versioned parsers (surviving format changes)

The shows page has changed format many times and will again. To keep the whole
archive **regenerable forever**, parsing is built as a registry of per-format
parsers in `schedule_archive.py`:

- Each format era is a `ScheduleParser` subclass with `detect(html)` (does this
  parser recognize the format?) and `parse(html)` (extract the week grid). The
  current one is `R34icsWeekGridParser`, version `2026-05_r34ics_week_grid`.
- `select_parser()` picks the right parser for any HTML blob by detection, so a
  given day is always handled by the parser for *its* era. Each snapshot records
  the `parser_version` that produced it (and so does each transcript's
  `schedule_hint`).
- **Parsers are never deleted.** When the format changes:
  1. Write a new `ScheduleParser` subclass with a `detect()` specific to the new
     markup and a `parse()` for it.
  2. Append it to the `PARSERS` list — **do not touch the existing parsers**.
  3. Run `python schedule_archive.py --reparse-all` to regenerate every day's
     JSON from the saved raw HTML; old days match old parsers, new days match
     the new one.
  4. `python transcribe.py --retag` to refresh transcript hints.

So in, say, 2035 with 15 historical formats, all 15 parsers coexist and
`--reparse-all` rebuilds the entire history correctly — because the raw HTML for
every day was archived and each parser still recognizes its own era. If no
parser matches a file, that day's snapshot just records `parse_ok: false` with an
error and keeps its raw HTML, waiting for a parser to be added.

A parser whose `detect()` raises is logged at warning level by
`select_parser()` and skipped, so a buggy detector for a new format is
debuggable from the logs rather than silently masking other parsers.

### Running it daily

`schedule_archive.py` archives one dated snapshot per run. Schedule it once a day
with the OS scheduler:

- **Windows (Task Scheduler):**
  ```powershell
  schtasks /create /tn "Radio Schedule" /tr `
    "<project-dir>\.venv\Scripts\python.exe <project-dir>\schedule_archive.py" `
    /sc daily /st 04:00
  ```
  (Replace `<project-dir>` with the absolute path to your checkout, e.g. `C:\Users\you\radio-archiver`.)
- **macOS/Linux (cron):** `0 4 * * * cd /path/to/proj && .venv/bin/python schedule_archive.py`
  — cron runs with a minimal `PATH`; if ffmpeg isn't found, prepend
  `FFMPEG=/usr/bin/ffmpeg FFPROBE=/usr/bin/ffprobe` or set `PATH` (see
  [§10](#10-platform-notes)). For a managed setup, prefer the systemd timer there.

Because tagging uses the **nearest snapshot on or before** each recording, the
more days you archive, the more accurate historical tagging becomes. Early
snapshots that postdate existing recordings are flagged
`snapshot_after_recording: true`. After the daily job has run for a while, you
can `python transcribe.py --retag` to re-tag older transcripts against the
better-matched snapshots.

### Artifacts (`schedule/`)

```
2026-05-26.page.html      raw shows page (source of the AJAX nonce/args)
2026-05-26.calendar.html  raw rendered week grid (the actual schedule HTML) — the
                          re-parse source of record; never deleted
2026-05-26.json           parsed snapshot: parser_version + parse_ok/parse_error +
                          week_grid[dow] -> [ {start,end,title,status} ]
```

The raw `.calendar.html` is the durable source of truth — `--reparse-all` rebuilds
the `.json` files from it at any time. A snapshot records which `parser_version`
produced it; if the format changes and no parser matches, the snapshot records
`parse_ok: false` with an error but the **raw HTML is still saved** for later
re-parsing once a parser is added.

---

## 10. Platform notes

Runs on **Windows, Linux, and macOS** (incl. Apple Silicon). The code is
cross-platform (pathlib, UTF-8 everywhere, guarded signals); the only OS-specific
piece is ffmpeg/ffprobe discovery, handled below.

**ffmpeg / ffprobe discovery.** `quality.py` resolves the tools in this order:
the `FFMPEG` / `FFPROBE` env vars (full path) → `PATH` → well-known dirs (winget/
chocolatey/scoop on Windows; `/opt/homebrew/bin`, `/usr/local/bin`, `/usr/bin`,
`/snap/bin` on Unix).

> **Minimal-PATH gotcha (cron / launchd / systemd).** Background schedulers run
> with a stripped-down `PATH` that often omits `/opt/homebrew/bin` or
> `/usr/local/bin`, so a tool that works in your shell can be "not found" under a
> job. The well-known-dirs fallback covers the common cases; if your ffmpeg is
> elsewhere, set `FFMPEG`/`FFPROBE` (or `PATH`) explicitly in the job — every
> template below does this.

**Apple Silicon / no GPU.** Everything runs CPU/int8, exactly like the Intel dev
box — faster-whisper/ctranslate2/onnxruntime ship arm64 wheels. There's no CUDA
path on macOS; don't set `device="cuda"`.

**HuggingFace symlink warning** on first model download is **Windows-only**;
Linux/macOS support symlinks and won't show it.

**Graceful shutdown.** `archive.py` traps SIGINT and SIGTERM and asks ffmpeg
to stop cleanly so the in-progress hour's MP3 is closed with a proper
trailer. On POSIX this is plain SIGTERM; on Windows the recorder launches
ffmpeg with `CREATE_NEW_PROCESS_GROUP` and sends `CTRL_BREAK_EVENT` for the
same effect — so Ctrl+C on Windows now stops cleanly without truncating the
in-progress segment. If ffmpeg ignores the graceful stop, the recorder falls
back to `terminate()` after 10 s and `kill()` after another 5 s.

**venv interpreter path:** `.venv\Scripts\python.exe` (Windows) vs
`.venv/bin/python` (Linux/macOS) — used in the templates below.

### Running as services

The recorder and the `--watch` transcriber are long-running; the schedule fetch
is daily. Replace `/home/you/radio-archiver` (or `/Users/you/...`), the `User`,
and the ffmpeg path to match your box.

**Linux — systemd** (drop in `/etc/systemd/system/`, or `~/.config/systemd/user/`
and use `systemctl --user`):

```ini
# radio-archive.service — records the stream
[Unit]
Description=Radio archiver
After=network-online.target
Wants=network-online.target
[Service]
Type=simple
User=you
WorkingDirectory=/home/you/radio-archiver
Environment=FFMPEG=/usr/bin/ffmpeg FFPROBE=/usr/bin/ffprobe
ExecStart=/home/you/radio-archiver/.venv/bin/python archive.py
Restart=always
RestartSec=5
[Install]
WantedBy=multi-user.target
```

```ini
# radio-transcribe.service — transcribes each finished hour
[Unit]
Description=Radio transcriber (--watch)
After=network-online.target
[Service]
Type=simple
User=you
WorkingDirectory=/home/you/radio-archiver
Environment=FFMPEG=/usr/bin/ffmpeg FFPROBE=/usr/bin/ffprobe
ExecStart=/home/you/radio-archiver/.venv/bin/python transcribe.py --watch
Restart=always
RestartSec=10
[Install]
WantedBy=multi-user.target
```

```ini
# radio-schedule.service + radio-schedule.timer — daily schedule snapshot
# --- radio-schedule.service ---
[Unit]
Description=Radio schedule snapshot
[Service]
Type=oneshot
User=you
WorkingDirectory=/home/you/radio-archiver
ExecStart=/home/you/radio-archiver/.venv/bin/python schedule_archive.py
# --- radio-schedule.timer ---
[Unit]
Description=Daily radio schedule snapshot
[Timer]
OnCalendar=*-*-* 04:00:00
Persistent=true
[Install]
WantedBy=timers.target
```

```bash
sudo systemctl enable --now radio-archive.service radio-transcribe.service
sudo systemctl enable --now radio-schedule.timer
journalctl -u radio-transcribe -f         # follow logs
```

**macOS — launchd** (LaunchAgents in `~/Library/LaunchAgents/`). One plist per
long-running process — use `KeepAlive` for the recorder and watcher; for the
daily fetch swap `KeepAlive`/`RunAtLoad` for `StartCalendarInterval`:

```xml
<!-- ~/Library/LaunchAgents/com.radio.transcribe.plist -->
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0"><dict>
  <key>Label</key><string>com.radio.transcribe</string>
  <key>ProgramArguments</key>
  <array>
    <string>/Users/you/radio-archiver/.venv/bin/python</string>
    <string>transcribe.py</string><string>--watch</string>
  </array>
  <key>WorkingDirectory</key><string>/Users/you/radio-archiver</string>
  <key>EnvironmentVariables</key>
  <dict>
    <key>FFMPEG</key><string>/opt/homebrew/bin/ffmpeg</string>
    <key>FFPROBE</key><string>/opt/homebrew/bin/ffprobe</string>
  </dict>
  <key>KeepAlive</key><true/>
  <key>RunAtLoad</key><true/>
  <key>StandardOutPath</key><string>/Users/you/radio-archiver/transcribe.out.log</string>
  <key>StandardErrorPath</key><string>/Users/you/radio-archiver/transcribe.err.log</string>
</dict></plist>
```

```xml
<!-- daily schedule fetch: same shape, but instead of KeepAlive/RunAtLoad use -->
  <key>ProgramArguments</key>
  <array>
    <string>/Users/you/radio-archiver/.venv/bin/python</string>
    <string>schedule_archive.py</string>
  </array>
  <key>StartCalendarInterval</key>
  <dict><key>Hour</key><integer>4</integer><key>Minute</key><integer>0</integer></dict>
```

```bash
launchctl load ~/Library/LaunchAgents/com.radio.transcribe.plist   # start
launchctl unload ~/Library/LaunchAgents/com.radio.transcribe.plist # stop
```

Make a recorder plist the same way (`...archive.plist`, args `archive.py`).

**Windows** — run the two long-running processes from the venv (see [§3](#3-usage));
for the daily schedule fetch use the Task Scheduler command in [§9](#running-it-daily).

---

## 11. Merging two redundant recorders

For continuity across reboots / power outages / network blips, you can run two
machines recording the same stream in parallel and reconcile the results into
one canonical tree afterwards with `merge_archives.py`. Both recorders use
`-segment_atclocktime`, so `2026-05-27_16-00.mp3` on either side covers the
same wall-clock window — the filename is the merge key. The script **never
mutates the source directories** and writes everything into a fresh output
directory.

> **Run from the venv.** `merge_archives.py` imports `transcribe`, which needs
> faster-whisper, so launch it from the activated venv (or with the venv's
> python explicitly). Phase 2 only re-runs the transcriber when it has to
> regenerate the JSON for a freshly-spliced file.

The merge is two phases:

1. **Winner-takes-all per hour.** For each hour key, the per-hour JSON
   sidecar's `quality` block is the oracle: prefer the side with `quality.ok`,
   then the higher `size.ratio`, then less total silence. The chosen side's
   `.mp3` / `.json` / `.txt` trio is copied to the output.
2. **Cross-fill splice (default ON; `--no-splice` to disable).** When both
   candidates are bad, the script picks the good portion from each side using
   their `silence_periods`, concatenates them losslessly with ffmpeg
   `-c copy`, and re-runs the transcriber on the merged audio so the new JSON
   describes what's actually there.

```bash
python merge_archives.py \
  --archive-a /path/to/A/archive --archive-b /path/to/B/archive \
  --archive-out /path/to/merged \
  [--schedule-a /path/to/A/schedule --schedule-b /path/to/B/schedule \
   --schedule-out /path/to/merged_schedule] \
  [--no-splice] [--dry-run] [--force-out] [--report merge_log.json]
```

A merge report (`merge_log.json` in the output dir by default) records every
per-hour decision — the chosen side, the reason, and a compact view of each
candidate — so the merge is auditable after the fact.

Schedule dirs (`--schedule-*`) merge in parallel with simpler rules: prefer
`parse_ok`, then higher `event_count` per date.

---

## 12. Off-air detection and purging

Some stations don't broadcast 24/7 — overnight blocks may be carrier hiss,
station-tone loops, or true silence rather than programming. The recorder
captures these hours indistinguishably at full bitrate (~86 MB/hour at 192
kbps), so an unattended deployment can accumulate a lot of disk usage on
non-content.

`quality.py` measures peak / mean amplitude in dBFS via `volumedetect` in
the same ffmpeg pass as `silencedetect`. Combined with the existing Silero
VAD speech count, an hour is positively identified as **off-air** when ALL
of these hold:

- `max_volume_db < OFF_AIR_PEAK_DB` (default `-30 dBFS` — well below any
  music programming, which routinely peaks at `0 dBFS`)
- `speech_seconds == 0` (Silero VAD detected no speech)
- `audio_seconds >= OFF_AIR_MIN_AUDIO_S` (default `3300 s` — a near-full
  hour; partial `.partN` files are explicitly excluded by their `partial`
  flag too)
- `tool_error` is `None` (the measurement itself is trustworthy)

The criteria are deliberately strict — instrumental music with peaks at
0 dB stays even when there's no speech, and a single VAD frame is enough
to keep a file regardless of volume. The intended false-positive rate is
zero. `quality.is_off_air()` is the single source of truth; the
boolean lands in each sidecar as `quality.is_off_air`.

The `.txt` header shows `quality: OFF-AIR (peak X dB)` for these files,
and the body line below the header says `(no speech detected — off-air
signal, no broadcast content)` instead of the music-only-hour wording.

### Purging the MP3s

```bash
python purge_silent.py            # dry-run, walks the whole archive
python purge_silent.py --apply    # actually deletes
python purge_silent.py FILE.json [...]   # check specific sidecars
```

`purge_silent.py` deletes only the `.mp3` for sidecars where
`quality.is_off_air == true`. The `.json` and `.txt` sidecars are
**kept** as tombstones — they remain searchable, document what was
recorded (peak / mean volume, silence breakdown, schedule_hint), and
prevent the transcriber from re-processing the missing `.mp3` on the
next `--watch` poll. After deletion:

- The sidecar gets a new `quality.mp3_deleted_utc` field (ISO date).
- A `# off-air detected; mp3 deleted YYYY-MM-DD UTC` line is appended to
  the `.txt`.

The purge is safe to run repeatedly — already-tombstoned hours are
skipped (no `.mp3` to delete). Run it from cron / a systemd timer after
your B2 rotation if you want off-air hours never to even reach cloud
storage.

### Re-tagging existing sidecars

Older sidecars produced before this feature don't have `max_volume_db`
or `is_off_air`. Backfill them with:
```bash
python transcribe.py --retag
```
`--retag` recomputes the quality block for any sidecar missing the new
volume fields, then re-evaluates `is_off_air`. Idempotent — running it
again on already-up-to-date sidecars is a no-op except for the schedule
hint refresh.

---

## License

MIT — see [`LICENSE`](LICENSE).
