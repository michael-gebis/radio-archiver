#!/usr/bin/env python3
"""
Radio Archive Transcriber

Pipeline for turning recorded hourly .mp3 segments into searchable text:

  1. Silero VAD (bundled with faster-whisper) isolates *speech* regions and
     skips music / silence. This is what stops Whisper from hallucinating
     lyrics or junk over songs — only talk is sent to the transcriber.
  2. faster-whisper transcribes the speech regions on CPU (int8 quantized).
  3. Two sidecar files are written next to each .mp3:
         <name>.txt   — human-readable, one timestamped line per segment
         <name>.json  — structured segments + metadata for programmatic use

The model is loaded once and reused, so transcribing many files (or running
inside the archiver's watcher) only pays the load cost a single time.

Usage:
    python transcribe.py FILE.mp3 [FILE2.mp3 ...]   # transcribe specific files
    python transcribe.py --all                      # all archive/*.mp3 missing a sidecar
    python transcribe.py --all --force              # re-transcribe everything
    python transcribe.py --watch                    # run continuously alongside archive.py,
                                                    #   transcribing each hour as it completes
    python transcribe.py --retag                    # only refresh schedule hints

Environment overrides:
    WHISPER_MODEL     model size (default: small.en). e.g. base.en, medium.en
    WHISPER_COMPUTE   compute type (default: int8). e.g. int8_float32, float32
"""

import json
import logging
import os
import platform
import re
import signal
import time
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from types import FrameType
from typing import TYPE_CHECKING

import quality
from config import CONFIG

if TYPE_CHECKING:                       # import only for type-checkers, not at runtime
    from faster_whisper import WhisperModel

# Only consulted by --all / --watch / --retag. When you pass explicit file
# paths to transcribe_file(), sidecars are written next to the input regardless
# of CWD, so an mp3 outside this directory transcribes correctly.
ARCHIVE_DIR = Path(CONFIG["paths"]["archive_dir"])

# Bumped when the JSON sidecar layout changes, so future tooling can adapt.
#   1: added language / realtime_factor / provenance
#   2: added schedule_hint (best-effort, non-authoritative show tagging)
#   3: added quality (size / silence / decode-error checks, folded in from archive.py)
SCHEMA_VERSION = 3

# Filenames look like 2026-05-24_12-00.mp3 (clock-aligned hour segments).
FILENAME_DT_FORMAT = "%Y-%m-%d_%H-%M"

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
# small.en is English-only: faster and more accurate on English speech than the
# multilingual model, and it avoids language-detection wobble on music.
MODEL_SIZE   = os.environ.get("WHISPER_MODEL", "small.en")
DEVICE       = "cpu"                                   # no NVIDIA GPU on this box
COMPUTE_TYPE = os.environ.get("WHISPER_COMPUTE", "int8")

# VAD tuning. A higher threshold makes the detector more conservative about
# what counts as speech, which matters for a music-heavy station: we would
# rather miss a little quiet talk than feed sung vocals to the transcriber.
VAD_PARAMETERS = {
    "threshold": 0.6,                 # Silero speech probability cutoff (default 0.5)
    "min_speech_duration_ms": 500,    # ignore sub-half-second blips
    "min_silence_duration_ms": 1000,  # require 1s of quiet to close a speech run
    "speech_pad_ms": 200,             # keep a little context around each region
}

# Decoder options (recorded in provenance so a future reader can judge whether
# re-running with different settings would be worthwhile).
BEAM_SIZE = 5
CONDITION_ON_PREVIOUS_TEXT = False

# Watch mode: how often to scan for new segments, and how long a file must sit
# unmodified before the newest one is treated as finished (covers the case where
# the recorder has stopped and there's no newer file to signal completion).
WATCH_INTERVAL_SECONDS = 30
WATCH_STABLE_SECONDS = 120

log = logging.getLogger(__name__)

_model: "WhisperModel | None" = None  # lazily-loaded singleton


def get_model() -> "WhisperModel":
    """Load (once) and return the faster-whisper model."""
    global _model
    if _model is None:
        from faster_whisper import WhisperModel
        log.info("Loading Whisper model '%s' (%s/%s)...",
                 MODEL_SIZE, DEVICE, COMPUTE_TYPE)
        t0 = time.time()
        _model = WhisperModel(MODEL_SIZE, device=DEVICE, compute_type=COMPUTE_TYPE)
        log.info("Model loaded in %.1fs.", time.time() - t0)
    return _model


def _pkg_version(name: str) -> str | None:
    try:
        return version(name)
    except PackageNotFoundError:
        return None


def provenance() -> dict:
    """
    Everything needed to reproduce or evaluate this transcription run: the
    model + decode settings, the library versions, and the machine it ran on.
    A future reader can compare against a newer model / stronger machine /
    different parameters to decide whether re-transcribing is worthwhile.
    """
    return {
        "settings": {
            "model": MODEL_SIZE,
            "device": DEVICE,
            "compute_type": COMPUTE_TYPE,
            "vad_filter": True,
            "vad_parameters": dict(VAD_PARAMETERS),
            "beam_size": BEAM_SIZE,
            "condition_on_previous_text": CONDITION_ON_PREVIOUS_TEXT,
        },
        "versions": {
            "python": platform.python_version(),
            "faster_whisper": _pkg_version("faster-whisper"),
            "ctranslate2": _pkg_version("ctranslate2"),
            "onnxruntime": _pkg_version("onnxruntime"),
            "av": _pkg_version("av"),
        },
        "machine": {
            "platform": platform.platform(),
            "processor": platform.processor() or platform.machine(),
            "cpu_count": os.cpu_count(),
        },
    }


# Matches a `.partN` suffix on the filename stem (N is one or more digits),
# produced by archive.py when it rotates a mid-hour file on restart.
_PART_SUFFIX_RE = re.compile(r"\.part\d+$")


def _is_partial(mp3: Path) -> bool:
    """A `.partN.mp3` file (archive.py's mid-hour rotation product) covers
    only a slice of its parent hour; quality and downstream tooling should
    treat it as a partial."""
    return bool(_PART_SUFFIX_RE.search(mp3.stem))


def _recording_datetime(mp3: Path) -> datetime | None:
    """Parse the clock-aligned start time out of the segment filename. Handles
    canonical names like `2026-06-04_01-00` and the `.partN`-suffixed siblings
    that archive.py produces on a mid-hour restart."""
    stem = _PART_SUFFIX_RE.sub("", mp3.stem)
    try:
        return datetime.strptime(stem, FILENAME_DT_FORMAT)
    except ValueError:
        return None


def _schedule_hint(mp3: Path) -> dict:
    """
    Best-effort 'what show was listed for this hour' tag. Never raises and never
    blocks transcription — any problem is reported inside the returned dict.
    The schedule is not authoritative (see schedule_archive), hence the naming.
    """
    dt = _recording_datetime(mp3)
    if dt is None:
        return {"error": f"could not parse a timestamp from filename {mp3.name!r}"}
    try:
        import schedule_archive
    except ImportError:
        return {"error": "schedule_archive module not importable"}
    try:
        return schedule_archive.schedule_hint_for(dt)
    except Exception as e:  # defensive: tagging must never break transcription
        return {"error": f"schedule lookup failed: {e!r}"}


def sidecar_paths(mp3: Path) -> tuple[Path, Path]:
    """Return the (.txt, .json) sidecar paths for an mp3."""
    return mp3.with_suffix(".txt"), mp3.with_suffix(".json")


def transcribe_file(mp3: Path, force: bool = False) -> dict | None:
    """
    Transcribe a single .mp3 and write .txt / .json sidecars beside it.

    Returns the result dict, or None if skipped (sidecar already present and
    not forced). Speech-free hours still write sidecars so they are not
    re-processed on every pass.
    """
    txt_path, json_path = sidecar_paths(mp3)
    if json_path.exists() and not force:
        log.info("Skip (already transcribed): %s", mp3.name)
        return None

    model = get_model()
    log.info("Transcribing %s ...", mp3.name)
    t0 = time.time()

    # vad_filter=True runs Silero VAD first; faster-whisper remaps the segment
    # timestamps back onto the original audio timeline for us.
    # condition_on_previous_text=False reduces runaway repetition loops.
    segments_iter, info = model.transcribe(
        str(mp3),
        vad_filter=True,
        vad_parameters=VAD_PARAMETERS,
        condition_on_previous_text=CONDITION_ON_PREVIOUS_TEXT,
        beam_size=BEAM_SIZE,
    )

    segments = []
    speech_seconds = 0.0
    for seg in segments_iter:
        text = seg.text.strip()
        if not text:
            continue
        segments.append({
            "start": round(seg.start, 2),
            "end": round(seg.end, 2),
            "text": text,
        })
        speech_seconds += seg.end - seg.start

    elapsed = time.time() - t0
    audio_seconds = float(info.duration) if info and info.duration else 0.0
    result = {
        "schema_version": SCHEMA_VERSION,
        "file": mp3.name,
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "audio_seconds": round(audio_seconds, 1),
        "speech_seconds": round(speech_seconds, 1),
        "talk_ratio": round(speech_seconds / audio_seconds, 3) if audio_seconds else 0.0,
        "language": getattr(info, "language", None),
        "language_probability": round(info.language_probability, 3)
            if getattr(info, "language_probability", None) is not None else None,
        "transcribe_seconds": round(elapsed, 1),
        "realtime_factor": round(audio_seconds / elapsed, 1) if elapsed else None,
        "segment_count": len(segments),
        "quality": quality.analyze(mp3, bitrate_kbps=CONFIG["stream"]["bitrate_kbps"],
                                   partial=_is_partial(mp3)),
        "schedule_hint": _schedule_hint(mp3),
        "provenance": provenance(),
        "segments": segments,
    }
    result["quality"]["is_off_air"] = quality.is_off_air(
        result["quality"], speech_seconds, audio_seconds)

    _write_sidecars(mp3, result)
    log.info(
        "Done %s — %d speech segments, %.0f%% talk, %.1fs to process %.0f min of audio.",
        mp3.name, len(segments), result["talk_ratio"] * 100, elapsed,
        audio_seconds / 60 if audio_seconds else 0,
    )
    return result


def _format_hint(hint: dict | None) -> str:
    """One-line summary of the schedule hint for the .txt header."""
    if not hint:
        return "none"
    if hint.get("error"):
        return f"unavailable ({hint['error']})"
    shows = hint.get("listed_shows") or []
    if not shows:
        return "no show listed for this hour"
    return "; ".join(
        f"{s['title']} ({s['start']}-{s.get('end') or '?'})" for s in shows
    )


def _write_sidecars(mp3: Path, result: dict) -> None:
    txt_path, json_path = sidecar_paths(mp3)

    json_path.write_text(json.dumps(result, indent=2, ensure_ascii=False),
                         encoding="utf-8")

    settings = result["provenance"]["settings"]
    fw_version = result["provenance"]["versions"].get("faster_whisper")
    lines = [
        f"# Transcript: {result['file']}",
        f"# model={settings['model']}  faster-whisper={fw_version}  "
        f"talk={result['talk_ratio'] * 100:.0f}%  segments={result['segment_count']}  "
        f"generated={result['generated_utc']}",
        f"# quality: {quality.summarize(result.get('quality'))}",
        f"# schedule hint (per published schedule, may differ): {_format_hint(result.get('schedule_hint'))}",
        "# Full quality/settings/versions/machine/schedule recorded in the .json sidecar.",
        "",
    ]
    if result["segments"]:
        for seg in result["segments"]:
            lines.append(f"[{quality.fmt_hms(seg['start'])} -> {quality.fmt_hms(seg['end'])}]  {seg['text']}")
    elif (result.get("quality") or {}).get("is_off_air"):
        lines.append("(no speech detected — off-air signal, no broadcast content)")
    else:
        lines.append("(no speech detected — likely a music-only hour)")
    lines.append("")
    txt_path.write_text("\n".join(lines), encoding="utf-8")


def transcribe_all(force: bool = False) -> None:
    """
    Transcribe every completed segment in the archive dir, skipping the
    in-progress hour so it's never captured as a partial. Idempotent: files that
    already have a transcript are skipped (unless force). Safe to run anytime,
    whether or not the recorder is running.
    """
    files = _completed_segments()
    if not files:
        log.info("No completed .mp3 files found in %s", ARCHIVE_DIR)
        return
    log.info("Found %d completed file(s).", len(files))
    for f in files:
        try:
            transcribe_file(f, force=force)
        except Exception as e:
            log.error("Error transcribing %s: %s", f.name, e)


def retag_all() -> None:
    """
    Refresh schedule_hint on every existing transcript JSON without
    re-transcribing, and backfill any v3 fields the sidecar is missing
    (`quality`, `schedule_hint`) so the `schema_version` stamp honestly
    reflects the file's shape. Rewrites both sidecars.

    If the matching .mp3 is missing we leave `schema_version` alone — never
    upgrade the stamp without being able to fill the fields.
    """
    files = sorted(ARCHIVE_DIR.rglob("*.json"))
    if not files:
        log.info("No transcript JSONs found in %s", ARCHIVE_DIR)
        return
    log.info("Re-tagging %d transcript(s) with current schedule snapshots.", len(files))
    updated = 0
    bitrate = CONFIG["stream"]["bitrate_kbps"]
    for jp in files:
        try:
            result = json.loads(jp.read_text(encoding="utf-8"))
            mp3 = jp.with_suffix(".mp3")
            result["schedule_hint"] = _schedule_hint(mp3)
            if not mp3.exists():
                log.warning("Re-tag: %s has no matching .mp3; leaving schema_version untouched.",
                            jp.name)
                _write_sidecars(mp3, result)
                updated += 1
                continue
            q = result.get("quality") or {}
            if not q or "max_volume_db" not in q:
                # Backfill quality (or re-run it to populate the new
                # max_volume_db / mean_volume_db fields needed by is_off_air).
                log.info("Re-tag: (re)computing quality for %s", jp.name)
                result["quality"] = quality.analyze(mp3, bitrate_kbps=bitrate,
                                                    partial=_is_partial(mp3))
            result["quality"]["is_off_air"] = quality.is_off_air(
                result["quality"],
                result.get("speech_seconds") or 0.0,
                result.get("audio_seconds") or 0.0,
            )
            result["schema_version"] = SCHEMA_VERSION
            _write_sidecars(mp3, result)
            updated += 1
        except Exception as e:
            log.error("Error re-tagging %s: %s", jp.name, e)
    log.info("Re-tagged %d transcript(s).", updated)


# ---------------------------------------------------------------------------
# Watch mode — decoupled from the recorder
# ---------------------------------------------------------------------------
# Run this alongside `archive.py` (which only records). It transcribes each hour
# as it completes, so transcription can be started/stopped/upgraded without ever
# touching the live recorder.

_watch_running = True


def _completed_segments() -> list[Path]:
    """
    All finished segments, excluding any that look in-progress. A file is
    treated as in-progress when BOTH its mtime is recent (< WATCH_STABLE_SECONDS)
    AND its size is below 95% of an expected full-hour file. The combined check
    is applied to every candidate, not just "the newest by path sort": with
    `.partN.mp3` rotation files in the mix the path-sort heuristic can put
    a stale-but-complete part *after* the actually-live canonical file, and
    we don't want the live one to slip through as "completed". The size-floor
    means a fully-recorded hour with recent mtime (just closed) is still
    correctly included.
    """
    mp3s = sorted(p for p in ARCHIVE_DIR.rglob("*.mp3") if p.stat().st_size > 0)
    if not mp3s:
        return []
    now = time.time()
    # Expected full-hour size from the configured stream bitrate (kbps -> bytes
    # for an hour). 95% covers normal jitter between expected and actual.
    expected_full = int(quality.EXPECTED_SEGMENT_SECONDS
                        * CONFIG["stream"]["bitrate_kbps"] * 1000 / 8 * 0.95)
    out = []
    for f in mp3s:
        stat = f.stat()
        recent = (now - stat.st_mtime) < WATCH_STABLE_SECONDS
        small  = stat.st_size < expected_full
        if recent and small:
            continue  # still being written
        out.append(f)
    return out


def _completed_untranscribed() -> list[Path]:
    """Completed segments that don't yet have a transcript (used by --watch)."""
    return [f for f in _completed_segments() if not f.with_suffix(".json").exists()]


def _handle_watch_signal(sig: int, frame: FrameType | None) -> None:
    global _watch_running
    log.info("Stop signal received — exiting after the current file.")
    _watch_running = False


def watch(interval: int = WATCH_INTERVAL_SECONDS) -> None:
    """Continuously transcribe completed segments as they appear. Ctrl+C to stop."""
    signal.signal(signal.SIGINT, _handle_watch_signal)
    if hasattr(signal, "SIGTERM"):
        signal.signal(signal.SIGTERM, _handle_watch_signal)

    log.info("Watching %s for completed segments every %ds. Ctrl+C to stop.",
             ARCHIVE_DIR.resolve(), interval)
    while _watch_running:
        try:
            for f in _completed_untranscribed():
                if not _watch_running:
                    break
                transcribe_file(f)
        except Exception as e:
            log.error("Watch scan error (continuing): %s", e)
        # Interruptible sleep so Ctrl+C is responsive between scans.
        for _ in range(interval):
            if not _watch_running:
                break
            time.sleep(1)
    log.info("Watch stopped.")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Transcribe radio archive segments.")
    parser.add_argument("files", nargs="*", help="Specific .mp3 files to transcribe.")
    parser.add_argument("--all", action="store_true",
                        help="Transcribe every .mp3 in the archive dir.")
    parser.add_argument("--force", action="store_true",
                        help="Re-transcribe even if a sidecar already exists.")
    parser.add_argument("--retag", action="store_true",
                        help="Only refresh schedule hints on existing transcripts "
                             "(no re-transcription).")
    parser.add_argument("--watch", action="store_true",
                        help="Run continuously, transcribing each hour as it completes. "
                             "Decoupled from the recorder — run alongside archive.py.")
    parser.add_argument("--watch-interval", type=int, default=WATCH_INTERVAL_SECONDS,
                        help=f"Seconds between scans in --watch mode "
                             f"(default {WATCH_INTERVAL_SECONDS}).")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    if args.watch:
        watch(interval=args.watch_interval)
    elif args.retag:
        retag_all()
    elif args.all:
        transcribe_all(force=args.force)
    elif args.files:
        for arg in args.files:
            transcribe_file(Path(arg), force=args.force)
    else:
        parser.error("Provide files, or use --all, --retag, or --watch.")
