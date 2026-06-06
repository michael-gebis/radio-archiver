#!/usr/bin/env python3
"""
Audio quality checks for recorded segments.

Pure functions: `analyze()` returns a structured result dict (it logs nothing and
never raises on a missing tool), so callers fold the result into their own
output. The transcriber stores it in each hour's JSON sidecar, which makes the
checks idempotent — they're computed once per file, not re-run on every restart.

Shared by:
  - transcribe.py  → records the result in the transcript JSON
  - archive.py     → reuses the located ffmpeg binary for recording

Checks:
  - size     : actual vs expected file size (catches stream outages)
  - silence  : ffmpeg silencedetect (flags dead air >= SILENCE_MIN_SECS)
  - volume   : ffmpeg volumedetect — peak / mean amplitude in dBFS. Used by
               the off-air detector: an entire hour with peak << 0 dB and no
               VAD speech is broadcast-side carrier noise, not content.
  - errors   : full ffmpeg decode pass via null muxer (catches mid-stream
               frame errors, not just unopenable containers)

`is_off_air()` is the helper transcribers call after both `analyze()` and the
VAD/Whisper pass have produced their numbers. It returns True only when ALL of
the following hold for the file:
  - `max_volume_db < OFF_AIR_PEAK_DB` (peak amplitude inaudible)
  - `speech_seconds == 0` (Silero VAD detected no speech)
  - `audio_seconds >= OFF_AIR_MIN_AUDIO_S` (full-hour file, not a partial)
  - `tool_error is None` (we trust the measurements)
The combination is intentionally narrow — instrumental music hits 0 dB peaks
even when the mean is low, so it's NOT marked off-air. A single VAD frame is
enough to keep a file.
"""

import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Locate ffmpeg / ffprobe across platforms.
#
# Order: explicit env override (FFMPEG / FFPROBE) -> PATH (shutil.which) ->
# well-known install dirs per OS. The last step matters under cron / launchd /
# systemd, where the inherited PATH is often minimal and omits the dirs where
# Homebrew/winget put the binaries.
# ---------------------------------------------------------------------------
if sys.platform == "win32":
    _WINGET_BIN = Path.home() / "AppData/Local/Microsoft/WinGet/Packages"
    _EXTRA_PATHS = [
        *sorted(_WINGET_BIN.glob("Gyan.FFmpeg*/ffmpeg-*/bin"), reverse=True),
        Path("C:/ProgramData/chocolatey/bin"),
        Path.home() / "scoop/apps/ffmpeg/current/bin",
    ]
else:
    _EXTRA_PATHS = [
        Path("/opt/homebrew/bin"),   # Homebrew on Apple Silicon
        Path("/usr/local/bin"),      # Homebrew on Intel macOS / common Linux
        Path("/usr/bin"),            # apt / dnf
        Path("/snap/bin"),           # snap
    ]


def _find_tool(name: str) -> str:
    """
    Return the full path to `name` (e.g. "ffmpeg"). Checks, in order: an explicit
    env override (FFMPEG / FFPROBE), the PATH, then well-known per-OS install
    dirs. Falls back to the bare name (subprocess will raise FileNotFoundError).
    """
    override = os.environ.get(name.upper())
    if override and Path(override).is_file():
        return override

    found = shutil.which(name)
    if found:
        return found

    exe = name + ".exe" if sys.platform == "win32" else name
    for directory in _EXTRA_PATHS:
        candidate = Path(directory) / exe
        if candidate.is_file():
            # Add parent to PATH so the sibling tool is findable too.
            os.environ["PATH"] = str(candidate.parent) + os.pathsep + os.environ["PATH"]
            return str(candidate)
    return name  # fall back to bare name; subprocess will raise FileNotFoundError


FFMPEG  = _find_tool("ffmpeg")
FFPROBE = _find_tool("ffprobe")

# ---------------------------------------------------------------------------
# Thresholds
# ---------------------------------------------------------------------------
DEFAULT_BITRATE_KBPS     = 192      # fallback if the caller doesn't pass one
EXPECTED_SEGMENT_SECONDS = 3600     # a full hour
SIZE_RATIO_WARN          = 0.80     # flag if a segment is < this fraction of expected
SILENCE_THRESHOLD        = "-40dB"  # audio level considered silence
SILENCE_MIN_SECS         = 10       # minimum silence duration to flag (seconds)

# Off-air detection (see is_off_air()). -30 dBFS as the peak threshold sits
# well below typical music programming (which routinely peaks at 0 dBFS) but
# comfortably above broadcast-side carrier hiss / encoder noise (which on
# UCLA Radio measured at ~-36 dBFS peak across full-hour off-air files).
#
# The audio-seconds floor is set against the *measured* normal full-hour
# duration on this stream (~3430-3462 s — not 3600). 3300 is comfortably
# below that floor and comfortably above the largest .partN partial seen
# in production (~2880 s).
OFF_AIR_PEAK_DB     = -30.0
OFF_AIR_MIN_AUDIO_S = 3300


def fmt_hms(secs: float) -> str:
    """Format a number of seconds as `H:MM:SS` (no leading zero on hours)."""
    s = int(secs)
    return f"{s // 3600}:{(s % 3600) // 60:02d}:{s % 60:02d}"


def _check_size(path: Path, expected_seconds: int, bitrate_kbps: int) -> dict:
    expected_bytes = expected_seconds * bitrate_kbps * 1000 // 8
    actual_bytes = path.stat().st_size
    ratio = (actual_bytes / expected_bytes) if expected_bytes else 1.0
    return {
        "actual_mb": round(actual_bytes / 1e6, 1),
        "expected_mb": round(expected_bytes / 1e6, 1),
        "ratio": round(ratio, 3),
        "ok": ratio >= SIZE_RATIO_WARN,
    }


def _silence_and_volume(path: Path) -> tuple[list[dict], float | None, float | None, str | None]:
    """
    Combined ffmpeg pass: silencedetect + volumedetect, parsed out of one
    stderr stream. Returns (silence_periods, max_volume_db, mean_volume_db,
    tool_error).

    Each silence period is {start, end, duration}; end/duration are None if
    the file ended while still silent. Volume values come from volumedetect's
    summary lines (printed once at end-of-input). `tool_error` is set if
    ffmpeg exited non-zero and no periods were parsed — i.e. a genuine
    failure rather than a clean noisy run.
    """
    cmd = [
        FFMPEG, "-hide_banner", "-nostdin", "-i", str(path),
        "-af", f"silencedetect=noise={SILENCE_THRESHOLD}:duration={SILENCE_MIN_SECS},volumedetect",
        "-f", "null", "-",
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    out = proc.stderr
    starts = [float(m) for m in re.findall(r"silence_start: ([0-9.]+)", out)]
    ends   = [float(m) for m in re.findall(r"silence_end: ([0-9.]+)", out)]
    periods = []
    for i, start in enumerate(starts):
        end = ends[i] if i < len(ends) else None
        periods.append({
            "start": round(start, 1),
            "end": round(end, 1) if end is not None else None,
            "duration": round(end - start, 1) if end is not None else None,
        })
    mean_m = re.search(r"mean_volume:\s*(-?[\d.]+)\s*dB", out)
    max_m  = re.search(r"max_volume:\s*(-?[\d.]+)\s*dB", out)
    mean_db = round(float(mean_m.group(1)), 1) if mean_m else None
    max_db  = round(float(max_m.group(1)), 1)  if max_m  else None
    err = None
    if proc.returncode != 0 and not periods and mean_db is None and max_db is None:
        tail = (out or "").strip().splitlines()[-3:]
        err = f"ffmpeg silence+volume exit {proc.returncode}: {' | '.join(tail)[:300]}"
    return periods, max_db, mean_db, err


def _decode_scan(path: Path) -> tuple[str | None, int]:
    """Full ffmpeg decode pass via the null muxer. Returns (error_text, rc).
    `error_text` is the joined stderr from `-v error` (each frame-level error
    appears as a line), or None if clean. Catches mid-stream frame errors that
    a metadata-only probe would miss."""
    cmd = [
        FFMPEG, "-v", "error", "-nostdin",
        "-i", str(path),
        "-f", "null", "-",
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    err = proc.stderr.strip() or None
    return err, proc.returncode


def analyze(path: str | Path, expected_seconds: int = EXPECTED_SEGMENT_SECONDS,
            bitrate_kbps: int = DEFAULT_BITRATE_KBPS, partial: bool = False) -> dict:
    """
    Run all quality checks on a completed segment. `bitrate_kbps` is the stream's
    expected bitrate (from config) used for the size estimate. Returns a
    structured dict and never raises: a missing ffmpeg/ffprobe is reported in
    `tool_error` with the ffmpeg-dependent checks left empty (the size check
    still works regardless).

    When `partial=True` (used for archive.py's `.partN.mp3` rotation products,
    which cover only a slice of the hour), the size check still reports its
    measured / expected values but `size.ok` is set to None so a small partial
    doesn't fail the overall `ok` — for a partial we can't fairly estimate
    expected bytes.
    """
    path = Path(path)
    result = {
        "expected_seconds": expected_seconds,
        "partial": partial,
        "size": _check_size(path, expected_seconds, bitrate_kbps),
        "silence_periods": [],
        "max_volume_db": None,
        "mean_volume_db": None,
        "decode_errors": None,
        "decode_exit_code": None,
        "tool_error": None,
        "thresholds": {
            "stream_bitrate_kbps": bitrate_kbps,
            "size_ratio_warn": SIZE_RATIO_WARN,
            "silence_threshold": SILENCE_THRESHOLD,
            "silence_min_secs": SILENCE_MIN_SECS,
            "off_air_peak_db": OFF_AIR_PEAK_DB,
            "off_air_min_audio_s": OFF_AIR_MIN_AUDIO_S,
        },
    }
    if partial:
        # Inputs for size estimation aren't meaningful for an arbitrary
        # subset of an hour. Keep the measured values, drop the pass/fail.
        result["size"]["ok"] = None
    try:
        periods, max_db, mean_db, sil_err = _silence_and_volume(path)
        result["silence_periods"] = periods
        result["max_volume_db"] = max_db
        result["mean_volume_db"] = mean_db
        decode_err, decode_rc = _decode_scan(path)
        result["decode_errors"] = decode_err
        result["decode_exit_code"] = decode_rc
        if sil_err:
            result["tool_error"] = sil_err
    except FileNotFoundError as e:
        result["tool_error"] = f"ffmpeg/ffprobe not found: {e}"
    except Exception as e:
        result["tool_error"] = f"quality check failed: {e!r}"
    size_ok = result["size"]["ok"]
    result["ok"] = (
        (size_ok is True or size_ok is None)
        and not result["silence_periods"]
        and not result["decode_errors"]
        and not result["tool_error"]
    )
    return result


def is_off_air(quality: dict, speech_seconds: float, audio_seconds: float) -> bool:
    """
    Decide whether a file is off-air: near-full-hour duration, not flagged
    as a `.partN` partial, no VAD speech, peak amplitude inaudible, and the
    quality measurement itself trusted.

    Each criterion is intentionally strict (any single failure → False), so
    instrumental music (peak 0 dB), a single VAD frame, partial files, and
    files where ffmpeg failed all stay safely non-off-air.
    """
    if not quality:
        return False
    if quality.get("tool_error"):
        return False
    if quality.get("partial"):
        return False
    max_db = quality.get("max_volume_db")
    if max_db is None:
        return False
    return (
        max_db < OFF_AIR_PEAK_DB
        and speech_seconds == 0
        and audio_seconds >= OFF_AIR_MIN_AUDIO_S
    )


def summarize(quality: dict | None) -> str:
    """One-line human-readable summary for logs / .txt headers."""
    if not quality:
        return "n/a"
    if quality.get("tool_error"):
        return f"unavailable ({quality['tool_error']})"
    if quality.get("is_off_air"):
        peak = quality.get("max_volume_db")
        peak_str = f"{peak} dB" if peak is not None else "?"
        return f"OFF-AIR (peak {peak_str})"
    parts = []
    if quality.get("partial"):
        size = quality.get("size", {})
        parts.append(f"PARTIAL HOUR ({size.get('actual_mb')}MB)")
    else:
        size = quality.get("size", {})
        if size.get("ok") is False:
            parts.append(f"SMALL SEGMENT {size.get('actual_mb')}MB "
                         f"({size.get('ratio', 0) * 100:.0f}%)")
    silence = quality.get("silence_periods") or []
    if silence:
        total = sum(p["duration"] for p in silence if p.get("duration"))
        parts.append(f"{len(silence)} silence period(s) ~{total:.0f}s")
    if quality.get("decode_errors"):
        parts.append("DECODE ERRORS")
    return "OK" if not parts else "; ".join(parts)
