#!/usr/bin/env python3
"""
Radio Live-Stream Archiver

Records the configured stream URL (see config.json) in hourly segments named by
clock time (e.g. archive/2026-05-23_14-00.mp3) and reconnects automatically on
drops.

This process only *records*. Per-hour analysis (transcription + quality checks)
is handled by transcribe.py, which writes one JSON sidecar per hour. Run them as
two processes:

    python archive.py              # record only
    python transcribe.py --watch   # transcribe + quality-check each finished hour

For convenience a single-process mode is available (`--transcribe`), which runs
the same per-hour processing inline via a watcher thread.

Usage:
    python archive.py              # record forever
    python archive.py --test       # record 60 s to verify setup, then exit
    python archive.py --transcribe # also transcribe each completed hour inline
"""

import argparse
import logging
import subprocess
import sys
import threading
import time
from datetime import date, datetime
from pathlib import Path
from types import FrameType

from config import CONFIG
from quality import FFMPEG  # ffmpeg binary, located (cross-platform) in quality.py

# ---------------------------------------------------------------------------
# Configuration (station-specific values come from config.json)
# ---------------------------------------------------------------------------
LABEL           = CONFIG["label"]
STREAM_URL      = CONFIG["stream"]["url"]
ARCHIVE_DIR     = Path(CONFIG["paths"]["archive_dir"])
SEGMENT_SECONDS = 3600         # length of each output file (1 hour)
RECONNECT_DELAY = 5            # seconds to wait after a drop before retrying
LOG_FILE        = "archiver.log"
# ---------------------------------------------------------------------------

running: bool = True


def setup_logging() -> None:
    fmt = logging.Formatter("%(asctime)s  %(levelname)-8s  %(message)s",
                            datefmt="%Y-%m-%d %H:%M:%S")
    root = logging.getLogger()
    root.setLevel(logging.INFO)

    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    root.addHandler(sh)

    fh = logging.FileHandler(LOG_FILE, encoding="utf-8")
    fh.setFormatter(fmt)
    root.addHandler(fh)


log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Signal handling
# ---------------------------------------------------------------------------
import signal

_current_proc: subprocess.Popen | None = None


def _handle_signal(sig: int, frame: FrameType | None) -> None:
    """
    Ask ffmpeg to stop cleanly so the in-progress MP3 segment is closed with
    a proper trailer.

    On POSIX, `terminate()` sends SIGTERM and ffmpeg flushes on its own. On
    Windows, SIGTERM maps to `TerminateProcess` which kills the child instantly
    and truncates the segment — so we instead send CTRL_BREAK_EVENT to the
    process group (ffmpeg was launched with CREATE_NEW_PROCESS_GROUP for this
    reason). If ffmpeg ignores the break event we fall back to terminate so
    shutdown isn't blocked forever.
    """
    global running
    log.info("Shutdown signal received — stopping after current ffmpeg exits.")
    running = False
    proc = _current_proc
    if not (proc and proc.poll() is None):
        return
    if sys.platform == "win32":
        try:
            proc.send_signal(signal.CTRL_BREAK_EVENT)
        except (OSError, ValueError) as e:
            log.warning("CTRL_BREAK_EVENT to ffmpeg failed (%s); falling back to terminate.", e)
            proc.terminate()
    else:
        proc.terminate()


signal.signal(signal.SIGINT, _handle_signal)
if hasattr(signal, "SIGTERM"):
    signal.signal(signal.SIGTERM, _handle_signal)


# ---------------------------------------------------------------------------
# Inline transcription (single-process --transcribe mode only)
# ---------------------------------------------------------------------------

def _transcribe_segment(path: Path) -> None:
    """
    Transcribe a completed segment via transcribe.py (which also runs the quality
    checks and writes them into the JSON sidecar). Failures here must never
    affect recording, so everything is caught.
    """
    try:
        import transcribe
    except ImportError:
        log.warning(
            "Transcription requested but faster-whisper isn't importable. "
            "Run the archiver from the project venv (.venv) where it's installed."
        )
        return
    try:
        transcribe.transcribe_file(path)
    except Exception as e:
        log.error("Error transcribing %s: %s", path.name, e)


def _watcher(stop_event: threading.Event) -> None:
    """
    Inline watcher for --transcribe mode: transcribe each segment once it's
    complete (a newer file exists). Skips files that already have a transcript,
    so a restart never re-does finished work.
    """
    done: set[Path] = set()

    while not stop_event.is_set():
        files = sorted(ARCHIVE_DIR.rglob("*.mp3"))
        complete = files[:-1] if len(files) > 1 else []   # last is still recording
        for f in complete:
            if f not in done:
                done.add(f)
                if not f.with_suffix(".json").exists():
                    _transcribe_segment(f)
        stop_event.wait(30)

    # On shutdown, process any completed files not yet handled.
    files = sorted(ARCHIVE_DIR.rglob("*.mp3"))
    for f in (files[:-1] if len(files) > 1 else []):
        if f not in done and not f.with_suffix(".json").exists():
            _transcribe_segment(f)


# ---------------------------------------------------------------------------
# ffmpeg recording
# ---------------------------------------------------------------------------

def _drain_stderr(proc: subprocess.Popen) -> None:
    for raw in proc.stderr:
        line = raw.strip()
        if line:
            log.warning("[ffmpeg] %s", line)


def _current_hour_path() -> Path:
    """Path the segment muxer will open right now for the current clock hour."""
    now = datetime.now()
    return (ARCHIVE_DIR
            / f"{now.year:04d}" / f"{now.month:02d}"
            / now.strftime("%Y-%m-%d_%H-00.mp3"))


def _rotate_in_progress_hour() -> None:
    """
    If the current hour's canonical file already exists and is non-empty, rename
    it to a `.partN.mp3` sibling so the next ffmpeg invocation can open the
    canonical name fresh. Called immediately before every ffmpeg invocation in
    normal mode, so:

    - On script restart mid-hour, the previous run's partial is preserved.
    - On an ffmpeg crash/reconnect mid-hour, the just-closed file is preserved
      so the new ffmpeg doesn't truncate it.

    `N` is the smallest non-negative integer that produces a free filename, so
    multiple restarts within the same hour stack as `.part0`, `.part1`, ...
    """
    path = _current_hour_path()
    try:
        if not path.exists() or path.stat().st_size == 0:
            return
    except OSError:
        return
    n = 0
    while True:
        candidate = path.with_name(f"{path.stem}.part{n}{path.suffix}")
        if not candidate.exists():
            break
        n += 1
    path.rename(candidate)
    log.info("Preserved in-progress hour as %s (%.1f MB).",
             candidate.name, candidate.stat().st_size / 1e6)


def _ensure_month_dirs(months_ahead: int = 24) -> None:
    """
    Create archive/YYYY/MM/ for the current month and the next `months_ahead`
    months. ffmpeg's segment muxer does not create intermediate directories,
    so we have to prepare them up-front — otherwise the first segment after a
    month rollover would fail to open. 24 months of buffer means the recorder
    can run uninterrupted for two years without manual help; each restart
    (e.g. on a stream-drop reconnect) re-extends the buffer.
    """
    ARCHIVE_DIR.mkdir(exist_ok=True)
    today = date.today()
    year, month = today.year, today.month
    for _ in range(months_ahead + 1):
        (ARCHIVE_DIR / f"{year:04d}" / f"{month:02d}").mkdir(parents=True, exist_ok=True)
        month += 1
        if month > 12:
            month = 1
            year += 1


def _build_cmd(segment_seconds: int, test_out: Path | None = None) -> list[str]:
    """
    Build the ffmpeg command. Two modes:

    - Normal: segment muxer, `-segment_atclocktime`, files land in
      `archive/YYYY/MM/YYYY-MM-DD_HH-00.mp3`. ffmpeg keeps running, segmenting
      forever until told to stop.
    - Test (`test_out` set): a single fixed-duration capture (`-t`) with no
      segment muxer, so ffmpeg exits cleanly after `segment_seconds`. Output
      goes to the literal `test_out` path. The segment-muxer behavior is wrong
      for test mode because (a) ffmpeg never self-exits and (b) within a single
      hour the strftime filename doesn't change, so sub-hour segments overwrite
      each other.
    """
    base = [
        FFMPEG,
        "-hide_banner",
        "-loglevel", "warning",
        "-reconnect",           "1",
        "-reconnect_streamed",  "1",
        "-reconnect_delay_max", "30",
        "-i", STREAM_URL,
        "-c", "copy",
    ]
    if test_out is not None:
        test_out.parent.mkdir(parents=True, exist_ok=True)
        return base + ["-t", str(segment_seconds), str(test_out)]
    _ensure_month_dirs()
    output_pattern = str(ARCHIVE_DIR / "%Y" / "%m" / "%Y-%m-%d_%H-00.mp3")
    return base + [
        "-f",                   "segment",
        "-segment_time",        str(segment_seconds),
        "-segment_atclocktime", "1",
        "-reset_timestamps",    "1",
        "-strftime",            "1",
        output_pattern,
    ]


def _run_ffmpeg(segment_seconds: int, test_out: Path | None = None) -> int:
    global _current_proc
    cmd = _build_cmd(segment_seconds, test_out=test_out)
    log.info("Starting ffmpeg:  %s", " ".join(cmd))
    # On Windows, launching ffmpeg in its own process group lets us send
    # CTRL_BREAK_EVENT for a graceful stop (see _handle_signal). On POSIX this
    # flag does not exist and the default behavior is already correct.
    popen_kwargs: dict = {"stderr": subprocess.PIPE, "text": True}
    if sys.platform == "win32":
        popen_kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
    _current_proc = subprocess.Popen(cmd, **popen_kwargs)
    drain = threading.Thread(target=_drain_stderr, args=(_current_proc,), daemon=True)
    drain.start()
    try:
        _current_proc.wait()
    except KeyboardInterrupt:
        # Handler already signaled the child; just give it a moment to flush.
        try:
            _current_proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            log.warning("ffmpeg did not exit after graceful stop; terminating.")
            _current_proc.terminate()
            try:
                _current_proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                _current_proc.kill()
                _current_proc.wait()
    drain.join(timeout=2)
    code = _current_proc.returncode
    _current_proc = None
    return code


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def run(test_mode: bool = False, transcribe: bool = False) -> None:
    global running
    segment_seconds = 60 if test_mode else SEGMENT_SECONDS

    # In test mode, write to a single timestamped file at the top of
    # ARCHIVE_DIR. The timestamp keeps successive --test runs from
    # overwriting each other, and the top-level location keeps test files
    # from mixing into the YYYY/MM/ hourly archive tree.
    test_out: Path | None = None
    if test_mode:
        ARCHIVE_DIR.mkdir(exist_ok=True)
        test_out = ARCHIVE_DIR / datetime.now().strftime("test-%Y-%m-%d_%H-%M-%S.mp3")

    log.info("%s archiver starting.", LABEL)
    log.info("Stream  : %s", STREAM_URL)
    log.info("Output  : %s",
             test_out.resolve() if test_out else ARCHIVE_DIR.resolve())
    log.info("Segments: %d s%s", segment_seconds,
             "  [TEST MODE — single capture, ffmpeg exits at end]" if test_mode else "")
    if transcribe:
        log.info("Transcription: ENABLED inline (per-hour .txt/.json sidecars). "
                 "For a decoupled setup, run `transcribe.py --watch` instead.")

    stop_watcher = None
    watcher = None
    if transcribe:
        stop_watcher = threading.Event()
        watcher = threading.Thread(target=_watcher, args=(stop_watcher,), daemon=True)
        watcher.start()

    while running:
        if not test_mode:
            _rotate_in_progress_hour()
        try:
            code = _run_ffmpeg(segment_seconds, test_out=test_out)
        except FileNotFoundError:
            log.error(
                "ffmpeg not found (tried: %s). Install ffmpeg and make sure it's on "
                "PATH (Windows: winget install Gyan.FFmpeg; macOS: brew install ffmpeg; "
                "Debian/Ubuntu: sudo apt install ffmpeg), or set the FFMPEG env var to "
                "its full path.",
                FFMPEG,
            )
            sys.exit(1)

        if not running or test_mode:
            break

        if code == 0:
            log.info("ffmpeg exited cleanly — restarting immediately.")
        else:
            log.warning("ffmpeg exited with code %d — reconnecting in %d s.",
                        code, RECONNECT_DELAY)
            time.sleep(RECONNECT_DELAY)

    if watcher is not None:
        log.info("Archiver stopped. Waiting for inline transcription to finish...")
        stop_watcher.set()
        watcher.join(timeout=120)
    log.info("Done. Files saved to: %s", ARCHIVE_DIR.resolve())


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Radio stream archiver (records only).")
    parser.add_argument("--test", action="store_true",
                        help="Record one 60-second segment, then exit.")
    parser.add_argument("--transcribe", action="store_true",
                        help="Also transcribe + quality-check each completed segment "
                             "inline (single-process mode; requires the venv). Prefer "
                             "running `transcribe.py --watch` as a separate process.")
    args = parser.parse_args()

    setup_logging()
    run(test_mode=args.test, transcribe=args.transcribe)
