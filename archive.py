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
from datetime import date, datetime, timedelta
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
# Touch this file to ask the recorder + watcher to exit cleanly at the next
# safe moment (recorder: next hour boundary + 5s; watcher: end of current
# file). systemd Restart=always brings them back up with fresh code on disk.
# This file is gone (in tmpfs) after a reboot — no manual cleanup needed.
RESTART_SENTINEL = Path("/tmp/RADIO_ARCH_RESTART_REQUIRED")
# How long after the clock-hour boundary to signal ffmpeg, so the previous
# hour is fully closed and the new hour has rolled to a fresh file.
RESTART_BOUNDARY_BUFFER_SECONDS = 5
RESTART_POLL_INTERVAL_SECONDS = 60
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


def _ask_ffmpeg_to_exit() -> None:
    """
    Signal the running ffmpeg child to exit cleanly so the in-progress MP3
    segment is closed with a proper trailer. POSIX SIGTERM works directly; on
    Windows we send CTRL_BREAK_EVENT to the process group (ffmpeg was launched
    with CREATE_NEW_PROCESS_GROUP for this), falling back to terminate() if
    the break event is ignored. Safe to call when there is no live child.
    """
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


def _handle_signal(sig: int, frame: FrameType | None) -> None:
    """
    Ask ffmpeg to stop cleanly so the in-progress MP3 segment is closed with
    a proper trailer, then let the main loop exit.
    """
    global running
    log.info("Shutdown signal received — stopping after current ffmpeg exits.")
    running = False
    _ask_ffmpeg_to_exit()


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


def _ensure_month_dirs(months_ahead: int = 1) -> None:
    """
    Create archive/YYYY/MM/ for the current month and the next `months_ahead`
    months. ffmpeg's segment muxer does not create intermediate directories,
    so we have to prepare them up-front — otherwise the first segment after a
    month rollover would fail to open. The default `months_ahead=1` keeps the
    archive tree tidy (at most one empty future directory) and is refreshed
    by `_month_dir_watchdog` on a periodic tick so an arbitrarily long
    uninterrupted run still stays one month ahead of the wall clock.
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


def _month_dir_watchdog(stop: threading.Event, interval_seconds: int = 3600) -> None:
    """
    Daemon-thread tick that keeps current + next month's archive dirs alive
    over any uptime. Runs immediately, then again every `interval_seconds`
    until `stop` is set. Hourly is well below the once-per-month frequency
    actually needed — overhead is one or two mkdir-noops.
    """
    while True:
        try:
            _ensure_month_dirs(months_ahead=1)
        except Exception as e:
            log.warning("month-dir watchdog: %r (will retry next tick).", e)
        if stop.wait(interval_seconds):
            return


def _sentinel_mtime() -> float:
    """mtime of the restart sentinel, or 0 if it doesn't exist."""
    try:
        return RESTART_SENTINEL.stat().st_mtime
    except OSError:
        return 0.0


def _restart_watchdog(stop: threading.Event,
                      poll_seconds: int = RESTART_POLL_INTERVAL_SECONDS) -> None:
    """
    Watch for `RESTART_SENTINEL` and, when found, schedule a graceful exit at
    the next clock-hour boundary + RESTART_BOUNDARY_BUFFER_SECONDS so the
    in-progress hour is fully captured. Then signal ffmpeg, set `running = False`,
    and unlink the sentinel.

    Method-A race-handling: the baseline mtime is fixed when the thread starts,
    so a sentinel that already existed at startup is treated as "previous
    incarnation already handled this" and does NOT trigger a fresh restart.
    The watchdog reacts only to mtime values strictly greater than the
    baseline — i.e. a `touch` after the service started. The transcriber does
    NOT delete the sentinel; the recorder is intentionally the slow service so
    that by the time it exits, the transcriber has already responded.
    """
    global running
    baseline = _sentinel_mtime()
    while not stop.wait(poll_seconds):
        mt = _sentinel_mtime()
        if mt <= baseline:
            continue
        # Sentinel touched after we started.
        log.info("Restart sentinel detected (%s, mtime=%s); will exit at next "
                 "hour boundary + %ds.",
                 RESTART_SENTINEL, datetime.fromtimestamp(mt).isoformat(),
                 RESTART_BOUNDARY_BUFFER_SECONDS)
        now = datetime.now()
        next_hour = (now + timedelta(hours=1)).replace(
            minute=0, second=0, microsecond=0)
        wait_until = next_hour + timedelta(seconds=RESTART_BOUNDARY_BUFFER_SECONDS)
        wait_s = max(1.0, (wait_until - now).total_seconds())
        log.info("Sleeping %.0fs until %s, then signaling ffmpeg.",
                 wait_s, wait_until.isoformat(timespec="seconds"))
        if stop.wait(wait_s):
            return  # process shutting down for another reason
        log.info("Restart time reached. Signaling ffmpeg and unlinking sentinel.")
        running = False
        _ask_ffmpeg_to_exit()
        try:
            RESTART_SENTINEL.unlink()
        except FileNotFoundError:
            pass
        except OSError as e:
            log.warning("Could not unlink %s: %s", RESTART_SENTINEL, e)
        return


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

    # Keep the YYYY/MM dirs alive over the lifetime of this process. Skipped
    # in test mode (test mode writes to a single explicit path at archive
    # root and never relies on the YYYY/MM tree).
    stop_month_watchdog: threading.Event | None = None
    month_watchdog: threading.Thread | None = None
    stop_restart_watchdog: threading.Event | None = None
    restart_watchdog: threading.Thread | None = None
    if not test_mode:
        stop_month_watchdog = threading.Event()
        month_watchdog = threading.Thread(
            target=_month_dir_watchdog,
            args=(stop_month_watchdog,),
            daemon=True,
            name="month-dir-watchdog",
        )
        month_watchdog.start()
        # Restart-on-sentinel watcher. See _restart_watchdog() for the design.
        stop_restart_watchdog = threading.Event()
        restart_watchdog = threading.Thread(
            target=_restart_watchdog,
            args=(stop_restart_watchdog,),
            daemon=True,
            name="restart-watchdog",
        )
        restart_watchdog.start()

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
    if stop_month_watchdog is not None:
        stop_month_watchdog.set()
        if month_watchdog is not None:
            month_watchdog.join(timeout=5)
    if stop_restart_watchdog is not None:
        stop_restart_watchdog.set()
        if restart_watchdog is not None:
            restart_watchdog.join(timeout=5)
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
