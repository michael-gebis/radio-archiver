#!/usr/bin/env python3
"""
Best-effort song identification for archived hours (Tier 1).

Pipeline:
  - For each transcribed hour, compute the "music-suspected" regions —
    the complement of speech (from Whisper `segments`) and silence
    (from `quality.silence_periods`).
  - For each music region above `MIN_MUSIC_SECONDS`, extract a chunk via
    ffmpeg, generate a Chromaprint fingerprint via `fpcalc`, and query
    AcoustID's free API for matches.
  - Write a `music_hint` block into the JSON sidecar capturing both the
    lookup audit trail (`lookups[]`) and the chosen match (`match`, or
    null when nothing meets the threshold).

The `_hint` suffix on the field, the prose `note`, and the gracefully
nullable `match` mirror the `schedule_hint` convention — non-authoritative
data treated as best-effort.

This is Tier 1 of the music-ID design captured in deploy-notes.md §15.
Tier 1 uses AcoustID only (free, ~30-50% hit rate expected on
college radio). The `lookups[]` array shape is forward-compatible with
adding a paid provider as a cascade (deploy-notes §15.4): when added,
each segment's `lookups[]` will gain a second entry without any
schema migration.

Usage:
    python music_id.py FILE.json [FILE2.json ...]      # specific sidecars
    python music_id.py --all                           # walk archive/ dir
    python music_id.py --all --force                   # re-process all
    python music_id.py FILE.json --apply               # actually call API
    # Without --apply, runs as a dry-run (no API calls, no writes).

Requirements:
    sudo apt install libchromaprint-tools  # provides fpcalc
    pip install pyacoustid                 # in the project venv
    export ACOUSTID_API_KEY="..."          # from your acoustid.org account
"""

import argparse
import json
import logging
import os
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

from config import CONFIG
from quality import FFMPEG

ARCHIVE_DIR = Path(CONFIG["paths"]["archive_dir"])

# ---------------------------------------------------------------------------
# Tunables
# ---------------------------------------------------------------------------
# Music regions shorter than this aren't worth fingerprinting (too short
# to plausibly be a song; AcoustID needs ~30 s minimum for a useful match).
MIN_MUSIC_SECONDS = 30

# How much of each music region to fingerprint. AcoustID matches well on
# ~60 s; longer doesn't help and just costs more transcode time.
FINGERPRINT_LENGTH_SECONDS = 60

# AcoustID returns a `score` 0.0-1.0 per match. Above this threshold, we
# accept the top hit as the `match`. Below, we still record the lookup
# (so future tuning can see what was rejected), but `match` stays null.
# See deploy-notes.md §15.4 for the calibration rationale; 0.85 is the
# conservative starting point.
ACOUSTID_THRESHOLD = 0.85

# Path to fpcalc (Chromaprint CLI). Installed by `libchromaprint-tools`
# on Debian/Ubuntu. Override via env var if it's installed somewhere odd.
FPCALC = os.environ.get("FPCALC", "fpcalc")

# AcoustID API key. Required for --apply. Sign up at https://acoustid.org/
# for a free key.
ACOUSTID_API_KEY = os.environ.get("ACOUSTID_API_KEY")

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Music-region detection (complement of speech + silence)
# ---------------------------------------------------------------------------

def music_regions(sidecar: dict) -> list[tuple[float, float]]:
    """
    Compute the music-suspected regions of an hour from the existing
    sidecar fields, as a list of `(start, end)` tuples in seconds.

    The hour's audio is partitioned into three categories:

      - Speech    — covered by `segments[]` from the Whisper transcript.
      - Silence   — covered by `quality.silence_periods[]`.
      - Music     — the complement of the above two over [0, audio_seconds].

    Regions shorter than `MIN_MUSIC_SECONDS` are filtered out (too short
    to plausibly be a song; usually just a DJ-overlapped intro).

    Returns an empty list when the file is shorter than `MIN_MUSIC_SECONDS`
    or when no music is detected.
    """
    audio_seconds = float(sidecar.get("audio_seconds") or 0.0)
    if audio_seconds < MIN_MUSIC_SECONDS:
        return []

    # Collect non-music intervals (speech + silence).
    non_music: list[tuple[float, float]] = []
    for seg in sidecar.get("segments") or []:
        if "start" in seg and "end" in seg:
            non_music.append((float(seg["start"]), float(seg["end"])))
    for sp in (sidecar.get("quality") or {}).get("silence_periods") or []:
        s = float(sp.get("start") or 0.0)
        e = sp.get("end")
        if e is None:
            d = sp.get("duration")
            e = s + float(d) if d else audio_seconds
        non_music.append((s, float(e)))

    if not non_music:
        # Whole hour is music (rare but possible for an instrumental block).
        return [(0.0, audio_seconds)]

    # Merge overlapping non-music intervals.
    non_music.sort()
    merged = [non_music[0]]
    for s, e in non_music[1:]:
        if s <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], e))
        else:
            merged.append((s, e))

    # Music = complement over [0, audio_seconds].
    music: list[tuple[float, float]] = []
    cursor = 0.0
    for s, e in merged:
        if s > cursor:
            music.append((cursor, s))
        cursor = max(cursor, e)
    if cursor < audio_seconds:
        music.append((cursor, audio_seconds))

    return [(s, e) for s, e in music if (e - s) >= MIN_MUSIC_SECONDS]


# ---------------------------------------------------------------------------
# Fingerprinting + lookup
# ---------------------------------------------------------------------------

def _fpcalc_version() -> str | None:
    """fpcalc -version, or None if unreachable."""
    try:
        result = subprocess.run(
            [FPCALC, "-version"], capture_output=True, text=True, check=True,
        )
        return result.stdout.strip().splitlines()[0]
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None


def _pkg_version(name: str) -> str | None:
    try:
        return version(name)
    except PackageNotFoundError:
        return None


def fingerprint_chunk(mp3: Path, start: float, length: float) -> tuple[float, str]:
    """
    Extract a chunk of `mp3` starting at `start` seconds, of length up to
    `length` seconds, and return `(duration, chromaprint_fingerprint)`
    via `fpcalc`.

    Uses ffmpeg's stream-copy when possible (fast, lossless) and falls
    back to a transcode if copy fails (some MP3s with sync errors won't
    re-mux cleanly). The temp file is cleaned up before returning.
    """
    with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp:
        tmp_path = Path(tmp.name)

    try:
        cmd_copy = [
            FFMPEG, "-y", "-hide_banner", "-loglevel", "error", "-nostdin",
            "-ss", f"{start:.3f}",
            "-t", f"{length:.3f}",
            "-i", str(mp3),
            "-c", "copy",
            str(tmp_path),
        ]
        result = subprocess.run(cmd_copy, capture_output=True, text=True)
        if result.returncode != 0 or tmp_path.stat().st_size == 0:
            # Fall back to re-encode.
            cmd_xcode = list(cmd_copy)
            cmd_xcode[cmd_xcode.index("-c") + 1] = "libmp3lame"
            cmd_xcode.insert(-1, "-b:a")
            cmd_xcode.insert(-1, "128k")
            subprocess.run(cmd_xcode, capture_output=True, text=True, check=True)

        fp_result = subprocess.run(
            [FPCALC, "-json", str(tmp_path)],
            capture_output=True, text=True, check=True,
        )
        data = json.loads(fp_result.stdout)
        return float(data["duration"]), data["fingerprint"]
    finally:
        try:
            tmp_path.unlink()
        except OSError:
            pass


def lookup_acoustid(api_key: str, fingerprint: str, duration: float) -> dict:
    """Query AcoustID and return the raw response dict. Raises on errors."""
    import acoustid  # lazy: dry-run doesn't need the package installed
    return acoustid.lookup(
        api_key, fingerprint, duration,
        meta="recordings releasegroups compress",
    )


def parse_acoustid_response(response: dict) -> list[dict]:
    """
    Flatten AcoustID's JSON response into a list of compact match dicts,
    sorted by `score` desc:

        [{score, recording_id, title, artists, release}, ...]

    `artists` is a list of names (often one, occasionally multiple
    collaborators). `release` is the first releasegroup's title.
    """
    out: list[dict] = []
    if response.get("status") != "ok":
        return out
    for res in response.get("results") or []:
        score = float(res.get("score") or 0.0)
        for rec in res.get("recordings") or []:
            artists = [a.get("name") or "" for a in rec.get("artists") or []]
            release = ""
            for rg in rec.get("releasegroups") or []:
                release = rg.get("title") or ""
                break
            out.append({
                "score": round(score, 3),
                "recording_id": rec.get("id"),
                "title": rec.get("title") or "",
                "artists": [a for a in artists if a],
                "release": release,
            })
    out.sort(key=lambda x: x["score"], reverse=True)
    return out


# ---------------------------------------------------------------------------
# Sidecar processing
# ---------------------------------------------------------------------------

def _should_skip(sidecar: dict) -> str | None:
    """Return a skip reason or None."""
    if (sidecar.get("quality") or {}).get("is_off_air"):
        return "off-air"
    audio_seconds = sidecar.get("audio_seconds") or 0
    if audio_seconds < MIN_MUSIC_SECONDS:
        return f"audio too short ({audio_seconds:.0f}s)"
    return None


def _build_hint_meta(reason: str | None = None) -> dict:
    """Common metadata for the music_hint block (excluding the segments)."""
    meta = {
        "source": "acoustid",
        "note": ("Best-effort song identification via Chromaprint/AcoustID; "
                 "coverage of indie/college-radio music is uneven. `match` "
                 "is present only when the AcoustID score >= "
                 f"{ACOUSTID_THRESHOLD}."),
        "fpcalc_version": _fpcalc_version(),
        "pyacoustid_version": _pkg_version("pyacoustid"),
        "looked_up_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "threshold": ACOUSTID_THRESHOLD,
    }
    if reason:
        meta["reason"] = reason
    return meta


def process_sidecar(jp: Path, apply: bool, force: bool) -> dict:
    """
    Process a single sidecar. Returns a stats dict suitable for the
    end-of-run summary.
    """
    stats = {
        "skipped": False,
        "reason": None,
        "regions_found": 0,
        "lookups_made": 0,
        "matches": 0,
    }

    try:
        d = json.loads(jp.read_text(encoding="utf-8"))
    except Exception as e:
        stats["skipped"] = True
        stats["reason"] = f"bad json: {e}"
        return stats

    if d.get("music_hint") and not force:
        stats["skipped"] = True
        stats["reason"] = "music_hint already present (--force to redo)"
        return stats

    skip = _should_skip(d)
    if skip:
        stats["skipped"] = True
        stats["reason"] = skip
        return stats

    mp3 = jp.with_suffix(".mp3")
    if not mp3.exists():
        stats["skipped"] = True
        stats["reason"] = "mp3 missing (rotated to B2?)"
        return stats

    regions = music_regions(d)
    stats["regions_found"] = len(regions)

    if not regions:
        log.info("%s: no music regions >= %ds", jp.name, MIN_MUSIC_SECONDS)
        if apply:
            d["music_hint"] = {**_build_hint_meta(reason="no music regions detected"),
                               "segments": []}
            jp.write_text(json.dumps(d, indent=2, ensure_ascii=False), encoding="utf-8")
        return stats

    if not apply:
        log.info("%s: would fingerprint %d region(s):",
                 jp.name, len(regions))
        for s, e in regions:
            log.info("    %.1fs - %.1fs  (%.1fs)", s, e, e - s)
        return stats

    log.info("%s: %d music region(s); querying AcoustID...",
             jp.name, len(regions))
    segments_data = []
    for start, end in regions:
        duration_region = end - start
        seg = {
            "start": round(start, 1),
            "end": round(end, 1),
            "duration": round(duration_region, 1),
            "lookups": [],
            "match": None,
        }
        try:
            fp_duration, fp = fingerprint_chunk(
                mp3, start, min(duration_region, FINGERPRINT_LENGTH_SECONDS),
            )
            response = lookup_acoustid(ACOUSTID_API_KEY, fp, fp_duration)
            parsed = parse_acoustid_response(response)
            stats["lookups_made"] += 1
            if parsed:
                top = parsed[0]
                seg["lookups"].append({
                    "provider": "acoustid",
                    "status": "ok",
                    "score": top["score"],
                })
                if top["score"] >= ACOUSTID_THRESHOLD:
                    seg["match"] = {
                        "source": "acoustid",
                        "score": top["score"],
                        "title": top["title"],
                        "artists": top["artists"],
                        "release": top["release"],
                        "recording_id": top["recording_id"],
                    }
                    stats["matches"] += 1
                    log.info("    %.1fs match: %s — %s (score %.2f)",
                             start,
                             ", ".join(top["artists"]) or "?",
                             top["title"] or "?",
                             top["score"])
                else:
                    log.info("    %.1fs no_match: top score %.2f below threshold %.2f",
                             start, top["score"], ACOUSTID_THRESHOLD)
            else:
                seg["lookups"].append({"provider": "acoustid", "status": "no_match"})
                log.info("    %.1fs no AcoustID match", start)
        except Exception as e:
            log.warning("    %.1fs lookup failed: %s", start, e)
            seg["lookups"].append({
                "provider": "acoustid",
                "status": "error",
                "error": str(e)[:200],
            })
        segments_data.append(seg)

    d["music_hint"] = {**_build_hint_meta(), "segments": segments_data}
    jp.write_text(json.dumps(d, indent=2, ensure_ascii=False), encoding="utf-8")
    return stats


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Identify songs in archived hours via AcoustID (Tier 1).",
    )
    parser.add_argument("files", nargs="*",
                        help="Specific sidecar .json files to process.")
    parser.add_argument("--all", action="store_true",
                        help="Walk the archive dir for sidecars to process.")
    parser.add_argument("--force", action="store_true",
                        help="Re-process even if music_hint is already present.")
    parser.add_argument("--apply", action="store_true",
                        help="Actually call AcoustID and write sidecars. "
                             "Default is a dry-run that lists candidate regions.")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    if args.apply and not ACOUSTID_API_KEY:
        log.error("ACOUSTID_API_KEY env var is required for --apply.")
        log.error("Sign up at https://acoustid.org/ for a free key, then:")
        log.error("    export ACOUSTID_API_KEY='your-key-here'")
        sys.exit(2)

    if args.files:
        targets = [Path(f) for f in args.files]
    elif args.all:
        targets = sorted(ARCHIVE_DIR.rglob("*.json"))
    else:
        parser.error("Provide files, or use --all.")

    if not targets:
        log.info("No sidecars to process.")
        sys.exit(0)

    mode = "APPLY" if args.apply else "dry-run"
    log.info("%s mode; %d sidecar(s) to consider.", mode, len(targets))

    totals = {"processed": 0, "skipped": 0,
              "regions": 0, "lookups": 0, "matches": 0}
    for jp in targets:
        result = process_sidecar(jp, apply=args.apply, force=args.force)
        if result["skipped"]:
            totals["skipped"] += 1
            log.info("  SKIP %s: %s", jp.name, result["reason"])
        else:
            totals["processed"] += 1
            totals["regions"] += result["regions_found"]
            totals["lookups"] += result["lookups_made"]
            totals["matches"] += result["matches"]

    log.info("Done. processed=%d skipped=%d  regions=%d lookups=%d matches=%d",
             totals["processed"], totals["skipped"],
             totals["regions"], totals["lookups"], totals["matches"])
    if totals["lookups"]:
        rate = 100.0 * totals["matches"] / totals["lookups"]
        log.info("Tier 1 hit-rate: %d/%d (%.1f%%) above threshold %.2f",
                 totals["matches"], totals["lookups"], rate, ACOUSTID_THRESHOLD)
    if not args.apply:
        log.info("(dry-run; nothing was sent to AcoustID or written. "
                 "Pass --apply to actually do it.)")
