#!/usr/bin/env python3
"""
Archive merge tool — reconcile two redundant recordings into one canonical tree.

When two machines record the same stream in parallel (for redundancy across
reboots / power outages / network blips), this script merges their archives
hour-by-hour into a fresh output directory, **never modifying the sources**.

Both machines name files by the clock hour (`-segment_atclocktime`), so
`2026-05-27_16-00.mp3` on each side covers the same wall-clock window — the
filename is the merge key. The per-hour JSON sidecar's `quality` block (size
ratio, silence periods, decode errors) is the decision oracle, so no audio
inspection is needed for the common case.

Two phases:
  1. **Winner-takes-all per hour** using `quality.ok` / `size.ratio` / total
     silence. Copies the chosen `.mp3` + `.json` + `.txt` trio.
  2. **Cross-fill splice** (default ON; disable with --no-splice). When both
     candidates are bad, pick the good portion from each via their
     `silence_periods` and concat losslessly with ffmpeg `-c copy`. The merged
     `.mp3` is then re-transcribed (force) so its JSON describes the new audio.

Schedule directories are merged in parallel (simpler logic: prefer `parse_ok`
then higher `event_count` per date).

Usage:
    python merge_archives.py
        --archive-a <pathA> --archive-b <pathB> --archive-out <merged>
       [--schedule-a <pathA> --schedule-b <pathB> --schedule-out <merged_sched>]
       [--no-splice] [--dry-run] [--force-out] [--report <file>]
"""

import argparse
import json
import logging
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import quality
import transcribe

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Candidate model — one side's view of a given hour
# ---------------------------------------------------------------------------

@dataclass
class Side:
    name: str
    archive_dir: Path


@dataclass
class Candidate:
    """One side's files for a single hour. `data` is the parsed JSON sidecar
    (or None if missing/unreadable)."""
    side: Side
    mp3: Path
    json_path: Path
    txt_path: Path
    data: dict | None


def _nested_hour_path(archive_dir: Path, hour: str, ext: str) -> Path:
    """archive_dir/<YYYY>/<MM>/<hour><ext> — the canonical write location."""
    dt = datetime.strptime(hour, transcribe.FILENAME_DT_FORMAT)
    return archive_dir / f"{dt.year:04d}" / f"{dt.month:02d}" / f"{hour}{ext}"


def _hour_path(archive_dir: Path, hour: str, ext: str) -> Path:
    """
    Resolve where a given hour's `.mp3` / `.json` / `.txt` already lives on a
    side. New archives are nested under `<year>/<month>/`; legacy archives are
    flat. Prefer the nested location if it exists, otherwise fall back to flat.
    """
    nested = _nested_hour_path(archive_dir, hour, ext)
    if nested.exists():
        return nested
    return archive_dir / f"{hour}{ext}"


def _load_candidate(side: Side, hour: str) -> Candidate | None:
    """Return a Candidate if the side has a non-empty .mp3 for `hour`, else None."""
    mp3 = _hour_path(side.archive_dir, hour, ".mp3")
    if not mp3.exists() or mp3.stat().st_size == 0:
        return None
    json_path = _hour_path(side.archive_dir, hour, ".json")
    txt_path = _hour_path(side.archive_dir, hour, ".txt")
    data: dict | None = None
    if json_path.exists():
        try:
            data = json.loads(json_path.read_text(encoding="utf-8"))
        except Exception as e:
            log.warning("Bad JSON %s: %s", json_path, e)
    return Candidate(side=side, mp3=mp3, json_path=json_path, txt_path=txt_path, data=data)


def _quality(c: Candidate | None) -> dict:
    if not c or not c.data:
        return {}
    return c.data.get("quality") or {}


def _ok(c: Candidate | None) -> bool:
    return bool(_quality(c).get("ok"))


def _size_ratio(c: Candidate | None) -> float:
    return float((_quality(c).get("size") or {}).get("ratio") or 0.0)


def _silence_total(c: Candidate | None) -> float:
    """Total silent seconds reported by quality. Inf if unknown (penalizes
    candidates without metadata in tie-breaks)."""
    if not c or not c.data:
        return float("inf")
    return sum((p.get("duration") or 0.0) for p in (_quality(c).get("silence_periods") or []))


def _summary(c: Candidate | None) -> dict:
    """Compact view of a candidate for the merge log."""
    if c is None:
        return {"present": False}
    return {
        "present": True,
        "size_ratio": _size_ratio(c),
        "size_ok": bool((_quality(c).get("size") or {}).get("ok")),
        "ok": _ok(c),
        "silence_total_s": round(_silence_total(c), 1) if _silence_total(c) != float("inf") else None,
        "audio_seconds": (c.data or {}).get("audio_seconds"),
        "mtime": c.mp3.stat().st_mtime,
    }


# ---------------------------------------------------------------------------
# Phase 1 decision
# ---------------------------------------------------------------------------

def _decide(a: Candidate | None, b: Candidate | None, splice: bool) -> tuple[str, str]:
    """Return (mode, reason). Modes: a_only, b_only, use_a, use_b, spliced, neither."""
    if a and not b:  return "a_only", "B missing"
    if b and not a:  return "b_only", "A missing"
    if not a and not b:
        return "neither", "both missing"

    if _ok(a) and not _ok(b):  return "use_a", "A quality.ok, B not"
    if _ok(b) and not _ok(a):  return "use_b", "B quality.ok, A not"

    if _ok(a) and _ok(b):
        ra, rb = _size_ratio(a), _size_ratio(b)
        if ra > rb:                   return "use_a", f"both ok, A ratio {ra:.3f} > {rb:.3f}"
        if rb > ra:                   return "use_b", f"both ok, B ratio {rb:.3f} > {ra:.3f}"
        if a.mp3.stat().st_mtime <= b.mp3.stat().st_mtime:
            return "use_a", "both ok, A mtime earlier"
        return "use_b", "both ok, B mtime earlier"

    # neither ok
    if splice:
        return "spliced", "neither quality.ok; splicing good regions"
    ra, rb = _size_ratio(a), _size_ratio(b)
    if ra != rb:
        return ("use_a", f"both bad, A ratio {ra:.3f} > {rb:.3f}") if ra > rb \
            else ("use_b", f"both bad, B ratio {rb:.3f} > {ra:.3f}")
    sa, sb = _silence_total(a), _silence_total(b)
    return ("use_a", f"both bad, A less silence ({sa:.0f}s ≤ {sb:.0f}s)") if sa <= sb \
        else ("use_b", f"both bad, B less silence ({sb:.0f}s < {sa:.0f}s)")


# ---------------------------------------------------------------------------
# Copy / splice actions
# ---------------------------------------------------------------------------

def _copy_trio(src: Candidate, out_dir: Path) -> None:
    """Copy the chosen side's mp3 + json + txt into the output dir, written
    into the canonical YYYY/MM/ subdir."""
    dest_mp3 = _nested_hour_path(out_dir, src.mp3.stem, ".mp3")
    dest_mp3.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src.mp3, dest_mp3)
    if src.json_path.exists():
        shutil.copy2(src.json_path, dest_mp3.with_suffix(".json"))
    if src.txt_path.exists():
        shutil.copy2(src.txt_path, dest_mp3.with_suffix(".txt"))


def _good_regions(c: Candidate) -> list[tuple[float, float]]:
    """`[0, audio_seconds]` minus this side's silence_periods (open-ended
    silences are clipped to end-of-audio)."""
    end = float((c.data or {}).get("audio_seconds") or 0.0)
    if end <= 0:
        return []
    sil: list[tuple[float, float]] = []
    for p in (_quality(c).get("silence_periods") or []):
        s = float(p.get("start") or 0.0)
        e = p.get("end")
        e = float(e) if e is not None else end
        sil.append((s, min(e, end)))
    sil.sort()
    good: list[tuple[float, float]] = []
    cursor = 0.0
    for s, e in sil:
        if s > cursor:
            good.append((cursor, s))
        cursor = max(cursor, e)
    if cursor < end:
        good.append((cursor, end))
    return good


def _splice_plan(a: Candidate, b: Candidate) -> list[tuple[Candidate, float, float]]:
    """Return a chronological list of (source, start, end) chunks. Picks the
    side that's `good` at each point; if both are good, prefers the side with
    the larger MP3 (stable tie-break)."""
    ga = _good_regions(a)
    gb = _good_regions(b)
    if not ga and not gb:
        return []

    breakpoints = sorted({0.0}
                         | {p for s, e in ga for p in (s, e)}
                         | {p for s, e in gb for p in (s, e)})

    def good_at(regions: list[tuple[float, float]], t: float) -> bool:
        for s, e in regions:
            if s <= t < e:
                return True
        return False

    sa = a.mp3.stat().st_size
    sb = b.mp3.stat().st_size

    plan: list[tuple[Candidate, float, float]] = []
    for i in range(len(breakpoints) - 1):
        s, e = breakpoints[i], breakpoints[i + 1]
        if e - s <= 0.05:
            continue
        mid = (s + e) / 2.0
        ag, bg = good_at(ga, mid), good_at(gb, mid)
        if ag and bg:
            src = a if sa >= sb else b
        elif ag:
            src = a
        elif bg:
            src = b
        else:
            continue  # neither side has audio here; unreconstructable
        # merge with previous chunk if same source and contiguous
        if plan and plan[-1][0] is src and abs(plan[-1][2] - s) < 0.05:
            prev_src, prev_s, _ = plan[-1]
            plan[-1] = (prev_src, prev_s, e)
        else:
            plan.append((src, s, e))
    return plan


def _splice(a: Candidate, b: Candidate, out_mp3: Path) -> int:
    """Splice good regions from both inputs into out_mp3 (lossless `-c copy`).
    Returns the number of chunks concatenated. Raises if nothing usable."""
    plan = _splice_plan(a, b)
    if not plan:
        raise RuntimeError("nothing usable to splice (no good regions on either side)")
    with tempfile.TemporaryDirectory() as td_str:
        td = Path(td_str)
        pieces: list[Path] = []
        for i, (src, s, e) in enumerate(plan):
            piece = td / f"piece_{i:03d}.mp3"
            cmd = [
                quality.FFMPEG, "-y", "-hide_banner", "-loglevel", "error",
                "-ss", f"{s:.3f}", "-t", f"{e - s:.3f}",
                "-i", str(src.mp3),
                "-c", "copy", str(piece),
            ]
            subprocess.run(cmd, check=True)
            pieces.append(piece)
        list_path = td / "concat.txt"
        # concat demuxer requires single-quoted POSIX-style paths
        list_path.write_text(
            "\n".join(f"file '{p.as_posix()}'" for p in pieces) + "\n",
            encoding="utf-8",
        )
        cmd = [
            quality.FFMPEG, "-y", "-hide_banner", "-loglevel", "error",
            "-f", "concat", "-safe", "0", "-i", str(list_path),
            "-c", "copy", str(out_mp3),
        ]
        subprocess.run(cmd, check=True)
    return len(plan)


# ---------------------------------------------------------------------------
# Schedule merge — simpler decision per date
# ---------------------------------------------------------------------------

def _schedule_decide(a: dict | None, b: dict | None) -> str:
    if a and not b:    return "use_a"
    if b and not a:    return "use_b"
    if a and b:
        oa = bool(a.get("parse_ok"))
        ob = bool(b.get("parse_ok"))
        if oa and not ob:  return "use_a"
        if ob and not oa:  return "use_b"
        ea = int(a.get("event_count") or 0)
        eb = int(b.get("event_count") or 0)
        return "use_a" if ea >= eb else "use_b"
    return "neither"


def merge_schedules(sched_a: Path, sched_b: Path, out: Path,
                    dry: bool, force_out: bool) -> list[dict]:
    out.mkdir(parents=True, exist_ok=True)
    if any(out.iterdir()) and not force_out and not dry:
        raise SystemExit(f"--schedule-out {out} not empty (use --force-out).")

    dates: set[str] = set()
    for d in (sched_a, sched_b):
        if d.is_dir():
            for j in d.glob("*.json"):
                dates.add(j.stem)
    log.info("Schedule: %d date(s) to consider.", len(dates))

    decisions: list[dict] = []
    for date in sorted(dates):
        ja = sched_a / f"{date}.json"
        jb = sched_b / f"{date}.json"
        ad = json.loads(ja.read_text(encoding="utf-8")) if ja.exists() else None
        bd = json.loads(jb.read_text(encoding="utf-8")) if jb.exists() else None
        mode = _schedule_decide(ad, bd)
        decisions.append({"date": date, "mode": mode})
        if dry or mode == "neither":
            continue
        src_dir = sched_a if mode == "use_a" else sched_b
        for ext in (".json", ".calendar.html", ".page.html"):
            p = src_dir / f"{date}{ext}"
            if p.exists():
                shutil.copy2(p, out / p.name)
    counts = {m: sum(1 for x in decisions if x["mode"] == m) for m in ("use_a", "use_b", "neither")}
    log.info("Schedule merge: %s", counts)
    return decisions


# ---------------------------------------------------------------------------
# Main audio merge
# ---------------------------------------------------------------------------

def merge_archives(arch_a: Path, arch_b: Path, out: Path,
                   splice: bool, dry: bool, force_out: bool) -> dict:
    if not arch_a.is_dir() or not arch_b.is_dir():
        raise SystemExit(f"archive dirs must exist: {arch_a}, {arch_b}")
    out.mkdir(parents=True, exist_ok=True)
    if any(out.iterdir()) and not force_out and not dry:
        raise SystemExit(f"--archive-out {out} not empty (use --force-out).")

    side_a = Side("A", arch_a)
    side_b = Side("B", arch_b)

    # Collect hour-keys; validate the filename pattern. rglob picks up both
    # the new YYYY/MM/ layout and the legacy flat layout.
    hours: set[str] = set()
    for d in (arch_a, arch_b):
        for f in d.rglob("*.mp3"):
            try:
                datetime.strptime(f.stem, transcribe.FILENAME_DT_FORMAT)
                hours.add(f.stem)
            except ValueError:
                continue
    log.info("Audio: %d unique hour(s) to consider.", len(hours))

    # Clock-skew sanity warning.
    a_hours = {f.stem for f in arch_a.rglob("*.mp3")} & hours
    b_hours = {f.stem for f in arch_b.rglob("*.mp3")} & hours
    if a_hours and b_hours and not (a_hours & b_hours):
        log.warning(
            "No overlapping hour-keys between A (%d) and B (%d). Check NTP / clock skew.",
            len(a_hours), len(b_hours),
        )

    entries: list[dict] = []
    summary = {k: 0 for k in
               ("a_only", "b_only", "use_a", "use_b", "spliced", "neither", "errors")}

    for hour in sorted(hours):
        ca = _load_candidate(side_a, hour)
        cb = _load_candidate(side_b, hour)
        mode, reason = _decide(ca, cb, splice=splice)
        entry: dict = {"hour": hour, "mode": mode, "reason": reason,
                       "a": _summary(ca), "b": _summary(cb)}
        try:
            if dry:
                pass
            elif mode in ("a_only", "use_a"):
                _copy_trio(ca, out)  # type: ignore[arg-type]
            elif mode in ("b_only", "use_b"):
                _copy_trio(cb, out)  # type: ignore[arg-type]
            elif mode == "spliced":
                out_mp3 = _nested_hour_path(out, hour, ".mp3")
                out_mp3.parent.mkdir(parents=True, exist_ok=True)
                chunks = _splice(ca, cb, out_mp3)  # type: ignore[arg-type]
                entry["splice_chunks"] = chunks
                # Regenerate the JSON/TXT for the new audio. transcribe_file
                # writes the sidecars next to the mp3 path it's given.
                try:
                    transcribe.transcribe_file(out_mp3, force=True)
                except Exception as e:
                    log.error("Regenerate JSON for spliced %s failed: %s", hour, e)
                    entry["regen_error"] = str(e)
            # mode == "neither": nothing to do
        except Exception as e:
            log.error("Failed on %s (%s): %s", hour, mode, e)
            entry["error"] = str(e)
            summary["errors"] += 1
        summary[mode] = summary.get(mode, 0) + 1
        entries.append(entry)

    log.info("Audio merge: %s", {k: v for k, v in summary.items() if v})
    return {
        "merged_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "inputs": {"a": str(arch_a.resolve()), "b": str(arch_b.resolve())},
        "splice_enabled": splice,
        "dry_run": dry,
        "summary": summary,
        "hours": entries,
    }


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def _main() -> None:
    parser = argparse.ArgumentParser(
        description="Merge two redundant archive trees into one canonical tree."
    )
    parser.add_argument("--archive-a", type=Path, required=True)
    parser.add_argument("--archive-b", type=Path, required=True)
    parser.add_argument("--archive-out", type=Path, required=True)
    parser.add_argument("--schedule-a", type=Path)
    parser.add_argument("--schedule-b", type=Path)
    parser.add_argument("--schedule-out", type=Path)
    parser.add_argument("--no-splice", action="store_true",
                        help="Disable Phase 2 (cross-fill splicing); winner-takes-all only.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Log decisions without copying or splicing.")
    parser.add_argument("--force-out", action="store_true",
                        help="Allow non-empty output directories.")
    parser.add_argument("--report", type=Path, default=None,
                        help="Explicit path for merge_log.json (default: <archive-out>/merge_log.json).")
    args = parser.parse_args()

    sched_flags = (args.schedule_a, args.schedule_b, args.schedule_out)
    if any(sched_flags) and not all(sched_flags):
        parser.error("--schedule-a, --schedule-b, and --schedule-out must be given together.")

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s  %(levelname)-8s  %(message)s",
                        datefmt="%Y-%m-%d %H:%M:%S")

    report = merge_archives(args.archive_a, args.archive_b, args.archive_out,
                            splice=not args.no_splice,
                            dry=args.dry_run,
                            force_out=args.force_out)
    if all(sched_flags):
        report["schedule"] = merge_schedules(args.schedule_a, args.schedule_b,
                                             args.schedule_out,
                                             dry=args.dry_run,
                                             force_out=args.force_out)

    report_path = args.report or (args.archive_out / "merge_log.json")
    if not args.dry_run:
        report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False),
                               encoding="utf-8")
        log.info("Wrote %s", report_path)


if __name__ == "__main__":
    _main()
