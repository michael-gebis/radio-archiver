#!/usr/bin/env python3
"""
Delete `.mp3` files that have been positively identified as off-air carrier
noise / dead air, while preserving the `.json` and `.txt` sidecars as
tombstones.

The criterion comes from `quality.is_off_air()` — it is *intentionally
narrow*. A file is only deleted when:
  - `quality.is_off_air == true` in its JSON sidecar, AND
  - the matching `.mp3` is on disk.

The sidecars are NOT deleted — they remain searchable and document what
was recorded (peak / mean volume, silence breakdown, schedule_hint), and
they prevent the transcriber from re-processing the (now-missing) `.mp3`
on the next `--watch` poll. After deletion:
  - `quality.mp3_deleted_utc` is added (ISO date) so the audit trail is
    in the sidecar.
  - A line is appended to the `.txt` noting the deletion.

Default behavior is **dry-run**. Pass `--apply` to actually delete.

Usage:
    python purge_silent.py                  # dry-run, scan whole archive
    python purge_silent.py --apply          # actually delete
    python purge_silent.py FILE.json [...]  # check specific sidecars
"""

import argparse
import json
import logging
from datetime import datetime, timezone
from pathlib import Path

from config import CONFIG

ARCHIVE_DIR = Path(CONFIG["paths"]["archive_dir"])

log = logging.getLogger(__name__)


def _candidate_sidecars(args_files: list[str]) -> list[Path]:
    if args_files:
        return [Path(f) for f in args_files]
    return sorted(ARCHIVE_DIR.rglob("*.json"))


def _append_deletion_note(txt_path: Path, today: str) -> None:
    if not txt_path.exists():
        return
    note = f"\n# off-air detected; mp3 deleted {today} UTC\n"
    txt_path.write_text(txt_path.read_text(encoding="utf-8") + note,
                        encoding="utf-8")


def purge(apply: bool, files: list[str]) -> dict:
    today = datetime.now(timezone.utc).date().isoformat()
    sidecars = _candidate_sidecars(files)
    if not sidecars:
        log.info("No JSON sidecars found.")
        return {"matched": 0, "would_delete": 0, "deleted": 0, "bytes_freed": 0}

    stats = {"matched": 0, "would_delete": 0, "deleted": 0, "bytes_freed": 0}
    for jp in sidecars:
        try:
            d = json.loads(jp.read_text(encoding="utf-8"))
        except Exception as e:
            log.warning("Bad sidecar %s: %s", jp.name, e)
            continue
        q = d.get("quality") or {}
        if not q.get("is_off_air"):
            continue
        stats["matched"] += 1
        mp3 = jp.with_suffix(".mp3")
        if not mp3.exists():
            # Already tombstoned by a prior run.
            continue
        size = mp3.stat().st_size
        verb = "DELETING" if apply else "WOULD DELETE"
        log.info("%s %s (%.1f MB, peak %s dB)",
                 verb, mp3.name, size / 1e6, q.get("max_volume_db"))
        if not apply:
            stats["would_delete"] += 1
            stats["bytes_freed"] += size
            continue
        mp3.unlink()
        q["mp3_deleted_utc"] = today
        d["quality"] = q
        jp.write_text(json.dumps(d, indent=2, ensure_ascii=False),
                      encoding="utf-8")
        _append_deletion_note(jp.with_suffix(".txt"), today)
        stats["deleted"] += 1
        stats["bytes_freed"] += size
    return stats


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Delete off-air MP3s, keep sidecars.")
    ap.add_argument("files", nargs="*", help="Specific .json sidecars to check.")
    ap.add_argument("--apply", action="store_true",
                    help="Actually delete. Default is dry-run.")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s  %(levelname)-8s  %(message)s",
                        datefmt="%Y-%m-%d %H:%M:%S")
    stats = purge(apply=args.apply, files=args.files)
    suffix = "" if args.apply else " (dry-run; pass --apply to delete)"
    if args.apply:
        log.info("Matched %d, deleted %d, freed %.1f MB.%s",
                 stats["matched"], stats["deleted"],
                 stats["bytes_freed"] / 1e6, suffix)
    else:
        log.info("Matched %d, would delete %d, would free %.1f MB.%s",
                 stats["matched"], stats["would_delete"],
                 stats["bytes_freed"] / 1e6, suffix)
