#!/usr/bin/env python3
"""
Show-Schedule Archiver

Fetches and archives the station's published show schedule daily, and parses it
into a structured snapshot. The schedule URL and source type come from
config.json.

The default source adapter, `r34ics`, targets the "ICS Calendar" WordPress
plugin (r34ics): the page HTML only contains an empty AJAX container, and the
real week grid is fetched from admin-ajax.php. This module reproduces that AJAX
call, archives the raw artifacts, and parses the week grid. New source types can
be added alongside it (see the `source_type` guard in `fetch_calendar_html` and
the parser registry below).

IMPORTANT — the published schedule is *not authoritative*. Shows are sometimes
changed without the schedule being updated, and the page format has changed
repeatedly over time. Everything here is a best-effort record of *what was
listed*, never a guarantee of what actually aired. Downstream consumers should
treat it as a hint (see `schedule_hint_for`).

Artifacts written per run, under the configured schedule dir:
    YYYY-MM-DD.page.html      raw shows page (the source of the nonce + args)
    YYYY-MM-DD.calendar.html  raw rendered week grid (the actual schedule HTML)
    YYYY-MM-DD.json           parsed snapshot, incl. parse_ok / parse_error

Only the Python standard library is used (urllib + html.parser), so there are no
extra dependencies to install.

Usage:
    python schedule_archive.py            # fetch + archive today's snapshot
    python schedule_archive.py --force    # re-fetch even if today's exists
    python schedule_archive.py --reparse  # re-parse today's saved calendar HTML
"""

import argparse
import html as _html
import json
import logging
import re
import urllib.parse
import urllib.request
from datetime import date, datetime, timezone
from html.parser import HTMLParser
from pathlib import Path

from config import CONFIG

SOURCE_URL   = CONFIG["schedule"]["url"]
SOURCE_TYPE  = CONFIG["schedule"]["source_type"]
SCHEDULE_DIR = Path(CONFIG["paths"]["schedule_dir"])
#   1: week_grid + parse_ok/parse_error
#   2: added parser_version (which registered parser produced the data)
SCHEMA_VERSION = 2
_UA = "Mozilla/5.0 (compatible; radio-schedule-archiver)"

# Calendar day-of-week numbering used by the plugin (Sunday = 0).
DOW_NAMES = {0: "Sunday", 1: "Monday", 2: "Tuesday", 3: "Wednesday",
             4: "Thursday", 5: "Friday", 6: "Saturday"}

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Fetching
# ---------------------------------------------------------------------------

def _get(url: str) -> str:
    req = urllib.request.Request(url, headers={"User-Agent": _UA})
    with urllib.request.urlopen(req, timeout=30) as r:
        return r.read().decode("utf-8", "replace")


def _post(url: str, data: dict) -> str:
    body = urllib.parse.urlencode(data).encode()
    req = urllib.request.Request(
        url, data=body,
        headers={"User-Agent": _UA,
                 "Content-Type": "application/x-www-form-urlencoded"},
    )
    with urllib.request.urlopen(req, timeout=30) as r:
        return r.read().decode("utf-8", "replace")


def _extract_ajax_params(page_html: str) -> dict:
    """
    Pull the bits needed to reproduce the calendar AJAX request out of the
    shows page. Raises ValueError if the page no longer matches the expected
    shape (so the caller can record the failure and keep the raw HTML).
    """
    nonce = re.search(r'"r34ics_nonce":"([^"]+)"', page_html)
    ajaxurl = re.search(r'"ajaxurl":"([^"]+)"', page_html)
    container = re.search(
        r'<div class="r34ics-ajax-container[^"]*"[^>]*'
        r'data-args="([^"]+)"[^>]*data-js-args="([^"]*)"',
        page_html,
    )
    if not (nonce and ajaxurl and container):
        missing = [n for n, m in
                   (("nonce", nonce), ("ajaxurl", ajaxurl), ("container", container))
                   if not m]
        raise ValueError(f"shows page format not recognized (missing: {', '.join(missing)})")
    return {
        "ajaxurl": ajaxurl.group(1).replace(r"\/", "/"),
        "nonce": nonce.group(1),
        "args": container.group(1),
        "js_args": _html.unescape(container.group(2)) or '{"debug":"0"}',
    }


def fetch_calendar_html() -> tuple[str, str]:
    """
    Return (page_html, calendar_html) for the configured source. Raises on
    network/format failure. Only the `r34ics` source type is implemented; add a
    new branch here (and a matching parser) to support another schedule system.
    """
    if SOURCE_TYPE != "r34ics":
        raise NotImplementedError(
            f"schedule source_type {SOURCE_TYPE!r} is not supported "
            f"(only 'r34ics' is implemented)."
        )
    page_html = _get(SOURCE_URL)
    params = _extract_ajax_params(page_html)
    calendar_html = _post(params["ajaxurl"], {
        "action": "r34ics_ajax",
        "r34ics_nonce": params["nonce"],
        "subaction": "display_calendar",
        "args": params["args"],
        "js_args": params["js_args"],
    })
    return page_html, calendar_html


# ---------------------------------------------------------------------------
# Parser registry
# ---------------------------------------------------------------------------
# The shows page has changed format many times and will change again. Each
# distinct format gets its own parser class with a detect()/parse() pair.
#
#   * Parsers are NEVER removed. Old archived HTML must stay re-parseable
#     forever, so every historical format keeps its parser.
#   * To support a new format, write a NEW ScheduleParser subclass and append it
#     to PARSERS. Do not edit existing parsers.
#   * select_parser() picks the right parser for a given HTML blob by detection,
#     so `--reparse-all` regenerates the whole archive correctly: each file is
#     handled by whichever parser recognizes its era.
#
# Parser `version` strings are date-prefixed so the timeline is self-evident.
# ---------------------------------------------------------------------------

def _parse_ampm(text: str) -> str | None:
    """'9:00 am' / '12:30 pm' -> 'HH:MM' (24h). None if not found."""
    m = re.search(r"(\d{1,2}):(\d{2})\s*([ap])m", text.lower().replace("\xa0", " "))
    if not m:
        return None
    h, mn, ap = int(m.group(1)), int(m.group(2)), m.group(3)
    if ap == "p" and h != 12:
        h += 12
    if ap == "a" and h == 12:
        h = 0
    return f"{h:02d}:{mn:02d}"


class _WeekGridParser(HTMLParser):
    """
    Walks the plugin's <td data-dow=N> ... <li class="event tHHMMSS"> structure
    and collects events per day-of-week. Start time comes from the machine
    -readable `tHHMMSS` class; end time and title from the inner spans.
    """

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.events: dict[int, list[dict]] = {}
        self._dow: int | None = None
        self._cur: dict | None = None
        self._cap_title: bool = False
        self._title_done: bool = False
        self._cap_end: bool = False

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        a = dict(attrs)
        cls = (a.get("class") or "").split()
        if tag == "td" and a.get("data-dow") is not None:
            try:
                self._dow = int(a["data-dow"])
            except ValueError:
                self._dow = None
        elif tag == "li" and "event" in cls and self._dow is not None:
            start = None
            for tok in cls:
                if re.fullmatch(r"t\d{6}", tok):
                    start = f"{tok[1:3]}:{tok[3:5]}"
            self._cur = {"start": start, "end": None, "title": "", "status": None}
            self._title_done = False
        elif self._cur is not None and tag == "span":
            if "title" in cls and not self._title_done:
                self._cap_title = True
                status = [c for c in cls if c not in ("title", "has_desc")]
                self._cur["status"] = status[0] if status else None
            elif "end_time" in cls:
                self._cap_end = True
                self._cur["_end_raw"] = ""

    def handle_data(self, data: str) -> None:
        if self._cap_title:
            self._cur["title"] += data
        elif self._cap_end:
            self._cur["_end_raw"] += data

    def handle_endtag(self, tag: str) -> None:
        if tag == "span":
            if self._cap_end:
                self._cap_end = False
            elif self._cap_title:
                self._cap_title = False
                self._title_done = True
        elif tag == "li" and self._cur is not None:
            cur = self._cur
            self._cur = None
            cur["title"] = re.sub(r"\s+", " ", cur["title"].replace("\xa0", " ")).strip()
            cur["end"] = _parse_ampm(cur.pop("_end_raw", "") or "")
            if cur["start"] and cur["title"]:
                self.events.setdefault(self._dow, []).append(cur)


class ScheduleParser:
    """
    Base class for a single schedule-format parser.

    Subclasses set a unique `version`, a human `description`, and implement:
      detect(html) -> bool   True iff this parser recognizes the HTML's format.
      parse(html)  -> dict   {dow_str: [ {start, end, title, status}, ... ]}.
    """
    version: str = "base"
    description: str = ""

    def detect(self, calendar_html: str) -> bool:
        raise NotImplementedError

    def parse(self, calendar_html: str) -> dict:
        raise NotImplementedError


class R34icsWeekGridParser(ScheduleParser):
    """
    ICS Calendar (r34ics) plugin 'week' view, in use as of 2026-05: day columns
    marked `<td data-dow=N>`, each event a `<li class="event tHHMMSS">` with the
    title in `<span class="title ...">` and the end time in `<span class="end_time">`.
    """
    version = "2026-05_r34ics_week_grid"
    description = "r34ics ICS Calendar week grid (data-dow cells, tHHMMSS event classes)"

    def detect(self, calendar_html: str) -> bool:
        return ("data-dow=" in calendar_html
                and re.search(r'class="event[^"]*\bt\d{6}\b', calendar_html) is not None)

    def parse(self, calendar_html: str) -> dict:
        p = _WeekGridParser()
        p.feed(calendar_html)
        grid = {}
        for dow, evs in p.events.items():
            evs.sort(key=lambda e: e["start"])
            grid[str(dow)] = evs
        return grid


# Registration order = the order detectors are tried. Keep each detector
# specific enough that only one parser matches a given format. Append new
# parsers here; never delete old ones.
PARSERS: list[ScheduleParser] = [
    R34icsWeekGridParser(),
]


def select_parser(calendar_html: str) -> ScheduleParser | None:
    """Return the first registered parser that recognizes this HTML, or None.
    A parser whose `detect()` raises is logged (so a buggy detector for a new
    format is debuggable from the logs) and skipped."""
    for parser in PARSERS:
        try:
            if parser.detect(calendar_html):
                return parser
        except Exception as e:
            log.warning("Parser %r detect() raised %r; skipping.",
                        getattr(parser, "version", parser.__class__.__name__), e)
            continue
    return None


def parse_calendar_html(calendar_html: str) -> tuple[dict, str]:
    """
    Parse rendered calendar HTML using whichever registered parser matches.
    Returns (week_grid, parser_version). Raises ValueError if no parser matches
    (i.e. the format changed and a new parser needs to be added).
    """
    parser = select_parser(calendar_html)
    if parser is None:
        raise ValueError("no registered schedule parser matches this HTML "
                         "(format may have changed — add a new ScheduleParser)")
    return parser.parse(calendar_html), parser.version


# ---------------------------------------------------------------------------
# Archiving (daily snapshot)
# ---------------------------------------------------------------------------

def _new_snapshot(fetched_utc: str) -> dict:
    return {
        "schema_version": SCHEMA_VERSION,
        "fetched_utc": fetched_utc,
        "source_url": SOURCE_URL,
        "note": "Published schedule as listed; NOT authoritative — actual "
                "broadcast may differ and the page format changes over time.",
        "parser_version": None,
        "parse_ok": False,
        "parse_error": None,
        "event_count": 0,
        "week_grid": {},
    }


def _parse_into(snapshot: dict, calendar_html: str) -> dict:
    """Run the registry parser and fill the snapshot's parse fields in place."""
    try:
        grid, parser_version = parse_calendar_html(calendar_html)
        snapshot["parser_version"] = parser_version
        snapshot["week_grid"] = grid
        snapshot["event_count"] = sum(len(v) for v in grid.values())
        snapshot["parse_ok"] = snapshot["event_count"] > 0
        if not snapshot["parse_ok"]:
            snapshot["parse_error"] = f"parser {parser_version} produced no events"
    except Exception as e:
        snapshot["parse_error"] = f"parse failed: {e!r}"
    return snapshot


def _write_snapshot(json_path: Path, snapshot: dict) -> None:
    json_path.write_text(json.dumps(snapshot, indent=2, ensure_ascii=False),
                         encoding="utf-8")


def _rotate_to_bak(paths: list[Path]) -> dict[Path, Path]:
    """
    Rename any existing files in `paths` to `*.bak`, overwriting any prior
    `.bak`. Returns a {original: bak} map so a failed re-fetch can be rolled
    back. Files that don't exist are skipped (no entry in the map).
    """
    rotated: dict[Path, Path] = {}
    for p in paths:
        if not p.exists():
            continue
        bak = p.with_name(p.name + ".bak")
        if bak.exists():
            bak.unlink()
        p.replace(bak)
        rotated[p] = bak
    return rotated


def _restore_from_bak(rotated: dict[Path, Path]) -> None:
    """Roll back `_rotate_to_bak` — restore originals from `.bak` files."""
    for original, bak in rotated.items():
        if original.exists():
            original.unlink()
        bak.replace(original)


def archive_snapshot(force: bool = False, reparse: bool = False) -> dict:
    """
    Fetch + archive today's snapshot. Always tries to leave the raw artifacts on
    disk even when parsing fails. Returns the parsed snapshot dict.

    When `force` re-fetches over an existing day, the existing
    `.page.html` / `.calendar.html` / `.json` trio is renamed to `*.bak` first
    so a bad re-fetch can be rolled back instead of silently destroying the
    previous good snapshot.
    """
    SCHEDULE_DIR.mkdir(exist_ok=True)
    today = date.today().isoformat()
    page_path = SCHEDULE_DIR / f"{today}.page.html"
    cal_path  = SCHEDULE_DIR / f"{today}.calendar.html"
    json_path = SCHEDULE_DIR / f"{today}.json"

    if json_path.exists() and not force and not reparse:
        log.info("Snapshot for %s already exists (use --force to refetch).", today)
        return json.loads(json_path.read_text(encoding="utf-8"))

    now = datetime.now(timezone.utc).isoformat(timespec="seconds")

    if reparse and cal_path.exists():
        log.info("Re-parsing saved calendar HTML for %s.", today)
        fetched_utc = now
        if json_path.exists():
            try:
                fetched_utc = json.loads(json_path.read_text(encoding="utf-8")) \
                    .get("fetched_utc", now)
            except Exception:
                pass
        snapshot = _new_snapshot(fetched_utc)
        snapshot["reparsed_utc"] = now
        _parse_into(snapshot, cal_path.read_text(encoding="utf-8"))
    else:
        # Preserve any existing trio before we touch it. If the fetch fails
        # we restore from .bak; on success we leave the .bak behind for a
        # one-deep undo.
        rotated = _rotate_to_bak([page_path, cal_path, json_path]) if force else {}
        snapshot = _new_snapshot(now)
        try:
            page_html, calendar_html = fetch_calendar_html()
            page_path.write_text(page_html, encoding="utf-8")
            cal_path.write_text(calendar_html, encoding="utf-8")
            log.info("Fetched + saved raw page and calendar HTML for %s.", today)
        except Exception as e:
            snapshot["parse_error"] = f"fetch failed: {e!r}"
            log.error("Schedule fetch failed: %s", e)
            if rotated:
                log.warning("Restoring previous snapshot from .bak.")
                _restore_from_bak(rotated)
                # Don't overwrite the restored json with a fetch-failure record.
                return snapshot
            _write_snapshot(json_path, snapshot)
            return snapshot
        _parse_into(snapshot, calendar_html)

    if snapshot["parse_ok"]:
        log.info("Parsed %d events with %s.",
                 snapshot["event_count"], snapshot["parser_version"])
    else:
        log.warning("Parse incomplete (raw HTML kept): %s", snapshot["parse_error"])
    _write_snapshot(json_path, snapshot)
    return snapshot


def reparse_all() -> dict:
    """
    Re-parse every archived raw calendar HTML, regenerating each JSON snapshot
    with whichever registered parser matches that file's format. This is how the
    full history is regenerated after a new parser is added: each day's archive
    is handled by the parser for its era, old days included.
    """
    stats = {"reparsed": 0, "ok": 0, "failed": 0}
    cal_files = sorted(SCHEDULE_DIR.glob("*.calendar.html")) if SCHEDULE_DIR.is_dir() else []
    if not cal_files:
        log.info("No archived calendar HTML to re-parse.")
        return stats

    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    log.info("Re-parsing %d archived calendar file(s)...", len(cal_files))
    for cal in cal_files:
        day = cal.name[: -len(".calendar.html")]
        json_path = SCHEDULE_DIR / f"{day}.json"

        fetched_utc = now
        if json_path.exists():
            try:
                fetched_utc = json.loads(json_path.read_text(encoding="utf-8")) \
                    .get("fetched_utc", now)
            except Exception:
                pass

        snapshot = _new_snapshot(fetched_utc)
        snapshot["reparsed_utc"] = now
        _parse_into(snapshot, cal.read_text(encoding="utf-8"))
        _write_snapshot(json_path, snapshot)

        stats["reparsed"] += 1
        if snapshot["parse_ok"]:
            stats["ok"] += 1
            log.info("  %s  ->  %s  (%d events)",
                     day, snapshot["parser_version"], snapshot["event_count"])
        else:
            stats["failed"] += 1
            log.warning("  %s  ->  unparsed: %s", day, snapshot["parse_error"])

    log.info("Re-parsed %d file(s): %d ok, %d unparsed.",
             stats["reparsed"], stats["ok"], stats["failed"])
    return stats


# ---------------------------------------------------------------------------
# Lookup helpers (used by the transcriber to tag hours)
# ---------------------------------------------------------------------------

def load_snapshot_for(d: date) -> tuple[dict | None, date | None, bool | None]:
    """
    Find the most useful archived snapshot for date `d`: the most recent one on
    or before `d`; if none predates it, the earliest available. Returns
    (snapshot, snapshot_date, snapshot_is_after_d).
    """
    snaps = []
    if SCHEDULE_DIR.is_dir():
        for f in SCHEDULE_DIR.glob("*.json"):
            try:
                snaps.append((datetime.strptime(f.stem, "%Y-%m-%d").date(), f))
            except ValueError:
                continue
    if not snaps:
        return None, None, None
    on_before = [s for s in snaps if s[0] <= d]
    if on_before:
        sd, f = max(on_before, key=lambda s: s[0])
        after = False
    else:
        sd, f = min(snaps, key=lambda s: s[0])
        after = True
    return json.loads(f.read_text(encoding="utf-8")), sd, after


def _to_min(hhmm: str) -> int:
    h, m = hhmm.split(":")
    return int(h) * 60 + int(m)


def _events_overlapping_hour(events: list[dict], hour: int) -> list[dict]:
    ws, we = hour * 60, hour * 60 + 60
    out: list[dict] = []
    for e in events:
        if not e.get("start"):
            continue
        s = _to_min(e["start"])
        en = _to_min(e["end"]) if e.get("end") else s + 60
        if en <= s:
            en = s + 60
        if s < we and en > ws:
            out.append({"start": e["start"], "end": e.get("end"),
                        "title": e.get("title"), "status": e.get("status")})
    out.sort(key=lambda x: x["start"])
    return out


def schedule_hint_for(recording_dt: datetime) -> dict:
    """
    Best-effort lookup of which show(s) were *listed* for the hour a recording
    started. Deliberately named/structured as a hint: never authoritative.
    Always returns a dict; failures are reported in an `error` field.
    """
    hint = {
        "source": SOURCE_URL,
        "note": "Best-effort match from the published weekly schedule; the "
                "actual broadcast may differ and is not guaranteed.",
        "snapshot_date": None,
        "snapshot_after_recording": None,
        "parser_version": None,
        "day_of_week": None,
        "listed_shows": [],
    }
    try:
        snapshot, sd, after = load_snapshot_for(recording_dt.date())
        if snapshot is None:
            hint["error"] = "no schedule snapshot archived yet"
            return hint
        hint["snapshot_date"] = sd.isoformat()
        hint["snapshot_after_recording"] = after
        hint["parser_version"] = snapshot.get("parser_version")
        if not snapshot.get("parse_ok"):
            hint["error"] = (f"snapshot {sd.isoformat()} not parseable: "
                             f"{snapshot.get('parse_error')}")
            return hint
        cal_dow = (recording_dt.weekday() + 1) % 7  # Python Mon=0 -> plugin Sun=0
        hint["day_of_week"] = DOW_NAMES[cal_dow]
        events = snapshot.get("week_grid", {}).get(str(cal_dow), [])
        hint["listed_shows"] = _events_overlapping_hour(events, recording_dt.hour)
    except Exception as e:
        hint["error"] = f"schedule tagging failed: {e!r}"
    return hint


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Archive the station's show schedule.")
    parser.add_argument("--force", action="store_true",
                        help="Re-fetch even if today's snapshot already exists.")
    parser.add_argument("--reparse", action="store_true",
                        help="Re-parse today's already-saved calendar HTML.")
    parser.add_argument("--reparse-all", action="store_true",
                        help="Re-parse EVERY archived calendar HTML, regenerating all "
                             "snapshots with the matching registered parser. Run this "
                             "after adding a new parser for a changed format.")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s  %(levelname)-8s  %(message)s",
                        datefmt="%Y-%m-%d %H:%M:%S")

    if args.reparse_all:
        reparse_all()
    else:
        snap = archive_snapshot(force=args.force, reparse=args.reparse)
        if snap["parse_ok"]:
            log.info("Snapshot OK: %d events via %s.",
                     snap["event_count"], snap["parser_version"])
        else:
            log.warning("Snapshot incomplete: %s", snap["parse_error"])
