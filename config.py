#!/usr/bin/env python3
"""
Deployment configuration.

Station-specific values — the stream URL, the schedule URL, the stream bitrate,
and the output directories — live in `config.json` so the code itself stays
station-agnostic. Copy `config.example.json` to `config.json` and edit it; the
path can be overridden with the `RADIO_CONFIG` environment variable.

`config.json` is *data*, so the station's domain names appearing there is fine —
the Python modules only reference config keys, never a specific station.
"""

import json
import os
from pathlib import Path

CONFIG_PATH = Path(os.environ.get("RADIO_CONFIG", "config.json"))

# Defaults for optional keys. The two URLs have no usable default and are
# validated as required in load().
_DEFAULTS = {
    "label": "Radio",                                  # display name for logs only
    "stream": {"url": None, "bitrate_kbps": 192},
    "schedule": {"url": None, "source_type": "r34ics"},  # schedule source adapter
    "paths": {"archive_dir": "archive", "schedule_dir": "schedule"},
}


def _merge(base: dict, override: dict) -> dict:
    """Recursively merge `override` into a copy of `base`."""
    out = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _merge(out[key], value)
        else:
            out[key] = value
    return out


def load(path: str | os.PathLike | None = None) -> dict:
    """Load and validate the config, filling defaults. Raises a clear error if the
    file is missing or a required URL is absent."""
    p = Path(path) if path is not None else CONFIG_PATH
    if not p.is_file():
        raise FileNotFoundError(
            f"Config file not found: {p}. Copy config.example.json to config.json "
            f"and set your stream/schedule URLs (or point RADIO_CONFIG at it)."
        )
    cfg = _merge(_DEFAULTS, json.loads(p.read_text(encoding="utf-8")))
    if not cfg["stream"]["url"]:
        raise ValueError(f"{p}: 'stream.url' is required.")
    if not cfg["schedule"]["url"]:
        raise ValueError(f"{p}: 'schedule.url' is required.")
    return cfg


# Loaded once at import; modules read station-specific values from here.
CONFIG = load()
