"""Named pitch types loaded from ``configs/pitches.yaml``.

The user references a type by name in a main pipeline yaml::

    pitch_type: kids_soccer

…or in a per-video overrides file::

    workspace/annotations/<stem>/overrides.yaml
        pitch_type: kids_soccer

The resolver (config_resolver / per_video_settings._expand_pitch_type)
expands that to the full ``pitch:`` block at load time so the rest of
the codebase only sees resolved dims.

The library is a single yaml file (mirrors ``configs/camera_profiles.yaml``)::

    profiles:
      fifa:
        label: ...
        pitch_length: 105.0
        pitch_width: 68.0
        ...
      futsal:
        ...

Add a new type by appending to the file — no code change required.
"""

from __future__ import annotations

import functools
import logging
from pathlib import Path

import yaml

logger = logging.getLogger(__name__)

# Pitches live next to the rest of the pipeline configs. Resolved once
# relative to the goalinsight package root so test runs / non-cwd
# invocations still find the file.
_PITCHES_FILE = (
    Path(__file__).resolve().parent.parent.parent / "configs" / "pitches.yaml"
)


@functools.lru_cache(maxsize=1)
def _load_registry() -> dict[str, dict]:
    """Parse ``configs/pitches.yaml`` and return a name→entry map.

    Each entry has the dims (pitch_length, pitch_width, ...) plus
    optional ``label`` / ``description`` strings flattened at the
    profile root — mirrors the shape of camera_profiles.yaml. Profiles
    missing the two required keys (pitch_length / pitch_width) are
    logged and dropped rather than crashing the annotator.
    """
    if not _PITCHES_FILE.is_file():
        logger.warning("Pitches library not found: %s", _PITCHES_FILE)
        return {}
    try:
        data = yaml.safe_load(_PITCHES_FILE.read_text()) or {}
    except (OSError, yaml.YAMLError) as exc:
        logger.warning("Malformed pitches yaml %s: %s", _PITCHES_FILE, exc)
        return {}
    profiles = data.get("profiles") or {}
    if not isinstance(profiles, dict):
        logger.warning(
            "Pitches yaml %s: top-level ``profiles:`` missing or not a dict",
            _PITCHES_FILE,
        )
        return {}
    out: dict[str, dict] = {}
    for name, entry in profiles.items():
        if not isinstance(entry, dict):
            logger.warning("Pitch profile %r is not a dict; skipping", name)
            continue
        if "pitch_length" not in entry or "pitch_width" not in entry:
            logger.warning(
                "Pitch profile %r missing pitch_length/pitch_width; skipping",
                name,
            )
            continue
        out[str(name)] = entry
    return out


def known_types() -> list[str]:
    """Sorted list of registered pitch type names."""
    return sorted(_load_registry().keys())


def resolve(name: str) -> dict:
    """Return the dims dict for *name*. Raises ``KeyError`` if unknown.

    Numeric values are coerced to ``float``; the optional
    ``penalty_area_shape`` string passes through unchanged. The returned
    dict is a fresh copy so callers may mutate it. Non-dim keys
    (``label`` / ``description``) are dropped so downstream code that
    treats the dict as a pitch geometry doesn't trip over them.
    """
    reg = _load_registry()
    if name not in reg:
        raise KeyError(
            f"unknown pitch_type {name!r}; known: {known_types()}"
        )
    entry = reg[name]
    out: dict = {}
    for k, v in entry.items():
        if k in ("label", "description"):
            continue
        if isinstance(v, (int, float)):
            out[k] = float(v)
        elif k == "penalty_area_shape" and isinstance(v, str):
            out[k] = v
        elif k == "valid_keypoint_ids" and isinstance(v, list):
            # Per-pitch whitelist of PnLCalib keypoint ids that are real +
            # correctly scaled on this pitch (see pitches.yaml). Passed
            # through so synthetic-data generation can skip ids that don't
            # exist on non-FIFA pitches. Left as ints.
            out[k] = [int(i) for i in v]
    return out


def describe(name: str) -> dict[str, str]:
    """Return ``{label, description}`` for *name* (synthesizes if missing)."""
    reg = _load_registry()
    entry = reg.get(name, {})
    return {
        "label": str(entry.get("label", name)),
        "description": str(entry.get("description", "")),
    }


def reload() -> None:
    """Drop the cached registry so the next access re-reads the yaml file.

    Useful in long-running processes (the web app) after a user edits
    the pitches yaml without restarting.
    """
    _load_registry.cache_clear()
