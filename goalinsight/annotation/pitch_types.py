"""Named pitch types loaded from ``configs/pitches/*.yaml``.

The user references a type by name in a main pipeline yaml::

    pitch_type: kids_soccer

…or in a per-video overrides file::

    workspace/annotations/<stem>/overrides.yaml
        pitch_type: kids_soccer

The resolver (config_resolver / per_video_settings._expand_pitch_type)
expands that to the full ``pitch:`` block at load time so the rest of
the codebase only sees resolved dims.

Each pitch type is one yaml file under ``configs/pitches/``:

    label:        # short human-readable label (shown in the annotate UI)
    description:  # optional one-line note
    pitch:
      pitch_length: <m>
      pitch_width: <m>
      ...

Add a new type by dropping in a new yaml file — no code change required.
"""

from __future__ import annotations

import functools
import logging
from pathlib import Path

import yaml

logger = logging.getLogger(__name__)

# Pitches live next to the rest of the pipeline configs. Resolved once
# relative to the goalinsight package root so test runs / non-cwd
# invocations still find them.
_PITCHES_DIR = (
    Path(__file__).resolve().parent.parent.parent / "configs" / "pitches"
)


@functools.lru_cache(maxsize=1)
def _load_registry() -> dict[str, dict]:
    """Discover and parse every ``configs/pitches/*.yaml``.

    Returns a dict keyed by file stem (e.g. ``"fifa"``) whose values
    carry the parsed top-level keys. Each entry must have a ``pitch:``
    sub-dict with at minimum ``pitch_length`` and ``pitch_width``. Files
    that fail to parse or miss those keys are logged and dropped — a
    bad single file shouldn't take the whole annotator down.
    """
    out: dict[str, dict] = {}
    if not _PITCHES_DIR.is_dir():
        logger.warning("Pitches dir not found: %s", _PITCHES_DIR)
        return out
    for path in sorted(_PITCHES_DIR.glob("*.yaml")):
        try:
            data = yaml.safe_load(path.read_text()) or {}
        except (OSError, yaml.YAMLError) as exc:
            logger.warning("Skipping malformed pitch yaml %s: %s", path, exc)
            continue
        if not isinstance(data, dict) or not isinstance(data.get("pitch"), dict):
            logger.warning(
                "Pitch yaml %s missing a top-level ``pitch:`` dict; skipping",
                path,
            )
            continue
        pitch = data["pitch"]
        if "pitch_length" not in pitch or "pitch_width" not in pitch:
            logger.warning(
                "Pitch yaml %s missing pitch_length/pitch_width; skipping", path,
            )
            continue
        out[path.stem] = data
    return out


def known_types() -> list[str]:
    """Sorted list of registered pitch type names."""
    return sorted(_load_registry().keys())


def resolve(name: str) -> dict:
    """Return the dims dict for *name*. Raises ``KeyError`` if unknown.

    Numeric values are coerced to ``float``; the optional
    ``penalty_area_shape`` string passes through unchanged. The returned
    dict is a fresh copy so callers may mutate it.
    """
    reg = _load_registry()
    if name not in reg:
        raise KeyError(
            f"unknown pitch_type {name!r}; known: {known_types()}"
        )
    pitch = reg[name]["pitch"]
    out: dict = {}
    for k, v in pitch.items():
        if isinstance(v, (int, float)):
            out[k] = float(v)
        elif k == "penalty_area_shape" and isinstance(v, str):
            out[k] = v
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
    """Drop the cached registry so the next access re-reads the yaml dir.

    Useful in long-running processes (the web app) after a user adds or
    edits a pitch yaml without restarting.
    """
    _load_registry.cache_clear()
