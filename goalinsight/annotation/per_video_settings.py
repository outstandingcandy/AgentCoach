"""Per-video sparse overrides for the unified web app.

Each annotated video gets a single
``workspace/annotations/<video_stem>/overrides.yaml`` with whatever
config keys differ from the user-selected base yaml. The user maintains
this file by hand — the web app only reads it.

This is a *sparse* deep-merge layer over the regular config tree.
Pitch dims are referenced by name from ``pitch_types.PITCH_TYPES``
instead of being repeated as 10 floats:

```yaml
# example overrides.yaml for kids_soccer_clip_1250_1310
pitch_type: kids_soccer
field_registration:
  physical:
    camera_position: [-5.557, -21.926, 2.820]
```

``pitch:`` is also accepted (and overrides ``pitch_type`` field-by-field)
for the rare case where the user wants a one-off dim that isn't in the
registry. After resolution the rest of the system sees an ordinary
``pitch:`` block; ``pitch_type`` is consumed and removed.

Annotator behaviour:
- ``load_pitch(...)``  → returns the resolved ``pitch:`` dict so the
  open-video hook can ``set_active_pitch(SoccerPitch(**pitch))``.

Pipeline behaviour:
- ``load(...)``  → returns the resolved yaml dict, deep-merged into the
  base config by ``goalinsight.web.jobs._merge_per_video_overrides``.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import yaml

from . import pitch_types

logger = logging.getLogger(__name__)

OVERRIDES_FILE = "overrides.yaml"


def video_dir(annotations_dir: Path, video_stem: str) -> Path:
    return Path(annotations_dir) / video_stem


def overrides_path(annotations_dir: Path, video_stem: str) -> Path:
    return video_dir(annotations_dir, video_stem) / OVERRIDES_FILE


def load(annotations_dir: Path, video_stem: str) -> dict[str, Any]:
    """Return the parsed overrides yaml as a dict (empty if missing/invalid).

    Resolves ``pitch_type: <name>`` against ``pitch_types.PITCH_TYPES``
    before returning so callers always see a fully expanded ``pitch:``
    block (or no pitch at all). Any inline ``pitch:`` keys override
    individual fields of the named type.
    """
    path = overrides_path(annotations_dir, video_stem)
    if not path.exists():
        return {}
    try:
        data = yaml.safe_load(path.read_text()) or {}
    except (OSError, yaml.YAMLError):
        return {}
    if not isinstance(data, dict):
        return {}
    return _expand_pitch_type(data, video_stem=video_stem)


def load_pitch(annotations_dir: Path, video_stem: str) -> dict[str, float] | None:
    """Return just the resolved ``pitch:`` block, or None if absent."""
    block = load(annotations_dir, video_stem).get("pitch")
    if not isinstance(block, dict):
        return None
    out: dict[str, float] = {}
    for k, v in block.items():
        if isinstance(v, (int, float)):
            out[k] = float(v)
    return out or None


def _expand_pitch_type(data: dict[str, Any], *, video_stem: str) -> dict[str, Any]:
    """Resolve ``pitch_type`` → ``pitch`` in a copy of *data*.

    - ``pitch_type: foo`` alone  →  ``pitch: <PITCH_TYPES['foo']>``
    - ``pitch_type`` + ``pitch`` →  type provides the baseline; inline
      ``pitch:`` keys override field-by-field.
    - Unknown ``pitch_type`` is logged and dropped (the user gets a clear
      log line rather than a hard crash on every video open).
    """
    if "pitch_type" not in data:
        return data
    out = dict(data)
    name = out.pop("pitch_type")
    try:
        resolved = pitch_types.resolve(str(name))
    except KeyError as exc:
        logger.warning("[%s] %s — overrides.yaml ignored for pitch", video_stem, exc)
        return out
    inline = out.get("pitch") if isinstance(out.get("pitch"), dict) else {}
    out["pitch"] = {**resolved, **inline}
    return out
