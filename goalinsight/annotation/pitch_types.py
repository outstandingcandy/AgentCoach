"""Named pitch types for per-video overrides.

The user references a type by name in
``workspace/annotations/<stem>/overrides.yaml``::

    pitch_type: kids_soccer

The resolver expands that to the full ``pitch:`` block at load time so
the annotator and pipeline never see the alias — they only see resolved
dims, same shape they used before.

Add a new type by appending to ``PITCH_TYPES`` below.
"""

from __future__ import annotations

from typing import Any

from .pitch.geometry import FIFA_DEFAULTS

# Registry. Keys are the names users type into overrides.yaml. Each value
# is a full pitch dict that the SoccerPitch ctor accepts as **kwargs.
PITCH_TYPES: dict[str, dict[str, float]] = {
    "fifa": {k: float(v) for k, v in FIFA_DEFAULTS.items()},

    # Non-FIFA youth pitch — same dims as configs/kids_soccer*.yaml.
    "kids_soccer": {
        "pitch_length": 66.28,
        "pitch_width": 43.15,
        "penalty_area_width": 21.84,
        "penalty_area_length": 9.96,
        "goal_area_width": 11.79,
        "goal_area_length": 4.13,
        "goal_line_to_penalty_mark": 8.0,
        "center_circle_radius": 7.0,
        "goal_height": 2.15,
        "goal_length": 4.44,
    },
}


def known_types() -> list[str]:
    return sorted(PITCH_TYPES.keys())


def resolve(name: str) -> dict[str, float]:
    """Return the dims for *name*. Raises KeyError on unknown type."""
    if name not in PITCH_TYPES:
        raise KeyError(
            f"unknown pitch_type {name!r}; known: {known_types()}"
        )
    return dict(PITCH_TYPES[name])
