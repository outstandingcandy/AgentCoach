"""Helper functions over the 57-point HRNet pitch keypoint system.

Keypoint coords live in ``goalinsight.annotation.pitch.keypoints.PITCH_POINTS``
(rebuilt when ``set_active_pitch`` is called). Names are HRNet-style strings
like ``"TL_PITCH_CORNER"``; saved annotation files store both the name and the
HRNet index, with the v3 finetune dataloader keying off the world coordinates.
"""

from .pitch import keypoints as _pk
from .pitch.keypoints import (
    PITCH_POINTS_TO_INTERSECTON,
    INTERSECTON_TO_PITCH_POINTS,
    NOT_ON_PLANE,
)


def get_hrnet_keypoints_2d() -> dict[str, tuple[float, float]]:
    """Return HRNet keypoints as (x, y) world coords, dropping z.

    Reads ``_pk.PITCH_POINTS`` at call time so non-FIFA pitches set via
    ``set_active_pitch`` are reflected.
    """
    return {name: (float(pt[0]), float(pt[1])) for name, pt in _pk.PITCH_POINTS.items()}


def get_hrnet_keypoint_choices() -> list[str]:
    """Dropdown-friendly choices: 'INDEX: NAME'."""
    return [f"{idx}: {name}" for idx, name in INTERSECTON_TO_PITCH_POINTS.items()]


def parse_keypoint_choice(choice: str) -> tuple[int, str]:
    parts = choice.split(": ", 1)
    idx = int(parts[0])
    name = parts[1] if len(parts) > 1 else INTERSECTON_TO_PITCH_POINTS[idx]
    return idx, name


# Cached snap threshold, keyed by id() of the active PITCH_POINTS dict so
# `set_active_pitch` (which rebuilds the dict) implicitly invalidates it.
_THRESH_CACHE: tuple[int, float] | None = None


def _active_threshold() -> float:
    """Half the smallest neighbour gap on the active pitch, capped at 5 m.

    Kids-spec pitches put PA-front and GA-front only ~4.13 m apart, so the
    legacy fixed 5 m radius pulls cross-side intersections in. Sweeping all
    57 keypoint pairs once and halving the minimum non-zero gap keeps every
    real intersection inside its own basin while still rejecting noise.
    """
    global _THRESH_CACHE
    pts = list(_pk.PITCH_POINTS.values())
    key = id(_pk.PITCH_POINTS)
    if _THRESH_CACHE is not None and _THRESH_CACHE[0] == key:
        return _THRESH_CACHE[1]

    min_gap = float("inf")
    for i, a in enumerate(pts):
        ax, ay = float(a[0]), float(a[1])
        for b in pts[i + 1:]:
            bx, by = float(b[0]), float(b[1])
            d = ((ax - bx) ** 2 + (ay - by) ** 2) ** 0.5
            if 1e-3 < d < min_gap:
                min_gap = d

    thr = 5.0 if min_gap == float("inf") else max(0.25, min(5.0, 0.5 * min_gap))
    _THRESH_CACHE = (key, thr)
    return thr


def find_nearest_keypoint(
    wx: float,
    wy: float,
    threshold: float | None = None,
) -> tuple[str | None, int | None]:
    """Find the nearest HRNet keypoint within `threshold` meters.

    When ``threshold`` is None, uses ``_active_threshold()`` — half the
    smallest pairwise gap on the active pitch. Pass an explicit value to
    override (e.g. legacy callers that already verified the candidate).
    """
    if threshold is None:
        threshold = _active_threshold()

    min_dist = float("inf")
    nearest_name = None
    nearest_idx = None

    for idx, name in INTERSECTON_TO_PITCH_POINTS.items():
        if name not in _pk.PITCH_POINTS:
            continue
        pt = _pk.PITCH_POINTS[name]
        kx, ky = float(pt[0]), float(pt[1])
        dist = ((wx - kx) ** 2 + (wy - ky) ** 2) ** 0.5
        if dist < min_dist and dist < threshold:
            min_dist = dist
            nearest_name = name
            nearest_idx = idx

    return nearest_name, nearest_idx


def abbreviate_line_name(name: str) -> str:
    abbrev = {
        "touchline": "TL",
        "goal_line": "GL",
        "center_line": "CL",
        "penalty": "P",
        "goal_area": "GA",
        "top": "T",
        "bottom": "B",
        "left": "L",
        "right": "R",
        "front": "F",
    }
    parts = name.split("_")
    return "".join(abbrev.get(p, p[0].upper()) for p in parts)
