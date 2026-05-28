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


def find_nearest_keypoint(
    wx: float,
    wy: float,
    threshold: float = 5.0,
) -> tuple[str | None, int | None]:
    """Find the nearest HRNet keypoint within `threshold` meters."""
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
