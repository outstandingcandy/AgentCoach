"""Minimal pitch-line overlay for annotation previews.

Given an image->world homography, project the pitch lines from
pitch_constants onto a BGR frame. No distortion model — this is just a
sanity-check visualization for the annotator.
"""

import cv2
import numpy as np

from .homography import project_world_to_image
from .pitch_constants import PITCH_LENGTH, PITCH_LINES

_L = PITCH_LENGTH / 2


def _clip_line_to_frame(
    p1: tuple[int, int],
    p2: tuple[int, int],
    w: int,
    h: int,
) -> tuple[tuple[int, int], tuple[int, int]] | None:
    """Cohen-Sutherland style clip against the frame rect."""
    x1, y1 = p1
    x2, y2 = p2
    INSIDE, LEFT, RIGHT, BOTTOM, TOP = 0, 1, 2, 4, 8

    def code(x: float, y: float) -> int:
        c = INSIDE
        if x < 0:
            c |= LEFT
        elif x >= w:
            c |= RIGHT
        if y < 0:
            c |= TOP
        elif y >= h:
            c |= BOTTOM
        return c

    c1 = code(x1, y1)
    c2 = code(x2, y2)

    while True:
        if not (c1 | c2):
            return (int(x1), int(y1)), (int(x2), int(y2))
        if c1 & c2:
            return None
        out = c1 or c2
        if out & TOP:
            x = x1 + (x2 - x1) * (0 - y1) / (y2 - y1)
            y = 0
        elif out & BOTTOM:
            x = x1 + (x2 - x1) * (h - 1 - y1) / (y2 - y1)
            y = h - 1
        elif out & RIGHT:
            y = y1 + (y2 - y1) * (w - 1 - x1) / (x2 - x1)
            x = w - 1
        else:
            y = y1 + (y2 - y1) * (0 - x1) / (x2 - x1)
            x = 0
        if out == c1:
            x1, y1 = x, y
            c1 = code(x1, y1)
        else:
            x2, y2 = x, y
            c2 = code(x2, y2)


def render_pitch_projection(
    frame_bgr: np.ndarray,
    H: np.ndarray,
    color: tuple[int, int, int] = (0, 255, 255),
    thickness: int = 2,
) -> np.ndarray:
    """Draw projected pitch lines on a BGR frame."""
    out = frame_bgr.copy()
    h, w = out.shape[:2]

    for (wx1, wy1), (wx2, wy2) in PITCH_LINES.values():
        p1 = project_world_to_image((wx1, wy1), H)
        p2 = project_world_to_image((wx2, wy2), H)
        if p1 is None or p2 is None:
            continue
        clipped = _clip_line_to_frame(
            (int(p1[0]), int(p1[1])), (int(p2[0]), int(p2[1])), w, h,
        )
        if clipped:
            cv2.line(out, clipped[0], clipped[1], color, thickness)

    # Landmarks
    for wx, wy, r in [(0.0, 0.0, 5), (-_L + 11, 0.0, 4), (_L - 11, 0.0, 4)]:
        pt = project_world_to_image((wx, wy), H)
        if pt is not None and 0 <= pt[0] < w and 0 <= pt[1] < h:
            cv2.circle(out, (int(pt[0]), int(pt[1])), r, color, -1)

    return out
