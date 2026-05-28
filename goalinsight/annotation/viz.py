"""Minimal pitch-line overlay for annotation previews.

Given an image->world homography, project the pitch lines from
pitch_constants onto a BGR frame. No distortion model — this is just a
sanity-check visualization for the annotator.
"""

import cv2
import numpy as np

from . import pitch_constants


def _project_world_with_depth(
    world_2d: tuple[float, float],
    H_world_to_image: np.ndarray,
) -> tuple[float, float] | None:
    """Project a ground-plane world point and return None if behind horizon.

    H_world_to_image is `inv(H_image_to_world)`. The homogeneous w-coordinate
    (third row) goes through zero on the horizon line; world points beyond the
    horizon have w < 0 and ``cv2.perspectiveTransform`` would mirror them
    (visually they show up "in the sky"). Skip those.
    """
    homo = H_world_to_image @ np.array([world_2d[0], world_2d[1], 1.0],
                                       dtype=np.float64)
    if homo[2] <= 1e-6:
        return None
    return (float(homo[0] / homo[2]), float(homo[1] / homo[2]))


def render_pitch_projection(
    frame_bgr: np.ndarray,
    H: np.ndarray,
    color: tuple[int, int, int] = (0, 255, 255),
    thickness: int = 2,
) -> np.ndarray:
    """Draw projected pitch lines on a BGR frame.

    H is the image->world homography. Endpoints "behind the horizon" — where
    the homogeneous w-coordinate is non-positive — are skipped so the overlay
    doesn't paint phantom lines into the sky.
    """
    out = frame_bgr.copy()
    h, w = out.shape[:2]

    try:
        H_world_to_image = np.linalg.inv(H)
    except np.linalg.LinAlgError:
        return out

    pitch = pitch_constants.get_active_pitch()
    L = pitch.PITCH_LENGTH / 2
    pen = pitch.GOAL_LINE_TO_PENALTY_MARK

    for (wx1, wy1), (wx2, wy2) in pitch_constants.PITCH_LINES.values():
        p1 = _project_world_with_depth((wx1, wy1), H_world_to_image)
        p2 = _project_world_with_depth((wx2, wy2), H_world_to_image)
        if p1 is None or p2 is None:
            continue
        ok, q1, q2 = cv2.clipLine(
            (0, 0, w, h),
            (int(p1[0]), int(p1[1])),
            (int(p2[0]), int(p2[1])),
        )
        if ok:
            cv2.line(out, q1, q2, color, thickness)

    # Landmarks
    for wx, wy, r in [(0.0, 0.0, 5), (-L + pen, 0.0, 4), (L - pen, 0.0, 4)]:
        pt = _project_world_with_depth((wx, wy), H_world_to_image)
        if pt is not None and 0 <= pt[0] < w and 0 <= pt[1] < h:
            cv2.circle(out, (int(pt[0]), int(pt[1])), r, color, -1)

    return out
