"""Camera projection helpers.

Every field-registration backend (PnLCalib, BroadTrack, physical, NBJW)
plus several downstream consumers (event detectors, ball trajectory,
top-down viz) call ``cv2.projectPoints`` and rebuild the same
``(N, 1, 3)`` / ``(3, 1)`` reshape boilerplate around it. Centralising
that lets calibrators focus on the actual residual / visualisation
logic.

These wrappers are intentionally thin — they don't validate inputs
or hide errors. Hot-loop callers (LM cost functions) pay only one
extra ``asarray`` + two reshapes per invocation.
"""

from __future__ import annotations

import cv2
import numpy as np


def project_points_2d(
    world_pts: np.ndarray,
    rvec: np.ndarray,
    tvec: np.ndarray,
    K: np.ndarray,
    dist: np.ndarray | None = None,
) -> np.ndarray:
    """Project ``world_pts`` to image pixels with the full camera model.

    Args:
        world_pts: ``(N, 3)`` or ``(N, 2)`` array of world points. ``(N, 2)``
            inputs are lifted to ``z=0`` (ground plane).
        rvec: rotation vector, any 3-element shape.
        tvec: translation vector, any 3-element shape.
        K: ``(3, 3)`` camera matrix.
        dist: ``(5,)`` distortion coefficients. ``None`` → zeros.

    Returns:
        ``(N, 2)`` projected pixel coordinates as ``float64``.
    """
    pts = np.asarray(world_pts, dtype=np.float64)
    if pts.ndim == 2 and pts.shape[1] == 2:
        pts = np.column_stack([pts, np.zeros(len(pts), dtype=np.float64)])
    pts = pts.reshape(-1, 1, 3)
    rv = np.asarray(rvec, dtype=np.float64).reshape(3, 1)
    tv = np.asarray(tvec, dtype=np.float64).reshape(3, 1)
    if dist is None:
        dist = np.zeros(5, dtype=np.float64)
    img, _ = cv2.projectPoints(pts, rv, tv, K, dist)
    return img.reshape(-1, 2)


def project_points_with_visibility(
    world_pts: np.ndarray,
    rvec: np.ndarray,
    tvec: np.ndarray,
    K: np.ndarray,
    dist: np.ndarray | None = None,
    *,
    z_eps: float = 0.1,
) -> tuple[np.ndarray, np.ndarray]:
    """Project + flag points that are behind / on the camera plane.

    A point is considered "in front" iff its camera-space Z exceeds
    ``z_eps``. Image coordinates for behind-camera points are returned
    unchanged (whatever ``cv2.projectPoints`` produced) — callers must
    consult the mask before consuming them.

    Returns:
        ``(img_pts (N, 2), in_front (N,) bool)``.
    """
    pts = np.asarray(world_pts, dtype=np.float64)
    if pts.ndim == 2 and pts.shape[1] == 2:
        pts = np.column_stack([pts, np.zeros(len(pts), dtype=np.float64)])
    pts_3d = pts.reshape(-1, 3)
    img_pts = project_points_2d(pts_3d, rvec, tvec, K, dist)
    R, _ = cv2.Rodrigues(np.asarray(rvec, dtype=np.float64).reshape(3, 1))
    cam_pts = (R @ pts_3d.T).T + np.asarray(tvec, dtype=np.float64).flatten()
    in_front = cam_pts[:, 2] > z_eps
    return img_pts, in_front
