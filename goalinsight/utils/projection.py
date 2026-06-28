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

from typing import Iterable, Sequence

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


def draw_world_polylines(
    img: np.ndarray,
    polylines: Iterable[Sequence],
    rvec: np.ndarray,
    tvec: np.ndarray,
    K: np.ndarray,
    dist: np.ndarray | None = None,
    color: tuple[int, int, int] = (0, 255, 255),
    thickness: int = 2,
) -> None:
    """Project world-space polylines and draw them on *img* in-place.

    The shared rendering path for every pitch overlay in the project:
    annotate-page projection, field-registration calibration visualisation,
    and tracking minimap all funnel through this so they paint the same
    yellow lines through the same camera model. Centralising means a fix
    to clipping / visibility logic propagates everywhere.

    Args:
        img: BGR image to draw on (mutated).
        polylines: Iterable of polylines. Each polyline is a sequence of
            ``(x, y)`` or ``(x, y, z)`` world points in metres.
            ``(x, y)`` entries are lifted to ``z=0`` (ground plane).
        rvec, tvec, K, dist: Camera extrinsics + intrinsics + Brown-Conrady
            distortion. ``dist`` defaults to zeros.
        color: BGR colour.
        thickness: cv2.line thickness.

    Points whose camera-space z ≤ ``z_eps`` are treated as behind the
    camera and skipped, so a polyline crossing the horizon breaks into
    two visible segments instead of painting phantom sky.
    """
    h_img, w_img = img.shape[:2]
    clip_rect = (0, 0, w_img, h_img)

    for polyline in polylines:
        pts = np.asarray(polyline, dtype=np.float64)
        if pts.ndim != 2 or pts.shape[0] < 2:
            continue
        if pts.shape[1] == 2:
            pts = np.column_stack([pts, np.zeros(len(pts), dtype=np.float64)])
        img_pts, in_front = project_points_with_visibility(
            pts, rvec, tvec, K, dist,
        )
        prev_xy = None
        prev_ok = False
        for i, (xy, vis) in enumerate(zip(img_pts, in_front)):
            cur_xy = (int(xy[0]), int(xy[1])) if vis else None
            if prev_ok and cur_xy is not None and prev_xy is not None:
                ok, q1, q2 = cv2.clipLine(clip_rect, prev_xy, cur_xy)
                if ok:
                    cv2.line(img, q1, q2, color, thickness)
            prev_xy = cur_xy
            prev_ok = cur_xy is not None


def draw_world_landmarks(
    img: np.ndarray,
    world_pts_xyz: Sequence,
    rvec: np.ndarray,
    tvec: np.ndarray,
    K: np.ndarray,
    dist: np.ndarray | None = None,
    color: tuple[int, int, int] = (0, 255, 255),
    radius: int = 4,
) -> None:
    """Project world points and draw filled circles where visible.

    Companion to :func:`draw_world_polylines` for the "single dot"
    landmarks (centre spot, penalty marks, keypoint labels).
    """
    h_img, w_img = img.shape[:2]
    pts = np.asarray(world_pts_xyz, dtype=np.float64)
    if pts.ndim != 2:
        return
    if pts.shape[1] == 2:
        pts = np.column_stack([pts, np.zeros(len(pts), dtype=np.float64)])
    img_pts, in_front = project_points_with_visibility(
        pts, rvec, tvec, K, dist,
    )
    for (px, py), vis in zip(img_pts, in_front):
        if not vis:
            continue
        if 0 <= px < w_img and 0 <= py < h_img:
            cv2.circle(img, (int(px), int(py)), radius, color, -1)
