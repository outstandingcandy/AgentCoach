"""Homography and PnP helpers for manual annotations.

The annotator's solve+project path delegates to ``pnlcalib_orig`` — the
faithful port of upstream PnLCalib (``cv2.calibrateCamera`` Zhang +
3-plane reparameterization for crossbars + 18-combo heuristic voting).
``solve_camera`` returns a ``Camera`` populated from the upstream result,
converted from upstream's y-down centered world frame to the project's
y-up centered frame so existing project/visualize callers keep working.
"""

import numpy as np

from goalinsight.annotation.pitch import keypoints as _pk
from goalinsight.field_registration.pnlcalib.camera import Camera
from goalinsight.field_registration.pnlcalib_orig import (
    FramebyFrameCalib,
    PnLCalibIdMap,
)


def line_intersection(
    line1: tuple[tuple[float, float], tuple[float, float]],
    line2: tuple[tuple[float, float], tuple[float, float]],
) -> tuple[float, float] | None:
    """Intersection of two lines, or None if parallel."""
    (x1, y1), (x2, y2) = line1
    (x3, y3), (x4, y4) = line2

    denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    if abs(denom) < 1e-10:
        return None

    t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / denom
    return (x1 + t * (x2 - x1), y1 + t * (y2 - y1))


# ---------------------------------------------------------------------------
# pnlcalib_orig-backed solve + project
# ---------------------------------------------------------------------------

# Minimum point count for upstream's ``cv2.calibrateCamera`` path (Zhang
# multi-plane). Mirrors the project's prior 6-point floor.
MIN_POINTS_FOR_PNLCALIB = 6

# Y-axis flip between upstream's y-down centered world frame and the
# project's y-up centered frame. Conjugating by J swaps the sign of world
# Y while leaving the optical axis untouched, so a pose solved in y-down
# coords becomes a y-up pose via ``R_up = R_dn @ J``, ``t_up = t_dn``.
_J_FLIP_Y = np.diag([1.0, -1.0, 1.0])


def _camera_from_upstream_result(
    result: dict, img_size: tuple[int, int],
) -> Camera:
    """Wrap a ``heuristic_voting`` result as a populated project ``Camera``.

    Converts the upstream y-down centered world frame into the project's
    y-up centered frame by post-multiplying R with ``diag(1, -1, 1)`` and
    recomputing the camera position from the new (R, t).
    """
    cp = result["cam_params"]
    fx = float(cp["x_focal_length"])
    fy = float(cp["y_focal_length"])
    px, py = float(cp["principal_point"][0]), float(cp["principal_point"][1])

    R_dn = np.asarray(cp["rotation_matrix"], dtype=np.float64).reshape(3, 3)
    pos_dn = np.asarray(cp["position_meters"], dtype=np.float64).reshape(3)
    t_dn = -R_dn @ pos_dn

    R_up = R_dn @ _J_FLIP_Y
    t_up = t_dn
    pos_up = -R_up.T @ t_up

    K = np.array([[fx, 0.0, px], [0.0, fy, py], [0.0, 0.0, 1.0]], dtype=np.float64)

    w, h = int(img_size[0]), int(img_size[1])
    cam = Camera(iwidth=w, iheight=h)
    cam.calibration = K
    cam.xfocal_length = fx
    cam.yfocal_length = fy
    cam.principal_point = (px, py)
    cam.rotation = R_up
    cam.position = pos_up
    cam.distortion = np.zeros(5, dtype=np.float64)
    return cam


def _project_world_to_pixel(
    cam: Camera, pt_3d: np.ndarray,
) -> tuple[float, float] | None:
    """Manual projection: K @ R @ (P - C). Returns None for points behind cam.

    Bypasses ``cv2.projectPoints`` because the y-up world frame stored on
    the project ``Camera`` produces a reflection rotation (det = -1) —
    the upstream solver works in y-down, and conjugating by ``diag(1,-1,
    1)`` to swap conventions makes ``R_up = R_dn @ J``. cv2.Rodrigues /
    projectPoints expect a proper rotation, but the linear algebra
    ``K @ R_up @ (P - C_up)`` still produces the correct pixel.
    """
    p = cam.rotation @ (pt_3d - cam.position)
    if p[2] <= 1e-3:
        return None
    x = p[0] / p[2] * cam.xfocal_length + cam.principal_point[0]
    y = p[1] / p[2] * cam.yfocal_length + cam.principal_point[1]
    return float(x), float(y)


def _active_pitch_dims() -> dict:
    """Read the active SoccerPitch as the dict ``FramebyFrameCalib`` expects."""
    pitch = _pk.get_active_pitch()
    return {
        "pitch_length": pitch.PITCH_LENGTH,
        "pitch_width": pitch.PITCH_WIDTH,
        "penalty_area_width": pitch.PENALTY_AREA_WIDTH,
        "penalty_area_length": pitch.PENALTY_AREA_LENGTH,
        "goal_area_width": pitch.GOAL_AREA_WIDTH,
        "goal_area_length": pitch.GOAL_AREA_LENGTH,
        "goal_line_to_penalty_mark": pitch.GOAL_LINE_TO_PENALTY_MARK,
        "center_circle_radius": pitch.CENTER_CIRCLE_RADIUS,
        "goal_height": pitch.GOAL_HEIGHT,
        "goal_length": pitch.GOAL_LENGTH,
    }


def solve_camera(
    pixel_pts: list[tuple[float, float]],
    world_3d_pts: list[tuple[float, float, float]],
    img_size: tuple[int, int],
) -> tuple[Camera | None, float, dict]:
    """Solve a ``Camera`` from 3D-2D point correspondences via pnlcalib_orig.

    Delegates to upstream's ``FramebyFrameCalib.update`` +
    ``heuristic_voting`` (18 mode×ransac combos, ``cv2.calibrateCamera``
    Zhang). World points are matched against the upstream-indexed pitch
    table (``world_xyz_to_upstream_id``) so the upstream solver sees the
    same correspondences in its own keypoint id space.

    Returns ``(camera, mean_error_px, diagnostics)`` on success, ``(None,
    inf, {...})`` on failure (diagnostics still carries ``unresolved`` so
    the caller can tell apart "too few resolvable pts" from "PnP failed").
    Diagnostics keys: ``median_err``, ``max_err``, ``back_facing``,
    ``fx``, ``mode``, ``ransac``, ``calib_plane``, ``unresolved``.
    """
    if len(pixel_pts) < MIN_POINTS_FOR_PNLCALIB:
        return None, float("inf"), {"unresolved": 0}

    pitch_dims = _active_pitch_dims()
    pitch = _pk.get_active_pitch()
    id_map = PnLCalibIdMap(pitch=pitch)

    w, h = int(img_size[0]), int(img_size[1])
    calib = FramebyFrameCalib(
        iwidth=w, iheight=h, denormalize=False, pitch_dims=pitch_dims,
    )

    keypoints_dict: dict[int, dict[str, float]] = {}
    unresolved = 0
    for (px, py), w3d in zip(pixel_pts, world_3d_pts, strict=True):
        up_id = id_map.world_xyz_to_upstream_id(
            (float(w3d[0]), float(w3d[1]), float(w3d[2])),
        )
        if up_id is None:
            unresolved += 1
            continue
        # Multiple project pixels may collapse onto the same upstream id
        # (e.g. derived & raw at the same corner). Keep the highest-prob
        # one — the upstream solver only reads one obs per id anyway.
        prev = keypoints_dict.get(up_id)
        if prev is None:
            keypoints_dict[up_id] = {"x": float(px), "y": float(py), "p": 1.0}

    if len(keypoints_dict) < MIN_POINTS_FOR_PNLCALIB:
        return None, float("inf"), {"unresolved": unresolved}

    calib.update(keypoints_dict, {})
    result = calib.heuristic_voting(refine=True, refine_lines=False, th=5.0)
    if result is None:
        return None, float("inf"), {"unresolved": unresolved}

    cam = _camera_from_upstream_result(result, img_size)

    errs = []
    for (px, py), w3d in zip(pixel_pts, world_3d_pts, strict=True):
        proj = _project_world_to_pixel(cam, np.asarray(w3d, dtype=np.float64))
        if proj is None:
            errs.append(float("inf"))
            continue
        errs.append(float(np.hypot(proj[0] - px, proj[1] - py)))
    err_arr = np.asarray(errs, dtype=np.float64)
    finite = err_arr[np.isfinite(err_arr)]
    median_err = float(np.median(finite)) if finite.size else float("inf")
    max_err = float(err_arr.max())
    mean_err = float(np.mean(err_arr))

    # Camera looks along +Z in *its own* frame, so the world's depth at
    # the camera's principal axis is just t_z. Reuse the upstream
    # ``position_meters`` (y-down): if its z-component (height) is < 0,
    # the camera is below the pitch — same physical check as before.
    pos_dn_z = float(result["cam_params"]["position_meters"][2])

    diagnostics = {
        "median_err": median_err,
        "max_err": max_err,
        "back_facing": bool(pos_dn_z >= 0),
        "fx": float(cam.calibration[0, 0]),
        "mode": result.get("mode"),
        "ransac": result.get("use_ransac"),
        "calib_plane": result.get("calib_plane"),
        "unresolved": unresolved,
    }
    return cam, mean_err, diagnostics


def camera_to_image_to_world(cam: Camera) -> np.ndarray | None:
    """Invert Camera.to_homography() into the image->world H the annotator stores."""
    H_w2i = cam.to_homography()
    try:
        return np.linalg.inv(H_w2i).astype(np.float32)
    except np.linalg.LinAlgError:
        return None


def project_camera_point(
    cam: Camera, pt_3d: tuple[float, float, float],
) -> tuple[float, float] | None:
    """Project a 3D world point through ``cam`` to pixel coordinates."""
    return _project_world_to_pixel(cam, np.asarray(pt_3d, dtype=np.float64))
