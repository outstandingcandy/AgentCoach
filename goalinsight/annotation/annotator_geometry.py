"""Homography solve and projection for the annotator.

Module-level functions taking the annotator state as their first argument.
Reads/writes ``state.H0``, ``state.reprojection_error``, ``state.derived_*``,
``state.auto_*`` directly.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from .homography import (
    MIN_POINTS_FOR_PNLCALIB,
    camera_to_image_to_world,
    line_intersection,
    project_camera_point,
    solve_camera,
    solve_camera_physical,
)
from .keypoint_utils import find_nearest_keypoint
from .pitch import keypoints as _pk
from .pitch.keypoints import (
    INTERSECTON_TO_PITCH_POINTS,
    NOT_ON_PLANE,
)

if TYPE_CHECKING:
    from .annotator_state import AnchorAnnotator


def collect_pnp_points(
    state: "AnchorAnnotator",
) -> tuple[list[tuple[float, float]], list[tuple[float, float, float]]]:
    """Gather (pixel, world_xyz) pairs from manual + accepted-derived points.

    Manual keypoints contribute their full 3D pitch coord. Accepted
    derived points are line intersections, always on the ground plane.

    ``pitch/geometry.py`` stores crossbar keypoints at ``z = +GOAL_HEIGHT``
    (positive-up), which is what the OpenCV ``solvePnP`` family this
    annotator's physical backend goes through expects — no sign flip
    needed here.
    """
    pixel_pts: list[tuple[float, float]] = []
    world_3d_pts: list[tuple[float, float, float]] = []

    for i, name in enumerate(state.keypoint_names):
        pixel_pts.append(state.clicked_points[i])
        pt_3d = _pk.PITCH_POINTS[name]
        world_3d_pts.append(
            (float(pt_3d[0]), float(pt_3d[1]), float(pt_3d[2])),
        )
    for i, (pixel, world, _) in enumerate(state.derived_points):
        accepted = state.derived_accepted[i] if i < len(state.derived_accepted) else False
        if not accepted:
            continue
        pixel_pts.append(pixel)
        world_3d_pts.append((float(world[0]), float(world[1]), 0.0))

    return pixel_pts, world_3d_pts


def compute_line_intersections(state: "AnchorAnnotator") -> None:
    """Recompute derived points from pairwise line intersections."""
    state.derived_points = []
    state.derived_accepted = []
    if len(state.annotated_lines) < 2:
        return

    for i, line1 in enumerate(state.annotated_lines):
        for line2 in state.annotated_lines[i + 1:]:
            pixel_int = line_intersection(
                (line1["pixels"][0], line1["pixels"][1]),
                (line2["pixels"][0], line2["pixels"][1]),
            )
            world_int = line_intersection(line1["world"], line2["world"])
            if not (pixel_int and world_int and state.current_frame is not None):
                continue
            px, py = pixel_int
            h, w = state.current_frame.shape[:2]
            if not (0 <= px < w and 0 <= py < h):
                continue
            wx, wy = world_int
            name, _ = find_nearest_keypoint(wx, wy)
            if name is None:
                continue
            state.derived_points.append((pixel_int, world_int, name))
            state.derived_accepted.append(False)


def compute_auto_projections(state: "AnchorAnnotator") -> None:
    """Auto-project all un-annotated HRNet keypoints using the solved camera."""
    state.auto_projected_points = []
    state.auto_accepted = []
    if state.H0 is None or state.current_frame is None:
        return

    h, w = state.current_frame.shape[:2]
    annotated_names = set(state.keypoint_names)

    pixel_pts, world_3d_pts = collect_pnp_points(state)
    cam, _, _ = solve_camera(pixel_pts, world_3d_pts, (w, h))
    if cam is None:
        return

    for idx, name in INTERSECTON_TO_PITCH_POINTS.items():
        if name in annotated_names or name not in _pk.PITCH_POINTS:
            continue
        pt_3d = _pk.PITCH_POINTS[name]
        if not np.all(np.isfinite(pt_3d)):
            continue

        pixel = project_camera_point(
            cam, (float(pt_3d[0]), float(pt_3d[1]), float(pt_3d[2])),
        )
        if pixel is None:
            continue
        px, py = pixel
        margin = 50
        if not (-margin <= px < w + margin and -margin <= py < h + margin):
            continue
        is_ground = idx not in NOT_ON_PLANE
        state.auto_projected_points.append((
            (px, py),
            (float(pt_3d[0]), float(pt_3d[1])),
            name,
            idx,
            is_ground,
        ))
        state.auto_accepted.append(False)


def compute_homography(state: "AnchorAnnotator") -> str:
    """Solve a Camera (and image->world H) from manual + accepted-derived points.

    Returns a status string for the UI. Mutates ``state.H0``,
    ``state.reprojection_error``, and triggers ``compute_auto_projections``.
    """
    if state.current_frame is None:
        return "No frame loaded."

    pixel_pts, world_3d_pts = collect_pnp_points(state)

    total_points = len(pixel_pts)
    backend = getattr(state, "_solver_backend", "pnlcalib")
    # pnlcalib needs ≥6 (Zhang multi-plane); physical's EPNP RANSAC
    # needs only 4. Don't force users to over-annotate when the
    # solver they picked doesn't require it.
    min_pts = 4 if backend == "physical" else MIN_POINTS_FOR_PNLCALIB
    if total_points < min_pts:
        return (
            f"Need at least {min_pts} points "
            f"({backend} solver requirement). Current: {total_points} — "
            f"add more keypoints or annotate intersecting lines to "
            f"derive corner points."
        )
    unique_world = {
        (round(x, 3), round(y, 3), round(z, 3)) for x, y, z in world_3d_pts
    }
    if len(unique_world) < min_pts:
        return (
            f"Need {min_pts} DIFFERENT world points. "
            f"Current unique: {len(unique_world)}"
        )

    h, w = state.current_frame.shape[:2]
    if backend == "physical":
        cam, mean_error, diag = solve_camera_physical(
            pixel_pts, world_3d_pts, (w, h),
            physical_cfg=state._physical_cfg or {},
            camera_profiles=state._camera_profiles or {},
        )
    else:
        cam, mean_error, diag = solve_camera(pixel_pts, world_3d_pts, (w, h))
    if cam is None:
        return (
            f"PnP RANSAC failed across all focal/distortion candidates "
            f"({total_points} pts). Add more diverse points/lines to "
            f"break the bearing-degenerate basin."
        )
    H = camera_to_image_to_world(cam)
    if H is None:
        return "Homography computation failed."

    # Reject degenerate solutions: solve_camera can return mean_error=inf when
    # the LM optimizer fails to descend (e.g. all keypoints clustered in one
    # half of the frame after a rename/move that introduced a co-linear set).
    # Letting this through would cascade an `inf` through state_dict() and
    # break JSON serialization downstream.
    import math
    if not math.isfinite(float(mean_error)):
        state.H0 = None
        state.reprojection_error = 0.0
        state._solved_cam_position = None
        state.auto_projected_points = []
        state.auto_accepted = []
        return (
            f"PnP solver returned a degenerate solution (mean_error=inf). "
            f"Re-check the recent edit — manual points may now be co-linear "
            f"or covering only one half of the pitch."
        )

    state.reprojection_error = float(mean_error)
    state.H0 = H
    # Store the y-up world position recovered by the solver so the
    # tactical view can draw it next to the configured prior. Both the
    # pnlcalib path (homography._camera_from_upstream_result) and the
    # physical path (solve_camera_physical) end up populating
    # cam.position in y-up coords, so this is uniform across backends.
    try:
        cam_pos = np.asarray(cam.position, dtype=np.float64).reshape(3)
        if np.all(np.isfinite(cam_pos)):
            state._solved_cam_position = (
                float(cam_pos[0]), float(cam_pos[1]), float(cam_pos[2]),
            )
        else:
            state._solved_cam_position = None
    except (AttributeError, ValueError):
        state._solved_cam_position = None
    # Stash full (K, dist, rvec, tvec) so draw_pitch_projection can
    # project the pitch lines through the distortion model instead of
    # the planar H. The planar H is a pinhole-only approximation; with
    # non-trivial dist_coeffs the K + dist + (rvec, tvec) path gives
    # the overlay that actually matches what the camera sees.
    try:
        state._solved_camera = {
            "K": np.asarray(cam.calibration, dtype=np.float64).copy(),
            "rvec": np.asarray(cam._pnp_rvec, dtype=np.float64).reshape(3, 1),
            "tvec": np.asarray(cam._pnp_tvec, dtype=np.float64).reshape(3, 1),
            "dist": np.asarray(cam._pnp_dist, dtype=np.float64).ravel(),
        }
    except (AttributeError, ValueError):
        state._solved_camera = None
    compute_auto_projections(state)

    warn = ""
    if diag.get("back_facing"):
        warn += " ⚠ back-facing pose (tvec_z≤0)"
    if diag.get("max_err", 0) > 200:
        warn += f" ⚠ max_err={diag['max_err']:.0f}px (median={diag['median_err']:.1f})"
    return (
        f"H0 from {total_points} pts. "
        f"mean={state.reprojection_error:.2f}px "
        f"median={diag.get('median_err', 0):.2f}px "
        f"fx={diag.get('fx', 0):.0f}, "
        f"projected: {len(state.auto_projected_points)}{warn}"
    )
