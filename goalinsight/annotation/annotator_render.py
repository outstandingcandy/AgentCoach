"""Rendering and JPEG encoding for the annotator.

Module-level functions taking the annotator state as their first argument.
No class — render concerns are kept separate from state and geometry so each
file can be read without holding the full annotator in mind.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Callable

import cv2
import numpy as np

from .keypoint_utils import abbreviate_line_name
from .pitch.keypoints import PITCH_POINTS_TO_INTERSECTON
from .pitch_diagram import draw_pitch_structure, get_line_color, make_pitch_canvas
from .viz import render_pitch_projection

if TYPE_CHECKING:
    from .annotator_state import AnchorAnnotator


# ---------------------------------------------------------------------------
# Encoding
# ---------------------------------------------------------------------------

def encode_jpeg(img_bgr: np.ndarray, quality: int = 85) -> bytes:
    ok, buf = cv2.imencode(".jpg", img_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    if not ok:
        raise RuntimeError("JPEG encode failed")
    return bytes(buf)


# ---------------------------------------------------------------------------
# Drawing primitives — used by both pixel-space and tactical-space renderers.
# `to_px(x, y) -> (int, int)` maps world coords to image pixels (or just casts
# pixel coords for pixel-space callers).
# ---------------------------------------------------------------------------

ToPx = Callable[[float, float], tuple[int, int]]


def _draw_lines_world(
    img: np.ndarray,
    lines: list[dict],
    to_px: ToPx,
) -> None:
    """Draw annotated lines on the tactical view using their world endpoints."""
    for line_data in lines:
        (wx1, wy1), (wx2, wy2) = line_data["world"]
        color = get_line_color(line_data["name"])
        p1 = to_px(wx1, wy1)
        p2 = to_px(wx2, wy2)
        cv2.line(img, p1, p2, color, 4)
        cv2.circle(img, p1, 5, color, -1)
        cv2.circle(img, p2, 5, color, -1)
        mid_x = (p1[0] + p2[0]) // 2
        mid_y = (p1[1] + p2[1]) // 2
        label = abbreviate_line_name(line_data["name"])
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
        cv2.rectangle(
            img, (mid_x - tw // 2 - 2, mid_y - th - 2),
            (mid_x + tw // 2 + 2, mid_y + 2), (0, 0, 0), -1,
        )
        cv2.putText(img, label, (mid_x - tw // 2, mid_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)


def _draw_lines_pixel(
    img: np.ndarray,
    lines: list[dict],
    selected_idx: int | None,
) -> None:
    """Draw annotated lines on the camera frame using their pixel endpoints."""
    for i, line_data in enumerate(lines):
        p1, p2 = line_data["pixels"]
        sel = (i == selected_idx)
        color = (255, 255, 255) if sel else (0, 255, 255)
        thick = 4 if sel else 2
        cv2.line(img, (int(p1[0]), int(p1[1])), (int(p2[0]), int(p2[1])), color, thick)
        cv2.circle(img, (int(p1[0]), int(p1[1])), 6 if sel else 5, color, -1)
        cv2.circle(img, (int(p2[0]), int(p2[1])), 6 if sel else 5, color, -1)
        mid_x = int((p1[0] + p2[0]) / 2)
        mid_y = int((p1[1] + p2[1]) / 2)
        short_name = abbreviate_line_name(line_data["name"])
        cv2.putText(img, f"L{i+1}:{short_name}", (mid_x, mid_y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)


# ---------------------------------------------------------------------------
# Tactical (world-coord) view
# ---------------------------------------------------------------------------

def _draw_cam_marker(
    img: np.ndarray,
    to_px,
    cam_pos: tuple,
    color: tuple[int, int, int],
    label_prefix: str,
    center_px: tuple[int, int],
    legend_slot: int = 0,
) -> None:
    """Draw a single camera marker on the tactical canvas.

    Plots a filled triangle (pointing toward the pitch centre) + a small
    crosshair dot at ``cam_pos`` (clipped to the canvas if it's outside)
    and writes ``"<prefix> (x, y, z)"`` into a fixed top-right legend
    column. ``legend_slot`` stacks subsequent labels (0 = top row).

    Coordinates of ``cam_pos`` are world-meters in the y-up frame; the
    z component is shown in the label but not used to position the
    marker. ``color`` is BGR.
    """
    cx, cy = float(cam_pos[0]), float(cam_pos[1])
    cz = float(cam_pos[2]) if len(cam_pos) >= 3 else None

    h, w = img.shape[:2]
    tx, ty = to_px(cx, cy)
    clipped = tx < 0 or tx >= w or ty < 0 or ty >= h
    tx_c = max(8, min(w - 8, tx))
    ty_c = max(8, min(h - 8, ty))

    cx_px, cy_px = center_px
    vx, vy = cx_px - tx_c, cy_px - ty_c
    vlen = max((vx * vx + vy * vy) ** 0.5, 1e-6)
    ux, uy = vx / vlen, vy / vlen
    perp_x, perp_y = -uy, ux

    tip = (int(tx_c + ux * 14), int(ty_c + uy * 14))
    base_l = (int(tx_c - ux * 4 + perp_x * 8),
              int(ty_c - uy * 4 + perp_y * 8))
    base_r = (int(tx_c - ux * 4 - perp_x * 8),
              int(ty_c - uy * 4 - perp_y * 8))
    tri = np.array([tip, base_l, base_r], dtype=np.int32)
    cv2.fillPoly(img, [tri], color)
    cv2.polylines(img, [tri], True, (0, 0, 0), 1)
    cv2.circle(img, (tx_c, ty_c), 3, (0, 0, 0), -1)
    cv2.circle(img, (tx_c, ty_c), 2, color, -1)

    # Fixed legend column in the top-right: each marker gets its own row,
    # so prior + solved can both be read at a glance even when the
    # triangles overlap or are clipped to the same edge.
    label = (
        f"{label_prefix} ({cx:.1f}, {cy:.1f}, {cz:.1f})"
        if cz is not None
        else f"{label_prefix} ({cx:.1f}, {cy:.1f})"
    )
    if clipped:
        label += " ↓"
    font_scale = 0.42
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)
    row_h = th + 8
    lx = w - tw - 12
    ly = 18 + legend_slot * row_h
    # Color chip + label background.
    cv2.rectangle(img, (lx - 18, ly - th - 3), (lx + tw + 3, ly + 4),
                  (0, 0, 0), -1)
    cv2.rectangle(img, (lx - 14, ly - th + 1), (lx - 6, ly - 1),
                  color, -1)
    cv2.putText(img, label, (lx, ly),
                cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, 1)


def render_tactical_view(state: "AnchorAnnotator") -> np.ndarray:
    """Render a tactical-style view of all annotations (y-up world)."""
    scale = 7  # match create_pitch_diagram / create_lines_diagram for visual parity
    img, to_px, _width, height = make_pitch_canvas(scale=scale, margin=50)
    draw_pitch_structure(img, to_px, scale=scale, landmark_radius=4)

    # User annotations.
    _draw_lines_world(img, state.annotated_lines, to_px)

    for i, _px in enumerate(state.clicked_points):
        if i < len(state.world_points):
            wx, wy = state.world_points[i]
            tx, ty = to_px(wx, wy)
            cv2.circle(img, (tx, ty), 10, (0, 0, 255), -1)
            cv2.circle(img, (tx, ty), 10, (255, 255, 255), 2)
            if i < len(state.keypoint_names):
                hrnet_idx = PITCH_POINTS_TO_INTERSECTON.get(state.keypoint_names[i], -1)
                label = f"{hrnet_idx}"
            else:
                label = f"P{i+1}"
            cv2.putText(img, label, (tx + 12, ty + 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    for i, (_pixel, world, _name) in enumerate(state.derived_points):
        tx, ty = to_px(world[0], world[1])
        accepted = state.derived_accepted[i] if i < len(state.derived_accepted) else False
        fill = -1 if accepted else 2
        cv2.circle(img, (tx, ty), 8, (255, 0, 255), fill)
        cv2.circle(img, (tx, ty), 8, (255, 255, 255), 2)
        cv2.putText(img, f"D{i+1}", (tx + 10, ty + 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 255), 1)

    for i, (_pixel, world, _name, _hrnet_idx, is_ground) in enumerate(
        state.auto_projected_points,
    ):
        tx, ty = to_px(world[0], world[1])
        color = (0, 128, 255) if is_ground else (180, 0, 255)
        accepted = state.auto_accepted[i] if i < len(state.auto_accepted) else False
        fill = -1 if accepted else 1
        cv2.circle(img, (tx, ty), 5, color, fill)
        cv2.circle(img, (tx, ty), 5, (255, 255, 255), 1)

    # Camera markers — yellow triangle = configured prior from the
    # per-video yaml; green triangle = position the solver actually
    # converged to (only present after Compute). Comparing the two is
    # the fastest way to see whether the LM honoured your prior or
    # drifted away (e.g. low position_weight + wide bounds = drift).
    h_img, w_img = img.shape[:2]
    cx_px, cy_px = to_px(0.0, 0.0)
    label_offset_y = 0  # tracks already-placed labels so they don't overlap

    cam_pos_prior = None
    phys_cfg = getattr(state, "_physical_cfg", None)
    if isinstance(phys_cfg, dict):
        cam_pos_prior = phys_cfg.get("camera_position")
    solved = getattr(state, "_solved_cam_position", None)

    # Both markers go to the legend column at the bottom-right so they
    # don't fight with the prior-marker label and so users can read both
    # values even when prior and solved end up close to each other (or
    # both clipped to the same canvas edge).
    if cam_pos_prior and len(cam_pos_prior) >= 2:
        _draw_cam_marker(
            img, to_px, cam_pos_prior, (0, 220, 220),
            label_prefix="prior", center_px=(cx_px, cy_px),
            legend_slot=0,
        )
    if solved is not None and len(solved) >= 2:
        _draw_cam_marker(
            img, to_px, solved, (0, 255, 80),
            label_prefix="solved", center_px=(cx_px, cy_px),
            legend_slot=1,
        )

    total = len(state.keypoint_names) + len(state.derived_points)
    status = (
        f"Points: {total} (manual: {len(state.keypoint_names)}, "
        f"derived: {len(state.derived_points)})"
    )
    cv2.putText(img, status, (10, height - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    if state.H0 is not None:
        cv2.putText(img, f"H0 computed (error: {state.reprojection_error:.2f}m)",
                    (10, height - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    return img


# ---------------------------------------------------------------------------
# Camera-frame overlay
# ---------------------------------------------------------------------------

def visualize_annotations(state: "AnchorAnnotator", frame_rgb: np.ndarray) -> np.ndarray:
    """Overlay all annotations on a single video frame (RGB in/out)."""
    vis = frame_rgb.copy()

    _draw_lines_pixel(vis, state.annotated_lines, state.selected_line_idx)

    for i, (px, py) in enumerate(state.line_clicks):
        cv2.circle(vis, (int(px), int(py)), 6, (255, 165, 0), -1)
        cv2.circle(vis, (int(px), int(py)), 8, (255, 255, 255), 2)
        cv2.putText(vis, f"L-{i+1}", (int(px) + 10, int(py)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 165, 0), 1)

    for i, (pixel, _world, _name) in enumerate(state.derived_points):
        accepted = state.derived_accepted[i] if i < len(state.derived_accepted) else False
        if not accepted:
            continue
        px, py = pixel
        cv2.circle(vis, (int(px), int(py)), 8, (255, 0, 255), -1)
        cv2.circle(vis, (int(px), int(py)), 10, (255, 255, 255), 2)
        cv2.putText(vis, f"D{i+1}", (int(px) + 12, int(py) + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)

    for i, (pixel, _world, _name, hrnet_idx, is_ground) in enumerate(
        state.auto_projected_points,
    ):
        accepted = state.auto_accepted[i] if i < len(state.auto_accepted) else False
        if not accepted:
            continue
        px, py = pixel
        color = (0, 128, 255) if is_ground else (180, 0, 255)
        cv2.circle(vis, (int(px), int(py)), 6, color, -1)
        cv2.circle(vis, (int(px), int(py)), 8, (255, 255, 255), 2)
        label = f"[{hrnet_idx}]" if is_ground else f"[{hrnet_idx}]T"
        cv2.putText(vis, label, (int(px) + 10, int(py) + 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 2)
        cv2.putText(vis, label, (int(px) + 10, int(py) + 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

    for i, (px, py) in enumerate(state.clicked_points):
        if i < len(state.keypoint_names):
            color = (0, 255, 0)
            kp_name = state.keypoint_names[i]
            hrnet_idx = PITCH_POINTS_TO_INTERSECTON.get(kp_name, -1)
            label = f"[{hrnet_idx}] {kp_name}"
        else:
            color = (255, 255, 0)
            label = f"P{i+1}: [click Add]"

        sel = (i == state.selected_manual_idx)
        radius = 11 if sel else 8
        cv2.circle(vis, (int(px), int(py)), radius, color, -1)
        cv2.circle(vis, (int(px), int(py)), radius + 2, (255, 255, 255), 3 if sel else 2)
        cv2.putText(vis, label, (int(px) + 15, int(py) + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        cv2.putText(vis, label, (int(px) + 15, int(py) + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    # Per-keypoint reprojection markers — shown only when a solve is on
    # file. For each manual annotation we run its world coord through
    # the same (K, dist, rvec, tvec) the PnP solver landed on, then
    # draw a small open circle at the projected pixel + a thin line to
    # the user's click. The visible gap between the two is the
    # per-point reprojection residual at solve time. Drawn LAST so the
    # markers sit on top of the green manual circles — otherwise a
    # near-zero residual would hide the cyan ring inside the green
    # blob and the user couldn't tell the solver fit each point.
    # Crossbar keypoints carry the legacy z = -GOAL_HEIGHT convention
    # from geometry.py; we mirror that here so the marker lands on the
    # same physical post-top the solver fit.
    cam = getattr(state, "_solved_camera", None)
    if cam is not None:
        from .pitch import keypoints as _pk
        K = np.asarray(cam["K"], dtype=np.float64)
        dist = np.asarray(cam["dist"], dtype=np.float64).ravel()
        rvec = np.asarray(cam["rvec"], dtype=np.float64).reshape(3, 1)
        tvec = np.asarray(cam["tvec"], dtype=np.float64).reshape(3, 1)
        for i, (px, py) in enumerate(state.clicked_points):
            if i >= len(state.keypoint_names):
                continue
            name = state.keypoint_names[i]
            pt_3d = _pk.PITCH_POINTS.get(name)
            if pt_3d is None:
                continue
            wx, wy = float(pt_3d[0]), float(pt_3d[1])
            # Flip z sign — see collect_pnp_points for the rationale.
            wz = -float(pt_3d[2]) if len(pt_3d) >= 3 else 0.0
            world = np.array([[wx, wy, wz]], dtype=np.float64)
            try:
                proj, _ = cv2.projectPoints(world, rvec, tvec, K, dist)
            except cv2.error:
                continue
            qx, qy = proj.reshape(2)
            if not (np.isfinite(qx) and np.isfinite(qy)):
                continue
            # Black outline first (so the cyan reads against bright
            # green / yellow / grass), then the cyan ring, then a thin
            # white connector to the click position. The connector
            # length IS the per-point residual; users can eyeball
            # which annotation the solver disagreed with most.
            qi = (int(qx), int(qy))
            cv2.line(vis, (int(px), int(py)), qi, (255, 255, 255), 1)
            cv2.circle(vis, qi, 7, (0, 0, 0), 3)         # black halo
            cv2.circle(vis, qi, 7, (0, 255, 255), 2)     # cyan ring (RGB)
            cv2.circle(vis, qi, 1, (0, 255, 255), -1)    # tiny dot at exact pixel

    confirmed = len(state.keypoint_names)
    derived = len(state.derived_points)
    total_pts = confirmed + derived
    mode_str = f"[{state.annotation_mode.upper()} mode]"
    info_text = (
        f"Frame {state.current_frame_idx} | "
        f"Points: {confirmed} manual + {derived} derived = {total_pts} | {mode_str}"
    )
    cv2.putText(vis, info_text, (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    return vis


def draw_pitch_projection(
    frame_rgb: np.ndarray,
    H: np.ndarray,
    cam: dict | None = None,
) -> np.ndarray:
    """Overlay the yellow pitch lines onto a frame.

    When *cam* (``{K, dist, rvec, tvec}`` from the most recent PnP solve)
    is provided, projection runs through ``cv2.projectPoints`` with the
    full distortion model — this matches what the camera actually
    captured on heavy-distortion lenses (k1 > 0.3-ish). Falls back to
    the planar H path (image↔world homography, pinhole-only) when
    ``cam`` is None, so callers without a stashed solve still work.
    """
    frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
    result = render_pitch_projection(
        frame_bgr, H, color=(0, 255, 255), thickness=2, cam=cam,
    )
    return cv2.cvtColor(result, cv2.COLOR_BGR2RGB)


