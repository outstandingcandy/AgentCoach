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


def draw_pitch_projection(frame_rgb: np.ndarray, H: np.ndarray) -> np.ndarray:
    frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
    result = render_pitch_projection(frame_bgr, H, color=(0, 255, 255), thickness=2)
    return cv2.cvtColor(result, cv2.COLOR_BGR2RGB)


