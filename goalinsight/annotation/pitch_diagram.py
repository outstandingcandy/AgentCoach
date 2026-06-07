"""Pitch diagram reference view showing all 57 HRNet keypoints.

World convention is y-up (top = +W/2); the rendering flips y so that points
with y > 0 appear at the top of the image.
"""

import math
from typing import Callable

import cv2
import numpy as np

from . import pitch_constants
from .keypoint_utils import abbreviate_line_name
from .pitch import keypoints as _pk
from .pitch.keypoints import (
    INTERSECTON_TO_PITCH_POINTS,
    NOT_ON_PLANE,
    pitch_name_to_pnlcalib_id,
)

ToPx = Callable[[float, float], tuple[int, int]]


def make_pitch_canvas(
    scale: int,
    margin: int,
    bg: tuple[int, int, int] = (34, 139, 34),
) -> tuple[np.ndarray, ToPx, int, int]:
    """Allocate a canvas sized to the active pitch + margin and return the y-up→pixel map.

    Returns ``(img, to_px, width, height)``. ``to_px(world_x, world_y)`` flips
    ``y`` around the pitch center so y-up world coords land y-down on screen.
    """
    pitch = pitch_constants.get_active_pitch()
    L = pitch.PITCH_LENGTH / 2.0
    W = pitch.PITCH_WIDTH / 2.0
    width = int(pitch.PITCH_LENGTH * scale + 2 * margin)
    height = int(pitch.PITCH_WIDTH * scale + 2 * margin)

    img = np.zeros((height, width, 3), dtype=np.uint8)
    img[:] = bg

    def to_px(x: float, y: float) -> tuple[int, int]:
        return (int((x + L) * scale + margin), int((W - y) * scale + margin))

    return img, to_px, width, height


def draw_pitch_structure(
    img: np.ndarray,
    to_px: ToPx,
    scale: int,
    color: tuple[int, int, int] = (255, 255, 255),
    thickness: int = 2,
    *,
    draw_landmarks: bool = True,
    landmark_radius: int = 4,
    draw_arcs: bool = True,
) -> None:
    """Draw outline + halfway line + circles + penalty/goal areas + (optionally) arcs.

    Geometry comes from ``pitch_constants.get_active_pitch()``. Used by all
    schematic pitch renderers (``render_tactical_view``, ``create_pitch_diagram``,
    ``create_lines_diagram``) so they stay visually consistent.
    """
    pitch = pitch_constants.get_active_pitch()
    L, W = pitch.PITCH_LENGTH / 2.0, pitch.PITCH_WIDTH / 2.0
    ccr = pitch.CENTER_CIRCLE_RADIUS
    pen = pitch.GOAL_LINE_TO_PENALTY_MARK
    pa_w, pa_d = pitch.PENALTY_AREA_WIDTH / 2.0, pitch.PENALTY_AREA_LENGTH
    ga_w, ga_d = pitch.GOAL_AREA_WIDTH / 2.0, pitch.GOAL_AREA_LENGTH

    pts = [to_px(-L, W), to_px(L, W), to_px(L, -W), to_px(-L, -W)]
    for i in range(4):
        cv2.line(img, pts[i], pts[(i + 1) % 4], color, thickness)

    cv2.line(img, to_px(0, W), to_px(0, -W), color, thickness)
    cv2.circle(img, to_px(0, 0), int(ccr * scale), color, thickness)

    cv2.rectangle(img, to_px(-L, pa_w), to_px(-L + pa_d, -pa_w), color, thickness)
    cv2.rectangle(img, to_px(L - pa_d, pa_w), to_px(L, -pa_w), color, thickness)
    cv2.rectangle(img, to_px(-L, ga_w), to_px(-L + ga_d, -ga_w), color, thickness)
    cv2.rectangle(img, to_px(L - ga_d, ga_w), to_px(L, -ga_w), color, thickness)

    if draw_landmarks:
        cv2.circle(img, to_px(0, 0), landmark_radius, color, -1)
        cv2.circle(img, to_px(-L + pen, 0), landmark_radius, color, -1)
        cv2.circle(img, to_px(L - pen, 0), landmark_radius, color, -1)

    if draw_arcs:
        # Penalty-arc visible-segment half-angle, from the geometry of the
        # circle (centered on the penalty mark) intersecting the PA-front
        # line. cos(theta) = (pa_d - pen) / ccr. Falls back to a full circle
        # if the front line is inside the circle (non-FIFA pitches where
        # ccr > pa_d - pen far enough that the arc wraps), and skips
        # entirely if the front line never reaches the circle.
        r = int(ccr * scale)
        dx = pa_d - pen
        ratio = dx / ccr if ccr > 0 else 1.0
        if -1.0 < ratio < 1.0:
            half = math.degrees(math.acos(ratio))
            # Left arc opens toward +x; right arc opens toward -x. The y-up
            # canvas flips angles, so the left arc is drawn at [-half, +half]
            # and the right at [180-half, 180+half].
            cv2.ellipse(img, to_px(-L + pen, 0), (r, r), 0,
                        -half, half, color, thickness)
            cv2.ellipse(img, to_px(L - pen, 0), (r, r), 0,
                        180 - half, 180 + half, color, thickness)
        elif ratio <= -1.0:
            # PA-front farther than ccr from goal — arc is entirely inside.
            cv2.ellipse(img, to_px(-L + pen, 0), (r, r), 0, 0, 360, color, thickness)
            cv2.ellipse(img, to_px(L - pen, 0), (r, r), 0, 0, 360, color, thickness)
        # ratio >= 1.0 → arc never crosses front line; skip.


def _draw_line_label(
    img: np.ndarray,
    p1: tuple[int, int],
    p2: tuple[int, int],
    text: str,
    color: tuple[int, int, int],
    font_scale: float = 0.35,
    thickness: int = 1,
    offset: int = 8,
    bg_pad: int = 2,
) -> None:
    """Draw ``text`` near the midpoint of segment ``p1``-``p2``, perpendicular-offset.

    Used by both ``create_pitch_diagram`` (offset=8) and ``create_lines_diagram``
    (offset=12).
    """
    mid_x = (p1[0] + p2[0]) // 2
    mid_y = (p1[1] + p2[1]) // 2
    dx, dy = p2[0] - p1[0], p2[1] - p1[1]
    length = max((dx * dx + dy * dy) ** 0.5, 1.0)
    nx, ny = -dy / length, dx / length
    cx = int(mid_x + nx * offset)
    cy = int(mid_y + ny * offset)
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
    x0 = cx - tw // 2
    y0 = cy + th // 2
    cv2.rectangle(
        img,
        (x0 - bg_pad, y0 - th - bg_pad),
        (x0 + tw + bg_pad, y0 + bg_pad),
        (0, 0, 0), -1,
    )
    cv2.putText(img, text, (x0, y0),
                cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness)


def create_pitch_diagram(
    highlight_keypoint: str | None = None,
    highlight_line: str | None = None,
    annotated_keypoints: list[str] | None = None,
    annotated_lines: list[str] | None = None,
    pitch_lines: dict | None = None,
) -> np.ndarray:
    """Create a pitch diagram with all 57 HRNet keypoints labelled."""
    annotated_keypoints = annotated_keypoints or []
    annotated_lines = annotated_lines or []

    scale = 7
    img, world_to_px, width, height = make_pitch_canvas(scale=scale, margin=50)
    draw_pitch_structure(img, world_to_px, scale=scale, landmark_radius=3)

    # Highlighted line (orange)
    if highlight_line and pitch_lines and highlight_line in pitch_lines:
        (x1, y1), (x2, y2) = pitch_lines[highlight_line]
        p1 = world_to_px(x1, y1)
        p2 = world_to_px(x2, y2)
        cv2.line(img, p1, p2, (255, 165, 0), 4)
        cv2.circle(img, p1, 6, (255, 165, 0), -1)
        cv2.circle(img, p2, 6, (255, 165, 0), -1)
        _draw_line_label(
            img, p1, p2, abbreviate_line_name(highlight_line),
            color=(255, 165, 0), font_scale=0.5, thickness=2,
        )

    # Already-annotated lines (yellow)
    if pitch_lines:
        for line_name in annotated_lines:
            if line_name in pitch_lines:
                (x1, y1), (x2, y2) = pitch_lines[line_name]
                p1 = world_to_px(x1, y1)
                p2 = world_to_px(x2, y2)
                cv2.line(img, p1, p2, (255, 255, 0), 3)
                cv2.circle(img, p1, 5, (255, 255, 0), -1)
                cv2.circle(img, p2, 5, (255, 255, 0), -1)
                _draw_line_label(
                    img, p1, p2, abbreviate_line_name(line_name),
                    color=(255, 255, 0), font_scale=0.4, thickness=2,
                )

    category_colors = {
        "GOAL": (0, 0, 255),
        "GOAL_AREA": (255, 0, 128),
        "PENALTY": (255, 0, 255),
        "CORNER": (0, 255, 0),
        "CENTER": (255, 255, 0),
        "HALFWAY": (255, 165, 0),
        "DEFAULT": (200, 200, 0),
    }

    def get_category(name: str) -> str:
        if "GOAL_TL" in name or "GOAL_TR" in name or "GOAL_BL" in name or "GOAL_BR" in name:
            return "GOAL"
        if "GOAL_AREA" in name:
            return "GOAL_AREA"
        if "PENALTY" in name or "16M" in name or "CIRCLE_TANGENT" in name:
            return "PENALTY"
        if "PITCH_CORNER" in name:
            return "CORNER"
        if "CENTER" in name or "CIRCLE" in name:
            return "CENTER"
        if "HALFWAY" in name or "TOUCH" in name:
            return "HALFWAY"
        return "DEFAULT"

    # Goal-post overlap on the ground plane.
    # The 4 keypoints per goal collapse onto 2 unique (x, y) screen
    # positions because each post has a "crossbar end" (z=-G_H) and a
    # "ground end" (z=0) at the same (x, y). The two posts of one goal
    # already differ in y (+G_HW vs -G_HW), so only crossbar-vs-ground
    # needs a screen-side nudge. We push the crossbar end *outward* (away
    # from pitch center) and the ground end *inward* — clicking either
    # picks the right kp without ambiguity.
    NOT_ON_PLANE_SET = set(NOT_ON_PLANE)  # {0, 1, 24, 25} crossbar tops
    LEFT_GOAL_IDS = {0, 1, 2, 3}
    POST_OFFSET_M = 1.4    # at scale=7 → ~10 px lateral nudge, > radius

    hrnet_pitch_points = _pk.PITCH_POINTS
    for idx, name in INTERSECTON_TO_PITCH_POINTS.items():
        if name not in hrnet_pitch_points:
            continue
        pt = hrnet_pitch_points[name]
        wx, wy = float(pt[0]), float(pt[1])
        # Goal-post posts: nudge in screen-x to disambiguate crossbar
        # (z<0) from ground (z=0) ends that share (x, y).
        if 0 <= idx <= 3 or 24 <= idx <= 27:
            is_left = idx in LEFT_GOAL_IDS
            is_crossbar = idx in NOT_ON_PLANE_SET
            # Outward = away from pitch center.
            outward = -1.0 if is_left else +1.0
            sign = +1.0 if is_crossbar else -1.0  # crossbar outward, ground inward
            display_wx = wx + sign * outward * POST_OFFSET_M
            display_wy = wy
        else:
            display_wx, display_wy = wx, wy
        px, py = world_to_px(display_wx, display_wy)

        if name in annotated_keypoints:
            color = (0, 255, 0)
            radius = 8
        elif highlight_keypoint and name == highlight_keypoint:
            color = (255, 0, 0)
            radius = 10
        else:
            color = category_colors[get_category(name)]
            radius = 4 if idx in NOT_ON_PLANE else 6

        cv2.circle(img, (px, py), radius, color, -1)
        cv2.circle(img, (px, py), radius + 1, (0, 0, 0), 1)

        # Per-keypoint label only for the currently-selected one — clicks
        # use this image as a picker, not a reference chart, so the
        # densely-packed pnlcalib/hrnet-id labels were noise. The dropdown
        # already shows the chosen name; the highlight ring is enough
        # visual feedback.
        if highlight_keypoint and name == highlight_keypoint:
            pnl_id = pitch_name_to_pnlcalib_id(name)
            label = name if pnl_id < 0 else f"{name} (id {pnl_id})"
            font_scale = 0.4
            (lw_, lh_), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)
            label_x = px + 8
            label_y = py - 6
            cv2.rectangle(
                img,
                (label_x - 2, label_y - lh_ - 2),
                (label_x + lw_ + 2, label_y + 2),
                (0, 0, 0), -1,
            )
            cv2.putText(img, label, (label_x, label_y),
                        cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), 1)

    legend_y = height - 30
    cv2.putText(img, "Click to pick keypoint  |  Red = selected  |  Green = annotated",
                (10, legend_y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1)

    count_color = (0, 255, 0) if len(annotated_keypoints) >= 4 else (255, 100, 100)
    cv2.putText(img, f"Annotated: {len(annotated_keypoints)}/4+ needed",
                (10, legend_y + 14), cv2.FONT_HERSHEY_SIMPLEX, 0.35, count_color, 1)

    legend_x = width - 120
    legend_y = 20
    for cat, color in [
        ("GOAL", (0, 0, 255)), ("GOAL_AREA", (255, 0, 128)),
        ("PENALTY", (255, 0, 255)), ("CORNER", (0, 255, 0)),
        ("CENTER", (255, 255, 0)), ("HALFWAY", (255, 165, 0)),
    ]:
        cv2.circle(img, (legend_x, legend_y), 5, color, -1)
        cv2.putText(img, cat, (legend_x + 10, legend_y + 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)
        legend_y += 15

    return img


# Distinct color per line. BGR (OpenCV).
_LINE_COLORS = {
    "touchline_top":         (255, 80, 80),     # blue-ish (BGR)
    "touchline_bottom":      (255, 160, 60),    # azure
    "goal_line_left":        (60, 180, 255),    # orange
    "goal_line_right":       (60, 230, 255),    # light orange
    "center_line":           (0, 255, 255),     # yellow
    "penalty_left_top":      (255, 0, 255),     # magenta
    "penalty_left_bottom":   (200, 0, 200),     # dark magenta
    "penalty_left_front":    (255, 100, 200),   # pink-magenta
    "penalty_right_top":     (180, 0, 255),     # purple
    "penalty_right_bottom":  (140, 0, 220),     # dark purple
    "penalty_right_front":   (220, 100, 255),   # light purple
    "goal_area_left_top":    (80, 255, 80),     # green
    "goal_area_left_bottom": (60, 200, 60),     # dark green
    "goal_area_left_front":  (140, 255, 140),   # light green
    "goal_area_right_top":   (180, 255, 0),     # cyan-green
    "goal_area_right_bottom":(140, 220, 0),     # teal
    "goal_area_right_front": (220, 255, 140),   # light cyan-green
}


def get_line_color(name: str) -> tuple[int, int, int]:
    """Return the canonical BGR color for ``name`` (white if unknown)."""
    return _LINE_COLORS.get(name, (255, 255, 255))


def create_lines_diagram(
    highlight_line: str | None = None,
    annotated_lines: list[str] | None = None,
    pitch_lines: dict | None = None,
) -> np.ndarray:
    """Draw a pitch outline plus every named line in a unique color, with labels.

    Companion to ``create_pitch_diagram`` — that one focuses on keypoints, this
    one focuses on lines so the user can pick the correct line name from the
    dropdown without ambiguity.
    """
    if pitch_lines is None:
        pitch_lines = pitch_constants.PITCH_LINES
    annotated_lines = annotated_lines or []

    scale = 7
    img, world_to_px, width, height = make_pitch_canvas(
        scale=scale, margin=60, bg=(34, 100, 34),  # darker green so colored lines pop
    )
    # Faint pitch outline as a subtle reference (so colored lines are the focus).
    pitch = pitch_constants.get_active_pitch()
    L, W = pitch.PITCH_LENGTH / 2.0, pitch.PITCH_WIDTH / 2.0
    ccr = pitch.CENTER_CIRCLE_RADIUS
    faint = (60, 130, 60)
    pts = [world_to_px(-L, W), world_to_px(L, W), world_to_px(L, -W), world_to_px(-L, -W)]
    for i in range(4):
        cv2.line(img, pts[i], pts[(i + 1) % 4], faint, 1)
    cv2.line(img, world_to_px(0, W), world_to_px(0, -W), faint, 1)
    cv2.circle(img, world_to_px(0, 0), int(ccr * scale), faint, 1)

    # Draw every line with its unique color + abbreviated label.
    for line_name, ((x1, y1), (x2, y2)) in pitch_lines.items():
        color = get_line_color(line_name)
        is_highlight = (line_name == highlight_line)
        is_annotated = (line_name in annotated_lines)

        if is_highlight:
            thickness = 5
        elif is_annotated:
            thickness = 4
        else:
            thickness = 2

        p1 = world_to_px(x1, y1)
        p2 = world_to_px(x2, y2)
        cv2.line(img, p1, p2, color, thickness)
        cv2.circle(img, p1, 3, color, -1)
        cv2.circle(img, p2, 3, color, -1)

        # Highlight ring for the selected line.
        if is_highlight:
            cv2.circle(img, p1, 7, (255, 255, 255), 1)
            cv2.circle(img, p2, 7, (255, 255, 255), 1)

        # Only label the currently-selected line and (compactly) any
        # already-annotated ones — every-line abbreviated labels were
        # cluttering up clicks.
        if is_highlight:
            _draw_line_label(img, p1, p2, abbreviate_line_name(line_name),
                             color=color, font_scale=0.5, thickness=2,
                             offset=12, bg_pad=3)
        elif is_annotated:
            _draw_line_label(img, p1, p2, "*", color=color,
                             font_scale=0.45, thickness=1,
                             offset=10, bg_pad=2)

    # Header / legend
    cv2.putText(img, f"Pitch Lines ({len(pitch_lines)}) - pick from the dropdown",
                (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
    cv2.putText(img,
                f"selected = white ring + thick | annotated = thick + * | "
                f"{len(annotated_lines)}/{len(pitch_lines)} done",
                (10, height - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.35,
                (220, 220, 220), 1)

    return img


# ----------------------------------------------------------------------
# Layout helpers — return pixel coordinates of every clickable element on
# the reference diagrams, mirroring the exact same scale/margin/offsets
# used in create_pitch_diagram / create_lines_diagram. The frontend
# click-to-pick UI uses these to hit-test image clicks.
# ----------------------------------------------------------------------

# Goal-post offset (mirror of create_pitch_diagram). Kept here so the
# frontend hit-test sees the same screen positions the user sees.
_NOT_ON_PLANE_SET = set(NOT_ON_PLANE)
_LEFT_GOAL_IDS = {0, 1, 2, 3}
_POST_OFFSET_M = 1.4


def compute_pitch_layout() -> dict:
    """Pixel coordinates of every keypoint on the reference pitch diagram.

    Returns ``{width, height, keypoints: [{name, hrnet_index, x, y}, ...]}``
    matching ``create_pitch_diagram`` exactly (scale=7, margin=50, with
    crossbar-vs-ground horizontal nudge for the 8 goal posts).
    """
    scale = 7
    _, world_to_px, width, height = make_pitch_canvas(scale=scale, margin=50)

    keypoints = []
    hrnet_pitch_points = _pk.PITCH_POINTS
    for idx, name in INTERSECTON_TO_PITCH_POINTS.items():
        if name not in hrnet_pitch_points:
            continue
        pt = hrnet_pitch_points[name]
        wx, wy = float(pt[0]), float(pt[1])
        if 0 <= idx <= 3 or 24 <= idx <= 27:
            is_left = idx in _LEFT_GOAL_IDS
            is_crossbar = idx in _NOT_ON_PLANE_SET
            outward = -1.0 if is_left else +1.0
            sign = +1.0 if is_crossbar else -1.0
            display_wx = wx + sign * outward * _POST_OFFSET_M
            display_wy = wy
        else:
            display_wx, display_wy = wx, wy
        px, py = world_to_px(display_wx, display_wy)
        keypoints.append({
            "name": name,
            "hrnet_index": int(idx),
            "x": int(px),
            "y": int(py),
        })

    return {"width": int(width), "height": int(height), "keypoints": keypoints}


def compute_lines_layout(pitch_lines: dict | None = None) -> dict:
    """Pixel coordinates of every line on the reference lines diagram.

    Returns ``{width, height, lines: [{name, x1,y1,x2,y2}, ...]}`` matching
    ``create_lines_diagram`` (scale=7, margin=60).
    """
    if pitch_lines is None:
        pitch_lines = pitch_constants.PITCH_LINES

    scale = 7
    _, world_to_px, width, height = make_pitch_canvas(scale=scale, margin=60)

    lines = []
    for line_name, ((x1, y1), (x2, y2)) in pitch_lines.items():
        p1 = world_to_px(x1, y1)
        p2 = world_to_px(x2, y2)
        lines.append({
            "name": line_name,
            "x1": int(p1[0]), "y1": int(p1[1]),
            "x2": int(p2[0]), "y2": int(p2[1]),
        })

    return {"width": int(width), "height": int(height), "lines": lines}
