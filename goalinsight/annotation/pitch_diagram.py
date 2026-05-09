"""Pitch diagram reference view showing all 57 HRNet keypoints.

World convention is y-up (top = +W/2); the rendering flips y so that points
with y > 0 appear at the top of the image.
"""

import cv2
import numpy as np

from .keypoint_utils import abbreviate_line_name
from .pitch.keypoints import (
    PITCH_POINTS as HRNET_PITCH_POINTS,
    INTERSECTON_TO_PITCH_POINTS,
    NOT_ON_PLANE,
)
from .pitch_constants import PITCH_LENGTH, PITCH_WIDTH


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
    margin = 50
    width = int(PITCH_LENGTH * scale + 2 * margin)
    height = int(PITCH_WIDTH * scale + 2 * margin)

    img = np.zeros((height, width, 3), dtype=np.uint8)
    img[:] = (34, 139, 34)

    L, W = PITCH_LENGTH / 2, PITCH_WIDTH / 2

    def world_to_px(x: float, y: float) -> tuple[int, int]:
        # y-up world -> y-down pixels: flip y around the pitch center.
        px = int((x + L) * scale + margin)
        py = int((W - y) * scale + margin)
        return (px, py)

    white = (255, 255, 255)

    pts = [
        world_to_px(-L, W), world_to_px(L, W),
        world_to_px(L, -W), world_to_px(-L, -W),
    ]
    for i in range(4):
        cv2.line(img, pts[i], pts[(i + 1) % 4], white, 2)

    # Center line and circle
    cv2.line(img, world_to_px(0, W), world_to_px(0, -W), white, 2)
    cv2.circle(img, world_to_px(0, 0), int(9.15 * scale), white, 2)
    cv2.circle(img, world_to_px(0, 0), 3, white, -1)

    # Penalty areas
    pa_w, pa_d = 40.32 / 2, 16.5
    cv2.rectangle(img, world_to_px(-L, pa_w), world_to_px(-L + pa_d, -pa_w), white, 2)
    cv2.rectangle(img, world_to_px(L - pa_d, pa_w), world_to_px(L, -pa_w), white, 2)

    # Goal areas
    ga_w, ga_d = 18.32 / 2, 5.5
    cv2.rectangle(img, world_to_px(-L, ga_w), world_to_px(-L + ga_d, -ga_w), white, 2)
    cv2.rectangle(img, world_to_px(L - ga_d, ga_w), world_to_px(L, -ga_w), white, 2)

    # Penalty spots
    cv2.circle(img, world_to_px(-L + 11, 0), 3, white, -1)
    cv2.circle(img, world_to_px(L - 11, 0), 3, white, -1)

    # Penalty arcs (arc on the center-circle side of the penalty area)
    cv2.ellipse(img, world_to_px(-L + 11, 0), (int(9.15 * scale), int(9.15 * scale)), 0, -60, 60, white, 2)
    cv2.ellipse(img, world_to_px(L - 11, 0), (int(9.15 * scale), int(9.15 * scale)), 0, 120, 240, white, 2)

    # Highlighted line (orange)
    if highlight_line and pitch_lines and highlight_line in pitch_lines:
        (x1, y1), (x2, y2) = pitch_lines[highlight_line]
        p1 = world_to_px(x1, y1)
        p2 = world_to_px(x2, y2)
        cv2.line(img, p1, p2, (255, 165, 0), 4)
        cv2.circle(img, p1, 6, (255, 165, 0), -1)
        cv2.circle(img, p2, 6, (255, 165, 0), -1)
        mid_x = (p1[0] + p2[0]) // 2
        mid_y = (p1[1] + p2[1]) // 2
        cv2.putText(
            img, abbreviate_line_name(highlight_line), (mid_x - 20, mid_y - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 165, 0), 2,
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

    # Nudge overlapping goal posts apart so their labels don't overlap.
    # In y-up world, +y = top; goal post tops (crossbar, NOT_ON_PLANE ids 0,1,24,25)
    # get a slight +y (toward top of screen); grounded posts (2,3,26,27) get -y.
    GOAL_POST_DISPLAY_OFFSET = {
        0: 1.5, 1: 1.5, 24: 1.5, 25: 1.5,
        2: -1.5, 3: -1.5, 26: -1.5, 27: -1.5,
    }

    for idx, name in INTERSECTON_TO_PITCH_POINTS.items():
        if name not in HRNET_PITCH_POINTS:
            continue
        pt = HRNET_PITCH_POINTS[name]
        wx, wy = float(pt[0]), float(pt[1])
        display_wy = wy + GOAL_POST_DISPLAY_OFFSET.get(idx, 0)
        px, py = world_to_px(wx, display_wy)

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

        if idx in [0, 1, 24, 25]:
            label = f"{idx}T"
        elif idx in [2, 3, 26, 27]:
            label = f"{idx}B"
        else:
            label = str(idx)

        font_scale = 0.3
        label_x = px + 6
        label_y = py - 4

        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)
        bg_color = (0, 80, 0) if name in annotated_keypoints else (40, 40, 40)
        cv2.rectangle(
            img, (label_x - 1, label_y - th - 1), (label_x + tw + 1, label_y + 1),
            bg_color, -1,
        )
        cv2.putText(img, label, (label_x, label_y),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), 1)

    legend_y = height - 60
    cv2.putText(img, "HRNet 57 Keypoints (numbers = index)", (10, legend_y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    cv2.putText(img, "Red=Selected | Green=Annotated", (10, legend_y + 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1)

    count_color = (0, 255, 0) if len(annotated_keypoints) >= 4 else (255, 100, 100)
    cv2.putText(img, f"Annotated: {len(annotated_keypoints)}/4+ needed",
                (10, legend_y + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.35, count_color, 1)

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
