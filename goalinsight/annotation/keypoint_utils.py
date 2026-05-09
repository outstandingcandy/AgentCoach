"""Keypoint name mappings and helpers for the annotator.

Two naming schemes exist:
- Legacy (shorthand like "corner_top_left"), used by PITCH_KEYPOINTS in
  pitch_constants — convenient for simple dropdowns.
- HRNet (full names like "TL_PITCH_CORNER"), used by the 57-point pitch model.

The user annotates via HRNet names; saved annotations store both the name and
the hrnet_index, which the v3 finetune dataloader ignores in favor of the
world coordinates.
"""

from .pitch.keypoints import (
    PITCH_POINTS as HRNET_PITCH_POINTS,
    PITCH_POINTS_TO_INTERSECTON,
    INTERSECTON_TO_PITCH_POINTS,
    NOT_ON_PLANE,
)


LEGACY_TO_HRNET = {
    "corner_top_left": "TL_PITCH_CORNER",
    "corner_top_right": "TR_PITCH_CORNER",
    "corner_bottom_left": "BL_PITCH_CORNER",
    "corner_bottom_right": "BR_PITCH_CORNER",
    "center_spot": "CENTER_MARK",
    "center_top": "T_TOUCH_AND_HALFWAY_LINES_INTERSECTION",
    "center_bottom": "B_TOUCH_AND_HALFWAY_LINES_INTERSECTION",
    "center_circle_top": "T_HALFWAY_LINE_AND_CENTER_CIRCLE_INTERSECTION",
    "center_circle_bottom": "B_HALFWAY_LINE_AND_CENTER_CIRCLE_INTERSECTION",
    "center_circle_left": "CENTER_CIRCLE_L",
    "center_circle_right": "CENTER_CIRCLE_R",
    "penalty_left_top_outer": "L_PENALTY_AREA_TL_CORNER",
    "penalty_left_top_inner": "L_PENALTY_AREA_TR_CORNER",
    "penalty_left_bottom_outer": "L_PENALTY_AREA_BL_CORNER",
    "penalty_left_bottom_inner": "L_PENALTY_AREA_BR_CORNER",
    "penalty_left_arc_top": "TL_16M_LINE_AND_PENALTY_ARC_INTERSECTION",
    "penalty_left_arc_bottom": "BL_16M_LINE_AND_PENALTY_ARC_INTERSECTION",
    "penalty_right_top_outer": "R_PENALTY_AREA_TL_CORNER",
    "penalty_right_top_inner": "R_PENALTY_AREA_TR_CORNER",
    "penalty_right_bottom_outer": "R_PENALTY_AREA_BL_CORNER",
    "penalty_right_bottom_inner": "R_PENALTY_AREA_BR_CORNER",
    "penalty_right_arc_top": "TR_16M_LINE_AND_PENALTY_ARC_INTERSECTION",
    "penalty_right_arc_bottom": "BR_16M_LINE_AND_PENALTY_ARC_INTERSECTION",
    "goal_area_left_top_outer": "L_GOAL_AREA_TL_CORNER",
    "goal_area_left_top_inner": "L_GOAL_AREA_TR_CORNER",
    "goal_area_left_bottom_outer": "L_GOAL_AREA_BL_CORNER",
    "goal_area_left_bottom_inner": "L_GOAL_AREA_BR_CORNER",
    "goal_area_right_top_outer": "R_GOAL_AREA_TL_CORNER",
    "goal_area_right_top_inner": "R_GOAL_AREA_TR_CORNER",
    "goal_area_right_bottom_outer": "R_GOAL_AREA_BL_CORNER",
    "goal_area_right_bottom_inner": "R_GOAL_AREA_BR_CORNER",
    "penalty_spot_left": "L_PENALTY_MARK",
    "penalty_spot_right": "R_PENALTY_MARK",
    "goal_left_top": "L_GOAL_BL_POST",
    "goal_left_bottom": "L_GOAL_BR_POST",
    "goal_right_top": "R_GOAL_BL_POST",
    "goal_right_bottom": "R_GOAL_BR_POST",
}


def get_hrnet_keypoints_2d() -> dict[str, tuple[float, float]]:
    """Return HRNet keypoints as (x, y) world coords, dropping z."""
    return {name: (float(pt[0]), float(pt[1])) for name, pt in HRNET_PITCH_POINTS.items()}


def get_hrnet_keypoint_choices() -> list[str]:
    """Dropdown-friendly choices: 'INDEX: NAME'."""
    return [f"{idx}: {name}" for idx, name in INTERSECTON_TO_PITCH_POINTS.items()]


def parse_keypoint_choice(choice: str) -> tuple[int, str]:
    parts = choice.split(": ", 1)
    idx = int(parts[0])
    name = parts[1] if len(parts) > 1 else INTERSECTON_TO_PITCH_POINTS[idx]
    return idx, name


def convert_keypoint_name(name: str) -> str:
    if name in HRNET_PITCH_POINTS:
        return name
    return LEGACY_TO_HRNET.get(name, name)


def find_nearest_keypoint(
    wx: float,
    wy: float,
    threshold: float = 5.0,
) -> tuple[str | None, int | None]:
    """Find the nearest HRNet keypoint within `threshold` meters."""
    min_dist = float("inf")
    nearest_name = None
    nearest_idx = None

    for idx, name in INTERSECTON_TO_PITCH_POINTS.items():
        if name not in HRNET_PITCH_POINTS:
            continue
        pt = HRNET_PITCH_POINTS[name]
        kx, ky = float(pt[0]), float(pt[1])
        dist = ((wx - kx) ** 2 + (wy - ky) ** 2) ** 0.5
        if dist < min_dist and dist < threshold:
            min_dist = dist
            nearest_name = name
            nearest_idx = idx

    return nearest_name, nearest_idx


def abbreviate_line_name(name: str) -> str:
    abbrev = {
        "touchline": "TL",
        "goal_line": "GL",
        "center_line": "CL",
        "penalty": "P",
        "goal_area": "GA",
        "top": "T",
        "bottom": "B",
        "left": "L",
        "right": "R",
        "front": "F",
    }
    parts = name.split("_")
    return "".join(abbrev.get(p, p[0].upper()) for p in parts)


# 2D world-coords of HRNet keypoints for homography.
PITCH_KEYPOINTS = get_hrnet_keypoints_2d()
