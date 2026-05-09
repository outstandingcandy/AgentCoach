"""57 HRNet pitch keypoints (y-up convention).

Pitch center is the origin. X grows toward the right goal. Y grows toward the
top of the image (y-up). Crossbar points have z = -GOAL_HEIGHT.

Ported from goal-sight-v2 `pitch/keypoints.py` with the y-axis flipped so that
labels like CENTER_CIRCLE_TANGENT_TR denote the point with y > 0 and x > 0.
"""

from typing import List, Tuple

import numpy as np

from .geometry import SoccerPitch


def _circle_tangent_points(
    circle_center: Tuple[float, float],
    radius: float,
    point: Tuple[float, float],
):
    """Return the two tangent points from `point` to the circle."""
    dx = point[0] - circle_center[0]
    dy = point[1] - circle_center[1]
    hypotenuse = float(np.sqrt(dx * dx + dy * dy))
    th = float(np.arccos(radius / hypotenuse))
    d = float(np.arctan2(dy, dx))
    t1 = (
        circle_center[0] + radius * np.cos(d + th),
        circle_center[1] + radius * np.sin(d + th),
    )
    t2 = (
        circle_center[0] + radius * np.cos(d - th),
        circle_center[1] + radius * np.sin(d - th),
    )
    return t1, t2


def _pick_by_x(a, b, want_positive: bool):
    """Pick the tangent with the desired x sign."""
    if (a[0] > b[0]) == want_positive:
        return a, b
    return b, a


def _pick_by_y(a, b, want_positive: bool):
    """Pick the tangent with the desired y sign."""
    if (a[1] > b[1]) == want_positive:
        return a, b
    return b, a


def _to_arr(xy, z: float = 0.0) -> np.ndarray:
    return np.array([xy[0], xy[1], z], dtype=float)


def _build_pitch_points():
    pitch = SoccerPitch()
    points = {**pitch.point_dict}
    R = pitch.CENTER_CIRCLE_RADIUS

    # Center-circle tangents from the touchline-halfway intersections.
    # Top tangents (from y>0 point) split into TR (x>0) and TL (x<0).
    tt1, tt2 = _circle_tangent_points(
        points["CENTER_MARK"][:2], R,
        points["T_TOUCH_AND_HALFWAY_LINES_INTERSECTION"][:2],
    )
    tr, tl = _pick_by_x(tt1, tt2, want_positive=True)
    points["CENTER_CIRCLE_TANGENT_TR"] = _to_arr(tr)
    points["CENTER_CIRCLE_TANGENT_TL"] = _to_arr(tl)

    bt1, bt2 = _circle_tangent_points(
        points["CENTER_MARK"][:2], R,
        points["B_TOUCH_AND_HALFWAY_LINES_INTERSECTION"][:2],
    )
    br, bl = _pick_by_x(bt1, bt2, want_positive=True)
    points["CENTER_CIRCLE_TANGENT_BR"] = _to_arr(br)
    points["CENTER_CIRCLE_TANGENT_BL"] = _to_arr(bl)

    # Center-circle diagonals: ±R/sqrt(2) in both axes.
    sq = float(np.sqrt(2.0) * R / 2.0)
    points["CENTER_CIRCLE_TR"] = np.array([sq, sq, 0], dtype=float)
    points["CENTER_CIRCLE_TL"] = np.array([-sq, sq, 0], dtype=float)
    points["CENTER_CIRCLE_BR"] = np.array([sq, -sq, 0], dtype=float)
    points["CENTER_CIRCLE_BL"] = np.array([-sq, -sq, 0], dtype=float)

    # Center-circle axis points (on x-axis).
    points["CENTER_CIRCLE_R"] = np.array([R, 0, 0], dtype=float)
    points["CENTER_CIRCLE_L"] = np.array([-R, 0, 0], dtype=float)

    # Penalty arc "axis" points — CENTER_CIRCLE_R displaced to penalty marks.
    points["LEFT_CIRCLE_R"] = points["L_PENALTY_MARK"] + points["CENTER_CIRCLE_R"]
    points["RIGHT_CIRCLE_L"] = points["R_PENALTY_MARK"] + points["CENTER_CIRCLE_L"]

    # Left penalty-arc tangents: from the TR/BR corners of the left penalty area.
    lt1, lt2 = _circle_tangent_points(
        points["L_PENALTY_MARK"][:2], R,
        points["L_PENALTY_AREA_TR_CORNER"][:2],
    )
    left_t, _ = _pick_by_y(lt1, lt2, want_positive=True)
    points["LEFT_CIRCLE_TANGENT_T"] = _to_arr(left_t)

    lb1, lb2 = _circle_tangent_points(
        points["L_PENALTY_MARK"][:2], R,
        points["L_PENALTY_AREA_BR_CORNER"][:2],
    )
    left_b, _ = _pick_by_y(lb1, lb2, want_positive=False)
    points["LEFT_CIRCLE_TANGENT_B"] = _to_arr(left_b)

    # L_MIDDLE_PENALTY: x at penalty-area front line, y = 0.
    l_mid = points["L_PENALTY_AREA_BR_CORNER"].copy()
    l_mid[1] = 0.0
    points["L_MIDDLE_PENALTY"] = l_mid

    # Right penalty-arc tangents: from the TL/BL corners of the right penalty area.
    rt1, rt2 = _circle_tangent_points(
        points["R_PENALTY_MARK"][:2], R,
        points["R_PENALTY_AREA_TL_CORNER"][:2],
    )
    right_t, _ = _pick_by_y(rt1, rt2, want_positive=True)
    points["RIGHT_CIRCLE_TANGENT_T"] = _to_arr(right_t)

    rb1, rb2 = _circle_tangent_points(
        points["R_PENALTY_MARK"][:2], R,
        points["R_PENALTY_AREA_BL_CORNER"][:2],
    )
    right_b, _ = _pick_by_y(rb1, rb2, want_positive=False)
    points["RIGHT_CIRCLE_TANGENT_B"] = _to_arr(right_b)

    r_mid = points["R_PENALTY_AREA_BL_CORNER"].copy()
    r_mid[1] = 0.0
    points["R_MIDDLE_PENALTY"] = r_mid

    return points


PITCH_POINTS = _build_pitch_points()


INTERSECTON_TO_PITCH_POINTS = {
    0: "L_GOAL_TL_POST",
    1: "L_GOAL_TR_POST",
    2: "L_GOAL_BL_POST",
    3: "L_GOAL_BR_POST",
    4: "L_GOAL_AREA_BR_CORNER",
    5: "L_GOAL_AREA_TR_CORNER",
    6: "L_GOAL_AREA_BL_CORNER",
    7: "L_GOAL_AREA_TL_CORNER",
    8: "L_PENALTY_AREA_BR_CORNER",
    9: "L_PENALTY_AREA_TR_CORNER",
    10: "L_PENALTY_AREA_BL_CORNER",
    11: "L_PENALTY_AREA_TL_CORNER",
    12: "BL_PITCH_CORNER",
    13: "TL_PITCH_CORNER",
    14: "B_TOUCH_AND_HALFWAY_LINES_INTERSECTION",
    15: "T_TOUCH_AND_HALFWAY_LINES_INTERSECTION",
    16: "R_PENALTY_AREA_BL_CORNER",
    17: "R_PENALTY_AREA_TL_CORNER",
    18: "R_PENALTY_AREA_BR_CORNER",
    19: "R_PENALTY_AREA_TR_CORNER",
    20: "R_GOAL_AREA_BL_CORNER",
    21: "R_GOAL_AREA_TL_CORNER",
    22: "R_GOAL_AREA_BR_CORNER",
    23: "R_GOAL_AREA_TR_CORNER",
    24: "R_GOAL_TL_POST",
    25: "R_GOAL_TR_POST",
    26: "R_GOAL_BL_POST",
    27: "R_GOAL_BR_POST",
    28: "BR_PITCH_CORNER",
    29: "TR_PITCH_CORNER",
    30: "CENTER_CIRCLE_TANGENT_TR",
    31: "CENTER_CIRCLE_TANGENT_TL",
    32: "CENTER_CIRCLE_TANGENT_BR",
    33: "CENTER_CIRCLE_TANGENT_BL",
    34: "CENTER_CIRCLE_TR",
    35: "CENTER_CIRCLE_TL",
    36: "CENTER_CIRCLE_BR",
    37: "CENTER_CIRCLE_BL",
    38: "CENTER_CIRCLE_R",
    39: "CENTER_CIRCLE_L",
    40: "T_HALFWAY_LINE_AND_CENTER_CIRCLE_INTERSECTION",
    41: "B_HALFWAY_LINE_AND_CENTER_CIRCLE_INTERSECTION",
    42: "CENTER_MARK",
    43: "LEFT_CIRCLE_R",
    44: "BL_16M_LINE_AND_PENALTY_ARC_INTERSECTION",
    45: "TL_16M_LINE_AND_PENALTY_ARC_INTERSECTION",
    46: "LEFT_CIRCLE_TANGENT_T",
    47: "LEFT_CIRCLE_TANGENT_B",
    48: "L_PENALTY_MARK",
    49: "L_MIDDLE_PENALTY",
    50: "RIGHT_CIRCLE_L",
    51: "BR_16M_LINE_AND_PENALTY_ARC_INTERSECTION",
    52: "TR_16M_LINE_AND_PENALTY_ARC_INTERSECTION",
    53: "RIGHT_CIRCLE_TANGENT_T",
    54: "RIGHT_CIRCLE_TANGENT_B",
    55: "R_PENALTY_MARK",
    56: "R_MIDDLE_PENALTY",
}

PITCH_POINTS_TO_INTERSECTON = {v: k for k, v in INTERSECTON_TO_PITCH_POINTS.items()}

# Crossbar-top points (z != 0): ids of the four posts at z = -GOAL_HEIGHT.
NOT_ON_PLANE: List[int] = [0, 1, 24, 25]
