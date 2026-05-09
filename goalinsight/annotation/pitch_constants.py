"""FIFA pitch keypoints and lines (y-up convention).

Pitch center is the origin. X grows toward the right goal. Y grows toward the
top of the image / away from the camera (y-up). This matches the PnLCalib
world-coordinate convention used by the v3 finetune dataloader.
"""

PITCH_LENGTH = 105.0
PITCH_WIDTH = 68.0

PENALTY_AREA_WIDTH = 40.32
PENALTY_AREA_DEPTH = 16.5
GOAL_AREA_WIDTH = 18.32
GOAL_AREA_DEPTH = 5.5
PENALTY_SPOT_DISTANCE = 11.0
CENTER_CIRCLE_RADIUS = 9.15
CORNER_ARC_RADIUS = 1.0
GOAL_WIDTH = 7.32

_L = PITCH_LENGTH / 2
_W = PITCH_WIDTH / 2
_PA_W = PENALTY_AREA_WIDTH / 2
_GA_W = GOAL_AREA_WIDTH / 2
_ARC_Y = 7.31
_GOAL_W = GOAL_WIDTH / 2

# Keypoints in world coordinates (meters). y-up: top = +W/2, bottom = -W/2.
PITCH_KEYPOINTS = {
    "corner_top_left": (-_L, _W),
    "corner_top_right": (_L, _W),
    "corner_bottom_left": (-_L, -_W),
    "corner_bottom_right": (_L, -_W),
    "center_spot": (0.0, 0.0),
    "center_top": (0.0, _W),
    "center_bottom": (0.0, -_W),
    "center_circle_top": (0.0, CENTER_CIRCLE_RADIUS),
    "center_circle_bottom": (0.0, -CENTER_CIRCLE_RADIUS),
    "center_circle_left": (-CENTER_CIRCLE_RADIUS, 0.0),
    "center_circle_right": (CENTER_CIRCLE_RADIUS, 0.0),
    "penalty_left_top_outer": (-_L, _PA_W),
    "penalty_left_top_inner": (-_L + PENALTY_AREA_DEPTH, _PA_W),
    "penalty_left_bottom_outer": (-_L, -_PA_W),
    "penalty_left_bottom_inner": (-_L + PENALTY_AREA_DEPTH, -_PA_W),
    "penalty_left_arc_top": (-_L + PENALTY_AREA_DEPTH, _ARC_Y),
    "penalty_left_arc_bottom": (-_L + PENALTY_AREA_DEPTH, -_ARC_Y),
    "penalty_right_top_outer": (_L, _PA_W),
    "penalty_right_top_inner": (_L - PENALTY_AREA_DEPTH, _PA_W),
    "penalty_right_bottom_outer": (_L, -_PA_W),
    "penalty_right_bottom_inner": (_L - PENALTY_AREA_DEPTH, -_PA_W),
    "penalty_right_arc_top": (_L - PENALTY_AREA_DEPTH, _ARC_Y),
    "penalty_right_arc_bottom": (_L - PENALTY_AREA_DEPTH, -_ARC_Y),
    "goal_area_left_top_outer": (-_L, _GA_W),
    "goal_area_left_top_inner": (-_L + GOAL_AREA_DEPTH, _GA_W),
    "goal_area_left_bottom_outer": (-_L, -_GA_W),
    "goal_area_left_bottom_inner": (-_L + GOAL_AREA_DEPTH, -_GA_W),
    "goal_area_right_top_outer": (_L, _GA_W),
    "goal_area_right_top_inner": (_L - GOAL_AREA_DEPTH, _GA_W),
    "goal_area_right_bottom_outer": (_L, -_GA_W),
    "goal_area_right_bottom_inner": (_L - GOAL_AREA_DEPTH, -_GA_W),
    "penalty_spot_left": (-_L + PENALTY_SPOT_DISTANCE, 0.0),
    "penalty_spot_right": (_L - PENALTY_SPOT_DISTANCE, 0.0),
    "goal_left_top": (-_L, _GOAL_W),
    "goal_left_bottom": (-_L, -_GOAL_W),
    "goal_right_top": (_L, _GOAL_W),
    "goal_right_bottom": (_L, -_GOAL_W),
    "touchline_top_left": (-_L / 2, _W),
    "touchline_top_right": (_L / 2, _W),
    "touchline_bottom_left": (-_L / 2, -_W),
    "touchline_bottom_right": (_L / 2, -_W),
}

# Pitch lines: (start, end) endpoints. y-up convention.
PITCH_LINES = {
    "touchline_top": ((-_L, _W), (_L, _W)),
    "touchline_bottom": ((-_L, -_W), (_L, -_W)),
    "goal_line_left": ((-_L, _W), (-_L, -_W)),
    "goal_line_right": ((_L, _W), (_L, -_W)),
    "center_line": ((0.0, _W), (0.0, -_W)),
    "penalty_left_top": ((-_L, _PA_W), (-_L + PENALTY_AREA_DEPTH, _PA_W)),
    "penalty_left_bottom": ((-_L, -_PA_W), (-_L + PENALTY_AREA_DEPTH, -_PA_W)),
    "penalty_left_front": ((-_L + PENALTY_AREA_DEPTH, _PA_W), (-_L + PENALTY_AREA_DEPTH, -_PA_W)),
    "penalty_right_top": ((_L, _PA_W), (_L - PENALTY_AREA_DEPTH, _PA_W)),
    "penalty_right_bottom": ((_L, -_PA_W), (_L - PENALTY_AREA_DEPTH, -_PA_W)),
    "penalty_right_front": ((_L - PENALTY_AREA_DEPTH, _PA_W), (_L - PENALTY_AREA_DEPTH, -_PA_W)),
    "goal_area_left_top": ((-_L, _GA_W), (-_L + GOAL_AREA_DEPTH, _GA_W)),
    "goal_area_left_bottom": ((-_L, -_GA_W), (-_L + GOAL_AREA_DEPTH, -_GA_W)),
    "goal_area_left_front": ((-_L + GOAL_AREA_DEPTH, _GA_W), (-_L + GOAL_AREA_DEPTH, -_GA_W)),
    "goal_area_right_top": ((_L, _GA_W), (_L - GOAL_AREA_DEPTH, _GA_W)),
    "goal_area_right_bottom": ((_L, -_GA_W), (_L - GOAL_AREA_DEPTH, -_GA_W)),
    "goal_area_right_front": ((_L - GOAL_AREA_DEPTH, _GA_W), (_L - GOAL_AREA_DEPTH, -_GA_W)),
}


def get_all_keypoint_names() -> list[str]:
    return list(PITCH_KEYPOINTS.keys())


def get_all_line_names() -> list[str]:
    return list(PITCH_LINES.keys())
