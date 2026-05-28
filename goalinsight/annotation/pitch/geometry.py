"""SoccerPitch geometry with y-up convention.

Pitch center is the origin. X grows toward the right goal. Y grows toward the
top of the image / away from the camera (y-up). Goal crossbars have z = -GOAL_HEIGHT.

Adapted from soccernet-calibration-sportlight/baseline/soccerpitch.py via
goal-sight-v2, with the y-axis flipped so top = +W/2 (matches PnLCalib).

Default dimensions are FIFA-spec; pass overrides to ``__init__`` (or
``SoccerPitch.from_dict``) for non-standard fields (e.g. 7-a-side).
"""

import numpy as np


# FIFA defaults — used when a SoccerPitch is constructed with no overrides.
FIFA_DEFAULTS: dict[str, float] = {
    "pitch_length": 105.0,
    "pitch_width": 68.0,
    "penalty_area_width": 40.32,
    "penalty_area_length": 16.5,
    "goal_area_width": 18.32,
    "goal_area_length": 5.5,
    "goal_line_to_penalty_mark": 11.0,
    "center_circle_radius": 9.15,
    "goal_height": 2.44,
    "goal_length": 7.32,
}


class SoccerPitch:
    def __init__(
        self,
        pitch_length: float = FIFA_DEFAULTS["pitch_length"],
        pitch_width: float = FIFA_DEFAULTS["pitch_width"],
        penalty_area_width: float = FIFA_DEFAULTS["penalty_area_width"],
        penalty_area_length: float = FIFA_DEFAULTS["penalty_area_length"],
        goal_area_width: float = FIFA_DEFAULTS["goal_area_width"],
        goal_area_length: float = FIFA_DEFAULTS["goal_area_length"],
        goal_line_to_penalty_mark: float = FIFA_DEFAULTS["goal_line_to_penalty_mark"],
        center_circle_radius: float = FIFA_DEFAULTS["center_circle_radius"],
        goal_height: float = FIFA_DEFAULTS["goal_height"],
        goal_length: float = FIFA_DEFAULTS["goal_length"],
    ):
        self.PITCH_LENGTH = pitch_length
        self.PITCH_WIDTH = pitch_width
        self.PENALTY_AREA_WIDTH = penalty_area_width
        self.PENALTY_AREA_LENGTH = penalty_area_length
        self.GOAL_AREA_WIDTH = goal_area_width
        self.GOAL_AREA_LENGTH = goal_area_length
        self.GOAL_LINE_TO_PENALTY_MARK = goal_line_to_penalty_mark
        self.CENTER_CIRCLE_RADIUS = center_circle_radius
        self.GOAL_HEIGHT = goal_height
        self.GOAL_LENGTH = goal_length

        hL = pitch_length / 2.0
        hW = pitch_width / 2.0
        pa_w = self.PENALTY_AREA_WIDTH / 2.0
        pa_d = self.PENALTY_AREA_LENGTH
        ga_w = self.GOAL_AREA_WIDTH / 2.0
        ga_d = self.GOAL_AREA_LENGTH
        g_w = self.GOAL_LENGTH / 2.0
        g_h = self.GOAL_HEIGHT
        ccr = self.CENTER_CIRCLE_RADIUS

        def pt(x: float, y: float, z: float = 0.0) -> np.ndarray:
            return np.array([x, y, z], dtype=float)

        center_mark = pt(0, 0)
        t_touch_halfway = pt(0, hW)
        b_touch_halfway = pt(0, -hW)
        t_halfway_circle = pt(0, ccr)
        b_halfway_circle = pt(0, -ccr)

        tr_corner = pt(hL, hW)
        tl_corner = pt(-hL, hW)
        br_corner = pt(hL, -hW)
        bl_corner = pt(-hL, -hW)

        l_goal_bl_post = pt(-hL, -g_w, 0.0)
        l_goal_tl_post = pt(-hL, -g_w, -g_h)
        l_goal_br_post = pt(-hL, g_w, 0.0)
        l_goal_tr_post = pt(-hL, g_w, -g_h)
        r_goal_bl_post = pt(hL, g_w, 0.0)
        r_goal_tl_post = pt(hL, g_w, -g_h)
        r_goal_br_post = pt(hL, -g_w, 0.0)
        r_goal_tr_post = pt(hL, -g_w, -g_h)

        l_pen_mark = pt(-hL + self.GOAL_LINE_TO_PENALTY_MARK, 0)
        r_pen_mark = pt(hL - self.GOAL_LINE_TO_PENALTY_MARK, 0)

        # Penalty areas (y-up: top = +pa_w, bottom = -pa_w)
        l_pa_tl = pt(-hL, pa_w)
        l_pa_tr = pt(-hL + pa_d, pa_w)
        l_pa_bl = pt(-hL, -pa_w)
        l_pa_br = pt(-hL + pa_d, -pa_w)
        r_pa_tr = pt(hL, pa_w)
        r_pa_tl = pt(hL - pa_d, pa_w)
        r_pa_br = pt(hL, -pa_w)
        r_pa_bl = pt(hL - pa_d, -pa_w)

        # Goal areas
        l_ga_tl = pt(-hL, ga_w)
        l_ga_tr = pt(-hL + ga_d, ga_w)
        l_ga_bl = pt(-hL, -ga_w)
        l_ga_br = pt(-hL + ga_d, -ga_w)
        r_ga_tr = pt(hL, ga_w)
        r_ga_tl = pt(hL - ga_d, ga_w)
        r_ga_br = pt(hL, -ga_w)
        r_ga_bl = pt(hL - ga_d, -ga_w)

        # Penalty-arc intersections with penalty-area front line.
        # dx from pen-mark to front line; y via circle equation.
        dx = pa_d - self.GOAL_LINE_TO_PENALTY_MARK
        # Non-FIFA pitches may have center_circle_radius < dx, which makes the
        # arc not intersect the front line. In that case the keypoint is
        # geometrically undefined; emit NaN so consumers can detect & skip.
        radicand = ccr ** 2 - dx ** 2
        arc_y = float(np.sqrt(radicand)) if radicand > 0 else float("nan")
        l_front_x = -hL + pa_d
        r_front_x = hL - pa_d
        tl_16m = pt(l_front_x, arc_y)
        bl_16m = pt(l_front_x, -arc_y)
        tr_16m = pt(r_front_x, arc_y)
        br_16m = pt(r_front_x, -arc_y)

        self.point_dict = {
            "CENTER_MARK": center_mark,
            "L_PENALTY_MARK": l_pen_mark,
            "R_PENALTY_MARK": r_pen_mark,
            "TL_PITCH_CORNER": tl_corner,
            "BL_PITCH_CORNER": bl_corner,
            "TR_PITCH_CORNER": tr_corner,
            "BR_PITCH_CORNER": br_corner,
            "L_PENALTY_AREA_TL_CORNER": l_pa_tl,
            "L_PENALTY_AREA_TR_CORNER": l_pa_tr,
            "L_PENALTY_AREA_BL_CORNER": l_pa_bl,
            "L_PENALTY_AREA_BR_CORNER": l_pa_br,
            "R_PENALTY_AREA_TL_CORNER": r_pa_tl,
            "R_PENALTY_AREA_TR_CORNER": r_pa_tr,
            "R_PENALTY_AREA_BL_CORNER": r_pa_bl,
            "R_PENALTY_AREA_BR_CORNER": r_pa_br,
            "L_GOAL_AREA_TL_CORNER": l_ga_tl,
            "L_GOAL_AREA_TR_CORNER": l_ga_tr,
            "L_GOAL_AREA_BL_CORNER": l_ga_bl,
            "L_GOAL_AREA_BR_CORNER": l_ga_br,
            "R_GOAL_AREA_TL_CORNER": r_ga_tl,
            "R_GOAL_AREA_TR_CORNER": r_ga_tr,
            "R_GOAL_AREA_BL_CORNER": r_ga_bl,
            "R_GOAL_AREA_BR_CORNER": r_ga_br,
            "L_GOAL_TL_POST": l_goal_tl_post,
            "L_GOAL_TR_POST": l_goal_tr_post,
            "L_GOAL_BL_POST": l_goal_bl_post,
            "L_GOAL_BR_POST": l_goal_br_post,
            "R_GOAL_TL_POST": r_goal_tl_post,
            "R_GOAL_TR_POST": r_goal_tr_post,
            "R_GOAL_BL_POST": r_goal_bl_post,
            "R_GOAL_BR_POST": r_goal_br_post,
            "T_TOUCH_AND_HALFWAY_LINES_INTERSECTION": t_touch_halfway,
            "B_TOUCH_AND_HALFWAY_LINES_INTERSECTION": b_touch_halfway,
            "T_HALFWAY_LINE_AND_CENTER_CIRCLE_INTERSECTION": t_halfway_circle,
            "B_HALFWAY_LINE_AND_CENTER_CIRCLE_INTERSECTION": b_halfway_circle,
            "TL_16M_LINE_AND_PENALTY_ARC_INTERSECTION": tl_16m,
            "BL_16M_LINE_AND_PENALTY_ARC_INTERSECTION": bl_16m,
            "TR_16M_LINE_AND_PENALTY_ARC_INTERSECTION": tr_16m,
            "BR_16M_LINE_AND_PENALTY_ARC_INTERSECTION": br_16m,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "SoccerPitch":
        """Construct from a dict (e.g. parsed YAML).

        Unknown keys are ignored. Missing keys default to FIFA values.
        """
        kwargs = {k: data[k] for k in FIFA_DEFAULTS if k in data}
        return cls(**kwargs)
