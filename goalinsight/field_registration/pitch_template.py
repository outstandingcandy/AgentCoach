"""Shared pitch template builder for field-registration backends.

Single source of truth for the 57 PnLCalib keypoint world coordinates plus
the 23 line definitions and ground-plane line intersections. Used by:

- ``physical_calibrator.PhysicalCalibrator`` (directly)
- ``pnlcalib_orig.pitch_template.build_keypoint_table`` (adapter that
  re-formats the output to upstream's 1-indexed / y-down conventions)

Conventions follow v3 across the rest of the codebase (see auto-memory
``pnlcalib_y_up``):

- Origin at pitch center.
- ``+x`` toward the right goal.
- ``+y`` toward the top of the image / away from the camera (y-up).
- Keypoint IDs are **0-indexed** (kp 0 = TL pitch corner). The PnLCalib
  upstream port uses 1-indexed and y-down — that translation lives in
  ``pnlcalib_orig/pitch_template.py``, not here.
- 4 crossbar-top keypoints (IDs 12, 14, 16, 18) carry ``z = +goal_height``
  (z-positive = up; OpenCV's ``solvePnP`` places no requirement on which
  world-frame sign means "up" — the choice is ours, and this is it).
"""

from __future__ import annotations

import math


# Crossbar-top kp IDs (0-indexed, v3 convention). Only z != 0 entries.
NON_GROUND_KEYPOINT_IDS: set[int] = {12, 14, 16, 18}


# Default marking dimensions (FIFA standard). Used when a pitch_dims dict
# omits a field. Match `goalinsight.annotation.pitch.geometry` defaults.
FIFA_DEFAULTS: dict[str, float] = {
    "pitch_length": 105.0,
    "pitch_width": 68.0,
    "penalty_area_width": 40.32,    # full width
    "penalty_area_length": 16.5,
    "goal_area_width": 18.32,
    "goal_area_length": 5.5,
    "goal_line_to_penalty_mark": 11.0,
    "center_circle_radius": 9.15,
    "goal_height": 2.44,
    "goal_length": 7.32,             # full goal width (post-to-post)
}


def _resolve(pitch_dims: dict | None) -> dict:
    out = dict(FIFA_DEFAULTS)
    if pitch_dims:
        out.update(pitch_dims)
    return out


def build_field_template(
    pitch_dims: dict | None = None,
) -> tuple[list[list[float]], dict, dict]:
    """Build the 57 keypoint coords, 23 line defs, and intersection table.

    Args:
        pitch_dims: Mapping of pitch geometry to floats (meters). Missing
            keys fall back to FIFA. Recognized keys:

            - ``pitch_length``, ``pitch_width``
            - ``penalty_area_length``, ``penalty_area_width``
            - ``goal_area_length``, ``goal_area_width``
            - ``goal_line_to_penalty_mark``
            - ``center_circle_radius``
            - ``goal_height``, ``goal_length``

            Pass ``None`` (or an empty dict) for FIFA-spec output.

    Returns:
        ``(world_coords_2d, line_defs, line_intersections)``.

        - ``world_coords_2d`` — list of 57 ``[x, y]`` (meters), origin at
          pitch center, y-up. Index = 0-indexed keypoint ID. The 4
          crossbar-top entries (IDs 12/14/16/18) only differ from their
          ground twins (11/13/15/17) by z, which is reflected in
          ``line_defs`` rather than this 2D table.
        - ``line_defs`` — ``{line_id: {"p1": (x,y,z), "p2": (x,y,z)}}``,
          23 lines, IDs 0-22. Goal-frame lines (6-11) carry ``z = -gh``.
        - ``line_intersections`` — ``{(lid_a, lid_b): (wx, wy, 0, kp_id)}``,
          ground-plane intersections of the 22 ground keypoints with their
          containing lines. ``kp_id`` is the keypoint that sits at this
          intersection; if it's already detected, the intersection is a
          duplicate.
    """
    d = _resolve(pitch_dims)

    L = float(d["pitch_length"])
    W = float(d["pitch_width"])
    PA_DEPTH = float(d["penalty_area_length"])
    PA_HW = float(d["penalty_area_width"]) / 2.0
    GA_DEPTH = float(d["goal_area_length"])
    GA_HW = float(d["goal_area_width"]) / 2.0
    G_HW = float(d["goal_length"]) / 2.0
    G_H = float(d["goal_height"])
    PS_DIST = float(d["goal_line_to_penalty_mark"])
    CR = float(d["center_circle_radius"])

    HL = L / 2.0
    HW = W / 2.0

    # Penalty arc ∩ penalty-area front line (only real if ccr > dx).
    dx = PA_DEPTH - PS_DIST
    radicand = CR * CR - dx * dx
    arc_dy = math.sqrt(radicand) if radicand > 0 else float("nan")

    # In a pre-centered y-down frame: top edge y=0, bottom edge y=W.
    # We compute everything there and flip y at the end (y-up centered).
    pa_top_y = HW - PA_HW
    pa_bot_y = HW + PA_HW
    ga_top_y = HW - GA_HW
    ga_bot_y = HW + GA_HW
    g_top_y = HW - G_HW
    g_bot_y = HW + G_HW

    # Penalty marks and pitch center (pre-centered y-down).
    left_ps = (PS_DIST, HW)
    right_ps = (L - PS_DIST, HW)
    center = (HL, HW)

    # Tangents from a point to a circle.
    def _circle_tangents(cx, cy, r, sx, sy):
        h2 = (sx - cx) ** 2 + (sy - cy) ** 2
        if h2 <= r * r:
            return None  # source inside
        h = math.sqrt(h2)
        th = math.acos(r / h)
        d = math.atan2(sy - cy, sx - cx)
        return (
            (cx + r * math.cos(d + th), cy + r * math.sin(d + th)),
            (cx + r * math.cos(d - th), cy + r * math.sin(d - th)),
        )

    # 38, 39 (kp 38/39 = center-circle top tangents from halfway-line
    # endpoint at pitch top). y-pick: smaller-x tangent → "left", larger-x
    # → "right". Source point (HL, 0) sits on top touchline.
    cc_top = _circle_tangents(center[0], center[1], CR, HL, 0.0)
    if cc_top is not None:
        a, b = cc_top
        cc_top_l, cc_top_r = (a, b) if a[0] < b[0] else (b, a)
    else:
        cc_top_l = cc_top_r = (float("nan"), float("nan"))
    cc_bot = _circle_tangents(center[0], center[1], CR, HL, W)
    if cc_bot is not None:
        a, b = cc_bot
        cc_bot_l, cc_bot_r = (a, b) if a[0] < b[0] else (b, a)
    else:
        cc_bot_l = cc_bot_r = (float("nan"), float("nan"))

    # Penalty-arc inner tangents (from pa front-line corners to penalty arc).
    def _tan_inner(corner, ps_pt, prefer_y_low):
        t = _circle_tangents(ps_pt[0], ps_pt[1], CR, corner[0], corner[1])
        if t is None:
            return (float("nan"), float("nan"))
        a, b = t
        if prefer_y_low:
            return a if a[1] < b[1] else b
        return a if a[1] > b[1] else b

    lpa_inner_top = _tan_inner((PA_DEPTH, pa_top_y), left_ps, prefer_y_low=False)
    lpa_inner_bot = _tan_inner((PA_DEPTH, pa_bot_y), left_ps, prefer_y_low=True)
    rpa_inner_top = _tan_inner((L - PA_DEPTH, pa_top_y), right_ps, prefer_y_low=False)
    rpa_inner_bot = _tan_inner((L - PA_DEPTH, pa_bot_y), right_ps, prefer_y_low=True)

    sq = CR / math.sqrt(2.0)

    # 57 keypoints in pre-centered y-down (0,0 = pitch top-left).
    raw = [
        # 0-2: top edge
        (0.0, 0.0), (HL, 0.0), (L, 0.0),
        # 3-6: penalty area top
        (0.0, pa_top_y), (PA_DEPTH, pa_top_y),
        (L - PA_DEPTH, pa_top_y), (L, pa_top_y),
        # 7-10: goal area top
        (0.0, ga_top_y), (GA_DEPTH, ga_top_y),
        (L - GA_DEPTH, ga_top_y), (L, ga_top_y),
        # 11-14: top goal posts (ground twin / crossbar twin)
        (0.0, g_top_y), (0.0, g_top_y),
        (L,   g_top_y), (L,   g_top_y),
        # 15-18: bottom goal posts
        (0.0, g_bot_y), (0.0, g_bot_y),
        (L,   g_bot_y), (L,   g_bot_y),
        # 19-22: goal area bottom
        (0.0, ga_bot_y), (GA_DEPTH, ga_bot_y),
        (L - GA_DEPTH, ga_bot_y), (L, ga_bot_y),
        # 23-26: penalty area bottom
        (0.0, pa_bot_y), (PA_DEPTH, pa_bot_y),
        (L - PA_DEPTH, pa_bot_y), (L, pa_bot_y),
        # 27-29: bottom edge
        (0.0, W), (HL, W), (L, W),
        # 30-35: penalty arc top tangent + center circle top + symmetric
        (PA_DEPTH,         HW - arc_dy),         # 30 left arc top
        (HL,               HW - CR),             # 31 cc top
        (L - PA_DEPTH,     HW - arc_dy),         # 32 right arc top
        (PA_DEPTH,         HW + arc_dy),         # 33 left arc bottom
        (HL,               HW + CR),             # 34 cc bottom
        (L - PA_DEPTH,     HW + arc_dy),         # 35 right arc bottom
        # 36-43: penalty arc inner + center circle 45° points
        lpa_inner_top,                            # 36
        cc_top_l,                                 # 37
        cc_top_r,                                 # 38
        rpa_inner_top,                            # 39
        lpa_inner_bot,                            # 40
        cc_bot_l,                                 # 41
        cc_bot_r,                                 # 42
        rpa_inner_bot,                            # 43
        # 44-46: left penalty arc keypoints on halfway-y line
        (left_ps[0],            left_ps[1]),               # 44 left penalty mark
        (left_ps[0] + 5.5,      left_ps[1]),               # 45 (legacy offset)
        (left_ps[0] + CR,       left_ps[1]),               # 46 arc rightmost
        # 47-53: center-circle x-axis points and 45° corners
        (center[0] - sq,        center[1] - sq),            # 47 cc TL 45°
        (center[0] + sq,        center[1] - sq),            # 48 cc TR 45°
        (center[0] - CR,        center[1]),                 # 49 cc leftmost
        (center[0],             center[1]),                 # 50 center spot
        (center[0] + CR,        center[1]),                 # 51 cc rightmost
        (center[0] - sq,        center[1] + sq),            # 52 cc BL 45°
        (center[0] + sq,        center[1] + sq),            # 53 cc BR 45°
        # 54-56: right penalty area arc points on halfway-y line
        (right_ps[0] - CR,      right_ps[1]),               # 54
        (right_ps[0] - 5.5,     right_ps[1]),               # 55
        (right_ps[0],           right_ps[1]),               # 56 right penalty mark
    ]

    # Center origin + flip y to y-up: x' = x - HL, y' = HW - y.
    world_coords_2d = [[x - HL, HW - y] for (x, y) in raw]

    # Line definitions (3D, centered, y-up).
    line_defs = {
        # left penalty area
        0:  {"p1": (-HL, PA_HW, 0), "p2": (-HL + PA_DEPTH, PA_HW, 0)},
        1:  {"p1": (-HL + PA_DEPTH, PA_HW, 0), "p2": (-HL + PA_DEPTH, -PA_HW, 0)},
        2:  {"p1": (-HL + PA_DEPTH, -PA_HW, 0), "p2": (-HL, -PA_HW, 0)},
        # right penalty area
        3:  {"p1": (HL, PA_HW, 0), "p2": (HL - PA_DEPTH, PA_HW, 0)},
        4:  {"p1": (HL - PA_DEPTH, PA_HW, 0), "p2": (HL - PA_DEPTH, -PA_HW, 0)},
        5:  {"p1": (HL - PA_DEPTH, -PA_HW, 0), "p2": (HL, -PA_HW, 0)},
        # Goal frames (z=+G_H, positive-up — matches KeypointMapper.
        # GOAL_CROSSBAR_HEIGHT and geometry.py's PITCH_POINTS).
        # left goal frame
        6:  {"p1": (-HL, -G_HW, G_H), "p2": (-HL, G_HW, G_H)},   # crossbar
        7:  {"p1": (-HL, -G_HW, 0),   "p2": (-HL, -G_HW, G_H)},  # bot-side post
        8:  {"p1": (-HL, G_HW, 0),    "p2": (-HL, G_HW, G_H)},   # top-side post
        # right goal frame
        9:  {"p1": (HL, -G_HW, G_H), "p2": (HL, G_HW, G_H)},
        10: {"p1": (HL, -G_HW, 0),   "p2": (HL, -G_HW, G_H)},
        11: {"p1": (HL, G_HW, 0),    "p2": (HL, G_HW, G_H)},
        # pitch outline
        12: {"p1": (0, -HW, 0),  "p2": (0, HW, 0)},
        13: {"p1": (-HL, HW, 0), "p2": (HL, HW, 0)},
        14: {"p1": (-HL, -HW, 0), "p2": (-HL, HW, 0)},
        15: {"p1": (HL, -HW, 0),  "p2": (HL, HW, 0)},
        16: {"p1": (-HL, -HW, 0), "p2": (HL, -HW, 0)},
        # left goal area
        17: {"p1": (-HL, GA_HW, 0), "p2": (-HL + GA_DEPTH, GA_HW, 0)},
        18: {"p1": (-HL + GA_DEPTH, GA_HW, 0), "p2": (-HL + GA_DEPTH, -GA_HW, 0)},
        19: {"p1": (-HL + GA_DEPTH, -GA_HW, 0), "p2": (-HL, -GA_HW, 0)},
        # right goal area
        20: {"p1": (HL, GA_HW, 0), "p2": (HL - GA_DEPTH, GA_HW, 0)},
        21: {"p1": (HL - GA_DEPTH, GA_HW, 0), "p2": (HL - GA_DEPTH, -GA_HW, 0)},
        22: {"p1": (HL - GA_DEPTH, -GA_HW, 0), "p2": (HL, -GA_HW, 0)},
    }

    # Ground-plane line-line intersections that coincide with a known
    # keypoint. Used to add extra correspondences when both lines fire.
    pa_front_x = HL - PA_DEPTH
    ga_front_x = HL - GA_DEPTH
    line_intersections = {
        # left PA corners
        (0, 1):   (-pa_front_x,  PA_HW, 0.0,  4),
        (1, 2):   (-pa_front_x, -PA_HW, 0.0, 24),
        (0, 14):  (-HL,          PA_HW, 0.0,  3),
        (2, 14):  (-HL,         -PA_HW, 0.0, 23),
        # right PA corners
        (3, 4):   ( pa_front_x,  PA_HW, 0.0,  5),
        (4, 5):   ( pa_front_x, -PA_HW, 0.0, 25),
        (3, 15):  ( HL,          PA_HW, 0.0,  6),
        (5, 15):  ( HL,         -PA_HW, 0.0, 26),
        # left GA corners
        (17, 18): (-ga_front_x,  GA_HW, 0.0,  8),
        (18, 19): (-ga_front_x, -GA_HW, 0.0, 20),
        (17, 14): (-HL,          GA_HW, 0.0,  7),
        (19, 14): (-HL,         -GA_HW, 0.0, 19),
        # right GA corners
        (20, 21): ( ga_front_x,  GA_HW, 0.0,  9),
        (21, 22): ( ga_front_x, -GA_HW, 0.0, 21),
        (20, 15): ( HL,          GA_HW, 0.0, 10),
        (22, 15): ( HL,         -GA_HW, 0.0, 22),
        # pitch corners
        (13, 14): (-HL,          HW,    0.0,  0),
        (13, 15): ( HL,          HW,    0.0,  2),
        (16, 14): (-HL,         -HW,    0.0, 27),
        (16, 15): ( HL,         -HW,    0.0, 29),
        # halfway line × touchlines
        (12, 13): ( 0.0,         HW,    0.0,  1),
        (12, 16): ( 0.0,        -HW,    0.0, 28),
    }

    return world_coords_2d, line_defs, line_intersections


# Keypoints that lie on each ground line, ordered by arc-length parameter t.
# Used to derive image-line endpoints from detected keypoints when the line
# detector is unavailable. Keys are 0-indexed v3 line IDs.
LINE_KEYPOINTS: dict[int, list[int]] = {
    0:  [3, 4],                          # left PA top
    1:  [4, 30, 45, 33, 24],             # left PA front
    2:  [24, 23],                        # left PA bottom
    3:  [6, 5],                          # right PA top
    4:  [5, 32, 55, 35, 25],             # right PA front
    5:  [25, 26],                        # right PA bottom
    12: [28, 34, 50, 31, 1],             # halfway line
    13: [0, 1, 2],                       # top touchline
    14: [27, 23, 19, 15, 11, 7, 3, 0],   # left goal line
    15: [29, 26, 22, 17, 13, 10, 6, 2],  # right goal line
    16: [27, 28, 29],                    # bottom touchline
    17: [7, 8],                          # left GA top
    18: [8, 20],                         # left GA front
    19: [20, 19],                        # left GA bottom
    20: [10, 9],                         # right GA top
    21: [9, 21],                         # right GA front
    22: [21, 22],                        # right GA bottom
}
