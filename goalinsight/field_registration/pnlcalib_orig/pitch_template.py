"""Parametric upstream PnLCalib pitch tables.

Upstream's `utils_calib.py` hardcodes 57 keypoint world coords + 16 aux
keypoint coords + 23 line endpoints for FIFA-spec dimensions
(105 × 68 × 2.44). For non-FIFA pitches (e.g. kids_soccer 66.28 × 43.15 ×
2.15) those literals are wrong.

``build_keypoint_table(pitch_dims)`` reconstructs all three tables from
``SoccerPitch`` so a single ``FramebyFrameCalib`` can switch pitch sizes
at runtime. Coordinates use upstream's convention: **centered at pitch
center, y-down** (top of pitch = small y, then negated by W/2 to get
y_centered ∈ [-W/2, +W/2] with negative = top).

Crossbar-top points (upstream 1-indexed IDs ``{12, 15, 16, 19}``) carry
``z = -goal_height``; everything else is on the ground plane (``z = 0``).

Returned dict::

    {
        "keypoint_world_coords_2D":      list[(x, y)]   # 57 entries, 1-indexed
        "keypoint_aux_world_coords_2D":  list[(x, y)]   # 16 entries, 1-indexed within aux
        "line_world_coords_3D":          list[((x,y,z), (x,y,z))]  # 23 entries
        "non_ground_ids":                set[int]       # crossbar tops, z!=0
    }

The aux table is concatenated to the main table at runtime (upstream IDs
58-73 = aux 1-16). NaN is emitted for the four penalty-arc-front
intersections (IDs 31, 33, 34, 36) when ``ccr² < (pa_length − gltpm)²``
(arc doesn't reach the front line — happens on small kids pitches).
``coords_to_dict`` and ``get_keypoints_subsets`` both NaN-skip.
"""

from __future__ import annotations

import numpy as np

# FIFA-spec defaults inlined here to avoid importing
# ``goalinsight.annotation.pitch.geometry`` at module load — that path
# triggers ``goalinsight.annotation.__init__``, which itself imports
# ``annotation.homography``, which imports this package. Lazy SoccerPitch
# import below breaks the cycle.
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


# Crossbar-top upstream IDs (1-indexed). These are the *only* z != 0 entries
# in the main 57-keypoint table; everything else is on the ground plane.
NON_GROUND_IDS: set[int] = {12, 15, 16, 19}


def _circle_tangents(
    center: tuple[float, float], radius: float, source: tuple[float, float]
) -> tuple[tuple[float, float], tuple[float, float]] | None:
    """Two tangent points on a circle from an external source point.

    Returns None if ``source`` lies inside the circle (no real tangents).
    """
    dx = source[0] - center[0]
    dy = source[1] - center[1]
    h = float(np.hypot(dx, dy))
    if h <= radius:
        return None
    th = float(np.arccos(radius / h))
    d = float(np.arctan2(dy, dx))
    t1 = (center[0] + radius * np.cos(d + th), center[1] + radius * np.sin(d + th))
    t2 = (center[0] + radius * np.cos(d - th), center[1] + radius * np.sin(d - th))
    return t1, t2


def build_keypoint_table(pitch_dims: dict | None = None) -> dict:
    """Build upstream-format pitch tables from ``pitch_dims``.

    ``pitch_dims`` maps SoccerPitch param names to floats. Missing keys
    fall back to FIFA defaults. Pass ``None`` (or an empty dict) for a
    FIFA-spec pitch — output then matches upstream's literal table
    bit-for-bit.
    """
    if pitch_dims is None:
        pitch_dims = {}
    # Lazy import: see module docstring on circular import.
    from goalinsight.annotation.pitch.geometry import SoccerPitch
    pitch = SoccerPitch.from_dict({**FIFA_DEFAULTS, **pitch_dims})

    L = pitch.PITCH_LENGTH
    W = pitch.PITCH_WIDTH
    pa_d = pitch.PENALTY_AREA_LENGTH
    pa_w = pitch.PENALTY_AREA_WIDTH
    ga_d = pitch.GOAL_AREA_LENGTH
    ga_w = pitch.GOAL_AREA_WIDTH
    gl = pitch.GOAL_LENGTH
    gh = pitch.GOAL_HEIGHT
    gltpm = pitch.GOAL_LINE_TO_PENALTY_MARK
    ccr = pitch.CENTER_CIRCLE_RADIUS

    # Y values along the (pre-centered, y-down) axis.
    T0 = (W - pa_w) / 2.0      # top of penalty area
    B0 = (W + pa_w) / 2.0      # bottom of penalty area
    TG = (W - ga_w) / 2.0      # top of goal area
    BG = (W + ga_w) / 2.0      # bottom of goal area
    TP = (W - gl) / 2.0        # top goal post (small y)
    BP = (W + gl) / 2.0        # bottom goal post (large y)
    HW = W / 2.0
    HL = L / 2.0

    # Penalty arc ∩ pa front line (only real if ccr > dx).
    dx = pa_d - gltpm
    radicand = ccr * ccr - dx * dx
    arc_dy = float(np.sqrt(radicand)) if radicand > 0 else float("nan")

    # Tangents from the touchline-halfway intersection to the center circle.
    # Both source points sit on the halfway line at y = 0 (top) or y = W (bottom).
    cc_top_tangents = _circle_tangents((HL, HW), ccr, (HL, 0.0))
    cc_bot_tangents = _circle_tangents((HL, HW), ccr, (HL, W))
    # Pick by x: the tangent with x > HL is the "right" tangent.
    if cc_top_tangents is not None:
        a, b = cc_top_tangents
        cc_top_r, cc_top_l = (a, b) if a[0] > b[0] else (b, a)
    else:
        cc_top_r = cc_top_l = (float("nan"), float("nan"))
    if cc_bot_tangents is not None:
        a, b = cc_bot_tangents
        cc_bot_r, cc_bot_l = (a, b) if a[0] > b[0] else (b, a)
    else:
        cc_bot_r = cc_bot_l = (float("nan"), float("nan"))

    # Tangents from penalty-area corners to the penalty arc (centered at
    # penalty mark).
    l_pen_mark = (gltpm, HW)
    r_pen_mark = (L - gltpm, HW)

    def _tan_select(tangents, prefer_y_low: bool):
        """Pick the tangent whose y is lower (prefer_y_low=True) or higher."""
        if tangents is None:
            return (float("nan"), float("nan"))
        a, b = tangents
        if prefer_y_low:
            return a if a[1] < b[1] else b
        return a if a[1] > b[1] else b

    # Left penalty arc tangents from L_PA_TR (top corner, small y) and
    # L_PA_BR (bottom corner, large y). The "inner top" tangent is the one
    # closer to the halfway line — i.e. lower y from the top corner.
    lpa_inner_top = _tan_select(_circle_tangents(l_pen_mark, ccr, (pa_d, T0)), prefer_y_low=False)
    lpa_inner_bot = _tan_select(_circle_tangents(l_pen_mark, ccr, (pa_d, B0)), prefer_y_low=True)
    rpa_inner_top = _tan_select(_circle_tangents(r_pen_mark, ccr, (L - pa_d, T0)), prefer_y_low=False)
    rpa_inner_bot = _tan_select(_circle_tangents(r_pen_mark, ccr, (L - pa_d, B0)), prefer_y_low=True)

    # Center-circle 45° points (relative to pitch center, in y-down frame).
    sq = ccr / float(np.sqrt(2.0))
    cc_diag_tl = (HL - sq, HW - sq)   # TL in y-down: x<HL, y<HW
    cc_diag_tr = (HL + sq, HW - sq)
    cc_diag_bl = (HL - sq, HW + sq)
    cc_diag_br = (HL + sq, HW + sq)

    # 1-indexed main table (57 entries). All in pre-centered y-down.
    main_pre = [
        (0.0, 0.0),                  # 1  TL pitch corner
        (HL, 0.0),                   # 2  T center (touchline)
        (L, 0.0),                    # 3  TR pitch corner
        (0.0, T0),                   # 4  L pa top corner
        (pa_d, T0),                  # 5  L pa top front
        (L - pa_d, T0),              # 6  R pa top front
        (L, T0),                     # 7  R pa top corner
        (0.0, TG),                   # 8  L ga top corner
        (ga_d, TG),                  # 9  L ga top front
        (L - ga_d, TG),              # 10 R ga top front
        (L, TG),                     # 11 R ga top corner
        (0.0, TP),                   # 12 L crossbar TR end (z=-gh)
        (0.0, TP),                   # 13 L post (top side, ground)
        (L, TP),                     # 14 R post (top side, ground)
        (L, TP),                     # 15 R crossbar TL end (z=-gh)
        (0.0, BP),                   # 16 L crossbar BR end (z=-gh)
        (0.0, BP),                   # 17 L post (bottom side, ground)
        (L, BP),                     # 18 R post (bottom side, ground)
        (L, BP),                     # 19 R crossbar BL end (z=-gh)
        (0.0, BG),                   # 20 L ga bottom corner
        (ga_d, BG),                  # 21 L ga bottom front
        (L - ga_d, BG),              # 22 R ga bottom front
        (L, BG),                     # 23 R ga bottom corner
        (0.0, B0),                   # 24 L pa bottom corner
        (pa_d, B0),                  # 25 L pa bottom front
        (L - pa_d, B0),              # 26 R pa bottom front
        (L, B0),                     # 27 R pa bottom corner
        (0.0, W),                    # 28 BL pitch corner
        (HL, W),                     # 29 B center
        (L, W),                      # 30 BR pitch corner
        # Penalty-arc ∩ penalty-area front line.
        (pa_d, HW - arc_dy),         # 31 L arc top tangent on pa_d front
        (HL, HW - ccr),              # 32 center circle top tangent
        (L - pa_d, HW - arc_dy),     # 33 R arc top tangent
        (pa_d, HW + arc_dy),         # 34 L arc bottom tangent
        (HL, HW + ccr),              # 35 center circle bottom tangent
        (L - pa_d, HW + arc_dy),     # 36 R arc bottom tangent
        # Penalty-arc tangents from pa corners (inner top/bottom).
        lpa_inner_top,               # 37 L pa arc inner top  (y < HW)
        cc_top_l,                    # 38 center circle top tangent left
        cc_top_r,                    # 39 center circle top tangent right
        rpa_inner_top,               # 40 R pa arc inner top
        lpa_inner_bot,               # 41 L pa arc inner bottom (y > HW)
        cc_bot_l,                    # 42 center circle bot tangent left
        cc_bot_r,                    # 43 center circle bot tangent right
        rpa_inner_bot,               # 44 R pa arc inner bottom
        # Penalty mark + penalty-arc keypoints on halfway-y line.
        (gltpm, HW),                 # 45 L penalty mark
        (pa_d, HW),                  # 46 L pa front line center (mid-y)
        (gltpm + ccr, HW),           # 47 L pa arc rightmost point (on x-axis)
        # Center-circle 45° points (TL, TR, BL, BR).
        cc_diag_tl,                  # 48 cc top-left 45°
        cc_diag_tr,                  # 49 cc top-right 45°
        # Center circle on x-axis & center spot.
        (HL - ccr, HW),              # 50 cc leftmost
        (HL, HW),                    # 51 center spot
        (HL + ccr, HW),              # 52 cc rightmost
        cc_diag_bl,                  # 53 cc bottom-left 45°
        cc_diag_br,                  # 54 cc bottom-right 45°
        # Right penalty area arc keypoints on halfway-y line.
        (L - gltpm - ccr, HW),       # 55 R pa arc leftmost
        (L - pa_d, HW),              # 56 R pa front line center
        (L - gltpm, HW),             # 57 R penalty mark
    ]

    # Center to (0,0) origin and y-flip-free (still y-down, just centered).
    main_centered = [(x - HL, y - HW) for x, y in main_pre]

    # Aux 16 entries (1-indexed within aux; concatenated to main as 58-73).
    aux_pre = [
        (ga_d, 0.0),                 # 58
        (pa_d, 0.0),                 # 59
        (L - pa_d, 0.0),             # 60
        (L - ga_d, 0.0),             # 61
        (ga_d, T0),                  # 62
        (L - ga_d, T0),              # 63
        (pa_d, TG),                  # 64
        (L - pa_d, TG),              # 65
        (pa_d, BG),                  # 66
        (L - pa_d, BG),              # 67
        (ga_d, B0),                  # 68
        (L - ga_d, B0),              # 69
        (ga_d, W),                   # 70
        (pa_d, W),                   # 71
        (L - pa_d, W),               # 72
        (L - ga_d, W),               # 73
    ]
    aux_centered = [(x - HL, y - HW) for x, y in aux_pre]

    # Lines 1-23 (3D, centered, y-down). Goal-frame uses z = -gh.
    lines_pre = [
        # 1-3: Left penalty area lines (excluding L goal line).
        ((0.0, B0, 0.0), (pa_d, B0, 0.0)),         # 1 L pa bottom edge
        ((pa_d, T0, 0.0), (pa_d, B0, 0.0)),        # 2 L pa front line
        ((pa_d, T0, 0.0), (0.0, T0, 0.0)),         # 3 L pa top edge
        # 4-6: Right penalty area lines.
        ((L - pa_d, B0, 0.0), (L, B0, 0.0)),       # 4 R pa bottom edge
        ((L - pa_d, T0, 0.0), (L - pa_d, B0, 0.0)),# 5 R pa front line
        ((L - pa_d, T0, 0.0), (L, T0, 0.0)),       # 6 R pa top edge
        # 7-9: Left goal frame.
        ((0.0, BP, -gh), (0.0, TP, -gh)),          # 7 L crossbar (top)
        ((0.0, BP, 0.0), (0.0, BP, -gh)),          # 8 L bottom-side post
        ((0.0, TP, 0.0), (0.0, TP, -gh)),          # 9 L top-side post
        # 10-12: Right goal frame.
        ((L, BP, -gh), (L, TP, -gh)),              # 10 R crossbar
        ((L, TP, 0.0), (L, TP, -gh)),              # 11 R top-side post
        ((L, BP, 0.0), (L, BP, -gh)),              # 12 R bottom-side post
        # 13-17: pitch outline.
        ((HL, 0.0, 0.0), (HL, W, 0.0)),            # 13 halfway line
        ((0.0, W, 0.0), (L, W, 0.0)),              # 14 bottom touchline
        ((0.0, 0.0, 0.0), (0.0, W, 0.0)),          # 15 left goal line
        ((L, 0.0, 0.0), (L, W, 0.0)),              # 16 right goal line
        ((0.0, 0.0, 0.0), (L, 0.0, 0.0)),          # 17 top touchline
        # 18-20: Left goal area lines.
        ((0.0, BG, 0.0), (ga_d, BG, 0.0)),         # 18 L ga bottom edge
        ((ga_d, BG, 0.0), (ga_d, TG, 0.0)),        # 19 L ga front line
        ((ga_d, TG, 0.0), (0.0, TG, 0.0)),         # 20 L ga top edge
        # 21-23: Right goal area lines.
        ((L - ga_d, BG, 0.0), (L, BG, 0.0)),       # 21 R ga bottom edge
        ((L - ga_d, BG, 0.0), (L - ga_d, TG, 0.0)),# 22 R ga front line
        ((L - ga_d, TG, 0.0), (L, TG, 0.0)),       # 23 R ga top edge
    ]
    lines_centered = [
        ((x1 - HL, y1 - HW, z1), (x2 - HL, y2 - HW, z2))
        for ((x1, y1, z1), (x2, y2, z2)) in lines_pre
    ]

    return {
        "keypoint_world_coords_2D": main_centered,
        "keypoint_aux_world_coords_2D": aux_centered,
        "line_world_coords_3D": lines_centered,
        "non_ground_ids": set(NON_GROUND_IDS),
        "pitch_length": L,
        "pitch_width": W,
        "goal_height": gh,
    }
