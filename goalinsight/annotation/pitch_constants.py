"""Pitch lines and active-pitch state (y-up convention).

Pitch center is the origin. X grows toward the right goal. Y grows toward the
top of the image / away from the camera (y-up). This matches the PnLCalib
world-coordinate convention used by the v3 finetune dataloader.

The active SoccerPitch and the line endpoints derived from it can be replaced
at runtime via ``set_active_pitch(SoccerPitch(...))`` — see
``goalinsight.annotation.pitch.geometry`` for the spec class. Keypoints live
in ``goalinsight.annotation.pitch.keypoints`` (57-point HRNet system).

IMPORTANT: consumers should import this *module* (``from . import
pitch_constants``) and read attributes at call time, not import them by value
(``from .pitch_constants import PITCH_LINES`` captures a stale value).
"""

import math

from .pitch.geometry import SoccerPitch


def _build_lines_for_pitch(
    pitch: SoccerPitch,
) -> dict[str, tuple[tuple[float, float], tuple[float, float]]]:
    """Build the line endpoints dict for a given pitch spec."""
    L = pitch.PITCH_LENGTH / 2.0
    W = pitch.PITCH_WIDTH / 2.0
    pa_w = pitch.PENALTY_AREA_WIDTH / 2.0
    pa_d = pitch.PENALTY_AREA_LENGTH
    ga_w = pitch.GOAL_AREA_WIDTH / 2.0
    ga_d = pitch.GOAL_AREA_LENGTH

    return {
        "touchline_top": ((-L, W), (L, W)),
        "touchline_bottom": ((-L, -W), (L, -W)),
        "goal_line_left": ((-L, W), (-L, -W)),
        "goal_line_right": ((L, W), (L, -W)),
        "center_line": ((0.0, W), (0.0, -W)),
        "penalty_left_top": ((-L, pa_w), (-L + pa_d, pa_w)),
        "penalty_left_bottom": ((-L, -pa_w), (-L + pa_d, -pa_w)),
        "penalty_left_front": ((-L + pa_d, pa_w), (-L + pa_d, -pa_w)),
        "penalty_right_top": ((L, pa_w), (L - pa_d, pa_w)),
        "penalty_right_bottom": ((L, -pa_w), (L - pa_d, -pa_w)),
        "penalty_right_front": ((L - pa_d, pa_w), (L - pa_d, -pa_w)),
        "goal_area_left_top": ((-L, ga_w), (-L + ga_d, ga_w)),
        "goal_area_left_bottom": ((-L, -ga_w), (-L + ga_d, -ga_w)),
        "goal_area_left_front": ((-L + ga_d, ga_w), (-L + ga_d, -ga_w)),
        "goal_area_right_top": ((L, ga_w), (L - ga_d, ga_w)),
        "goal_area_right_bottom": ((L, -ga_w), (L - ga_d, -ga_w)),
        "goal_area_right_front": ((L - ga_d, ga_w), (L - ga_d, -ga_w)),
    }


# Module-level state. Initialized to FIFA defaults; replace via set_active_pitch.
_active_pitch: SoccerPitch = SoccerPitch()
PITCH_LINES = _build_lines_for_pitch(_active_pitch)


def set_active_pitch(pitch: SoccerPitch) -> None:
    """Rebuild module-level pitch geometry for ``pitch``.

    Also delegates to ``annotation.pitch.keypoints.set_active_pitch`` so the
    HRNet 57-point dict and pnlcalib mapping stay in sync.
    """
    global _active_pitch, PITCH_LINES
    _active_pitch = pitch
    PITCH_LINES = _build_lines_for_pitch(pitch)

    # Sync the pitch.keypoints module too.
    from .pitch import keypoints as _kp
    _kp.set_active_pitch(pitch)


def get_active_pitch() -> SoccerPitch:
    return _active_pitch


def get_all_line_names() -> list[str]:
    return list(PITCH_LINES.keys())


def build_d_penalty_arcs(
    pitch: SoccerPitch | None = None,
    samples_per_arc: int = 48,
) -> list[list[tuple[float, float]]]:
    """Return D-shape penalty area as a list of world-space polylines.

    Single source of truth for the futsal D-region geometry. Each arc is a
    post-centered quarter-arc (radius = PENALTY_AREA_LENGTH); the two
    chord segments connect arc apices at x = ±(L - pa_d) along ±g_w.

    Output is a list of 6 polylines (4 arcs + 2 chords). Annotate-side
    rendering (``annotation/viz.py``) and pipeline-side rendering
    (``utils/pitch.py``) both consume this so the projected D shape is
    identical on both pages.
    """
    pitch = pitch if pitch is not None else _active_pitch
    L = pitch.PITCH_LENGTH / 2.0
    pa_d = pitch.PENALTY_AREA_LENGTH
    g_w = pitch.GOAL_LENGTH / 2.0

    # (center_x, center_y, start_deg, end_deg). Standard math angles (CCW
    # from +x) in the y-up world frame. The sweep direction matters: each
    # quarter-arc must walk from the goal-line side (radius along ±x) to
    # the chord-apex side, staying inside the pitch.
    arc_specs = [
        # Left side, top post & bottom post.
        (-L, +g_w, 90.0, 0.0),
        (-L, -g_w, 0.0, -90.0),
        # Right side, top post & bottom post.
        (+L, +g_w, 90.0, 180.0),
        (+L, -g_w, 180.0, 270.0),
    ]
    polylines: list[list[tuple[float, float]]] = []
    for cx, cy, a0, a1 in arc_specs:
        arc: list[tuple[float, float]] = []
        for k in range(samples_per_arc):
            t = math.radians(a0) + (
                math.radians(a1) - math.radians(a0)
            ) * k / (samples_per_arc - 1)
            arc.append((cx + pa_d * math.cos(t), cy + pa_d * math.sin(t)))
        polylines.append(arc)
    # Chord segments at x = ±(L - pa_d) connecting (-g_w) and (+g_w).
    for sign in (-1.0, 1.0):
        chord_x = sign * (L - pa_d)
        polylines.append([(chord_x, -g_w), (chord_x, +g_w)])
    return polylines
