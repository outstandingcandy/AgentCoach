"""Temporal windowing utilities for highlight clipping."""

from __future__ import annotations

import logging

from ._context import MatchContext

logger = logging.getLogger(__name__)


def find_buildup_start(
    ctx: MatchContext,
    goal_frame: int,
    goal_side: str,
    max_seconds: float = 10.0,
    padding_seconds: float = 2.0,
) -> int:
    """Find the start frame of the build-up play before a goal.

    Walks backwards from *goal_frame* looking for:
    1. Ball crossing the halfway line (x sign change), or
    2. Reaching the maximum lookback.

    Then applies a padding before that point.

    Returns:
        Start frame for the build-up segment (clamped to >= 0).
    """
    fps = ctx.fps
    max_lookback = int(max_seconds * fps)
    padding_frames = int(padding_seconds * fps)
    earliest = max(0, goal_frame - max_lookback)

    # Walk backwards through ball observations
    buildup_frame = earliest
    for f in range(goal_frame - 1, earliest - 1, -1):
        ball = ctx.get_ball_at_frame(f)
        if ball is None:
            continue
        pp = ball.get("pitch_position")
        if pp is None:
            continue
        x = pp[0]

        # If ball is on the other side of the halfway line, the build-up
        # started around here.
        if goal_side == "right" and x < 0:
            buildup_frame = f
            break
        if goal_side == "left" and x > 0:
            buildup_frame = f
            break
    else:
        # Never crossed halfway — use the earliest frame with ball data
        buildup_frame = earliest

    return max(0, buildup_frame - padding_frames)


def find_celebration_end(
    goal_frame: int,
    fps: float,
    duration_seconds: float = 4.0,
    total_frames: int | None = None,
) -> int:
    """Return the end frame of the celebration window.

    Simply extends *duration_seconds* after the goal frame, clamped to
    total_frames if provided.
    """
    end = goal_frame + int(duration_seconds * fps)
    if total_frames is not None:
        end = min(end, total_frames - 1)
    return end
