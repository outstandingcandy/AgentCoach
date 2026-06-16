"""Shared utilities for event detectors."""

from __future__ import annotations

import math
from collections.abc import Callable
from typing import Any

from .._context import EventDetectionContext


def find_nearest_player(
    ctx: EventDetectionContext,
    frame: int,
    ball_pos: list[float] | tuple[float, float],
    *,
    team_filter: str | None = None,
    exclude_referee: bool = True,
    exclude_fn: Callable[[dict[str, Any], list[float]], bool] | None = None,
) -> tuple[int | str | None, str | None, float]:
    """Find the nearest player to a ball position at a given frame.

    Args:
        ctx: Event detection context with player/team data.
        frame: Frame number to query.
        ball_pos: Ball pitch position [x, y].
        team_filter: If set, only consider players on this team.
        exclude_referee: If True, skip players classified as referees.
        exclude_fn: Optional predicate ``(player_dict, pitch_position) -> bool``.
            If it returns True for a player, that player is skipped.

    Returns:
        ``(track_id, team_id, distance)`` — ``track_id`` is an int when
        running on raw tracker output, a consolidated player_id string
        (``"A-7"``) when running after track_consolidation. Returns
        ``(None, None, inf)`` if no player qualifies.
    """
    players = ctx.get_players_at_frame(frame)
    best_id: int | str | None = None
    best_team: str | None = None
    best_dist = float("inf")

    bx, by = ball_pos[0], ball_pos[1]

    for p in players:
        pp = p.get("pitch_position")
        if pp is None:
            continue
        tid = p["track_id"]
        team = ctx.get_team_for_track(tid)
        if exclude_referee and team == "referee":
            continue
        if team_filter is not None and team != team_filter:
            continue
        if exclude_fn is not None and exclude_fn(p, pp):
            continue
        dist = math.hypot(pp[0] - bx, pp[1] - by)
        if dist < best_dist:
            best_dist = dist
            best_id = tid
            best_team = team

    return best_id, best_team, best_dist
