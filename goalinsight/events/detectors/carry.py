"""CarryDetector — detects ball carries (dribbles with forward progress)."""

from __future__ import annotations

import logging
import math
from collections import defaultdict
from typing import Any

from .._base import BaseEventDetector
from .._context import EventDetectionContext
from .._registry import register_detector
from .._types import EventType, MatchEvent

logger = logging.getLogger(__name__)


@register_detector
class CarryDetector(BaseEventDetector):
    """Detect carries: sustained possession with significant displacement."""

    name = "carry"
    depends_on = ["possession"]

    def detect(
        self, ctx: EventDetectionContext, config: dict[str, Any]
    ) -> list[MatchEvent]:
        cfg = config.get("events", {}).get("carry", {})
        min_duration_sec = cfg.get("min_duration_seconds", 2.0)
        min_total_dist = cfg.get("min_total_distance", 5.0)
        min_forward_dist = cfg.get("min_forward_distance", 3.0)

        # Determine attacking direction per team
        attack_dir = self._infer_attacking_directions(ctx)

        events: list[MatchEvent] = []
        seq = 0

        for span in ctx.possession_spans:
            duration_sec = (span.end_frame - span.start_frame) / ctx.fps
            if duration_sec < min_duration_sec:
                continue

            # Compute player displacement during possession
            player_start = self._get_player_position(
                ctx, span.player_id, span.start_frame
            )
            player_end = self._get_player_position(
                ctx, span.player_id, span.end_frame
            )
            if player_start is None or player_end is None:
                continue

            dx = player_end[0] - player_start[0]
            dy = player_end[1] - player_start[1]
            total_dist = math.hypot(dx, dy)

            # Forward distance: displacement in the attacking direction
            sign = attack_dir.get(span.team_id, 1)
            forward_dist = dx * sign

            if total_dist >= min_total_dist or forward_dist >= min_forward_dist:
                seq += 1
                events.append(
                    MatchEvent(
                        event_id=f"carry_{seq:04d}",
                        event_type=EventType.CARRY,
                        frame=span.start_frame,
                        match_time=span.start_frame / ctx.fps,
                        player_id=span.player_id,
                        team_id=span.team_id,
                        start_frame=span.start_frame,
                        end_frame=span.end_frame,
                        start_position=player_start,
                        end_position=player_end,
                        confidence=1.0,
                        metadata={
                            "distance_m": round(total_dist, 2),
                            "forward_distance_m": round(forward_dist, 2),
                        },
                    )
                )

        logger.info("CarryDetector: %d carry(ies) detected", len(events))
        return events

    @staticmethod
    def _infer_attacking_directions(
        ctx: EventDetectionContext,
    ) -> dict[str, int]:
        """Infer which direction (+x or -x) each team attacks.

        Heuristic: team with lower median x is attacking right (+1),
        team with higher median x is attacking left (-1).

        Returns dict mapping team_id → sign (+1 or -1).
        """
        team_xs: dict[str, list[float]] = defaultdict(list)

        # Sample player positions from a subset of frames
        frames = sorted(ctx.player_tracks.keys(), key=int)
        sample_step = max(1, len(frames) // 50)

        for frame_str in frames[::sample_step]:
            for p in ctx.player_tracks[frame_str]:
                pp = p.get("pitch_position")
                if pp is None:
                    continue
                team = ctx.get_team_for_track(p["track_id"])
                if team in ("referee", "unknown"):
                    continue
                team_xs[team].append(pp[0])

        if len(team_xs) < 2:
            return {}

        # Sort teams by median x
        teams_sorted = sorted(
            team_xs.keys(),
            key=lambda t: (
                sorted(team_xs[t])[len(team_xs[t]) // 2]
                if team_xs[t]
                else 0
            ),
        )

        # Lower-x team attacks right, higher-x team attacks left
        return {teams_sorted[0]: 1, teams_sorted[1]: -1}

    @staticmethod
    def _get_player_position(
        ctx: EventDetectionContext,
        track_id: int | str,
        frame: int,
        search_radius: int = 5,
    ) -> list[float] | None:
        """Get a player's pitch position at or near a frame."""
        for offset in range(search_radius + 1):
            for f in (frame + offset, frame - offset):
                if f < 0:
                    continue
                for p in ctx.get_players_at_frame(f):
                    if p["track_id"] == track_id:
                        pp = p.get("pitch_position")
                        if pp is not None:
                            return pp
        return None
