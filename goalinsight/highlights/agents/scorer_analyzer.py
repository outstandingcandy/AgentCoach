"""ScorerAnalyzer — player attribution and segment planning for goal events."""

from __future__ import annotations

import logging
from typing import Any

from .._context import MatchContext
from .._temporal import find_buildup_start, find_celebration_end
from .._types import AnalyzedEvent, ClipSegment, Event
from .base import BaseSceneAnalyzer

logger = logging.getLogger(__name__)


class ScorerAnalyzer(BaseSceneAnalyzer):
    """Analyze goal events: identify scorer and plan clip segments."""

    supported_events = ["goal"]

    def analyze(
        self,
        event: Event,
        ctx: MatchContext,
        config: dict[str, Any],
    ) -> AnalyzedEvent:
        goal_frame = event.frame
        goal_side = event.metadata.get("goal_side", "right")
        fps = ctx.fps

        temporal_cfg = config.get("temporal", {})
        buildup_max = temporal_cfg.get("buildup_max_seconds", 10.0)
        buildup_pad = temporal_cfg.get("buildup_padding_seconds", 2.0)
        celebration_dur = temporal_cfg.get("celebration_seconds", 4.0)

        # --- Player attribution (from event detector) ---
        event_player = event.metadata.get("player_id")
        event_team = event.metadata.get("team_id")
        if event_player is None:
            raise ValueError(
                f"Goal event at frame {goal_frame} has no player_id — "
                "run event_detection stage first to get shooter attribution"
            )

        shooter_frame = event.metadata.get("shooter_frame", goal_frame)
        scorer = {
            "track_id": event_player,
            "role": "scorer",
            "team": event_team or "unknown",
            "distance": 0.0,
            "frame": shooter_frame,
        }
        logger.info(
            "Scorer from event detector: track_id=%s, team=%s",
            event_player, event_team,
        )
        key_players: list[dict[str, Any]] = [scorer]

        # --- Temporal windows ---
        buildup_start = find_buildup_start(
            ctx, goal_frame, goal_side,
            max_seconds=buildup_max, padding_seconds=buildup_pad,
        )

        scorer_id = scorer["track_id"]
        celebration_start = goal_frame
        celebration_max_end = find_celebration_end(
            celebration_start, fps,
            duration_seconds=celebration_dur,
            total_frames=ctx.frame_count or None,
        )

        # Truncate celebration when scorer track is lost
        scorer_trajectory = ctx.get_player_trajectory(
            scorer_id, celebration_start, celebration_max_end,
        )
        if scorer_trajectory:
            last_seen = max(t["frame"] for t in scorer_trajectory)
            celebration_end = min(celebration_max_end, last_seen)
            if celebration_end < celebration_max_end:
                logger.info(
                    "Celebration truncated: scorer track %s lost at frame %d "
                    "(max was %d)",
                    scorer_id, last_seen, celebration_max_end,
                )
        else:
            celebration_end = celebration_max_end

        # --- Segment plan ---
        # Buildup extends past goal frame; celebration starts at goal frame.
        # The two segments overlap — buildup shows the ball entering the net,
        # celebration tracks the scorer from the goal moment.
        buildup_ext = int(temporal_cfg.get("buildup_extension_seconds", 3.0) * fps)
        buildup_end = min(goal_frame + buildup_ext, celebration_end)

        segments: list[ClipSegment] = []

        # 1. Build-up → goal moment + extension: follow the ball
        buildup_view = temporal_cfg.get("buildup_view", "medium")
        segments.append(
            ClipSegment(
                name="buildup",
                start_frame=buildup_start,
                end_frame=buildup_end,
                view_type=buildup_view,
                focus_target="ball",
            )
        )

        # 2. Celebration: follow the scorer from goal frame
        celebration_view = temporal_cfg.get("celebration_view", "medium")
        if celebration_start <= celebration_end:
            segments.append(
                ClipSegment(
                    name="celebration",
                    start_frame=celebration_start,
                    end_frame=celebration_end,
                    view_type=celebration_view,
                    focus_target="player",
                    focus_track_id=scorer_id,
                )
            )

        return AnalyzedEvent(
            event=event,
            key_players=key_players,
            segments=segments,
        )

