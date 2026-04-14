"""ScorerAnalyzer — player attribution and segment planning for goal events.

Produces a broadcast-style 4-segment highlight:

    buildup → strike → celebration → replay

- **Buildup**: wide/medium view following the ball through the attacking play.
- **Strike**: closeup on the ball from the moment of the kick through the
  ball entering the net — the narrative climax.
- **Celebration**: medium view tracking the scorer's reaction.
- **Replay**: slow-motion replay of the strike with ball trail overlay.
"""

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
        scorer_id = scorer["track_id"]

        # --- Temporal boundaries ---
        buildup_max = temporal_cfg.get("buildup_max_seconds", 10.0)
        buildup_pad = temporal_cfg.get("buildup_padding_seconds", 2.0)
        buildup_start = find_buildup_start(
            ctx, goal_frame, goal_side,
            max_seconds=buildup_max, padding_seconds=buildup_pad,
        )

        # Strike window: the shot and ball flight into the net.
        # Starts 0.5s before the kick (minimal pre-read so the closeup is
        # already locked on when the ball leaves the foot), runs until the
        # ball is in the net + post buffer.
        strike_pre = int(temporal_cfg.get("strike_pre_seconds", 0.5) * fps)
        strike_post = int(temporal_cfg.get("strike_post_seconds", 1.5) * fps)
        strike_start = max(0, shooter_frame - strike_pre)
        strike_end = goal_frame + strike_post

        # Buildup runs until the kick — hard cut so the zoom change
        # coincides with the ball being struck (maximum dramatic impact).
        buildup_end = shooter_frame

        # Celebration window
        celebration_dur = temporal_cfg.get("celebration_seconds", 10.0)
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

        # --- Segment plan: buildup → strike → celebration → replay ---
        segments: list[ClipSegment] = []
        min_segment_frames = int(1.0 * fps)

        # 1. Build-up: wide/medium view following the ball
        buildup_view = temporal_cfg.get("buildup_view", "wide")
        if buildup_end - buildup_start >= min_segment_frames:
            segments.append(
                ClipSegment(
                    name="buildup",
                    start_frame=buildup_start,
                    end_frame=buildup_end,
                    view_type=buildup_view,
                    focus_target="ball",
                    transition="cut",
                )
            )

        # 2. Strike: closeup following the ball through the shot.
        #    Hard cut from buildup — the zoom change hits at the same
        #    instant the ball is kicked (no crossfade blurring the moment).
        strike_view = temporal_cfg.get("strike_view", "closeup")
        segments.append(
            ClipSegment(
                name="strike",
                start_frame=strike_start,
                end_frame=strike_end,
                view_type=strike_view,
                focus_target="ball",
                transition="cut",
            )
        )

        # 3. Celebration: medium view following the scorer
        celebration_view = temporal_cfg.get("celebration_view", "medium")
        if celebration_end - celebration_start >= min_segment_frames:
            segments.append(
                ClipSegment(
                    name="celebration",
                    start_frame=celebration_start,
                    end_frame=celebration_end,
                    view_type=celebration_view,
                    focus_target="player",
                    focus_track_id=scorer_id,
                    transition="flash",
                )
            )

        # 4. Replay: slow-motion of the strike
        replay_enabled = temporal_cfg.get("replay_enabled", True)
        replay_speed = temporal_cfg.get("replay_speed", 0.4)
        if replay_enabled:
            segments.append(
                ClipSegment(
                    name="replay",
                    start_frame=strike_start,
                    end_frame=strike_end,
                    view_type=strike_view,
                    focus_target="ball",
                    transition="crossfade",
                    speed=replay_speed,
                    overlays=[
                        {"type": "text", "text": "REPLAY",
                         "position": "top_center"},
                    ],
                )
            )

        return AnalyzedEvent(
            event=event,
            key_players=key_players,
            segments=segments,
        )

