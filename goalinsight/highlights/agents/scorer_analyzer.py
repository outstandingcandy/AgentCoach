"""ScorerAnalyzer — player attribution and segment planning for goal events.

Produces a two-pass broadcast highlight of the same play:

    Pass 1 (original / wide view)  →  Pass 2 (closeup replay)

- **Pass 1 — original view**: the whole play (buildup → shot → ball into
  the net → celebration) shown once in the uncropped broadcast angle, no
  camera-follow. This is the "as it happened" pass.
- **Pass 2 — closeup replay**: the same play again, zoomed in with the
  camera following the action — whoever is on the ball (passer → shooter),
  then the ball flying into the net, then the scorer's celebration.

The celebration is hard-capped at 10 s after the shot in either pass.
No ball-trail effect is drawn; the ball stays visible because the closeup
pass tracks it directly through the flight.
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

        # Celebration window. Hard-cap at 10 s *after the shot* (the kick,
        # shooter_frame) regardless of the configured celebration length —
        # the requirement is "celebration no longer than 10 s post-shot".
        celebration_dur = temporal_cfg.get("celebration_seconds", 10.0)
        celebration_start = goal_frame
        celebration_max_end = find_celebration_end(
            celebration_start, fps,
            duration_seconds=celebration_dur,
            total_frames=ctx.frame_count or None,
        )
        post_shot_cap = shooter_frame + int(10.0 * fps)
        if celebration_max_end > post_shot_cap:
            logger.info(
                "Celebration capped to 10s after shot: %d → %d",
                celebration_max_end, post_shot_cap,
            )
            celebration_max_end = post_shot_cap

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

        # --- Two-pass segment plan ---
        # Pass 1: the whole play in the original (wide, uncropped) view —
        #         "as it happened", no camera-follow.
        # Pass 2: the same play again, zoomed in with the camera following
        #         the action (possession → shooter → ball → scorer).
        segments: list[ClipSegment] = []
        min_segment_frames = int(1.0 * fps)
        play_end = max(strike_end, celebration_end)

        # ---- Pass 1: original view (single continuous wide shot) ----
        segments.append(
            ClipSegment(
                name="original",
                start_frame=buildup_start,
                end_frame=play_end,
                view_type="wide",
                focus_target="ball",  # unused in wide view (full frame)
                transition="cut",
            )
        )

        # ---- Pass 2: closeup replay following the action ----
        # Each sub-segment carries a REPLAY overlay so the repeat reads as
        # a replay, not a continuity error. The first sub-segment crossfades
        # in to separate the two passes.
        buildup_view = temporal_cfg.get("buildup_view", "medium")
        strike_view = temporal_cfg.get("strike_view", "closeup")
        celebration_view = temporal_cfg.get("celebration_view", "medium")
        replay_overlay = [
            {"type": "text", "text": "REPLAY", "position": "top_center"},
        ]

        first_pass2 = True

        def _pass2_transition() -> str:
            nonlocal first_pass2
            t = "crossfade" if first_pass2 else "cut"
            first_pass2 = False
            return t

        # Build-up: follow whoever is on the ball (passer → shooter).
        if buildup_end - buildup_start >= min_segment_frames:
            segments.append(
                ClipSegment(
                    name="replay_buildup",
                    start_frame=buildup_start,
                    end_frame=buildup_end,
                    view_type=buildup_view,
                    focus_target="possession",
                    transition=_pass2_transition(),
                    overlays=replay_overlay,
                )
            )

        # Strike wind-up: lock the closeup on the shooter until the ball
        # leaves the foot.
        if shooter_frame - strike_start >= 1:
            segments.append(
                ClipSegment(
                    name="replay_strike",
                    start_frame=strike_start,
                    end_frame=shooter_frame,
                    view_type=strike_view,
                    focus_target="player",
                    focus_track_id=scorer_id,
                    transition=_pass2_transition(),
                    overlays=replay_overlay,
                )
            )

        # Flight: track the ball from the kick into the net.
        segments.append(
            ClipSegment(
                name="replay_flight",
                start_frame=shooter_frame,
                end_frame=strike_end,
                view_type=strike_view,
                focus_target="ball",
                transition=_pass2_transition(),
                overlays=replay_overlay,
            )
        )

        # Celebration: follow the scorer (already capped to 10 s post-shot).
        if celebration_end - celebration_start >= min_segment_frames:
            segments.append(
                ClipSegment(
                    name="replay_celebration",
                    start_frame=celebration_start,
                    end_frame=celebration_end,
                    view_type=celebration_view,
                    focus_target="player",
                    focus_track_id=scorer_id,
                    transition=_pass2_transition(),
                    overlays=replay_overlay,
                )
            )

        return AnalyzedEvent(
            event=event,
            key_players=key_players,
            segments=segments,
        )

