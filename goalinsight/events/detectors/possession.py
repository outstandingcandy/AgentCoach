"""PossessionDetector — foundation state machine for all event detection."""

from __future__ import annotations

import logging
import math
from typing import Any

from .._base import BaseEventDetector
from .._context import EventDetectionContext
from .._registry import register_detector
from .._types import EventType, MatchEvent, PossessionSpan

logger = logging.getLogger(__name__)


@register_detector
class PossessionDetector(BaseEventDetector):
    """Detect ball possession by player proximity over consecutive frames."""

    name = "possession"
    depends_on: list[str] = []

    def detect(
        self, ctx: EventDetectionContext, config: dict[str, Any]
    ) -> list[MatchEvent]:
        cfg = config.get("events", {}).get("possession", {})
        dist_threshold = cfg.get("distance_threshold", 1.5)
        min_seconds = cfg.get("min_consecutive_seconds", 1.0)
        speed_break = cfg.get("speed_break_threshold", 15.0)

        spans = self._build_spans(ctx, dist_threshold, min_seconds, speed_break)

        # Store on context for downstream detectors
        ctx.possession_spans = spans
        ctx.possession_at_frame = {}
        for span in spans:
            for f in range(span.start_frame, span.end_frame + 1):
                ctx.possession_at_frame[f] = span

        # Convert to MatchEvents
        events: list[MatchEvent] = []
        for i, span in enumerate(spans):
            duration = (span.end_frame - span.start_frame) / ctx.fps
            events.append(
                MatchEvent(
                    event_id=f"pos_{i:04d}",
                    event_type=EventType.POSSESSION,
                    frame=span.start_frame,
                    match_time=span.start_frame / ctx.fps,
                    player_id=span.player_id,
                    team_id=span.team_id,
                    start_frame=span.start_frame,
                    end_frame=span.end_frame,
                    start_position=span.start_position,
                    end_position=span.end_position,
                    confidence=1.0,
                    metadata={"duration_sec": round(duration, 2)},
                )
            )

        logger.info(
            "PossessionDetector: %d spans detected", len(spans)
        )
        return events

    def _build_spans(
        self,
        ctx: EventDetectionContext,
        dist_threshold: float,
        min_seconds: float,
        speed_break: float,
    ) -> list[PossessionSpan]:
        """Walk through frames and build continuous possession spans."""
        if not ctx.ball_states:
            return []

        spans: list[PossessionSpan] = []

        # Candidate tracking
        candidate_id: int | None = None
        candidate_team: str | None = None
        candidate_frames: list[int] = []
        candidate_positions: list[list[float]] = []

        for bs in ctx.ball_states:
            # Speed break: ball kicked hard → no one possesses
            if bs.speed > speed_break:
                self._flush_span(
                    spans, candidate_id, candidate_team,
                    candidate_frames, candidate_positions,
                    min_seconds, ctx.fps,
                )
                candidate_id = None
                candidate_frames = []
                candidate_positions = []
                continue

            nearest_id, nearest_team, nearest_dist = self._find_nearest_player(
                ctx, bs.frame, bs.position
            )

            if nearest_id is None or nearest_dist > dist_threshold:
                # No one close enough
                self._flush_span(
                    spans, candidate_id, candidate_team,
                    candidate_frames, candidate_positions,
                    min_seconds, ctx.fps,
                )
                candidate_id = None
                candidate_frames = []
                candidate_positions = []
                continue

            if nearest_id == candidate_id:
                # Same player still in control
                candidate_frames.append(bs.frame)
                candidate_positions.append(bs.position)
            else:
                # Different player — flush previous and start new candidate
                self._flush_span(
                    spans, candidate_id, candidate_team,
                    candidate_frames, candidate_positions,
                    min_seconds, ctx.fps,
                )
                candidate_id = nearest_id
                candidate_team = nearest_team
                candidate_frames = [bs.frame]
                candidate_positions = [bs.position]

        # Flush remaining
        self._flush_span(
            spans, candidate_id, candidate_team,
            candidate_frames, candidate_positions,
            min_seconds, ctx.fps,
        )

        return spans

    @staticmethod
    def _flush_span(
        spans: list[PossessionSpan],
        player_id: int | None,
        team_id: str | None,
        frames: list[int],
        positions: list[list[float]],
        min_seconds: float,
        fps: float,
    ) -> None:
        """Append a PossessionSpan if it meets the minimum duration."""
        if player_id is None or not frames:
            return
        duration = (frames[-1] - frames[0]) / fps
        if duration < min_seconds:
            return
        spans.append(
            PossessionSpan(
                player_id=player_id,
                team_id=team_id or "unknown",
                start_frame=frames[0],
                end_frame=frames[-1],
                start_position=positions[0],
                end_position=positions[-1],
            )
        )

    @staticmethod
    def _find_nearest_player(
        ctx: EventDetectionContext,
        frame: int,
        ball_pos: list[float],
    ) -> tuple[int | None, str | None, float]:
        """Find the nearest player to the ball at a given frame.

        Returns (track_id, team_id, distance). If no player has a pitch
        position, returns (None, None, inf).
        """
        players = ctx.get_players_at_frame(frame)
        best_id: int | None = None
        best_team: str | None = None
        best_dist = float("inf")

        bx, by = ball_pos

        for p in players:
            pp = p.get("pitch_position")
            if pp is None:
                continue
            tid = p["track_id"]
            team = ctx.get_team_for_track(tid)
            if team == "referee":
                continue
            dist = math.hypot(pp[0] - bx, pp[1] - by)
            if dist < best_dist:
                best_dist = dist
                best_id = tid
                best_team = team

        return best_id, best_team, best_dist
