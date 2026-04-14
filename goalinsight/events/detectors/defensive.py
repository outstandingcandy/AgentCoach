"""DefensiveActionDetector — detects tackles and interceptions."""

from __future__ import annotations

import logging
import math
from typing import Any

from .._base import BaseEventDetector
from .._context import EventDetectionContext
from .._registry import register_detector
from .._types import EventType, MatchEvent, PassOutcome
from ._utils import find_nearest_player

logger = logging.getLogger(__name__)


@register_detector
class DefensiveActionDetector(BaseEventDetector):
    """Detect tackles and interceptions from possession transitions."""

    name = "defensive"
    depends_on = ["possession", "pass"]

    def detect(
        self, ctx: EventDetectionContext, config: dict[str, Any]
    ) -> list[MatchEvent]:
        cfg = config.get("events", {}).get("defensive", {})
        tackle_proximity = cfg.get("tackle_proximity", 2.0)
        deflection_angle = cfg.get("tackle_deflection_angle", 45.0)
        tackle_max_gap_sec = cfg.get("tackle_max_gap_seconds", 0.5)
        deflection_window_sec = cfg.get("deflection_window_seconds", 0.3)

        events: list[MatchEvent] = []

        # --- Interceptions: failed passes where the defender gained possession ---
        interceptions = self._detect_interceptions(ctx)
        events.extend(interceptions)

        # --- Tackles: team-to-team possession changes without a pass ---
        tackles = self._detect_tackles(
            ctx, tackle_proximity, deflection_angle,
            tackle_max_gap_sec, deflection_window_sec,
        )
        events.extend(tackles)

        logger.info(
            "DefensiveActionDetector: %d interception(s), %d tackle(s)",
            len(interceptions),
            len(tackles),
        )
        return events

    def _detect_interceptions(
        self,
        ctx: EventDetectionContext,
    ) -> list[MatchEvent]:
        """Interceptions from failed pass events."""
        # Find failed passes from accumulated events
        failed_passes = [
            e
            for e in ctx.events
            if (
                e.event_type == EventType.PASS
                and e.metadata.get("outcome") == PassOutcome.FAILED.value
            )
        ]

        events: list[MatchEvent] = []
        seq = 0

        for pass_event in failed_passes:
            receiver_id = pass_event.metadata.get("receiver_id")
            receiver_team = pass_event.metadata.get("receiver_team")
            if receiver_id is None or receiver_team is None:
                continue

            seq += 1
            events.append(
                MatchEvent(
                    event_id=f"interception_{seq:04d}",
                    event_type=EventType.INTERCEPTION,
                    frame=pass_event.frame,
                    match_time=pass_event.match_time,
                    player_id=receiver_id,
                    team_id=receiver_team,
                    start_position=pass_event.end_position,
                    confidence=0.8,
                    metadata={
                        "target_event_id": pass_event.event_id,
                        "opponent_id": pass_event.player_id,
                        "opponent_team": pass_event.team_id,
                        "possession_won": True,
                    },
                )
            )

        return events

    def _detect_tackles(
        self,
        ctx: EventDetectionContext,
        proximity: float,
        deflection_angle: float,
        max_gap_sec: float,
        deflection_window_sec: float,
    ) -> list[MatchEvent]:
        """Tackles: team-to-team possession changes without a preceding pass."""
        spans = ctx.possession_spans
        if len(spans) < 2:
            return []

        # Collect frames covered by pass events to exclude
        pass_frames = set()
        for e in ctx.events:
            if e.event_type == EventType.PASS:
                pass_frames.add(e.frame)

        events: list[MatchEvent] = []
        seq = 0

        for i in range(len(spans) - 1):
            span_a = spans[i]
            span_b = spans[i + 1]

            # Must be a team change
            if span_a.team_id == span_b.team_id:
                continue

            # Skip if this transition was already explained by a pass
            if span_a.end_frame in pass_frames:
                continue

            gap_sec = (span_b.start_frame - span_a.end_frame) / ctx.fps
            if gap_sec > max_gap_sec:
                continue

            # Check if defender was close to the ball at transition
            transition_frame = span_a.end_frame
            ball = ctx.get_ball_at_frame(transition_frame)
            if ball is None:
                continue

            defender_id, _, defender_dist = find_nearest_player(
                ctx, transition_frame, ball.position,
                team_filter=span_b.team_id,
            )
            if defender_id is None or defender_dist > proximity:
                continue

            # Check ball trajectory deflection
            angle = self._compute_deflection(
                ctx, transition_frame, deflection_window_sec
            )
            if angle < deflection_angle:
                continue

            seq += 1
            events.append(
                MatchEvent(
                    event_id=f"tackle_{seq:04d}",
                    event_type=EventType.TACKLE,
                    frame=transition_frame,
                    match_time=transition_frame / ctx.fps,
                    player_id=defender_id,
                    team_id=span_b.team_id,
                    start_position=ball.position,
                    confidence=0.7,
                    metadata={
                        "opponent_id": span_a.player_id,
                        "opponent_team": span_a.team_id,
                        "possession_won": True,
                        "deflection_angle": round(angle, 1),
                    },
                )
            )

        return events

    @staticmethod
    def _compute_deflection(
        ctx: EventDetectionContext,
        frame: int,
        window_seconds: float = 0.3,
    ) -> float:
        """Compute ball trajectory deflection angle at a frame.

        Compares the ball's direction before and after the frame.
        Returns angle in degrees (0 = no change, 180 = reversal).
        """
        window_frames = window_seconds * ctx.fps

        # Find ball states around the frame
        before: list[float] = []
        after: list[float] = []

        for bs in ctx.ball_states:
            if bs.velocity is None:
                continue
            if bs.frame < frame - window_frames:
                continue
            if bs.frame > frame + window_frames:
                break
            if bs.frame <= frame:
                before = bs.velocity
            elif not after:
                after = bs.velocity

        if not before or not after:
            return 180.0  # No data → assume deflection

        # Angle between two velocity vectors
        dot = before[0] * after[0] + before[1] * after[1]
        mag_b = math.hypot(before[0], before[1])
        mag_a = math.hypot(after[0], after[1])

        if mag_b < 1e-6 or mag_a < 1e-6:
            return 0.0

        cos_angle = max(-1.0, min(1.0, dot / (mag_b * mag_a)))
        return math.degrees(math.acos(cos_angle))
