"""PassDetector — detects passes from possession transitions."""

from __future__ import annotations

import logging
import math
from typing import Any

from .._base import BaseEventDetector
from .._context import EventDetectionContext
from .._registry import register_detector
from .._types import EventType, MatchEvent, PassOutcome

logger = logging.getLogger(__name__)


@register_detector
class PassDetector(BaseEventDetector):
    """Detect passes by analyzing possession transitions with ball speed jumps."""

    name = "pass"
    depends_on = ["possession"]

    def detect(
        self, ctx: EventDetectionContext, config: dict[str, Any]
    ) -> list[MatchEvent]:
        cfg = config.get("events", {}).get("pass", {})
        speed_threshold = cfg.get("pass_speed_threshold", 5.0)
        max_transit_sec = cfg.get("max_transit_seconds", 1.0)

        spans = ctx.possession_spans
        if len(spans) < 2:
            return []

        events: list[MatchEvent] = []
        seq = 0

        for i in range(len(spans) - 1):
            span_a = spans[i]
            span_b = spans[i + 1]

            gap_sec = (span_b.start_frame - span_a.end_frame) / ctx.fps
            if gap_sec > max_transit_sec or gap_sec < 0:
                continue

            # Check ball speed at transition
            max_speed = self._max_speed_in_range(
                ctx, span_a.end_frame, span_b.start_frame
            )
            if max_speed < speed_threshold:
                continue

            # Classify outcome
            if span_a.team_id == span_b.team_id:
                outcome = PassOutcome.SUCCESSFUL
            else:
                outcome = PassOutcome.FAILED

            pass_length = math.hypot(
                span_b.start_position[0] - span_a.end_position[0],
                span_b.start_position[1] - span_a.end_position[1],
            )

            seq += 1
            events.append(
                MatchEvent(
                    event_id=f"pass_{seq:04d}",
                    event_type=EventType.PASS,
                    frame=span_a.end_frame,
                    match_time=span_a.end_frame / ctx.fps,
                    player_id=span_a.player_id,
                    team_id=span_a.team_id,
                    start_position=span_a.end_position,
                    end_position=span_b.start_position,
                    confidence=1.0,
                    metadata={
                        "outcome": outcome.value,
                        "receiver_id": span_b.player_id,
                        "receiver_team": span_b.team_id,
                        "pass_length": round(pass_length, 2),
                        "is_successful": outcome == PassOutcome.SUCCESSFUL,
                    },
                )
            )

        logger.info(
            "PassDetector: %d pass(es) (%d successful, %d failed)",
            len(events),
            sum(
                1
                for e in events
                if e.metadata.get("outcome") == PassOutcome.SUCCESSFUL.value
            ),
            sum(
                1
                for e in events
                if e.metadata.get("outcome") == PassOutcome.FAILED.value
            ),
        )
        return events

    @staticmethod
    def _max_speed_in_range(
        ctx: EventDetectionContext, start_frame: int, end_frame: int
    ) -> float:
        """Find the maximum ball speed between two frames."""
        max_speed = 0.0
        for bs in ctx.ball_states:
            if bs.frame < start_frame:
                continue
            if bs.frame > end_frame:
                break
            if bs.speed > max_speed:
                max_speed = bs.speed
        return max_speed
