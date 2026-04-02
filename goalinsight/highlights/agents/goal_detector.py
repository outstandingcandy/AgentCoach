"""GoalEventDetector — wraps existing goal_detection module."""

from __future__ import annotations

import logging
from typing import Any

from goalinsight.goal_detection import detect_goals_from_output

from .._context import MatchContext
from .._types import Event
from .base import BaseEventDetector

logger = logging.getLogger(__name__)


class GoalEventDetector(BaseEventDetector):
    """Detect goal events by delegating to goalinsight.goal_detection."""

    event_type = "goal"

    def detect(self, ctx: MatchContext, config: dict[str, Any]) -> list[Event]:
        goal_cfg = config.get("goal_detection", {})
        min_confidence = goal_cfg.get("min_confidence", 0.15)

        raw_goals = detect_goals_from_output(
            output_dir=ctx.pipeline_output_dir,
            pitch_length=ctx.pitch_length,
            pitch_width=ctx.pitch_width,
            fps=ctx.fps,
            min_confidence=min_confidence,
        )

        events: list[Event] = []
        for g in raw_goals:
            events.append(
                Event(
                    event_type="goal",
                    frame=g["frame"],
                    timestamp=g["timestamp"],
                    confidence=g["confidence"],
                    metadata={
                        "goal_side": g["goal_side"],
                        "ball_position": g["ball_position"],
                        "ball_pixel": g.get("ball_pixel"),
                        "ball_speed_mps": g.get("ball_speed_mps", 0.0),
                        "crossbar_validation": g.get("crossbar_validation"),
                    },
                )
            )

        logger.info("GoalEventDetector found %d goal(s)", len(events))
        return events
