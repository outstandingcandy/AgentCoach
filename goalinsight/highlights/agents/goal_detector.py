"""GoalEventDetector — detects goals via the events module."""

from __future__ import annotations

import json
import logging
from typing import Any

from .._context import MatchContext
from .._types import Event
from .base import BaseEventDetector

logger = logging.getLogger(__name__)


class GoalEventDetector(BaseEventDetector):
    """Detect goal events using goalinsight.events module."""

    event_type = "goal"

    def detect(self, ctx: MatchContext, config: dict[str, Any]) -> list[Event]:
        # Try pre-computed events.json first
        events_json = self._load_events_json(ctx)
        if events_json is not None:
            return self._from_events_json(events_json, ctx.fps)

        # Fall back to running event detection directly
        return self._run_detection(ctx, config)

    def _load_events_json(self, ctx: MatchContext) -> list[dict] | None:
        """Try to load events.json from pipeline output."""
        for stage_dir in ("event_detection", "goal_detection"):
            path = ctx.pipeline_output_dir / stage_dir / "events.json"
            if path.exists():
                with open(path) as f:
                    data = json.load(f)
                logger.info("Loaded events from %s", path)
                return data
        return None

    def _from_events_json(
        self, events_data: list[dict], fps: float
    ) -> list[Event]:
        """Convert events.json entries to highlight Event objects."""
        events: list[Event] = []
        for e in events_data:
            if e.get("type") != "goal":
                continue
            meta = e.get("metadata", {})
            events.append(
                Event(
                    event_type="goal",
                    frame=e["frame"],
                    timestamp=e.get("match_time", e["frame"] / fps),
                    confidence=e.get("confidence", 1.0),
                    metadata={
                        "goal_side": meta.get("goal_side"),
                        "ball_position": meta.get("ball_position_3d"),
                        "ball_pixel": meta.get("ball_pixel"),
                        "ball_speed_mps": meta.get("ball_speed_mps", 0.0),
                        "crossbar_validation": meta.get(
                            "crossbar_validation"
                        ),
                    },
                )
            )
        logger.info(
            "GoalEventDetector found %d goal(s) from events.json",
            len(events),
        )
        return events

    def _run_detection(
        self, ctx: MatchContext, config: dict[str, Any]
    ) -> list[Event]:
        """Run event detection directly and filter to goals."""
        from goalinsight.events import EventType, detect_events_from_output

        goal_cfg = config.get("goal_detection", {})
        min_confidence = goal_cfg.get("min_confidence", 0.15)

        events_config = {
            "events": {
                "detectors": ["possession", "shot"],
                "shot": {"min_confidence": min_confidence},
            }
        }

        raw_events = detect_events_from_output(
            pipeline_output_dir=ctx.pipeline_output_dir,
            config=events_config,
            pitch_length=ctx.pitch_length,
            pitch_width=ctx.pitch_width,
            fps=ctx.fps,
        )

        events: list[Event] = []
        for e in raw_events:
            if e.event_type != EventType.GOAL:
                continue
            meta = e.metadata or {}
            events.append(
                Event(
                    event_type="goal",
                    frame=e.frame,
                    timestamp=e.match_time,
                    confidence=e.confidence,
                    metadata={
                        "goal_side": meta.get("goal_side"),
                        "ball_position": meta.get("ball_position_3d"),
                        "ball_pixel": meta.get("ball_pixel"),
                        "ball_speed_mps": meta.get("ball_speed_mps", 0.0),
                        "crossbar_validation": meta.get(
                            "crossbar_validation"
                        ),
                    },
                )
            )

        logger.info("GoalEventDetector found %d goal(s)", len(events))
        return events
