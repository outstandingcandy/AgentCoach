"""Event detection module for GoalInsight.

Detects match events (possession, passes, shots/goals, carries, defensive
actions) from pipeline tracking output.

Usage as pipeline stage:
    Configure 'event_detection' in pipeline.stages.

Usage standalone:
    from goalinsight.events import detect_events_from_output
    events = detect_events_from_output("output/run_xxx/")

Usage programmatic:
    from goalinsight.events import detect_events_from_data
    events = detect_events_from_data(ball_tracks, player_tracks, ...)
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ._context import EventDetectionContext
from ._orchestrator import EventOrchestrator
from ._types import (
    BallState,
    EventType,
    MatchEvent,
    PassOutcome,
    PossessionSpan,
    ShotOutcome,
)


def detect_events_from_output(
    pipeline_output_dir: str | Path,
    config: dict[str, Any] | None = None,
    pitch_length: float | None = None,
    pitch_width: float | None = None,
    goal_length: float | None = None,
    goal_height: float | None = None,
    fps: float | None = None,
) -> list[MatchEvent]:
    """Detect events from a pipeline output directory.

    Pitch / goal dims default to ``calibration_metadata.video_info`` when
    omitted; falls back to FIFA only if neither caller nor metadata supply
    them. Pass an explicit value to override metadata.
    """
    ctx = EventDetectionContext.from_output_dir(
        pipeline_output_dir,
        pitch_length=pitch_length,
        pitch_width=pitch_width,
        goal_length=goal_length,
        goal_height=goal_height,
        fps=fps,
    )
    orchestrator = EventOrchestrator(config or {})
    return orchestrator.detect_all(ctx)


def detect_events_from_dirs(
    tracking_dir: str | Path,
    calibration_dir: str | Path | None = None,
    config: dict[str, Any] | None = None,
    pitch_length: float | None = None,
    pitch_width: float | None = None,
    goal_length: float | None = None,
    goal_height: float | None = None,
    fps: float | None = None,
) -> list[MatchEvent]:
    """Detect events from separate stage directories (pipeline stage use)."""
    ctx = EventDetectionContext.from_dirs(
        tracking_dir,
        calibration_dir,
        pitch_length=pitch_length,
        pitch_width=pitch_width,
        goal_length=goal_length,
        goal_height=goal_height,
        fps=fps,
    )
    orchestrator = EventOrchestrator(config or {})
    return orchestrator.detect_all(ctx)


def detect_events_from_data(
    ball_tracks: dict,
    player_tracks: dict,
    team_assignments: dict,
    fps: float = 30.0,
    pitch_length: float = 105.0,
    pitch_width: float = 68.0,
    goal_length: float = 7.32,
    goal_height: float = 2.44,
    camera_poses: dict | None = None,
    config: dict[str, Any] | None = None,
) -> list[MatchEvent]:
    """Detect events from pre-loaded data dicts.

    Pitch / goal dims default to FIFA — direct programmatic callers should
    pass explicit values for non-FIFA pitches.
    """
    ctx = EventDetectionContext(
        ball_tracks=ball_tracks,
        player_tracks=player_tracks,
        team_assignments=team_assignments,
        camera_poses=camera_poses,
        fps=fps,
        pitch_length=pitch_length,
        pitch_width=pitch_width,
        goal_length=goal_length,
        goal_height=goal_height,
    )
    orchestrator = EventOrchestrator(config or {})
    return orchestrator.detect_all(ctx)


__all__ = [
    "detect_events_from_output",
    "detect_events_from_dirs",
    "detect_events_from_data",
    "EventDetectionContext",
    "EventOrchestrator",
    "MatchEvent",
    "EventType",
    "PossessionSpan",
    "BallState",
    "ShotOutcome",
    "PassOutcome",
]
