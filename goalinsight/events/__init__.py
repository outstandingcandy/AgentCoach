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


def _dims_from_config(
    config: dict[str, Any] | None,
) -> dict[str, float | None]:
    """Pull pitch / goal dims out of a pipeline config.

    The config carries the pitch as either an already-expanded ``pitch:``
    block or the ``pitch_type: <name>`` shorthand; expand the shorthand so
    a per-video config that only says ``pitch_type: futsal`` still yields
    the futsal goal width (3 m) instead of the FIFA fallback (7.32 m).
    Without this, a shot that misses a small futsal goal but crosses the
    goal line within ±3.66 m is mis-scored as a goal.

    Returns ``None`` for any dim the config doesn't define, so callers keep
    the caller > config > video_info > FIFA priority.
    """
    if not config:
        return {k: None for k in
                ("pitch_length", "pitch_width", "goal_length", "goal_height")}
    pitch = config.get("pitch")
    if not isinstance(pitch, dict) and "pitch_type" in config:
        # Expand ``pitch_type`` without mutating the caller's config.
        try:
            from ..annotation import pitch_types
            pitch = pitch_types.resolve(str(config["pitch_type"]))
        except (KeyError, ImportError):
            pitch = {}
    pitch = pitch if isinstance(pitch, dict) else {}
    return {
        k: (float(pitch[k]) if k in pitch else None)
        for k in ("pitch_length", "pitch_width", "goal_length", "goal_height")
    }


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

    Pitch / goal dims resolve caller > config (``pitch``/``pitch_type``) >
    ``calibration_metadata.video_info`` > FIFA fallback. Pass an explicit
    value to override.
    """
    cfg_dims = _dims_from_config(config)
    ctx = EventDetectionContext.from_output_dir(
        pipeline_output_dir,
        pitch_length=pitch_length if pitch_length is not None else cfg_dims["pitch_length"],
        pitch_width=pitch_width if pitch_width is not None else cfg_dims["pitch_width"],
        goal_length=goal_length if goal_length is not None else cfg_dims["goal_length"],
        goal_height=goal_height if goal_height is not None else cfg_dims["goal_height"],
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
    """Detect events from separate stage directories (pipeline stage use).

    Pitch / goal dims resolve caller > config (``pitch``/``pitch_type``) >
    calibration ``video_info`` > FIFA fallback.
    """
    cfg_dims = _dims_from_config(config)
    ctx = EventDetectionContext.from_dirs(
        tracking_dir,
        calibration_dir,
        pitch_length=pitch_length if pitch_length is not None else cfg_dims["pitch_length"],
        pitch_width=pitch_width if pitch_width is not None else cfg_dims["pitch_width"],
        goal_length=goal_length if goal_length is not None else cfg_dims["goal_length"],
        goal_height=goal_height if goal_height is not None else cfg_dims["goal_height"],
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
