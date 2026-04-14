"""Core data types for the highlight clipping agent system."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class Event:
    """A detected match event (goal, fast break, etc.)."""

    event_type: str         # "goal", "fast_break", "corner_kick", ...
    frame: int              # Key frame index
    timestamp: float        # Seconds into the video
    confidence: float       # Detection confidence [0, 1]
    metadata: dict[str, Any] = field(default_factory=dict)
    # Event-specific data lives in metadata, e.g.:
    #   goal: {goal_side, ball_position, ball_speed_mps, crossbar_validation}


@dataclass
class ClipSegment:
    """A segment of the highlight clip to render."""

    name: str               # "buildup", "strike", "celebration", "replay"
    start_frame: int
    end_frame: int
    view_type: str          # "wide", "closeup", "medium"
    focus_target: str = "ball"          # "ball" or "player"
    focus_track_id: int | None = None   # Player to focus on (when focus_target="player")
    overlays: list[dict[str, Any]] = field(default_factory=list)
    # overlay example: {"type": "text", "text": "GOAL!", "position": "center"}
    transition: str = "crossfade"       # "cut", "crossfade", "flash"
    speed: float = 1.0                  # < 1.0 for slow-motion replay


@dataclass
class AnalyzedEvent:
    """An event enriched with player attribution and clip segments."""

    event: Event
    key_players: list[dict[str, Any]] = field(default_factory=list)
    # e.g. [{"track_id": 7, "role": "scorer", "team": "team_A"}]
    segments: list[ClipSegment] = field(default_factory=list)
