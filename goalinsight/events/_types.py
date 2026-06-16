"""Core data types for the event detection module."""

from __future__ import annotations

import enum
from dataclasses import dataclass, field
from typing import Any


class EventType(str, enum.Enum):
    POSSESSION = "possession"
    PASS = "pass"
    SHOT = "shot"
    CARRY = "carry"
    TACKLE = "tackle"
    INTERCEPTION = "interception"
    GOAL = "goal"


class ShotOutcome(str, enum.Enum):
    GOAL = "Goal"
    SAVED = "Saved"
    OFF_TARGET = "Off_Target"
    BLOCKED = "Blocked"


class PassOutcome(str, enum.Enum):
    SUCCESSFUL = "successful"
    FAILED = "failed"


@dataclass
class MatchEvent:
    """A detected match event."""

    event_id: str
    event_type: EventType
    frame: int
    match_time: float  # seconds into match
    # ``player_id`` is an int track_id when event_detection runs on raw
    # tracker output, or a consolidated string (``"A-7"``) when it runs
    # after track_consolidation. Equality comparison and dict-key use are
    # the only operations on it, so both forms flow through unchanged.
    player_id: int | str | None = None
    team_id: str | None = None
    start_frame: int | None = None
    end_frame: int | None = None
    start_position: list[float] | None = None  # [x, y] pitch coords
    end_position: list[float] | None = None
    confidence: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to JSON-compatible dict."""
        d: dict[str, Any] = {
            "event_id": self.event_id,
            "type": self.event_type.value,
            "frame": self.frame,
            "match_time": round(self.match_time, 3),
        }
        if self.player_id is not None:
            d["player_id"] = self.player_id
        if self.team_id is not None:
            d["team_id"] = self.team_id
        if self.start_frame is not None:
            d["start_frame"] = self.start_frame
        if self.end_frame is not None:
            d["end_frame"] = self.end_frame
        if self.start_position is not None:
            d["start_position"] = [round(v, 2) for v in self.start_position]
        if self.end_position is not None:
            d["end_position"] = [round(v, 2) for v in self.end_position]
        d["confidence"] = round(self.confidence, 3)
        if self.metadata:
            d["metadata"] = self.metadata
        return d


@dataclass
class PossessionSpan:
    """A continuous interval where a single player controls the ball."""

    player_id: int | str
    team_id: str
    start_frame: int
    end_frame: int
    start_position: list[float]  # [x, y]
    end_position: list[float]  # [x, y]


@dataclass
class BallState:
    """Per-frame ball state with derived velocity."""

    frame: int
    position: list[float]  # [x, y] pitch coords
    position_3d: list[float] | None  # [x, y, z]
    height: float
    velocity: list[float] | None  # [vx, vy] m/s, None for first frame
    speed: float  # |velocity| m/s
    confidence: float
    pixel_center: list[float] | None
