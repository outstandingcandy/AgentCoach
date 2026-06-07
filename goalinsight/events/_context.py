"""EventDetectionContext — shared state for all event detectors."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ._types import BallState, MatchEvent, PossessionSpan

logger = logging.getLogger(__name__)


@dataclass
class EventDetectionContext:
    """Shared state passed to all event detectors."""

    # Input data
    ball_tracks: dict[str, dict]
    player_tracks: dict[str, list[dict]]
    team_assignments: dict[str, str]
    camera_poses: dict[str, dict] | None
    fps: float
    pitch_length: float
    pitch_width: float
    # Goal-frame geometry (FIFA defaults — non-FIFA pitches override via
    # video_info metadata or constructor kwargs).
    goal_length: float = 7.32
    goal_height: float = 2.44

    # Pre-computed by orchestrator
    ball_states: list[BallState] = field(default_factory=list)
    frame_to_ball: dict[int, BallState] = field(default_factory=dict)

    # Cross-detector shared state
    possession_spans: list[PossessionSpan] = field(default_factory=list)
    possession_at_frame: dict[int, PossessionSpan | None] = field(
        default_factory=dict
    )

    # Accumulated results
    events: list[MatchEvent] = field(default_factory=list)

    # ------------------------------------------------------------------
    # Convenience accessors
    # ------------------------------------------------------------------

    def get_players_at_frame(self, frame: int) -> list[dict]:
        return self.player_tracks.get(str(frame), [])

    def get_team_for_track(self, track_id: int) -> str:
        return self.team_assignments.get(str(track_id), "unknown")

    def get_ball_at_frame(self, frame: int) -> BallState | None:
        return self.frame_to_ball.get(frame)

    def get_possession_at_frame(self, frame: int) -> PossessionSpan | None:
        return self.possession_at_frame.get(frame)

    # ------------------------------------------------------------------
    # Factory methods
    # ------------------------------------------------------------------

    @classmethod
    def from_output_dir(
        cls,
        pipeline_output_dir: str | Path,
        pitch_length: float | None = None,
        pitch_width: float | None = None,
        goal_length: float | None = None,
        goal_height: float | None = None,
        fps: float | None = None,
    ) -> EventDetectionContext:
        """Build context from a pipeline output directory.

        Pitch / goal dimensions resolve in this order: caller override →
        ``calibration_metadata.video_info`` → FIFA defaults. ``None`` means
        "fall through to the next source"; the previous sentinel-on-105.0
        scheme silently ignored real FIFA configs.
        """
        d = Path(pipeline_output_dir)

        ball_tracks = _load_json(d, "tracking", "ball_tracks.json") or {}
        player_tracks = _load_json(d, "tracking", "tracks.json") or {}
        team_assignments = (
            _load_json(d, "tracking", "team_assignments.json") or {}
        )
        camera_poses = _load_json(
            d, "field_registration", "camera_poses.json"
        )

        # Auto-detect fps and pitch dimensions from metadata.
        meta = _load_json(
            d, "field_registration", "calibration_metadata.json"
        )
        vi = meta.get("video_info", {}) if meta else {}
        if fps is None:
            fps = vi.get("fps", 30.0)

        return cls(
            ball_tracks=ball_tracks,
            player_tracks=player_tracks,
            team_assignments=team_assignments,
            camera_poses=camera_poses,
            fps=fps,
            **_resolve_dims(vi, pitch_length, pitch_width, goal_length, goal_height),
        )

    @classmethod
    def from_dirs(
        cls,
        tracking_dir: str | Path,
        calibration_dir: str | Path | None = None,
        pitch_length: float | None = None,
        pitch_width: float | None = None,
        goal_length: float | None = None,
        goal_height: float | None = None,
        fps: float | None = None,
    ) -> EventDetectionContext:
        """Build context from separate stage directories.

        See :meth:`from_output_dir` for the dim resolution order.
        """
        tracking_dir = Path(tracking_dir)
        ball_tracks = _load_json_file(tracking_dir / "ball_tracks.json") or {}
        player_tracks = _load_json_file(tracking_dir / "tracks.json") or {}
        team_assignments = (
            _load_json_file(tracking_dir / "team_assignments.json") or {}
        )

        camera_poses = None
        vi: dict = {}
        if calibration_dir:
            cal = Path(calibration_dir)
            camera_poses = _load_json_file(cal / "camera_poses.json")
            meta = _load_json_file(cal / "calibration_metadata.json")
            vi = meta.get("video_info", {}) if meta else {}
            if fps is None:
                fps = vi.get("fps", 30.0)

        if fps is None:
            fps = 30.0

        return cls(
            ball_tracks=ball_tracks,
            player_tracks=player_tracks,
            team_assignments=team_assignments,
            camera_poses=camera_poses,
            fps=fps,
            **_resolve_dims(vi, pitch_length, pitch_width, goal_length, goal_height),
        )


# ------------------------------------------------------------------
# File helpers
# ------------------------------------------------------------------


def _load_json(
    base: Path, stage: str, filename: str
) -> dict | None:
    """Try new-style then legacy-style stage directory names."""
    path = base / stage / filename
    if path.exists():
        return _load_json_file(path)
    # Legacy fallback
    legacy_map = {
        "tracking": "stage2",
        "field_registration": "stage1",
    }
    legacy = legacy_map.get(stage)
    if legacy:
        path = base / legacy / filename
        if path.exists():
            return _load_json_file(path)
    return None


def _load_json_file(path: Path) -> dict | None:
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


# FIFA fallback for pitch / goal dims when neither caller nor metadata supplied
# them. Kept here (not imported) so the events package stays self-contained.
_FIFA_FALLBACK = {
    "pitch_length": 105.0,
    "pitch_width": 68.0,
    "goal_length": 7.32,
    "goal_height": 2.44,
}


def _resolve_dims(
    video_info: dict,
    pitch_length: float | None,
    pitch_width: float | None,
    goal_length: float | None,
    goal_height: float | None,
) -> dict[str, float]:
    """Resolve pitch / goal dims with caller > video_info > FIFA priority."""
    out = {}
    for key, override in (
        ("pitch_length", pitch_length),
        ("pitch_width", pitch_width),
        ("goal_length", goal_length),
        ("goal_height", goal_height),
    ):
        if override is not None:
            out[key] = float(override)
        elif key in video_info:
            out[key] = float(video_info[key])
        else:
            out[key] = _FIFA_FALLBACK[key]
    return out
