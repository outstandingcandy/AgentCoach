"""MatchContext — lazy-loads all pipeline output into a single queryable object."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class MatchContext:
    """Provides indexed access to all pipeline output data.

    Lazily loads JSON files on first access to keep construction cheap.
    """

    video_path: Path
    pipeline_output_dir: Path
    fps: float = 30.0
    frame_count: int = 0
    width: int = 1920
    height: int = 1080
    pitch_length: float = 105.0
    pitch_width: float = 68.0

    # Lazy-loaded caches (private)
    _player_tracks: dict[str, list[dict]] | None = field(
        default=None, repr=False
    )
    _ball_tracks: dict[str, dict] | None = field(default=None, repr=False)
    _camera_poses: dict[str, dict] | None = field(default=None, repr=False)
    _team_assignments: dict[str, str] | None = field(default=None, repr=False)
    _events: list[dict] | None = field(default=None, repr=False)

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def from_output_dir(
        cls,
        pipeline_output_dir: str | Path,
        video_path: str | Path | None = None,
        pitch_length: float = 105.0,
        pitch_width: float = 68.0,
    ) -> MatchContext:
        """Build a MatchContext by reading video info from calibration metadata.

        If *video_path* is None, it is auto-resolved from the pipeline
        metadata (``calibration_metadata.json``).
        """
        pipeline_output_dir = Path(pipeline_output_dir)

        fps = 30.0
        frame_count = 0
        width, height = 1920, 1080
        meta_video_path: str | None = None

        meta_path = pipeline_output_dir / "field_registration" / "calibration_metadata.json"
        if not meta_path.exists():
            meta_path = pipeline_output_dir / "stage1" / "calibration_metadata.json"
        if meta_path.exists():
            with open(meta_path) as f:
                meta = json.load(f)
            vi = meta.get("video_info", {})
            fps = vi.get("fps", fps)
            frame_count = vi.get("total_frames", frame_count)
            width = vi.get("width", width)
            height = vi.get("height", height)
            meta_video_path = vi.get("path")
            # Auto-detect pitch dimensions from calibration metadata
            # (overridden if caller passes explicit values)
            if pitch_length == 105.0 and "pitch_length" in vi:
                pitch_length = vi["pitch_length"]
            if pitch_width == 68.0 and "pitch_width" in vi:
                pitch_width = vi["pitch_width"]

        # Auto-resolve video path from metadata when not provided
        if video_path is None:
            if meta_video_path is None:
                raise ValueError(
                    "video_path not provided and not found in calibration metadata"
                )
            video_path = Path(meta_video_path)
            logger.info("Auto-resolved video path from metadata: %s", video_path)
        else:
            video_path = Path(video_path)
            # Warn if the provided path differs from what the pipeline used
            if meta_video_path and Path(meta_video_path).name != video_path.name:
                logger.warning(
                    "Provided video '%s' differs from pipeline source '%s'",
                    video_path.name, Path(meta_video_path).name,
                )

        # track_consolidation is a hard precondition for downstream stages.
        # It writes players.json and rewrites tracks.json with stable
        # player_ids; without it tracks.json lacks team/role/jersey.
        players_path = pipeline_output_dir / "track_consolidation" / "players.json"
        if not players_path.exists():
            raise RuntimeError(
                "track_consolidation must be run before this stage. "
                f"Expected {players_path} to exist. "
                "Run `--stages track_consolidation` first."
            )

        return cls(
            video_path=video_path,
            pipeline_output_dir=pipeline_output_dir,
            fps=fps,
            frame_count=frame_count,
            width=width,
            height=height,
            pitch_length=pitch_length,
            pitch_width=pitch_width,
        )

    # ------------------------------------------------------------------
    # Lazy-loading properties
    # ------------------------------------------------------------------

    @property
    def player_tracks(self) -> dict[str, list[dict]]:
        if self._player_tracks is None:
            from ..track_consolidation import load_tracks
            # Highlights need the consolidated player-id-keyed tracks.
            self._player_tracks = (
                load_tracks(self.pipeline_output_dir) or {}
            )
        return self._player_tracks

    @property
    def ball_tracks(self) -> dict[str, dict]:
        if self._ball_tracks is None:
            self._ball_tracks = (
                self._load_json("tracking", "ball_tracks.json")
                or self._load_json("stage2", "ball_tracks.json")
                or {}
            )
        return self._ball_tracks

    @property
    def camera_poses(self) -> dict[str, dict]:
        if self._camera_poses is None:
            self._camera_poses = (
                self._load_json("field_registration", "camera_poses.json")
                or self._load_json("stage1", "camera_poses.json")
                or {}
            )
        return self._camera_poses

    @property
    def team_assignments(self) -> dict[str, str]:
        if self._team_assignments is None:
            from ..track_consolidation import load_team_assignments
            self._team_assignments = (
                load_team_assignments(self.pipeline_output_dir) or {}
            )
        return self._team_assignments

    @property
    def events(self) -> list[dict]:
        """Load detected events from the event_detection stage output."""
        if self._events is None:
            for stage in ("event_detection", "goal_detection"):
                path = self.pipeline_output_dir / stage / "events.json"
                if path.exists():
                    with open(path) as f:
                        data = json.load(f)
                    if isinstance(data, list):
                        self._events = data
                        break
            if self._events is None:
                self._events = []
        return self._events

    # ------------------------------------------------------------------
    # Convenience accessors
    # ------------------------------------------------------------------

    def get_tracks_at_frame(self, frame: int) -> list[dict]:
        """Return player tracks at *frame* (preferring refined if available)."""
        return self.player_tracks.get(str(frame), [])

    def get_ball_at_frame(self, frame: int) -> dict | None:
        return self.ball_tracks.get(str(frame))

    def get_player_trajectory(
        self, track_id: int, start_frame: int, end_frame: int
    ) -> list[dict[str, Any]]:
        """Return all observations of *track_id* between start and end frames.

        Each returned dict has at least: frame, bbox, pitch_position.
        """
        result: list[dict[str, Any]] = []
        for f in range(start_frame, end_frame + 1):
            for t in self.player_tracks.get(str(f), []):
                if t.get("track_id") == track_id:
                    result.append({"frame": f, **t})
                    break
        return result

    def get_team_for_track(self, track_id: int) -> str:
        return self.team_assignments.get(str(track_id), "unknown")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load_json(self, stage: str, filename: str) -> dict | None:
        path = self.pipeline_output_dir / stage / filename
        if not path.exists():
            return None
        with open(path) as f:
            return json.load(f)
