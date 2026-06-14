"""MatchContext — flat copy of goalinsight/highlights/_context.py.

Loads pipeline output JSON files from a directory. The runtime fills
that directory by syncing the relevant slice from S3.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def _load_json(path: Path) -> Any | None:
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def _load_tracks(run_dir: Path) -> dict[str, list[dict[str, Any]]] | None:
    consolidated = run_dir / "track_consolidation" / "tracks.json"
    if consolidated.exists():
        return _load_json(consolidated)
    return _load_json(run_dir / "tracking" / "tracks.json")


def _load_team_assignments(run_dir: Path) -> dict[str, str] | None:
    consolidated = run_dir / "track_consolidation" / "team_assignments.json"
    if consolidated.exists():
        return _load_json(consolidated)
    return _load_json(run_dir / "tracking" / "team_assignments.json")


@dataclass
class MatchContext:
    video_path: Path
    pipeline_output_dir: Path
    fps: float = 30.0
    frame_count: int = 0
    width: int = 1920
    height: int = 1080
    pitch_length: float = 105.0
    pitch_width: float = 68.0

    _player_tracks: dict[str, list[dict]] | None = field(default=None, repr=False)
    _ball_tracks: dict[str, dict] | None = field(default=None, repr=False)
    _team_assignments: dict[str, str] | None = field(default=None, repr=False)
    _events: list[dict] | None = field(default=None, repr=False)

    @classmethod
    def from_output_dir(
        cls,
        pipeline_output_dir: str | Path,
        video_path: str | Path | None = None,
        pitch_length: float = 105.0,
        pitch_width: float = 68.0,
    ) -> MatchContext:
        pipeline_output_dir = Path(pipeline_output_dir)

        fps = 30.0
        frame_count = 0
        width, height = 1920, 1080

        meta_path = pipeline_output_dir / "field_registration" / "calibration_metadata.json"
        if meta_path.exists():
            meta = json.loads(meta_path.read_text())
            vi = meta.get("video_info", {})
            fps = vi.get("fps", fps)
            frame_count = vi.get("total_frames", frame_count)
            width = vi.get("width", width)
            height = vi.get("height", height)
            if pitch_length == 105.0 and "pitch_length" in vi:
                pitch_length = vi["pitch_length"]
            if pitch_width == 68.0 and "pitch_width" in vi:
                pitch_width = vi["pitch_width"]

        # In the runtime container we never touch the actual video file —
        # the chat path only inspects ctx.video_path.name for the schema
        # brief — so we accept a stub path when nothing is supplied.
        if video_path is None:
            video_path = Path("video.mp4")
        else:
            video_path = Path(video_path)

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

    @property
    def player_tracks(self) -> dict[str, list[dict]]:
        if self._player_tracks is None:
            self._player_tracks = _load_tracks(self.pipeline_output_dir) or {}
        return self._player_tracks

    @property
    def ball_tracks(self) -> dict[str, dict]:
        if self._ball_tracks is None:
            self._ball_tracks = (
                _load_json(self.pipeline_output_dir / "tracking" / "ball_tracks.json")
                or {}
            )
        return self._ball_tracks

    @property
    def team_assignments(self) -> dict[str, str]:
        if self._team_assignments is None:
            self._team_assignments = _load_team_assignments(self.pipeline_output_dir) or {}
        return self._team_assignments

    @property
    def events(self) -> list[dict]:
        if self._events is None:
            path = self.pipeline_output_dir / "event_detection" / "events.json"
            if path.exists():
                data = json.loads(path.read_text())
                self._events = data if isinstance(data, list) else []
            else:
                self._events = []
        return self._events
