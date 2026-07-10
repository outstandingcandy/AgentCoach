"""Workspace layout for the unified web app.

A workspace is a single on-disk directory that holds everything the unified
web product needs across sessions: source videos the user uploaded, manual
annotations, fine-tuned model weights, and pipeline run outputs. Subsequent
modules (library, jobs, analytics) read and write through this layout.

Layout::

    workspace/
      videos/                   # uploaded / discovered source videos
      annotations/<video_stem>/ # AnchorAnnotator output (frame_*.json, ...)
      configs/<video_stem>.yaml # per-video pipeline config authored from /library
      calibrations/<name>.json  # saved fixed-camera pose presets (reusable)
      models/<run_ts>/          # train_finetune.py outputs (best_model.pt, ...)
      runs/<run_name>/          # pipeline run outputs
        field_registration/
        tracking/
        track_consolidation/
        event_detection/
        highlights/
        annotated_video/
        logs/<stage>.log
      jobs.json                 # JobManager state
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

VIDEO_EXTS = (".mp4", ".mov", ".mkv", ".avi")


@dataclass(frozen=True)
class Workspace:
    root: Path

    @property
    def videos_dir(self) -> Path:
        return self.root / "videos"

    @property
    def annotations_dir(self) -> Path:
        return self.root / "annotations"

    @property
    def models_dir(self) -> Path:
        return self.root / "models"

    @property
    def runs_dir(self) -> Path:
        return self.root / "runs"

    @property
    def configs_dir(self) -> Path:
        return self.root / "configs"

    @property
    def calibrations_dir(self) -> Path:
        """Saved fixed-camera calibration presets (<name>.json). Each holds a
        solved camera pose + pitch_type so it can be reused across videos
        shot with the same camera at the same fixed position."""
        return self.root / "calibrations"

    @property
    def jobs_file(self) -> Path:
        return self.root / "jobs.json"

    def run_dir(self, run_name: str) -> Path:
        return self.runs_dir / run_name

    def annotations_for(self, video_path: Path) -> Path:
        """Per-video annotations directory, keyed by file stem."""
        return self.annotations_dir / Path(video_path).stem

    def config_for(self, video_path: Path) -> Path:
        """Per-video config yaml, keyed by file stem (may not exist)."""
        return self.configs_dir / f"{Path(video_path).stem}.yaml"

    def ensure(self) -> None:
        for d in (
            self.videos_dir,
            self.annotations_dir,
            self.configs_dir,
            self.calibrations_dir,
            self.models_dir,
            self.runs_dir,
        ):
            d.mkdir(parents=True, exist_ok=True)


def resolve_workspace(path: str | Path) -> Workspace:
    ws = Workspace(Path(path).resolve())
    ws.ensure()
    return ws
