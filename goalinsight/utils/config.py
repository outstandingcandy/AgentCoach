"""Configuration loading utilities.

This module provides configuration loading from YAML files,
frame sampling, and FPS management.

Factory functions for creating pipeline components have been moved to
``goalinsight.utils.factories`` and are re-exported here for backwards
compatibility.
"""

from pathlib import Path
from typing import Any

import yaml


# =============================================================================
# Configuration Loading
# =============================================================================


def load_config(config_path: str | Path) -> dict[str, Any]:
    """Load configuration from YAML file.

    Args:
        config_path: Path to the configuration file.

    Returns:
        Configuration dictionary.
    """
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    return config


def load_pitch_template(template_path: str | Path | None = None) -> dict[str, Any]:
    """Load pitch template from YAML file.

    Args:
        template_path: Path to pitch template file. If None, uses default.

    Returns:
        Pitch template dictionary with keypoints and lines.
    """
    if template_path is None:
        # Use default template (project_root/configs/)
        template_path = Path(__file__).parent.parent.parent / "configs" / "pitch_template.yaml"

    template_path = Path(template_path)
    if not template_path.exists():
        raise FileNotFoundError(f"Pitch template not found: {template_path}")

    with open(template_path, "r") as f:
        template = yaml.safe_load(f)

    return template


def get_default_config() -> dict[str, Any]:
    """Get default configuration.

    Returns:
        Default configuration dictionary.
    """
    # project_root/configs/default.yaml
    default_path = Path(__file__).parent.parent.parent / "configs" / "default.yaml"
    return load_config(default_path)


def merge_configs(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Deep merge two configuration dictionaries.

    Args:
        base: Base configuration.
        override: Override configuration (takes precedence).

    Returns:
        Merged configuration dictionary.
    """
    result = base.copy()

    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = merge_configs(result[key], value)
        else:
            result[key] = value

    return result


def get_frame_indices(
    total_frames: int,
    video_fps: float,
    process_fps: float | None = None,
) -> list[int]:
    """Get list of frame indices to process based on target FPS.

    Args:
        total_frames: Total number of frames in video.
        video_fps: Original video frame rate.
        process_fps: Target frames per second to process.
                    If None or 0, returns all frames.

    Returns:
        List of frame indices to process.

    Example:
        >>> get_frame_indices(1000, video_fps=50, process_fps=10)
        [0, 5, 10, 15, ...]  # Every 5th frame (50/10=5)
    """
    if process_fps is None or process_fps <= 0 or process_fps >= video_fps:
        # Process all frames
        return list(range(total_frames))

    # Calculate frame interval
    frame_interval = video_fps / process_fps

    # Generate frame indices
    indices = []
    frame_idx = 0.0
    while int(frame_idx) < total_frames:
        indices.append(int(frame_idx))
        frame_idx += frame_interval

    return indices


def get_process_fps_from_config(config: dict[str, Any] | None = None) -> float | None:
    """Get process_fps setting from configuration.

    Args:
        config: Configuration dictionary. If None, loads default config.

    Returns:
        Target process FPS, or None if not set.
    """
    if config is None:
        config = get_default_config()

    video_config = config.get("video", {})
    return video_config.get("process_fps")


class FrameSampler:
    """Helper class for frame sampling based on target FPS."""

    def __init__(
        self,
        total_frames: int,
        video_fps: float,
        process_fps: float | None = None,
    ):
        """Initialize frame sampler.

        Args:
            total_frames: Total frames in video.
            video_fps: Original video FPS.
            process_fps: Target processing FPS. If None, process all frames.
        """
        self.total_frames = total_frames
        self.video_fps = video_fps
        self.process_fps = process_fps

        # Compute frame indices to process
        self.frame_indices = get_frame_indices(total_frames, video_fps, process_fps)
        self.num_frames_to_process = len(self.frame_indices)

        # Create set for fast lookup
        self._frame_set = set(self.frame_indices)

    def should_process(self, frame_idx: int) -> bool:
        """Check if a frame should be processed."""
        return frame_idx in self._frame_set

    def get_nearest_processed_frame(self, frame_idx: int) -> int | None:
        """Get nearest processed frame index."""
        if not self.frame_indices:
            return None

        import bisect
        pos = bisect.bisect_left(self.frame_indices, frame_idx)

        if pos == 0:
            return self.frame_indices[0]
        if pos == len(self.frame_indices):
            return self.frame_indices[-1]

        before = self.frame_indices[pos - 1]
        after = self.frame_indices[pos]

        if frame_idx - before <= after - frame_idx:
            return before
        return after

    @property
    def frame_interval(self) -> float:
        """Get frame interval (frames between processed frames)."""
        if self.process_fps is None or self.process_fps <= 0:
            return 1.0
        return self.video_fps / self.process_fps

    def __len__(self) -> int:
        return self.num_frames_to_process

    def __iter__(self):
        return iter(self.frame_indices)


# Backwards compatibility: factory functions moved to factories.py
from .factories import (  # noqa: F401
    get_calibrator,
    get_reid_extractor,
    get_jersey_recognizer,
    get_team_classifier,
    get_visualizer,
    get_side_labeler,
)
