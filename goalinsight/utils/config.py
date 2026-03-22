"""Configuration loading utilities and factory functions.

This module provides:
1. Configuration loading from YAML files
2. Factory functions for creating pipeline components based on config
"""

from pathlib import Path
from typing import Any, TYPE_CHECKING

import yaml

if TYPE_CHECKING:
    from ..interfaces import (
        BaseCalibrator,
        BaseReIDExtractor,
        BaseJerseyRecognizer,
        BaseTeamClassifier,
        BaseVisualizer,
    )


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


# =============================================================================
# Factory Functions
# =============================================================================


def get_calibrator(config: dict[str, Any] | None = None) -> "BaseCalibrator":
    """Create a calibrator based on configuration.

    Args:
        config: Configuration dict with 'field_registration' section.
            Expected keys:
            - backend: "pnlcalib" or "nbjw"
            - pnlcalib: {...} backend-specific config
            - nbjw: {...} backend-specific config

    Returns:
        Calibrator instance implementing BaseCalibrator.
    """
    if config is None:
        config = get_default_config()

    fr_config = config.get("field_registration", {})
    backend = fr_config.get("backend", "pnlcalib")

    if backend == "nbjw":
        from ..field_registration.nbjw import NbjwCalibrator
        backend_config = fr_config.get("nbjw", {})
        backend_config["device"] = config.get("device", "cuda")
        return NbjwCalibrator(backend_config)
    elif backend == "physical":
        from ..field_registration.physical_calibrator import PhysicalCalibrator
        return PhysicalCalibrator
    else:
        # Default: PnLCalib (use existing FramebyFrameCalib)
        from ..field_registration.pnlcalib import FramebyFrameCalib
        # Note: FramebyFrameCalib doesn't implement BaseCalibrator interface directly
        # For now, return it and let stage1.py handle the differences
        # TODO: Create a PnlCalibrator wrapper class
        return FramebyFrameCalib


def get_reid_extractor(config: dict[str, Any] | None = None) -> "BaseReIDExtractor":
    """Create a ReID extractor based on configuration.

    Args:
        config: Configuration dict with 'reid' section.
            Expected keys:
            - backend: "osnet" or "prtreid"
            - osnet: {...} backend-specific config
            - prtreid: {...} backend-specific config

    Returns:
        ReID extractor instance implementing BaseReIDExtractor.
    """
    if config is None:
        config = get_default_config()

    reid_config = config.get("reid", {})
    backend = reid_config.get("backend", "osnet")

    if backend == "prtreid":
        from ..tracking.reid import PRTReIDExtractor
        backend_config = reid_config.get("prtreid", {})
        backend_config["device"] = config.get("device", "cuda")
        return PRTReIDExtractor(backend_config)
    else:
        # Default: OSNet
        from ..tracking.reid import OSNetExtractor
        extractor_config = {
            "model": reid_config.get("model", "osnet_x1_0"),
            "feature_dim": reid_config.get("feature_dim", 512),
            "batch_size": reid_config.get("batch_size", 32),
            "device": config.get("device", "cuda"),
        }
        return OSNetExtractor(extractor_config)


def get_jersey_recognizer(config: dict[str, Any] | None = None) -> "BaseJerseyRecognizer":
    """Create a jersey recognizer based on configuration.

    Args:
        config: Configuration dict with 'jersey_recognition' section.
            Expected keys:
            - backend: "qwen_vl" or "mmocr"
            - enabled: Whether jersey recognition is enabled
            - qwen_vl: {...} backend-specific config
            - mmocr: {...} backend-specific config

    Returns:
        Jersey recognizer instance implementing BaseJerseyRecognizer.
    """
    if config is None:
        config = get_default_config()

    jr_config = config.get("jersey_recognition", {})
    backend = jr_config.get("backend", "qwen_vl")

    if backend == "mmocr":
        from ..jersey import MMOCRJerseyRecognizer
        backend_config = jr_config.get("mmocr", {})
        backend_config["device"] = config.get("device", "cuda")
        return MMOCRJerseyRecognizer(backend_config)
    else:
        # Default: Qwen VL
        from ..jersey import QwenJerseyRecognizer
        recognizer_config = {
            "mode": jr_config.get("mode", "local"),
            "local": jr_config.get("local", {}),
            "api": jr_config.get("api", {}),
        }
        return QwenJerseyRecognizer(recognizer_config)


def get_team_classifier(config: dict[str, Any] | None = None) -> "BaseTeamClassifier":
    """Create a team classifier based on configuration.

    Args:
        config: Configuration dict with 'team_classification' section.
            Expected keys:
            - backend: "kmeans" or "tracklet"
            - kmeans: {...} backend-specific config
            - tracklet: {...} backend-specific config

    Returns:
        Team classifier instance implementing BaseTeamClassifier.
    """
    if config is None:
        config = get_default_config()

    tc_config = config.get("team_classification", {})
    backend = tc_config.get("backend", "kmeans")

    if backend == "tracklet":
        from ..tracking.team import TrackletTeamClustering
        backend_config = tc_config.get("tracklet", {})
        return TrackletTeamClustering(backend_config)
    else:
        # Default: KMeans
        from ..tracking.team import KMeansTeamClassifier
        role_config = config.get("role_classification", {})
        classifier_config = {
            "n_teams": role_config.get("n_clusters", 2),
            "use_position": role_config.get("use_field_position", True),
            "position_weight": tc_config.get("position_weight", 0.1),
            "min_samples_per_team": tc_config.get("min_samples_per_team", 5),
        }
        return KMeansTeamClassifier(classifier_config)


def get_visualizer(
    config: dict[str, Any] | None = None,
    output_dir: Path | str | None = None,
) -> "BaseVisualizer":
    """Create a visualizer based on configuration.

    Args:
        config: Configuration dict with 'visualization' section.
            Expected keys:
            - backend: "minimal" or "step"
        output_dir: Output directory for saving visualizations.

    Returns:
        Visualizer instance implementing BaseVisualizer.
    """
    if config is None:
        config = get_default_config()

    vis_config = config.get("visualization", {})
    backend = vis_config.get("backend", "minimal")

    if backend == "step":
        from .visualizers import StepVisualizer
        return StepVisualizer(output_dir)
    else:
        # Default: Minimal
        from .visualizers import MinimalVisualizer
        return MinimalVisualizer(output_dir)


def get_side_labeler(config: dict[str, Any] | None = None):
    """Create a team side labeler based on configuration.

    Args:
        config: Configuration dict.

    Returns:
        Side labeler instance.
    """
    if config is None:
        config = get_default_config()

    from ..tracking.team import TrackletTeamSideLabeling
    return TrackletTeamSideLabeling(config.get("team_classification", {}))
