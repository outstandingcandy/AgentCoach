"""Factory functions for creating pipeline components based on config.

Each factory function lazily imports the appropriate backend implementation
based on configuration, avoiding unnecessary heavy imports at module load time.
"""

from pathlib import Path
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from ..interfaces import (
        BaseCalibrator,
        BaseReIDExtractor,
        BaseJerseyRecognizer,
        BaseTeamClassifier,
        BaseVisualizer,
    )


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
        from .config import get_default_config
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
        from .config import get_default_config
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
        from .config import get_default_config
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
        from .config import get_default_config
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
        from .config import get_default_config
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
        from .config import get_default_config
        config = get_default_config()

    from ..tracking.team import TrackletTeamSideLabeling
    return TrackletTeamSideLabeling(config.get("team_classification", {}))
