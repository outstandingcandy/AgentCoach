"""Homography Chain Calibration Module.

Propagates calibration from high-confidence anchor frames to
low-confidence frames using frame-to-frame feature matching.

Usage (with manual anchors):
    from goalinsight.field_registration.homography_chain import ChainCalibrator

    calibrator = ChainCalibrator(mode="offline")
    calibrator.add_anchor(661, camera_params=anchor_params_661)
    calibrator.add_anchor(1127, camera_params=anchor_params_1127)
    results = calibrator.calibrate_range(
        video_path="video.mp4",
        start_frame=681,
        end_frame=1127,
    )
    calibrator.export_results("chain_calib.json")

Usage (with automatic anchor detection):
    from goalinsight.field_registration.homography_chain import VideoCalibrator

    calibrator = VideoCalibrator()
    results = calibrator.calibrate_video(
        video_path="video.mp4",
        start_frame=680,
        end_frame=1128,
    )
    calibrator.export_results("auto_calib.json")
"""

from .auto_anchor_selector import (
    AnchorSelectionError,
    AutoAnchorSelector,
    FrameScore,
)
from .bidirectional_smoother import BidirectionalSmoother
from .camera_param_converter import CameraParamConverter
from .chain_calibrator import ChainCalibrator
from .drift_detector import DriftDetector, DriftMetrics
from .dynamic_masker import DynamicMasker
from .feature_matcher import FeatureMatcher
from .homography_propagator import HomographyPropagator, PropagationState
from .param_interpolator import ParamInterpolator
from .video_calibrator import VideoCalibrator

__all__ = [
    # High-level API
    "VideoCalibrator",
    "AutoAnchorSelector",
    "FrameScore",
    "AnchorSelectionError",
    # Core calibration
    "ChainCalibrator",
    "FeatureMatcher",
    "DynamicMasker",
    "HomographyPropagator",
    "PropagationState",
    "DriftDetector",
    "DriftMetrics",
    "CameraParamConverter",
    "BidirectionalSmoother",
    "ParamInterpolator",
]
