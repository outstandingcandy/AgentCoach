"""PnLCalib integration for soccer field registration.

This module provides HRNet-based keypoint and line detection models
following the PnLCalib approach for improved camera calibration.

Reference:
    Gutierrez-Perez & Agudo, "PnLCalib: Single-View Camera-Field Calibration
    using Points and Lines", arXiv 2024. https://github.com/mguti97/PnLCalib
"""

from .hrnet import HRNetKeypointModel
from .hrnet_line import HRNetLineModel
from .heatmap_utils import (
    extract_keypoints_from_heatmap,
    extract_line_extremities,
    get_keypoints_from_heatmap_maxpool,
    get_lines_from_heatmap_maxpool,
    soft_argmax_2d,
)
from .frame_calibrator import FramebyFrameCalib, iterative_pnp_calibrate
from .keypoint_mapping import KeypointMapper
from .line_mapping import LineMapper
from .hough_line_matcher import HoughLineMatcher
from .pnl_optimizer import PnLOptimizer
from .weight_downloader import PnLCalibWeightDownloader
from .camera import Camera, is_good_camera

__all__ = [
    "HRNetKeypointModel",
    "HRNetLineModel",
    "extract_keypoints_from_heatmap",
    "extract_line_extremities",
    "get_keypoints_from_heatmap_maxpool",
    "get_lines_from_heatmap_maxpool",
    "soft_argmax_2d",
    "FramebyFrameCalib",
    "iterative_pnp_calibrate",
    "KeypointMapper",
    "LineMapper",
    "HoughLineMatcher",
    "PnLOptimizer",
    "PnLCalibWeightDownloader",
    "Camera",
    "is_good_camera",
]
