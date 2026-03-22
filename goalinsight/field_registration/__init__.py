"""Stage 1: Field Registration modules.

Supports multiple calibration backends:
- PnLCalib (default): HRNet keypoint/line detection with PnL optimization
- NBJW: Uses NBJW's FramebyFrameCalib approach
"""

from .keypoint_detector import KeypointDetector
from .line_detector import LineDetector
from .pnl_solver import PnLSolver, FieldRegistrar
from .distortion import DistortionCorrector

__all__ = [
    "KeypointDetector",
    "LineDetector",
    "PnLSolver",
    "FieldRegistrar",
    "DistortionCorrector",
]

# Submodules are imported on demand to avoid requiring all dependencies:
# - PnLCalib: from .pnlcalib import FramebyFrameCalib, KeypointMapper, LineMapper
# - NBJW: from .nbjw import NbjwCalibrator
# - Homography Chain: from .homography_chain import ChainCalibrator
# - Distortion: from .distortion import DistortionCorrector, LineSampler, DistortionVisualizer
