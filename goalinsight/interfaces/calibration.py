"""Abstract base class for field calibration backends."""

from abc import ABC, abstractmethod
from typing import Any

import numpy as np


class BaseCalibrator(ABC):
    """Abstract base class for field calibration.

    Implementations:
    - PnLCalib: Uses HRNet keypoint/line detection with PnL optimization
    - NBJW: Uses NBJW's FramebyFrameCalib approach
    """

    @abstractmethod
    def __init__(self, config: dict[str, Any] | None = None):
        """Initialize calibrator.

        Args:
            config: Backend-specific configuration.
        """
        pass

    @abstractmethod
    def load_models(self) -> None:
        """Load required models (keypoint detector, line detector, etc.)."""
        pass

    @abstractmethod
    def detect_keypoints(self, frame: np.ndarray) -> dict:
        """Detect pitch keypoints in a frame.

        Args:
            frame: Input frame (BGR format from OpenCV).

        Returns:
            Dictionary with detected keypoints. Format may vary by backend,
            but should include at minimum:
            - 'keypoints': dict or list of keypoint detections
            - 'lines': dict or list of line detections (optional)
        """
        pass

    @abstractmethod
    def compute_homography(
        self,
        keypoints: dict,
        image_width: int,
        image_height: int,
    ) -> np.ndarray | None:
        """Compute homography from detected keypoints.

        Args:
            keypoints: Detected keypoints from detect_keypoints().
            image_width: Frame width in pixels.
            image_height: Frame height in pixels.

        Returns:
            3x3 homography matrix (image -> world), or None if failed.
            The world coordinate system should be centered at pitch center
            with x-axis along pitch length and y-axis along pitch width.
        """
        pass

    @abstractmethod
    def project_to_pitch(
        self,
        image_point: tuple[float, float],
        homography: np.ndarray,
    ) -> tuple[float, float] | None:
        """Project image point to pitch coordinates.

        Args:
            image_point: (x, y) in image coordinates (pixels).
            homography: Image -> world homography matrix.

        Returns:
            (x, y) in pitch coordinates (meters), or None if projection failed.
            Pitch center is at (0, 0), with x in [-52.5, 52.5] and y in [-34, 34].
        """
        pass

    def calibrate_frame(
        self,
        frame: np.ndarray,
        image_width: int | None = None,
        image_height: int | None = None,
    ) -> dict[str, Any]:
        """Convenience method to detect keypoints and compute homography.

        Args:
            frame: Input frame.
            image_width: Frame width (defaults to frame.shape[1]).
            image_height: Frame height (defaults to frame.shape[0]).

        Returns:
            Dictionary with:
            - 'keypoints': Detected keypoints
            - 'lines': Detected lines (if available)
            - 'homography': Computed homography matrix or None
            - 'success': Whether calibration succeeded
        """
        if image_width is None:
            image_height, image_width = frame.shape[:2]
        elif image_height is None:
            image_height = frame.shape[0]

        keypoints_result = self.detect_keypoints(frame)
        keypoints = keypoints_result.get('keypoints', {})
        lines = keypoints_result.get('lines', {})

        homography = self.compute_homography(keypoints, image_width, image_height)

        return {
            'keypoints': keypoints,
            'lines': lines,
            'homography': homography,
            'success': homography is not None,
        }

    def reset(self) -> None:
        """Reset calibrator state (e.g., temporal smoothing buffers)."""
        pass

    @property
    def name(self) -> str:
        """Return backend name for logging."""
        return self.__class__.__name__
