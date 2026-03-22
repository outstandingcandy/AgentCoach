"""Abstract base class for visualization backends."""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import numpy as np


class BaseVisualizer(ABC):
    """Abstract base class for pipeline visualization.

    Implementations:
    - Minimal: Simple bounding box and label visualization
    - Step: Detailed step-by-step visualization with radar view
    """

    @abstractmethod
    def __init__(self, output_dir: Path | str | None = None):
        """Initialize visualizer.

        Args:
            output_dir: Directory to save visualization outputs.
        """
        pass

    @abstractmethod
    def draw_detections(
        self,
        frame: np.ndarray,
        detections: list[dict[str, Any]],
        frame_idx: int = 0,
    ) -> np.ndarray:
        """Draw detection bounding boxes.

        Args:
            frame: Input frame (BGR format).
            detections: List of detection dicts with 'bbox' and 'confidence'.
            frame_idx: Frame index for display.

        Returns:
            Frame with drawn detections.
        """
        pass

    @abstractmethod
    def draw_tracking(
        self,
        frame: np.ndarray,
        tracks: list[dict[str, Any]],
        track_history: dict[int, list[tuple[int, int]]] | None = None,
        frame_idx: int = 0,
    ) -> np.ndarray:
        """Draw tracking results with trajectories.

        Args:
            frame: Input frame (BGR format).
            tracks: List of track dicts with 'track_id', 'bbox'.
            track_history: Dictionary of track trajectories.
            frame_idx: Frame index for display.

        Returns:
            Frame with tracking visualization.
        """
        pass

    @abstractmethod
    def draw_final_result(
        self,
        frame: np.ndarray,
        tracks: list[dict[str, Any]],
        team_sides: dict[int, str],
        roles: dict[int, str],
    ) -> np.ndarray:
        """Draw final result with team sides and roles.

        Args:
            frame: Input frame (BGR format).
            tracks: List of track dicts.
            team_sides: {track_id: 'left'/'right'/'referee'}.
            roles: {track_id: role}.

        Returns:
            Frame with final visualization.
        """
        pass

    def draw_calibration(
        self,
        frame: np.ndarray,
        keypoints: dict,
        lines: dict,
        homography: np.ndarray | None,
        image_width: int,
        image_height: int,
    ) -> np.ndarray:
        """Draw calibration results.

        Default implementation just returns the frame.
        Override in subclasses for detailed visualization.

        Args:
            frame: Input frame (BGR format).
            keypoints: Detected keypoints.
            lines: Detected line segments.
            homography: Image -> world homography matrix.
            image_width: Frame width.
            image_height: Frame height.

        Returns:
            Frame with calibration visualization.
        """
        return frame

    def draw_team_clustering(
        self,
        frame: np.ndarray,
        tracks: list[dict[str, Any]],
        team_clusters: dict[int, int],
    ) -> np.ndarray:
        """Draw team clustering results.

        Default implementation just returns the frame.
        Override in subclasses for detailed visualization.

        Args:
            frame: Input frame (BGR format).
            tracks: List of track dicts.
            team_clusters: {track_id: cluster_id (0 or 1)}.

        Returns:
            Frame with cluster visualization.
        """
        return frame

    def draw_radar_view(
        self,
        frame: np.ndarray,
        tracks: list[dict[str, Any]],
        team_sides: dict[int, str],
        homography: np.ndarray | None,
        position: str = 'bottom-right',
    ) -> np.ndarray:
        """Draw radar/minimap view of pitch.

        Default implementation just returns the frame.
        Override in subclasses for radar visualization.

        Args:
            frame: Input frame (BGR format).
            tracks: List of track dicts with 'bbox' and 'track_id'.
            team_sides: {track_id: team_side}.
            homography: Image -> world homography.
            position: Where to place radar.

        Returns:
            Frame with radar overlay.
        """
        return frame

    def draw_ball(
        self,
        frame: np.ndarray,
        ball_track: dict[str, Any] | None,
        trajectory: list[tuple[float, float]] | None = None,
        show_velocity: bool = True,
    ) -> np.ndarray:
        """Draw ball position and trajectory.

        Default implementation just returns the frame.
        Override in subclasses for ball visualization.

        Args:
            frame: Input frame (BGR format).
            ball_track: Ball track dict with 'center', 'bbox', 'velocity'.
            trajectory: List of recent (x, y) positions for trail.
            show_velocity: Whether to show velocity arrow.

        Returns:
            Frame with ball visualization.
        """
        return frame

    @property
    def name(self) -> str:
        """Return backend name for logging."""
        return self.__class__.__name__
