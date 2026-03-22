"""Shot boundary detection for identifying camera cuts in video.

Uses a multi-method fusion approach with adaptive thresholding:
1. Frame difference detection (grayscale pixel-level difference)
2. Histogram difference (HSV color histogram comparison)
"""

from collections import deque
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

from . import ShotBoundary


class ShotDetector:
    """Detects shot boundaries (camera cuts) in video.

    Uses adaptive thresholding with multi-method fusion for robust detection.
    """

    def __init__(self, config: dict | None = None):
        """Initialize shot detector.

        Args:
            config: Configuration dict with shot_detection settings.
        """
        config = config or {}
        sd_config = config.get("shot_detection", {})

        self.method = sd_config.get("method", "adaptive")

        # Frame difference settings
        fd_config = sd_config.get("frame_diff", {})
        self.adaptive_ratio = fd_config.get("adaptive_ratio", 3.0)
        self.window_size = fd_config.get("window_size", 50)

        # Histogram settings
        hist_config = sd_config.get("histogram", {})
        self.color_space = hist_config.get("color_space", "hsv")
        self.hist_bins = hist_config.get("bins", [8, 12, 3])

        # Minimum shot duration (to avoid false positives from flashes)
        self.min_shot_duration = sd_config.get("min_shot_duration", 1.0)

        # Detection weights
        self.frame_diff_weight = 0.7
        self.histogram_weight = 0.3

    def detect(self, video_path: str | Path) -> list[ShotBoundary]:
        """Detect shot boundaries in video.

        Args:
            video_path: Path to video file.

        Returns:
            List of detected shot boundaries.
        """
        video_path = Path(video_path)
        if not video_path.exists():
            raise FileNotFoundError(f"Video not found: {video_path}")

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {video_path}")

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)

        # Minimum frames between shots
        min_shot_frames = int(self.min_shot_duration * fps)

        # Compute all frame differences first
        frame_diffs, hist_diffs = self._compute_differences(cap, total_frames)
        cap.release()

        if len(frame_diffs) == 0:
            return []

        # Detect boundaries based on method
        if self.method == "adaptive":
            boundaries = self._detect_adaptive(
                frame_diffs, hist_diffs, min_shot_frames
            )
        elif self.method == "histogram":
            boundaries = self._detect_histogram_only(hist_diffs, min_shot_frames)
        elif self.method == "fixed":
            boundaries = self._detect_fixed_threshold(
                frame_diffs, hist_diffs, min_shot_frames
            )
        else:
            boundaries = self._detect_adaptive(
                frame_diffs, hist_diffs, min_shot_frames
            )

        return boundaries

    def _compute_differences(
        self, cap: cv2.VideoCapture, total_frames: int
    ) -> tuple[list[float], list[float]]:
        """Compute frame and histogram differences for all frames.

        Args:
            cap: OpenCV video capture object.
            total_frames: Total number of frames.

        Returns:
            Tuple of (frame_diffs, hist_diffs) lists.
        """
        frame_diffs = []
        hist_diffs = []

        prev_frame = None
        prev_hist = None

        for _ in tqdm(range(total_frames), desc="Stage 0: Computing differences"):
            ret, frame = cap.read()
            if not ret:
                break

            # Convert to grayscale for frame difference
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Compute frame difference
            if prev_frame is not None:
                diff = cv2.absdiff(gray, prev_frame)
                frame_diff = np.mean(diff)
                frame_diffs.append(frame_diff)

                # Compute histogram difference
                hist = self._compute_histogram(frame)
                hist_diff = cv2.compareHist(prev_hist, hist, cv2.HISTCMP_BHATTACHARYYA)
                hist_diffs.append(hist_diff)
                prev_hist = hist
            else:
                prev_hist = self._compute_histogram(frame)

            prev_frame = gray.copy()

        return frame_diffs, hist_diffs

    def _compute_histogram(self, frame: np.ndarray) -> np.ndarray:
        """Compute color histogram for frame.

        Args:
            frame: BGR image.

        Returns:
            Normalized histogram.
        """
        if self.color_space == "hsv":
            converted = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        else:
            converted = frame

        # Compute 3D histogram
        hist = cv2.calcHist(
            [converted],
            [0, 1, 2],
            None,
            self.hist_bins,
            [0, 180, 0, 256, 0, 256] if self.color_space == "hsv" else [0, 256, 0, 256, 0, 256],
        )
        cv2.normalize(hist, hist)
        return hist.flatten()

    def _detect_adaptive(
        self,
        frame_diffs: list[float],
        hist_diffs: list[float],
        min_shot_frames: int,
    ) -> list[ShotBoundary]:
        """Detect shot boundaries using adaptive thresholding.

        Uses sliding window to compute local statistics and adaptive threshold.

        Args:
            frame_diffs: Frame difference values.
            hist_diffs: Histogram difference values.
            min_shot_frames: Minimum frames between shots.

        Returns:
            List of detected shot boundaries.
        """
        boundaries = []
        n = len(frame_diffs)

        # Use sliding window for local statistics
        window = deque(maxlen=self.window_size)
        last_shot_frame = -min_shot_frames

        for i in range(n):
            # Add to window
            window.append(frame_diffs[i])

            # Need enough samples for statistics
            if len(window) < self.window_size // 2:
                continue

            # Compute local statistics
            local_mean = np.mean(window)
            local_std = np.std(window)

            # Adaptive threshold
            frame_threshold = local_mean + self.adaptive_ratio * local_std

            # Combine scores
            frame_score = frame_diffs[i] / max(frame_threshold, 1e-6)
            hist_score = hist_diffs[i] / 0.5 if hist_diffs[i] > 0.3 else 0

            combined_score = (
                self.frame_diff_weight * frame_score
                + self.histogram_weight * hist_score
            )

            # Detect shot boundary
            frame_idx = i + 1  # +1 because differences start from frame 1
            if (
                combined_score > 1.0
                and frame_idx - last_shot_frame >= min_shot_frames
            ):
                boundaries.append(
                    ShotBoundary(
                        frame_idx=frame_idx,
                        score=min(combined_score, 1.0),
                        method="adaptive",
                    )
                )
                last_shot_frame = frame_idx

        return boundaries

    def _detect_histogram_only(
        self, hist_diffs: list[float], min_shot_frames: int
    ) -> list[ShotBoundary]:
        """Detect shot boundaries using histogram difference only.

        Args:
            hist_diffs: Histogram difference values.
            min_shot_frames: Minimum frames between shots.

        Returns:
            List of detected shot boundaries.
        """
        boundaries = []
        threshold = 0.5  # Bhattacharyya distance threshold
        last_shot_frame = -min_shot_frames

        for i, diff in enumerate(hist_diffs):
            frame_idx = i + 1
            if diff > threshold and frame_idx - last_shot_frame >= min_shot_frames:
                boundaries.append(
                    ShotBoundary(
                        frame_idx=frame_idx,
                        score=min(diff / threshold, 1.0),
                        method="histogram",
                    )
                )
                last_shot_frame = frame_idx

        return boundaries

    def _detect_fixed_threshold(
        self,
        frame_diffs: list[float],
        hist_diffs: list[float],
        min_shot_frames: int,
    ) -> list[ShotBoundary]:
        """Detect shot boundaries using fixed thresholds.

        Args:
            frame_diffs: Frame difference values.
            hist_diffs: Histogram difference values.
            min_shot_frames: Minimum frames between shots.

        Returns:
            List of detected shot boundaries.
        """
        boundaries = []
        frame_threshold = 30.0
        hist_threshold = 0.5
        last_shot_frame = -min_shot_frames

        for i in range(len(frame_diffs)):
            frame_score = frame_diffs[i] / frame_threshold
            hist_score = hist_diffs[i] / hist_threshold

            combined_score = (
                self.frame_diff_weight * frame_score
                + self.histogram_weight * hist_score
            )

            frame_idx = i + 1
            if combined_score > 1.0 and frame_idx - last_shot_frame >= min_shot_frames:
                boundaries.append(
                    ShotBoundary(
                        frame_idx=frame_idx,
                        score=min(combined_score, 1.0),
                        method="fixed",
                    )
                )
                last_shot_frame = frame_idx

        return boundaries
