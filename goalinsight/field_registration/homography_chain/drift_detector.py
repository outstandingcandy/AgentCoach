"""Drift detection and confidence scoring for homography chains.

Monitors calibration quality and triggers re-anchoring when drift
exceeds thresholds or when higher-confidence frames become available.
"""

from collections import deque
from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class DriftMetrics:
    """Metrics for drift detection."""

    inlier_ratio: float
    reprojection_error: float
    translation_magnitude: float
    rotation_magnitude: float
    cumulative_drift: float
    confidence: float


class DriftDetector:
    """Monitor calibration drift and trigger re-anchoring.

    Tracks multiple metrics to detect when propagated calibration
    is drifting too far from ground truth and needs re-anchoring.
    """

    def __init__(
        self,
        drift_threshold: float = 50.0,
        inlier_threshold: float = 0.4,
        confidence_threshold: float = 0.3,
        history_size: int = 10,
    ):
        """Initialize drift detector.

        Args:
            drift_threshold: Max cumulative drift before re-anchor.
            inlier_threshold: Min inlier ratio for good match.
            confidence_threshold: Min confidence for valid calibration.
            history_size: Number of frames to track in history.
        """
        self.drift_threshold = drift_threshold
        self.inlier_threshold = inlier_threshold
        self.confidence_threshold = confidence_threshold
        self.history_size = history_size

        self.history: deque[DriftMetrics] = deque(maxlen=history_size)
        self.cumulative_drift = 0.0

    def update(
        self,
        H: np.ndarray,
        match_metadata: dict[str, Any] | None = None,
    ) -> DriftMetrics:
        """Update drift tracking with new homography.

        Args:
            H: Frame-to-frame homography.
            match_metadata: Feature matching metadata.

        Returns:
            Current drift metrics.
        """
        if H is None:
            metrics = DriftMetrics(
                inlier_ratio=0.0,
                reprojection_error=float("inf"),
                translation_magnitude=0.0,
                rotation_magnitude=0.0,
                cumulative_drift=self.cumulative_drift + 10.0,
                confidence=0.0,
            )
            self.cumulative_drift += 10.0
            self.history.append(metrics)
            return metrics

        # Extract metrics from homography
        translation_mag = np.sqrt(H[0, 2]**2 + H[1, 2]**2)
        rotation_mag = self._estimate_rotation(H)

        # Update cumulative drift
        drift_increment = translation_mag + rotation_mag * 10
        self.cumulative_drift += drift_increment

        # Get inlier ratio from metadata
        inlier_ratio = 0.5
        reprojection_error = 0.0
        if match_metadata:
            inlier_ratio = match_metadata.get("inlier_ratio", 0.5)

        # Compute confidence
        confidence = self._compute_confidence(
            inlier_ratio, self.cumulative_drift
        )

        metrics = DriftMetrics(
            inlier_ratio=inlier_ratio,
            reprojection_error=reprojection_error,
            translation_magnitude=translation_mag,
            rotation_magnitude=rotation_mag,
            cumulative_drift=self.cumulative_drift,
            confidence=confidence,
        )

        self.history.append(metrics)
        return metrics

    def _estimate_rotation(self, H: np.ndarray) -> float:
        """Estimate rotation angle from homography.

        Args:
            H: Homography matrix.

        Returns:
            Rotation angle in radians.
        """
        # Extract rotation from upper-left 2x2 block
        # For small rotations: H[0,1] ~ -sin(theta), H[1,0] ~ sin(theta)
        sin_theta = (H[1, 0] - H[0, 1]) / 2
        sin_theta = np.clip(sin_theta, -1, 1)
        return abs(np.arcsin(sin_theta))

    def _compute_confidence(
        self,
        inlier_ratio: float,
        cumulative_drift: float,
    ) -> float:
        """Compute confidence score.

        Args:
            inlier_ratio: Feature match inlier ratio.
            cumulative_drift: Total accumulated drift.

        Returns:
            Confidence score [0, 1].
        """
        # Inlier contribution
        inlier_conf = min(1.0, inlier_ratio / 0.7)

        # Drift penalty
        drift_factor = np.exp(-cumulative_drift / self.drift_threshold)

        return inlier_conf * drift_factor

    def should_re_anchor(self) -> bool:
        """Check if re-anchoring is needed.

        Returns:
            True if drift exceeds thresholds.
        """
        if not self.history:
            return False

        latest = self.history[-1]

        # Check cumulative drift
        if latest.cumulative_drift > self.drift_threshold:
            return True

        # Check confidence
        if latest.confidence < self.confidence_threshold:
            return True

        # Check recent inlier ratio trend
        if len(self.history) >= 3:
            recent_inliers = [m.inlier_ratio for m in list(self.history)[-3:]]
            if np.mean(recent_inliers) < self.inlier_threshold:
                return True

        return False

    def check_anchor_opportunity(
        self,
        external_confidence: float,
        external_frame_idx: int,
    ) -> bool:
        """Check if external calibration should become new anchor.

        Args:
            external_confidence: Confidence from external calibrator.
            external_frame_idx: Frame index of external calibration.

        Returns:
            True if external should be used as new anchor.
        """
        if not self.history:
            return external_confidence > self.confidence_threshold

        current_conf = self.history[-1].confidence

        # Re-anchor if external is significantly better
        return external_confidence > current_conf + 0.2

    def get_average_confidence(self, n_frames: int = 5) -> float:
        """Get average confidence over recent frames.

        Args:
            n_frames: Number of recent frames to average.

        Returns:
            Average confidence score.
        """
        if not self.history:
            return 0.0

        recent = list(self.history)[-n_frames:]
        return np.mean([m.confidence for m in recent])

    def get_drift_rate(self, n_frames: int = 5) -> float:
        """Get recent drift rate (drift per frame).

        Args:
            n_frames: Number of frames for rate calculation.

        Returns:
            Drift rate (drift units per frame).
        """
        if len(self.history) < 2:
            return 0.0

        recent = list(self.history)[-n_frames:]
        if len(recent) < 2:
            return 0.0

        drift_diff = recent[-1].cumulative_drift - recent[0].cumulative_drift
        return drift_diff / (len(recent) - 1)

    def reset(self) -> None:
        """Reset drift tracking."""
        self.history.clear()
        self.cumulative_drift = 0.0

    def reset_cumulative(self) -> None:
        """Reset cumulative drift only (keep history)."""
        self.cumulative_drift = 0.0
