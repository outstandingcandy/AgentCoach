"""Kalman filter for bbox tracking.

State: ``[cx, cy, aspect_ratio, height, vx, vy, va, vh]``. Process and
measurement noise scale with target height so the covariance is
properly calibrated for Mahalanobis gating (DeepSORT/StrongSORT
convention).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.linalg


# Chi-squared inverse CDF at 95% confidence for 1-9 degrees of freedom.
# Used for Mahalanobis gating thresholds.
chi2inv95 = {
    1: 3.8415, 2: 5.9915, 3: 7.8147, 4: 9.4877,
    5: 11.070, 6: 12.592, 7: 14.067, 8: 15.507, 9: 16.919,
}


@dataclass
class KalmanState:
    """Kalman filter state for bounding box tracking."""
    mean: np.ndarray
    covariance: np.ndarray


class KalmanFilter:
    """Kalman filter for bbox tracking with height-dependent noise."""

    def __init__(self, frame_interval: float = 1.0):
        ndim = 4

        # State transition matrix (constant velocity model)
        self.F = np.eye(2 * ndim)
        for i in range(ndim):
            self.F[i, ndim + i] = frame_interval

        # Observation matrix (we observe cx, cy, a, h)
        self.H = np.eye(ndim, 2 * ndim)

        # Noise weights (relative to target height), scaled by frame interval.
        # Base values from DeepSORT assume 30fps (dt=1); scale for actual dt.
        self._std_weight_position = (1.0 / 20) * frame_interval
        self._std_weight_velocity = (1.0 / 160) * frame_interval

    def initiate(self, bbox: list) -> KalmanState:
        """Initialize state from bounding box [x1, y1, x2, y2]."""
        x1, y1, x2, y2 = bbox
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2
        w = x2 - x1
        h = max(y2 - y1, 1.0)
        a = w / h

        mean = np.array([cx, cy, a, h, 0.0, 0.0, 0.0, 0.0])
        std = [
            2 * self._std_weight_position * h,
            2 * self._std_weight_position * h,
            1e-2,
            2 * self._std_weight_position * h,
            10 * self._std_weight_velocity * h,
            10 * self._std_weight_velocity * h,
            1e-5,
            10 * self._std_weight_velocity * h,
        ]
        covariance = np.diag(np.square(std))
        return KalmanState(mean=mean, covariance=covariance)

    def predict(self, state: KalmanState) -> KalmanState:
        """Predict next state with height-dependent process noise."""
        h = state.mean[3]
        std_pos = [
            self._std_weight_position * h,
            self._std_weight_position * h,
            1e-2,
            self._std_weight_position * h,
        ]
        std_vel = [
            self._std_weight_velocity * h,
            self._std_weight_velocity * h,
            1e-5,
            self._std_weight_velocity * h,
        ]
        motion_cov = np.diag(np.square(np.r_[std_pos, std_vel]))

        mean = self.F @ state.mean
        covariance = self.F @ state.covariance @ self.F.T + motion_cov
        return KalmanState(mean=mean, covariance=covariance)

    def _project(self, state: KalmanState):
        """Project state to measurement space."""
        h = state.mean[3]
        std = [
            self._std_weight_position * h,
            self._std_weight_position * h,
            1e-1,
            self._std_weight_position * h,
        ]
        innovation_cov = np.diag(np.square(std))

        mean = self.H @ state.mean
        covariance = self.H @ state.covariance @ self.H.T + innovation_cov
        return mean, covariance

    def update(self, state: KalmanState, bbox: list) -> KalmanState:
        """Update state with measurement."""
        x1, y1, x2, y2 = bbox
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2
        w = x2 - x1
        h = max(y2 - y1, 1.0)
        a = w / h

        measurement = np.array([cx, cy, a, h])
        projected_mean, projected_cov = self._project(state)

        K = state.covariance @ self.H.T @ np.linalg.inv(projected_cov)
        mean = state.mean + K @ (measurement - projected_mean)
        covariance = (np.eye(8) - K @ self.H) @ state.covariance

        return KalmanState(mean=mean, covariance=covariance)

    def gating_distance(
        self, state: KalmanState, measurement: np.ndarray,
    ) -> float:
        """Squared Mahalanobis distance between predicted state and measurement."""
        projected_mean, projected_cov = self._project(state)
        diff = measurement - projected_mean
        chol = np.linalg.cholesky(projected_cov)
        z = scipy.linalg.solve_triangular(chol, diff, lower=True)
        return float(z @ z)

    def state_to_bbox(
        self, state: KalmanState, img_w: int = 1920, img_h: int = 1080,
    ) -> list:
        """Convert state to bounding box with bounds clipping."""
        cx, cy, a, h = state.mean[:4]
        w = a * h

        # Ensure positive dimensions
        w = max(1, abs(w))
        h = max(1, abs(h))

        x1 = cx - w / 2
        y1 = cy - h / 2
        x2 = cx + w / 2
        y2 = cy + h / 2

        # Clip to image bounds
        x1 = max(0, min(img_w - 1, x1))
        y1 = max(0, min(img_h - 1, y1))
        x2 = max(x1 + 1, min(img_w, x2))
        y2 = max(y1 + 1, min(img_h, y2))

        return [x1, y1, x2, y2]
