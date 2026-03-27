"""Ball tracking using Kalman filter.

Specialized tracker for soccer ball with:
- Position and velocity state model
- No ReID features (ball has uniform appearance)
- Longer max_age for occlusion handling
- Trajectory history and velocity estimation
"""

from collections import deque
from enum import Enum
from typing import Any

import numpy as np


class BallStatus(Enum):
    """Ball track status."""
    TENTATIVE = "tentative"
    CONFIRMED = "confirmed"
    LOST = "lost"


class BallKalmanFilter:
    """Kalman filter for ball tracking with position and velocity.

    State vector: [x, y, vx, vy]
    Measurement vector: [x, y]
    """

    def __init__(self, initial_position: tuple[float, float]):
        """Initialize Kalman filter with initial position.

        Args:
            initial_position: (x, y) initial ball position.
        """
        # State: [x, y, vx, vy]
        self.state = np.array([
            initial_position[0],
            initial_position[1],
            0.0,  # vx
            0.0,  # vy
        ], dtype=np.float64)

        # State transition matrix (constant velocity model)
        self.F = np.array([
            [1, 0, 1, 0],
            [0, 1, 0, 1],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ], dtype=np.float64)

        # Measurement matrix (observe position only)
        self.H = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
        ], dtype=np.float64)

        # Process noise covariance (tuned for ball motion)
        q_pos = 5.0  # Position uncertainty
        q_vel = 10.0  # Velocity uncertainty (ball can accelerate quickly)
        self.Q = np.diag([q_pos, q_pos, q_vel, q_vel])

        # Measurement noise covariance
        r = 5.0  # Detection position uncertainty
        self.R = np.diag([r, r])

        # Initial state covariance
        self.P = np.diag([10.0, 10.0, 100.0, 100.0])

    def predict(self) -> np.ndarray:
        """Predict next state.

        Returns:
            Predicted position [x, y].
        """
        # State prediction
        self.state = self.F @ self.state
        # Covariance prediction
        self.P = self.F @ self.P @ self.F.T + self.Q

        return self.state[:2].copy()

    def update(self, measurement: tuple[float, float]) -> np.ndarray:
        """Update state with measurement.

        Args:
            measurement: Observed position (x, y).

        Returns:
            Updated position [x, y].
        """
        z = np.array(measurement, dtype=np.float64)

        # Innovation
        y = z - self.H @ self.state

        # Innovation covariance
        S = self.H @ self.P @ self.H.T + self.R

        # Kalman gain
        K = self.P @ self.H.T @ np.linalg.inv(S)

        # State update
        self.state = self.state + K @ y

        # Covariance update
        I = np.eye(4)
        self.P = (I - K @ self.H) @ self.P

        return self.state[:2].copy()

    @property
    def position(self) -> tuple[float, float]:
        """Get current position estimate."""
        return (float(self.state[0]), float(self.state[1]))

    @property
    def velocity(self) -> tuple[float, float]:
        """Get current velocity estimate."""
        return (float(self.state[2]), float(self.state[3]))


class BallTrack:
    """Single ball track."""

    def __init__(
        self,
        track_id: int,
        initial_detection: dict[str, Any],
        max_age: int = 60,
        n_init: int = 2,
    ):
        """Initialize ball track.

        Args:
            track_id: Unique track identifier.
            initial_detection: First detection dict with 'center', 'bbox'.
            max_age: Maximum frames to keep track without detection.
            n_init: Number of detections to confirm track.
        """
        self.track_id = track_id
        self.max_age = max_age
        self.n_init = n_init

        # Initialize Kalman filter
        center = initial_detection["center"]
        self.kf = BallKalmanFilter((center[0], center[1]))

        # Track state
        self.status = BallStatus.TENTATIVE
        self.hits = 1
        self.age = 0
        self.time_since_update = 0

        # Store last detection
        self.last_detection = initial_detection.copy()

        # Trajectory history (for visualization)
        self.trajectory: deque[tuple[float, float]] = deque(maxlen=60)
        self.trajectory.append((center[0], center[1]))

    def predict(self) -> tuple[float, float]:
        """Predict next position.

        Returns:
            Predicted (x, y) position.
        """
        predicted_pos = self.kf.predict()
        self.age += 1
        self.time_since_update += 1
        return (float(predicted_pos[0]), float(predicted_pos[1]))

    def update(self, detection: dict[str, Any]) -> None:
        """Update track with new detection.

        Args:
            detection: Detection dict with 'center', 'bbox'.
        """
        center = detection["center"]
        self.kf.update((center[0], center[1]))

        self.hits += 1
        self.time_since_update = 0
        self.last_detection = detection.copy()

        # Update trajectory
        self.trajectory.append((center[0], center[1]))

        # Update status
        if self.status == BallStatus.TENTATIVE and self.hits >= self.n_init:
            self.status = BallStatus.CONFIRMED

    def mark_lost(self) -> None:
        """Mark track as lost."""
        self.status = BallStatus.LOST

    def is_expired(self) -> bool:
        """Check if track should be deleted."""
        return self.time_since_update > self.max_age

    def to_dict(self) -> dict[str, Any]:
        """Convert track to dictionary.

        Returns:
            Track information dictionary.
        """
        pos = self.kf.position
        predicted = self.time_since_update > 0

        return {
            "track_id": self.track_id,
            "center": list(pos),
            "bbox": self.last_detection.get("bbox", [pos[0]-10, pos[1]-10, pos[0]+10, pos[1]+10]),
            "confidence": 0.0 if predicted else self.last_detection.get("confidence", 0.0),
            "status": self.status.value,
            "velocity": [0.0, 0.0] if predicted else list(self.kf.velocity),
            "pitch_position": self.last_detection.get("pitch_position"),
            "predicted": predicted,
            "time_since_update": self.time_since_update,
        }

    def get_trajectory(self, num_frames: int = 30) -> list[tuple[float, float]]:
        """Get recent trajectory points.

        Args:
            num_frames: Number of recent frames to return.

        Returns:
            List of (x, y) positions.
        """
        traj = list(self.trajectory)
        return traj[-num_frames:]


class BallTracker:
    """Multi-object tracker specialized for soccer ball.

    Features:
    - Kalman filter based motion prediction
    - No ReID (ball appearance is uniform)
    - Hungarian matching by position distance
    - Single primary ball selection
    """

    def __init__(self, config: dict[str, Any] | None = None):
        """Initialize ball tracker.

        Args:
            config: Tracker configuration. max_age is in seconds and converted
                to frames using the fps value from config.
        """
        self.config = config or {}

        # Convert max_age from seconds to frames
        fps = self.config.get("fps", 10)
        max_age_sec = self.config.get("max_age", 1.5)
        self.max_age = max(1, int(max_age_sec * fps))

        self.n_init = self.config.get("n_init", 2)
        self.max_position_distance = self.config.get("max_position_distance", 200)
        self.trajectory_length = self.config.get("trajectory_length", 30)

        # Frame bounds for out-of-bounds detection
        self.frame_width = self.config.get("frame_width", 1920)
        self.frame_height = self.config.get("frame_height", 1080)

        self.tracks: list[BallTrack] = []
        self.next_id = 1

    def update(self, detections: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Update tracker with new detections.

        Args:
            detections: List of ball detections with 'center', 'bbox'.

        Returns:
            List of active track dictionaries.
        """
        # Predict existing tracks, mark out-of-bounds as lost
        predicted_positions = []
        for track in self.tracks:
            pos = track.predict()
            if (pos[0] < -50 or pos[0] > self.frame_width + 50
                    or pos[1] < -50 or pos[1] > self.frame_height + 50):
                track.mark_lost()
            predicted_positions.append(pos)

        # Match detections to tracks
        if self.tracks and detections:
            cost_matrix = self._compute_cost_matrix(
                predicted_positions, detections
            )
            matches, unmatched_tracks, unmatched_detections = self._hungarian_match(
                cost_matrix
            )
        else:
            matches = []
            unmatched_tracks = list(range(len(self.tracks)))
            unmatched_detections = list(range(len(detections)))

        # Update matched tracks
        for track_idx, det_idx in matches:
            self.tracks[track_idx].update(detections[det_idx])

        # Create new tracks for unmatched detections
        for det_idx in unmatched_detections:
            self._create_track(detections[det_idx])

        # Mark unmatched tracks
        for track_idx in unmatched_tracks:
            if self.tracks[track_idx].time_since_update > self.max_age // 2:
                self.tracks[track_idx].mark_lost()

        # Remove expired tracks
        self.tracks = [t for t in self.tracks if not t.is_expired()]

        # Return active tracks
        return [t.to_dict() for t in self.tracks if t.status != BallStatus.LOST]

    def _compute_cost_matrix(
        self,
        predicted_positions: list[tuple[float, float]],
        detections: list[dict[str, Any]],
    ) -> np.ndarray:
        """Compute cost matrix for Hungarian matching.

        Args:
            predicted_positions: List of predicted (x, y) positions.
            detections: List of detections.

        Returns:
            Cost matrix (tracks x detections).
        """
        n_tracks = len(predicted_positions)
        n_dets = len(detections)
        cost_matrix = np.full((n_tracks, n_dets), self.max_position_distance * 2)

        for i, pred_pos in enumerate(predicted_positions):
            for j, det in enumerate(detections):
                det_pos = det["center"]
                dist = np.sqrt(
                    (pred_pos[0] - det_pos[0]) ** 2 +
                    (pred_pos[1] - det_pos[1]) ** 2
                )
                if dist < self.max_position_distance:
                    cost_matrix[i, j] = dist

        return cost_matrix

    def _hungarian_match(
        self,
        cost_matrix: np.ndarray,
    ) -> tuple[list[tuple[int, int]], list[int], list[int]]:
        """Perform Hungarian matching.

        Args:
            cost_matrix: Cost matrix (tracks x detections).

        Returns:
            Tuple of (matches, unmatched_tracks, unmatched_detections).
        """
        try:
            from scipy.optimize import linear_sum_assignment
        except ImportError:
            # Fallback to greedy matching
            return self._greedy_match(cost_matrix)

        n_tracks, n_dets = cost_matrix.shape
        if n_tracks == 0 or n_dets == 0:
            return [], list(range(n_tracks)), list(range(n_dets))

        row_indices, col_indices = linear_sum_assignment(cost_matrix)

        matches = []
        unmatched_tracks = list(range(n_tracks))
        unmatched_detections = list(range(n_dets))

        for row, col in zip(row_indices, col_indices):
            if cost_matrix[row, col] < self.max_position_distance:
                matches.append((row, col))
                unmatched_tracks.remove(row)
                unmatched_detections.remove(col)

        return matches, unmatched_tracks, unmatched_detections

    def _greedy_match(
        self,
        cost_matrix: np.ndarray,
    ) -> tuple[list[tuple[int, int]], list[int], list[int]]:
        """Greedy matching fallback when scipy not available."""
        n_tracks, n_dets = cost_matrix.shape
        matches = []
        matched_tracks = set()
        matched_dets = set()

        # Sort by cost
        indices = np.argsort(cost_matrix.flatten())
        for idx in indices:
            row = idx // n_dets
            col = idx % n_dets

            if row in matched_tracks or col in matched_dets:
                continue
            if cost_matrix[row, col] >= self.max_position_distance:
                break

            matches.append((row, col))
            matched_tracks.add(row)
            matched_dets.add(col)

        unmatched_tracks = [i for i in range(n_tracks) if i not in matched_tracks]
        unmatched_dets = [i for i in range(n_dets) if i not in matched_dets]

        return matches, unmatched_tracks, unmatched_dets

    def _create_track(self, detection: dict[str, Any]) -> None:
        """Create new track from detection.

        Args:
            detection: Detection dict.
        """
        track = BallTrack(
            track_id=self.next_id,
            initial_detection=detection,
            max_age=self.max_age,
            n_init=self.n_init,
        )
        self.tracks.append(track)
        self.next_id += 1

    def get_primary_ball(self) -> dict[str, Any] | None:
        """Get the primary (most confident) ball track.

        Returns:
            Primary ball track dict or None.
        """
        confirmed = [t for t in self.tracks if t.status == BallStatus.CONFIRMED]
        if not confirmed:
            # Fall back to tentative tracks
            confirmed = [t for t in self.tracks if t.status != BallStatus.LOST]

        if not confirmed:
            return None

        # Select by most recent update and highest confidence
        def score(track: BallTrack) -> float:
            recency = 1.0 / (1.0 + track.time_since_update)
            conf = track.last_detection.get("confidence", 0.5)
            return recency * 0.6 + conf * 0.4

        best = max(confirmed, key=score)
        return best.to_dict()

    def get_ball_trajectory(self, num_frames: int = 30) -> list[tuple[float, float]]:
        """Get trajectory of primary ball.

        Args:
            num_frames: Number of recent frames.

        Returns:
            List of (x, y) positions.
        """
        primary = self.get_primary_ball()
        if not primary:
            return []

        for track in self.tracks:
            if track.track_id == primary["track_id"]:
                return track.get_trajectory(num_frames)

        return []

    def get_ball_velocity(self) -> tuple[float, float] | None:
        """Get velocity of primary ball.

        Returns:
            (vx, vy) velocity or None.
        """
        primary = self.get_primary_ball()
        if not primary or "velocity" not in primary:
            return None

        return tuple(primary["velocity"])

    def get_pitch_position(self, homography: np.ndarray | None = None) -> tuple[float, float] | None:
        """Get pitch position of primary ball.

        Args:
            homography: Image -> world homography (optional, uses stored if available).

        Returns:
            (x, y) pitch position in meters or None.
        """
        primary = self.get_primary_ball()
        if not primary:
            return None

        # Use stored pitch position if available
        if primary.get("pitch_position"):
            return tuple(primary["pitch_position"])

        # Compute if homography provided
        if homography is not None:
            center = primary["center"]
            pt_h = np.array([center[0], center[1], 1.0])
            world_h = homography @ pt_h

            if abs(world_h[2]) > 1e-6:
                return (float(world_h[0] / world_h[2]), float(world_h[1] / world_h[2]))

        return None

    def reset(self) -> None:
        """Reset tracker state."""
        self.tracks = []
        self.next_id = 1
