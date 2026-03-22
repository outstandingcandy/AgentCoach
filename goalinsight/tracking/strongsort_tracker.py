"""StrongSORT-style tracking with external ReID features.

This implementation provides proper multi-object tracking with:
1. Kalman filter for motion prediction
2. External ReID features for appearance matching
3. Cascaded matching (appearance -> IoU)
4. Track management with confirmation logic
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist


class TrackStatus(Enum):
    """Track lifecycle status."""
    TENTATIVE = 1  # Not yet confirmed
    CONFIRMED = 2  # Confirmed track
    DELETED = 3    # Marked for deletion


@dataclass
class KalmanState:
    """Kalman filter state for bounding box tracking."""
    # State: [cx, cy, aspect_ratio, height, vx, vy, va, vh]
    mean: np.ndarray
    covariance: np.ndarray


@dataclass
class Track:
    """Single object track."""
    track_id: int
    status: TrackStatus = TrackStatus.TENTATIVE
    hits: int = 1
    age: int = 1
    time_since_update: int = 0

    # Kalman state
    kalman_state: KalmanState | None = None

    # Appearance features (ReID embeddings)
    features: list = field(default_factory=list)
    smooth_feature: np.ndarray | None = None

    # Detection info
    bbox: list = field(default_factory=list)  # [x1, y1, x2, y2]
    confidence: float = 0.0
    class_id: int = 0

    # Attributes
    team: str | None = None
    jersey_number: int | None = None
    role: str = "player"

    def update_feature(self, feature: np.ndarray, alpha: float = 0.9):
        """Update smooth feature with EMA."""
        self.features.append(feature)
        # Keep only last 100 features
        if len(self.features) > 100:
            self.features = self.features[-100:]

        if self.smooth_feature is None:
            self.smooth_feature = feature.copy()
        else:
            self.smooth_feature = alpha * self.smooth_feature + (1 - alpha) * feature
            # Normalize
            self.smooth_feature /= np.linalg.norm(self.smooth_feature) + 1e-8

    def get_mean_feature(self) -> np.ndarray | None:
        """Get mean of all features."""
        if not self.features:
            return None
        return np.mean(self.features, axis=0)


class KalmanFilter:
    """Simple Kalman filter for bounding box tracking."""

    def __init__(self):
        # Motion model: constant velocity
        self.dt = 1.0

        # State transition matrix
        self.F = np.eye(8)
        self.F[0, 4] = self.dt  # cx += vx * dt
        self.F[1, 5] = self.dt  # cy += vy * dt
        self.F[2, 6] = self.dt  # a += va * dt
        self.F[3, 7] = self.dt  # h += vh * dt

        # Observation matrix (we observe cx, cy, a, h)
        self.H = np.eye(4, 8)

        # Process noise
        self.Q = np.eye(8) * 0.01
        self.Q[4:, 4:] *= 10  # Higher noise for velocities

        # Measurement noise
        self.R = np.eye(4) * 1.0

    def initiate(self, bbox: list) -> KalmanState:
        """Initialize state from bounding box [x1, y1, x2, y2]."""
        x1, y1, x2, y2 = bbox
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2
        w = x2 - x1
        h = y2 - y1
        a = w / h if h > 0 else 1.0

        mean = np.array([cx, cy, a, h, 0, 0, 0, 0])
        covariance = np.eye(8) * 10
        covariance[4:, 4:] *= 100  # Higher uncertainty for velocities

        return KalmanState(mean=mean, covariance=covariance)

    def predict(self, state: KalmanState) -> KalmanState:
        """Predict next state."""
        mean = self.F @ state.mean
        covariance = self.F @ state.covariance @ self.F.T + self.Q
        return KalmanState(mean=mean, covariance=covariance)

    def update(self, state: KalmanState, bbox: list) -> KalmanState:
        """Update state with measurement."""
        x1, y1, x2, y2 = bbox
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2
        w = x2 - x1
        h = y2 - y1
        a = w / h if h > 0 else 1.0

        measurement = np.array([cx, cy, a, h])

        # Kalman update
        S = self.H @ state.covariance @ self.H.T + self.R
        K = state.covariance @ self.H.T @ np.linalg.inv(S)

        mean = state.mean + K @ (measurement - self.H @ state.mean)
        covariance = (np.eye(8) - K @ self.H) @ state.covariance

        return KalmanState(mean=mean, covariance=covariance)

    def state_to_bbox(self, state: KalmanState, img_w: int = 1920, img_h: int = 1080) -> list:
        """Convert state to bounding box with bounds checking."""
        cx, cy, a, h = state.mean[:4]
        w = a * h

        # Ensure positive dimensions
        w = max(1, abs(w))
        h = max(1, abs(h))

        x1 = cx - w/2
        y1 = cy - h/2
        x2 = cx + w/2
        y2 = cy + h/2

        # Clip to image bounds
        x1 = max(0, min(img_w - 1, x1))
        y1 = max(0, min(img_h - 1, y1))
        x2 = max(x1 + 1, min(img_w, x2))
        y2 = max(y1 + 1, min(img_h, y2))

        return [x1, y1, x2, y2]


class StrongSORTTracker:
    """StrongSORT-style multi-object tracker.

    Features:
    - Kalman filter motion prediction
    - External ReID features for appearance matching
    - Cascaded matching strategy
    - EMA feature smoothing
    """

    def __init__(self, config: dict[str, Any] | None = None):
        """Initialize tracker.

        Args:
            config: Tracker configuration with keys:
                - max_age: Max frames to keep unmatched track (default: 30)
                - n_init: Min hits to confirm track (default: 3)
                - max_iou_distance: Max IoU distance for matching (default: 0.7)
                - max_cosine_distance: Max cosine distance for ReID (default: 0.3)
                - feature_alpha: EMA alpha for feature smoothing (default: 0.9)
        """
        config = config or {}
        self.max_age = config.get("max_age", 30)
        self.n_init = config.get("n_init", 3)
        self.max_iou_distance = config.get("max_iou_distance", 0.7)
        self.max_cosine_distance = config.get("max_cosine_distance", 0.3)
        self.feature_alpha = config.get("feature_alpha", 0.9)

        self.kalman = KalmanFilter()
        self.tracks: list[Track] = []
        self.next_id = 1
        self.img_w = 1920  # Default, can be updated
        self.img_h = 1080

    def reset(self):
        """Reset tracker state."""
        self.tracks = []
        self.next_id = 1

    def predict(self):
        """Predict track states for current frame."""
        for track in self.tracks:
            if track.kalman_state is not None:
                track.kalman_state = self.kalman.predict(track.kalman_state)
                track.bbox = self.kalman.state_to_bbox(
                    track.kalman_state, self.img_w, self.img_h
                )

    def update(
        self,
        detections: list[dict[str, Any]],
        embeddings: np.ndarray | None = None,
    ) -> list[dict[str, Any]]:
        """Update tracker with new detections.

        Args:
            detections: List of detection dicts with 'bbox', 'confidence', etc.
            embeddings: ReID embeddings for each detection, shape (N, D).

        Returns:
            List of confirmed tracks with track_id, bbox, etc.
        """
        # Predict existing tracks
        self.predict()

        # Split tracks by status
        confirmed_tracks = [t for t in self.tracks if t.status == TrackStatus.CONFIRMED]
        tentative_tracks = [t for t in self.tracks if t.status == TrackStatus.TENTATIVE]

        # Cascaded matching
        # Step 1: Match confirmed tracks using appearance (ReID)
        unmatched_tracks_idx = list(range(len(confirmed_tracks)))
        unmatched_detections_idx = list(range(len(detections)))
        matches = []

        if embeddings is not None and len(confirmed_tracks) > 0 and len(detections) > 0:
            # Get track features
            track_features = []
            valid_track_idx = []
            for i, track in enumerate(confirmed_tracks):
                if track.smooth_feature is not None:
                    track_features.append(track.smooth_feature)
                    valid_track_idx.append(i)

            if track_features and len(embeddings) > 0:
                track_features = np.array(track_features)

                # Compute cosine distance
                cost_matrix = cdist(track_features, embeddings, metric='cosine')

                # Apply threshold
                cost_matrix[cost_matrix > self.max_cosine_distance] = 1e5

                # Hungarian matching
                if cost_matrix.size > 0:
                    row_indices, col_indices = linear_sum_assignment(cost_matrix)

                    for row, col in zip(row_indices, col_indices):
                        if cost_matrix[row, col] < self.max_cosine_distance:
                            track_idx = valid_track_idx[row]
                            matches.append((track_idx, col))
                            if track_idx in unmatched_tracks_idx:
                                unmatched_tracks_idx.remove(track_idx)
                            if col in unmatched_detections_idx:
                                unmatched_detections_idx.remove(col)

        # Step 2: Match remaining tracks using IoU
        if unmatched_tracks_idx and unmatched_detections_idx:
            remaining_tracks = [confirmed_tracks[i] for i in unmatched_tracks_idx]
            remaining_dets = [detections[i] for i in unmatched_detections_idx]

            # Compute IoU matrix
            iou_matrix = self._compute_iou_matrix(remaining_tracks, remaining_dets)

            # Convert to cost (1 - IoU)
            cost_matrix = 1 - iou_matrix
            cost_matrix[cost_matrix > self.max_iou_distance] = 1e5

            if cost_matrix.size > 0:
                row_indices, col_indices = linear_sum_assignment(cost_matrix)

                for row, col in zip(row_indices, col_indices):
                    if cost_matrix[row, col] < self.max_iou_distance:
                        track_idx = unmatched_tracks_idx[row]
                        det_idx = unmatched_detections_idx[col]
                        matches.append((track_idx, det_idx))

        # Update matched tracks
        matched_track_indices = set()
        matched_det_indices = set()

        for track_idx, det_idx in matches:
            track = confirmed_tracks[track_idx]
            det = detections[det_idx]

            # Update Kalman state
            track.kalman_state = self.kalman.update(track.kalman_state, det["bbox"])
            track.bbox = det["bbox"]
            track.confidence = det.get("confidence", 1.0)
            track.hits += 1
            track.time_since_update = 0

            # Update appearance feature
            if embeddings is not None and det_idx < len(embeddings):
                track.update_feature(embeddings[det_idx], self.feature_alpha)

            matched_track_indices.add(track_idx)
            matched_det_indices.add(det_idx)

        # Step 3: Match tentative tracks using IoU only
        unmatched_det_after_confirmed = [
            i for i in range(len(detections)) if i not in matched_det_indices
        ]

        if tentative_tracks and unmatched_det_after_confirmed:
            remaining_dets = [detections[i] for i in unmatched_det_after_confirmed]
            iou_matrix = self._compute_iou_matrix(tentative_tracks, remaining_dets)
            cost_matrix = 1 - iou_matrix
            cost_matrix[cost_matrix > self.max_iou_distance] = 1e5

            if cost_matrix.size > 0:
                row_indices, col_indices = linear_sum_assignment(cost_matrix)

                for row, col in zip(row_indices, col_indices):
                    if cost_matrix[row, col] < self.max_iou_distance:
                        track = tentative_tracks[row]
                        det_idx = unmatched_det_after_confirmed[col]
                        det = detections[det_idx]

                        track.kalman_state = self.kalman.update(track.kalman_state, det["bbox"])
                        track.bbox = det["bbox"]
                        track.confidence = det.get("confidence", 1.0)
                        track.hits += 1
                        track.time_since_update = 0

                        if embeddings is not None and det_idx < len(embeddings):
                            track.update_feature(embeddings[det_idx], self.feature_alpha)

                        # Promote to confirmed if enough hits
                        if track.hits >= self.n_init:
                            track.status = TrackStatus.CONFIRMED

                        matched_det_indices.add(det_idx)

        # Create new tracks for unmatched detections
        for i in range(len(detections)):
            if i not in matched_det_indices:
                det = detections[i]
                track = Track(
                    track_id=self.next_id,
                    status=TrackStatus.TENTATIVE,
                    bbox=det["bbox"],
                    confidence=det.get("confidence", 1.0),
                    class_id=det.get("class", 0),
                )
                track.kalman_state = self.kalman.initiate(det["bbox"])

                if embeddings is not None and i < len(embeddings):
                    track.update_feature(embeddings[i], self.feature_alpha)

                self.tracks.append(track)
                self.next_id += 1

        # Update unmatched tracks
        for track in self.tracks:
            if track.status != TrackStatus.DELETED:
                track.age += 1
                if track not in [confirmed_tracks[i] for i in matched_track_indices if i < len(confirmed_tracks)]:
                    track.time_since_update += 1

        # Delete old tracks
        self.tracks = [
            t for t in self.tracks
            if t.status != TrackStatus.DELETED and t.time_since_update <= self.max_age
        ]

        # Delete tentative tracks that didn't get confirmed in time
        for track in self.tracks:
            if track.status == TrackStatus.TENTATIVE and track.age > self.n_init + 2:
                track.status = TrackStatus.DELETED

        self.tracks = [t for t in self.tracks if t.status != TrackStatus.DELETED]

        # Return confirmed tracks
        return [
            {
                "track_id": t.track_id,
                "bbox": t.bbox,
                "confidence": t.confidence,
                "class_id": t.class_id,
                "team": t.team,
                "jersey_number": t.jersey_number,
                "role": t.role,
            }
            for t in self.tracks
            if t.status == TrackStatus.CONFIRMED
        ]

    def _compute_iou_matrix(
        self,
        tracks: list[Track],
        detections: list[dict],
    ) -> np.ndarray:
        """Compute IoU matrix between tracks and detections."""
        if not tracks or not detections:
            return np.array([])

        n_tracks = len(tracks)
        n_dets = len(detections)
        iou_matrix = np.zeros((n_tracks, n_dets))

        for i, track in enumerate(tracks):
            for j, det in enumerate(detections):
                iou_matrix[i, j] = self._iou(track.bbox, det["bbox"])

        return iou_matrix

    def _iou(self, box1: list, box2: list) -> float:
        """Compute IoU between two boxes."""
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])

        inter = max(0, x2 - x1) * max(0, y2 - y1)

        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])

        union = area1 + area2 - inter

        return inter / union if union > 0 else 0

    def get_track_features(self) -> dict[int, np.ndarray]:
        """Get mean features for all confirmed tracks."""
        return {
            t.track_id: t.get_mean_feature()
            for t in self.tracks
            if t.status == TrackStatus.CONFIRMED and t.get_mean_feature() is not None
        }
