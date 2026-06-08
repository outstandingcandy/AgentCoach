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
import scipy.linalg
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

    # Recent bbox-center history for stationary-detection.  Stores at
    # most ``stationary_window`` (cx, cy) tuples so we can detect
    # ghost tracks where YOLO keeps re-detecting the same static
    # background object.
    center_history: list = field(default_factory=list)

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


# Chi-squared inverse CDF at 95% confidence for 1-9 degrees of freedom
chi2inv95 = {1: 3.8415, 2: 5.9915, 3: 7.8147, 4: 9.4877,
             5: 11.070, 6: 12.592, 7: 14.067, 8: 15.507, 9: 16.919}


class KalmanFilter:
    """Kalman filter for bounding box tracking with height-dependent noise.

    Following DeepSORT/StrongSORT: process and measurement noise scale with
    target height so that the covariance is properly calibrated for
    Mahalanobis distance gating.

    State: [cx, cy, aspect_ratio, height, vx, vy, va, vh]
    """

    def __init__(self, frame_interval: float = 1.0):
        ndim = 4

        # State transition matrix (constant velocity model)
        self.F = np.eye(2 * ndim)
        for i in range(ndim):
            self.F[i, ndim + i] = frame_interval

        # Observation matrix (we observe cx, cy, a, h)
        self.H = np.eye(ndim, 2 * ndim)

        # Noise weights (relative to target height), scaled by frame interval
        # Base values from DeepSORT assume 30fps (dt=1); scale for actual dt
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

    def gating_distance(self, state: KalmanState, measurement: np.ndarray) -> float:
        """Compute squared Mahalanobis distance between predicted state and measurement."""
        projected_mean, projected_cov = self._project(state)
        diff = measurement - projected_mean
        chol = np.linalg.cholesky(projected_cov)
        z = scipy.linalg.solve_triangular(chol, diff, lower=True)
        return float(z @ z)

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
        # Stationary-track killer: delete a track if its bbox centre has
        # not moved more than ``stationary_max_pixels`` over the last
        # ``stationary_window`` updates (regardless of how many "matches"
        # it received in that window).  Defends against YOLO
        # re-detecting the same static background object — a banner /
        # spectator / fence-post / parked car — which would otherwise
        # keep matching to an old track via ReID and produce a 5-15s
        # ghost bbox.  Set ``stationary_window`` to 0 to disable.
        self.stationary_window = int(config.get("stationary_window", 30))
        self.stationary_max_pixels = float(
            config.get("stationary_max_pixels", 5.0))

        frame_interval = config.get("frame_interval", 1.0)
        self.kalman = KalmanFilter(frame_interval=frame_interval)
        self.tracks: list[Track] = []
        self.next_id = 1
        self.img_w = 1920  # Default, can be updated
        self.img_h = 1080

        # Kill-zones: locations where the stationary killer recently
        # deleted a track.  New detections falling inside one of these
        # zones are suppressed for ``stationary_zone_ttl`` updates so a
        # persistent YOLO false-positive can't immediately respawn the
        # ghost as a fresh track.  Each entry: (cx, cy, ttl).
        self.stationary_zones: list[tuple[float, float, int]] = []
        self.stationary_zone_radius = float(
            config.get("stationary_zone_radius", 25.0))
        self.stationary_zone_ttl = int(
            config.get("stationary_zone_ttl", 300))

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

                # Mark tracks as deleted if predicted center is outside frame
                cx, cy = track.kalman_state.mean[0], track.kalman_state.mean[1]
                if cx < 0 or cx > self.img_w or cy < 0 or cy > self.img_h:
                    track.status = TrackStatus.DELETED

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

                # Gate by Mahalanobis distance: reject matches where detection
                # falls outside 95% confidence region of Kalman prediction.
                # For tracks with large time_since_update, also enforce a hard
                # pixel distance cap — their inflated covariance makes
                # Mahalanobis gating alone ineffective.
                gating_threshold = chi2inv95[4]  # 4-dim observation space
                stale_age = 10  # frames without update before applying pixel cap
                max_pixel_dist_sq = 400.0 ** 2  # hard cap for stale tracks
                for ri in range(len(valid_track_idx)):
                    t = confirmed_tracks[valid_track_idx[ri]]
                    if t.kalman_state is None:
                        continue
                    is_stale = t.time_since_update > stale_age
                    pred_cx, pred_cy = t.kalman_state.mean[0], t.kalman_state.mean[1]
                    for ci in range(len(detections)):
                        if cost_matrix[ri, ci] >= 1e5:
                            continue  # Already rejected by cosine threshold
                        d = detections[ci]
                        x1, y1, x2, y2 = d["bbox"]
                        dcx = (x1 + x2) / 2
                        dcy = (y1 + y2) / 2
                        # Hard pixel cap for stale tracks (inflated covariance)
                        if is_stale:
                            pixel_dist_sq = (dcx - pred_cx) ** 2 + (dcy - pred_cy) ** 2
                            if pixel_dist_sq > max_pixel_dist_sq:
                                cost_matrix[ri, ci] = 1e5
                                continue
                        dw = x2 - x1
                        dh = y2 - y1
                        meas = np.array([dcx, dcy, dw / dh if dh > 0 else 1.0, dh])
                        if self.kalman.gating_distance(t.kalman_state, meas) > gating_threshold:
                            cost_matrix[ri, ci] = 1e5
                # Debug stats (uncomment to monitor gating behavior)
                # if gated_count and total_pairs:
                #     self._gating_stats = (gated_count, total_pairs)

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
            self._record_center(track)

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
                        self._record_center(track)

                        if embeddings is not None and det_idx < len(embeddings):
                            track.update_feature(embeddings[det_idx], self.feature_alpha)

                        # Promote to confirmed if enough hits
                        if track.hits >= self.n_init:
                            track.status = TrackStatus.CONFIRMED

                        matched_det_indices.add(det_idx)

        # Create new tracks for unmatched detections — but suppress any
        # detection landing inside a kill-zone left behind by the
        # stationary-track killer.
        for i in range(len(detections)):
            if i in matched_det_indices:
                continue
            det = detections[i]
            x1, y1, x2, y2 = det["bbox"]
            dcx = (x1 + x2) / 2.0
            dcy = (y1 + y2) / 2.0
            if self._in_kill_zone(dcx, dcy):
                continue
            track = Track(
                track_id=self.next_id,
                status=TrackStatus.TENTATIVE,
                bbox=det["bbox"],
                confidence=det.get("confidence", 1.0),
                class_id=det.get("class", 0),
            )
            track.kalman_state = self.kalman.initiate(det["bbox"])
            self._record_center(track)

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

        # Stationary-track killer: drop confirmed tracks whose bbox
        # centre has not moved more than ``stationary_max_pixels``
        # over the last ``stationary_window`` updates.  These are
        # ghosts — typically YOLO false positives on a fixed-position
        # banner, fence post, or distant spectator.
        if self.stationary_window > 0:
            for track in self.tracks:
                if track.status != TrackStatus.CONFIRMED:
                    continue
                if len(track.center_history) < self.stationary_window:
                    continue
                hist = track.center_history[-self.stationary_window:]
                xs = [c[0] for c in hist]
                ys = [c[1] for c in hist]
                span = max(max(xs) - min(xs), max(ys) - min(ys))
                if span <= self.stationary_max_pixels:
                    track.status = TrackStatus.DELETED
                    cx = sum(xs) / len(xs)
                    cy = sum(ys) / len(ys)
                    self.stationary_zones.append((
                        cx, cy, self.stationary_zone_ttl,
                    ))

        # Decay kill-zones (drop expired ones).
        self.stationary_zones = [
            (x, y, ttl - 1) for (x, y, ttl) in self.stationary_zones
            if ttl - 1 > 0
        ]

        self.tracks = [t for t in self.tracks if t.status != TrackStatus.DELETED]

        # Return confirmed + tentative tracks (tentative flagged for backfill).
        # Skip tracks that didn't get a measurement this frame — their bbox
        # is a Kalman extrapolation, not a real detection. After 5-10 coast
        # steps the constant-velocity model produces grossly wrong bboxes
        # (e.g. a 30x40 player extrapolated for 18 sample-frames at +24px/h
        # ends up 480x600 covering OSD + bench), and tracks.json would
        # otherwise leak those into downstream stages as "real" detections.
        # The track itself is kept alive — it just doesn't emit a bbox on
        # coast frames; when YOLO re-detects it the same tid resumes.
        result = []
        for t in self.tracks:
            if t.status not in (TrackStatus.CONFIRMED, TrackStatus.TENTATIVE):
                continue
            if t.time_since_update > 0:
                continue
            result.append({
                "track_id": t.track_id,
                "bbox": t.bbox,
                "confidence": t.confidence,
                "class_id": t.class_id,
                "team": t.team,
                "jersey_number": t.jersey_number,
                "role": t.role,
                "confirmed": t.status == TrackStatus.CONFIRMED,
            })
        return result

    def _in_kill_zone(self, cx: float, cy: float) -> bool:
        """Return True if (cx, cy) sits within ``stationary_zone_radius``
        of any active kill-zone."""
        if not self.stationary_zones:
            return False
        r2 = self.stationary_zone_radius ** 2
        for zx, zy, _ttl in self.stationary_zones:
            if (cx - zx) ** 2 + (cy - zy) ** 2 <= r2:
                return True
        return False

    def _record_center(self, track: Track) -> None:
        """Push the current bbox centre onto a per-track history ring,
        capped at ``stationary_window`` entries.  Used by the
        stationary-track killer at the end of :meth:`update`."""
        if not track.bbox or self.stationary_window <= 0:
            return
        x1, y1, x2, y2 = track.bbox
        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0
        track.center_history.append((cx, cy))
        if len(track.center_history) > self.stationary_window:
            del track.center_history[: -self.stationary_window]

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
