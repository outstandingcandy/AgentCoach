"""StrongSORT-style multi-object tracker.

Coordinates Kalman prediction, ReID feature management, cascaded
matching, lifecycle (creation / promotion / deletion), and emission.

The matching / gating / lifecycle pieces are split into sibling
modules (matching.py, gates.py, lifecycle.py) so each can be reasoned
about and tested independently — see their docstrings.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist

from .kalman import KalmanFilter
from .track import Track, TrackStatus


class StrongSORTTracker:
    """StrongSORT-style multi-object tracker.

    Features:
    - Kalman filter motion prediction
    - External ReID features for appearance matching
    - Cascaded matching strategy (confirmed-ReID, confirmed-IoU, tentative-IoU)
    - EMA feature smoothing
    - Pitch-space metric gating (when calibration is available)
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
                - pitch_gate_m: Max pitch-space distance gate in metres (default: 3.0)
                - stationary_window / stationary_max_pixels / stationary_zone_*: ghost killer
        """
        config = config or {}
        self.max_age = config.get("max_age", 30)
        self.n_init = config.get("n_init", 3)
        self.max_iou_distance = config.get("max_iou_distance", 0.7)
        self.max_cosine_distance = config.get("max_cosine_distance", 0.3)
        self.feature_alpha = config.get("feature_alpha", 0.9)
        # Pitch-space gating threshold (metres). At process_fps=10 a
        # full-sprint 8 m/s player travels ~0.8m/sample; 3m leaves
        # comfortable headroom for projection jitter on distant
        # players. Calibration failures skip the gate entirely.
        self.pitch_gate_m = float(config.get("pitch_gate_m", 3.0))
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
        """Predict track states for current frame.

        Tracks whose Kalman prediction crosses the image border are NOT
        immediately deleted — a player walking off-screen briefly (then
        coming back) was getting wiped within one frame because the
        constant-velocity model overshoots the edge by a few pixels.
        ``time_since_update > max_age`` already provides natural
        cleanup; an off-screen-then-reappearing player can re-match by
        ReID + pitch-position when they come back into the field of
        view, preserving their tid.
        """
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

                # Gate by pitch-space distance — reject matches where the
                # detection's foot-point is more than ``pitch_gate_m`` metres
                # from the track's last known pitch position. Aspect ratio
                # and bbox height are NOT gated — in sports footage they
                # vary too much per pose (e.g. a side-on player has
                # aspect 0.32, the same player mid-run lunge has 0.72)
                # and force-rejecting on those just orphans real
                # detections (frame 12: tid=10 lost det_LO purely because
                # det_LO's aspect spiked to 0.72).
                #
                # When calibration is unavailable (no pitch_pos on the
                # detection or on the track), gating is skipped and
                # matching falls back to ReID cosine alone — better an
                # ungated match than a wrongly-rejected one.
                pitch_gate_m_sq = (
                    float(getattr(self, "pitch_gate_m", 3.0)) ** 2
                )
                for ri in range(len(valid_track_idx)):
                    t = confirmed_tracks[valid_track_idx[ri]]
                    if t.pitch_pos is None:
                        continue  # no track pitch anchor → can't gate
                    tx, ty = t.pitch_pos
                    for ci in range(len(detections)):
                        if cost_matrix[ri, ci] >= 1e5:
                            continue  # already rejected by cosine threshold
                        det_pp = detections[ci].get("pitch_pos")
                        if det_pp is None:
                            continue  # detection lacks calibration → skip gate
                        dx = tx - det_pp[0]
                        dy = ty - det_pp[1]
                        if dx * dx + dy * dy > pitch_gate_m_sq:
                            cost_matrix[ri, ci] = 1e5

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
        # Tracks that received a real detection update this frame —
        # used at end-of-update so the time_since_update bump only
        # applies to truly unmatched tracks. Without this, tentative
        # matches and freshly-created tracks would tick to >0 right
        # after creation and the emit-stage coast filter would drop
        # them, hiding the first n_init frames of every track.
        updated_this_frame: set[int] = set()

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
            if det.get("pitch_pos") is not None:
                track.pitch_pos = tuple(det["pitch_pos"])

            # Update appearance feature
            if embeddings is not None and det_idx < len(embeddings):
                track.update_feature(embeddings[det_idx], self.feature_alpha)

            matched_track_indices.add(track_idx)
            matched_det_indices.add(det_idx)
            updated_this_frame.add(id(track))

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
                        if det.get("pitch_pos") is not None:
                            track.pitch_pos = tuple(det["pitch_pos"])

                        if embeddings is not None and det_idx < len(embeddings):
                            track.update_feature(embeddings[det_idx], self.feature_alpha)

                        # Promote to confirmed if enough hits
                        if track.hits >= self.n_init:
                            track.status = TrackStatus.CONFIRMED

                        matched_det_indices.add(det_idx)
                        updated_this_frame.add(id(track))

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
            if det.get("pitch_pos") is not None:
                track.pitch_pos = tuple(det["pitch_pos"])

            if embeddings is not None and i < len(embeddings):
                track.update_feature(embeddings[i], self.feature_alpha)

            self.tracks.append(track)
            self.next_id += 1
            updated_this_frame.add(id(track))

        # Update unmatched tracks — bump time_since_update only on
        # tracks that didn't get a measurement this frame.
        for track in self.tracks:
            if track.status == TrackStatus.DELETED:
                continue
            track.age += 1
            if id(track) not in updated_this_frame:
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
