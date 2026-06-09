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

from .gates import PitchGate
from .kalman import KalmanFilter
from .lifecycle import TrackLifecycle
from .matching import MatchingStage, cosine_cost, iou_cost, run_stage
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
        self.n_init = config.get("n_init", 3)
        self.max_iou_distance = config.get("max_iou_distance", 0.7)
        self.max_cosine_distance = config.get("max_cosine_distance", 0.3)
        self.feature_alpha = config.get("feature_alpha", 0.9)
        # Pitch-space gating threshold (metres). At process_fps=10 a
        # full-sprint 8 m/s player travels ~0.8m/sample; 3m leaves
        # comfortable headroom for projection jitter on distant
        # players. Calibration failures skip the gate entirely.
        self.pitch_gate_m = float(config.get("pitch_gate_m", 3.0))

        self.lifecycle = TrackLifecycle(
            max_age=config.get("max_age", 30),
            n_init=self.n_init,
            stationary_window=int(config.get("stationary_window", 30)),
            stationary_max_pixels=float(
                config.get("stationary_max_pixels", 5.0)),
            stationary_zone_radius=float(
                config.get("stationary_zone_radius", 25.0)),
            stationary_zone_ttl=int(config.get("stationary_zone_ttl", 300)),
        )

        frame_interval = config.get("frame_interval", 1.0)
        self.kalman = KalmanFilter(frame_interval=frame_interval)
        self.tracks: list[Track] = []
        self.next_id = 1
        self.img_w = 1920  # Default, can be updated
        self.img_h = 1080

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

    def _matching_stages(self) -> list[MatchingStage]:
        """Build the cascaded matching pipeline for this update().

        Three stages — same behaviour as the original DeepSORT cascade
        — but expressed as data so adding/swapping a stage is local:

        1. Confirmed × ReID cosine, gated by pitch distance.
        2. Confirmed remaining × IoU (Kalman-predicted bbox).
        3. Tentative × IoU only (no ReID, EMA isn't stable yet).
        """
        return [
            MatchingStage(
                name="confirmed-reid",
                track_filter=lambda t: (
                    t.status == TrackStatus.CONFIRMED
                    and t.smooth_feature is not None
                ),
                cost_fn=cosine_cost,
                gates=[PitchGate(self.pitch_gate_m)],
                threshold=self.max_cosine_distance,
            ),
            MatchingStage(
                name="confirmed-iou",
                track_filter=lambda t: t.status == TrackStatus.CONFIRMED,
                cost_fn=iou_cost,
                gates=[],
                threshold=self.max_iou_distance,
            ),
            MatchingStage(
                name="tentative-iou",
                track_filter=lambda t: t.status == TrackStatus.TENTATIVE,
                cost_fn=iou_cost,
                gates=[],
                threshold=self.max_iou_distance,
            ),
        ]

    def _apply_match(
        self,
        track: Track,
        det_idx: int,
        detections: list[dict[str, Any]],
        embeddings: np.ndarray | None,
    ) -> None:
        """Apply a (track, detection) match: Kalman update, feature
        EMA, pitch_pos refresh, hits++ and tentative-promotion."""
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
        if (
            track.status == TrackStatus.TENTATIVE
            and track.hits >= self.n_init
        ):
            track.status = TrackStatus.CONFIRMED

    def _spawn_track(
        self,
        det: dict[str, Any],
        det_idx: int,
        embeddings: np.ndarray | None,
    ) -> Track:
        """Create a fresh TENTATIVE track from an unmatched detection."""
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
        if embeddings is not None and det_idx < len(embeddings):
            track.update_feature(embeddings[det_idx], self.feature_alpha)
        self.tracks.append(track)
        self.next_id += 1
        return track

    def update(
        self,
        detections: list[dict[str, Any]],
        embeddings: np.ndarray | None = None,
    ) -> list[dict[str, Any]]:
        """Update tracker with new detections.

        Pipeline:
          1. Kalman predict for all tracks.
          2. Run cascaded matching stages, applying each match before
             the next stage so later stages see updated state. (The
             original DeepSORT cascade applied step-1+step-2 matches
             together then ran step-3 separately; for behavioural
             parity we apply each stage's matches before moving on.)
          3. Spawn TENTATIVE tracks for still-unmatched detections.
          4. Tick age / time_since_update on tracks that didn't get a
             measurement; cleanup pass deletes stale + ghost tracks.
          5. Emit one dict per non-coasting track.
        """
        self.predict()

        # ---- Match ---------------------------------------------------
        unmatched_track_ids: set[int] = {id(t) for t in self.tracks}
        unmatched_det_idx: set[int] = set(range(len(detections)))
        # Tracks that received a real detection update this frame —
        # used at end-of-update so the time_since_update bump only
        # applies to truly unmatched tracks. Without this, tentative
        # matches and freshly-created tracks would tick to >0 right
        # after creation and the emit-stage coast filter would drop
        # them, hiding the first n_init frames of every track.
        updated_this_frame: set[int] = set()

        stages = self._matching_stages()
        # Apply confirmed stages together (step 1 + step 2 in the
        # original cascade collected matches then applied as a batch);
        # apply tentative stages immediately after their match. This
        # matches the original ordering bit-for-bit.
        confirmed_matches: list[tuple[Track, int]] = []
        for stage in stages:
            stage_matches = run_stage(
                stage, self.tracks, detections, embeddings,
                unmatched_track_ids, unmatched_det_idx,
            )
            if stage.name.startswith("confirmed-"):
                confirmed_matches.extend(stage_matches)
            else:
                # Apply confirmed matches before the tentative stage so
                # spawn() decisions match the original ordering.
                for track, det_idx in confirmed_matches:
                    self._apply_match(track, det_idx, detections, embeddings)
                    updated_this_frame.add(id(track))
                confirmed_matches = []
                for track, det_idx in stage_matches:
                    self._apply_match(track, det_idx, detections, embeddings)
                    updated_this_frame.add(id(track))
        # Any leftover confirmed matches (no tentative stage ran).
        for track, det_idx in confirmed_matches:
            self._apply_match(track, det_idx, detections, embeddings)
            updated_this_frame.add(id(track))

        # ---- Spawn new tracks for unmatched detections --------------
        # Suppress any detection landing inside a kill-zone left behind
        # by the stationary-track killer (a persistent YOLO false-
        # positive can't immediately respawn the ghost as a fresh
        # track).
        for i in sorted(unmatched_det_idx):
            det = detections[i]
            x1, y1, x2, y2 = det["bbox"]
            dcx = (x1 + x2) / 2.0
            dcy = (y1 + y2) / 2.0
            if self._in_kill_zone(dcx, dcy):
                continue
            track = self._spawn_track(det, i, embeddings)
            updated_this_frame.add(id(track))

        # ---- Tick + cleanup -----------------------------------------
        self.lifecycle.tick(self.tracks, updated_this_frame)
        self.tracks = self.lifecycle.cleanup(self.tracks)

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
        return self.lifecycle.in_kill_zone(cx, cy)

    def _record_center(self, track: Track) -> None:
        self.lifecycle.record_center(track)

    def get_track_features(self) -> dict[int, np.ndarray]:
        """Get mean features for all confirmed tracks."""
        return {
            t.track_id: t.get_mean_feature()
            for t in self.tracks
            if t.status == TrackStatus.CONFIRMED and t.get_mean_feature() is not None
        }
