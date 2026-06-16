"""Ball tracking using ByteTrack or BoT-SORT.

Wraps ultralytics BYTETracker / BOTSORT for multi-object ball tracking with:
- Two-threshold matching (high-conf first, low-conf center-distance rescue)
- Center-distance matching instead of IoU in BOTH stages (better for tiny ball bboxes)
- Hard distance gate that fuse_score cannot bypass
- Kalman filter motion prediction
- Multi-track output for trajectory-based filtering

Tracker type is selected via config["tracker_type"]: "bytetrack" or "botsort".
"""

from contextlib import contextmanager
from types import SimpleNamespace
from typing import Any

import numpy as np

from ultralytics.trackers.byte_tracker import BYTETracker, STrack
from ultralytics.trackers.bot_sort import BOTSORT, BOTrack
from ultralytics.trackers.utils import matching


class _DetectionAdapter:
    """Adapts a list of detection dicts to the interface BYTETracker.update() expects.

    BYTETracker expects an object with .conf, .xywh, .cls attributes and
    supports boolean/integer array indexing and len().
    """

    def __init__(self, detections: list[dict[str, Any]]):
        n = len(detections)
        self._xywh = np.zeros((n, 4), dtype=np.float32)
        self._conf = np.zeros(n, dtype=np.float32)
        self._cls = np.zeros(n, dtype=np.float32)

        for i, det in enumerate(detections):
            bbox = det["bbox"]  # [x1, y1, x2, y2]
            x1, y1, x2, y2 = bbox[0], bbox[1], bbox[2], bbox[3]
            cx = (x1 + x2) / 2
            cy = (y1 + y2) / 2
            w = x2 - x1
            h = y2 - y1
            self._xywh[i] = [cx, cy, w, h]
            self._conf[i] = det.get("confidence", 0.0)
            self._cls[i] = det.get("class", 32)  # sports ball

    @property
    def xywh(self) -> np.ndarray:
        return self._xywh

    @property
    def conf(self) -> np.ndarray:
        return self._conf

    @property
    def cls(self) -> np.ndarray:
        return self._cls

    @property
    def xyxy(self) -> np.ndarray:
        xyxy = np.zeros_like(self._xywh)
        xyxy[:, 0] = self._xywh[:, 0] - self._xywh[:, 2] / 2
        xyxy[:, 1] = self._xywh[:, 1] - self._xywh[:, 3] / 2
        xyxy[:, 2] = self._xywh[:, 0] + self._xywh[:, 2] / 2
        xyxy[:, 3] = self._xywh[:, 1] + self._xywh[:, 3] / 2
        return xyxy

    def __len__(self) -> int:
        return len(self._conf)

    def __getitem__(self, idx):
        new = _DetectionAdapter.__new__(_DetectionAdapter)
        new._xywh = self._xywh[idx]
        new._conf = self._conf[idx]
        new._cls = self._cls[idx]
        return new


def _center_distance_matrix(
    tracks: list,
    detections: list,
    max_distance: float,
    *,
    current_frame_id: int | None = None,
    gap_scale_cap: float = 8.0,
) -> np.ndarray:
    """Compute normalized center-distance matrix with hard gate.

    Per-track gate scaling: a track that has been coasting on Kalman
    predict for N frames (``current_frame_id - track.frame_id``) gets its
    gate scaled by ``max(1, N)``, capped at ``gap_scale_cap``. Without
    this scaling a track that loses its detection during a kick (16+
    frames of motion blur) can never re-associate when the ball reappears
    hundreds of pixels away — the base 120 px gate was sized for the
    fresh-track case (≤2 frame gap, ≤95 px displacement) and rejects
    every long-gap reacquisition. ultralytics's BYTETracker doesn't
    maintain ``time_since_update``, so we derive the gap from the
    tracker's ``frame_id`` counter instead.

    Returns:
        (dists, hard_mask) — dists[i,j] in [0,1], hard_mask[i,j]=True
        means the pair is gated out regardless of score fusion.
    """
    if len(tracks) == 0 or len(detections) == 0:
        empty = np.zeros((len(tracks), len(detections)), dtype=np.float32)
        return empty, np.zeros_like(empty, dtype=bool)

    track_centers = np.array([t.xywh[:2] for t in tracks], dtype=np.float32)
    det_centers = np.array([d.xywh[:2] for d in detections], dtype=np.float32)

    diff = track_centers[:, None, :] - det_centers[None, :, :]
    dists = np.sqrt((diff ** 2).sum(axis=2))

    if current_frame_id is None:
        per_track_max = np.full(len(tracks), max_distance, dtype=np.float32)
    else:
        gaps = np.array(
            [max(1.0, min(float(current_frame_id - int(t.frame_id)),
                          gap_scale_cap))
             for t in tracks],
            dtype=np.float32,
        )
        per_track_max = max_distance * gaps

    dists = np.minimum(dists / per_track_max[:, None], 1.0)

    hard_mask = dists >= 1.0

    return dists, hard_mask


def _center_distance_as_iou(max_distance: float, frame_id_getter):
    """Return a function with the same signature as matching.iou_distance
    but using center-distance instead of IoU.

    ``frame_id_getter`` is a zero-arg callable returning the tracker's
    current frame id; it's threaded through so the gap-aware gate inside
    ``_center_distance_matrix`` works for the second-stage call too.
    """
    def _iou_replacement(tracks, detections):
        dists, _ = _center_distance_matrix(
            tracks, detections, max_distance,
            current_frame_id=frame_id_getter(),
        )
        return dists
    return _iou_replacement


@contextmanager
def _patch_second_stage(max_distance: float, frame_id_getter):
    """Temporarily replace matching functions used by the second stage.

    BYTETracker.update() hardcodes ``matching.iou_distance`` and
    ``matching.fuse_score`` for the second-stage (low-confidence) association.

    Problems for tiny-ball tracking:
    1. IoU is useless for 10-30px bboxes → swap with center-distance.
    2. ``fuse_score`` computes ``1 - (1-dist)*score``.  With the second-stage
       threshold hardcoded at 0.5, any detection with score < 0.5 can NEVER
       match (the fused cost is always > 0.5).  Since the second stage by
       definition only sees low-score detections, fuse_score must be disabled.
    """
    orig_iou = matching.iou_distance
    orig_fuse = matching.fuse_score
    matching.iou_distance = _center_distance_as_iou(max_distance, frame_id_getter)
    matching.fuse_score = lambda cost, dets: cost  # no-op
    try:
        yield
    finally:
        matching.iou_distance = orig_iou
        matching.fuse_score = orig_fuse


class _BallBYTETracker(BYTETracker):
    """BYTETracker subclass that uses center-distance instead of IoU for matching.

    Overrides get_dists() for first-stage matching and patches
    matching.iou_distance during update() so the second-stage (low-confidence
    rescue) also uses center-distance instead of IoU.
    """

    def __init__(self, args, frame_rate: int = 30, max_distance: float = 200.0):
        super().__init__(args, frame_rate=frame_rate)
        self.max_distance = max_distance

    def get_dists(self, tracks: list[STrack], detections: list[STrack]) -> np.ndarray:
        """Center-distance matrix normalized to [0, 1] range with hard gate."""
        dists, hard_mask = _center_distance_matrix(
            tracks, detections, self.max_distance,
            current_frame_id=self.frame_id,
        )

        if self.args.fuse_score:
            det_scores = np.array([d.score for d in detections], dtype=np.float32)
            dists = dists * (1.0 - det_scores[None, :])

        dists[hard_mask] = 1.0

        return dists

    def update(self, results, img=None, feats=None):
        with _patch_second_stage(self.max_distance, lambda: self.frame_id):
            return super().update(results, img=img, feats=feats)


class _BallBOTSORTTracker(BOTSORT):
    """BOTSORT subclass that uses center-distance instead of IoU for matching.

    Same center-distance + hard gate logic as _BallBYTETracker, but inherits
    BOTSORT's additional features:
    - KalmanFilterXYWH (center+width+height) — better for ball tracking than
      BYTETracker's KalmanFilterXYAH (center+aspect+height)
    - GMC camera motion compensation (optional, set gmc_method="none" to disable)
    - BOTrack with EMA feature smoothing (for future ReID support)
    """

    def __init__(self, args, frame_rate: int = 30, max_distance: float = 200.0):
        super().__init__(args, frame_rate=frame_rate)
        self.max_distance = max_distance

    def get_dists(self, tracks: list[BOTrack], detections: list[BOTrack]) -> np.ndarray:
        """Center-distance matrix normalized to [0, 1] range with hard gate."""
        dists, hard_mask = _center_distance_matrix(
            tracks, detections, self.max_distance,
            current_frame_id=self.frame_id,
        )

        if self.args.fuse_score:
            det_scores = np.array([d.score for d in detections], dtype=np.float32)
            dists = dists * (1.0 - det_scores[None, :])

        dists[hard_mask] = 1.0

        return dists

    def update(self, results, img=None, feats=None):
        with _patch_second_stage(self.max_distance, lambda: self.frame_id):
            return super().update(results, img=img, feats=feats)


class BallTracker:
    """Multi-object ball tracker wrapping BYTETracker or BOTSORT.

    Converts between our detection dict format and the tracker's internal
    format, preserving the same update() -> list[dict] API.

    Config key ``tracker_type`` selects the backend:
    - ``"bytetrack"`` — BYTETracker with center-distance matching
    - ``"botsort"``   — BOTSORT with center-distance matching + GMC + XYWH Kalman
    """

    def __init__(self, config: dict[str, Any] | None = None):
        self.config = config or {}

        tracker_type = self.config.get("tracker_type", "botsort")
        fps = self.config.get("fps", 10)
        max_distance = self.config.get("max_position_distance", 200.0)

        # Common args shared by both trackers
        args = SimpleNamespace(
            track_high_thresh=self.config.get("track_high_thresh", 0.4),
            track_low_thresh=self.config.get("track_low_thresh", 0.1),
            track_buffer=self.config.get("track_buffer", 15),
            match_thresh=self.config.get("match_thresh", 0.8),
            new_track_thresh=self.config.get("new_track_thresh", 0.3),
            fuse_score=self.config.get("fuse_score", True),
        )

        if tracker_type == "botsort":
            args.gmc_method = self.config.get("gmc_method", "none")
            args.proximity_thresh = self.config.get("proximity_thresh", 0.5)
            args.appearance_thresh = self.config.get("appearance_thresh", 0.25)
            args.with_reid = self.config.get("with_reid", False)
            args.model = self.config.get("reid_model", "auto")
            self._tracker = _BallBOTSORTTracker(args, frame_rate=fps, max_distance=max_distance)
        else:
            self._tracker = _BallBYTETracker(args, frame_rate=fps, max_distance=max_distance)

    def update(self, detections: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Update tracker with new detections."""
        if not detections:
            adapter = _DetectionAdapter([])
        else:
            adapter = _DetectionAdapter(detections)

        results = self._tracker.update(adapter)

        tracks = []
        for row in results:
            x1, y1, x2, y2 = float(row[0]), float(row[1]), float(row[2]), float(row[3])
            track_id = int(row[4])
            score = float(row[5])

            tracks.append({
                "track_id": track_id,
                "center": [(x1 + x2) / 2, (y1 + y2) / 2],
                "bbox": [x1, y1, x2, y2],
                "confidence": score,
                "predicted": False,
            })

        return tracks

    def reset(self) -> None:
        """Reset tracker state."""
        self._tracker.reset()
