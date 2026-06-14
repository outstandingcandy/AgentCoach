"""Cost-matrix gates.

A *gate* answers a yes/no question per (track, detection) pair:
"is this match physically plausible?". When the answer is no, the
cost matrix entry is forced to :data:`INF` so the Hungarian assignment
algorithm cannot pick it.

Why a separate abstraction: the matching code used to inline three
different gates as nested for-loops directly mutating ``cost_matrix``.
Pulling them out as objects lets each matching stage compose whichever
gates it needs and makes the rules unit-testable in isolation.

Gates do NOT compute matching costs — they only veto pairs. The
costs come from a separate ``cost_fn`` (cosine, IoU, pitch distance),
typically used by a :class:`MatchingStage`.
"""

from __future__ import annotations

from typing import Protocol

import numpy as np

from .track import Track


# Sentinel cost for forbidden pairings. Hungarian will avoid any pair
# with cost ≥ this value as long as a feasible assignment exists.
INF = 1e5


class Gate(Protocol):
    """A gate is callable; returns True to allow the pair, False to veto.

    Detection ReID embeddings (when needed) are read from
    ``detection['embedding']`` — the tracker injects them at update()
    entry so gates / cost functions don't need a parallel array.
    """

    def __call__(
        self,
        track: Track,
        detection: dict,
    ) -> bool: ...


class PitchGate:
    """Reject (track, detection) pairs whose pitch-space distance exceeds
    a per-frame budget that scales with how long the track has been
    coasting (``time_since_update``).

    Limit formula:

        limit = threshold_m × max(1, time_since_update)

    Rationale: ``threshold_m`` is the budget for one *processed* frame
    (= 0.27 m sprint + jitter ≈ 0.5 m typical). When the track has
    been unmatched for K frames (occluded, missed by YOLO), the player
    can legitimately cover K × that budget before the next detection.
    A static ceiling would correctly reject single-frame teleports but
    wrongly reject legitimate sprint reappearances after a 5-frame
    occlusion (the orig 109 → 173 case in the kids clip).

    Skips (returns True) when either side lacks a pitch projection —
    better an ungated match than a wrongly-rejected one on
    calibration failures.
    """

    # See MotionPitchGate.DEFAULT_ABSOLUTE_MAX_M.
    DEFAULT_ABSOLUTE_MAX_M = 8.0

    def __init__(
        self,
        threshold_m: float,
        absolute_max_m: float | None = None,
    ) -> None:
        self.threshold_m = float(threshold_m)
        self.absolute_max_m = float(
            absolute_max_m if absolute_max_m is not None
            else self.DEFAULT_ABSOLUTE_MAX_M
        )

    def __call__(self, track: Track, detection: dict) -> bool:
        if track.pitch_pos is None:
            return True
        det_pp = detection.get("pitch_pos")
        if det_pp is None:
            return True
        dx = track.pitch_pos[0] - det_pp[0]
        dy = track.pitch_pos[1] - det_pp[1]
        # Linear scaling with coast length, capped at the absolute
        # physical max — a 30-frame coast at 30fps shouldn't authorise
        # 30 × 0.5 = 15m of displacement; that's beyond what a sprint
        # could cover and just lets cross-pitch teleports through.
        scale = int(getattr(track, "time_since_update", 0)) + 1
        threshold = min(self.threshold_m * scale, self.absolute_max_m)
        return dx * dx + dy * dy <= threshold * threshold


class MotionPitchGate:
    """Hybrid motion + pitch gate: pair passes if either signal looks
    consistent, with a coast-aware hard pitch ceiling.

    Logic per call:

        scale = max(1, track.time_since_update)
        pass if  (IoU(pred_bbox, det_bbox) >= iou_min
                  OR pitch_dist <= pitch_max_m * scale)
              AND pitch_dist <= pitch_hard_limit_m * scale

    Why the scale: ``pitch_max_m`` and ``pitch_hard_limit_m`` are
    per-processed-frame budgets. When the track has been unmatched
    for K frames (occlusion / YOLO miss) the player can legitimately
    have moved K × the per-frame budget, so the gate must dilate.
    Without scaling, the orig 109 → orig 173 reappearance after a
    6-frame YOLO drop (3.05 m, well within 6 × 0.5 = 3 m sprint
    budget) gets rejected by the static 1.5 m ceiling.

    Why both halves AND the ceiling:
      - Pitch alone (4 m default) lets Hungarian pick a same-team
        player who happens to be 2 m away.
      - IoU alone fails for fast wingers whose bbox crosses its own
        width in one sample.
      - The OR ("either is consistent") fixes both, BUT a far-camera
        perspective failure can still produce a high-IoU bbox match
        where the two boxes overlap on screen yet sit at very
        different pitch depths (frame 246 in the kids clip: IoU≈0.19
        between an orig 72 player bbox and a referee bbox 4 m closer
        to camera — same screen-space rectangle, totally different
        pitch positions). The hard pitch ceiling rejects those even
        when IoU is fine.

    ``pitch_hard_limit_m`` defaults to ``3 * pitch_max_m`` — generous
    enough to absorb projection jitter on far players (where 1 px
    image error can be multiple metres on the pitch) but tight enough
    at scale=1 to kill near-camera teleports.

    Skips (returns True) when pitch projections unavailable on either
    side — calibration failure shouldn't block all matching.
    """

    # Physical absolute ceiling. Even a Usain Bolt-class kid can't
    # cover more than ~10m/s × 1s = 10m per second, so a 30-frame
    # coast at 30fps should not authorise more than 10m of pitch
    # displacement. Keeps the dt-scaling from going wild on max_age-
    # length coasts and authorising cross-pitch teleports (the
    # frame-98→128 orig 42 case in the kids clip: 30 frames coast,
    # scale=31 dilated the hard limit to 46.5m and let the track
    # snap onto a different player 13m away). Conservatively set to
    # 8m — kids sprint is ~6-7m/s, this leaves headroom for a 1s
    # gap. Caller can override via the ``absolute_max_m`` arg.
    DEFAULT_ABSOLUTE_MAX_M = 8.0

    def __init__(
        self,
        iou_min: float,
        pitch_max_m: float,
        pitch_hard_limit_m: float | None = None,
        absolute_max_m: float | None = None,
    ) -> None:
        self.iou_min = float(iou_min)
        self.pitch_max_m = float(pitch_max_m)
        self.pitch_hard_limit_m = float(
            pitch_hard_limit_m if pitch_hard_limit_m is not None
            else 3.0 * pitch_max_m
        )
        self.absolute_max_m = float(
            absolute_max_m if absolute_max_m is not None
            else self.DEFAULT_ABSOLUTE_MAX_M
        )

    def __call__(self, track: Track, detection: dict) -> bool:
        det_pp = detection.get("pitch_pos") if isinstance(detection, dict) else None
        # +1 so a freshly-updated track (time_since_update = 0) still
        # gets the single-frame per-frame budget; +k for a track that
        # has already missed k frames. Cap at the physical max so a
        # long coast can't authorise teleports.
        scale = int(getattr(track, "time_since_update", 0)) + 1
        pitch_max = min(self.pitch_max_m * scale, self.absolute_max_m)
        pitch_hard = min(
            self.pitch_hard_limit_m * scale, self.absolute_max_m,
        )
        # Hard pitch ceiling — overrides the IoU "stayed put" path.
        # Only enforced when both projections exist; calibration-
        # failure pairs fall through to the IoU/leniency path.
        pitch_dx_sq = None
        if track.pitch_pos is not None and det_pp is not None:
            dx = track.pitch_pos[0] - det_pp[0]
            dy = track.pitch_pos[1] - det_pp[1]
            pitch_dx_sq = dx * dx + dy * dy
            if pitch_dx_sq > pitch_hard * pitch_hard:
                return False
            # Pitch path — small motion accepted.
            if pitch_dx_sq <= pitch_max * pitch_max:
                return True
        # IoU path — when we have both bboxes.
        det_bbox = detection.get("bbox") if isinstance(detection, dict) else None
        if track.bbox is not None and det_bbox is not None:
            iou = _iou_bbox(track.bbox, det_bbox)
            if iou >= self.iou_min:
                return True
        # Neither check available → be lenient (calibration / Kalman
        # state edge case); Hungarian's threshold and the cost itself
        # still filter obviously bad pairs.
        if track.pitch_pos is None and (
            track.bbox is None or det_bbox is None
        ):
            return True
        # Both checks ran but neither passed → reject.
        return False


def _iou_bbox(a, b) -> float:
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    aa = (a[2] - a[0]) * (a[3] - a[1])
    bb = (b[2] - b[0]) * (b[3] - b[1])
    union = aa + bb - inter
    return inter / union if union > 0 else 0.0


def apply_gates(
    cost_matrix: np.ndarray,
    gates: list[Gate],
    tracks: list[Track],
    detections: list[dict],
) -> np.ndarray:
    """Set ``cost_matrix[r, c] = INF`` for any pair where any gate vetoes.

    Operates in-place on ``cost_matrix`` and also returns it so callers
    can chain calls. Skips pairs already marked INF (cheap short-
    circuit when stacking gates).
    """
    if not gates or cost_matrix.size == 0:
        return cost_matrix
    n_tracks, n_dets = cost_matrix.shape
    for r in range(n_tracks):
        for c in range(n_dets):
            if cost_matrix[r, c] >= INF:
                continue
            det = detections[c]
            for gate in gates:
                if not gate(tracks[r], det):
                    cost_matrix[r, c] = INF
                    break
    return cost_matrix
