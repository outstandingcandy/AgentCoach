"""Matching pipeline.

The cascaded matching used to be three near-identical for-loops:

  Step 1: confirmed × all detections, ReID cosine + pitch gate
  Step 2: confirmed remaining × remaining detections, IoU
  Step 3: tentative × remaining detections, IoU

Each loop built its own cost matrix, applied gates, called
:func:`scipy.optimize.linear_sum_assignment`, and threshold-filtered
the result. This module collapses that into a single :func:`run_stage`
helper driven by a :class:`MatchingStage` config, so future tweaks
(adding a gate, swapping a cost function, reordering stages) are local.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist

from .gates import INF, Gate, apply_gates
from .track import Track


# Type alias for a cost function: (tracks, detections) → (T, D) array.
# Detection ReID embeddings are read from ``det['embedding']`` when
# present (the tracker injects them at update() entry).
CostFn = Callable[
    [list[Track], list[dict]],
    np.ndarray,
]


def cosine_cost(
    tracks: list[Track],
    detections: list[dict],
) -> np.ndarray:
    """ReID cosine distance between track ``smooth_feature`` and
    detection ``embedding``. Tracks/detections missing either feature
    get an INF row/column.
    """
    if not tracks or not detections:
        return np.zeros((len(tracks), len(detections)))
    det_embs = []
    valid_dets = []
    for j, d in enumerate(detections):
        e = d.get("embedding")
        if e is not None:
            det_embs.append(e)
            valid_dets.append(j)
    if not det_embs:
        return np.full((len(tracks), len(detections)), INF, dtype=np.float64)
    feats = []
    valid_tracks = []
    for i, t in enumerate(tracks):
        if t.smooth_feature is not None:
            feats.append(t.smooth_feature)
            valid_tracks.append(i)
    if not feats:
        return np.full((len(tracks), len(detections)), INF, dtype=np.float64)
    sub = cdist(np.asarray(feats), np.asarray(det_embs), metric="cosine")
    cost = np.full((len(tracks), len(detections)), INF, dtype=np.float64)
    for k, i in enumerate(valid_tracks):
        for m, j in enumerate(valid_dets):
            cost[i, j] = sub[k, m]
    return cost


def iou_cost(
    tracks: list[Track],
    detections: list[dict],
) -> np.ndarray:
    """1 - IoU between track bbox and detection bbox."""
    if not tracks or not detections:
        return np.zeros((len(tracks), len(detections)))
    cost = np.ones((len(tracks), len(detections)), dtype=np.float64)
    for i, t in enumerate(tracks):
        if not t.bbox:
            continue
        for j, d in enumerate(detections):
            cost[i, j] = 1.0 - _iou(t.bbox, d["bbox"])
    return cost


def pitch_distance_cost(
    tracks: list[Track],
    detections: list[dict],
) -> np.ndarray:
    """Pitch-space distance (metres) between track and detection.

    Used for TENTATIVE-track matching where IoU fails on fast-moving
    or distant players (a 30 px/sample winger has zero bbox overlap
    across samples even though his pitch displacement is < 1 m). When
    either side lacks a pitch projection (calibration failure) the
    pair is marked INF — the IoU stage above this one will catch
    those cases.
    """
    if not tracks or not detections:
        return np.zeros((len(tracks), len(detections)))
    cost = np.full((len(tracks), len(detections)), INF, dtype=np.float64)
    for i, t in enumerate(tracks):
        if t.pitch_pos is None:
            continue
        tx, ty = t.pitch_pos
        for j, d in enumerate(detections):
            det_pp = d.get("pitch_pos")
            if det_pp is None:
                continue
            dx = tx - det_pp[0]
            dy = ty - det_pp[1]
            cost[i, j] = (dx * dx + dy * dy) ** 0.5
    return cost


def _iou(box1: list, box2: list) -> float:
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - inter
    return inter / union if union > 0 else 0


@dataclass
class MatchingStage:
    """One step in the cascaded matching pipeline.

    Attributes:
        name: Human-readable label for logs / debugging.
        track_filter: Predicate that selects which tracks participate.
            Stages typically split on status (CONFIRMED vs TENTATIVE)
            but any per-track property works.
        cost_fn: Builds a (T, D) cost matrix.
        gates: Vetoes applied after the cost matrix is built.
        threshold: Post-Hungarian per-pair max cost; assignments above
            this are discarded (the same value the gate uses for the
            cosine threshold typically).
    """
    name: str
    track_filter: Callable[[Track], bool]
    cost_fn: CostFn
    threshold: float
    gates: list[Gate] = field(default_factory=list)


def run_stage(
    stage: MatchingStage,
    tracks: list[Track],
    detections: list[dict],
    unmatched_track_ids: set[int],
    unmatched_det_idx: set[int],
) -> list[tuple[Track, int]]:
    """Run a single matching stage. Returns list of (track, det_idx) pairs.

    ``unmatched_track_ids`` and ``unmatched_det_idx`` are mutated in
    place so subsequent stages see only the leftovers.
    ``unmatched_track_ids`` keys on ``Track.track_id`` (which is unique
    and stable for the lifetime of a track).
    """
    candidates = [
        t for t in tracks
        if t.track_id in unmatched_track_ids and stage.track_filter(t)
    ]
    if not candidates or not detections:
        return []

    det_indices = sorted(unmatched_det_idx)
    if not det_indices:
        return []
    sub_dets = [detections[i] for i in det_indices]

    cost = stage.cost_fn(candidates, sub_dets)
    if cost.size == 0:
        return []

    # Pre-threshold: any pair already worse than the stage's threshold
    # is clamped to INF so Hungarian's global optimum is computed only
    # over feasible pairs (matches the original cascade — without this
    # clamp the assignment can prefer a chain of bad pairs over a
    # single good one).
    cost[cost > stage.threshold] = INF

    apply_gates(cost, stage.gates, candidates, sub_dets)

    rows, cols = linear_sum_assignment(cost)

    matches: list[tuple[Track, int]] = []
    for r, c in zip(rows, cols):
        if cost[r, c] >= stage.threshold:
            continue
        track = candidates[r]
        det_idx = det_indices[c]
        matches.append((track, det_idx))
        unmatched_track_ids.discard(track.track_id)
        unmatched_det_idx.discard(det_idx)
    return matches
