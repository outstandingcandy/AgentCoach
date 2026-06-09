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


# Type alias for a cost function: (tracks, detections, embeddings) → (T, D) array.
CostFn = Callable[
    [list[Track], list[dict], np.ndarray | None],
    np.ndarray,
]


def cosine_cost(
    tracks: list[Track],
    detections: list[dict],
    embeddings: np.ndarray | None,
) -> np.ndarray:
    """ReID cosine distance between track ``smooth_feature`` and detection
    embedding. Tracks without ``smooth_feature`` get an INF row.
    """
    if embeddings is None or not tracks or not detections:
        return np.zeros((len(tracks), len(detections)))
    feats = []
    valid = []
    for i, t in enumerate(tracks):
        if t.smooth_feature is not None:
            feats.append(t.smooth_feature)
            valid.append(i)
    if not feats:
        return np.full((len(tracks), len(detections)), INF, dtype=np.float64)
    feats_arr = np.asarray(feats)
    sub = cdist(feats_arr, embeddings, metric="cosine")
    cost = np.full((len(tracks), len(detections)), INF, dtype=np.float64)
    for k, i in enumerate(valid):
        cost[i] = sub[k]
    return cost


def iou_cost(
    tracks: list[Track],
    detections: list[dict],
    embeddings: np.ndarray | None = None,
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
    embeddings: np.ndarray | None,
    unmatched_track_ids: set[int],
    unmatched_det_idx: set[int],
) -> list[tuple[Track, int]]:
    """Run a single matching stage. Returns list of (track, det_idx) pairs.

    ``unmatched_track_ids`` and ``unmatched_det_idx`` are mutated in
    place so subsequent stages see only the leftovers.

    Tracks must come from a stable list — we use ``id(track)`` as key
    in the unmatched set, so the caller is responsible for tracking
    object identity (typically by passing in ``self.tracks`` directly).
    """
    candidates = [
        t for t in tracks
        if id(t) in unmatched_track_ids and stage.track_filter(t)
    ]
    if not candidates or not detections:
        return []

    det_indices = sorted(unmatched_det_idx)
    if not det_indices:
        return []
    sub_dets = [detections[i] for i in det_indices]
    sub_embs = (
        embeddings[det_indices] if embeddings is not None else None
    )

    cost = stage.cost_fn(candidates, sub_dets, sub_embs)
    if cost.size == 0:
        return []

    # Pre-threshold: any pair already worse than the stage's threshold
    # is clamped to INF so Hungarian's global optimum is computed only
    # over feasible pairs (matches the original cascade — without this
    # clamp the assignment can prefer a chain of bad pairs over a
    # single good one).
    cost[cost > stage.threshold] = INF

    apply_gates(cost, stage.gates, candidates, sub_dets, sub_embs)

    rows, cols = linear_sum_assignment(cost)

    matches: list[tuple[Track, int]] = []
    for r, c in zip(rows, cols):
        if cost[r, c] >= stage.threshold:
            continue
        track = candidates[r]
        det_idx = det_indices[c]
        matches.append((track, det_idx))
        unmatched_track_ids.discard(id(track))
        unmatched_det_idx.discard(det_idx)
    return matches
