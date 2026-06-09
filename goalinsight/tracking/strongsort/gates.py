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
    ``threshold_m`` metres.

    Skips the gate (returns True) when either side lacks a pitch
    projection — better an ungated match than a wrongly-rejected one
    on calibration failures.

    At process_fps=10 a sprinting 8 m/s player covers ~0.8 m / sample;
    3 m leaves headroom for projection jitter on distant detections.
    """

    def __init__(self, threshold_m: float) -> None:
        self.threshold_m = float(threshold_m)
        self._threshold_sq = self.threshold_m ** 2

    def __call__(self, track: Track, detection: dict) -> bool:
        if track.pitch_pos is None:
            return True
        det_pp = detection.get("pitch_pos")
        if det_pp is None:
            return True
        dx = track.pitch_pos[0] - det_pp[0]
        dy = track.pitch_pos[1] - det_pp[1]
        return dx * dx + dy * dy <= self._threshold_sq


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
