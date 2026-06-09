"""Tests for matching pipeline (cost functions + run_stage)."""

from __future__ import annotations

import numpy as np
import pytest

from goalinsight.tracking.strongsort.gates import INF, PitchGate
from goalinsight.tracking.strongsort.matching import (
    MatchingStage,
    cosine_cost,
    iou_cost,
    run_stage,
)
from goalinsight.tracking.strongsort.track import Track, TrackStatus


def _track(track_id: int, smooth_feat=None, bbox=None, pitch_pos=None,
           status=TrackStatus.CONFIRMED) -> Track:
    t = Track(track_id=track_id, status=status, bbox=bbox or [0, 0, 10, 30],
              pitch_pos=pitch_pos)
    t.smooth_feature = smooth_feat
    return t


def _det(bbox=None, pitch_pos=None, embedding=None) -> dict:
    return {
        "bbox": bbox or [0, 0, 10, 30],
        "pitch_pos": pitch_pos,
        "embedding": embedding,
    }


def _norm(v: np.ndarray) -> np.ndarray:
    return v / (np.linalg.norm(v) + 1e-9)


class TestCosineCost:
    def test_identical_features_cost_zero(self):
        f = _norm(np.array([1.0, 0.0, 0.0]))
        tracks = [_track(1, smooth_feat=f)]
        dets = [_det(embedding=f)]
        cost = cosine_cost(tracks, dets)
        assert cost.shape == (1, 1)
        assert cost[0, 0] == pytest.approx(0.0, abs=1e-6)

    def test_orthogonal_features_cost_one(self):
        f1 = _norm(np.array([1.0, 0.0]))
        f2 = _norm(np.array([0.0, 1.0]))
        tracks = [_track(1, smooth_feat=f1)]
        dets = [_det(embedding=f2)]
        cost = cosine_cost(tracks, dets)
        assert cost[0, 0] == pytest.approx(1.0, abs=1e-6)

    def test_track_without_feature_row_inf(self):
        f = _norm(np.array([1.0, 0.0]))
        tracks = [_track(1, smooth_feat=None), _track(2, smooth_feat=f)]
        dets = [_det(embedding=f)]
        cost = cosine_cost(tracks, dets)
        assert cost[0, 0] >= INF        # no feature → INF
        assert cost[1, 0] == pytest.approx(0.0, abs=1e-6)

    def test_detection_without_embedding_column_inf(self):
        f = _norm(np.array([1.0, 0.0]))
        tracks = [_track(1, smooth_feat=f)]
        dets = [_det(embedding=None), _det(embedding=f)]
        cost = cosine_cost(tracks, dets)
        assert cost[0, 0] >= INF
        assert cost[0, 1] == pytest.approx(0.0, abs=1e-6)

    def test_empty_inputs(self):
        assert cosine_cost([], []).shape == (0, 0)
        f = _norm(np.array([1.0, 0.0]))
        assert cosine_cost([_track(1, smooth_feat=f)], []).shape == (1, 0)


class TestIouCost:
    def test_identical_bbox_cost_zero(self):
        tracks = [_track(1, bbox=[0, 0, 10, 30])]
        dets = [_det(bbox=[0, 0, 10, 30])]
        cost = iou_cost(tracks, dets)
        assert cost[0, 0] == pytest.approx(0.0, abs=1e-6)

    def test_disjoint_bbox_cost_one(self):
        tracks = [_track(1, bbox=[0, 0, 10, 10])]
        dets = [_det(bbox=[100, 100, 110, 110])]
        cost = iou_cost(tracks, dets)
        assert cost[0, 0] == pytest.approx(1.0, abs=1e-6)

    def test_half_overlap(self):
        # Two 10x10 boxes overlapping 5x10 → IoU = 50/(100+100-50)=1/3
        tracks = [_track(1, bbox=[0, 0, 10, 10])]
        dets = [_det(bbox=[5, 0, 15, 10])]
        cost = iou_cost(tracks, dets)
        assert cost[0, 0] == pytest.approx(1.0 - 1.0/3.0, abs=1e-6)

    def test_track_without_bbox_row_unchanged(self):
        # Empty bbox → cost stays at default (1.0)
        t = Track(track_id=1, bbox=[])
        cost = iou_cost([t], [_det(bbox=[0, 0, 10, 10])])
        assert cost[0, 0] == 1.0


class TestRunStage:
    def test_simple_match(self):
        stage = MatchingStage(
            name="iou", track_filter=lambda t: True,
            cost_fn=iou_cost, threshold=0.7,
        )
        tracks = [_track(1, bbox=[0, 0, 10, 10])]
        dets = [_det(bbox=[0, 0, 10, 10])]
        unmatched_t, unmatched_d = {1}, {0}
        matches = run_stage(stage, tracks, dets, unmatched_t, unmatched_d)
        assert len(matches) == 1
        assert matches[0][0].track_id == 1
        assert matches[0][1] == 0
        assert unmatched_t == set()
        assert unmatched_d == set()

    def test_filter_excludes_track(self):
        # track_filter rejects tid 1 — should not match.
        stage = MatchingStage(
            name="only-confirmed",
            track_filter=lambda t: t.status == TrackStatus.CONFIRMED,
            cost_fn=iou_cost, threshold=0.7,
        )
        tracks = [_track(1, bbox=[0, 0, 10, 10], status=TrackStatus.TENTATIVE)]
        dets = [_det(bbox=[0, 0, 10, 10])]
        matches = run_stage(stage, tracks, dets, {1}, {0})
        assert matches == []

    def test_threshold_rejects_bad_match(self):
        # IoU 0 → cost 1.0 > threshold 0.7 → no match.
        stage = MatchingStage(
            name="iou", track_filter=lambda t: True,
            cost_fn=iou_cost, threshold=0.7,
        )
        tracks = [_track(1, bbox=[0, 0, 10, 10])]
        dets = [_det(bbox=[100, 100, 110, 110])]
        unmatched_t, unmatched_d = {1}, {0}
        matches = run_stage(stage, tracks, dets, unmatched_t, unmatched_d)
        assert matches == []
        # neither side consumed
        assert unmatched_t == {1}
        assert unmatched_d == {0}

    def test_pitch_gate_blocks_far_match(self):
        stage = MatchingStage(
            name="reid+pitch", track_filter=lambda t: True,
            cost_fn=cosine_cost,
            threshold=0.3,
            gates=[PitchGate(threshold_m=3.0)],
        )
        f = _norm(np.array([1.0, 0.0]))
        tracks = [_track(1, smooth_feat=f, pitch_pos=(0.0, 0.0))]
        # Same feature (cosine=0) but 10m away → gated out.
        dets = [_det(embedding=f, pitch_pos=(10.0, 0.0))]
        matches = run_stage(stage, tracks, dets, {1}, {0})
        assert matches == []

    def test_only_unmatched_tracks_considered(self):
        # tid 1 already matched → only tid 2 should compete.
        stage = MatchingStage(
            name="iou", track_filter=lambda t: True,
            cost_fn=iou_cost, threshold=0.7,
        )
        tracks = [
            _track(1, bbox=[0, 0, 10, 10]),    # better fit, but unmatched={2}
            _track(2, bbox=[0, 0, 10, 10]),
        ]
        dets = [_det(bbox=[0, 0, 10, 10])]
        unmatched_t = {2}
        matches = run_stage(stage, tracks, dets, unmatched_t, {0})
        assert [m[0].track_id for m in matches] == [2]

    def test_global_optimum_not_greedy(self):
        # Greedy would give A→det0 (best for A), forcing B→det1 worse.
        # Hungarian should find the global minimum total.
        stage = MatchingStage(
            name="iou", track_filter=lambda t: True,
            cost_fn=iou_cost, threshold=0.99,
        )
        tracks = [
            _track(1, bbox=[0, 0, 10, 10]),
            _track(2, bbox=[100, 0, 110, 10]),
        ]
        dets = [
            _det(bbox=[0, 0, 10, 10]),       # ideal for tid 1
            _det(bbox=[100, 0, 110, 10]),    # ideal for tid 2
        ]
        matches = run_stage(stage, tracks, dets, {1, 2}, {0, 1})
        # Each track should match its co-located detection.
        match_map = {m[0].track_id: m[1] for m in matches}
        assert match_map == {1: 0, 2: 1}


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
