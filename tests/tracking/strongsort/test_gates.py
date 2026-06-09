"""Tests for Gate abstractions and apply_gates helper."""

from __future__ import annotations

import numpy as np
import pytest

from goalinsight.tracking.strongsort.gates import INF, PitchGate, apply_gates
from goalinsight.tracking.strongsort.track import Track


def _track(track_id: int = 1, pitch_pos=(0.0, 0.0)) -> Track:
    return Track(track_id=track_id, pitch_pos=pitch_pos)


def _det(pitch_pos=None, bbox=None) -> dict:
    return {"bbox": bbox or [0, 0, 10, 30], "pitch_pos": pitch_pos}


class TestPitchGate:
    def test_within_threshold_passes(self):
        gate = PitchGate(threshold_m=3.0)
        assert gate(_track(pitch_pos=(0.0, 0.0)), _det(pitch_pos=(2.0, 0.0)))

    def test_at_threshold_passes(self):
        # exactly equal to threshold → passes (squared comparison <=)
        gate = PitchGate(threshold_m=3.0)
        assert gate(_track(pitch_pos=(0.0, 0.0)), _det(pitch_pos=(3.0, 0.0)))

    def test_outside_threshold_vetoes(self):
        gate = PitchGate(threshold_m=3.0)
        assert not gate(_track(pitch_pos=(0.0, 0.0)), _det(pitch_pos=(4.0, 0.0)))

    def test_diagonal_distance(self):
        # (3, 4) → 5 m, should fail at threshold 4
        gate = PitchGate(threshold_m=4.0)
        assert not gate(_track(pitch_pos=(0.0, 0.0)), _det(pitch_pos=(3.0, 4.0)))
        # threshold 5 → passes (=)
        gate = PitchGate(threshold_m=5.0)
        assert gate(_track(pitch_pos=(0.0, 0.0)), _det(pitch_pos=(3.0, 4.0)))

    def test_track_missing_pitch_skips(self):
        # No track anchor → cannot gate, allow
        gate = PitchGate(threshold_m=3.0)
        t = Track(track_id=1, pitch_pos=None)
        assert gate(t, _det(pitch_pos=(100.0, 100.0)))

    def test_detection_missing_pitch_skips(self):
        gate = PitchGate(threshold_m=3.0)
        assert gate(_track(pitch_pos=(0.0, 0.0)), _det(pitch_pos=None))


class TestApplyGates:
    def test_no_gates_is_identity(self):
        cost = np.array([[0.1, 0.2], [0.3, 0.4]])
        out = apply_gates(cost, [], [_track(1), _track(2)],
                          [_det(), _det()])
        np.testing.assert_array_equal(out, [[0.1, 0.2], [0.3, 0.4]])

    def test_pitch_gate_marks_far_pairs_inf(self):
        tracks = [_track(1, (0.0, 0.0)), _track(2, (10.0, 0.0))]
        dets = [_det(pitch_pos=(0.0, 0.0)), _det(pitch_pos=(10.0, 0.0))]
        cost = np.array([[0.1, 0.2], [0.3, 0.4]])
        apply_gates(cost, [PitchGate(3.0)], tracks, dets)
        # diagonal close, off-diagonal 10m apart → veto'd
        assert cost[0, 0] == 0.1
        assert cost[1, 1] == 0.4
        assert cost[0, 1] >= INF
        assert cost[1, 0] >= INF

    def test_skips_already_inf_cells(self):
        # Already-INF cells aren't redundantly tested by gates.
        # We can't observe the skip directly, but the result must
        # remain INF (and not flip to a smaller value).
        tracks = [_track(1, (0.0, 0.0))]
        dets = [_det(pitch_pos=(100.0, 100.0))]
        cost = np.array([[INF * 2]])
        apply_gates(cost, [PitchGate(3.0)], tracks, dets)
        assert cost[0, 0] >= INF

    def test_empty_cost_matrix_returns_unchanged(self):
        cost = np.zeros((0, 0))
        out = apply_gates(cost, [PitchGate(3.0)], [], [])
        assert out.shape == (0, 0)

    def test_multiple_gates_any_veto_marks_inf(self):
        # Custom gate that always vetoes; combined with PitchGate (passes).
        tracks = [_track(1, (0.0, 0.0))]
        dets = [_det(pitch_pos=(0.0, 0.0))]
        cost = np.array([[0.5]])

        class AlwaysVeto:
            def __call__(self, track, detection):
                return False

        apply_gates(cost, [PitchGate(3.0), AlwaysVeto()], tracks, dets)
        assert cost[0, 0] >= INF


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
