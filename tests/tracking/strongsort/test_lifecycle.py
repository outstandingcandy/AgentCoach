"""Tests for TrackLifecycle (tick + cleanup + stationary killer)."""

from __future__ import annotations

import pytest

from goalinsight.tracking.strongsort.lifecycle import TrackLifecycle
from goalinsight.tracking.strongsort.track import Track, TrackStatus


_DEFAULT_BBOX = [0, 0, 10, 30]


def _track(track_id, status=TrackStatus.CONFIRMED, age=1, tsu=0,
           hits=3, bbox=None, history=None):
    t = Track(track_id=track_id, status=status, age=age,
              time_since_update=tsu, hits=hits,
              bbox=_DEFAULT_BBOX if bbox is None else bbox)
    if history is not None:
        t.center_history = list(history)
    return t


class TestTick:
    def test_unmatched_track_advances_tsu(self):
        lc = TrackLifecycle()
        t = _track(1, age=5, tsu=0)
        lc.tick([t], updated_ids=set())
        assert t.age == 6
        assert t.time_since_update == 1

    def test_matched_track_keeps_tsu_zero(self):
        lc = TrackLifecycle()
        t = _track(1, age=5, tsu=0)
        lc.tick([t], updated_ids={1})
        assert t.age == 6  # always increments
        assert t.time_since_update == 0

    def test_deleted_tracks_skipped(self):
        lc = TrackLifecycle()
        t = _track(1, status=TrackStatus.DELETED, age=5, tsu=0)
        lc.tick([t], updated_ids=set())
        assert t.age == 5  # unchanged
        assert t.time_since_update == 0


class TestCleanup:
    def test_drops_max_age_exceeded(self):
        lc = TrackLifecycle(max_age=10)
        old = _track(1, tsu=11)
        survivors = lc.cleanup([old])
        assert survivors == []

    def test_drops_already_marked_deleted(self):
        lc = TrackLifecycle()
        t = _track(1, status=TrackStatus.DELETED)
        survivors = lc.cleanup([t])
        assert survivors == []

    def test_ages_out_unconfirmed_tentative(self):
        lc = TrackLifecycle(n_init=3)
        # tentative + age > n_init+2 = 5 → DELETED
        t = _track(1, status=TrackStatus.TENTATIVE, age=6)
        survivors = lc.cleanup([t])
        assert survivors == []

    def test_keeps_young_tentative(self):
        lc = TrackLifecycle(n_init=3)
        t = _track(1, status=TrackStatus.TENTATIVE, age=4)
        survivors = lc.cleanup([t])
        assert len(survivors) == 1
        assert survivors[0].track_id == 1

    def test_stationary_killer_drops_motionless_track(self):
        lc = TrackLifecycle(stationary_window=3, stationary_max_pixels=5.0)
        # 3 entries, all within 5 px → killed.
        t = _track(1, history=[(100.0, 200.0)] * 3)
        survivors = lc.cleanup([t])
        assert survivors == []
        # Kill-zone recorded.
        assert len(lc.stationary_zones) == 1
        zx, zy, ttl = lc.stationary_zones[0]
        assert zx == pytest.approx(100.0)
        assert zy == pytest.approx(200.0)
        # TTL decays once at end of cleanup() → ttl_default - 1.
        assert ttl == lc.stationary_zone_ttl - 1

    def test_stationary_killer_keeps_moving_track(self):
        lc = TrackLifecycle(stationary_window=3, stationary_max_pixels=5.0)
        # spread > 5 px → not stationary.
        t = _track(1, history=[(100.0, 200.0), (110.0, 200.0), (120.0, 200.0)])
        survivors = lc.cleanup([t])
        assert len(survivors) == 1

    def test_stationary_killer_disabled_when_window_zero(self):
        lc = TrackLifecycle(stationary_window=0)
        # 30 entries, all motionless — but killer disabled.
        t = _track(1, history=[(100.0, 200.0)] * 30)
        survivors = lc.cleanup([t])
        assert len(survivors) == 1
        assert lc.stationary_zones == []

    def test_stationary_killer_skips_tentative(self):
        lc = TrackLifecycle(stationary_window=3, stationary_max_pixels=5.0)
        t = _track(1, status=TrackStatus.TENTATIVE,
                   history=[(100.0, 200.0)] * 3)
        survivors = lc.cleanup([t])
        # Tentative + age=1 < n_init+2=5 → kept; killer doesn't apply.
        assert len(survivors) == 1
        assert lc.stationary_zones == []


class TestKillZones:
    def test_in_kill_zone_within_radius(self):
        lc = TrackLifecycle(stationary_zone_radius=10.0)
        lc.stationary_zones = [(100.0, 200.0, 50)]
        assert lc.in_kill_zone(105.0, 200.0) is True
        assert lc.in_kill_zone(100.0, 209.0) is True

    def test_outside_radius(self):
        lc = TrackLifecycle(stationary_zone_radius=10.0)
        lc.stationary_zones = [(100.0, 200.0, 50)]
        assert lc.in_kill_zone(115.0, 200.0) is False

    def test_no_zones(self):
        lc = TrackLifecycle()
        assert lc.in_kill_zone(0, 0) is False

    def test_ttl_decay(self):
        lc = TrackLifecycle(stationary_window=0)  # disable killer
        lc.stationary_zones = [(100.0, 200.0, 2)]
        lc.cleanup([])  # decay step
        assert lc.stationary_zones[0][2] == 1
        lc.cleanup([])
        # ttl=0 → expired
        assert lc.stationary_zones == []


class TestRecordCenter:
    def test_appends_center_from_bbox(self):
        lc = TrackLifecycle(stationary_window=5)
        t = _track(1, bbox=[0, 0, 10, 30])
        lc.record_center(t)
        assert t.center_history == [(5.0, 15.0)]

    def test_caps_at_window(self):
        lc = TrackLifecycle(stationary_window=3)
        t = _track(1, bbox=[0, 0, 10, 30])
        for _ in range(5):
            lc.record_center(t)
        assert len(t.center_history) == 3

    def test_no_op_when_window_zero(self):
        lc = TrackLifecycle(stationary_window=0)
        t = _track(1, bbox=[0, 0, 10, 30])
        lc.record_center(t)
        assert t.center_history == []

    def test_no_op_with_empty_bbox(self):
        lc = TrackLifecycle(stationary_window=5)
        t = _track(1, bbox=[])
        lc.record_center(t)
        assert t.center_history == []


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
