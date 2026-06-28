"""Tests for TrackLifecycle (tick + cleanup)."""

from __future__ import annotations

from goalinsight.tracking.strongsort.lifecycle import TrackLifecycle
from goalinsight.tracking.strongsort.track import Track, TrackStatus


_DEFAULT_BBOX = [0, 0, 10, 30]


def _track(track_id, status=TrackStatus.CONFIRMED, age=1, tsu=0,
           hits=3, bbox=None):
    return Track(
        track_id=track_id, status=status, age=age,
        time_since_update=tsu, hits=hits,
        bbox=_DEFAULT_BBOX if bbox is None else bbox,
    )


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

    def test_keeps_motionless_confirmed_track(self):
        # No stationary-killer any more: a player who stands still must
        # stay tracked, otherwise re-detection is suppressed forever by
        # the missing killer + kill-zone path. Regression test.
        lc = TrackLifecycle()
        t = _track(1, status=TrackStatus.CONFIRMED, age=100, tsu=0)
        survivors = lc.cleanup([t])
        assert len(survivors) == 1


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])
