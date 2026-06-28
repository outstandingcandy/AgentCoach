"""Track lifecycle policies — promotion, ageing, deletion.

Pulls the per-frame "what should happen to each track" rules into one
place so the tracker's update() doesn't have to inline them:

1. **Tick** age + time_since_update for unmatched tracks.
2. **Age out** tentative tracks that never got promoted in time.
3. **Stale cleanup** — delete tracks whose ``time_since_update``
   exceeds ``max_age`` and tracks already marked DELETED.

(The earlier "stationary killer + kill-zone" path that suppressed
detections at persistent banner/fence positions was removed: it
permanently banned legit players who stood still for a few seconds —
common in youth/futsal — from being re-detected for the next 300 frames.
False-positive banners are now expected to be filtered upstream via the
pitch-bound + size gates, not via a position-based spawn block.)
"""

from __future__ import annotations

from dataclasses import dataclass

from .track import Track, TrackStatus


@dataclass
class TrackLifecycle:
    """Configures and applies the per-frame lifecycle pass."""

    max_age: int = 30
    n_init: int = 3

    def tick(
        self,
        tracks: list[Track],
        updated_ids: set[int],
    ) -> None:
        """Bump age (always) + time_since_update (when no measurement)."""
        for track in tracks:
            if track.status == TrackStatus.DELETED:
                continue
            track.age += 1
            if track.track_id not in updated_ids:
                track.time_since_update += 1

    def cleanup(self, tracks: list[Track]) -> list[Track]:
        """Apply the deletion rules and return the survivors."""
        # 1. Drop tracks that exceeded max_age or were already marked.
        survivors = [
            t for t in tracks
            if t.status != TrackStatus.DELETED
            and t.time_since_update <= self.max_age
        ]

        # 2. Tentative tracks that never got promoted in time.
        for track in survivors:
            if (
                track.status == TrackStatus.TENTATIVE
                and track.age > self.n_init + 2
            ):
                track.status = TrackStatus.DELETED

        # 3. Final DELETED filter.
        return [t for t in survivors if t.status != TrackStatus.DELETED]
