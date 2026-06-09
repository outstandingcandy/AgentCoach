"""Track lifecycle policies — promotion, ageing, deletion.

Pulls the four "what should happen to a track this frame" rules into
one place so the tracker's update() doesn't have to inline them:

1. **Tick** age + time_since_update for unmatched tracks.
2. **Age out** tentative tracks that never got promoted in time.
3. **Stationary killer** — delete confirmed tracks whose bbox centre
   barely moved over the last ``window`` updates (YOLO false-positives
   on banners / fence posts / parked cars). Records a kill-zone so a
   persistent false-positive can't immediately respawn a fresh track.
4. **Stale cleanup** — delete tracks whose ``time_since_update``
   exceeds ``max_age`` and tracks already marked DELETED.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .track import Track, TrackStatus


@dataclass
class TrackLifecycle:
    """Configures and applies the per-frame lifecycle pass."""

    max_age: int = 30
    n_init: int = 3
    stationary_window: int = 30
    stationary_max_pixels: float = 5.0
    stationary_zone_radius: float = 25.0
    stationary_zone_ttl: int = 300

    # Active kill-zones: ``(cx, cy, ttl)``. Detections falling inside
    # one of these zones are suppressed at spawn time so a persistent
    # YOLO false-positive can't immediately respawn the ghost.
    stationary_zones: list[tuple[float, float, int]] = field(default_factory=list)

    def in_kill_zone(self, cx: float, cy: float) -> bool:
        """True if (cx, cy) is within radius of any active kill-zone."""
        if not self.stationary_zones:
            return False
        r2 = self.stationary_zone_radius ** 2
        for zx, zy, _ttl in self.stationary_zones:
            if (cx - zx) ** 2 + (cy - zy) ** 2 <= r2:
                return True
        return False

    def record_center(self, track: Track) -> None:
        """Push the current bbox centre onto the track's history ring,
        capped at ``stationary_window`` entries."""
        if not track.bbox or self.stationary_window <= 0:
            return
        x1, y1, x2, y2 = track.bbox
        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0
        track.center_history.append((cx, cy))
        if len(track.center_history) > self.stationary_window:
            del track.center_history[: -self.stationary_window]

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
        """Apply the four deletion rules and return the survivors.

        Order matters: stale-cleanup first (drops ancient tracks),
        then tentative aging, then stationary killer (which adds to
        ``stationary_zones``), then a final DELETED filter.
        """
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

        # 3. Stationary killer (only for confirmed tracks).
        if self.stationary_window > 0:
            for track in survivors:
                if track.status != TrackStatus.CONFIRMED:
                    continue
                if len(track.center_history) < self.stationary_window:
                    continue
                hist = track.center_history[-self.stationary_window:]
                xs = [c[0] for c in hist]
                ys = [c[1] for c in hist]
                span = max(max(xs) - min(xs), max(ys) - min(ys))
                if span <= self.stationary_max_pixels:
                    track.status = TrackStatus.DELETED
                    cx = sum(xs) / len(xs)
                    cy = sum(ys) / len(ys)
                    self.stationary_zones.append(
                        (cx, cy, self.stationary_zone_ttl)
                    )

        # 4. Decay kill-zones.
        self.stationary_zones = [
            (x, y, ttl - 1)
            for (x, y, ttl) in self.stationary_zones
            if ttl - 1 > 0
        ]

        # 5. Final DELETED filter (after stationary killer flagged some).
        return [t for t in survivors if t.status != TrackStatus.DELETED]
