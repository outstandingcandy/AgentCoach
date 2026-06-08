"""Merge fragmented track_ids into stable player_ids.

Stage A — cluster tracks with the same (team, jersey_number) when their
          ReID centroids are cosine-similar enough.
Stage B — absorb orphan tracks (no jersey or low-confidence) into the
          nearest team-matched player centroid when cosine > threshold.
Stage C — name players: ``A-9``, ``B-10``, ``A-GK``, ``B-unk-01``, etc.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class TrackMeta:
    track_id: int
    team: str           # "team_A", "team_B"
    role: str           # "player", "goalkeeper"
    frame_count: int
    jersey_number: int | None
    jersey_confidence: float
    reid_centroid: np.ndarray | None = None   # L2-normalised


@dataclass
class PlayerCluster:
    player_id: str
    team: str
    jersey_number: int | None
    jersey_confidence: float
    source_tracks: list[int] = field(default_factory=list)
    reid_centroid: np.ndarray | None = None
    role: str = "player"  # "player" or "goalkeeper"

    def to_dict(self) -> dict[str, Any]:
        return {
            "player_id": self.player_id,
            "team": self.team,
            "role": self.role,
            "jersey_number": self.jersey_number,
            "jersey_confidence": round(self.jersey_confidence, 3),
            "source_tracks": sorted(self.source_tracks),
            "reid_centroid_norm": (
                float(np.linalg.norm(self.reid_centroid))
                if self.reid_centroid is not None else None
            ),
        }


def l2_normalize(vec: np.ndarray) -> np.ndarray | None:
    if vec is None:
        return None
    n = float(np.linalg.norm(vec))
    if n < 1e-6:
        return None
    return vec / n


def compute_centroid(embs: list[list[float]] | list[np.ndarray]) -> np.ndarray | None:
    if not embs:
        return None
    arr = np.asarray(embs, dtype=np.float32)
    if arr.ndim == 1:
        arr = arr[None, :]
    mean = arr.mean(axis=0)
    return l2_normalize(mean)


def cosine(a: np.ndarray | None, b: np.ndarray | None) -> float:
    if a is None or b is None:
        return 0.0
    return float(np.dot(a, b))


def _format_player_id(team: str, jersey: int | None, role: str,
                      unk_ordinal: int | None = None) -> str:
    suffix = team[5:] if team.startswith("team_") else team  # A / B
    if role == "goalkeeper" and jersey is None:
        return f"{suffix}-GK"
    if jersey is not None:
        return f"{suffix}-{jersey}"
    return f"{suffix}-unk-{unk_ordinal:02d}"


def consolidate(
    metas: list[TrackMeta],
    same_person_threshold: float = 0.0,
    orphan_absorb_threshold: float = 0.92,
    min_jersey_confidence: float = 0.6,
    cooccur_pairs: set[tuple[int, int]] | None = None,
) -> tuple[dict[int, str], list[PlayerCluster]]:
    """Return (track_id → player_id map, cluster list).

    Design note: in low-res broadcast footage the OSNet ReID embeddings
    of different same-team players overlap heavily (observed p50 ≈ 0.89
    same-team, p50 ≈ 0.81 cross-team), so ReID cannot reliably separate
    individuals on its own.  We therefore trust Claude's jersey reading
    as the primary identity signal and only use ReID as a gate for
    orphan absorption at a *high* cosine threshold.

    Args:
        metas: one :class:`TrackMeta` per candidate track.
        same_person_threshold: ReID cosine gate inside a (team, jersey)
            group.  Default 0.0 = merge unconditionally when Claude agrees
            on (team, jersey) and confidence is high enough.
        orphan_absorb_threshold: cosine above which an orphan track is
            glued onto the nearest team-matched player centroid.
        min_jersey_confidence: below this Claude confidence, the jersey
            number is treated as unknown (goes to orphan stage).
        cooccur_pairs: set of (lo_tid, hi_tid) pairs that were ever
            visible in the same video frame. Two such tids cannot be the
            same person (one body, one place) — :func:`_can_merge`
            rejects any merge whose target cluster contains a
            co-occurring source tid. Pass ``None`` to disable the guard.
    """
    cooccur_pairs = cooccur_pairs or set()

    def _conflicts(cluster: PlayerCluster, m: TrackMeta) -> bool:
        """True if merging ``m`` into ``cluster`` would put two
        co-occurring tids in the same player_id."""
        for tid in cluster.source_tracks:
            lo, hi = (tid, m.track_id) if tid < m.track_id else (m.track_id, tid)
            if (lo, hi) in cooccur_pairs:
                return True
        return False
    clusters: list[PlayerCluster] = []
    track_to_player: dict[int, str] = {}

    # ---- Stage A: confident-number clustering ---------------------------
    # Include role in the grouping key so that two unknown-team
    # goalkeepers (one per side, both possibly read as the same number)
    # don't collapse together; assign_goalkeepers can then place each
    # on its goal side.  For 'player' role, role-vs-team merging is
    # already implicit so this is a no-op.
    confident: dict[tuple[str, str, int], list[TrackMeta]] = {}
    orphans: list[TrackMeta] = []
    for m in metas:
        if (m.jersey_number is not None
                and m.jersey_confidence >= min_jersey_confidence):
            key = (m.team, m.role, m.jersey_number)
            confident.setdefault(key, []).append(m)
        else:
            orphans.append(m)

    for (team, _role, jersey), group in confident.items():
        # All tracks Claude tags as the same (team, jersey) collapse to
        # one player.  If same_person_threshold > 0, enforce a ReID gate
        # (useful if Claude is suspected to confuse similar numbers).
        group_sorted = sorted(group, key=lambda g: -g.frame_count)
        local_clusters: list[PlayerCluster] = []
        for m in group_sorted:
            # Pick the most-similar cluster that doesn't co-occur with m.
            # If two tids in this same-jersey group were on screen at the
            # same time, Claude misread the number on at least one of them
            # — keep them as separate clusters (will get -a/-b suffix).
            best_cluster: PlayerCluster | None = None
            best_sim = -1.0
            for c in local_clusters:
                if _conflicts(c, m):
                    continue
                sim = cosine(m.reid_centroid, c.reid_centroid)
                if sim > best_sim:
                    best_sim = sim
                    best_cluster = c
            if best_cluster is not None and (
                same_person_threshold <= 0
                or best_sim >= same_person_threshold
            ):
                _merge_into(best_cluster, m)
            else:
                local_clusters.append(_new_cluster(team, jersey, m))

        # If a (team, jersey) group split into >1 cluster (only possible
        # with same_person_threshold > 0), disambiguate with a suffix.
        for i, c in enumerate(local_clusters):
            if len(local_clusters) > 1:
                c.player_id = f"{c.player_id}-{chr(ord('a') + i)}"
            clusters.append(c)
            for tid in c.source_tracks:
                track_to_player[tid] = c.player_id

    # ---- Stage B: orphan absorption ------------------------------------
    unk_counters: dict[str, int] = {"team_A": 0, "team_B": 0}
    # Sort orphans so the largest get processed first (they seed new
    # unknown players; tiny fragments get absorbed into them).
    orphans.sort(key=lambda m: -m.frame_count)
    for m in orphans:
        # Try to absorb into a same-team confident cluster — but skip
        # clusters that share a frame with m (they belong to a different
        # body even if ReID looks similar).
        best_cluster: PlayerCluster | None = None
        best_sim = -1.0
        for c in clusters:
            if c.team != m.team:
                continue
            if _conflicts(c, m):
                continue
            sim = cosine(m.reid_centroid, c.reid_centroid)
            if sim > best_sim:
                best_sim = sim
                best_cluster = c
        if best_cluster is not None and best_sim >= orphan_absorb_threshold:
            _merge_into(best_cluster, m)
            track_to_player[m.track_id] = best_cluster.player_id
            continue

        # Try to absorb into another orphan-derived cluster (same team)
        best_cluster = None
        best_sim = -1.0
        for c in clusters:
            if c.team != m.team or c.jersey_number is not None:
                continue
            if _conflicts(c, m):
                continue
            sim = cosine(m.reid_centroid, c.reid_centroid)
            if sim > best_sim:
                best_sim = sim
                best_cluster = c
        if best_cluster is not None and best_sim >= orphan_absorb_threshold:
            _merge_into(best_cluster, m)
            track_to_player[m.track_id] = best_cluster.player_id
            continue

        # Seed a new unknown-number cluster
        unk_counters[m.team] = unk_counters.get(m.team, 0) + 1
        player_id = _format_player_id(m.team, None, m.role,
                                      unk_ordinal=unk_counters[m.team])
        cluster = PlayerCluster(
            player_id=player_id,
            team=m.team,
            jersey_number=None,
            jersey_confidence=0.0,
            source_tracks=[m.track_id],
            reid_centroid=m.reid_centroid,
        )
        clusters.append(cluster)
        track_to_player[m.track_id] = player_id

    return track_to_player, clusters


def _new_cluster(team: str, jersey: int, m: TrackMeta) -> PlayerCluster:
    return PlayerCluster(
        player_id=_format_player_id(team, jersey, m.role),
        team=team,
        jersey_number=jersey,
        jersey_confidence=m.jersey_confidence,
        source_tracks=[m.track_id],
        reid_centroid=m.reid_centroid,
    )


def _merge_into(c: PlayerCluster, m: TrackMeta) -> None:
    c.source_tracks.append(m.track_id)
    # Weighted centroid update (weight by track frame_count proxy = 1 for
    # simplicity; already L2-normalised means)
    if m.reid_centroid is not None:
        if c.reid_centroid is None:
            c.reid_centroid = m.reid_centroid
        else:
            stacked = np.stack([c.reid_centroid, m.reid_centroid], axis=0)
            merged = stacked.mean(axis=0)
            c.reid_centroid = l2_normalize(merged)
    # Update jersey confidence to the max (most-confident observation)
    c.jersey_confidence = max(c.jersey_confidence, m.jersey_confidence)


# ----------------------------------------------------------------------
# Post-consolidation goalkeeper identification
# ----------------------------------------------------------------------


def assign_goalkeepers(
    clusters: list[PlayerCluster],
    track_positions: dict[int, list[list[float]]],
    pitch_length: float,
    pitch_width: float,
    goal_area_depth: float = 16.5,
) -> list[tuple[str, str]]:
    """Identify each team's goalkeeper *after* consolidation.

    ``tracking.GoalkeeperDetector`` runs at the raw-track level; when
    tracking fragments the real goalkeeper into dozens of short tracks
    the "lead by 5m over second-closest" test never fires and no GK
    is ever reported.  By re-running the test on consolidated clusters
    — which pool the fragments back into one identity — we recover the
    GK cleanly.

    For each team:

    1. Compute each cluster's mean pitch position across all its source
       tracks' observations.
    2. Determine the team's **own goal** side from the team center-of-mass
       on the pitch — the team defending the left goal has more of its
       mass at negative x, vice versa for the right goal.
    3. Pick the cluster whose mean x is closest to that own-goal line.
       If its distance is ≤ ``goal_area_depth`` (16.5 m default), mark
       that cluster as the team's goalkeeper.

    Updates each selected cluster in-place: ``role='goalkeeper'`` and
    ``player_id`` is rewritten to ``A-GK`` / ``B-GK`` (or ``A-GK-<N>``
    preserving a known jersey number).

    Returns a list of ``(old_player_id, new_player_id)`` renames so the
    caller can update ``track_to_player`` mappings.
    """
    half_l = pitch_length / 2
    renames: list[tuple[str, str]] = []

    # Mean pitch position per cluster
    cluster_mean: dict[str, tuple[float, float]] = {}
    for c in clusters:
        xs, ys = [], []
        for tid in c.source_tracks:
            for pos in track_positions.get(tid, ()):
                if pos is None:
                    continue
                # Filter obvious calibration blowups (y outside plausible range)
                if abs(pos[0]) > pitch_length or abs(pos[1]) > pitch_width:
                    continue
                xs.append(float(pos[0]))
                ys.append(float(pos[1]))
        if xs:
            cluster_mean[c.player_id] = (float(np.mean(xs)), float(np.mean(ys)))

    # Team center-of-mass to decide own-goal side
    team_mean_x: dict[str, float] = {}
    for team in ("team_A", "team_B"):
        xs = [
            cluster_mean[c.player_id][0]
            for c in clusters
            if c.team == team and c.player_id in cluster_mean
        ]
        if xs:
            team_mean_x[team] = float(np.mean(xs))

    if len(team_mean_x) < 2:
        return renames

    # Assign own goal side (team with smaller mean x → defends left)
    sorted_teams = sorted(team_mean_x.items(), key=lambda kv: kv[1])
    team_goal_x: dict[str, float] = {
        sorted_teams[0][0]: -half_l,
        sorted_teams[1][0]: half_l,
    }

    # For each goal, consider ALL clusters (not just the cluster's nominal
    # team) and pick the single closest one.  This handles the common case
    # where KMeans on jersey colour misassigns the goalkeeper to the
    # opposing team because GK kit differs from teammates.
    goal_to_team: dict[float, str] = {v: k for k, v in team_goal_x.items()}

    assigned_pids: set[str] = set()
    for goal_x, defending_team in goal_to_team.items():
        # Score = -distance + small bonus per source track, so fragmented
        # goalkeepers (many short tracks near the own goal) win over
        # an isolated 1-source cluster that happens to be a metre closer.
        def gk_score(c: PlayerCluster) -> float:
            dist = abs(cluster_mean[c.player_id][0] - goal_x)
            return -dist + 0.25 * len(c.source_tracks)

        # Allow team='unknown' clusters too — Step 2 sometimes returns
        # goalkeeper with team unknown (the GK kit colour doesn't match
        # either outfield kit so the model declines to commit).  We'll
        # set the team from the goal side below.
        near_enough = [
            c for c in clusters
            if c.player_id in cluster_mean
            and abs(cluster_mean[c.player_id][0] - goal_x) <= goal_area_depth
            and c.player_id not in assigned_pids
            and c.role not in ("referee", "linesman")
            and c.team in (
                "team_A", "team_B", "unknown", "gk_left", "gk_right",
            )
        ]
        # Prefer clusters whose role is already 'goalkeeper' (Step 2
        # voted) — give them a large bonus so a fragmented but
        # explicitly-tagged GK wins over a random outfielder who
        # happens to be standing close to the goal.
        def _scored(c: PlayerCluster) -> float:
            base = gk_score(c)
            return base + (5.0 if c.role == "goalkeeper" else 0.0)
        gk = max(near_enough, key=_scored) if near_enough else None
        if gk is None:
            logger.info(
                "No goalkeeper found for %s (no cluster within %.1fm of own goal)",
                defending_team, goal_area_depth,
            )
            continue

        old_pid = gk.player_id
        old_team = gk.team
        # Correct team assignment if needed (KMiss-clustered GK)
        if gk.team != defending_team:
            logger.info(
                "GK team corrected: %s %s → %s (defends x=%.1f goal)",
                old_pid, old_team, defending_team, goal_x,
            )
            gk.team = defending_team

        suffix = defending_team[5:] if defending_team.startswith("team_") \
            else defending_team
        if gk.jersey_number is not None:
            new_pid = f"{suffix}-GK-{gk.jersey_number}"
        else:
            new_pid = f"{suffix}-GK"
        gk.role = "goalkeeper"
        gk.player_id = new_pid
        assigned_pids.add(new_pid)
        logger.info(
            "GK identified: %s → %s (team %s, dist=%.1fm, n_sources=%d)",
            old_pid, new_pid, defending_team,
            abs(cluster_mean[old_pid][0] - goal_x),
            len(gk.source_tracks),
        )
        if old_pid != new_pid:
            renames.append((old_pid, new_pid))

    return renames
