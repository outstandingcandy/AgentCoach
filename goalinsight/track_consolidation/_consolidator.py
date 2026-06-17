"""Merge fragmented track_ids into stable player_ids.

Stage A — ReID-first greedy clustering: pair up tracks whose centroids
          are cosine-similar (≥ same_person_threshold) and that never
          appeared in the same frame. ReID is the primary identity
          signal; jersey is used as a downstream label.
Stage B — within each ReID cluster, vote on a jersey number using the
          confidence-weighted Claude votes of the cluster's source
          tracks. This both fuses redundant high-conf jerseys and
          rescues a cluster from a single low-conf misread.
Stage C — split by team: tracks of different teams in the same ReID
          cluster get separated (cross-team ReID is unreliable on Veo
          footage; team is the cheaper hard constraint).
Stage D — orphan absorption: tracks that didn't reach the ReID
          threshold against any seed get a second chance against
          existing clusters at a (looser) absorb threshold.
Stage E — name players: ``A-9``, ``B-10``, ``A-GK``, ``B-unk-01``, etc.
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
    # Per-track jersey-vote breakdown from the LLM (number, weighted
    # score). The first entry is the winner used for jersey_number;
    # the remaining entries carry runner-up readings so the
    # consolidator can detect tracks that mix multiple physical
    # players (e.g. a tracker ID switch where the winning number is
    # only marginally ahead of a competing reading).
    jersey_candidates: list[tuple[int, float]] = field(default_factory=list)


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
    same_person_threshold: float = 0.9,
    orphan_absorb_threshold: float = 0.92,
    min_jersey_confidence: float = 0.6,
    cooccur_pairs: set[tuple[int, int]] | None = None,
    jersey_aware_merge_threshold: float = 0.8,
) -> tuple[dict[int, str], list[PlayerCluster]]:
    """Return (track_id → player_id map, cluster list).

    ReID-first design. Earlier versions trusted the LLM jersey vote as
    the bucket key and used ReID only as a tiebreaker; that mis-merged
    same-jersey-different-person cases (the kids dataset really has
    two #18s) and silently split same-person-misread-number cases. We
    now lead with ReID — every initial cluster is a connected
    component of (cosine ≥ ``same_person_threshold`` ∧ no cooccur), and
    the jersey number is voted on inside the cluster afterwards.

    The hand-tuned thresholds reflect PRTReID's distribution on Veo
    kids footage: same-person centroid cosine ≈ 0.93–0.96, different-
    same-team-player ≈ 0.78–0.87 (heavy overlap with the lower tail
    of same-person). 0.9 is the conservative cut.

    Args:
        metas: one :class:`TrackMeta` per candidate track.
        same_person_threshold: ReID cosine gate for the primary
            ReID-first clustering. Default 0.9.
        orphan_absorb_threshold: cosine above which a track that
            didn't seed a cluster in Stage A gets absorbed into an
            existing cluster afterwards. Should be ≥ same_person_thr;
            default 0.92.
        min_jersey_confidence: below this Claude confidence, a track's
            jersey vote is ignored when the cluster votes for its
            label. Higher-conf votes still win.
        cooccur_pairs: set of (lo_tid, hi_tid) pairs that were ever
            visible in the same video frame. Two such tids cannot be
            the same person; merges that would put both into the same
            cluster are rejected.
        jersey_aware_merge_threshold: in Stage C, two clusters with
            the same (team, jersey_number) and no cooccur are merged
            when their ReID cosine is at least this value. Lower than
            ``same_person_threshold`` because the jersey number is a
            second independent identity signal — same kit + same
            number + ReID-near-miss is still very likely the same
            player. Default 0.8.
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

    def _team_conflicts(cluster: PlayerCluster, m: TrackMeta) -> bool:
        """True if cluster and m belong to different *known* teams.

        team='unknown' is treated as compatible with anything so a
        track Claude couldn't team-classify can still join its real
        team via ReID. Two known-but-different teams are never the
        same body — keep them separate.
        """
        if cluster.team in ("unknown", "") or m.team in ("unknown", ""):
            return False
        return cluster.team != m.team

    clusters: list[PlayerCluster] = []
    track_to_player: dict[int, str] = {}

    # ---- Stage A: constrained connected-components clustering ----------
    # Build a graph where two tids share an edge iff:
    #   1. cosine(reid_i, reid_j) ≥ same_person_threshold
    #   2. (i, j) ∉ cooccur_pairs                  (physical impossibility)
    #   3. team_compatible(i, j)
    #   4. jersey numbers don't STRONGLY disagree  (cannot-link if both
    #      tids have a strong-confidence jersey vote pointing at
    #      different numbers — independent OCR observations of two
    #      different numbers is much stronger evidence of "two people"
    #      than ReID alone, especially on uniformed-team footage where
    #      cross-player cosine is 0.89-0.96 and overlaps same-player).
    # Then connected-components on the resulting graph, processing edges
    # in descending cosine order with an anti-transitivity guard so a
    # chain a-b-c never sneaks past a (a, c) cooccur or jersey-conflict.
    #
    # Why this replaces greedy ReID-first: the previous "process tracks
    # largest-first, attach to best existing cluster ≥ threshold" was
    # order-dependent and let a single tid pull a competing-jersey
    # cluster into its bucket once their centroids merged. Constrained
    # CC is order-invariant, transitive, and uses the LLM jersey vote
    # as a hard cannot-link signal at the right place.
    meta_by_tid = {m.track_id: m for m in metas}

    def _strong_jersey(m: TrackMeta) -> int | None:
        """Top-candidate jersey number iff its weighted score ≥ 1.0.

        Stage A uses a stricter bar than Stage B's
        ``min_jersey_confidence`` because here we're using OCR as a
        cannot-link signal — false positives split same-person tids.
        Weighted score ≥ 1.0 typically means at least 2 frames
        independently agreed on the number.
        """
        if not m.jersey_candidates:
            return None
        try:
            n, s = m.jersey_candidates[0]
        except (TypeError, ValueError):
            return None
        if float(s) < 1.0:
            return None
        try:
            return int(n)
        except (TypeError, ValueError):
            return None

    def _jersey_conflicts(a: TrackMeta, b: TrackMeta) -> bool:
        ja, jb = _strong_jersey(a), _strong_jersey(b)
        return ja is not None and jb is not None and ja != jb

    # Order-stable but determinism-friendly: union-find indexed by tid.
    parent = {m.track_id: m.track_id for m in metas}

    def _find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def _comp_members(root: int) -> list[int]:
        return [t for t in parent if _find(t) == root]

    # Pre-build cluster-level cannot-link checker. We keep it as a
    # function so the anti-transitivity guard sees the live components.
    def _would_violate(ra: int, rb: int) -> bool:
        members_a = _comp_members(ra)
        members_b = _comp_members(rb)
        for x in members_a:
            mx = meta_by_tid[x]
            for y in members_b:
                my = meta_by_tid[y]
                lo, hi = (x, y) if x < y else (y, x)
                if (lo, hi) in cooccur_pairs:
                    return True
                if _jersey_conflicts(mx, my):
                    return True
                # Different known teams stay split.
                if (mx.team not in ("unknown", "")
                        and my.team not in ("unknown", "")
                        and mx.team != my.team):
                    return True
        return False

    # Build the candidate edge list (passes the local pair filter).
    edges: list[tuple[float, int, int]] = []
    metas_indexed = list(metas)
    for i in range(len(metas_indexed)):
        a = metas_indexed[i]
        for j in range(i + 1, len(metas_indexed)):
            b = metas_indexed[j]
            if a.reid_centroid is None or b.reid_centroid is None:
                continue
            lo, hi = (a.track_id, b.track_id) if a.track_id < b.track_id \
                else (b.track_id, a.track_id)
            if (lo, hi) in cooccur_pairs:
                continue
            if (a.team not in ("unknown", "") and b.team not in ("unknown", "")
                    and a.team != b.team):
                continue
            if _jersey_conflicts(a, b):
                continue
            sim = cosine(a.reid_centroid, b.reid_centroid)
            if sim >= same_person_threshold:
                edges.append((sim, a.track_id, b.track_id))

    # Strongest edges merge first (deterministic tie-break by tid).
    edges.sort(key=lambda e: (-e[0], e[1], e[2]))
    for _sim, a_tid, b_tid in edges:
        ra, rb = _find(a_tid), _find(b_tid)
        if ra == rb:
            continue
        if _would_violate(ra, rb):
            continue
        # Lower tid wins as the root (stable, deterministic).
        if ra < rb:
            parent[rb] = ra
        else:
            parent[ra] = rb

    # Materialise clusters from the connected components. Each cluster's
    # centroid is the L2-normalised mean of its tids' centroids; the
    # seed metadata (team / role) is taken from the largest member —
    # Stage B will re-vote on it anyway.
    components: dict[int, list[TrackMeta]] = {}
    for m in metas:
        components.setdefault(_find(m.track_id), []).append(m)

    seeded: list[PlayerCluster] = []
    for tids in components.values():
        # Pick the largest tid by frame_count as the seed metadata
        # provider; stage B re-votes anyway so this is just a stub.
        seed = max(tids, key=lambda m: m.frame_count)
        cents = [m.reid_centroid for m in tids if m.reid_centroid is not None]
        merged_centroid = (
            l2_normalize(np.mean(np.stack(cents, axis=0), axis=0))
            if cents else None
        )
        seeded.append(PlayerCluster(
            player_id="",
            team=seed.team,
            role=seed.role,
            jersey_number=None,
            jersey_confidence=max(
                (m.jersey_confidence for m in tids), default=0.0
            ),
            source_tracks=[m.track_id for m in tids],
            reid_centroid=merged_centroid,
        ))

    # ---- Stage B: confidence-weighted jersey vote per cluster ----------
    # For each cluster, look at all source tracks and pick the jersey
    # with highest summed confidence (only votes ≥ min_jersey_confidence
    # count). A single high-conf vote can override several mid-conf
    # disagreements, but two equally-confident different numbers will
    # leave the cluster's number as None (handled at orphan stage).
    # ``meta_by_tid`` was already built in Stage A.
    for c in seeded:
        c.jersey_number, c.jersey_confidence = _vote_jersey_for_cluster(
            c, meta_by_tid, min_jersey_confidence,
        )
        # Refresh team / role from the cluster majority too — the seed
        # tid's values may not represent the merged group well.
        c.team = _vote_team_for_cluster(c, meta_by_tid)
        c.role = _vote_role_for_cluster(c, meta_by_tid)

    # ---- Stage B.5: jersey-aware second-pass merge -------------------
    # ReID alone misses long-interval reappearances (lighting / viewpoint
    # drift makes the same player's centroid cosine drop to 0.85 even
    # when no other player matches). When two clusters land on the same
    # (team, jersey_number), don't co-occur, and have ReID cosine ≥
    # ``jersey_aware_merge_threshold`` (default 0.8 — well above the
    # cross-team noise floor), merge them: jersey is an independent
    # identity signal that ReID alone shouldn't override.
    #
    # Symptom this fixes on the kids clip: orig 5 (A-20, frames 0-346,
    # 344 frames) and orig 199 (also #20, frames 353-403, 44 frames) had
    # ReID cosine 0.847. They are the same #20 player walking out then
    # back in. Without this stage they end up as A-20 and A-unk-NN.
    #
    # Largest cluster keeps its identity; smaller ones merge in. Process
    # at most one merge per cluster per pass — repeat until stable.
    def _jersey_aware_merge() -> bool:
        merged_any = False
        # Group by (team, jersey) so we only consider plausible pairs.
        groups: dict[tuple[str, int], list[PlayerCluster]] = {}
        for c in seeded:
            if c.jersey_number is None:
                continue
            groups.setdefault((c.team, c.jersey_number), []).append(c)
        for group in groups.values():
            if len(group) < 2:
                continue
            # Largest first; smaller fragments fold into larger ones.
            group.sort(key=lambda c: -len(c.source_tracks))
            target = group[0]
            for src in group[1:]:
                if src is target or src not in seeded:
                    continue
                if _conflicts_clusters(target, src, cooccur_pairs):
                    continue
                sim = cosine(target.reid_centroid, src.reid_centroid)
                if sim < jersey_aware_merge_threshold:
                    continue
                _absorb_cluster(target, src)
                seeded.remove(src)
                merged_any = True
        return merged_any

    # Loop in case A absorbs B and now A's centroid drifts close enough
    # to absorb C. Bounded by cluster count.
    for _ in range(len(seeded)):
        if not _jersey_aware_merge():
            break

    # ---- Stage C: name and finalise ----------------------------------
    # ReID is the trusted identity signal. If two ReID-distinct clusters
    # STILL agree on (team, jersey) after Stage B.5 it means the LLM
    # read the same number on two physically different people
    # (genuinely two #18s, OR LLM error on one of them). The cluster
    # with the most source observations keeps the canonical "A-18"
    # name; the rest get demoted to "A-unk-NN".
    by_label: dict[tuple[str, int | None], list[PlayerCluster]] = {}
    for c in seeded:
        if c.jersey_number is None:
            continue  # orphans named in Stage D
        by_label.setdefault((c.team, c.jersey_number), []).append(c)

    unk_counters: dict[str, int] = {"team_A": 0, "team_B": 0, "unknown": 0}
    for (team, jersey), group in by_label.items():
        # Largest cluster (by source_tracks count, ties broken by min tid)
        # wins the canonical name; rest demote.
        group.sort(key=lambda c: (-len(c.source_tracks), min(c.source_tracks)))
        for i, c in enumerate(group):
            if i == 0:
                c.player_id = _format_player_id(team, jersey, c.role)
            else:
                team_key = c.team if c.team in unk_counters else "unknown"
                unk_counters[team_key] = unk_counters.get(team_key, 0) + 1
                # Demote: keep jersey_number on the cluster meta for
                # debugging but the player_id reflects "ReID says
                # different person, jersey collision irrelevant".
                c.player_id = _format_player_id(
                    c.team, None, c.role,
                    unk_ordinal=unk_counters[team_key],
                )
            clusters.append(c)
            for tid in c.source_tracks:
                track_to_player[tid] = c.player_id

    # ---- Stage D: orphan naming + absorption -------------------------
    # Clusters that came out of Stage A without a confident jersey
    # vote are now the orphans. First try to absorb each into an
    # already-named cluster (same team, no cooccur) at the higher
    # orphan_absorb_threshold; if that fails, give them an unk-NN id.
    # ``unk_counters`` continues from Stage C so demoted-by-jersey-
    # collision and orphan-by-no-jersey share the same NN sequence.
    orphans = [c for c in seeded if c.jersey_number is None]
    orphans.sort(key=lambda c: -len(c.source_tracks))
    for orph in orphans:
        # Try absorption into a same-team named cluster.
        best_cluster: PlayerCluster | None = None
        best_sim = -1.0
        for c in clusters:
            if c.team != orph.team and orph.team != "unknown":
                continue
            if _conflicts_clusters(c, orph, cooccur_pairs):
                continue
            sim = cosine(orph.reid_centroid, c.reid_centroid)
            if sim > best_sim:
                best_sim = sim
                best_cluster = c
        if best_cluster is not None and best_sim >= orphan_absorb_threshold:
            _absorb_cluster(best_cluster, orph)
            for tid in orph.source_tracks:
                track_to_player[tid] = best_cluster.player_id
            continue
        # Otherwise: name as unk-NN and keep as its own cluster.
        team_key = orph.team if orph.team in unk_counters else "unknown"
        unk_counters[team_key] = unk_counters.get(team_key, 0) + 1
        orph.player_id = _format_player_id(
            orph.team, None, orph.role, unk_ordinal=unk_counters[team_key],
        )
        clusters.append(orph)
        for tid in orph.source_tracks:
            track_to_player[tid] = orph.player_id

    # ---- Stage E: dedupe colliding player_ids ------------------------
    # Pure jersey-less GK clusters share ``<suffix>-GK`` because
    # _format_player_id has no ordinal slot for them; if 2+ such
    # clusters survive (LLM mis-labels several distinct off-pitch
    # detections as goalkeeper) they'd render as one ID. Append
    # ``-a / -b / ...`` so the front-end can show them as siblings.
    counts: dict[str, int] = {}
    for c in clusters:
        counts[c.player_id] = counts.get(c.player_id, 0) + 1
    seen_ord: dict[str, int] = {}
    for c in clusters:
        if counts.get(c.player_id, 0) <= 1:
            continue
        # Skip referee / linesman: those are deliberately pooled by the
        # caller (_build_officiating_clusters) into REF-NN tags.
        if c.role in ("referee", "linesman"):
            continue
        idx = seen_ord.get(c.player_id, 0)
        seen_ord[c.player_id] = idx + 1
        new_pid = f"{c.player_id}-{chr(ord('a') + idx)}"
        old_pid = c.player_id
        c.player_id = new_pid
        for tid in c.source_tracks:
            if track_to_player.get(tid) == old_pid:
                track_to_player[tid] = new_pid

    return track_to_player, clusters


def _vote_jersey_for_cluster(
    cluster: PlayerCluster,
    meta_by_tid: dict[int, "TrackMeta"],
    min_conf: float,
) -> tuple[int | None, float]:
    """Pick the cluster's representative jersey number from its source
    tracks' per-crop voting breakdown.

    For each source track, sum over all candidate (number, score) pairs
    the LLM produced — not just the winning number. This means a
    track whose 15 crops voted ``[(20, 0.8), (10, 0.4)]`` contributes
    to BOTH the 20 and 10 buckets when ReID merges it with another
    track whose breakdown was ``[(20, 1.5)]`` — total: 20 → 2.3, 10 →
    0.4, so the cluster correctly resolves to 20 even though one
    member had a contaminated reading. Falls back to the legacy
    ``jersey_number / jersey_confidence`` pair when ``jersey_candidates``
    is empty (caches written before per-crop voting landed).

    Votes from tracks whose top-line confidence is below ``min_conf``
    are skipped entirely so that low-confidence noise tracks don't
    drown out a small-but-clear vote.
    """
    score: dict[int, float] = {}
    max_conf: dict[int, float] = {}
    for tid in cluster.source_tracks:
        m = meta_by_tid.get(tid)
        if m is None:
            continue
        # Skip tracks the LLM was uncertain about overall.
        if m.jersey_confidence < min_conf and m.jersey_number is not None:
            continue
        if m.jersey_candidates:
            for n, s in m.jersey_candidates:
                score[n] = score.get(n, 0.0) + float(s)
                max_conf[n] = max(max_conf.get(n, 0.0), float(s))
        elif m.jersey_number is not None and m.jersey_confidence >= min_conf:
            score[m.jersey_number] = score.get(m.jersey_number, 0.0) + m.jersey_confidence
            max_conf[m.jersey_number] = max(
                max_conf.get(m.jersey_number, 0.0), m.jersey_confidence)
    if not score:
        return None, 0.0
    winner = max(score.items(), key=lambda kv: kv[1])[0]
    return winner, max_conf[winner]


def _vote_team_for_cluster(
    cluster: PlayerCluster,
    meta_by_tid: dict[int, "TrackMeta"],
) -> str:
    """Pick the cluster's team by frame-count-weighted majority of its
    source tracks' team labels. team='unknown' counts only as a
    fallback when no labelled vote exists."""
    score: dict[str, int] = {}
    for tid in cluster.source_tracks:
        m = meta_by_tid.get(tid)
        if m is None:
            continue
        if m.team in ("unknown", ""):
            continue
        score[m.team] = score.get(m.team, 0) + max(1, m.frame_count)
    if not score:
        # All votes were 'unknown' — keep whatever the seed had.
        return cluster.team
    return max(score.items(), key=lambda kv: kv[1])[0]


def _vote_role_for_cluster(
    cluster: PlayerCluster,
    meta_by_tid: dict[int, "TrackMeta"],
) -> str:
    """Pick the cluster's role by majority. 'goalkeeper' wins if any
    source voted GK (rare false-positive; assign_goalkeepers re-checks
    by position later)."""
    has_gk = False
    score: dict[str, int] = {}
    for tid in cluster.source_tracks:
        m = meta_by_tid.get(tid)
        if m is None:
            continue
        if m.role == "goalkeeper":
            has_gk = True
        score[m.role] = score.get(m.role, 0) + max(1, m.frame_count)
    if has_gk:
        return "goalkeeper"
    if not score:
        return cluster.role
    return max(score.items(), key=lambda kv: kv[1])[0]


def _conflicts_clusters(
    a: PlayerCluster,
    b: PlayerCluster,
    cooccur_pairs: set[tuple[int, int]],
) -> bool:
    """True if any tid in ``a`` co-occurs with any tid in ``b``."""
    for ta in a.source_tracks:
        for tb in b.source_tracks:
            lo, hi = (ta, tb) if ta < tb else (tb, ta)
            if (lo, hi) in cooccur_pairs:
                return True
    return False


def _absorb_cluster(target: PlayerCluster, src: PlayerCluster) -> None:
    """Fold ``src``'s source tracks + ReID centroid into ``target``."""
    target.source_tracks.extend(src.source_tracks)
    if src.reid_centroid is not None:
        if target.reid_centroid is None:
            target.reid_centroid = src.reid_centroid
        else:
            stacked = np.stack(
                [target.reid_centroid, src.reid_centroid], axis=0)
            merged = stacked.mean(axis=0)
            target.reid_centroid = l2_normalize(merged)
    target.jersey_confidence = max(
        target.jersey_confidence, src.jersey_confidence)


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
        # Hard y-offset filter: real GKs sit within ~6m of the goal
        # centreline. Anything further off-axis (corner area, behind-
        # goal coach / spectator) cannot be the goalkeeper, regardless
        # of how close they are to the goal line in x. orig 95 in the
        # kids clip — coach behind the corner at y=10.2m, x=-33.5 —
        # was getting picked because the LLM saw "dark kit + close to
        # goal line" and tagged it goalkeeper.
        GK_MAX_Y_OFFSET = 6.0
        near_enough = [
            c for c in clusters
            if c.player_id in cluster_mean
            and abs(cluster_mean[c.player_id][0] - goal_x) <= goal_area_depth
            and abs(cluster_mean[c.player_id][1]) <= GK_MAX_Y_OFFSET
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
