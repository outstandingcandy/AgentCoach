"""Track consolidation runner — ties sampler, Claude recognizer, and
consolidator together, then rewrites tracks.json / team_assignments.json
with stable player_ids.
"""

from __future__ import annotations

import json
import logging
import shutil
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np

from ._consolidator import (
    PlayerCluster,
    TrackMeta,
    assign_goalkeepers,
    compute_centroid,
    consolidate,
)
from ._sampler import sample_crops_for_tracks

logger = logging.getLogger(__name__)


def run_track_consolidation(
    output_dir: str | Path,
    pipeline_output_dir: str | Path,
    video_path: str | Path,
    config: dict[str, Any],
) -> dict[str, Any]:
    """Run the consolidation pipeline.

    Args:
        output_dir: This stage's output dir (``<run>/track_consolidation``).
        pipeline_output_dir: Root pipeline run dir (contains ``tracking/``).
        video_path: Source video.
        config: ``track_consolidation:`` section from the active YAML.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    pipeline_output_dir = Path(pipeline_output_dir)
    tracking_dir = pipeline_output_dir / "tracking"
    if not (tracking_dir / "tracks.json").exists():
        raise RuntimeError(
            f"tracking/tracks.json not found under {pipeline_output_dir}; "
            "run the tracking stage first.")

    # --- Load inputs --------------------------------------------------
    tracks_by_frame = _load_json(tracking_dir / "tracks.json")
    track_features = _load_json(tracking_dir / "track_features.json") or {}

    # --- Collect per-track frame counts + positions ------------------
    # Role / team are assigned here (tracker no longer writes them).
    frame_count_by_track: dict[int, int] = Counter()
    positions_by_track: dict[int, list[list[float]]] = {}
    for fk, tracks in tracks_by_frame.items():
        for t in tracks:
            tid = int(t["track_id"])
            frame_count_by_track[tid] += 1
            pp = t.get("pitch_position")
            if pp is not None:
                positions_by_track.setdefault(tid, []).append(pp)

    pitch_length, pitch_width = _pitch_dims(pipeline_output_dir)

    # --- Position labelling (spatial context for Claude) --------------
    # Every track (including off-field ones) gets a coarse label so
    # Claude can use spatial context to judge role (e.g. a track that
    # sits on the sideline is more likely a linesman or coach).
    position_label_by_track = _compute_position_labels(
        positions_by_track,
        pitch_length=pitch_length,
        pitch_width=pitch_width,
    )
    label_counts: dict[str, int] = {}
    for lab in position_label_by_track.values():
        label_counts[lab] = label_counts.get(lab, 0) + 1
    logger.info("Position labels: %s", label_counts)

    # --- Movement-pattern descriptions (strongest role signal) -------
    # Pre-compute a textual movement summary per candidate track so the
    # VLM can use motion (range, sideline-hugging, goal-tightness) as
    # the primary cue for goalkeeper / referee / linesman.
    movement_by_track = _compute_movement_descriptions(
        positions_by_track,
        pitch_length=pitch_length,
        pitch_width=pitch_width,
    )

    # --- Team classification ------------------------------------------
    # OSNet ReID is not reliable for team colour separation on Veo
    # footage (measured cross-team cosines overlap same-team cosines),
    # so we let Claude group a small set of seed crops into
    # team_A / team_B / official / other, then propagate those labels.
    team_by_track: dict[int, str] = {}

    # Tracker does not assign roles any more; treat every track as a
    # player entering consolidation. GK identification happens later.
    role_by_track: dict[int, str] = {
        tid: "player" for tid in frame_count_by_track
    }

    min_frames = int(config.get("min_track_frames", 30))
    # All tracks with enough observations go to Claude, regardless of
    # KMeans team label or pitch position — Claude decides the final
    # role (player / coach / referee / linesman / other).
    candidates = [
        tid for tid, n in frame_count_by_track.items()
        if n >= min_frames
    ]
    logger.info(
        "Candidates: %d tracks (of %d total) pass min_frames=%d",
        len(candidates), len(frame_count_by_track), min_frames,
    )

    # --- Cached votes? Skip LLM passes if jersey_votes.json exists.
    # Useful for iterating on cluster-merge / GK / rendering logic
    # without re-paying for the LLM round-trip.
    cached_votes_path = output_dir / "jersey_votes.json"
    cached_scene_path = output_dir / "scene.json"
    use_cached = (
        bool(config.get("reuse_cached_votes"))
        and cached_votes_path.exists()
    )
    if use_cached:
        print(
            f"  Reusing cached jersey_votes.json + scene.json from "
            f"{output_dir} (skipping all LLM calls).",
            flush=True,
        )

    # --- Sample crops (skip when reusing cached votes — sampler is the
    # second-most expensive step after the LLM) ----------------------
    sampler_cfg = config.get("sampler", {})
    k = int(config.get("frames_per_track", 15))
    sampled: dict = {}
    if not use_cached:
        print(f"  Sampling {k} crops × {len(candidates)} tracks...", flush=True)
        t0 = time.time()
        sampled = sample_crops_for_tracks(
            video_path=video_path,
            tracks_by_frame=tracks_by_frame,
            track_ids=candidates,
            k=k,
            min_bbox_height=int(sampler_cfg.get("min_bbox_height", 80)),
            upper_ratio=float(sampler_cfg.get("upper_body_ratio", 0.65)),
        )
        print(f"  Sampling done in {time.time() - t0:.1f}s", flush=True)

    # --- Jersey recognition via Claude -------------------------------
    jersey_cfg = config.get("jersey", {})
    recognizer = _build_recognizer(jersey_cfg) if not use_cached else None
    votes: dict[int, dict[str, Any]] = {}
    concurrency = int(jersey_cfg.get("max_concurrency", 8))

    # --- Step 1: Global scene understanding (skip when reusing cache) -
    if use_cached:
        scene = json.loads(cached_scene_path.read_text()) \
            if cached_scene_path.exists() else {}
        print("  Step 1 (cached): "
              f"team_A={scene.get('team_A_kit', '-')} | "
              f"team_B={scene.get('team_B_kit', '-')}",
              flush=True)
    else:
        wide_frame, wide_fid = _pick_wide_frame(
            video_path=video_path,
            tracks_by_frame=tracks_by_frame,
        )
        n_scene_persons = int(jersey_cfg.get("scene_person_samples", 20))
        scene_person_crops = _pick_scene_person_crops(
            sampled=sampled,
            positions_by_track=positions_by_track,
            pitch_length=pitch_length,
            pitch_width=pitch_width,
            n_total=n_scene_persons,
        )
        print(f"  Step 1 (scene understanding): wide frame @ fid={wide_fid} "
              f"+ {len(scene_person_crops)} person samples...", flush=True)
        t0 = time.time()
        scene = recognizer.describe_scene(wide_frame, scene_person_crops) \
            if wide_frame is not None else {}
        print(f"  Step 1 done in {time.time() - t0:.1f}s", flush=True)
        if scene:
            with open(output_dir / "scene.json", "w") as _f:
                json.dump(scene, _f, indent=2)
            print("    team_A_kit: " + (scene.get("team_A_kit") or "-"),
                  flush=True)
            print("    team_B_kit: " + (scene.get("team_B_kit") or "-"),
                  flush=True)

    # Step 1 only commits team_A / team_B exemplars — referees,
    # goalkeepers, linesmen are decided in Step 2 from movement.
    for tid in scene.get("team_A_ref_ids", []):
        team_by_track[tid] = "team_A"
    for tid in scene.get("team_B_ref_ids", []):
        team_by_track[tid] = "team_B"

    if not use_cached:
        n_exemplars = int(jersey_cfg.get("team_exemplars_per_team", 5))
        team_exemplars = _pick_team_exemplars_from_refs(
            sampled=sampled,
            ref_tids_by_team={
                "team_A": scene.get("team_A_ref_ids", []),
                "team_B": scene.get("team_B_ref_ids", []),
            },
            n=n_exemplars,
        )
        if not team_exemplars.get("team_A") or not team_exemplars.get("team_B"):
            logger.warning(
                "Step 1 returned insufficient refs; falling back to "
                "frame-count-based exemplars.")
            team_exemplars = _pick_team_exemplars(
                sampled=sampled,
                team_by_track=team_by_track,
                frame_count_by_track=frame_count_by_track,
                n=n_exemplars,
            )
        logger.info(
            "Team kit exemplars: team_A=%d, team_B=%d",
            len(team_exemplars.get("team_A", [])),
            len(team_exemplars.get("team_B", [])),
        )

        def _call(tid: int) -> tuple[int, tuple]:
            frames = sampled.get(tid, [])
            if not frames:
                return tid, (None, 0.0, "no crops", "unknown", "unknown")
            crops = [f.crop for f in frames]
            pos_lab = position_label_by_track.get(tid, "unknown")
            mov_desc = movement_by_track.get(tid, "")
            return tid, recognizer.recognize_multi(
                crops,
                position_label=pos_lab,
                team_exemplars=team_exemplars,
                scene_description=scene,
                movement_description=mov_desc,
            )

        backend_name = jersey_cfg.get("backend", "claude")
        print(f"  Jersey recognition: {len(candidates)} tracks × {k} crops "
              f"(backend={backend_name}, concurrency={concurrency})...",
              flush=True)
        t0 = time.time()
        with ThreadPoolExecutor(max_workers=concurrency) as pool:
            futs = [pool.submit(_call, tid) for tid in candidates]
            for i, fut in enumerate(as_completed(futs), 1):
                try:
                    tid, (num, conf, reason, role, team) = fut.result()
                except Exception as exc:  # noqa: BLE001
                    logger.warning("recognize_multi failed: %s", exc)
                    continue
                votes[tid] = {
                    "jersey_number": num,
                    "confidence": conf,
                    "reasoning": reason,
                    "role": role,
                    "team": team,
                    "position_label": position_label_by_track.get(tid, "unknown"),
                    "n_crops": len(sampled.get(tid, [])),
                }
                if team in ("team_A", "team_B") and tid not in team_by_track:
                    team_by_track[tid] = team
                if i % 20 == 0 or i == len(candidates):
                    elapsed = time.time() - t0
                    rate = i / elapsed if elapsed > 0 else 0
                    print(f"    [{i}/{len(candidates)}] {rate:.1f} req/s",
                          flush=True)
        print(f"  Jersey recognition done in {time.time() - t0:.1f}s",
              flush=True)

        with open(output_dir / "jersey_votes.json", "w") as f:
            json.dump({str(tid): v for tid, v in votes.items()}, f, indent=2)
    else:
        cached = json.loads(cached_votes_path.read_text())
        for k_str, v in cached.items():
            try:
                votes[int(k_str)] = v
            except (TypeError, ValueError):
                continue
            t_lab = v.get("team")
            if t_lab in ("team_A", "team_B"):
                team_by_track.setdefault(int(k_str), t_lab)
        print(f"  Reused {len(votes)} cached votes.", flush=True)

    # --- Build metas + consolidate -----------------------------------
    # Route tracks by Step 2's role verdict:
    #   - coach / other          → dropped (no cluster written)
    #   - referee                → referee pool (merged as REF-01)
    #   - linesman               → linesman pool (merged as LIN-01)
    #   - goalkeeper             → enters player pool with role=goalkeeper
    #                              (later refined by assign_goalkeepers)
    #   - player / unknown       → normal player consolidation
    dropped_roles = {"coach", "other"}
    dropped_tracks: dict[int, str] = {}
    referee_tracks: list[int] = []
    linesman_tracks: list[int] = []
    gk_voted_tracks: set[int] = set()
    metas: list[TrackMeta] = []
    for tid in candidates:
        v = votes.get(tid, {})
        claude_role = (v.get("role") or "unknown").lower()
        if claude_role in dropped_roles:
            dropped_tracks[tid] = claude_role
            continue
        if claude_role == "referee":
            referee_tracks.append(tid)
            continue
        if claude_role == "linesman":
            linesman_tracks.append(tid)
            continue
        # player / goalkeeper / unknown → enter the player consolidation.
        # Goalkeepers carry team='unknown' from Step 2 (per prompt rules);
        # assign_goalkeepers will fix the team from goal-side later.
        emb_list = track_features.get(str(tid)) or []
        centroid = compute_centroid(emb_list)
        num = v.get("jersey_number")
        conf = float(v.get("confidence", 0.0) or 0.0)
        meta_role = "goalkeeper" if claude_role == "goalkeeper" else \
            role_by_track.get(tid, "player")
        if claude_role == "goalkeeper":
            gk_voted_tracks.add(tid)
        team_label = team_by_track.get(tid, "unknown")
        # When Step 2 returned goalkeeper without a team, split by goal
        # side so the two GKs (one per goal) don't collapse if they
        # happen to share a jersey number reading.
        if claude_role == "goalkeeper" and team_label == "unknown":
            poss = positions_by_track.get(tid, [])
            if poss:
                med_x = float(np.median(np.asarray(
                    poss, dtype=np.float32), axis=0)[0])
                team_label = "gk_left" if med_x < 0 else "gk_right"
        metas.append(TrackMeta(
            track_id=tid,
            team=team_label,
            role=meta_role,
            frame_count=frame_count_by_track[tid],
            jersey_number=num if isinstance(num, int) else None,
            jersey_confidence=conf,
            reid_centroid=centroid,
        ))

    if dropped_tracks or referee_tracks or linesman_tracks:
        role_counts: dict[str, int] = {}
        for r in dropped_tracks.values():
            role_counts[r] = role_counts.get(r, 0) + 1
        role_counts["referee"] = len(referee_tracks)
        role_counts["linesman"] = len(linesman_tracks)
        print(
            "  Claude role breakdown: "
            + ", ".join(f"{r}={n}" for r, n in sorted(role_counts.items())
                        if n > 0),
            flush=True,
        )

    reid_cfg = config.get("reid", {})
    track_to_player, clusters = consolidate(
        metas,
        same_person_threshold=float(reid_cfg.get("same_person_threshold", 0.55)),
        orphan_absorb_threshold=float(reid_cfg.get("orphan_absorb_threshold", 0.65)),
        min_jersey_confidence=float(jersey_cfg.get("min_confidence", 0.6)),
    )
    logger.info("Consolidated %d tracks → %d players", len(metas), len(clusters))
    print(f"  Consolidated {len(metas)} tracks → {len(clusters)} players "
          f"({sum(1 for c in clusters if c.jersey_number is not None)} "
          f"with jersey numbers)", flush=True)

    # Add referee + linesman clusters (one cluster per role; ReID is not
    # reliable enough to separate individual officials).
    ref_clusters, ref_map = _build_officiating_clusters(
        referee_tracks, linesman_tracks,
        track_features=track_features,
    )
    clusters.extend(ref_clusters)
    track_to_player.update(ref_map)
    if ref_clusters:
        print(
            "  Officiating clusters: " +
            ", ".join(f"{c.player_id}({len(c.source_tracks)})"
                      for c in ref_clusters),
            flush=True,
        )

    # --- Prune lonely unknown clusters -------------------------------
    # An ``unknown-unk-NN`` cluster with only one source track is a
    # track the LLM could not identify (truncated reply, blurry crops,
    # or a "jumping ghost" bbox that drifts across the frame edge).
    # These are by definition unreliable: no kit, no team, no role,
    # and they couldn't be absorbed into any other cluster either.
    # Drop them so they don't pollute the rendered video / downstream
    # consumers with a labelled bbox that doesn't correspond to a
    # confident identity.
    if config.get("drop_lonely_unknown", True):
        keep: list = []
        dropped_lonely = 0
        for c in clusters:
            if (
                c.team == "unknown"
                and c.role not in ("goalkeeper", "referee", "linesman")
                and len(c.source_tracks) <= 1
            ):
                # Detach the orphan track so it isn't rendered.
                for tid in c.source_tracks:
                    track_to_player.pop(tid, None)
                dropped_lonely += 1
                continue
            keep.append(c)
        if dropped_lonely:
            print(
                f"  Dropped {dropped_lonely} lonely unknown clusters "
                "(1-source, unidentifiable).",
                flush=True,
            )
        clusters = keep

    # --- Goalkeeper identification at cluster level ------------------
    gk_cfg = config.get("goalkeeper", {})
    if gk_cfg.get("enabled", True):
        pitch_length, pitch_width = _pitch_dims(pipeline_output_dir)
        renames = assign_goalkeepers(
            clusters,
            track_positions=positions_by_track,
            pitch_length=pitch_length,
            pitch_width=pitch_width,
            goal_area_depth=float(gk_cfg.get("goal_area_depth", 16.5)),
        )
        # Drop the synthetic gk_left / gk_right team labels that
        # didn't get chosen as the canonical GK — they're not real
        # team affiliations and would pollute downstream consumers.
        for c in clusters:
            if c.team in ("gk_left", "gk_right"):
                c.team = "unknown"
        if renames:
            rename_map = dict(renames)
            for tid, pid in list(track_to_player.items()):
                if pid in rename_map:
                    track_to_player[tid] = rename_map[pid]
            print(
                f"  Goalkeepers identified: "
                f"{', '.join(f'{old}→{new}' for old, new in renames)}",
                flush=True,
            )

    # --- Write player map + cluster details --------------------------
    with open(output_dir / "player_map.json", "w") as f:
        json.dump({str(tid): pid for tid, pid in track_to_player.items()},
                  f, indent=2)
    with open(output_dir / "players.json", "w") as f:
        json.dump([c.to_dict() for c in clusters], f, indent=2)

    # --- Rewrite tracking outputs ------------------------------------
    out_cfg = config.get("output", {})
    if out_cfg.get("overwrite_tracks_json", True):
        _rewrite_tracking(
            tracking_dir=tracking_dir,
            track_to_player=track_to_player,
            clusters=clusters,
            backup=bool(out_cfg.get("backup_original", True)),
        )

    stats = {
        "candidate_tracks": len(candidates),
        "total_tracks": len(frame_count_by_track),
        "dropped_tracks": len(dropped_tracks),
        "dropped_roles": {
            r: sum(1 for rr in dropped_tracks.values() if rr == r)
            for r in set(dropped_tracks.values())
        },
        "clusters": len(clusters),
        "clusters_with_jersey": sum(
            1 for c in clusters if c.jersey_number is not None),
        "clusters_without_jersey": sum(
            1 for c in clusters if c.jersey_number is None),
        "goalkeepers": [c.player_id for c in clusters if c.role == "goalkeeper"],
        "referees": [c.player_id for c in clusters if c.role == "referee"],
        "linesmen": [c.player_id for c in clusters if c.role == "linesman"],
        "players_map": str(output_dir / "player_map.json"),
        "players_detail": str(output_dir / "players.json"),
    }
    with open(output_dir / "stats.json", "w") as f:
        json.dump(stats, f, indent=2)
    return stats


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------


def _load_json(path: Path) -> Any | None:
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def _pick_wide_frame(
    video_path: Any,
    tracks_by_frame: dict[str, list[dict[str, Any]]],
) -> "tuple[np.ndarray | None, int | None]":
    """Pick the frame with the most tracked people (best scene context)."""
    import cv2 as _cv2
    best_fid, best_n = None, 0
    for fk, ts in tracks_by_frame.items():
        if len(ts) > best_n:
            best_n = len(ts)
            try:
                best_fid = int(fk)
            except (TypeError, ValueError):
                continue
    if best_fid is None:
        return None, None
    cap = _cv2.VideoCapture(str(video_path))
    try:
        cap.set(_cv2.CAP_PROP_POS_FRAMES, best_fid)
        ok, frame = cap.read()
    finally:
        cap.release()
    if not ok:
        return None, None
    return frame, best_fid


def _pick_scene_person_crops(
    sampled: dict,
    positions_by_track: dict[int, list[list[float]]],
    pitch_length: float,
    pitch_width: float,
    n_total: int,
) -> list[tuple[int, "np.ndarray"]]:
    """Pick ``n_total`` person crops spread across pitch regions.

    Splits the pitch into 6 zones (left/mid/right × def/fwd) plus 2
    goal zones.  Distributes samples roughly evenly by drawing a
    largest-bbox crop from distinct tracks in each zone.
    """
    half_l = pitch_length / 2
    half_w = pitch_width / 2

    def _zone(mx: float, my: float) -> str:
        if abs(mx) > half_l + 0.5 or abs(my) > half_w + 0.5:
            return "outside"
        if mx < -half_l + 16.5:
            return "near_left_goal"
        if mx > half_l - 16.5:
            return "near_right_goal"
        # Three horizontal bands by x, two vertical (top/bottom) by y
        if mx < -half_l / 3:
            band_x = "left"
        elif mx > half_l / 3:
            band_x = "right"
        else:
            band_x = "mid"
        band_y = "top" if my > 0 else "bot"
        return f"{band_x}_{band_y}"

    zone_of: dict[int, str] = {}
    for tid, poss in positions_by_track.items():
        if poss and sampled.get(tid):
            mx, my = np.median(np.asarray(poss), axis=0)
            zone_of[tid] = _zone(float(mx), float(my))

    # Group tids by zone
    by_zone: dict[str, list[int]] = {}
    for tid, z in zone_of.items():
        by_zone.setdefault(z, []).append(tid)
    # Sort each zone's tracks by sampler-best-bbox area (largest first)
    def _best_area(tid: int) -> float:
        frames = sampled.get(tid) or []
        if not frames:
            return 0.0
        return max(f.score for f in frames)
    for z in list(by_zone):
        by_zone[z].sort(key=lambda t: -_best_area(t))

    zones = list(by_zone.keys())
    if not zones:
        return []
    quota = max(1, n_total // len(zones))
    picks: list[tuple[int, "np.ndarray"]] = []
    used = set()
    # First pass: quota per zone
    for z in zones:
        for tid in by_zone[z][:quota]:
            if tid in used:
                continue
            frames = sampled.get(tid) or []
            if not frames:
                continue
            best = max(frames, key=lambda f: f.score)
            picks.append((tid, best.crop))
            used.add(tid)
            if len(picks) >= n_total:
                return picks
    # Second pass: fill remaining from whatever zones have spare
    for z in zones:
        for tid in by_zone[z]:
            if tid in used:
                continue
            frames = sampled.get(tid) or []
            if not frames:
                continue
            best = max(frames, key=lambda f: f.score)
            picks.append((tid, best.crop))
            used.add(tid)
            if len(picks) >= n_total:
                return picks
    return picks


def _pick_team_exemplars_from_refs(
    sampled: dict,
    ref_tids_by_team: dict[str, list[int]],
    n: int,
) -> dict[str, list["np.ndarray"]]:
    """Build team exemplar crop list from Claude-verified ref tids.

    Takes each team's ref tids (typically 3 per team from Step 1),
    picks their best-scored crops, and returns up to ``n`` crops per
    team.  If a team has fewer refs than ``n``, we supplement from
    duplicate crops of the same tids to hit ``n``.
    """
    out: dict[str, list["np.ndarray"]] = {"team_A": [], "team_B": []}
    for team, tids in ref_tids_by_team.items():
        crops: list["np.ndarray"] = []
        for tid in tids:
            frames = sampled.get(tid) or []
            if not frames:
                continue
            # Take the single best crop per ref tid — avoids diluting
            # with blurry or small copies of the same person.
            best = max(frames, key=lambda f: f.score)
            crops.append(best.crop)
        if not crops:
            continue
        # Truncate or pad to n by repetition of distinct refs.
        if len(crops) >= n:
            crops = crops[:n]
        out[team] = crops
    return out


def _pick_team_exemplars(
    sampled: dict,
    team_by_track: dict[int, str],
    frame_count_by_track: dict[int, int],
    n: int,
) -> dict[str, list[np.ndarray]]:
    """Pick N exemplar crops per team for Claude's visual reference.

    Strategy: for each team take ``2*n`` candidate tracks with the
    highest frame count, then, across those candidates, pick crops
    spread through the match's time range (different frame_ids) so the
    references cover varied lighting, angles, and players.  This
    avoids a failure mode where all exemplars come from the same
    dominant player's crops in the same ~30 s window.
    """
    out: dict[str, list[np.ndarray]] = {"team_A": [], "team_B": []}
    for team in ("team_A", "team_B"):
        team_tids = [
            tid for tid in frame_count_by_track
            if team_by_track.get(tid) == team and sampled.get(tid)
        ]
        team_tids.sort(key=lambda t: -frame_count_by_track[t])
        # Candidate pool: up to 4*n longest tracks, gathering one
        # top-scored crop from each, then pick N spread by frame_id.
        pool: list[tuple[int, int, object]] = []  # (frame_id, tid, crop)
        for tid in team_tids[: max(n * 4, n)]:
            frames = sampled.get(tid) or []
            if not frames:
                continue
            best = max(frames, key=lambda f: f.score)
            pool.append((best.frame_id, tid, best.crop))
        if not pool:
            continue
        # Sort by frame_id, pick N evenly spaced indices
        pool.sort(key=lambda e: e[0])
        if len(pool) <= n:
            chosen = pool
        else:
            step = len(pool) / n
            chosen = [pool[int(i * step)] for i in range(n)]
        out[team] = [c for (_fid, _tid, c) in chosen]
    return out


def _build_officiating_clusters(
    referee_tids: list[int],
    linesman_tids: list[int],
    track_features: dict[str, list[list[float]]],
) -> tuple[list[PlayerCluster], dict[int, str]]:
    """Bundle all referee tracks into one cluster and all linesman tracks
    into individual clusters (usually 2 linesmen on a pitch).

    Strategy:
      - **Referees**: collapse all into a single cluster ``REF-01``.
        In practice there is only one main referee; if two are detected
        they'll share the ID.  OSNet cannot reliably separate them on
        Veo footage, so we don't try.
      - **Linesmen**: split into per-side clusters ``LIN-01``, ``LIN-02``
        based on median pitch y (one runs each touchline).  If we can't
        tell, lump into ``LIN-01``.
    """
    clusters: list[PlayerCluster] = []
    track_map: dict[int, str] = {}

    def _centroid_of(tids: list[int]) -> np.ndarray | None:
        embs: list[list[float]] = []
        for tid in tids:
            embs.extend(track_features.get(str(tid), []) or [])
        return compute_centroid(embs) if embs else None

    if referee_tids:
        c = PlayerCluster(
            player_id="REF-01",
            team="referee",
            role="referee",
            jersey_number=None,
            jersey_confidence=0.0,
            source_tracks=list(referee_tids),
            reid_centroid=_centroid_of(referee_tids),
        )
        clusters.append(c)
        for tid in referee_tids:
            track_map[tid] = c.player_id

    if linesman_tids:
        # Single lumped linesman cluster — per-side split would need
        # stored positions; keep it simple and merge.
        c = PlayerCluster(
            player_id="LIN-01",
            team="linesman",
            role="linesman",
            jersey_number=None,
            jersey_confidence=0.0,
            source_tracks=list(linesman_tids),
            reid_centroid=_centroid_of(linesman_tids),
        )
        clusters.append(c)
        for tid in linesman_tids:
            track_map[tid] = c.player_id

    return clusters, track_map


def _compute_position_labels(
    positions_by_track: dict[int, list[list[float]]],
    pitch_length: float,
    pitch_width: float,
) -> dict[int, str]:
    """Classify each track's median pitch position into a coarse label.

    Labels (priority order — first match wins):

    - ``outside``          : |x| > half_l + 0.5 or |y| > half_w + 0.5
    - ``near_left_goal``   : x < -half_l + goal_area_depth (16.5m)
    - ``near_right_goal``  : x >  half_l - goal_area_depth
    - ``sideline``         : |y| > 0.7 * half_w (but inside pitch)
    - ``midfield``         : everything else

    Off-field tracks keep their label (``outside``); they still enter
    the Claude pipeline so linesmen/coaches can be identified.
    """
    half_l = pitch_length / 2
    half_w = pitch_width / 2
    goal_area = 16.5
    labels: dict[int, str] = {}
    for tid, poss in positions_by_track.items():
        if not poss:
            labels[tid] = "unknown"
            continue
        x, y = [float(v) for v in np.median(
            np.asarray(poss, dtype=np.float32), axis=0)]
        if abs(x) > half_l + 0.5 or abs(y) > half_w + 0.5:
            labels[tid] = "outside"
        elif x < -half_l + goal_area:
            labels[tid] = "near_left_goal"
        elif x > half_l - goal_area:
            labels[tid] = "near_right_goal"
        elif abs(y) > 0.7 * half_w:
            labels[tid] = "sideline"
        else:
            labels[tid] = "midfield"
    return labels


def _compute_movement_descriptions(
    positions_by_track: dict[int, list[list[float]]],
    pitch_length: float,
    pitch_width: float,
) -> dict[int, str]:
    """Summarise each track's movement on the pitch in plain English so
    the VLM can use it as a strong role cue.

    Pattern taxonomy (matches the role decision table in the Step-2
    prompt):

    - ``tight_near_goal``  : motion confined to a small box right by
      a goal line — exclusive to goalkeepers.
    - ``covers_whole_pitch``: motion spans most of length AND width —
      exclusive to the main referee.
    - ``touchline_runner`` : y stays near a touchline (short y-range,
      large x-range along the line) — exclusive to a linesman.
    - ``off_pitch_lateral``: median position is off the pitch.
    - ``outfield``         : everything else (normal player).

    The returned string is a single line that the prompt drops in
    verbatim under "Movement pattern across the match".
    """
    half_l = pitch_length / 2
    half_w = pitch_width / 2
    out: dict[int, str] = {}

    # Thresholds (chosen for a 90-min match on a ~100×60m pitch)
    # Length covered must be > 60% of pitch_length to count as
    # "whole pitch"; same for width.  These are referees' typical
    # patterns; outfielders cover ~30-50% of length and ~20-40% of
    # width on each side.
    REF_LEN_FRAC = 0.6
    REF_WID_FRAC = 0.55
    GK_LEN_FRAC = 0.18   # GK rarely strays > 18% of length from goal line
    GK_WID_FRAC = 0.55   # GK can move side-to-side within penalty area
    LIN_Y_BAND = 5.0     # m: linesman y stays within 5m of a touchline

    for tid, poss in positions_by_track.items():
        if not poss:
            out[tid] = ""
            continue
        arr = np.asarray(poss, dtype=np.float32)
        # Filter calibration blowups
        ok = (np.abs(arr[:, 0]) <= pitch_length) & \
             (np.abs(arr[:, 1]) <= pitch_width)
        arr = arr[ok]
        if arr.shape[0] < 3:
            out[tid] = ""
            continue
        x_med = float(np.median(arr[:, 0]))
        y_med = float(np.median(arr[:, 1]))
        # Use 5th/95th percentiles to ignore outliers from track noise
        x_lo, x_hi = np.percentile(arr[:, 0], [5, 95])
        y_lo, y_hi = np.percentile(arr[:, 1], [5, 95])
        x_range = float(x_hi - x_lo)
        y_range = float(y_hi - y_lo)
        len_frac = x_range / pitch_length
        wid_frac = y_range / pitch_width

        # Distance from median to nearest goal line (left or right).
        d_left_goal = abs(x_med - (-half_l))
        d_right_goal = abs(x_med - half_l)
        d_goal = min(d_left_goal, d_right_goal)
        nearest_side = "left" if d_left_goal <= d_right_goal else "right"
        d_top_touch = abs(y_med - half_w)
        d_bot_touch = abs(y_med + half_w)
        d_touch = min(d_top_touch, d_bot_touch)
        nearest_touch = "top" if d_top_touch <= d_bot_touch else "bottom"

        on_pitch = (abs(x_med) <= half_l + 0.5
                    and abs(y_med) <= half_w + 0.5)

        # ----- Classify -----
        # Goalkeeper: very tight to one goal line, length range small.
        if (on_pitch and d_goal <= 18.0
                and len_frac <= GK_LEN_FRAC):
            pattern = "tight_near_goal"
            descr = (
                f"tight_near_goal: stays within ~{x_range:.1f}m of the "
                f"{nearest_side} goal line ({d_goal:.1f}m from goal). "
                "This is the goalkeeper pattern."
            )
        # Main referee: covers most of length AND most of width.
        elif (on_pitch and len_frac >= REF_LEN_FRAC
              and wid_frac >= REF_WID_FRAC):
            pattern = "covers_whole_pitch"
            descr = (
                f"covers_whole_pitch: x-range {x_range:.1f}m "
                f"({len_frac*100:.0f}% of pitch length), y-range "
                f"{y_range:.1f}m ({wid_frac*100:.0f}% of pitch width). "
                "Only the main referee covers this much area."
            )
        # Linesman: runs along ONE touchline.
        elif (on_pitch and d_touch <= LIN_Y_BAND
              and y_range <= 12.0
              and len_frac >= 0.35):
            pattern = "touchline_runner"
            descr = (
                f"touchline_runner: stays within {y_range:.1f}m of "
                f"the {nearest_touch} touchline (median {d_touch:.1f}m "
                f"from it), runs {x_range:.1f}m along it. "
                "This is the linesman pattern."
            )
        elif not on_pitch:
            pattern = "off_pitch_lateral"
            descr = (
                f"off_pitch_lateral: median position is off the playing "
                f"area (x={x_med:.1f}, y={y_med:.1f}). "
                "Coach / substitute / staff."
            )
        else:
            pattern = "outfield"
            descr = (
                f"outfield: x-range {x_range:.1f}m "
                f"({len_frac*100:.0f}% of length), y-range "
                f"{y_range:.1f}m ({wid_frac*100:.0f}% of width). "
                "Normal outfield player movement."
            )
        out[tid] = f"pattern={pattern}. {descr}"
    return out


def _drop_off_field(
    tracks_by_frame: dict[str, list[dict[str, Any]]],
    positions_by_track: dict[int, list[list[float]]],
    pitch_length: float,
    pitch_width: float,
    margin: float,
) -> dict[str, list[dict[str, Any]]]:
    """Drop tracks whose median pitch position sits outside the playing
    area (substitutes warming up, sideline staff).  Returns a filtered
    copy of tracks_by_frame.
    """
    half_l = pitch_length / 2
    half_w = pitch_width / 2
    mean_pos = {
        tid: np.median(np.asarray(poss, dtype=np.float32), axis=0)
        for tid, poss in positions_by_track.items() if poss
    }
    off_field = {
        tid for tid, pos in mean_pos.items()
        if abs(pos[0]) > half_l + margin or abs(pos[1]) > half_w + margin
    }
    if off_field:
        logger.info("  Off-field tracks removed: %s", sorted(off_field))
        print(f"  Off-field filter: dropped {len(off_field)} tracks",
              flush=True)
    return {
        fk: [t for t in ts if int(t["track_id"]) not in off_field]
        for fk, ts in tracks_by_frame.items()
    }


def _team_counts(team_by_track: dict[int, str]) -> dict[str, int]:
    out: dict[str, int] = {}
    for team in team_by_track.values():
        out[team] = out.get(team, 0) + 1
    return out


def _classify_teams_kmeans(
    track_features: dict[str, list[list[float]]],
    positions_by_track: dict[int, list[list[float]]],
    frame_count_by_track: dict[int, int],
    config: dict[str, Any],
) -> dict[int, str]:
    """Run the existing KMeans team classifier on per-track ReID centroids.

    Uses median pitch position per track as optional position feature.
    Returns ``{track_id: "team_A"|"team_B"|"referee"|"unknown"}``.
    """
    from ..tracking.team.kmeans_classifier import KMeansTeamClassifier

    tc_cfg = config.get("team_classification", {})
    classifier = KMeansTeamClassifier(tc_cfg)

    # Average ReID features per track — tracker writes each track's
    # embeddings as a list; take the L2-normalised mean as its identity.
    mean_features: dict[int, np.ndarray] = {}
    for tid_str, embs in track_features.items():
        if not embs:
            continue
        try:
            tid = int(tid_str)
        except (TypeError, ValueError):
            continue
        arr = np.asarray(embs, dtype=np.float32)
        if arr.ndim == 1:
            arr = arr[None, :]
        mean_features[tid] = arr.mean(axis=0)

    if not mean_features:
        logger.warning("No ReID features available — skipping team classification")
        return {}

    # Median pitch position per track (positions optional input)
    mean_positions = {
        tid: list(np.median(np.asarray(poss, dtype=np.float32), axis=0))
        for tid, poss in positions_by_track.items() if poss
    }
    bbox_heights = None  # tracker no longer surfaces this; KMeans handles absence
    frame_counts = dict(frame_count_by_track)

    assignments = classifier.fit(
        track_features=mean_features,
        track_positions=mean_positions,
        track_bbox_heights=bbox_heights,
        track_frame_counts=frame_counts,
    )
    return assignments


def _pitch_dims(pipeline_output_dir: Path) -> tuple[float, float]:
    """Read pitch dimensions from calibration metadata (defaults to FIFA)."""
    meta_path = pipeline_output_dir / "field_registration" / "calibration_metadata.json"
    if not meta_path.exists():
        meta_path = pipeline_output_dir / "stage1" / "calibration_metadata.json"
    if meta_path.exists():
        try:
            meta = json.loads(meta_path.read_text())
            vi = meta.get("video_info", {})
            return (
                float(vi.get("pitch_length", 105.0)),
                float(vi.get("pitch_width", 68.0)),
            )
        except (json.JSONDecodeError, KeyError, ValueError):
            pass
    return (105.0, 68.0)


def _build_recognizer(jersey_cfg: dict[str, Any]):
    backend = jersey_cfg.get("backend", "claude").lower()
    if backend == "claude":
        from ..jersey.claude_recognizer import ClaudeJerseyRecognizer
        return ClaudeJerseyRecognizer(config=jersey_cfg)
    if backend == "gemini":
        from ..jersey.gemini_recognizer import GeminiJerseyRecognizer
        return GeminiJerseyRecognizer(config=jersey_cfg)
    if backend == "qwen":
        from ..jersey.qwen_vllm_recognizer import QwenVLLMRecognizer
        return QwenVLLMRecognizer(config=jersey_cfg)
    raise ValueError(
        f"track_consolidation.jersey.backend must be one of "
        f"{{claude, gemini, qwen}}, got {backend!r}")


def _rewrite_tracking(
    tracking_dir: Path,
    track_to_player: dict[int, str],
    clusters: list[PlayerCluster],
    backup: bool,
) -> None:
    """Rewrite tracks.json and team_assignments.json with player_ids.

    The rewrite is *additive*: each track dict keeps ``bbox`` / ``pitch_position`` /
    ``team`` / ``role`` / ``confidence`` and gains a new ``track_id`` equal to
    the player_id string, plus updated ``jersey_number`` / ``jersey_confidence``.
    The original integer track_id is preserved under ``orig_track_id``.
    """
    tracks_path = tracking_dir / "tracks.json"
    team_path = tracking_dir / "team_assignments.json"

    if backup:
        bak = tracking_dir / "tracks.pre_consolidation.json"
        if not bak.exists():
            shutil.copy2(tracks_path, bak)
        bak2 = tracking_dir / "team_assignments.pre_consolidation.json"
        if team_path.exists() and not bak2.exists():
            shutil.copy2(team_path, bak2)

    by_player: dict[str, PlayerCluster] = {c.player_id: c for c in clusters}

    # tracks.json
    tracks_by_frame = _load_json(tracks_path) or {}
    for fk, tracks in tracks_by_frame.items():
        kept: list[dict[str, Any]] = []
        for t in tracks:
            tid = int(t["track_id"])
            pid = track_to_player.get(tid)
            if pid is None:
                # Unmapped track (short or Claude-rejected). Keep with
                # role=other + a synthetic id so rendering can show a
                # neutral box rather than a hole — real player attribution
                # still requires a real cluster.
                t["orig_track_id"] = tid
                t["track_id"] = f"unmapped-{tid}"
                t["role"] = "other"
                t["team"] = "unknown"
                kept.append(t)
                continue
            cluster = by_player.get(pid)
            t["orig_track_id"] = tid
            t["track_id"] = pid
            if cluster and cluster.jersey_number is not None:
                t["jersey_number"] = cluster.jersey_number
                t["jersey_confidence"] = round(cluster.jersey_confidence, 3)
            if cluster:
                t["role"] = cluster.role
                t["team"] = cluster.team
            kept.append(t)
        tracks_by_frame[fk] = kept
    with open(tracks_path, "w") as f:
        json.dump(tracks_by_frame, f)

    # team_assignments.json — key by player_id
    player_teams = {c.player_id: c.team for c in clusters}
    with open(team_path, "w") as f:
        json.dump(player_teams, f, indent=2)
