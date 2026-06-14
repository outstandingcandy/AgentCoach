"""Track consolidation runner — ties sampler, Claude recognizer, and
consolidator together, then rewrites tracks.json / team_assignments.json
with stable player_ids.
"""

from __future__ import annotations

import json
import logging
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import cv2
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

    # Per-track frame counts come from the *processed* sample stream — at
    # process_fps=5 a track that lasts 4 s contributes 20 observations,
    # not 40. The frame-count thresholds below were tuned at process_fps=10;
    # rescale by effective_fps/10 so they keep their intended duration.
    _ref_fps = 10.0
    _process_fps = config.get("_process_fps")
    _fps_scale = (float(_process_fps) / _ref_fps) if _process_fps else 1.0

    # --- Load inputs --------------------------------------------------
    tracks_by_frame = _load_json(tracking_dir / "tracks.json")
    track_features = _load_json(tracking_dir / "track_features.json") or {}

    # --- Collect per-track frame counts + positions ------------------
    # Role / team are assigned here (tracker no longer writes them).
    frame_count_by_track: dict[int, int] = Counter()
    positions_by_track: dict[int, list[list[float]]] = {}
    cooccur_pairs: set[tuple[int, int]] = set()  # tids ever in same frame
    for fk, tracks in tracks_by_frame.items():
        tids_here: list[int] = []
        for t in tracks:
            tid = int(t["track_id"])
            frame_count_by_track[tid] += 1
            tids_here.append(tid)
            pp = t.get("pitch_position")
            if pp is not None:
                positions_by_track.setdefault(tid, []).append(pp)
        # Any pair of tids visible in the same frame can't be the same
        # person — record them so consolidate() refuses to merge them.
        for i, a in enumerate(tids_here):
            for b in tids_here[i + 1:]:
                if a == b:
                    continue
                lo, hi = (a, b) if a < b else (b, a)
                cooccur_pairs.add((lo, hi))

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
    #
    # Short tracks (< min_frames_for_pattern) get an empty description.
    # Their x/y range is unreliably small just because we have few
    # samples — labelling a 20-frame yellow-jersey fragment as
    # tight_near_goal lets the prompt's "exclusive to goalkeeper" rule
    # override the visual evidence and route real outfielders to A-GK.
    movement_by_track = _compute_movement_descriptions(
        positions_by_track,
        pitch_length=pitch_length,
        pitch_width=pitch_width,
        min_frames_for_pattern=int(config.get(
            "min_frames_for_movement_pattern",
            max(1, round(40 * _fps_scale)))),
        frame_count_by_track=frame_count_by_track,
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

    # Pre-filter: drop tracks that spent most of their lifetime
    # outside the playing area (substitutes / sideline staff / fans
    # detected by YOLO whose pitch projection consistently lands beyond
    # the touchline). Without this, the candidate set is polluted with
    # off-pitch person detections that survive size/aspect filters but
    # represent neither players nor officials. The filter requires both
    # a minimum sample count and a high outside-fraction so brief
    # excursions over the touchline (throw-ins, corner kicks) don't
    # remove real players.
    off_cfg = config.get("off_field_filter", {}) or {}
    if off_cfg.get("enabled", True):
        off_field_tids = _detect_off_field_tracks(
            positions_by_track,
            pitch_length=pitch_length,
            pitch_width=pitch_width,
            margin=float(off_cfg.get("margin_m", 1.5)),
            min_outside_fraction=float(off_cfg.get("min_outside_fraction", 0.85)),
            min_observations=int(off_cfg.get(
                "min_observations", max(1, round(10 * _fps_scale)))),
        )
        if off_field_tids:
            logger.info(
                "Off-field tracks dropped (%d): %s",
                len(off_field_tids), sorted(off_field_tids),
            )
            print(
                f"  Off-field filter: dropped {len(off_field_tids)} tracks "
                f"({sorted(off_field_tids)})",
                flush=True,
            )
            tracks_by_frame = _filter_tracks_by_id(tracks_by_frame, off_field_tids)
            for tid in off_field_tids:
                frame_count_by_track.pop(tid, None)
                positions_by_track.pop(tid, None)
                position_label_by_track.pop(tid, None)
                movement_by_track.pop(tid, None)
                role_by_track.pop(tid, None)

    min_frames = int(config.get(
        "min_track_frames", max(1, round(30 * _fps_scale))))
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

    # --- Sample crops -----------------------------------------------
    # Always re-sample. The LLM cache used to live at
    # ``<stage>/jersey_votes.json`` and was reused across re-runs to
    # save Claude tokens, but it had no input fingerprint — when the
    # tracker re-ran with the same integer tids meaning different
    # physical people, the cache silently served stale answers.
    # Better-cheap-than-stale: pay the LLM cost every time.
    #
    # ``frames_per_track``: positive int → top-K sampling (legacy);
    # 0 or null → take EVERY frame in the track (LLM sees all
    # available bboxes), capped at ``max_crops_per_track``. Per-crop
    # OCR runs in parallel inside ``recognize_multi`` so 100s of crops
    # is tractable.
    sampler_cfg = config.get("sampler", {})
    k_raw = config.get("frames_per_track", 0)
    k = int(k_raw) if k_raw not in (None, "", "all", "none") else None
    if k is not None and k <= 0:
        k = None
    max_crops = int(config.get("max_crops_per_track", 200))
    # Stride sampling: take every Nth frame in temporal order. Cheaper
    # than all-frames with similar voting confidence — N=10 over a
    # 600-frame track gives ~60 OCR calls instead of 600. Defaults to
    # 10 when ocr_backend=rapidocr (no API rate limit, but each call
    # still costs ~50 ms on CPU). Setting ``frame_stride`` explicitly
    # overrides regardless of backend.
    jersey_cfg_pre = config.get("jersey", {}) or {}
    stride_raw = config.get("frame_stride")
    if stride_raw in (None, "", 0):
        if str(jersey_cfg_pre.get("ocr_backend", "llm")).lower() == "rapidocr":
            stride: int | None = 10
        else:
            stride = None
    else:
        stride = int(stride_raw)
        if stride <= 1:
            stride = None
    if stride:
        label = f"every {stride}th frame"
    else:
        label = (
            f"all crops (cap {max_crops})" if k is None else f"{k} crops"
        )
    print(f"  Sampling {label} × {len(candidates)} tracks...", flush=True)
    t0 = time.time()
    sampled = sample_crops_for_tracks(
        video_path=video_path,
        tracks_by_frame=tracks_by_frame,
        track_ids=candidates,
        k=k,
        min_bbox_height=int(sampler_cfg.get("min_bbox_height", 80)),
        upper_ratio=float(sampler_cfg.get("upper_body_ratio", 0.65)),
        max_cap=max_crops,
        stride=stride,
    )
    n_total_crops = sum(len(v) for v in sampled.values())
    print(f"  Sampling done in {time.time() - t0:.1f}s "
          f"({n_total_crops} total crops)", flush=True)

    # --- Jersey recognition via Claude -------------------------------
    jersey_cfg = config.get("jersey", {})
    recognizer = _build_recognizer(jersey_cfg)
    votes: dict[int, dict[str, Any]] = {}
    concurrency = int(jersey_cfg.get("max_concurrency", 8))

    # --- Step 1: Global scene understanding -------------------------
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

    # Team exemplars feed the LLM as reference images alongside each
    # target track. Picked once up front so the dump path below sees the
    # same set the LLM saw — including when votes are fully cached and
    # ``_do_jersey_recognition`` never runs.
    n_exemplars = int(jersey_cfg.get("team_exemplars_per_team", 5))
    team_exemplars: dict[str, list[np.ndarray]] = {}
    if sampled:
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

    # LLM call helper.
    def _do_jersey_recognition(target_tids: list[int]) -> None:
        def _call(tid: int) -> tuple[int, tuple]:
            frames = sampled.get(tid, [])
            if not frames:
                return tid, (None, 0.0, "no crops", "unknown", "unknown", [])
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
        print(f"  Jersey recognition: {len(target_tids)} tracks × {k} crops "
              f"(backend={backend_name}, concurrency={concurrency})...",
              flush=True)
        t0 = time.time()
        with ThreadPoolExecutor(max_workers=concurrency) as pool:
            futs = [pool.submit(_call, tid) for tid in target_tids]
            for i, fut in enumerate(as_completed(futs), 1):
                try:
                    tid, result = fut.result()
                except Exception as exc:  # noqa: BLE001
                    logger.warning("recognize_multi failed: %s", exc)
                    continue
                # Shape: (num, conf, reason, role, team, breakdown,
                #         per_crop_verdicts). Tolerate legacy shorter
                #         tuples from older code paths.
                if len(result) == 7:
                    num, conf, reason, role, team, breakdown, per_crop = result
                elif len(result) == 6:
                    num, conf, reason, role, team, breakdown = result
                    per_crop = []
                else:
                    num, conf, reason, role, team = result
                    breakdown = [(num, 1.0)] if num is not None else []
                    per_crop = []
                votes[tid] = {
                    "jersey_number": num,
                    "confidence": conf,
                    "reasoning": reason,
                    "role": role,
                    "team": team,
                    "position_label": position_label_by_track.get(tid, "unknown"),
                    "n_crops": len(sampled.get(tid, [])),
                    # Per-crop voting breakdown (number, weighted_score).
                    # The consolidator can inspect runner-up candidates
                    # to detect tracks that mix multiple physical
                    # players (e.g. a tracker ID switch that left two
                    # competing jersey readings inside one tid).
                    "jersey_candidates": [
                        [int(n), round(float(s), 3)] for n, s in breakdown
                    ],
                    # Per-crop OCR verdicts in the same order as
                    # sampled[tid]. Each entry: {reading, crop_confidence,
                    # visible_digits}. Surfaced so the /pipeline UI can
                    # show what every crop contributed (or didn't).
                    "per_crop_verdicts": per_crop,
                }
                if team in ("team_A", "team_B") and tid not in team_by_track:
                    team_by_track[tid] = team
                if i % 20 == 0 or i == len(target_tids):
                    elapsed = time.time() - t0
                    rate = i / elapsed if elapsed > 0 else 0
                    print(f"    [{i}/{len(target_tids)}] {rate:.1f} req/s",
                          flush=True)
        print(f"  Jersey recognition done in {time.time() - t0:.1f}s",
              flush=True)

    _do_jersey_recognition(candidates)
    # Persist for inspection / the /pipeline page — NOT used as input
    # cache on re-runs.
    with open(output_dir / "jersey_votes.json", "w") as f:
        json.dump({str(tid): v for tid, v in votes.items()}, f, indent=2)

    # --- Dump the exact crops + context that fed the LLM -------------
    # The /pipeline page reads this to surface what the model actually
    # saw. Cheap (sampled is already in memory).
    if sampled:
        _dump_llm_inputs(
            out_dir=output_dir / "llm_inputs",
            sampled=sampled,
            team_exemplars=team_exemplars,
            scene=scene,
            votes=votes,
            position_label_by_track=position_label_by_track,
            movement_by_track=movement_by_track,
            frame_count_by_track=frame_count_by_track,
        )

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
    # Sanity-check referees/linesmen after the LLM votes. The LLM's
    # vision on small/distant crops can mis-read e.g. a coach in a
    # white hoodie as a "white-uniform assistant referee" (orig 46
    # case at sideline (10.7, 18.9), 75 frames, completely static).
    # Real referees move across the pitch; a track that is short AND
    # static AND on the sideline cannot be the main ref. Demote those
    # to 'other' before they reach _build_officiating_clusters.
    half_l_for_ref = pitch_length / 2
    half_w_for_ref = pitch_width / 2
    def _looks_like_real_ref(tid: int) -> bool:
        poss = positions_by_track.get(tid) or []
        if len(poss) < 30:
            return False  # too short to demonstrate movement
        arr = np.asarray(poss, dtype=np.float32)
        x_lo, x_hi = np.percentile(arr[:, 0], [5, 95])
        y_lo, y_hi = np.percentile(arr[:, 1], [5, 95])
        x_range = x_hi - x_lo
        y_range = y_hi - y_lo
        y_med = np.median(arr[:, 1])
        # Reject static sideline tracks: nearly motionless AND median y
        # in the touchline band. Real refs may produce short fragments
        # mid-pitch with limited per-fragment range, but they don't sit
        # on the sideline for long stretches without moving.
        if x_range < 3.0 and y_range < 3.0 and abs(y_med) > 0.7 * half_w_for_ref:
            return False
        return True
    for tid in candidates:
        v = votes.get(tid, {})
        claude_role = (v.get("role") or "unknown").lower()
        if claude_role in dropped_roles:
            dropped_tracks[tid] = claude_role
            continue
        if claude_role in ("referee", "linesman"):
            if not _looks_like_real_ref(tid):
                # Visual mistake — demote.
                dropped_tracks[tid] = "other"
                continue
            if claude_role == "referee":
                referee_tracks.append(tid)
            else:
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
        # Surface the per-crop voting breakdown so the consolidator
        # can inspect runner-up readings (split-vote tracks indicate
        # tracker ID switches).
        cands_raw = v.get("jersey_candidates") or []
        cands: list[tuple[int, float]] = []
        for entry in cands_raw:
            try:
                if isinstance(entry, (list, tuple)) and len(entry) >= 2:
                    n = int(entry[0])
                    s = float(entry[1])
                    if 1 <= n <= 99:
                        cands.append((n, s))
            except (TypeError, ValueError):
                continue
        metas.append(TrackMeta(
            track_id=tid,
            team=team_label,
            role=meta_role,
            frame_count=frame_count_by_track[tid],
            jersey_number=num if isinstance(num, int) else None,
            jersey_confidence=conf,
            reid_centroid=centroid,
            jersey_candidates=cands,
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
        # ReID-first clustering: cosine ≥ same_person_threshold + no
        # cooccur is the primary identity gate. 0.9 reflects the
        # PRTReID same-person centroid distribution (≈0.93–0.96) on
        # Veo kids footage with margin for noisier short tracks.
        same_person_threshold=float(reid_cfg.get("same_person_threshold", 0.9)),
        orphan_absorb_threshold=float(reid_cfg.get("orphan_absorb_threshold", 0.92)),
        # Same-(team, jersey) clusters get a second-pass merge at this
        # lower cosine — the jersey number is an independent identity
        # signal, so ReID 0.85 + matching #20 is much stronger evidence
        # of "same player" than ReID 0.85 alone.
        jersey_aware_merge_threshold=float(
            reid_cfg.get("jersey_aware_merge_threshold", 0.8)),
        min_jersey_confidence=float(jersey_cfg.get("min_confidence", 0.6)),
        cooccur_pairs=cooccur_pairs,
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
        cooccur_pairs=cooccur_pairs,
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
        # Snapshot pre-rename pids so we can identify which cluster
        # objects were chosen as GKs (rename targets) regardless of
        # name collisions.
        pre_rename_pids = {id(c): c.player_id for c in clusters}
        renames = assign_goalkeepers(
            clusters,
            track_positions=positions_by_track,
            pitch_length=pitch_length,
            pitch_width=pitch_width,
            goal_area_depth=float(gk_cfg.get("goal_area_depth", 16.5)),
        )
        chosen_gk_cluster_ids = {
            id(c) for c in clusters
            if pre_rename_pids.get(id(c)) != c.player_id
            and c.role == "goalkeeper"
        }
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

        # Demote any remaining role=goalkeeper clusters that
        # ``assign_goalkeepers`` did NOT pick — Claude false-positives
        # the GK label on persistent off-field detections (goalpost,
        # banner, sideline staff) whose ReID happens to look uniform-
        # coloured. Without demotion they collide on a single
        # ``<suffix>-GK`` id (no ordinal). Identity check is on the
        # cluster object via ``id()`` so a cluster that *happened* to
        # already be named ``B-GK`` from a prior step but wasn't
        # selected by assign_goalkeepers also gets demoted.
        unk_extra: dict[str, int] = {}
        for c in clusters:
            if c.role != "goalkeeper":
                continue
            if id(c) in chosen_gk_cluster_ids:
                continue  # the real GK
            old_pid = c.player_id
            team_key = c.team if c.team in ("team_A", "team_B") else "unknown"
            unk_extra[team_key] = unk_extra.get(team_key, 0) + 1
            suffix = team_key[5:] if team_key.startswith("team_") else team_key
            new_pid = f"{suffix}-unk-gk-{unk_extra[team_key]:02d}"
            c.role = "player"  # demote: it's not really a goalkeeper
            c.player_id = new_pid
            for tid in c.source_tracks:
                if track_to_player.get(tid) == old_pid:
                    track_to_player[tid] = new_pid

        # Final dedupe: assign_goalkeepers may rename two clusters to
        # the same canonical id (e.g. both A-GK from a previous A-GK
        # collision). Append -a/-b on those.
        from collections import Counter as _Counter
        pid_counts = _Counter(c.player_id for c in clusters)
        dup_seen: dict[str, int] = {}
        for c in clusters:
            if pid_counts[c.player_id] <= 1:
                continue
            if c.role in ("referee", "linesman"):
                continue
            idx = dup_seen.get(c.player_id, 0)
            dup_seen[c.player_id] = idx + 1
            new_pid = f"{c.player_id}-{chr(ord('a') + idx)}"
            old_pid = c.player_id
            c.player_id = new_pid
            for tid in c.source_tracks:
                if track_to_player.get(tid) == old_pid:
                    track_to_player[tid] = new_pid

    # --- Write player map + cluster details --------------------------
    with open(output_dir / "player_map.json", "w") as f:
        json.dump({str(tid): pid for tid, pid in track_to_player.items()},
                  f, indent=2)
    with open(output_dir / "players.json", "w") as f:
        json.dump([c.to_dict() for c in clusters], f, indent=2)

    # --- Write consolidated tracks.json + team_assignments.json -----
    # Additive output: we write to ``track_consolidation/`` and leave
    # ``tracking/`` untouched. Downstream stages should read via
    # :func:`goalinsight.track_consolidation.load_tracks` which serves
    # the consolidated file when it exists and falls back to the raw
    # tracker output otherwise. Re-running consolidation no longer
    # destroys the source-of-truth.
    _write_consolidated_tracks(
        tracking_dir=tracking_dir,
        out_dir=output_dir,
        track_to_player=track_to_player,
        clusters=clusters,
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


def _dump_llm_inputs(
    out_dir: Path,
    sampled: dict[int, list[Any]],
    team_exemplars: dict[str, list[np.ndarray]],
    scene: dict[str, Any],
    votes: dict[int, dict[str, Any]],
    position_label_by_track: dict[int, str],
    movement_by_track: dict[int, str],
    frame_count_by_track: dict[int, int],
) -> None:
    """Persist the exact set of crops + context that fed the jersey LLM.

    Emits under ``<stage>/llm_inputs/``:
      - ``team_exemplars/team_{A,B}_<i>.jpg`` — same crops shown to the
        LLM as team-kit reference (orig sampler crop, no resize).
      - ``tracks/track_<NNNN>_f<NNNNN>.jpg`` — per source-track crops in
        the same upper-body / order the LLM consumed.
      - ``per_track.json`` — for each tid: vote (jersey/role/team/conf/
        reasoning), position_label, movement_description, n_frames seen
        in tracking, and the ordered crop filenames.
      - ``scene.json`` — copy of the scene-understanding output (kept
        here so the page has everything in one place).

    All paths are relative to ``out_dir``; consumers prepend their own
    URL prefix (``/runs_static/<run>/track_consolidation/llm_inputs/``).
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    tracks_dir = out_dir / "tracks"
    tracks_dir.mkdir(exist_ok=True)
    exemplars_dir = out_dir / "team_exemplars"
    exemplars_dir.mkdir(exist_ok=True)

    # Wipe stale artefacts so the dump always reflects the current run.
    for old in tracks_dir.glob("*.jpg"):
        old.unlink()
    for old in exemplars_dir.glob("*.jpg"):
        old.unlink()

    for team_label in ("team_A", "team_B"):
        for i, crop in enumerate(team_exemplars.get(team_label, []) or []):
            cv2.imwrite(
                str(exemplars_dir / f"{team_label}_{i:02d}.jpg"), crop)

    per_track: dict[str, Any] = {}
    for tid, frames in sampled.items():
        crop_files: list[dict[str, Any]] = []
        # Pull the per-crop OCR verdicts (same order as ``frames``) so
        # we can stamp the OCR reading + reasoning onto each thumb.
        verdicts = (votes.get(tid) or {}).get("per_crop_verdicts") or []
        for idx, sf in enumerate(frames):
            fname = f"track_{tid:04d}_f{sf.frame_id:05d}.jpg"
            cv2.imwrite(str(tracks_dir / fname), sf.crop)
            verdict = verdicts[idx] if idx < len(verdicts) else None
            crop_entry: dict[str, Any] = {
                "frame_id": int(sf.frame_id),
                "name": fname,
                "score": float(sf.score),
            }
            if verdict:
                crop_entry["ocr_reading"] = verdict.get("reading")
                crop_entry["ocr_confidence"] = round(
                    float(verdict.get("crop_confidence", 0.0) or 0.0), 3,
                )
                crop_entry["ocr_visible"] = verdict.get("visible_digits") or ""
            crop_files.append(crop_entry)
        vote = votes.get(tid) or {}
        per_track[str(tid)] = {
            "tid": int(tid),
            "vote": {
                "jersey_number": vote.get("jersey_number"),
                "confidence": vote.get("confidence"),
                "reasoning": vote.get("reasoning"),
                "role": vote.get("role"),
                "team": vote.get("team"),
            },
            "position_label": position_label_by_track.get(tid, "unknown"),
            "movement_description": movement_by_track.get(tid, ""),
            "n_frames": int(frame_count_by_track.get(tid, 0)),
            "n_crops_to_llm": len(crop_files),
            "crops": crop_files,
        }

    payload = {
        "n_tracks": len(per_track),
        "n_team_A_exemplars": len(team_exemplars.get("team_A", []) or []),
        "n_team_B_exemplars": len(team_exemplars.get("team_B", []) or []),
        "scene": scene or {},
        "tracks": per_track,
    }
    with open(out_dir / "per_track.json", "w") as f:
        json.dump(payload, f, indent=2)


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
    cooccur_pairs: set[tuple[int, int]] | None = None,
) -> tuple[list[PlayerCluster], dict[int, str]]:
    """Bundle referee tracks into co-occurrence-respecting clusters and
    linesman tracks into a single cluster.

    Strategy:
      - **Referees**: greedy cluster — each tid joins the first existing
        REF cluster that doesn't share a frame with it. Tids whose
        timeline overlaps every existing cluster start a new one
        (REF-02, REF-03, ...). On a clean clip this still ends up as a
        single REF-01; when the pool is polluted by sideline ghosts
        Claude routed via the kit-mismatch fallback rule, those pile
        into separate REFs and the real ref keeps REF-01.
      - **Linesmen**: collapse into a single cluster (typical match has
        2 but we don't try to split them; OSNet can't tell on Veo).
    """
    clusters: list[PlayerCluster] = []
    track_map: dict[int, str] = {}
    cooccur_pairs = cooccur_pairs or set()

    def _conflicts(group: list[int], tid: int) -> bool:
        for other in group:
            lo, hi = (other, tid) if other < tid else (tid, other)
            if (lo, hi) in cooccur_pairs:
                return True
        return False

    def _centroid_of(tids: list[int]) -> np.ndarray | None:
        embs: list[list[float]] = []
        for tid in tids:
            embs.extend(track_features.get(str(tid), []) or [])
        return compute_centroid(embs) if embs else None

    if referee_tids:
        ref_groups: list[list[int]] = []
        for tid in referee_tids:
            placed = False
            for g in ref_groups:
                if not _conflicts(g, tid):
                    g.append(tid)
                    placed = True
                    break
            if not placed:
                ref_groups.append([tid])
        for i, group in enumerate(ref_groups, 1):
            pid = f"REF-{i:02d}"
            c = PlayerCluster(
                player_id=pid,
                team="referee",
                role="referee",
                jersey_number=None,
                jersey_confidence=0.0,
                source_tracks=list(group),
                reid_centroid=_centroid_of(group),
            )
            clusters.append(c)
            for tid in group:
                track_map[tid] = pid

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
    min_frames_for_pattern: int = 40,
    frame_count_by_track: dict[int, int] | None = None,
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
        # Skip pattern classification for short tracks — see caller note.
        if (frame_count_by_track is not None
                and frame_count_by_track.get(tid, 0) < min_frames_for_pattern):
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
        # Goalkeeper: very tight to one goal line, small length range,
        # AND positioned roughly in front of the goal mouth (|y| small).
        # Without the y check, a stationary coach / spectator standing
        # just behind a goal corner — e.g. orig 95 in the kids clip at
        # (-33.5, 10.2), well past the corner flag — passes the depth
        # checks (d_goal=0.4m, len_frac≈0) and gets falsely tagged as a
        # GK. Real GKs sit within ~6m of the goal centreline; outside
        # that band you're in corner / coach / spectator territory.
        GK_MAX_Y_OFFSET = 6.0
        if (on_pitch and d_goal <= 18.0
                and len_frac <= GK_LEN_FRAC
                and abs(y_med) <= GK_MAX_Y_OFFSET):
            pattern = "tight_near_goal"
            descr = (
                f"tight_near_goal: stays within ~{x_range:.1f}m of the "
                f"{nearest_side} goal line ({d_goal:.1f}m from goal), "
                f"y_med={y_med:.1f}m. This is the goalkeeper pattern."
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


def _detect_off_field_tracks(
    positions_by_track: dict[int, list[list[float]]],
    pitch_length: float,
    pitch_width: float,
    margin: float,
    min_outside_fraction: float,
    min_observations: int,
) -> set[int]:
    """Identify tracks that spent most of their lifetime outside the
    playing area (substitutes warming up, sideline staff, observers).

    A track is flagged off-field iff it has at least ``min_observations``
    pitch projections AND at least ``min_outside_fraction`` of them fall
    beyond ``half_pitch + margin`` in either x or y.

    Tracks that briefly stray over the touchline (e.g. a winger taking
    a throw-in) are NOT flagged — only persistently-outside tracks are.
    Tracks with no projections at all are skipped (calibration failure
    isn't grounds for filtering).
    """
    half_l = pitch_length / 2
    half_w = pitch_width / 2
    off: set[int] = set()
    for tid, poss in positions_by_track.items():
        if not poss or len(poss) < min_observations:
            continue
        arr = np.asarray(poss, dtype=np.float32)
        outside = (
            (np.abs(arr[:, 0]) > half_l + margin)
            | (np.abs(arr[:, 1]) > half_w + margin)
        )
        if outside.mean() >= min_outside_fraction:
            off.add(tid)
    return off


def _filter_tracks_by_id(
    tracks_by_frame: dict[str, list[dict[str, Any]]],
    drop: set[int],
) -> dict[str, list[dict[str, Any]]]:
    """Return a copy of ``tracks_by_frame`` with all dropped tids removed."""
    return {
        fk: [t for t in ts if int(t["track_id"]) not in drop]
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


def _write_consolidated_tracks(
    tracking_dir: Path,
    out_dir: Path,
    track_to_player: dict[int, str],
    clusters: list[PlayerCluster],
) -> None:
    """Write consolidated tracks.json + team_assignments.json into the
    consolidation stage's own dir.

    Reads the raw tracker output from ``tracking/tracks.json`` and
    writes the consolidated copy to ``out_dir/tracks.json``. The raw
    tracker file is *never* mutated — re-running consolidation can't
    destroy the source-of-truth, and downstream stages can switch
    between raw / consolidated views via
    :func:`goalinsight.track_consolidation.load_tracks`.

    Each track dict in the output keeps ``bbox`` / ``pitch_position`` /
    ``confidence`` from the raw tracker and gains:

      - ``track_id``        = the player_id string from consolidation
      - ``player_id``       = same string, exposed under the canonical
                              field name so downstream consumers
                              (events, chat run_python sandbox) don't
                              need to know the historical key
      - ``orig_track_id``   = the raw integer tid from the tracker
      - ``team`` / ``role`` / ``jersey_number`` / ``jersey_confidence``
                            = cluster attributes
    """
    raw_tracks = _load_json(tracking_dir / "tracks.json") or {}
    by_player: dict[str, PlayerCluster] = {c.player_id: c for c in clusters}

    consolidated: dict[str, list[dict[str, Any]]] = {}
    for fk, tracks in raw_tracks.items():
        kept: list[dict[str, Any]] = []
        for raw in tracks:
            t = dict(raw)
            tid = int(t["track_id"])
            pid = track_to_player.get(tid)
            if pid is None:
                # Unmapped track (short or Claude-rejected). Keep with
                # role=other + a synthetic id so rendering can show a
                # neutral box rather than a hole — real player attribution
                # still requires a real cluster.
                t["orig_track_id"] = tid
                t["track_id"] = f"unmapped-{tid}"
                t["player_id"] = t["track_id"]
                t["role"] = "other"
                t["team"] = "unknown"
                kept.append(t)
                continue
            cluster = by_player.get(pid)
            t["orig_track_id"] = tid
            t["track_id"] = pid
            t["player_id"] = pid
            if cluster and cluster.jersey_number is not None:
                t["jersey_number"] = cluster.jersey_number
                t["jersey_confidence"] = round(cluster.jersey_confidence, 3)
            if cluster:
                t["role"] = cluster.role
                t["team"] = cluster.team
            kept.append(t)
        consolidated[fk] = kept

    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "tracks.json", "w") as f:
        json.dump(consolidated, f)

    # team_assignments.json keyed by player_id (not raw int tid).
    player_teams = {c.player_id: c.team for c in clusters}
    with open(out_dir / "team_assignments.json", "w") as f:
        json.dump(player_teams, f, indent=2)
