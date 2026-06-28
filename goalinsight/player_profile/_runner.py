"""Build per-player profiles: front/back crops, heatmap PNG, distance run.

Inputs (all from upstream stages):
- ``track_consolidation/players.json`` — list of consolidated players, each
  with ``player_id``, ``source_tracks: [tid, ...]``, ``team``, ``role``,
  ``jersey_number``.
- ``track_consolidation/llm_inputs/per_track.json`` — per-original-track
  metadata; the ``crops`` array gives us each saved crop's filename,
  sharpness ``score`` and OCR ``visible_digits`` verdict (the orientation
  signal we use to pick front vs back).
- ``track_consolidation/llm_inputs/tracks/<name>.jpg`` — the actual crop
  images sampled during consolidation.
- ``track_consolidation/tracks.json`` — per-frame, player-id-keyed track
  list (used both for heatmap and distance computation; identical to
  what MatchContext exposes as ``player_tracks``).

Outputs (under ``output/<run>/player_profile/``):
- ``players_profile.json`` — list of profile dicts, one per player.
- ``crops/<player_id>_front.jpg`` / ``<player_id>_back.jpg`` — picked crops.
- ``heatmaps/<player_id>.png`` — pitch heatmap PNG.
"""

from __future__ import annotations

import io
import json
import logging
import math
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

# Same speed/gap thresholds match_tools.get_player_stats uses, so the
# numbers reported on the match page agree with the stage output.
_MAX_GAP_S = 1.0
_MAX_REALISTIC_MPS = 12.0


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def build_player_profiles(
    pipeline_output_dir: Path,
    out_dir: Path,
    *,
    heatmap_bins: int = 30,
    video_path: Path | None = None,
    spotlights_cfg: dict[str, Any] | None = None,
    skip_existing: bool = False,
) -> dict[str, Any]:
    """Build per-player profiles, writing all artifacts under *out_dir*.

    Returns a stats dict for the pipeline driver — total players,
    counts of front / back crops actually picked, etc.

    When *video_path* + ``spotlights_cfg.enabled`` are provided, also
    renders ``spotlights/<pid>.mp4`` per player (a follow-cam clip with
    the player centered, ~2/3 frame height, plus optional ellipse /
    trail / name-badge overlays). Skipped silently if the source video
    or homographies are missing.
    """
    pipeline_output_dir = Path(pipeline_output_dir)
    out_dir = Path(out_dir)
    crops_dir = out_dir / "crops"
    heatmaps_dir = out_dir / "heatmaps"
    out_dir.mkdir(parents=True, exist_ok=True)
    crops_dir.mkdir(parents=True, exist_ok=True)
    heatmaps_dir.mkdir(parents=True, exist_ok=True)

    consol_dir = pipeline_output_dir / "track_consolidation"
    players_path = consol_dir / "players.json"
    if not players_path.exists():
        raise RuntimeError(
            f"player_profile stage requires track_consolidation output: "
            f"{players_path} missing"
        )
    players = json.loads(players_path.read_text())

    per_track_path = consol_dir / "llm_inputs" / "per_track.json"
    per_track_index = _load_per_track(per_track_path)

    tracks = _load_tracks(consol_dir / "tracks.json")
    fps = _load_fps(pipeline_output_dir)

    # Pre-compute per-player observations once instead of scanning
    # tracks.json N times (one per player). For spotlight rendering we
    # also need the per-frame bbox; build that side-table in the same
    # walk to avoid a second pass.
    obs_by_pid = _index_observations_by_player(tracks)
    bbox_by_pid = _index_bboxes_by_player(tracks)

    # Pitch dimensions for the heatmap come from calibration metadata so
    # non-FIFA pitches (kids 66.28×43.15) render at the right aspect.
    pitch_length, pitch_width = _load_pitch_dims(pipeline_output_dir)

    crops_root = consol_dir / "llm_inputs" / "tracks"
    profiles: list[dict[str, Any]] = []
    front_count = back_count = heatmap_count = 0

    for entry in players:
        pid = entry.get("player_id")
        if not pid:
            continue
        source_tids = entry.get("source_tracks") or []

        front_src, back_src = _pick_orientation_crops(
            source_tids, per_track_index, crops_root,
        )
        front_url = back_url = None
        if front_src is not None:
            dest = crops_dir / f"{_safe_name(pid)}_front.jpg"
            shutil.copyfile(front_src, dest)
            front_url = f"crops/{dest.name}"
            front_count += 1
        if back_src is not None:
            dest = crops_dir / f"{_safe_name(pid)}_back.jpg"
            shutil.copyfile(back_src, dest)
            back_url = f"crops/{dest.name}"
            back_count += 1

        positions = obs_by_pid.get(pid, [])
        distance_m, max_speed, avg_speed = _trajectory_stats(positions, fps)

        heatmap_url = None
        if positions:
            png = _render_heatmap_png(
                [(p[1], p[2]) for p in positions],
                bins=heatmap_bins,
                pitch_length=pitch_length,
                pitch_width=pitch_width,
                title=f"{pid}",
            )
            if png is not None:
                hpath = heatmaps_dir / f"{_safe_name(pid)}.png"
                hpath.write_bytes(png)
                heatmap_url = f"heatmaps/{hpath.name}"
                heatmap_count += 1

        profiles.append({
            "player_id": pid,
            "team_id": entry.get("team"),
            "role": entry.get("role"),
            "jersey_number": entry.get("jersey_number"),
            "frames_observed": len(positions),
            "distance_m": round(distance_m, 2),
            "max_speed_mps": round(max_speed, 2),
            "avg_speed_mps": round(avg_speed, 2),
            "front_crop": front_url,
            "back_crop": back_url,
            "heatmap": heatmap_url,
        })

    spot_count = _maybe_render_spotlights(
        profiles=profiles,
        out_dir=out_dir,
        pipeline_output_dir=pipeline_output_dir,
        video_path=video_path,
        bbox_by_pid=bbox_by_pid,
        obs_by_pid=obs_by_pid,
        fps=fps,
        cfg=spotlights_cfg or {},
        skip_existing=skip_existing,
    )

    profile_path = out_dir / "players_profile.json"
    profile_path.write_text(json.dumps(profiles, indent=2))

    return {
        "players": len(profiles),
        "front_crops": front_count,
        "back_crops": back_count,
        "heatmaps": heatmap_count,
        "spotlights": spot_count,
    }


# ---------------------------------------------------------------------------
# Crop picker — uses the OCR "visible_digits" verdict as a facing hint.
# ---------------------------------------------------------------------------


def _pick_orientation_crops(
    source_tids: list[int],
    per_track_index: dict[int, list[dict[str, Any]]],
    crops_root: Path,
) -> tuple[Path | None, Path | None]:
    """Pick the sharpest front-facing and back-facing crop across all
    source tracks of a consolidated player.

    Front: ``ocr_visible == 'back not visible'`` — VLM saw the chest /
    front of the kit, not the number.
    Back: ``ocr_visible`` is a digit string (or ``partial: <digit>``) —
    the number is at least partially readable, so the back is facing the
    camera.

    Falls back to the highest-score crop overall when no labelled
    candidate exists, so even orientation-ambiguous players still get a
    representative thumbnail.
    """
    fronts: list[tuple[float, Path]] = []
    backs: list[tuple[float, Path]] = []
    any_crop: list[tuple[float, Path]] = []

    for tid in source_tids:
        for crop in per_track_index.get(int(tid), []):
            name = crop.get("name")
            if not name:
                continue
            path = crops_root / name
            if not path.exists():
                continue
            score = float(crop.get("score") or 0.0)
            label = (crop.get("ocr_visible") or "").strip().lower()
            any_crop.append((score, path))
            if label == "back not visible":
                fronts.append((score, path))
            elif _looks_like_back(label):
                backs.append((score, path))

    front = max(fronts, key=lambda x: x[0])[1] if fronts else None
    back = max(backs, key=lambda x: x[0])[1] if backs else None

    # If only one orientation surfaced, fill the other slot with the
    # best remaining crop so users always see two thumbnails when crops
    # are available at all.
    if any_crop:
        any_crop.sort(key=lambda x: -x[0])
        used = {front, back}
        for _, path in any_crop:
            if front is None and path not in used:
                front = path
                used.add(path)
            elif back is None and path not in used:
                back = path
                used.add(path)
            if front is not None and back is not None:
                break
    return front, back


def _looks_like_back(label: str) -> bool:
    """Verdict implies the number is at least partially visible."""
    if not label:
        return False
    if label.startswith("partial: "):
        rest = label[len("partial: "):]
        return rest and rest[0].isdigit()
    return label[0].isdigit()


# ---------------------------------------------------------------------------
# Trajectory + heatmap utilities
# ---------------------------------------------------------------------------


def _trajectory_stats(
    positions: list[tuple[int, float, float]],
    fps: float,
) -> tuple[float, float, float]:
    """Total distance + speed stats. Mirrors match_tools.get_player_stats so
    the per-player numbers shown on the match page line up exactly."""
    if not positions:
        return 0.0, 0.0, 0.0
    positions = sorted(positions, key=lambda p: p[0])
    distance = 0.0
    max_speed = 0.0
    speeds: list[float] = []
    for (f1, x1, y1), (f2, x2, y2) in zip(positions, positions[1:]):
        dt = (f2 - f1) / fps if fps else 0
        if dt <= 0 or dt > _MAX_GAP_S:
            continue
        d = math.hypot(x2 - x1, y2 - y1)
        speed = d / dt
        if speed > _MAX_REALISTIC_MPS:
            continue
        distance += d
        speeds.append(speed)
        max_speed = max(max_speed, speed)
    avg = (sum(speeds) / len(speeds)) if speeds else 0.0
    return distance, max_speed, avg


def _render_heatmap_png(
    positions: list[tuple[float, float]],
    *,
    bins: int,
    pitch_length: float,
    pitch_width: float,
    title: str,
) -> bytes | None:
    """Render a heatmap PNG over a schematic top-down pitch.

    Reads geometry from the *active* SoccerPitch (set by the pipeline's
    stage 1 from the per-video config) so PA shape, goal-area
    dimensions, center-circle radius, etc. match the run's pitch type
    (futsal D, kids-soccer, FIFA, ...). Uses the shared
    ``make_pitch_canvas`` + ``draw_pitch_structure`` helpers so the
    schematic matches every other top-down panel exactly.
    """
    if not positions:
        return None
    # Lazy matplotlib import — keeps stage import cheap when heatmaps
    # aren't being generated (e.g. tests).
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from ..annotation import pitch_constants
    from ..annotation.pitch.geometry import SoccerPitch
    from ..annotation.pitch_diagram import (
        draw_pitch_structure, make_pitch_canvas,
    )

    # Prefer the process-global active pitch when its outer dims match
    # the run's pitch_length/width — that's the path the pipeline takes
    # (stage 1 sets active pitch from the per-video config). When they
    # diverge, fall back to a scaled FIFA so the schematic still tracks
    # the run's outer dimensions.
    active = pitch_constants.get_active_pitch()
    if (abs(active.PITCH_LENGTH - pitch_length) < 1e-3
            and abs(active.PITCH_WIDTH - pitch_width) < 1e-3):
        pitch = active
    else:
        base = SoccerPitch()
        sx = pitch_length / base.PITCH_LENGTH
        sy = pitch_width / base.PITCH_WIDTH
        pitch = SoccerPitch(
            pitch_length=pitch_length,
            pitch_width=pitch_width,
            penalty_area_length=base.PENALTY_AREA_LENGTH * sx,
            penalty_area_width=base.PENALTY_AREA_WIDTH * sy,
            goal_area_length=base.GOAL_AREA_LENGTH * sx,
            goal_area_width=base.GOAL_AREA_WIDTH * sy,
            goal_line_to_penalty_mark=base.GOAL_LINE_TO_PENALTY_MARK * sx,
            center_circle_radius=base.CENTER_CIRCLE_RADIUS * min(sx, sy),
        )

    L = pitch_length / 2.0
    W = pitch_width / 2.0

    # Build the schematic with the shared helpers (handles D-shape PA,
    # penalty arcs, landmarks). Convert to RGB for matplotlib.
    scale_px = 12  # px per metre — matches analytics' PITCH_SCALE
    margin_px = 30
    img, to_px, w_px, h_px = make_pitch_canvas(
        scale_px, margin_px, bg=(34, 100, 34), pitch=pitch,
    )
    draw_pitch_structure(
        img, to_px, scale_px,
        color=(207, 227, 255), thickness=2, pitch=pitch,
    )

    fig, ax = plt.subplots(figsize=(w_px / 100, h_px / 100), dpi=100)
    ax.imshow(img[:, :, ::-1])  # BGR → RGB
    ax.set_xlim(0, w_px)
    ax.set_ylim(h_px, 0)
    ax.set_axis_off()

    # Histogram-based heatmap, projected into the pitch canvas pixel grid.
    xs = np.array([p[0] for p in positions])
    ys = np.array([p[1] for p in positions])
    H, xedges, yedges = np.histogram2d(
        xs, ys, bins=bins, range=[[-L, L], [-W, W]],
    )
    xc = 0.5 * (xedges[:-1] + xedges[1:])
    yc = 0.5 * (yedges[:-1] + yedges[1:])
    xx, yy = np.meshgrid(xc, yc, indexing="ij")
    pxs = (xx + L) * scale_px + margin_px
    pys = (W - yy) * scale_px + margin_px
    ax.pcolormesh(pxs, pys, H, alpha=0.6, shading="auto", cmap="hot")
    ax.set_title(title, fontsize=11, color="#222")

    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return buf.getvalue()


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------


def _load_per_track(path: Path) -> dict[int, list[dict[str, Any]]]:
    if not path.exists():
        return {}
    blob = json.loads(path.read_text())
    raw = blob.get("tracks") or {}
    out: dict[int, list[dict[str, Any]]] = {}
    for k, v in raw.items():
        try:
            tid = int(k)
        except (TypeError, ValueError):
            continue
        out[tid] = v.get("crops") or []
    return out


def _load_tracks(path: Path) -> dict[str, list[dict[str, Any]]]:
    if not path.exists():
        return {}
    return json.loads(path.read_text())


def _load_fps(pipeline_output_dir: Path) -> float:
    meta = pipeline_output_dir / "field_registration" / "calibration_metadata.json"
    if meta.exists():
        try:
            return float(json.loads(meta.read_text()).get("video_info", {}).get("fps") or 30.0)
        except Exception:  # noqa: BLE001
            return 30.0
    return 30.0


def _load_pitch_dims(pipeline_output_dir: Path) -> tuple[float, float]:
    meta = pipeline_output_dir / "field_registration" / "calibration_metadata.json"
    if meta.exists():
        try:
            vi = json.loads(meta.read_text()).get("video_info", {}) or {}
            return float(vi.get("pitch_length") or 105.0), float(vi.get("pitch_width") or 68.0)
        except Exception:  # noqa: BLE001
            pass
    return 105.0, 68.0


def _index_observations_by_player(
    tracks: dict[str, list[dict[str, Any]]],
) -> dict[str, list[tuple[int, float, float]]]:
    out: dict[str, list[tuple[int, float, float]]] = defaultdict(list)
    for k, items in tracks.items():
        try:
            f = int(k)
        except (TypeError, ValueError):
            continue
        for t in items:
            pid = t.get("track_id")
            pp = t.get("pitch_position") or [None, None]
            if pid is None or pp[0] is None:
                continue
            out[str(pid)].append((f, float(pp[0]), float(pp[1])))
    return out


def _safe_name(pid: str) -> str:
    return "".join(c if c.isalnum() or c in "._-" else "_" for c in str(pid))


def _index_bboxes_by_player(
    tracks: dict[str, list[dict[str, Any]]],
) -> dict[str, list[dict[str, Any]]]:
    """``player_id -> [{frame, bbox}, ...]`` sorted by frame."""
    out: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for k, items in tracks.items():
        try:
            f = int(k)
        except (TypeError, ValueError):
            continue
        for t in items:
            pid = t.get("track_id")
            bbox = t.get("bbox")
            if pid is None or bbox is None:
                continue
            out[str(pid)].append({"frame": f, "bbox": bbox})
    for pid in out:
        out[pid].sort(key=lambda r: r["frame"])
    return out


# Default BGR colors for team-color badges. Resolved by team label
# (kmeans/tracklet emit team_a / team_b strings; we map both forms).
_TEAM_COLORS_BGR: dict[str, tuple[int, int, int]] = {
    "team_a": (180, 70, 40),    # blue
    "team_a_kit": (180, 70, 40),
    "A": (180, 70, 40),
    "team_b": (40, 60, 200),    # red
    "team_b_kit": (40, 60, 200),
    "B": (40, 60, 200),
    "referee": (45, 45, 45),
    "unknown": (130, 130, 130),
}


def _resolve_team_color(team: str | None) -> tuple[int, int, int]:
    if not team:
        return _TEAM_COLORS_BGR["unknown"]
    # Tracker emits "team_A" / "team_B"; consolidator sometimes lowercases
    # to "team_a". Match either by normalizing.
    key = str(team).lower()
    return _TEAM_COLORS_BGR.get(key, _TEAM_COLORS_BGR["unknown"])


def _badge_text(profile: dict[str, Any]) -> str:
    jersey = profile.get("jersey_number")
    role = profile.get("role") or ""
    if jersey not in (None, "", "unknown"):
        return f"#{jersey} {role}".strip()
    pid = str(profile.get("player_id") or "?")
    return f"{pid} {role}".strip()


def _maybe_render_spotlights(
    *,
    profiles: list[dict[str, Any]],
    out_dir: Path,
    pipeline_output_dir: Path,
    video_path: Path | None,
    bbox_by_pid: dict[str, list[dict[str, Any]]],
    obs_by_pid: dict[str, list[tuple[int, float, float]]],
    fps: float,
    cfg: dict[str, Any],
    skip_existing: bool,
) -> int:
    """Render per-player spotlight clips. No-op when disabled / impossible.

    Mutates *profiles* in place to add ``spotlight_video`` and
    ``spotlight_duration_s`` fields. Returns the count of clips rendered
    (or already-present skipped) so the stage stats reflect coverage.
    """
    if not cfg.get("enabled", True):
        return 0
    if video_path is None or not Path(video_path).exists():
        logger.warning(
            "player_profile: spotlight render skipped — video_path missing/unreadable"
        )
        return 0

    # Lazy imports: keep the rest of the stage import-cheap when we're
    # not rendering (e.g. legacy runs, --skip-existing already done).
    import pickle

    from ._spotlight import render_player_spotlight

    homographies = None
    h_path = pipeline_output_dir / "field_registration" / "homographies.pkl"
    if h_path.exists():
        try:
            homographies = pickle.load(open(h_path, "rb"))
        except Exception:  # noqa: BLE001 — bad pickle just disables trail
            logger.warning("player_profile: failed to load homographies for trail")
            homographies = None

    spotlight_dir = out_dir / "spotlights"
    spotlight_dir.mkdir(exist_ok=True)

    output_size = tuple(cfg.get("output_size") or (1920, 1080))
    target_frac = float(cfg.get("target_player_height_frac", 2 / 3))
    presence_gap_s = float(cfg.get("presence_gap_seconds", 1.0))
    enable_ellipse = bool(cfg.get("ellipse", True))
    enable_trail = bool(cfg.get("trail", True)) and homographies is not None
    enable_badge = bool(cfg.get("name_badge", True))
    trail_seconds = float(cfg.get("trail_seconds", 1.5))
    crf = int(cfg.get("crf", 23))
    preset = str(cfg.get("preset", "medium"))
    min_obs = int(cfg.get("min_observations", 30))

    rendered = 0
    for profile in profiles:
        pid = str(profile["player_id"])
        traj = bbox_by_pid.get(pid) or []
        if len(traj) < min_obs:
            continue

        out_mp4 = spotlight_dir / f"{_safe_name(pid)}.mp4"
        rel_url = f"spotlights/{out_mp4.name}"
        if out_mp4.exists() and skip_existing:
            profile["spotlight_video"] = rel_url
            # Best-effort fps×frames if we don't probe; Player_profile
            # doesn't run ffprobe to keep the stage hermetic. The frontend
            # will still play it; duration is informational.
            rendered += 1
            continue

        # Pitch positions for trail are already keyed by frame in obs_by_pid
        # (frame, x, y) tuples — flatten to dict[frame] -> [(x, y)].
        pps = obs_by_pid.get(pid) or []
        pp_by_frame: dict[int, list[tuple[float, float]]] = {}
        for f, x, y in pps:
            pp_by_frame.setdefault(f, []).append((x, y))

        try:
            meta = render_player_spotlight(
                video_path=Path(video_path),
                player_id=pid,
                trajectory=traj,
                pitch_positions_by_frame=pp_by_frame if enable_trail else None,
                homographies=homographies if enable_trail else None,
                fps=fps,
                output_path=out_mp4,
                output_size=output_size,
                target_player_height_frac=target_frac,
                presence_gap_seconds=presence_gap_s,
                enable_ellipse=enable_ellipse,
                enable_trail=enable_trail,
                enable_name_badge=enable_badge,
                trail_seconds=trail_seconds,
                name_badge_text=_badge_text(profile) if enable_badge else None,
                name_badge_team_color=_resolve_team_color(profile.get("team_id")),
                crf=crf,
                preset=preset,
            )
        except Exception:  # noqa: BLE001 — one player failing shouldn't kill the stage
            logger.exception("player_profile: spotlight render failed for %s", pid)
            continue

        profile["spotlight_video"] = rel_url
        profile["spotlight_duration_s"] = round(float(meta["duration_s"]), 2)

        # Sidecar maps spotlight clip frame N → broadcast frame, used by
        # the match-page frontend to keep the right-side 2D pitch view
        # in sync (highlight selected player + draw teammates) while the
        # spotlight clip plays. Tiny JSON (a few KB per player).
        sidecar = spotlight_dir / f"{_safe_name(pid)}.frames.json"
        sidecar.write_text(json.dumps({
            "fps": float(fps),
            "broadcast_frames": meta.get("broadcast_frames") or [],
        }))
        profile["spotlight_frames_json"] = f"spotlights/{sidecar.name}"

        rendered += 1
        logger.info(
            "[%s] spotlight: %d frames → %s (%.1f s)",
            pid, meta["frames_rendered"], rel_url, meta["duration_s"],
        )

    return rendered
