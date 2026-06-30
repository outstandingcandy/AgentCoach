"""Match detail page API: bundles roster, events with seek windows,
and per-player presence ranges into a single JSON payload.

The match detail page (``/match/{run_name}``) draws bbox / spotlight /
mini-pitch overlays client-side from ``tracks.json`` and
``ball_tracks.json`` served via ``/runs_static/...``. This route pack
just precomputes the things the frontend can't easily derive on its
own:

- per-event time windows (using highlight temporal helpers for goals
  and shots, ``start_frame``/``end_frame`` or ±2 s otherwise);
- per-track contiguous presence ranges, merged across short gaps so a
  single dropout doesn't cut a clip in half.

Results are cached on the ``RunHandle`` so the (slightly expensive)
scan over all per-frame tracks runs once per warm run.
"""

from __future__ import annotations

import json
import math
from collections import defaultdict
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse

from ..highlights._context import MatchContext
from ..highlights._temporal import find_buildup_start, find_celebration_end
from ._pitch import resolve_pitch_for_run
from ._runs import RunHandle, RunRegistry
from .match_tools import _frame_to_time

# Default ± window for events without their own start_frame/end_frame.
_DEFAULT_PRE_S = 2.0
_DEFAULT_POST_S = 2.0
# Tight ± window around shot/goal events for the page timeline. The
# wider buildup-+-celebration framing in highlights is for clip
# rendering, not for the live timeline — there it just smears the
# event marker across many seconds. 0.5 s on each side keeps the
# striker badge on for the kick + immediate aftermath only.
_SHOT_PRE_S = 0.5
_SHOT_POST_S = 0.5
# Buildup/celebration constants kept for the (now unused on the page)
# highlight clip path; left in place for cross-imports.
_BUILDUP_MAX_S = 8.0
_BUILDUP_PAD_S = 1.5
_CELEBRATION_S = 4.0
# Two presence observations within this gap are treated as the same clip.
_PLAYER_CLIP_GAP_S = 1.0


def register_match_detail_routes(app: FastAPI, runs: RunRegistry) -> None:
    @app.get("/api/runs/{run_name}/match/data")
    def match_data(run_name: str) -> JSONResponse:
        from ._runs import IncompleteRunError
        try:
            handle = runs.get(run_name)
        except FileNotFoundError as exc:
            raise HTTPException(404, str(exc))
        except IncompleteRunError as exc:
            raise HTTPException(
                status_code=409,
                detail={
                    "error": "incomplete_run",
                    "run_name": exc.run_name,
                    "message": str(exc),
                    "stages": exc.stages,
                    "missing_required": exc.missing_required,
                    "missing_recommended": exc.missing_recommended,
                },
            ) from exc
        return JSONResponse(_build_payload(handle, runs.workspace))


# ---------------------------------------------------------------------------
# Payload builder (cached on the RunHandle)
# ---------------------------------------------------------------------------


def _build_payload(handle: RunHandle, workspace: Any) -> dict[str, Any]:
    """Build the match-detail payload fresh on each request.

    Previously this was cached on the RunHandle, but that meant
    re-running event_detection (or any pipeline stage that rewrites
    its outputs) wasn't visible to the page until the web process
    restarted. We also drop the MatchContext-side ``_events`` cache
    here so the in-handle ctx re-reads ``events.json`` from disk.
    Player clips and camera-fov polygons are also dependent on the
    pipeline outputs and get re-derived per request.
    """
    ctx = handle.ctx
    # Invalidate the lazy MatchContext caches that read pipeline
    # outputs from disk. Static-once fields (fps, pitch dims, video
    # geometry) are kept; tracks / ball / events / team_assignments
    # are re-read on next access.
    ctx._events = None  # type: ignore[attr-defined]
    ctx._player_tracks = None  # type: ignore[attr-defined]
    ctx._ball_tracks = None  # type: ignore[attr-defined]
    ctx._team_assignments = None  # type: ignore[attr-defined]
    ctx._camera_poses = None  # type: ignore[attr-defined]
    payload = {
        "fps": ctx.fps,
        "frame_count": ctx.frame_count,
        "width": ctx.width,
        "height": ctx.height,
        "pitch_length": ctx.pitch_length,
        "pitch_width": ctx.pitch_width,
        "pitch": _pitch_geometry(ctx, handle.video_path, workspace),
        "tracks_url": (
            f"/runs_static/{handle.run_name}/track_consolidation/tracks.json"
        ),
        "ball_url": (
            f"/runs_static/{handle.run_name}/tracking/ball_tracks.json"
        ),
        "players": _players(ctx, handle.run_name, handle.run_dir),
        "events": _events(ctx),
        "player_clips": _player_clips(ctx),
        "camera_fov": _camera_fov_polygons(ctx),
    }
    return payload


def _pitch_geometry(
    ctx: MatchContext,
    video_path: Any = None,
    workspace: Any = None,
) -> dict[str, Any]:
    """Active pitch dimensions + 2D top-down polylines for the match panel.

    Builds world-space polylines (from ``pitch_constants.PITCH_LINES`` +
    ``build_d_penalty_arcs``) so the client just maps world (x, y) →
    pixel and strokes line strips. Same source as the annotate page,
    field-registration vis, and tracking minimap — single source of
    truth across every top-down panel.

    Pitch resolution reads ``workspace/configs/<video>.yaml`` directly
    (see :func:`web._pitch.resolve_pitch_for_run`) so per-run pitch
    survives the process-global active-pitch racing across runs.
    """
    import math
    from ..annotation import pitch_constants

    p = resolve_pitch_for_run(ctx, video_path, workspace)
    L = p.PITCH_LENGTH
    W = p.PITCH_WIDTH

    pa_shape = getattr(p, "PENALTY_AREA_SHAPE", "rect")

    # World-space polylines that paint the pitch. Each entry is a list
    # of (x, y) world coords (metres, centre-origin, y-up). The client
    # draws each as a connected line strip.
    polylines: list[list[list[float]]] = []

    # Rebuild PITCH_LINES against this run's pitch (don't trust the
    # process-global state).
    line_dict = pitch_constants._build_lines_for_pitch(p)
    pa_rect_names = {
        "penalty_left_top", "penalty_left_bottom", "penalty_left_front",
        "penalty_right_top", "penalty_right_bottom", "penalty_right_front",
    }
    for name, ((wx1, wy1), (wx2, wy2)) in line_dict.items():
        if pa_shape == "d" and name in pa_rect_names:
            continue
        polylines.append([[wx1, wy1], [wx2, wy2]])

    # Centre circle.
    ccr = p.CENTER_CIRCLE_RADIUS
    n = 96
    polylines.append([
        [ccr * math.cos(2 * math.pi * i / n),
         ccr * math.sin(2 * math.pi * i / n)]
        for i in range(n + 1)
    ])

    if pa_shape == "d":
        for poly in pitch_constants.build_d_penalty_arcs(p):
            polylines.append([[x, y] for x, y in poly])
    else:
        # 11-a-side penalty arc.
        pa_d = p.PENALTY_AREA_LENGTH
        pen = p.GOAL_LINE_TO_PENALTY_MARK
        dx = pa_d - pen
        ratio = dx / ccr if ccr > 0 else 1.0
        if -1.0 < ratio < 1.0:
            half = math.acos(ratio)
            for sign in (-1.0, 1.0):
                cx = sign * (p.PITCH_LENGTH / 2 - pen)
                if sign < 0:
                    ts = [-half + (2 * half) * i / 39 for i in range(40)]
                else:
                    ts = [math.pi - half + (2 * half) * i / 39 for i in range(40)]
                polylines.append([
                    [cx + ccr * math.cos(t), ccr * math.sin(t)] for t in ts
                ])

    pen = p.GOAL_LINE_TO_PENALTY_MARK
    landmarks = [
        [0.0, 0.0],
        [-p.PITCH_LENGTH / 2 + pen, 0.0],
        [p.PITCH_LENGTH / 2 - pen, 0.0],
    ]

    return {
        "length": L,
        "width": W,
        "penalty_area_length": p.PENALTY_AREA_LENGTH,
        "penalty_area_width": p.PENALTY_AREA_WIDTH,
        "goal_area_length": p.GOAL_AREA_LENGTH,
        "goal_area_width": p.GOAL_AREA_WIDTH,
        "center_circle_radius": p.CENTER_CIRCLE_RADIUS,
        "penalty_area_shape": pa_shape,
        "polylines_2d": polylines,
        "landmarks_2d": landmarks,
    }


def _players(
    ctx: MatchContext,
    run_name: str | None = None,
    run_dir: Any = None,
) -> list[dict[str, Any]]:
    """Roster with frames_observed + jersey + team, ordered by visibility.

    When ``run_dir`` is provided and a player_profile stage has run, each
    entry is enriched with crops/heatmap URLs and per-player distance —
    so the match page can show a profile pane without a second request.
    """
    counts: dict[str, int] = defaultdict(int)
    jerseys: dict[str, Any] = {}
    teams: dict[str, str] = {}
    roles: dict[str, str] = {}
    for items in ctx.player_tracks.values():
        for t in items:
            tid = t.get("track_id")
            if tid is None:
                continue
            tid = str(tid)
            # Skip the consolidator's synthetic ``unmapped-<orig_tid>``
            # tracks — these are short / off-field / LLM-rejected
            # fragments the consolidator preserved in tracks.json only
            # so the per-frame renderer can paint a neutral box rather
            # than a hole. They are NOT players and shouldn't pollute
            # the match roster or the right-side minimap dots.
            if tid.startswith("unmapped-"):
                continue
            counts[tid] += 1
            if tid not in jerseys and t.get("jersey_number") is not None:
                jerseys[tid] = t.get("jersey_number")
            if tid not in teams:
                teams[tid] = (
                    t.get("team")
                    or ctx.team_assignments.get(tid, "unknown")
                )
            if tid not in roles and t.get("role") is not None:
                roles[tid] = t.get("role")

    profiles = _load_profiles(run_dir, run_name) if run_dir is not None else {}

    out = []
    for tid, n in sorted(counts.items(), key=lambda kv: -kv[1]):
        entry = {
            "player_id": tid,
            "team_id": teams.get(tid, "unknown"),
            "role": roles.get(tid),
            "jersey_number": jerseys.get(tid),
            "frames_observed": n,
        }
        prof = profiles.get(tid)
        if prof:
            entry.update(prof)
        out.append(entry)
    return out


def _load_profiles(
    run_dir: Any, run_name: str | None,
) -> dict[str, dict[str, Any]]:
    """Read player_profile/players_profile.json + rewrite asset paths to
    /runs_static URLs the browser can fetch directly."""
    from pathlib import Path
    if not run_dir or not run_name:
        return {}
    profile_path = Path(run_dir) / "player_profile" / "players_profile.json"
    if not profile_path.exists():
        return {}
    try:
        rows = json.loads(profile_path.read_text())
    except (OSError, ValueError):
        return {}
    base = f"/runs_static/{run_name}/player_profile"
    out: dict[str, dict[str, Any]] = {}
    for row in rows:
        pid = str(row.get("player_id") or "")
        if not pid:
            continue
        out[pid] = {
            "distance_m": row.get("distance_m"),
            "max_speed_mps": row.get("max_speed_mps"),
            "avg_speed_mps": row.get("avg_speed_mps"),
            "front_crop_url": _abs_url(base, row.get("front_crop")),
            "back_crop_url": _abs_url(base, row.get("back_crop")),
            "heatmap_url": _abs_url(base, row.get("heatmap")),
            "spotlight_video_url": _abs_url(base, row.get("spotlight_video")),
            "spotlight_duration_s": row.get("spotlight_duration_s"),
            "spotlight_frames_url": _abs_url(base, row.get("spotlight_frames_json")),
        }
    return out


def _abs_url(base: str, rel: str | None) -> str | None:
    if not rel:
        return None
    return f"{base}/{rel.lstrip('/')}"


# ---------------------------------------------------------------------------
# Per-frame camera FOV polygon on the ground plane.
# ---------------------------------------------------------------------------


# Image-edge sampling. Matches the 30 points/side that FR's topdown
# vis uses (utils/pitch.py:_draw_topdown_pitch) so the match minimap
# FOV polygon is bit-identical to the FR vis FOV. Payload is
# (4 × 30 × 2) × 4 B ≈ ~1 kB per frame after gzip — affordable.
_FOV_POINTS_PER_SIDE = 30
# Drop ground hits more than this many metres outside the pitch — the
# unprojection of points near the horizon line goes to enormous values.
# Matches the ``max_range`` FR's ``utils/pitch._img_to_world`` uses
# (default 80 m) so the match-page FOV polygon and the FR stage-1
# topdown FOV polygon are bit-identical for the same pose.
_FOV_GROUND_CLAMP_M = 80.0


def _camera_fov_polygons(ctx: MatchContext) -> dict[str, Any]:
    """Per-frame ground-plane polygon of the camera's image rectangle.

    Computed only for runs whose calibration backend produced
    ``camera_poses.json`` (physical / pnlcalib). Homography-only runs
    skip this — the homography H itself already maps image→world but
    K/R/t aren't available, and the homography backend is rarely used
    in the kids workflow that drives this page.
    """
    poses = ctx.camera_poses or {}
    if not poses:
        return {"frames": []}
    img_w, img_h = ctx.width or 1920, ctx.height or 1080

    # Pre-build the boundary sampling grid in image space once.
    n = _FOV_POINTS_PER_SIDE
    boundary = []
    for i in range(n):
        boundary.append((img_w * i / n, 0.0))
    for i in range(n):
        boundary.append((float(img_w), img_h * i / n))
    for i in range(n):
        boundary.append((img_w * (1 - i / n), float(img_h)))
    for i in range(n):
        boundary.append((0.0, img_h * (1 - i / n)))

    out: list[dict[str, Any]] = []
    for k, pose in poses.items():
        try:
            frame = int(k)
        except (TypeError, ValueError):
            continue
        poly = _project_image_boundary_to_ground(pose, boundary)
        if poly is None or len(poly) < 3:
            continue
        entry: dict[str, Any] = {"frame": frame, "polygon": poly}
        # Camera position (world coords) — annotate-page-style marker on
        # the minimap. Computed from R, t so we don't depend on the
        # pose dict carrying a ``camera_position`` sub-dict (some
        # backends omit it).
        cam_xyz = _camera_position_from_pose(pose)
        if cam_xyz is not None:
            entry["camera"] = [
                round(cam_xyz[0], 3),
                round(cam_xyz[1], 3),
                round(cam_xyz[2], 3),
            ]
        out.append(entry)
    out.sort(key=lambda x: x["frame"])
    return {"frames": out}


def _camera_position_from_pose(pose: dict[str, Any]) -> tuple[float, float, float] | None:
    """Return world-coord (x, y, z) of the camera from a pose dict.

    Prefers an explicit ``camera_position`` sub-dict when present (the
    fixed-camera and physical backends both write one), falls back to
    ``-R.T @ tvec`` otherwise.
    """
    explicit = pose.get("camera_position")
    if isinstance(explicit, dict) and all(k in explicit for k in ("x", "y", "z")):
        try:
            return (float(explicit["x"]),
                    float(explicit["y"]),
                    float(explicit["z"]))
        except (TypeError, ValueError):
            pass

    rvec = pose.get("rvec")
    tvec = pose.get("tvec")
    if rvec is None or tvec is None:
        return None
    import cv2 as _cv2
    import numpy as _np
    R, _ = _cv2.Rodrigues(_np.asarray(rvec, dtype=_np.float64).reshape(3, 1))
    t = _np.asarray(tvec, dtype=_np.float64).reshape(3)
    pos = (-R.T @ t).ravel()
    return (float(pos[0]), float(pos[1]), float(pos[2]))


def _project_image_boundary_to_ground(
    pose: dict[str, Any], boundary_img: list[tuple[float, float]],
) -> list[list[float]] | None:
    """Back-project each image-boundary point to the z=0 ground plane.

    Uses the same math as ``utils/pitch.py:_img_to_world`` — applies the
    Brown-Conrady distortion via ``cv2.undistortPoints`` so wide-angle
    / fisheye lenses (futsal phone profile carries k1≈0.48) produce the
    same FOV polygon on the match minimap as the field-registration
    visualisation. Without distortion the polygon shape diverges at the
    edges where radial distortion is strongest.
    """
    import cv2 as _cv2
    import numpy as _np

    K = pose.get("K")
    if K is None:
        return None
    rvec = pose.get("rvec")
    tvec = pose.get("tvec")
    if rvec is None or tvec is None:
        return None

    K_arr = _np.asarray(K, dtype=_np.float64)
    rvec_arr = _np.asarray(rvec, dtype=_np.float64).reshape(3, 1)
    tvec_arr = _np.asarray(tvec, dtype=_np.float64).reshape(3)
    dist_arr = _np.asarray(
        pose.get("dist_coeffs") or [0.0, 0.0, 0.0, 0.0, 0.0],
        dtype=_np.float64,
    ).ravel()

    R, _ = _cv2.Rodrigues(rvec_arr)
    cam_center = -R.T @ tvec_arr

    pts = _np.asarray(boundary_img, dtype=_np.float64).reshape(-1, 1, 2)
    pts_norm = _cv2.undistortPoints(pts, K_arr, dist_arr).reshape(-1, 2)

    poly: list[list[float]] = []
    for xn, yn in pts_norm:
        ray_world = R.T @ _np.array([xn, yn, 1.0], dtype=_np.float64)
        if abs(ray_world[2]) < 1e-9:
            continue
        t = -cam_center[2] / ray_world[2]
        if t < 0:
            continue
        wx = float(cam_center[0] + t * ray_world[0])
        wy = float(cam_center[1] + t * ray_world[1])
        # Mirror FR's behaviour (utils/pitch._img_to_world): drop
        # near-horizon rays that back-project far outside the pitch
        # instead of clamping. The resulting polygon may be open at the
        # sky-end, but the client fills it the same way FR fills its
        # pitch.copy() overlay, so both panels look identical.
        if abs(wx) > _FOV_GROUND_CLAMP_M or abs(wy) > _FOV_GROUND_CLAMP_M:
            continue
        poly.append([round(wx, 3), round(wy, 3)])
    return poly


def _events(ctx: MatchContext) -> list[dict[str, Any]]:
    fps = ctx.fps or 30.0
    out: list[dict[str, Any]] = []
    for e in ctx.events:
        # Possession spans are dense and not useful as user-facing clips;
        # the page surfaces them implicitly via the active-possessor
        # outline rather than as filterable events.
        if e.get("type") == "possession":
            continue
        frame = int(e.get("frame", 0))
        start_f, end_f = _event_window_frames(ctx, e, fps)
        out.append({
            "event_id": e.get("event_id"),
            "type": e.get("type"),
            "frame": frame,
            "start_s": _frame_to_time(start_f, fps),
            "end_s": _frame_to_time(end_f, fps),
            "player_id": e.get("player_id"),
            "team_id": e.get("team_id"),
            "metadata": e.get("metadata") or {},
        })
    out.sort(key=lambda x: x["frame"])
    return out


def _event_window_frames(
    ctx: MatchContext, event: dict[str, Any], fps: float,
) -> tuple[int, int]:
    et = event.get("type")
    frame = int(event.get("frame", 0))
    total = ctx.frame_count or None

    if et in {"goal", "shot"}:
        # Tight window centred on the shot frame — just the kick and
        # the moment after. The previous build-up-to-celebration
        # framing was inherited from the highlight clip pipeline and
        # made the timeline marker span 10+ seconds, which read as
        # "the player was shooting for 10 s" on the live badge.
        pre = int(_SHOT_PRE_S * fps)
        post = int(_SHOT_POST_S * fps)
        start_f = max(0, frame - pre)
        end_f = frame + post
        if total is not None:
            end_f = min(end_f, total - 1)
        return start_f, end_f

    sf = event.get("start_frame")
    ef = event.get("end_frame")
    if sf is not None and ef is not None:
        return int(sf), int(ef)

    pre = int(_DEFAULT_PRE_S * fps)
    post = int(_DEFAULT_POST_S * fps)
    start_f = max(0, frame - pre)
    end_f = frame + post
    if total is not None:
        end_f = min(end_f, total - 1)
    return start_f, end_f


def _player_clips(ctx: MatchContext) -> dict[str, list[dict[str, float]]]:
    """Per-track contiguous presence ranges (gaps < 1 s merged)."""
    fps = ctx.fps or 30.0
    gap_frames = max(1, int(_PLAYER_CLIP_GAP_S * fps))

    # Collect sorted observation frames per track.
    frames_by_pid: dict[str, list[int]] = defaultdict(list)
    for f_str, items in ctx.player_tracks.items():
        try:
            f = int(f_str)
        except ValueError:
            continue
        for t in items:
            tid = t.get("track_id")
            if tid is None:
                continue
            frames_by_pid[str(tid)].append(f)

    out: dict[str, list[dict[str, float]]] = {}
    for pid, frames in frames_by_pid.items():
        frames.sort()
        clips: list[tuple[int, int]] = []
        run_start = run_end = frames[0]
        for f in frames[1:]:
            if f - run_end <= gap_frames:
                run_end = f
            else:
                clips.append((run_start, run_end))
                run_start = run_end = f
        clips.append((run_start, run_end))
        out[pid] = [
            {
                "start_s": _frame_to_time(s, fps),
                "end_s": _frame_to_time(e, fps),
                "start_frame": s,
                "end_frame": e,
            }
            for s, e in clips
        ]
    return out
