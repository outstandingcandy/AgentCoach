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
from ._runs import RunHandle, RunRegistry
from .match_tools import _frame_to_time

# Default ± window for events without their own start_frame/end_frame.
_DEFAULT_PRE_S = 2.0
_DEFAULT_POST_S = 2.0
# Goal/shot windows lean on highlights' build-up + celebration logic.
_BUILDUP_MAX_S = 8.0
_BUILDUP_PAD_S = 1.5
_CELEBRATION_S = 4.0
# Two presence observations within this gap are treated as the same clip.
_PLAYER_CLIP_GAP_S = 1.0


def register_match_detail_routes(app: FastAPI, runs: RunRegistry) -> None:
    @app.get("/api/runs/{run_name}/match/data")
    def match_data(run_name: str) -> JSONResponse:
        try:
            handle = runs.get(run_name)
        except FileNotFoundError as exc:
            raise HTTPException(404, str(exc))
        return JSONResponse(_build_payload(handle))


# ---------------------------------------------------------------------------
# Payload builder (cached on the RunHandle)
# ---------------------------------------------------------------------------


def _build_payload(handle: RunHandle) -> dict[str, Any]:
    cached = getattr(handle, "_match_detail_payload", None)
    if cached is not None:
        return cached
    ctx = handle.ctx
    payload = {
        "fps": ctx.fps,
        "frame_count": ctx.frame_count,
        "width": ctx.width,
        "height": ctx.height,
        "pitch_length": ctx.pitch_length,
        "pitch_width": ctx.pitch_width,
        "pitch": _pitch_geometry(ctx),
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
    handle._match_detail_payload = payload  # type: ignore[attr-defined]
    return payload


def _pitch_geometry(ctx: MatchContext) -> dict[str, float]:
    """Active pitch dimensions for drawing the top-down panel client-side.

    Falls back to FIFA defaults scaled by the context's pitch dims when
    the active SoccerPitch hasn't been overridden (e.g. workspace
    booted without a ``--pitch-config`` flag).
    """
    from ..annotation import pitch_constants
    p = pitch_constants.get_active_pitch()
    # Prefer the per-run dimensions stored in MatchContext (calibration
    # metadata) over the process-global active pitch when they disagree;
    # the active pitch is shared across runs but a ctx may legitimately
    # describe a different size.
    L = ctx.pitch_length or p.PITCH_LENGTH
    W = ctx.pitch_width or p.PITCH_WIDTH
    sx = L / p.PITCH_LENGTH if p.PITCH_LENGTH else 1.0
    sy = W / p.PITCH_WIDTH if p.PITCH_WIDTH else 1.0
    return {
        "length": L,
        "width": W,
        "penalty_area_length": p.PENALTY_AREA_LENGTH * sx,
        "penalty_area_width": p.PENALTY_AREA_WIDTH * sy,
        "goal_area_length": p.GOAL_AREA_LENGTH * sx,
        "goal_area_width": p.GOAL_AREA_WIDTH * sy,
        "center_circle_radius": p.CENTER_CIRCLE_RADIUS * min(sx, sy),
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
        }
    return out


def _abs_url(base: str, rel: str | None) -> str | None:
    if not rel:
        return None
    return f"{base}/{rel.lstrip('/')}"


# ---------------------------------------------------------------------------
# Per-frame camera FOV polygon on the ground plane.
# ---------------------------------------------------------------------------


# Image-edge sampling: 8 points per side gives a smooth polygon on the
# ground without ballooning the payload. With ~200 keyed frames per
# kids run that's 200 × 32 × 2 floats ≈ ~12 kB after gzip.
_FOV_POINTS_PER_SIDE = 8
# Drop ground hits more than this many metres outside the pitch — the
# unprojection of points near the horizon line goes to enormous values
# and would distort the auto-zoom on the top-down panel.
_FOV_GROUND_CLAMP_M = 60.0


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
        out.append({"frame": frame, "polygon": poly})
    out.sort(key=lambda x: x["frame"])
    return {"frames": out}


def _project_image_boundary_to_ground(
    pose: dict[str, Any], boundary_img: list[tuple[float, float]],
) -> list[list[float]] | None:
    """Back-project each image-boundary point to the z=0 ground plane.

    Mirrors ``utils/pitch.py:_img_to_world`` but in pure Python (no
    OpenCV dep here — distortion coefficients are zero for the physical
    backend kids configs use, so undistortPoints reduces to the linear
    pinhole model and we can do the math inline).
    """
    K = pose.get("K")
    if K is None:
        return None
    rvec = pose.get("rvec")
    tvec = pose.get("tvec")
    if rvec is None or tvec is None:
        return None

    fx = float(K[0][0])
    fy = float(K[1][1])
    cx = float(K[0][2])
    cy = float(K[1][2])

    # rvec → R via Rodrigues. Pure-Python implementation; rvec is small
    # so accuracy is fine. Skip OpenCV to keep this module light.
    R = _rodrigues(rvec)
    Rt = _transpose3(R)
    tv = [float(tvec[0]), float(tvec[1]), float(tvec[2])]
    cam_center = _mat_vec(Rt, [-tv[0], -tv[1], -tv[2]])

    # Distortion coefficients aren't applied — physical backend uses
    # zero distortion, and pnlcalib's residual is small enough that the
    # polygon shape is dominated by R/t. If a future backend ships
    # non-zero distortion this is the place to plug it in.

    poly: list[list[float]] = []
    for ix, iy in boundary_img:
        xn = (ix - cx) / fx
        yn = (iy - cy) / fy
        ray_world = _mat_vec(Rt, [xn, yn, 1.0])
        if abs(ray_world[2]) < 1e-9:
            continue
        t = -cam_center[2] / ray_world[2]
        if t < 0:
            continue
        wx = cam_center[0] + t * ray_world[0]
        wy = cam_center[1] + t * ray_world[1]
        if abs(wx) > _FOV_GROUND_CLAMP_M or abs(wy) > _FOV_GROUND_CLAMP_M:
            # Image points near the horizon back-project to enormous
            # ground coordinates. Clamp the ray to the boundary ring at
            # the configured radius so the polygon stays closed.
            r = _FOV_GROUND_CLAMP_M / max(abs(wx), abs(wy))
            wx *= r
            wy *= r
        poly.append([round(wx, 3), round(wy, 3)])
    return poly


def _rodrigues(rvec: list[float]) -> list[list[float]]:
    """Rotation matrix from a 3-vector axis-angle (Rodrigues' formula)."""
    rx, ry, rz = float(rvec[0]), float(rvec[1]), float(rvec[2])
    theta = math.sqrt(rx * rx + ry * ry + rz * rz)
    if theta < 1e-12:
        return [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
    kx, ky, kz = rx / theta, ry / theta, rz / theta
    c = math.cos(theta)
    s = math.sin(theta)
    C = 1.0 - c
    return [
        [c + kx * kx * C,        kx * ky * C - kz * s, kx * kz * C + ky * s],
        [ky * kx * C + kz * s,   c + ky * ky * C,      ky * kz * C - kx * s],
        [kz * kx * C - ky * s,   kz * ky * C + kx * s, c + kz * kz * C],
    ]


def _transpose3(M: list[list[float]]) -> list[list[float]]:
    return [[M[j][i] for j in range(3)] for i in range(3)]


def _mat_vec(M: list[list[float]], v: list[float]) -> list[float]:
    return [
        M[0][0] * v[0] + M[0][1] * v[1] + M[0][2] * v[2],
        M[1][0] * v[0] + M[1][1] * v[1] + M[1][2] * v[2],
        M[2][0] * v[0] + M[2][1] * v[1] + M[2][2] * v[2],
    ]


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
        md = event.get("metadata") or {}
        side = md.get("goal_side") or md.get("side") or "right"
        try:
            start_f = find_buildup_start(
                ctx, frame, side,
                max_seconds=_BUILDUP_MAX_S,
                padding_seconds=_BUILDUP_PAD_S,
            )
        except Exception:  # noqa: BLE001
            start_f = max(0, frame - int(_BUILDUP_MAX_S * fps))
        end_f = find_celebration_end(
            frame, fps,
            duration_seconds=_CELEBRATION_S,
            total_frames=total,
        )
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
