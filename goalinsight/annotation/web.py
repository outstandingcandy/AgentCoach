"""FastAPI backend for the pitch annotator UI.

All interaction goes through JSON endpoints under ``{prefix}/...``
(workspace app mounts this at ``/api/annotate``) plus a few image
endpoints that return JPEGs. The frontend that drives these routes is
``goalinsight/web/static/annotate.html``.

Public API: ``register_annotation_routes(app, annotator, *, prefix)``
mounts the routes onto an existing FastAPI app — the unified workspace
app uses this so the annotator and the viewer share one process.
"""

from __future__ import annotations

import json
import logging
import re
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, Response

logger = logging.getLogger(__name__)

# Lazy singleton for the pretrained-preview route. Loading HRNet is
# expensive (~1s + GPU mem), and the annotator UI can be opened and
# closed many times per session, so we cache the model across calls.
_KEYPOINT_DETECTOR = None


def _get_pretrained_detector():
    global _KEYPOINT_DETECTOR
    if _KEYPOINT_DETECTOR is None:
        from ..field_registration.keypoint_detector import KeypointDetector

        det = KeypointDetector({
            "backend": "pnlcalib",
            "pnlcalib": {"weights": "SV_kp", "confidence_threshold": 0.3},
        })
        det.load_model()
        _KEYPOINT_DETECTOR = det
    return _KEYPOINT_DETECTOR

from . import per_video_settings, pitch_constants
from .pitch import keypoints as _pk
from .pitch.geometry import SoccerPitch
from .pitch.keypoints import (
    INTERSECTON_TO_PITCH_POINTS,
    NOT_ON_PLANE,
)
from .pitch_constants import get_all_line_names
from .pitch_diagram import get_line_color
from .ui import AnchorAnnotator

VIDEO_EXTS = (".mp4", ".mov", ".mkv", ".avi")


def _discover_videos(videos_root: Path) -> list[Path]:
    if not videos_root.is_dir():
        return []
    out: list[Path] = []
    for ext in VIDEO_EXTS:
        out.extend(videos_root.glob(f"*{ext}"))
    return sorted(out, key=lambda p: p.name)


def video_group_prefix(video_name: str) -> str:
    """Group key for an annotation video — the part before the first ``_``.

    Examples:
        ``sunday_cup_0050-0110``     → ``sunday``
        ``sunday_soccer_10min``      → ``sunday``
        ``football_sunday_full``     → ``football``
        ``kids_soccer_match``        → ``kids``
        ``standalone``               → ``standalone`` (no underscore)

    Used by both the annotate-page UI (collapsible group headers) and
    the finetune launcher (so a "train sunday group" job concatenates
    every annotation directory whose video name shares the prefix).
    """
    head, _, _ = video_name.partition("_")
    return head or video_name


def register_annotation_routes(
    app: FastAPI,
    annotator: AnchorAnnotator,
    videos_root: Path,
    *,
    configs_root: Path | None = None,
    calibrations_root: Path | None = None,
    prefix: str = "/api",
) -> None:
    """Mount annotator JSON + JPEG endpoints onto an existing FastAPI app.

    The annotator state is shared via *annotator*; the caller is responsible
    for opening the initial video. *prefix* lets the unified app expose the
    annotator under ``/api/annotate/...`` to avoid colliding with the
    viewer's ``/api/*`` namespace.

    *configs_root* is the workspace's per-video config directory
    (``workspace/configs/``). When opening a video we look for
    ``<configs_root>/<stem>.yaml`` and, if present, apply it the same
    way the now-gone "Config" dropdown used to — backend selection,
    physical params, pitch, etc. The library page is the canonical
    place to author / edit these.
    """
    index = annotator.index
    videos_root_path = Path(videos_root)
    configs_root_path = Path(configs_root) if configs_root else None
    calibrations_root_path = Path(calibrations_root) if calibrations_root else None
    p = prefix.rstrip("/")

    def _apply_per_video_config(stem: str) -> bool:
        """Re-read ``workspace/configs/<stem>.yaml`` and apply it.

        Called from both the open-video hook and the compute endpoint so
        edits the user makes to the per-video yaml (camera_position,
        focal_hfov, pitch, etc.) take effect on the very next Compute
        click — without forcing them to switch videos to refresh.

        Returns True when a config was found and applied. Failures are
        surfaced via ``annotator._pending_pitch_mismatch`` (rendered as
        the next state response's ``status``) rather than raised, so a
        bad edit doesn't take the page down.
        """
        if configs_root_path is None:
            return False
        cfg_path = configs_root_path / f"{stem}.yaml"
        if not cfg_path.exists():
            return False
        try:
            annotator.set_solver_config(str(cfg_path))
            return True
        except (FileNotFoundError, ValueError) as exc:
            annotator._pending_pitch_mismatch = (
                f"per-video config {cfg_path.name} failed to load: {exc}"
            )
            return False

    # Auto-apply per-video config + pitch on every video open. Must run
    # BEFORE _check_pitch_consistency in the annotator so the active
    # pitch matches the saved coords. We wrap open_video at registration
    # time; switch_video delegates to it, so one hook covers both.
    _orig_open_video = annotator.open_video

    def _open_video_with_settings(video_path: str, start_frame: int = 0) -> int:
        stem = Path(video_path).stem

        # Per-video pipeline config (the file the library page edits)
        # wins. When the user hasn't authored one, fall back to the
        # legacy workspace/annotations/<stem>/overrides.yaml that older
        # builds wrote via the now-gone pitch_type dropdown.
        applied_from_config = _apply_per_video_config(stem)

        if not applied_from_config:
            pitch = per_video_settings.load_pitch(
                annotator.annotations_dir, stem,
            )
            if pitch:
                try:
                    pitch_constants.set_active_pitch(SoccerPitch(**pitch))
                except TypeError:
                    pass

        return _orig_open_video(video_path, start_frame=start_frame)

    annotator.open_video = _open_video_with_settings  # type: ignore[assignment]

    # ------------------------------------------------------------------
    # Read-only metadata
    # ------------------------------------------------------------------

    @app.get(f"{p}/keypoints")
    def keypoints() -> JSONResponse:
        import math as _math
        out = []
        for idx, name in INTERSECTON_TO_PITCH_POINTS.items():
            pt = _pk.PITCH_POINTS.get(name)
            if pt is None:
                continue
            wx, wy = float(pt[0]), float(pt[1])
            # Some tangent-from-corner keypoints don't exist for small
            # pitches (e.g. futsal where corners sit inside the center
            # circle) — SoccerPitch emits NaN; drop them so the JSON
            # serializer doesn't 500.
            if not (_math.isfinite(wx) and _math.isfinite(wy)):
                continue
            out.append({
                "hrnet_index": idx,
                "name": name,
                "world": [wx, wy],
                "is_ground": idx not in NOT_ON_PLANE,
            })
        return JSONResponse(out)

    @app.get(f"{p}/lines")
    def lines() -> JSONResponse:
        out = []
        for name in get_all_line_names():
            (wx1, wy1), (wx2, wy2) = pitch_constants.PITCH_LINES[name]
            b, g, r = get_line_color(name)
            out.append({
                "name": name,
                "world": [[float(wx1), float(wy1)], [float(wx2), float(wy2)]],
                "color": f"rgb({r}, {g}, {b})",
            })
        return JSONResponse(out)

    @app.get(f"{p}/state")
    def state() -> JSONResponse:
        return JSONResponse(annotator.state_dict())

    @app.get(f"{p}/info")
    def info() -> JSONResponse:
        return JSONResponse({
            "videos_root": str(videos_root_path),
            "video_path": annotator.video_path,
            "video_name": annotator.video_name,
            "total_frames": annotator.total_frames,
            "annotations_dir": str(annotator.annotations_dir),
            "active_frame": annotator.current_frame_idx,
        })

    @app.get(f"{p}/videos")
    def videos() -> JSONResponse:
        on_disk = {pp.stem: pp for pp in _discover_videos(videos_root_path)}
        in_index = set(index.get_all_video_names())
        names = sorted(set(on_disk) | in_index)
        out = []
        for name in names:
            path = on_disk.get(name)
            available = path is not None
            frames = index.get_annotated_frames(name)
            out.append({
                "name": name,
                "path": str(path) if path else None,
                "available": available,
                "annotated_count": len(frames),
                "is_active": name == annotator.video_name,
                "group": video_group_prefix(name),
            })
        return JSONResponse(out)

    @app.get(f"{p}/annotated_frames")
    def annotated_frames() -> JSONResponse:
        return JSONResponse(index.get_annotated_frame_stats())

    @app.post(f"{p}/train_group")
    async def train_group(request: Request) -> JSONResponse:
        """Submit a finetune job that trains on every annotated video in a
        given prefix-group at once.

        Body: ``{group: str, kind: "keypoint"|"line" (default keypoint)}``.
        We resolve the group's annotation directories from the index and
        forward them as a comma-separated ``--annotations_dir`` to the
        existing trainer (which already accepts that form).

        The pretrained weight file is auto-picked from
        ``workspace/models/SV_kp.pt`` / ``SV_lines.pt`` — same convention
        the pipeline page uses; users can still hit
        ``POST /api/jobs/train`` directly for fully-explicit submissions.
        """
        manager = getattr(request.app.state, "jobs", None)
        if manager is None:
            raise HTTPException(503, "JobManager not available")

        body = await request.json()
        group = (body.get("group") or "").strip()
        kind = (body.get("kind") or "keypoint").strip()
        if not group:
            raise HTTPException(400, "missing 'group'")
        if kind not in ("keypoint", "line"):
            raise HTTPException(400, f"unknown kind: {kind!r}")

        # Collect on-disk annotation dirs for every video in the group.
        ann_root = Path(annotator.annotations_dir)
        all_video_names = set(index.get_all_video_names())
        group_dirs: list[Path] = []
        for name in sorted(all_video_names):
            if video_group_prefix(name) != group:
                continue
            d = ann_root / name
            if d.is_dir():
                group_dirs.append(d)

        if not group_dirs:
            raise HTTPException(
                404,
                f"no annotation directories found for group '{group}'",
            )

        # Auto-pick the pretrained weights from the workspace mirror.
        ws = getattr(request.app.state, "workspace", None)
        if ws is None:
            raise HTTPException(503, "Workspace not available")
        weight_name = "SV_kp.pt" if kind == "keypoint" else "SV_lines.pt"
        pretrained = ws.models_dir / weight_name
        if not pretrained.is_file():
            raise HTTPException(
                400,
                f"{pretrained} not found — drop the SV_kp.pt / "
                f"SV_lines.pt weights into <workspace>/models/ first.",
            )

        rec = manager.submit_train(
            kind=kind,
            annotations_dir=group_dirs,
            pretrained=pretrained,
        )
        return JSONResponse({
            **rec.to_public(),
            "group": group,
            "dirs": [str(d) for d in group_dirs],
        }, status_code=202)

    @app.get(f"{p}/groups")
    def groups() -> JSONResponse:
        """Aggregate per-prefix-group stats for the UI + finetune button.

        Group key = ``video_group_prefix(stem)`` — the part before the
        first underscore. The returned list mirrors the per-video
        endpoint but rolls up annotation counts across siblings.
        """
        on_disk = {pp.stem: pp for pp in _discover_videos(videos_root_path)}
        in_index = set(index.get_all_video_names())
        names = sorted(set(on_disk) | in_index)

        by_prefix: dict[str, dict] = {}
        for name in names:
            prefix_key = video_group_prefix(name)
            entry = by_prefix.setdefault(prefix_key, {
                "prefix": prefix_key,
                "videos": [],
                "available_videos": 0,
                "annotated_videos": 0,
                "annotated_frames": 0,
                "annotation_dirs": [],
            })
            n_frames = len(index.get_annotated_frames(name))
            available = name in on_disk
            entry["videos"].append(name)
            if available:
                entry["available_videos"] += 1
            if n_frames > 0:
                entry["annotated_videos"] += 1
                entry["annotated_frames"] += n_frames
                # Resolve the on-disk annotation dir for this video so
                # the finetune launcher can pass them all to the trainer
                # as a comma-separated list.
                ann_dir = Path(annotator.annotations_dir) / name
                if ann_dir.is_dir():
                    entry["annotation_dirs"].append(str(ann_dir))
        return JSONResponse(sorted(
            by_prefix.values(), key=lambda g: g["prefix"]
        ))

    # ------------------------------------------------------------------
    # Image endpoints
    # ------------------------------------------------------------------

    @app.get(f"{p}/frame.jpg")
    def frame_jpg(overlay: int = 1) -> Response:
        try:
            data = annotator.render_frame_jpeg(with_overlay=bool(overlay))
        except RuntimeError as exc:
            raise HTTPException(500, str(exc))
        return Response(content=data, media_type="image/jpeg",
                        headers={"Cache-Control": "no-store"})

    @app.get(f"{p}/pretrained_preview.jpg")
    def pretrained_preview_jpg() -> Response:
        """Overlay pretrained HRNet keypoint detections on the current frame.

        Used by the scene-setup wizard's "preview calibration" step so
        the user can eyeball whether SV_kp weights already work on
        their pitch style, before deciding to annotate for a fine-tune
        or use the pretrained model as-is. Runs the same detector the
        physical / pnlcalib backends do at pipeline time.
        """
        frame = annotator.current_frame
        if frame is None:
            raise HTTPException(400, "no frame loaded — open a video first")
        try:
            det = _get_pretrained_detector()
        except Exception as exc:  # noqa: BLE001
            logger.exception("failed to init keypoint detector")
            raise HTTPException(500, f"detector init failed: {exc}") from exc

        try:
            keypoints = det.detect(frame, convert_to_soccernet=True)
        except Exception as exc:  # noqa: BLE001
            logger.exception("keypoint detection failed")
            raise HTTPException(500, f"detection failed: {exc}") from exc

        vis = frame.copy()
        for kp in keypoints:
            x = kp.get("x")
            y = kp.get("y")
            conf = float(kp.get("confidence", 0.0))
            if x is None or y is None:
                continue
            cv2.circle(vis, (int(round(x)), int(round(y))), 6, (0, 255, 255), -1)
            cv2.circle(vis, (int(round(x)), int(round(y))), 8, (0, 0, 0), 1)
            label = kp.get("name") or str(kp.get("id", ""))
            if label:
                cv2.putText(
                    vis, str(label),
                    (int(round(x)) + 8, int(round(y)) - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1, cv2.LINE_AA,
                )
        ok, buf = cv2.imencode(".jpg", vis, [cv2.IMWRITE_JPEG_QUALITY, 85])
        if not ok:
            raise HTTPException(500, "jpeg encode failed")
        return Response(
            content=bytes(buf),
            media_type="image/jpeg",
            headers={
                "Cache-Control": "no-store",
                "X-Keypoints-Detected": str(len(keypoints)),
            },
        )

    @app.get(f"{p}/pitch.jpg")
    def pitch_jpg(highlight_keypoint: str | None = None,
                  highlight_line: str | None = None) -> Response:
        data = annotator.render_pitch_diagram_jpeg(
            highlight_keypoint=highlight_keypoint,
            highlight_line=highlight_line,
        )
        return Response(content=data, media_type="image/jpeg",
                        headers={"Cache-Control": "no-store"})

    @app.get(f"{p}/tactical.jpg")
    def tactical_jpg() -> Response:
        # NB: we used to re-read the per-video config here on every
        # GET, but ``set_solver_config`` invalidates H0 + the solved
        # cam position as a side effect — so any periodic poll of this
        # endpoint silently wiped a fresh solve. The config is now
        # re-read in two narrower spots: on /api/annotate/jump (page
        # load / video switch) and on /api/annotate/compute (the user
        # clicked the button). Between solves the tactical view shows
        # the prior the user just saved, which is what they want when
        # iterating on camera_position.
        data = annotator.render_tactical_jpeg()
        return Response(content=data, media_type="image/jpeg",
                        headers={"Cache-Control": "no-store"})

    @app.get(f"{p}/lines.jpg")
    def lines_jpg(highlight_line: str | None = None) -> Response:
        data = annotator.render_lines_diagram_jpeg(highlight_line=highlight_line)
        return Response(content=data, media_type="image/jpeg",
                        headers={"Cache-Control": "no-store"})

    @app.get(f"{p}/pitch_layout")
    def pitch_layout() -> JSONResponse:
        from .pitch_diagram import compute_pitch_layout, compute_lines_layout
        return JSONResponse({
            "pitch": compute_pitch_layout(),
            "lines": compute_lines_layout(),
        })

    # ------------------------------------------------------------------
    # Mutating actions
    # ------------------------------------------------------------------

    def _ok(message: str = "") -> JSONResponse:
        body = annotator.state_dict()
        body["status"] = message
        return JSONResponse(body)

    @app.post(f"{p}/goto")
    async def goto(request: Request) -> JSONResponse:
        payload = await request.json()
        idx = int(payload.get("frame_idx", 0))
        if not annotator.goto_frame(idx):
            raise HTTPException(400, f"Invalid frame_idx: {idx}")
        return _ok(f"Switched to frame {idx}")

    @app.post(f"{p}/jump")
    async def jump(request: Request) -> JSONResponse:
        payload = await request.json()
        video_name = payload.get("video_name")
        frame_idx = payload.get("frame_idx")
        if not video_name:
            raise HTTPException(400, "video_name is required")

        on_disk = {pp.stem: pp for pp in _discover_videos(videos_root_path)}
        path = on_disk.get(video_name)
        if path is None:
            raise HTTPException(
                400,
                f"Video '{video_name}' not found in {videos_root_path}",
            )

        target_frame = int(frame_idx) if frame_idx is not None else 0
        try:
            if video_name == annotator.video_name:
                # Same video → no open_video, so no auto-apply hook
                # fires. Re-read the per-video config here so a user
                # who edited workspace/configs/<stem>.yaml and reloaded
                # the page still gets the latest pitch / backend.
                _apply_per_video_config(video_name)
                if not annotator.goto_frame(target_frame):
                    raise HTTPException(400, f"Invalid frame_idx: {target_frame}")
            else:
                annotator.switch_video(str(path), frame_idx=target_frame)
        except RuntimeError as exc:
            raise HTTPException(400, str(exc))
        return _ok(f"Jumped to {video_name} frame {annotator.current_frame_idx}")

    @app.post(f"{p}/mode")
    async def mode(request: Request) -> JSONResponse:
        payload = await request.json()
        try:
            annotator.set_mode(payload.get("mode", "point"))
        except ValueError as exc:
            raise HTTPException(400, str(exc))
        return _ok(f"Mode: {annotator.annotation_mode}")

    @app.post(f"{p}/click")
    async def click(request: Request) -> JSONResponse:
        payload = await request.json()
        msg = annotator.click(float(payload["x"]), float(payload["y"]))
        return _ok(msg)

    @app.post(f"{p}/add_keypoint")
    async def add_keypoint(request: Request) -> JSONResponse:
        payload = await request.json()
        msg = annotator.add_keypoint(payload["name"])
        return _ok(msg)

    @app.post(f"{p}/add_line")
    async def add_line(request: Request) -> JSONResponse:
        payload = await request.json()
        msg = annotator.add_line(payload["name"])
        return _ok(msg)

    @app.post(f"{p}/undo")
    async def undo() -> JSONResponse:
        return _ok(annotator.undo())

    @app.post(f"{p}/reset")
    async def reset() -> JSONResponse:
        annotator.reset()
        return _ok("All annotations reset")

    @app.post(f"{p}/compute")
    async def compute() -> JSONResponse:
        # Re-read the per-video config on every Compute click so users
        # who tweak camera_position / focal_hfov / pitch in their
        # workspace/configs/<stem>.yaml see the result without having to
        # re-open the video. Cheap (yaml parse + dict diff); the active
        # pitch and solver backend stay in sync with disk.
        if annotator.video_name:
            _apply_per_video_config(annotator.video_name)
        return _ok(annotator.compute_homography())

    @app.post(f"{p}/save")
    async def save() -> JSONResponse:
        msg = annotator.save()
        # After a successful save, thread the frame path into the
        # per-video pipeline config so the fixed_camera runner can replay
        # that pose. Only when the wizard's mode produced a
        # ``lock_camera_position: true`` config (fixed rig) — otherwise
        # PTZ / model users would get their config unexpectedly rewritten.
        try:
            _thread_annotation_into_config()
        except Exception as exc:  # noqa: BLE001
            logger.warning("failed to update per-video config: %s", exc)
        return _ok(msg)

    @app.post(f"{p}/save_preset")
    async def save_preset(request: Request) -> JSONResponse:
        """Persist the just-computed camera pose as a reusable calibration
        preset under ``workspace/calibrations/<name>.json``.

        For a fixed rig the annotate page's solved pose (K/rvec/tvec/dist,
        held on ``annotator._solved_camera`` after Compute) is the ground
        truth. Saving it lets the pipeline REUSE it verbatim — and lets
        other videos shot with the same camera/position pick it in the
        wizard — instead of re-solving PnP every run.
        """
        if calibrations_root_path is None:
            raise HTTPException(500, "calibrations directory not configured")
        cam = getattr(annotator, "_solved_camera", None)
        if not cam:
            raise HTTPException(
                400, "No computed camera yet — click Compute homography first.",
            )
        body = {}
        try:
            body = await request.json()
        except Exception:  # noqa: BLE001 — empty/invalid body is fine
            body = {}
        raw_name = (body.get("name") or "").strip()
        if not raw_name:
            raise HTTPException(400, "Preset name is required.")
        # Sanitise to a safe filename stem.
        name = re.sub(r"[^A-Za-z0-9_.-]+", "_", raw_name).strip("_.")
        if not name:
            raise HTTPException(400, f"Invalid preset name: {raw_name!r}")

        # Image size the pose was solved at (pose intrinsics are pixel-valued).
        if annotator.current_frame is not None:
            h, w = annotator.current_frame.shape[:2]
            img_size = [int(w), int(h)]
        else:
            img_size = None

        def _tolist(v):
            arr = np.asarray(v, dtype=np.float64)
            return arr.ravel().tolist() if arr.ndim == 1 else arr.tolist()

        K = np.asarray(cam["K"], dtype=np.float64)
        rvec = np.asarray(cam["rvec"], dtype=np.float64).reshape(3)
        tvec = np.asarray(cam["tvec"], dtype=np.float64).reshape(3)
        R = cv2.Rodrigues(rvec)[0]
        pos = (-R.T @ tvec).ravel()
        # Snapshot the pitch dimensions the pose was solved against, so the
        # preset is self-contained: the pose and the pitch geometry it
        # depends on stay locked together even if the named pitch profile is
        # later edited. Read from the live active pitch (authoritative — it's
        # what the solve used).
        ap = pitch_constants.get_active_pitch()
        pitch_snapshot = {
            "pitch_length": float(ap.PITCH_LENGTH),
            "pitch_width": float(ap.PITCH_WIDTH),
            "penalty_area_width": float(ap.PENALTY_AREA_WIDTH),
            "penalty_area_length": float(ap.PENALTY_AREA_LENGTH),
            "goal_area_width": float(ap.GOAL_AREA_WIDTH),
            "goal_area_length": float(ap.GOAL_AREA_LENGTH),
            "goal_line_to_penalty_mark": float(ap.GOAL_LINE_TO_PENALTY_MARK),
            "center_circle_radius": float(ap.CENTER_CIRCLE_RADIUS),
            "goal_height": float(ap.GOAL_HEIGHT),
            "goal_length": float(ap.GOAL_LENGTH),
        }
        pa_shape = getattr(ap, "PENALTY_AREA_SHAPE", None)
        if pa_shape:
            pitch_snapshot["penalty_area_shape"] = str(pa_shape)
        preset = {
            "name": name,
            "pitch_type": annotator._active_pitch_type,
            "pitch": pitch_snapshot,
            "image_size": img_size,
            "reprojection_error": float(annotator.reprojection_error),
            "created_at": datetime.now().isoformat(),
            "source_video": annotator.video_name or None,
            # ``pose`` mirrors the camera_poses.json single-frame schema so the
            # fixed_camera runner can load it with zero conversion.
            "pose": {
                "K": K.tolist(),
                "dist_coeffs": _tolist(cam.get("dist", np.zeros(5))),
                "rvec": rvec.tolist(),
                "tvec": tvec.tolist(),
                "camera_position": {
                    "x": float(pos[0]), "y": float(pos[1]), "z": float(pos[2]),
                },
                "focal_length": float(K[0, 0]),
            },
        }
        calibrations_root_path.mkdir(parents=True, exist_ok=True)
        out_path = calibrations_root_path / f"{name}.json"
        out_path.write_text(json.dumps(preset, indent=2, ensure_ascii=False))
        logger.info("saved calibration preset %s (reproj=%.2f px)",
                    out_path.name, preset["reprojection_error"])
        return _ok(f"Saved calibration preset '{name}'")

    def _thread_annotation_into_config() -> None:
        if configs_root_path is None:
            return
        video_name = getattr(annotator, "video_name", None)
        if not video_name:
            return
        stem = Path(str(video_name)).stem
        cfg_path = configs_root_path / f"{stem}.yaml"
        if not cfg_path.exists():
            return
        import yaml
        cfg = yaml.safe_load(cfg_path.read_text()) or {}
        phys = (cfg.get("field_registration") or {}).get("physical") or {}
        # Only threaded when the wizard picked "fixed rig" mode.
        if not phys.get("lock_camera_position"):
            return
        # Pick the most-recently-saved frame_*.json in the annotations dir.
        anns_dir = Path(annotator.annotations_dir) / stem
        if not anns_dir.is_dir():
            return
        candidates = sorted(
            (p for p in anns_dir.glob("frame_*.json")
             if not p.name.endswith("_all_points.json")),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        if not candidates:
            return
        latest = str(candidates[0].resolve())
        if phys.get("annotation_frame_path") == latest:
            return  # Nothing to do — already in sync.
        cfg.setdefault("field_registration", {}).setdefault("physical", {})[
            "annotation_frame_path"
        ] = latest
        cfg_path.write_text(yaml.safe_dump(cfg, sort_keys=False))
        logger.info(
            "annotation frame saved — wrote annotation_frame_path=%s to %s",
            latest, cfg_path.name,
        )

    @app.post(f"{p}/derived/toggle")
    async def derived_toggle(request: Request) -> JSONResponse:
        payload = await request.json()
        msg = annotator.toggle_derived(int(payload["idx"]))
        return _ok(msg)

    @app.post(f"{p}/derived/accept_all")
    async def derived_accept_all() -> JSONResponse:
        return _ok(annotator.accept_all_derived())

    @app.post(f"{p}/derived/reject_all")
    async def derived_reject_all() -> JSONResponse:
        return _ok(annotator.reject_all_derived())

    @app.post(f"{p}/auto/toggle")
    async def auto_toggle(request: Request) -> JSONResponse:
        payload = await request.json()
        msg = annotator.toggle_auto(int(payload["idx"]))
        return _ok(msg)

    @app.post(f"{p}/auto/accept_all")
    async def auto_accept_all() -> JSONResponse:
        return _ok(annotator.accept_all_auto())

    @app.post(f"{p}/auto/reject_all")
    async def auto_reject_all() -> JSONResponse:
        return _ok(annotator.reject_all_auto())

    @app.post(f"{p}/manual/select")
    async def manual_select(request: Request) -> JSONResponse:
        payload = await request.json()
        idx = payload.get("idx")
        return _ok(annotator.select_manual_point(None if idx is None else int(idx)))

    @app.post(f"{p}/manual/delete")
    async def manual_delete(request: Request) -> JSONResponse:
        payload = await request.json()
        return _ok(annotator.delete_manual_point(int(payload["idx"])))

    @app.post(f"{p}/manual/update_pixel")
    async def manual_update_pixel(request: Request) -> JSONResponse:
        payload = await request.json()
        return _ok(annotator.update_manual_pixel(
            int(payload["idx"]),
            float(payload["x"]),
            float(payload["y"]),
        ))

    @app.post(f"{p}/manual/update_name")
    async def manual_update_name(request: Request) -> JSONResponse:
        payload = await request.json()
        return _ok(annotator.update_manual_name(
            int(payload["idx"]),
            str(payload["name"]),
        ))

    @app.post(f"{p}/derived/promote")
    async def derived_promote(request: Request) -> JSONResponse:
        payload = await request.json()
        return _ok(annotator.promote_derived_to_manual(int(payload["idx"])))

    @app.post(f"{p}/auto/promote")
    async def auto_promote(request: Request) -> JSONResponse:
        payload = await request.json()
        return _ok(annotator.promote_auto_to_manual(int(payload["idx"])))

    @app.post(f"{p}/line/select")
    async def line_select(request: Request) -> JSONResponse:
        payload = await request.json()
        idx = payload.get("idx")
        return _ok(annotator.select_line(None if idx is None else int(idx)))

    @app.post(f"{p}/line/delete")
    async def line_delete(request: Request) -> JSONResponse:
        payload = await request.json()
        return _ok(annotator.delete_line(int(payload["idx"])))

    @app.post(f"{p}/projection")
    async def set_projection(request: Request) -> JSONResponse:
        payload = await request.json()
        return _ok(annotator.set_show_projection(bool(payload.get("show", True))))
