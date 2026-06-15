"""FastAPI backend for the pitch annotator UI.

The frontend is a single static HTML page served from
goalinsight/annotation/static/, and all interaction goes through JSON
endpoints under ``{prefix}/...`` (default ``/api``) plus a few image
endpoints that return JPEGs.

Public API: ``register_annotation_routes(app, annotator, *, prefix)``
mounts the routes onto an existing FastAPI app — the unified workspace
app uses this so the annotator and the viewer share one process.
"""

from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse, Response

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

STATIC_DIR = Path(__file__).parent / "static"

VIDEO_EXTS = (".mp4", ".mov", ".mkv", ".avi")


def _discover_videos(videos_root: Path) -> list[Path]:
    if not videos_root.is_dir():
        return []
    out: list[Path] = []
    for ext in VIDEO_EXTS:
        out.extend(videos_root.glob(f"*{ext}"))
    return sorted(out, key=lambda p: p.name)


def register_annotation_routes(
    app: FastAPI,
    annotator: AnchorAnnotator,
    videos_root: Path,
    *,
    prefix: str = "/api",
) -> None:
    """Mount annotator JSON + JPEG endpoints onto an existing FastAPI app.

    The annotator state is shared via *annotator*; the caller is responsible
    for opening the initial video. *prefix* lets the unified app expose the
    annotator under ``/api/annotate/...`` to avoid colliding with the
    viewer's ``/api/*`` namespace.
    """
    index = annotator.index
    videos_root_path = Path(videos_root)
    p = prefix.rstrip("/")

    # Auto-apply per-video pitch on every video open. Must run BEFORE
    # _check_pitch_consistency in the annotator so the active pitch matches
    # the saved coords. We wrap open_video at registration time; switch_video
    # delegates to it, so one hook covers both.
    _orig_open_video = annotator.open_video

    def _open_video_with_settings(video_path: str, start_frame: int = 0) -> int:
        stem = Path(video_path).stem
        pitch = per_video_settings.load_pitch(annotator.annotations_dir, stem)
        if pitch:
            try:
                pitch_constants.set_active_pitch(SoccerPitch(**pitch))
            except TypeError:
                # Unknown keys in overrides.yaml — fall through.
                pass
        return _orig_open_video(video_path, start_frame=start_frame)

    annotator.open_video = _open_video_with_settings  # type: ignore[assignment]

    # ------------------------------------------------------------------
    # Read-only metadata
    # ------------------------------------------------------------------

    @app.get(f"{p}/keypoints")
    def keypoints() -> JSONResponse:
        out = []
        for idx, name in INTERSECTON_TO_PITCH_POINTS.items():
            pt = _pk.PITCH_POINTS.get(name)
            if pt is None:
                continue
            out.append({
                "hrnet_index": idx,
                "name": name,
                "world": [float(pt[0]), float(pt[1])],
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
            })
        return JSONResponse(out)

    @app.get(f"{p}/annotated_frames")
    def annotated_frames() -> JSONResponse:
        return JSONResponse(index.get_annotated_frame_stats())

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
        return _ok(annotator.compute_homography())

    @app.post(f"{p}/save")
    async def save() -> JSONResponse:
        return _ok(annotator.save())

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

    @app.get(f"{p}/configs")
    async def list_configs() -> JSONResponse:
        """List config yamls under ./configs that include a
        ``field_registration`` block — these are the ones that make
        sense to bind as the active solver/pitch source."""
        from pathlib import Path as _P
        import yaml as _yaml
        out = []
        for cp in sorted(_P("configs").glob("*.yaml")):
            try:
                doc = _yaml.safe_load(cp.read_text()) or {}
            except (OSError, _yaml.YAMLError):
                continue
            fr = doc.get("field_registration") or {}
            if not isinstance(fr, dict) or not fr.get("backend"):
                # Skip non-pipeline configs (e.g. camera_profiles.yaml,
                # standalone helper docs that don't drive a backend).
                continue
            backend = fr["backend"].lower()
            pitch = doc.get("pitch") or {}
            out.append({
                "path": str(cp),
                "name": cp.name,
                "backend": backend,
                "pitch_length": pitch.get("pitch_length"),
                "pitch_width": pitch.get("pitch_width"),
                "active": (
                    cp.name == getattr(annotator, "_active_config_name", None)
                ),
            })
        return JSONResponse(out)

    @app.post(f"{p}/set_config")
    async def set_config(request: Request) -> JSONResponse:
        payload = await request.json()
        cfg = payload.get("config_path") or payload.get("path")
        if not cfg:
            raise HTTPException(400, "config_path is required")
        msg = annotator.set_solver_config(cfg)
        return _ok(msg)

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
