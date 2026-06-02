"""FastAPI backend for the pitch annotator UI.

Replaces the previous Gradio app. The frontend is a single static HTML page
served from goalinsight/annotation/static/, and all interaction goes through
JSON endpoints under /api/* (plus a few image endpoints that return JPEGs).
"""

from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse, Response

from . import pitch_constants
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


def create_app(
    videos_root: str,
    annotations_dir: str,
    start_frame: int = 0,
    pitch: SoccerPitch | None = None,
) -> FastAPI:
    videos_root_path = Path(videos_root)
    annotator = AnchorAnnotator(annotations_dir=annotations_dir, pitch=pitch)
    # Share the annotator's index — a separate instance would never see
    # frames added by ``_save_frame_annotation`` because each AnnotationIndex
    # caches its own copy of index.json in memory.
    index = annotator.index

    discovered = _discover_videos(videos_root_path)
    if not discovered:
        raise RuntimeError(
            f"No videos found in --videos-root: {videos_root_path}",
        )
    first_video = discovered[0]
    annotator.open_video(str(first_video), start_frame=start_frame)

    app = FastAPI(title="Soccer Pitch Annotator")

    # ------------------------------------------------------------------
    # Static files
    # ------------------------------------------------------------------

    @app.get("/")
    def index_page() -> FileResponse:
        return FileResponse(STATIC_DIR / "index.html")

    @app.get("/static/{filename}")
    def static_file(filename: str) -> FileResponse:
        path = STATIC_DIR / filename
        if not path.exists():
            raise HTTPException(404, "Not found")
        return FileResponse(path)

    # ------------------------------------------------------------------
    # Read-only metadata
    # ------------------------------------------------------------------

    @app.get("/api/keypoints")
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

    @app.get("/api/lines")
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

    @app.get("/api/state")
    def state() -> JSONResponse:
        return JSONResponse(annotator.state_dict())

    @app.get("/api/info")
    def info() -> JSONResponse:
        return JSONResponse({
            "videos_root": str(videos_root_path),
            "video_path": annotator.video_path,
            "video_name": annotator.video_name,
            "total_frames": annotator.total_frames,
            "annotations_dir": str(annotations_dir),
            "active_frame": annotator.current_frame_idx,
        })

    @app.get("/api/videos")
    def videos() -> JSONResponse:
        on_disk = {p.stem: p for p in _discover_videos(videos_root_path)}
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

    @app.get("/api/annotated_frames")
    def annotated_frames() -> JSONResponse:
        return JSONResponse(index.get_annotated_frame_stats())

    # ------------------------------------------------------------------
    # Image endpoints
    # ------------------------------------------------------------------

    @app.get("/api/frame.jpg")
    def frame_jpg(overlay: int = 1) -> Response:
        try:
            data = annotator.render_frame_jpeg(with_overlay=bool(overlay))
        except RuntimeError as exc:
            raise HTTPException(500, str(exc))
        return Response(content=data, media_type="image/jpeg",
                        headers={"Cache-Control": "no-store"})

    @app.get("/api/pitch.jpg")
    def pitch_jpg(highlight_keypoint: str | None = None,
                  highlight_line: str | None = None) -> Response:
        data = annotator.render_pitch_diagram_jpeg(
            highlight_keypoint=highlight_keypoint,
            highlight_line=highlight_line,
        )
        return Response(content=data, media_type="image/jpeg",
                        headers={"Cache-Control": "no-store"})

    @app.get("/api/tactical.jpg")
    def tactical_jpg() -> Response:
        data = annotator.render_tactical_jpeg()
        return Response(content=data, media_type="image/jpeg",
                        headers={"Cache-Control": "no-store"})

    @app.get("/api/lines.jpg")
    def lines_jpg(highlight_line: str | None = None) -> Response:
        data = annotator.render_lines_diagram_jpeg(highlight_line=highlight_line)
        return Response(content=data, media_type="image/jpeg",
                        headers={"Cache-Control": "no-store"})

    # ------------------------------------------------------------------
    # Mutating actions
    # ------------------------------------------------------------------

    def _ok(message: str = "") -> JSONResponse:
        body = annotator.state_dict()
        body["status"] = message
        return JSONResponse(body)

    @app.post("/api/goto")
    async def goto(request: Request) -> JSONResponse:
        payload = await request.json()
        idx = int(payload.get("frame_idx", 0))
        if not annotator.goto_frame(idx):
            raise HTTPException(400, f"Invalid frame_idx: {idx}")
        return _ok(f"Switched to frame {idx}")

    @app.post("/api/jump")
    async def jump(request: Request) -> JSONResponse:
        payload = await request.json()
        video_name = payload.get("video_name")
        frame_idx = payload.get("frame_idx")
        if not video_name:
            raise HTTPException(400, "video_name is required")

        on_disk = {p.stem: p for p in _discover_videos(videos_root_path)}
        path = on_disk.get(video_name)
        if path is None:
            raise HTTPException(
                400,
                f"Video '{video_name}' not found in {videos_root_path}",
            )

        target_frame = int(frame_idx) if frame_idx is not None else 0
        try:
            if video_name == annotator.video_name:
                if target_frame and not annotator.goto_frame(target_frame):
                    raise HTTPException(400, f"Invalid frame_idx: {target_frame}")
            else:
                annotator.switch_video(str(path), frame_idx=target_frame)
        except RuntimeError as exc:
            raise HTTPException(400, str(exc))
        return _ok(f"Jumped to {video_name} frame {annotator.current_frame_idx}")

    @app.post("/api/mode")
    async def mode(request: Request) -> JSONResponse:
        payload = await request.json()
        try:
            annotator.set_mode(payload.get("mode", "point"))
        except ValueError as exc:
            raise HTTPException(400, str(exc))
        return _ok(f"Mode: {annotator.annotation_mode}")

    @app.post("/api/click")
    async def click(request: Request) -> JSONResponse:
        payload = await request.json()
        msg = annotator.click(float(payload["x"]), float(payload["y"]))
        return _ok(msg)

    @app.post("/api/add_keypoint")
    async def add_keypoint(request: Request) -> JSONResponse:
        payload = await request.json()
        msg = annotator.add_keypoint(payload["name"])
        return _ok(msg)

    @app.post("/api/add_line")
    async def add_line(request: Request) -> JSONResponse:
        payload = await request.json()
        msg = annotator.add_line(payload["name"])
        return _ok(msg)

    @app.post("/api/undo")
    async def undo() -> JSONResponse:
        return _ok(annotator.undo())

    @app.post("/api/reset")
    async def reset() -> JSONResponse:
        annotator.reset()
        return _ok("All annotations reset")

    @app.post("/api/compute")
    async def compute() -> JSONResponse:
        return _ok(annotator.compute_homography())

    @app.post("/api/save")
    async def save() -> JSONResponse:
        return _ok(annotator.save())

    # Confirm / reject for derived points -------------------------------

    @app.post("/api/derived/toggle")
    async def derived_toggle(request: Request) -> JSONResponse:
        payload = await request.json()
        msg = annotator.toggle_derived(int(payload["idx"]))
        return _ok(msg)

    @app.post("/api/derived/accept_all")
    async def derived_accept_all() -> JSONResponse:
        return _ok(annotator.accept_all_derived())

    @app.post("/api/derived/reject_all")
    async def derived_reject_all() -> JSONResponse:
        return _ok(annotator.reject_all_derived())

    # Confirm / reject for auto-projected points ------------------------

    @app.post("/api/auto/toggle")
    async def auto_toggle(request: Request) -> JSONResponse:
        payload = await request.json()
        msg = annotator.toggle_auto(int(payload["idx"]))
        return _ok(msg)

    @app.post("/api/auto/accept_all")
    async def auto_accept_all() -> JSONResponse:
        return _ok(annotator.accept_all_auto())

    @app.post("/api/auto/reject_all")
    async def auto_reject_all() -> JSONResponse:
        return _ok(annotator.reject_all_auto())

    # Selection + deletion of manual points / lines ---------------------

    @app.post("/api/manual/select")
    async def manual_select(request: Request) -> JSONResponse:
        payload = await request.json()
        idx = payload.get("idx")
        return _ok(annotator.select_manual_point(None if idx is None else int(idx)))

    @app.post("/api/manual/delete")
    async def manual_delete(request: Request) -> JSONResponse:
        payload = await request.json()
        return _ok(annotator.delete_manual_point(int(payload["idx"])))

    @app.post("/api/line/select")
    async def line_select(request: Request) -> JSONResponse:
        payload = await request.json()
        idx = payload.get("idx")
        return _ok(annotator.select_line(None if idx is None else int(idx)))

    @app.post("/api/line/delete")
    async def line_delete(request: Request) -> JSONResponse:
        payload = await request.json()
        return _ok(annotator.delete_line(int(payload["idx"])))

    @app.post("/api/projection")
    async def set_projection(request: Request) -> JSONResponse:
        payload = await request.json()
        return _ok(annotator.set_show_projection(bool(payload.get("show", True))))

    return app


def run_server(
    videos_root: str,
    annotations_dir: str,
    host: str = "127.0.0.1",
    port: int = 7860,
    start_frame: int = 0,
    pitch: SoccerPitch | None = None,
) -> None:
    import uvicorn

    app = create_app(
        videos_root,
        annotations_dir,
        start_frame=start_frame,
        pitch=pitch,
    )
    uvicorn.run(app, host=host, port=port, log_level="info")
