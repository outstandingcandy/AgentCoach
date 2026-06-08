"""FastAPI app: video playback + LLM chat over a single pipeline run.

Two factories live here:

- ``create_app(run_dir, video_path)`` — single-run viewer used by
  ``scripts/run_web_viewer.py`` (kept intact for backwards compatibility).
- ``create_workspace_app(workspace_dir)`` — unified product that hosts the
  viewer, annotator, pipeline console, and library against a workspace
  directory (see ``_workspace.Workspace``). The annotator routes, library
  endpoints, jobs API, and analytics endpoints are attached by their own
  modules so this file stays a thin assembler.
"""

from __future__ import annotations

import logging
import shutil
from collections import Counter
from pathlib import Path
from typing import Any

import json

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from ..highlights._context import MatchContext
from ._runs import CHAT_ARTIFACTS_URL as RUNS_ARTIFACTS_URL, RunRegistry
from ._workspace import Workspace, resolve_workspace
from .analytics import register_analytics_routes
from .chat import ChatEngine
from .jobs import JobManager, register_jobs_routes
from .library import register_library_routes

logger = logging.getLogger(__name__)

STATIC_DIR = Path(__file__).parent / "static"
# Plots / images that run_python emits are stored here and mounted at
# /chat_artifacts/* so the chat UI can render them inline. Cleared on
# app start so old artifacts from a previous viewer session don't
# silently shadow new ones with the same name.
CHAT_ARTIFACTS_URL = "/chat_artifacts"


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    messages: list[ChatMessage]
    current_time: float = 0.0


def create_app(run_dir: Path, video_path: Path | None = None) -> FastAPI:
    run_dir = Path(run_dir).resolve()
    if video_path is not None:
        video_path = Path(video_path).resolve()
        if not video_path.exists():
            raise FileNotFoundError(f"video not found: {video_path}")
    else:
        # Prefer the web-optimized re-encode (small GOP + faststart) when
        # present so seeks/scrubbing are responsive over slow links.
        video_dir = run_dir / "annotated_video"
        web_path = video_dir / "annotated_web.mp4"
        full_path = video_dir / "annotated.mp4"
        if web_path.exists():
            video_path = web_path
        elif full_path.exists():
            video_path = full_path
        else:
            raise FileNotFoundError(f"annotated video not found: {full_path}")

    ctx = MatchContext.from_output_dir(run_dir)

    artifact_dir = run_dir / "chat_artifacts"
    if artifact_dir.exists():
        shutil.rmtree(artifact_dir)
    artifact_dir.mkdir(parents=True, exist_ok=True)

    engine = ChatEngine(
        ctx=ctx,
        artifact_dir=artifact_dir,
        artifact_url_prefix=CHAT_ARTIFACTS_URL,
    )

    app = FastAPI(title="GoalInsight Viewer")

    @app.on_event("shutdown")
    def _shutdown() -> None:
        engine.close()

    @app.get("/")
    def index() -> FileResponse:
        return FileResponse(STATIC_DIR / "index.html")

    @app.get("/video")
    def video() -> FileResponse:
        # FileResponse handles HTTP range requests automatically, so the
        # <video> element can seek.
        return FileResponse(video_path, media_type="video/mp4")

    @app.get("/api/meta")
    def meta() -> JSONResponse:
        teams = Counter(ctx.team_assignments.values())
        event_types = Counter(e.get("type", "unknown") for e in ctx.events)
        return JSONResponse({
            "video_name": ctx.video_path.name,
            "fps": ctx.fps,
            "frame_count": ctx.frame_count,
            "duration_s": ctx.frame_count / ctx.fps if ctx.fps else 0.0,
            "width": ctx.width,
            "height": ctx.height,
            "pitch_length": ctx.pitch_length,
            "pitch_width": ctx.pitch_width,
            "teams": dict(teams),
            "event_counts": dict(event_types),
        })

    @app.post("/api/chat")
    def chat(req: ChatRequest) -> dict[str, Any]:
        try:
            text = engine.respond(
                [m.model_dump() for m in req.messages],
                req.current_time,
            )
        except Exception as exc:  # noqa: BLE001
            logger.exception("chat failed")
            raise HTTPException(status_code=500, detail=str(exc)) from exc
        return {"text": text}

    @app.post("/api/chat/stream")
    def chat_stream(req: ChatRequest) -> StreamingResponse:
        messages = [m.model_dump() for m in req.messages]
        current_time = req.current_time

        def event_source():
            try:
                for delta in engine.stream(messages, current_time):
                    yield f"data: {json.dumps({'delta': delta})}\n\n"
                yield "data: {\"done\": true}\n\n"
            except Exception as exc:  # noqa: BLE001
                logger.exception("chat stream failed")
                yield f"data: {json.dumps({'error': str(exc)})}\n\n"

        return StreamingResponse(
            event_source(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",
            },
        )

    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
    app.mount(
        CHAT_ARTIFACTS_URL,
        StaticFiles(directory=artifact_dir),
        name="chat_artifacts",
    )
    return app


# ---------------------------------------------------------------------------
# Unified workspace app
# ---------------------------------------------------------------------------


def create_workspace_app(
    workspace_dir: str | Path,
    *,
    pitch: "SoccerPitch | None" = None,
) -> FastAPI:
    """Build the unified web app rooted at *workspace_dir*.

    This is the entry point used by ``goalinsight-web``. The skeleton mounts
    static assets and a placeholder index page; route packs (annotator,
    library, jobs, analytics, per-run viewer) are attached by their own
    modules below.

    *pitch* overrides the annotator's default FIFA pitch — required when
    annotations were saved against a non-FIFA pitch (e.g. youth fields
    where ``configs/kids_soccer_physical.yaml`` ships dims like 66.28×43.15).
    """
    ws = resolve_workspace(workspace_dir)

    app = FastAPI(title="GoalInsight")
    app.state.workspace = ws

    # Lazy-init annotator: a single instance backs the /annotate page across
    # videos. Auto-opens the first discovered video when one is present, but
    # does not fail boot if videos_dir is empty (uploads come later).
    from ..annotation.ui import AnchorAnnotator
    from ..annotation.web import (
        VIDEO_EXTS as _ANNOT_VIDEO_EXTS,
        register_annotation_routes,
    )

    annotator = AnchorAnnotator(
        annotations_dir=str(ws.annotations_dir), pitch=pitch,
    )
    initial = sorted(
        [p for ext in _ANNOT_VIDEO_EXTS for p in ws.videos_dir.glob(f"*{ext}")],
        key=lambda p: p.name,
    )
    if initial:
        try:
            annotator.open_video(str(initial[0]), start_frame=0)
        except Exception as exc:  # noqa: BLE001
            logger.warning("annotator: could not auto-open %s: %s", initial[0], exc)
    app.state.annotator = annotator
    register_annotation_routes(
        app, annotator, ws.videos_dir, prefix="/api/annotate",
    )

    register_library_routes(app, ws)

    job_manager = JobManager(ws)
    app.state.jobs = job_manager
    register_jobs_routes(app, job_manager)

    # Browsers cache HTML aggressively; we iterate the UI a lot during
    # development so always force a re-fetch for top-level pages.
    no_cache = {"Cache-Control": "no-store, no-cache, must-revalidate, max-age=0"}

    @app.get("/")
    def shell_page() -> FileResponse:
        path = STATIC_DIR / "shell.html"
        if not path.exists():
            # Pre-shell-page bootstrap: send the user to the viewer until the
            # static pages task lands.
            raise HTTPException(503, "shell.html not yet installed")
        return FileResponse(path, headers=no_cache)

    @app.get("/library")
    def library_page() -> FileResponse:
        path = STATIC_DIR / "library.html"
        if not path.exists():
            raise HTTPException(503, "library.html not yet installed")
        return FileResponse(path, headers=no_cache)

    @app.get("/annotate")
    def annotate_page() -> FileResponse:
        path = STATIC_DIR / "annotate.html"
        if not path.exists():
            raise HTTPException(503, "annotate.html not yet installed")
        return FileResponse(path, headers=no_cache)

    @app.get("/pipeline")
    def pipeline_page() -> FileResponse:
        path = STATIC_DIR / "pipeline.html"
        if not path.exists():
            raise HTTPException(503, "pipeline.html not yet installed")
        return FileResponse(path, headers=no_cache)

    @app.get("/insights/{run_name}")
    def viewer_page(run_name: str) -> FileResponse:
        # Path is just the entry point; client uses run_name to call
        # /api/runs/{run_name}/* endpoints (added by task 3).
        path = STATIC_DIR / "viewer.html"
        if not path.exists():
            # Until we rename index.html → viewer.html we still want a usable
            # fallback so the workspace app boots end-to-end.
            path = STATIC_DIR / "index.html"
        return FileResponse(path, headers=no_cache)

    # Per-run viewer state. ChatEngine boot is expensive (boto3 + Bedrock +
    # MatchContext load); the registry keeps a small LRU of warm runs.
    runs = RunRegistry(ws)
    app.state.runs = runs
    register_analytics_routes(app, runs)

    @app.on_event("shutdown")
    def _shutdown_runs() -> None:
        runs.close_all()

    @app.get("/api/runs/{run_name}/meta")
    def run_meta(run_name: str) -> JSONResponse:
        try:
            handle = runs.get(run_name)
        except FileNotFoundError as exc:
            raise HTTPException(404, str(exc))
        ctx = handle.ctx
        teams = Counter(ctx.team_assignments.values())
        event_types = Counter(e.get("type", "unknown") for e in ctx.events)
        return JSONResponse({
            "run_name": run_name,
            "video_name": ctx.video_path.name,
            "fps": ctx.fps,
            "frame_count": ctx.frame_count,
            "duration_s": ctx.frame_count / ctx.fps if ctx.fps else 0.0,
            "width": ctx.width,
            "height": ctx.height,
            "pitch_length": ctx.pitch_length,
            "pitch_width": ctx.pitch_width,
            "teams": dict(teams),
            "event_counts": dict(event_types),
        })

    @app.get("/api/runs/{run_name}/video")
    def run_video(run_name: str) -> FileResponse:
        try:
            handle = runs.get(run_name)
        except FileNotFoundError as exc:
            raise HTTPException(404, str(exc))
        return FileResponse(handle.video_path, media_type="video/mp4")

    @app.post("/api/runs/{run_name}/chat")
    def run_chat(run_name: str, req: ChatRequest) -> dict[str, Any]:
        try:
            handle = runs.get(run_name)
        except FileNotFoundError as exc:
            raise HTTPException(404, str(exc))
        try:
            text = handle.engine.respond(
                [m.model_dump() for m in req.messages],
                req.current_time,
            )
        except Exception as exc:  # noqa: BLE001
            logger.exception("chat failed (run=%s)", run_name)
            raise HTTPException(status_code=500, detail=str(exc)) from exc
        return {"text": text}

    @app.post("/api/runs/{run_name}/chat/stream")
    def run_chat_stream(run_name: str, req: ChatRequest) -> StreamingResponse:
        try:
            handle = runs.get(run_name)
        except FileNotFoundError as exc:
            raise HTTPException(404, str(exc))
        messages = [m.model_dump() for m in req.messages]
        current_time = req.current_time

        def event_source():
            try:
                for delta in handle.engine.stream(messages, current_time):
                    yield f"data: {json.dumps({'delta': delta})}\n\n"
                yield "data: {\"done\": true}\n\n"
            except Exception as exc:  # noqa: BLE001
                logger.exception("chat stream failed (run=%s)", run_name)
                yield f"data: {json.dumps({'error': str(exc)})}\n\n"

        return StreamingResponse(
            event_source(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    # Mount the workspace-wide chat artifact root at /chat_artifacts so the
    # existing IMG_RE in viewer.html ('/chat_artifacts/...') still matches.
    # ChatEngine writes per-run files under <root>/<run_name>/.
    app.mount(
        RUNS_ARTIFACTS_URL,
        StaticFiles(directory=runs.artifacts_root),
        name="chat_artifacts",
    )
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

    return app
