"""FastAPI app: video playback + LLM chat over a single pipeline run."""

from __future__ import annotations

import logging
from collections import Counter
from pathlib import Path
from typing import Any

import json

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from ..highlights._context import MatchContext
from .chat import ChatEngine

logger = logging.getLogger(__name__)

STATIC_DIR = Path(__file__).parent / "static"


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
    engine = ChatEngine(ctx=ctx)

    app = FastAPI(title="GoalInsight Viewer")

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
    return app
