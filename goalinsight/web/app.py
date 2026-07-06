"""FastAPI app: unified GoalInsight workspace product.

Single factory: ``create_workspace_app(workspace_dir)`` hosts the
viewer, annotator, pipeline console, library, and tracking diagnostics
against a workspace directory (see ``_workspace.Workspace``). The
annotator routes, library endpoints, jobs API, analytics endpoints,
tracking-diag, and pipeline-results route packs are attached by their
own modules so this file stays a thin assembler.
"""

from __future__ import annotations

import logging
import os
from collections import Counter
from pathlib import Path
from typing import Any

import json


def _chat_enabled() -> bool:
    """Read the GOALINSIGHT_DISABLE_CHAT env flag.

    Defaults to chat-enabled (legacy behaviour). The offline Docker
    image sets ``GOALINSIGHT_DISABLE_CHAT=1`` so the same code base can
    ship as either a credentials-free viewer or the full chat surface
    depending on deployment.
    """
    val = os.getenv("GOALINSIGHT_DISABLE_CHAT", "").strip().lower()
    # Treat any truthy value as "disabled"; empty / "0" / "false" keep
    # chat on.
    return val in ("", "0", "false", "no", "off")

from fastapi import Body, FastAPI, HTTPException
from fastapi.responses import FileResponse, JSONResponse, RedirectResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from ._runs import (
    CHAT_ARTIFACTS_URL as RUNS_ARTIFACTS_URL,
    IncompleteRunError,
    RunRegistry,
)
from ._s3_video import (
    presigned_url_for_key,
    presigned_video_url,
    s3_key_exists,
    s3_object_exists,
    source_video_key,
    upload_source_video,
    video_bucket,
)
from ._workspace import Workspace, resolve_workspace
from .analytics import register_analytics_routes
from .jobs import JobManager, register_jobs_routes
from .library import register_library_routes
from .match_detail import register_match_detail_routes
from .pipeline_results import register_pipeline_results_routes

logger = logging.getLogger(__name__)

STATIC_DIR = Path(__file__).parent / "static"
# Plots / images that run_python emits are stored here and mounted at
# /chat_artifacts/<run>/* so the chat UI can render them inline. Each
# RunHandle clears its own subdir on creation so old artifacts from a
# previous run don't silently shadow new ones with the same name.
CHAT_ARTIFACTS_URL = "/chat_artifacts"


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    messages: list[ChatMessage]
    current_time: float = 0.0


class SessionCreateRequest(BaseModel):
    title: str | None = None


class SessionUpdateRequest(BaseModel):
    title: str | None = None


class SessionChatRequest(BaseModel):
    # The current user message + the playback time. We don't take
    # the full history from the client because the server already
    # has it on disk (less round-tripping, less drift).
    message: str
    current_time: float = 0.0


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
    where ``configs/templates/children.yaml`` ships pitch_type kids_soccer).
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
    app.state.annotator = annotator
    register_annotation_routes(
        app, annotator, ws.videos_dir,
        configs_root=ws.configs_dir,
        prefix="/api/annotate",
    )
    # Open the first discovered video AFTER routes are mounted so the
    # open_video wrapper installed by register_annotation_routes (which
    # applies the per-video workspace config) runs on the initial open
    # too. Failing here doesn't kill the boot — uploads come later.
    initial = sorted(
        [p for ext in _ANNOT_VIDEO_EXTS for p in ws.videos_dir.glob(f"*{ext}")],
        key=lambda p: p.name,
    )
    if initial:
        try:
            annotator.open_video(str(initial[0]), start_frame=0)
        except Exception as exc:  # noqa: BLE001
            logger.warning("annotator: could not auto-open %s: %s", initial[0], exc)

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

    # The /insights/* routes are the LLM chat surface. Gated behind
    # GOALINSIGHT_DISABLE_CHAT so the offline / credentials-free
    # deployment doesn't serve a half-broken page that 500s on every
    # chat call.
    if _chat_enabled():
        @app.get("/insights")
        @app.get("/insights/")
        def insights_index_page() -> FileResponse:
            # Run picker — same pattern as /match/ / /library: lists every
            # run with a tracking + annotated_video output and lets the user
            # click into one. Matches the nav-tab convention so Insights
            # is always reachable, even when no run is in the URL yet.
            path = STATIC_DIR / "insights_index.html"
            if not path.exists():
                raise HTTPException(503, "insights_index.html not yet installed")
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

    @app.get("/match")
    @app.get("/match/")
    def match_index_page() -> FileResponse:
        path = STATIC_DIR / "match_index.html"
        if not path.exists():
            raise HTTPException(503, "match_index.html not yet installed")
        return FileResponse(path, headers=no_cache)

    @app.get("/match/{run_name}")
    def match_detail_page(run_name: str) -> FileResponse:
        path = STATIC_DIR / "match.html"
        if not path.exists():
            raise HTTPException(503, "match.html not yet installed")
        return FileResponse(path, headers=no_cache)

    # Per-run viewer state. ChatEngine boot is expensive (boto3 + Bedrock +
    # MatchContext load); the registry keeps a small LRU of warm runs.
    runs = RunRegistry(ws)
    app.state.runs = runs
    register_analytics_routes(app, runs)
    # Per-stage manifest endpoints used by the /pipeline page right pane.
    register_pipeline_results_routes(app, ws)
    # Match-detail page payload (roster + events with seek windows).
    register_match_detail_routes(app, runs)

    @app.on_event("shutdown")
    def _shutdown_runs() -> None:
        runs.close_all()

    def _get_run_or_http(run_name: str):
        """Resolve a run handle, mapping registry errors to clean HTTP codes.

        - 404 when the run dir doesn't exist.
        - 409 when the run exists but pipeline stages chat/viewer
          depend on (tracking, track_consolidation) haven't finished —
          the response body lists which stages are present so the
          frontend can show a "pipeline incomplete" banner instead of
          a stack trace.
        """
        try:
            return runs.get(run_name)
        except FileNotFoundError as exc:
            raise HTTPException(404, str(exc)) from exc
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

    @app.get("/api/features")
    def features() -> dict[str, bool]:
        """Feature-flag probe consumed by shell.html nav.

        ``chat_enabled`` reflects ``GOALINSIGHT_DISABLE_CHAT`` and lets
        the static shell hide the Insights tab without baking a
        deployment-specific HTML variant into the image.
        """
        return {"chat_enabled": _chat_enabled()}

    @app.get("/api/runs/{run_name}/meta")
    def run_meta(run_name: str) -> JSONResponse:
        handle = _get_run_or_http(run_name)
        ctx = handle.ctx
        teams = Counter(ctx.team_assignments.values())
        event_types = Counter(e.get("type", "unknown") for e in ctx.events)
        # In Runtime mode, run_python returns presigned S3 URLs from
        # this bucket; the viewer needs to know it to widen its
        # IMG_RE so the image renders inline. Empty in local mode.
        import os as _os
        chat_artifact_bucket = (
            _os.environ.get("GOALINSIGHT_CHAT_ARTIFACT_BUCKET")
            or _os.environ.get("GOALINSIGHT_S3_BUCKET", "")
        ) if _os.environ.get("GOALINSIGHT_AGENTCORE_RUNTIME_ARN") else ""
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
            "chat_artifact_bucket": chat_artifact_bucket,
        })

    @app.get("/api/runs/{run_name}/video")
    def run_video(run_name: str):
        """Serve the run's playable mp4.

        When ``GOALINSIGHT_VIDEO_S3_BUCKET`` is set, redirect the
        browser to a short-lived presigned S3 GET URL so EC2 stops
        being the byte-pump. Two key shapes are supported:

          - Run-derived mp4 (``runs/<run>/<stage>/<file>``) — when the
            chosen video lives inside the run dir.
          - Source video (``videos/<file>``) — when the run hasn't
            produced annotated_video / consolidated and we're
            falling back to the original input. Auto-uploaded on
            first hit so existing runs don't need a pre-bake.

        Falls back to streaming from local disk when S3 isn't
        configured or the object can't be found / signed.
        """
        handle = _get_run_or_http(run_name)
        bucket = video_bucket()
        if bucket:
            try:
                rel = handle.video_path.relative_to(handle.run_dir).as_posix()
                if s3_object_exists(run_name, rel, bucket=bucket):
                    url = presigned_video_url(run_name, rel, bucket=bucket)
                    if url:
                        return RedirectResponse(url, status_code=302)
            except ValueError:
                # Source video lives outside run_dir — key it by
                # basename and lazily upload on first miss.
                key = source_video_key(handle.video_path)
                if not s3_key_exists(key, bucket=bucket):
                    logger.info("uploading source video %s on first hit", key)
                    upload_source_video(handle.video_path, bucket=bucket)
                if s3_key_exists(key, bucket=bucket):
                    url = presigned_url_for_key(key, bucket=bucket)
                    if url:
                        return RedirectResponse(url, status_code=302)
        return FileResponse(handle.video_path, media_type="video/mp4")

    # ------------------------------------------------------------------
    # Chat API surface — gated behind GOALINSIGHT_DISABLE_CHAT so the
    # offline image doesn't expose endpoints that 500 without AWS creds.
    # The chat-disabled deployment also skips the ``/chat_artifacts``
    # static mount further down (see "Mount the workspace-wide chat
    # artifact root" below).
    # ------------------------------------------------------------------

    @app.post("/api/runs/{run_name}/chat")
    def run_chat(run_name: str, req: ChatRequest) -> dict[str, Any]:
        if not _chat_enabled():
            raise HTTPException(404, "chat disabled")
        handle = _get_run_or_http(run_name)
        try:
            text = handle.engine.respond(
                [m.model_dump() for m in req.messages],
                req.current_time,
            )
        except Exception as exc:  # noqa: BLE001
            logger.exception("chat failed (run=%s)", run_name)
            raise HTTPException(status_code=500, detail=str(exc)) from exc
        return {"text": text}

    @app.post("/api/runs/{run_name}/chat/prepare")
    def run_chat_prepare(run_name: str) -> StreamingResponse:
        """Stream session-preparation progress as SSE.

        When the chat agent is hosted on AgentCore Runtime this uploads
        the run's JSON to S3 and warms up the runtime; in local mode it
        immediately emits ``ready``. Front-end uses this to gate the
        chat input until preparation finishes.
        """
        if not _chat_enabled():
            raise HTTPException(404, "chat disabled")
        handle = _get_run_or_http(run_name)

        from .chat_prepare import stream_prepare

        def event_source():
            try:
                for evt in stream_prepare(run_name, handle.run_dir):
                    yield f"data: {json.dumps(evt)}\n\n"
            except Exception as exc:  # noqa: BLE001
                logger.exception("chat prepare failed (run=%s)", run_name)
                yield f"data: {json.dumps({'stage':'error','percent':0,'detail':str(exc)})}\n\n"

        return StreamingResponse(
            event_source(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    @app.post("/api/runs/{run_name}/chat/stream")
    def run_chat_stream(run_name: str, req: ChatRequest) -> StreamingResponse:
        if not _chat_enabled():
            raise HTTPException(404, "chat disabled")
        handle = _get_run_or_http(run_name)
        messages = [m.model_dump() for m in req.messages]
        current_time = req.current_time

        def event_source():
            try:
                for evt in handle.engine.stream(messages, current_time):
                    # Back-compat: text events ship as {'delta': '...'}
                    # so existing viewers keep working; new tool_use /
                    # tool_result events ship as {'event': {...}}.
                    if evt.get("type") == "text":
                        yield f"data: {json.dumps({'delta': evt['delta']})}\n\n"
                    else:
                        yield f"data: {json.dumps({'event': evt}, ensure_ascii=False)}\n\n"
                yield "data: {\"done\": true}\n\n"
            except Exception as exc:  # noqa: BLE001
                logger.exception("chat stream failed (run=%s)", run_name)
                yield f"data: {json.dumps({'error': str(exc)})}\n\n"

        return StreamingResponse(
            event_source(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    # ------------------------------------------------------------------
    # Multi-session chat: each /insights/<run> page can keep several
    # independent conversations. State lives on disk under
    # workspace/runs/<run>/chat_sessions/<sid>/, so a server restart
    # — or a returning user — sees the same list of conversations.
    # ------------------------------------------------------------------

    def _store(handle):
        if handle.sessions is None:
            raise HTTPException(500, "session store not available for this run")
        return handle.sessions

    @app.get("/api/runs/{run_name}/chat/sessions")
    def list_sessions(run_name: str) -> dict[str, Any]:
        if not _chat_enabled():
            raise HTTPException(404, "chat disabled")
        handle = _get_run_or_http(run_name)
        sessions = [m.to_json() for m in _store(handle).list()]
        return {"sessions": sessions}

    @app.post("/api/runs/{run_name}/chat/sessions")
    def create_session(
        run_name: str, req: SessionCreateRequest,
    ) -> dict[str, Any]:
        if not _chat_enabled():
            raise HTTPException(404, "chat disabled")
        handle = _get_run_or_http(run_name)
        meta = _store(handle).create(title=req.title)
        return meta.to_json()

    @app.get("/api/runs/{run_name}/chat/sessions/{sid}")
    def get_session(run_name: str, sid: str) -> dict[str, Any]:
        if not _chat_enabled():
            raise HTTPException(404, "chat disabled")
        handle = _get_run_or_http(run_name)
        store = _store(handle)
        try:
            meta = store.get_meta(sid)
        except KeyError:
            raise HTTPException(404, f"session not found: {sid}")
        return {
            **meta.to_json(),
            "messages": store.list_messages(sid),
        }

    @app.patch("/api/runs/{run_name}/chat/sessions/{sid}")
    def update_session(
        run_name: str, sid: str, req: SessionUpdateRequest,
    ) -> dict[str, Any]:
        if not _chat_enabled():
            raise HTTPException(404, "chat disabled")
        handle = _get_run_or_http(run_name)
        store = _store(handle)
        try:
            meta = store.update_meta(sid, title=req.title)
        except KeyError:
            raise HTTPException(404, f"session not found: {sid}")
        return meta.to_json()

    @app.delete("/api/runs/{run_name}/chat/sessions/{sid}")
    def delete_session(run_name: str, sid: str) -> dict[str, Any]:
        if not _chat_enabled():
            raise HTTPException(404, "chat disabled")
        handle = _get_run_or_http(run_name)
        _store(handle).delete(sid)
        return {"ok": True, "session_id": sid}

    @app.post("/api/runs/{run_name}/chat/sessions/{sid}/resume")
    def resume_session(run_name: str, sid: str) -> dict[str, Any]:
        """Warm the AgentCore sandbox for *sid* without sending a turn.

        Calls keepalive() on the engine's sandbox: if the cached
        AgentCore session id is still alive, this is a single ~200ms
        no-op invoke; if it's been reaped, _invoke transparently
        spawns a fresh one (and re-uploads data) before returning
        success. Either way the next chat turn lands on a warm
        sandbox.
        """
        if not _chat_enabled():
            raise HTTPException(404, "chat disabled")
        handle = _get_run_or_http(run_name)
        store = _store(handle)
        try:
            store.get_meta(sid)
        except KeyError:
            raise HTTPException(404, f"session not found: {sid}")
        engine = store.engine(sid)
        sandbox = getattr(engine, "_ensure_sandbox", lambda: None)()
        alive = bool(sandbox and sandbox.keepalive()) if sandbox else False
        store.persist_agentcore_session(sid)
        return {
            "ok": True,
            "session_id": sid,
            "agentcore_alive": alive,
            "agentcore_session_id": getattr(sandbox, "_session_id", None)
                if sandbox else None,
        }

    @app.post("/api/runs/{run_name}/chat/sessions/{sid}/stream")
    def session_chat_stream(
        run_name: str, sid: str, req: SessionChatRequest,
    ) -> StreamingResponse:
        if not _chat_enabled():
            raise HTTPException(404, "chat disabled")
        handle = _get_run_or_http(run_name)
        store = _store(handle)
        try:
            store.get_meta(sid)
        except KeyError:
            raise HTTPException(404, f"session not found: {sid}")

        # The on-disk history is the source of truth — append the new
        # user turn before we hand things to the engine, so a crash in
        # the middle of streaming doesn't lose the prompt.
        user_text = (req.message or "").strip()
        if not user_text:
            raise HTTPException(400, "empty message")
        store.append_message(sid, {"role": "user", "content": user_text})
        store.maybe_autotitle(sid, user_text)

        # The Bedrock chat engine should only see {role: user|assistant}
        # turns; the persisted tool_use/tool_result entries are UI
        # breadcrumbs and would confuse the model if fed back as
        # conversation. ``llm_history()`` filters them out.
        history = store.llm_history(sid)
        current_time = req.current_time
        engine = store.engine(sid)

        def event_source():
            # Only persist the FINAL round's text (commentary from
            # intermediate tool-using rounds is shown in real time but
            # discarded from the saved transcript so history replay
            # matches the original buffered behaviour).
            current_round_text: list[str] = []
            final_round_text: list[str] = []
            try:
                for evt in engine.stream(history, current_time):
                    et = evt.get("type")
                    if et == "round_start":
                        current_round_text = []
                        yield f"data: {json.dumps({'event': evt}, ensure_ascii=False)}\n\n"
                        continue
                    if et == "round_end":
                        # Last seen round is the most recent; if it
                        # ended on end_turn it's the final answer.
                        if evt.get("stop_reason") != "tool_use":
                            final_round_text = current_round_text
                        yield f"data: {json.dumps({'event': evt}, ensure_ascii=False)}\n\n"
                        continue
                    if et == "text":
                        current_round_text.append(evt["delta"])
                        yield f"data: {json.dumps({'delta': evt['delta']})}\n\n"
                    elif evt.get("type") == "tool_use":
                        # Persist before yielding so a stream that dies
                        # mid-flight still leaves a usable trail in
                        # messages.jsonl.
                        store.append_message(sid, {
                            "role": "tool_use",
                            "id": evt.get("id"),
                            "name": evt.get("name"),
                            "input": evt.get("input") or {},
                        })
                        yield f"data: {json.dumps({'event': evt}, ensure_ascii=False)}\n\n"
                    elif evt.get("type") == "tool_result":
                        store.append_message(sid, {
                            "role": "tool_result",
                            "id": evt.get("id"),
                            "name": evt.get("name"),
                            "result": evt.get("result") or {},
                        })
                        yield f"data: {json.dumps({'event': evt}, ensure_ascii=False)}\n\n"
                    else:
                        yield f"data: {json.dumps({'event': evt}, ensure_ascii=False)}\n\n"
                # Persist the assistant's final-round text + the
                # AgentCore session id so a server restart (or the user
                # closing the tab and coming back tomorrow) can resume.
                full = "".join(final_round_text)
                if full:
                    store.append_message(sid, {
                        "role": "assistant",
                        "content": full,
                    })
                store.persist_agentcore_session(sid)
                yield "data: {\"done\": true}\n\n"
            except Exception as exc:  # noqa: BLE001
                logger.exception(
                    "session chat stream failed (run=%s sid=%s)",
                    run_name, sid)
                yield f"data: {json.dumps({'error': str(exc)})}\n\n"

        return StreamingResponse(
            event_source(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    # Mount the workspace-wide chat artifact root at /chat_artifacts so the
    # existing IMG_RE in viewer.html ('/chat_artifacts/...') still matches.
    # ChatEngine writes per-run files under <root>/<run_name>/.
    # Skipped when chat is disabled — there's nothing under that path
    # without an LLM writing into it.
    if _chat_enabled():
        app.mount(
            RUNS_ARTIFACTS_URL,
            StaticFiles(directory=runs.artifacts_root),
            name="chat_artifacts",
        )
    # Read-only direct access to every file inside a run dir, used by the
    # /pipeline page to display per-stage vis JPGs / mp4s. Path layout
    # mirrors the on-disk one: /runs_static/<run>/<stage>/...
    #
    # Dynamic resize for vis JPGs: hitting the same URL with ``?w=720``
    # / 1080 / 1920 returns a downscaled copy. Used by the /pipeline
    # page so the user can pick a resolution and not have to download
    # 1.3 MB 4K frames at 10 fps. Cached in-memory (LRU).
    from io import BytesIO
    import cv2 as _cv2_resize
    from fastapi import HTTPException, Query
    from fastapi.responses import Response as _Response
    from functools import lru_cache as _lru_cache

    @_lru_cache(maxsize=256)
    def _resize_cached(path_str: str, target_w: int, mtime_ns: int) -> bytes:
        """Read JPG at ``path_str`` and return a JPEG bytes payload
        downscaled so the long edge is ``target_w``. ``mtime_ns`` is
        baked into the cache key so re-rendered frames invalidate the
        cached resize. Aspect ratio preserved.
        """
        img = _cv2_resize.imread(path_str)
        if img is None:
            raise FileNotFoundError(path_str)
        h, w = img.shape[:2]
        long_edge = max(w, h)
        if long_edge <= target_w:
            with open(path_str, "rb") as f:
                return f.read()
        scale = target_w / long_edge
        new_w = int(round(w * scale))
        new_h = int(round(h * scale))
        out = _cv2_resize.resize(
            img, (new_w, new_h), interpolation=_cv2_resize.INTER_AREA,
        )
        ok, enc = _cv2_resize.imencode(".jpg", out, [_cv2_resize.IMWRITE_JPEG_QUALITY, 82])
        if not ok:
            raise RuntimeError("jpeg encode failed")
        return bytes(enc)

    _ALLOWED_W = {720, 1080, 1920, 3840}

    # All `/runs_static/*` requests come through this handler. JPGs
    # with ``?w=720|1080|1920|3840`` are downscaled & cached on the
    # fly; everything else (other JPGs, mp4, json, …) is served as-is
    # via FileResponse so the original byte-range / streaming behaviour
    # is preserved (mp4 playback in particular needs ranged GETs).
    from fastapi.responses import FileResponse as _FileResponse
    import mimetypes as _mt

    @app.get("/runs_static/{path:path}")
    def _runs_static_handler(path: str, w: int | None = None):
        full = ws.runs_dir / path
        # Frame jpgs get re-rendered while the browser stays open;
        # disable client caching so the user always sees fresh bytes.
        # The server-side LRU still amortises decoding/encoding cost.
        no_cache = {"Cache-Control": "no-cache, no-store, must-revalidate"}
        is_jpg = path.lower().endswith((".jpg", ".jpeg"))

        # When S3 video bucket is configured AND this is an as-is
        # request for a heavy media file (mp4 / jpg / png), redirect
        # to a presigned URL so EC2 stops pumping bytes. Skip the
        # HEAD check — S3 returns 403/404 directly to the browser if
        # the key isn't there yet, and at scale a HEAD per request
        # would dominate latency. Resize requests (?w=720|1080|...)
        # still go through the local decoder + LRU cache.
        #
        # JSON / other small files stay local so the browser doesn't
        # have to round-trip to S3 for every API-shaped JSON the
        # /match and /pipeline pages fetch (tracks.json, events.json,
        # players.json, ...). Their bytes are negligible and they're
        # consumed once per page load.
        _S3_REDIRECT_EXTS = (".mp4", ".webm", ".jpg", ".jpeg", ".png", ".webp")
        if (
            w is None
            and path.lower().endswith(_S3_REDIRECT_EXTS)
            and video_bucket()
        ):
            key = f"runs/{path.lstrip('/')}"
            url = presigned_url_for_key(key)
            if url:
                return RedirectResponse(url, status_code=302)

        if not full.is_file():
            raise HTTPException(status_code=404, detail="not found")
        if w is None or not is_jpg:
            mime, _ = _mt.guess_type(str(full))
            return _FileResponse(
                str(full),
                media_type=mime or "application/octet-stream",
                headers=no_cache if is_jpg else None,
            )
        if w not in _ALLOWED_W:
            raise HTTPException(
                status_code=400, detail=f"w must be one of {sorted(_ALLOWED_W)}",
            )
        try:
            payload = _resize_cached(str(full), w, full.stat().st_mtime_ns)
        except FileNotFoundError:
            raise HTTPException(status_code=404, detail="not found")
        return _Response(payload, media_type="image/jpeg", headers=no_cache)
    # Static assets (JS / CSS / HTML fragments) evolve rapidly during
    # dev; browser HTTP cache would silently serve stale files that
    # confuse debugging (a familiar footgun: "why doesn't my JS
    # change show up"). Wrap StaticFiles so every asset carries
    # ``Cache-Control: no-store`` — no server-side caching cost, and
    # the browser always fetches fresh.
    class _NoCacheStatic(StaticFiles):
        async def get_response(self, path, scope):
            resp = await super().get_response(path, scope)
            resp.headers["Cache-Control"] = (
                "no-store, no-cache, must-revalidate, max-age=0"
            )
            return resp

    app.mount("/static", _NoCacheStatic(directory=STATIC_DIR), name="static")

    return app
