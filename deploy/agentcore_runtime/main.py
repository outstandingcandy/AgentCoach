"""AgentCore Runtime entrypoint for the GoalInsight chat agent.

Exposes the AgentCore HTTP contract:
    GET  /ping        -> {"status": "Healthy"}
    POST /invocations -> SSE stream of {"delta": str} frames, ending
                         with {"done": true}.

Each invocation receives the run_name plus the S3 prefix that holds
that run's pipeline output JSON. On first call for a session, the
container syncs those JSON files from S3 to a per-session tempdir,
builds a MatchContext + ChatEngine, and caches them keyed by the
AgentCore session id (header ``X-Amzn-Bedrock-AgentCore-Runtime-Session-Id``).
Subsequent calls reuse the cached engine — same lifecycle as today's
in-process ChatEngine.

Environment:
    AWS_REGION              — region for boto3 clients (default us-east-1)
    GOALINSIGHT_S3_BUCKET   — bucket holding run outputs; can be
                              overridden per-request via the s3_bucket
                              field in the invocation payload.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import tempfile
import threading
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import boto3
from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel

from chat_app._context import MatchContext
from chat_app.chat import ChatEngine

logging.basicConfig(
    level=os.environ.get("LOG_LEVEL", "INFO"),
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

AWS_REGION = os.environ.get("AWS_REGION", "us-east-1")
DEFAULT_BUCKET = os.environ.get("GOALINSIGHT_S3_BUCKET", "")
# When set, run_python uploads chat artifacts here instead of returning
# a 404 to Claude. Layout: s3://<bucket>/<prefix>/<run>/<file>.png
ARTIFACT_BUCKET = os.environ.get(
    "GOALINSIGHT_CHAT_ARTIFACT_BUCKET", DEFAULT_BUCKET,
)
ARTIFACT_PREFIX = os.environ.get(
    "GOALINSIGHT_CHAT_ARTIFACT_PREFIX", "chat_artifacts",
)

# AgentCore passes the session id via this header. We key the engine
# cache off it so repeat calls inside the same MicroVM-resident session
# reuse the warm ChatEngine.
SESSION_HEADER = "x-amzn-bedrock-agentcore-runtime-session-id"

# These are the only files the chat path reads. Anything else under
# the run dir (vis JPEGs, mp4s, weights) stays out of the runtime
# container.
RUN_FILES = [
    "field_registration/calibration_metadata.json",
    "tracking/tracks.json",
    "tracking/ball_tracks.json",
    "tracking/team_assignments.json",
    "track_consolidation/tracks.json",
    "track_consolidation/team_assignments.json",
    "track_consolidation/players.json",
    "event_detection/events.json",
]


class ChatMessage(BaseModel):
    role: str
    content: str


class InvocationRequest(BaseModel):
    run_name: str
    s3_prefix: str
    messages: list[ChatMessage]
    current_time: float = 0.0
    s3_bucket: str | None = None


class _SessionState:
    """Cached per-session resources held for the MicroVM's lifetime."""

    def __init__(
        self,
        engine: ChatEngine,
        tempdir: tempfile.TemporaryDirectory,
        s3_prefix: str,
    ) -> None:
        self.engine = engine
        self.tempdir = tempdir
        self.s3_prefix = s3_prefix

    def close(self) -> None:
        try:
            self.engine.close()
        finally:
            self.tempdir.cleanup()


_sessions: dict[str, _SessionState] = {}
_sessions_lock = threading.Lock()
_s3 = boto3.client("s3", region_name=AWS_REGION)


def _sync_run_from_s3(bucket: str, prefix: str, dest: Path) -> int:
    """Pull RUN_FILES from s3://bucket/prefix/ into *dest*. Missing files
    are skipped — MatchContext tolerates partial input (e.g. consolidated
    files not yet produced)."""
    pulled = 0
    for rel in RUN_FILES:
        key = f"{prefix.rstrip('/')}/{rel}"
        local = dest / rel
        local.parent.mkdir(parents=True, exist_ok=True)
        try:
            _s3.download_file(bucket, key, str(local))
            pulled += 1
        except Exception as exc:  # noqa: BLE001
            logger.debug("s3 skip %s/%s: %s", bucket, key, exc)
    return pulled


def _ensure_session(
    session_id: str, req: InvocationRequest,
) -> _SessionState:
    bucket = req.s3_bucket or DEFAULT_BUCKET
    if not bucket:
        raise HTTPException(
            status_code=400,
            detail="s3_bucket missing on payload and GOALINSIGHT_S3_BUCKET unset",
        )

    with _sessions_lock:
        existing = _sessions.get(session_id)
        if existing is not None and existing.s3_prefix == req.s3_prefix:
            return existing
        if existing is not None:
            existing.close()
            _sessions.pop(session_id, None)

        td = tempfile.TemporaryDirectory(prefix=f"goalinsight-{req.run_name}-")
        dest = Path(td.name)
        pulled = _sync_run_from_s3(bucket, req.s3_prefix, dest)
        if pulled == 0:
            td.cleanup()
            raise HTTPException(
                status_code=404,
                detail=f"no run files at s3://{bucket}/{req.s3_prefix}/",
            )
        logger.info(
            "session %s: synced %d files from s3://%s/%s",
            session_id, pulled, bucket, req.s3_prefix,
        )

        ctx = MatchContext.from_output_dir(
            pipeline_output_dir=dest,
            video_path=Path(req.run_name),
        )
        # run_python is enabled when an artifact bucket is configured.
        # match_tools.run_python detects the s3:// scheme and uploads
        # PNGs to s3://<bucket>/<prefix>/<run>/, returning presigned
        # URLs in the tool_result. ChatEngine writes its local copy
        # under artifact_dir (a temp dir per session) before upload.
        if ARTIFACT_BUCKET:
            artifact_dir = Path(td.name) / "_artifacts"
            artifact_dir.mkdir(parents=True, exist_ok=True)
            artifact_url_prefix = (
                f"s3://{ARTIFACT_BUCKET}/{ARTIFACT_PREFIX.strip('/')}/"
                f"{req.run_name}"
            )
        else:
            artifact_dir = None
            artifact_url_prefix = ""
        engine = ChatEngine(
            ctx=ctx,
            region=AWS_REGION,
            artifact_dir=artifact_dir,
            artifact_url_prefix=artifact_url_prefix,
        )
        state = _SessionState(engine=engine, tempdir=td, s3_prefix=req.s3_prefix)
        _sessions[session_id] = state
        return state


app = FastAPI(title="GoalInsight Chat Runtime")


@app.get("/ping")
def ping() -> dict[str, str]:
    return {"status": "Healthy"}


WARMUP_SENTINEL = "__warmup__"


@app.post("/invocations")
def invocations(
    req: InvocationRequest, request: Request,
) -> StreamingResponse:
    session_id = request.headers.get(SESSION_HEADER) or req.run_name
    state = _ensure_session(session_id, req)

    # Warmup pings from the FastAPI proxy — _ensure_session above is
    # the actual work; we don't want to call Bedrock for these. Detect
    # by the sentinel content of the only message.
    is_warmup = (
        len(req.messages) == 1
        and req.messages[0].role == "user"
        and req.messages[0].content == WARMUP_SENTINEL
    )
    messages = [m.model_dump() for m in req.messages]
    current_time = req.current_time

    def event_source() -> Iterator[bytes]:
        if is_warmup:
            yield b"data: {\"ready\": true}\n\n"
            yield b"data: {\"done\": true}\n\n"
            return
        try:
            for delta in state.engine.stream(messages, current_time):
                payload = json.dumps({"delta": delta}, ensure_ascii=False)
                yield f"data: {payload}\n\n".encode()
            yield b"data: {\"done\": true}\n\n"
        except Exception as exc:  # noqa: BLE001
            logger.exception("stream failed for session %s", session_id)
            err = json.dumps({"error": str(exc)})
            yield f"data: {err}\n\n".encode()

    return StreamingResponse(
        event_source(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.delete("/sessions/{session_id}")
def end_session(session_id: str) -> JSONResponse:
    with _sessions_lock:
        state = _sessions.pop(session_id, None)
    if state is None:
        return JSONResponse({"closed": False})
    state.close()
    return JSONResponse({"closed": True})
