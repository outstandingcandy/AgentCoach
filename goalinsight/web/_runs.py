"""Lazy per-run state for the unified viewer.

Each pipeline run under ``workspace/runs/<run_name>/`` owns a MatchContext +
ChatEngine. Building these is expensive (loads JSON, opens a Bedrock
client, eagerly parses tracks), so we build on first request and keep at
most ``RUN_CACHE_SIZE`` warm in memory. The least-recently-used run is
closed when capacity is exceeded.

When ``GOALINSIGHT_AGENTCORE_RUNTIME_ARN`` is set, chat is delegated to
an AgentCore Runtime container (see ``chat_remote.RemoteChatEngine``)
instead of running locally. Other endpoints (analytics, viewer, etc.)
are unaffected.
"""

from __future__ import annotations

import logging
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from ..highlights._context import MatchContext
from ._sessions import SessionStore
from ._workspace import VIDEO_EXTS, Workspace

# ``.chat`` / ``.chat_remote`` are imported lazily inside the engine
# factories below — they pull boto3 + Bedrock client setup into memory,
# and an offline / chat-disabled deployment shouldn't pay that cost
# just to list runs or render the match-detail page.


class ChatEngineLike(Protocol):
    def respond(self, messages: list[dict[str, str]], current_time: float) -> str: ...
    def stream(self, messages: list[dict[str, str]], current_time: float): ...
    def close(self) -> None: ...

logger = logging.getLogger(__name__)

RUN_CACHE_SIZE = 4
CHAT_ARTIFACTS_URL = "/chat_artifacts"


class IncompleteRunError(RuntimeError):
    """Raised when a run is missing pipeline stages chat depends on.

    Carries the per-stage completion map so the API layer can return a
    structured 409 instead of a bare 500. Chat needs at minimum
    track_consolidation (for the consolidated tracks + roster) and
    benefits from event_detection; tracking is implied by both.
    """

    REQUIRED_STAGES = ("tracking", "track_consolidation")
    RECOMMENDED_STAGES = ("event_detection",)

    def __init__(self, run_name: str, stages: dict[str, bool]):
        missing_required = [
            s for s in self.REQUIRED_STAGES if not stages.get(s, False)
        ]
        super().__init__(
            f"run {run_name!r} is missing required stages: "
            f"{', '.join(missing_required)}"
        )
        self.run_name = run_name
        self.stages = stages
        self.missing_required = missing_required
        self.missing_recommended = [
            s for s in self.RECOMMENDED_STAGES if not stages.get(s, False)
        ]


@dataclass
class RunHandle:
    run_name: str
    run_dir: Path
    video_path: Path
    ctx: MatchContext
    # Built lazily on first chat call so analytics-only requests (heatmap,
    # stats) don't pay the boto3 / Bedrock startup cost.
    # ``_engine`` is the legacy single-engine path (kept so /chat and
    # /chat/stream without a session id still work); ``sessions`` is
    # the per-session store the new UI uses.
    _engine: ChatEngineLike | None = None
    _engine_factory: Any = None
    sessions: SessionStore | None = None

    @property
    def engine(self) -> ChatEngineLike:
        if self._engine is None:
            if self._engine_factory is None:
                raise RuntimeError("engine factory missing for run handle")
            self._engine = self._engine_factory()
        return self._engine

    def close(self) -> None:
        if self._engine is not None:
            try:
                self._engine.close()
            except Exception:  # noqa: BLE001
                logger.exception("engine close failed for %s", self.run_name)
        if self.sessions is not None:
            self.sessions.close_all()


def resolve_run_video(run_dir: Path) -> Path:
    """Pick the video file the viewer should serve for *run_dir*.

    Preference order:
      1. ``annotated_video/annotated_web.mp4`` (browser-optimized HUD)
      2. ``annotated_video/annotated.mp4`` (full HUD render)
      3. ``track_consolidation/consolidated.mp4`` if it's a real
         render (>1 KB; consolidated.mp4 may exist as a 257-byte
         placeholder when the stage was run with ``--no-viz``)
      4. The source video recorded in ``calibration_metadata.json``
         (only the field_registration stage ran).
    """
    candidates = [
        run_dir / "annotated_video" / "annotated_web.mp4",
        run_dir / "annotated_video" / "annotated.mp4",
    ]
    for c in candidates:
        if c.exists():
            return c
    cons = run_dir / "track_consolidation" / "consolidated.mp4"
    # 1024 bytes is well above the empty-mp4 stub size (~257 B) but
    # safely below any real render even at the lowest bitrate.
    if cons.exists() and cons.stat().st_size > 1024:
        return cons
    # Fall back to the source video recorded in calibration metadata so
    # users can still open a viewer for runs that haven't produced
    # annotated_video yet (e.g. only field_registration ran).
    meta = run_dir / "field_registration" / "calibration_metadata.json"
    if meta.exists():
        import json
        try:
            vi = json.loads(meta.read_text()).get("video_info", {})
            src = vi.get("path")
            if src and Path(src).exists():
                return Path(src)
        except Exception:  # noqa: BLE001
            pass
    raise FileNotFoundError(
        f"no playable video found for run {run_dir.name}: "
        f"annotated_video/, track_consolidation/consolidated.mp4 "
        "and source-video fallback all missing"
    )


class RunRegistry:
    """LRU cache of warm RunHandle objects keyed by run_name."""

    def __init__(self, workspace: Workspace, *, capacity: int = RUN_CACHE_SIZE):
        self.workspace = workspace
        self.capacity = capacity
        self._cache: OrderedDict[str, RunHandle] = OrderedDict()
        # Per-run chat artifact dirs land under workspace/chat_artifacts/<run>;
        # the unified app mounts workspace/chat_artifacts at /chat_artifacts so
        # the existing IMG_RE in viewer.html still matches without changes.
        self.artifacts_root = workspace.root / "chat_artifacts"
        self.artifacts_root.mkdir(parents=True, exist_ok=True)

    def artifact_dir_for(self, run_name: str) -> Path:
        return self.artifacts_root / run_name

    def url_prefix_for(self, run_name: str) -> str:
        return f"{CHAT_ARTIFACTS_URL}/{run_name}"

    def get(self, run_name: str) -> RunHandle:
        if run_name in self._cache:
            self._cache.move_to_end(run_name)
            return self._cache[run_name]

        run_dir = self.workspace.run_dir(run_name).resolve()
        if not run_dir.is_dir():
            raise FileNotFoundError(f"run not found: {run_dir}")

        # Surface "pipeline not finished yet" as a typed error so the API
        # can return 409 + the stage map instead of a 500 from
        # MatchContext's hard precondition check. Cheaper than letting
        # MatchContext.from_output_dir raise: it skips the calibration
        # metadata / video resolution work for runs we already know
        # aren't usable.
        stages = _detect_stage_completion(run_dir)
        missing = [
            s for s in IncompleteRunError.REQUIRED_STAGES
            if not stages.get(s, False)
        ]
        if missing:
            raise IncompleteRunError(run_name, stages)

        video_path = resolve_run_video(run_dir)
        ctx = MatchContext.from_output_dir(run_dir, video_path=video_path)

        artifact_dir = self.artifact_dir_for(run_name)
        # Don't wipe the per-run artifact dir on reload: the new
        # multi-session UI persists each session's plots under
        # ``<artifact_dir>/<session_id>/``, and a server restart should
        # leave those alone. Legacy single-engine artifacts that landed
        # directly in <artifact_dir>/*.png stay too — they're cheap and
        # won't shadow new ones (per-session subdirs are URL-distinct).
        artifact_dir.mkdir(parents=True, exist_ok=True)

        prefix = self.url_prefix_for(run_name)

        def _build_engine() -> ChatEngineLike:
            from .chat import ChatEngine
            from .chat_remote import (
                RemoteChatEngine,
                runtime_arn_from_env,
                s3_bucket_from_env,
                s3_prefix_for,
            )

            runtime_arn = runtime_arn_from_env()
            s3_bucket = s3_bucket_from_env() if runtime_arn else None
            if runtime_arn:
                if not s3_bucket:
                    raise RuntimeError(
                        "GOALINSIGHT_AGENTCORE_RUNTIME_ARN is set but "
                        "GOALINSIGHT_S3_BUCKET is not — can't tell the "
                        "runtime where to read this run's outputs from."
                    )
                logger.info(
                    "run %s: chat via AgentCore Runtime %s",
                    run_name, runtime_arn,
                )
                return RemoteChatEngine(
                    ctx=ctx,
                    run_name=run_name,
                    agent_runtime_arn=runtime_arn,
                    s3_bucket=s3_bucket,
                    s3_prefix=s3_prefix_for(run_name),
                )
            return ChatEngine(
                ctx=ctx,
                artifact_dir=artifact_dir,
                artifact_url_prefix=prefix,
            )

        # Each chat session in the new multi-session API gets its own
        # engine + AgentCore sandbox + artifact dir. We reuse the same
        # factory above, but parametrised per-session.
        def _build_session_engine(
            *, sid: str, artifact_dir: Path,
            artifact_url_prefix: str,
            agentcore_session_id: str | None,
        ) -> ChatEngineLike:
            from .chat import ChatEngine
            from .chat_remote import (
                RemoteChatEngine,
                runtime_arn_from_env,
                s3_bucket_from_env,
                s3_prefix_for,
            )

            runtime_arn = runtime_arn_from_env()
            s3_bucket = s3_bucket_from_env() if runtime_arn else None
            if runtime_arn:
                # RemoteChatEngine doesn't expose AgentCore session
                # reuse; in Runtime mode every session shares the
                # remote runtime container's lifecycle.
                if not s3_bucket:
                    raise RuntimeError(
                        "GOALINSIGHT_AGENTCORE_RUNTIME_ARN is set but "
                        "GOALINSIGHT_S3_BUCKET is not — can't tell the "
                        "runtime where to read this run's outputs from."
                    )
                return RemoteChatEngine(
                    ctx=ctx,
                    run_name=run_name,
                    agent_runtime_arn=runtime_arn,
                    s3_bucket=s3_bucket,
                    s3_prefix=s3_prefix_for(run_name),
                )
            return ChatEngine(
                ctx=ctx,
                artifact_dir=artifact_dir,
                artifact_url_prefix=artifact_url_prefix,
                agentcore_session_id=agentcore_session_id,
            )

        sessions = SessionStore(
            run_name=run_name,
            run_dir=run_dir,
            ctx=ctx,
            artifact_root=artifact_dir,
            artifact_url_prefix_root=prefix,
            engine_factory=_build_session_engine,
        )

        handle = RunHandle(
            run_name=run_name,
            run_dir=run_dir,
            video_path=video_path,
            ctx=ctx,
            _engine_factory=_build_engine,
            sessions=sessions,
        )
        self._cache[run_name] = handle
        self._evict()
        return handle

    def _evict(self) -> None:
        while len(self._cache) > self.capacity:
            old_name, old_handle = self._cache.popitem(last=False)
            logger.info("evicting run %s from registry", old_name)
            old_handle.close()

    def close_all(self) -> None:
        for handle in self._cache.values():
            handle.close()
        self._cache.clear()


def list_runs(workspace: Workspace) -> list[dict]:
    """Lightweight enumeration for the library page (no MatchContext load)."""
    out: list[dict] = []
    if not workspace.runs_dir.is_dir():
        return out
    for run_dir in sorted(workspace.runs_dir.iterdir()):
        if not run_dir.is_dir():
            continue
        out.append({
            "run_name": run_dir.name,
            "stages": _detect_stage_completion(run_dir),
        })
    return out


def _detect_stage_completion(run_dir: Path) -> dict[str, bool]:
    return {
        "field_registration": (run_dir / "field_registration" / "homographies.pkl").exists(),
        "tracking": (run_dir / "tracking" / "tracks.json").exists(),
        "track_consolidation": (run_dir / "track_consolidation" / "players.json").exists(),
        "event_detection": (run_dir / "event_detection" / "events.json").exists(),
        "highlights": any((run_dir / "highlights").glob("*.mp4"))
                      if (run_dir / "highlights").is_dir() else False,
        "annotated_video": (run_dir / "annotated_video" / "annotated.mp4").exists(),
        "consolidated_overlay": (run_dir / "track_consolidation" / "players_overlay.mp4").exists(),
    }
