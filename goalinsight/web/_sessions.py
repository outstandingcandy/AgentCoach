"""Per-run chat session storage.

A ``ChatSession`` is one user conversation with the match-insights chat
inside a single run. Each session keeps:

  - its own message history (jsonl on disk so it survives restarts),
  - its own ChatEngine (so artifact URLs and the AgentCore Code
    Interpreter sandbox don't leak between sessions),
  - the AgentCore session id of the last live sandbox, persisted so
    "Resume" can re-attach to the same MicroVM if it's still alive
    instead of paying the cold-start + data-upload tax.

On disk::

    workspace/runs/<run>/chat_sessions/
        <session_id>/
            meta.json         # title, created_at, agentcore_session_id, ...
            messages.jsonl    # one JSON message per line

    workspace/chat_artifacts/<run>/<session_id>/<plot>.png

Sessions are CRUD-managed via ``SessionStore``; a single store instance
lives on the ``RunHandle`` so the API layer can list / create / resume /
delete from the same source of truth as the chat stream.
"""

from __future__ import annotations

import json
import logging
import shutil
import time
import uuid
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator

from ..highlights._context import MatchContext

logger = logging.getLogger(__name__)


CHAT_SESSIONS_SUBDIR = "chat_sessions"
DEFAULT_TITLE = "New chat"
SESSION_ID_LEN = 12
# Cap on persisted titles. Was 80 — that turned long Chinese first-turn
# prompts (each char roughly 2× as wide as Latin) into tooltips that
# blew out the chat dropdown. 40 chars is enough to disambiguate
# sessions without dominating the UI.
TITLE_MAX_LEN = 40


def _new_session_id() -> str:
    """URL-safe short id; collision space is fine for a per-run namespace."""
    return uuid.uuid4().hex[:SESSION_ID_LEN]


def _now() -> float:
    return time.time()


# ---------------------------------------------------------------------------
# Disk persistence
# ---------------------------------------------------------------------------


@dataclass
class SessionMeta:
    """The bits we persist about a session in meta.json.

    ``agentcore_session_id`` is the live sandbox id from the last chat
    turn. Keeping it lets a Resume re-attach to the still-warm MicroVM
    (no cold start, no data re-upload) when AgentCore hasn't reaped it
    yet — and the dead-session fallback in CodeSandbox handles the
    case where it has.
    """
    session_id: str
    title: str
    created_at: float
    updated_at: float
    agentcore_session_id: str | None = None
    # Free-form notes the UI may want to surface later (e.g. last
    # current_time we sent so we can resume the video at that mark).
    extra: dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> dict[str, Any]:
        return {
            "session_id": self.session_id,
            "title": self.title,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "agentcore_session_id": self.agentcore_session_id,
            "extra": self.extra,
        }

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> SessionMeta:
        return cls(
            session_id=data["session_id"],
            title=data.get("title") or DEFAULT_TITLE,
            created_at=float(data.get("created_at", _now())),
            updated_at=float(data.get("updated_at", _now())),
            agentcore_session_id=data.get("agentcore_session_id"),
            extra=data.get("extra") or {},
        )


# ---------------------------------------------------------------------------
# Store
# ---------------------------------------------------------------------------


class SessionStore:
    """Filesystem-backed session list for one run.

    Caches a small set of warm engines so back-to-back turns in the
    same session don't pay boto3 startup. Eviction is LRU when more
    than ``engine_cache_size`` sessions are active.
    """

    def __init__(
        self,
        run_name: str,
        run_dir: Path,
        ctx: MatchContext,
        artifact_root: Path,
        artifact_url_prefix_root: str,
        engine_factory,
        *,
        engine_cache_size: int = 4,
    ) -> None:
        self.run_name = run_name
        self.run_dir = run_dir
        self.ctx = ctx
        self.sessions_dir = run_dir / CHAT_SESSIONS_SUBDIR
        self.sessions_dir.mkdir(parents=True, exist_ok=True)
        self.artifact_root = artifact_root  # workspace/chat_artifacts/<run>
        self.artifact_url_prefix_root = artifact_url_prefix_root  # /chat_artifacts/<run>
        self._engine_factory = engine_factory
        self._engines: OrderedDict[str, Any] = OrderedDict()
        self._engine_cache_size = engine_cache_size

    # ---- meta / dirs ----------------------------------------------------

    def _session_dir(self, sid: str) -> Path:
        return self.sessions_dir / sid

    def _meta_path(self, sid: str) -> Path:
        return self._session_dir(sid) / "meta.json"

    def _messages_path(self, sid: str) -> Path:
        return self._session_dir(sid) / "messages.jsonl"

    def _artifact_dir(self, sid: str) -> Path:
        return self.artifact_root / sid

    def _artifact_url_prefix(self, sid: str) -> str:
        return f"{self.artifact_url_prefix_root}/{sid}"

    # ---- list / create / get / delete -----------------------------------

    def list(self) -> list[SessionMeta]:
        out: list[SessionMeta] = []
        if not self.sessions_dir.is_dir():
            return out
        for d in self.sessions_dir.iterdir():
            if not d.is_dir():
                continue
            mp = d / "meta.json"
            if not mp.exists():
                continue
            try:
                out.append(SessionMeta.from_json(json.loads(mp.read_text())))
            except Exception:  # noqa: BLE001
                logger.exception("skipping unreadable session: %s", d)
        out.sort(key=lambda m: m.updated_at, reverse=True)
        return out

    def create(self, title: str | None = None) -> SessionMeta:
        sid = _new_session_id()
        # Defensive: re-roll on the (vanishing) chance of collision.
        while self._meta_path(sid).exists():
            sid = _new_session_id()
        meta = SessionMeta(
            session_id=sid,
            title=(title or DEFAULT_TITLE).strip()[:TITLE_MAX_LEN] or DEFAULT_TITLE,
            created_at=_now(),
            updated_at=_now(),
        )
        self._session_dir(sid).mkdir(parents=True, exist_ok=True)
        self._artifact_dir(sid).mkdir(parents=True, exist_ok=True)
        self._write_meta(meta)
        # Empty messages file so subsequent appends don't have to mkdir/check.
        self._messages_path(sid).touch()
        return meta

    def get_meta(self, sid: str) -> SessionMeta:
        mp = self._meta_path(sid)
        if not mp.exists():
            raise KeyError(sid)
        return SessionMeta.from_json(json.loads(mp.read_text()))

    def update_meta(
        self,
        sid: str,
        *,
        title: str | None = None,
        agentcore_session_id: str | None = "__keep__",
    ) -> SessionMeta:
        meta = self.get_meta(sid)
        if title is not None:
            meta.title = title.strip()[:TITLE_MAX_LEN] or meta.title
        if agentcore_session_id != "__keep__":
            meta.agentcore_session_id = agentcore_session_id
        meta.updated_at = _now()
        self._write_meta(meta)
        return meta

    def delete(self, sid: str) -> None:
        # Drop the live engine FIRST so its sandbox is stopped before we
        # rm the meta that knows the AgentCore session id.
        eng = self._engines.pop(sid, None)
        if eng is not None:
            try:
                eng.close()
            except Exception:  # noqa: BLE001
                logger.exception("session %s engine close failed", sid)
        sd = self._session_dir(sid)
        if sd.exists():
            shutil.rmtree(sd, ignore_errors=True)
        ad = self._artifact_dir(sid)
        if ad.exists():
            shutil.rmtree(ad, ignore_errors=True)

    # ---- messages -------------------------------------------------------

    def list_messages(self, sid: str) -> list[dict[str, Any]]:
        mp = self._messages_path(sid)
        if not mp.exists():
            return []
        out: list[dict[str, Any]] = []
        with mp.open() as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    out.append(json.loads(line))
                except Exception:  # noqa: BLE001
                    logger.warning(
                        "skipping malformed history line in %s", mp)
        return out

    # Roles we'll let through ``append_message`` and that ``llm_history``
    # filters down to the conversational subset.
    USER_ROLES = ("user", "assistant")
    TOOL_ROLES = ("tool_use", "tool_result")

    def append_message(self, sid: str, message: dict[str, Any]) -> None:
        """Append one message and bump updated_at on the meta.

        Accepts both conversational turns (``role: user|assistant`` with
        ``content``) and tool breadcrumbs (``role: tool_use|tool_result``
        with ``id`` + ``name`` + ``input``/``result``). Tool entries are
        UI-only — ``llm_history`` strips them before they reach the model.
        """
        role = message.get("role")
        if role in self.USER_ROLES:
            if "content" not in message:
                raise ValueError("user/assistant message needs content")
        elif role in self.TOOL_ROLES:
            if "id" not in message or "name" not in message:
                raise ValueError("tool message needs id + name")
        else:
            raise ValueError(f"unknown message role: {role!r}")
        mp = self._messages_path(sid)
        mp.parent.mkdir(parents=True, exist_ok=True)
        with mp.open("a") as f:
            f.write(json.dumps(message, ensure_ascii=False) + "\n")
        # Bump updated_at; leave title untouched.
        meta = self.get_meta(sid)
        meta.updated_at = _now()
        self._write_meta(meta)

    def llm_history(self, sid: str) -> list[dict[str, Any]]:
        """Return only the conversational turns to feed back to Bedrock.

        Tool breadcrumbs (``role: tool_use|tool_result``) are persisted
        for the UI to replay but must NOT be fed back to the model as
        history — they're not in Anthropic's expected conversation
        format and would either be rejected or confuse it. The model
        re-issues its own tool calls each turn.
        """
        return [
            {"role": m["role"], "content": m.get("content", "")}
            for m in self.list_messages(sid)
            if m.get("role") in self.USER_ROLES
        ]

    def maybe_autotitle(self, sid: str, first_user_text: str) -> None:
        """If the session is still 'New chat', use the first user message as the title."""
        meta = self.get_meta(sid)
        if meta.title != DEFAULT_TITLE:
            return
        new_title = (
            first_user_text.strip().splitlines()[0][:TITLE_MAX_LEN]
            or DEFAULT_TITLE
        )
        meta.title = new_title
        meta.updated_at = _now()
        self._write_meta(meta)

    # ---- engine ---------------------------------------------------------

    def engine(self, sid: str):
        """Return (and lazily build) the chat engine bound to *sid*.

        The engine holds the live AgentCore sandbox; we cache up to
        ``engine_cache_size`` of these per run so back-to-back turns
        don't pay boto3 startup or data re-upload.
        """
        if sid in self._engines:
            self._engines.move_to_end(sid)
            return self._engines[sid]
        # Validate that the session exists on disk.
        self.get_meta(sid)

        engine = self._engine_factory(
            sid=sid,
            artifact_dir=self._artifact_dir(sid),
            artifact_url_prefix=self._artifact_url_prefix(sid),
            agentcore_session_id=self.get_meta(sid).agentcore_session_id,
        )
        # Wrap close to also persist the AgentCore session id last seen.
        # This is what lets a future Resume re-attach to the live MicroVM.
        store = self
        orig_close = engine.close

        def _close_and_persist() -> None:
            try:
                acs = getattr(engine, "_sandbox", None)
                if acs is not None:
                    sid_ac = getattr(acs, "_session_id", None)
                    if sid_ac is not None:
                        try:
                            store.update_meta(
                                sid, agentcore_session_id=sid_ac)
                        except Exception:  # noqa: BLE001
                            logger.exception(
                                "failed to persist agentcore_session_id "
                                "for %s", sid)
            finally:
                orig_close()

        engine.close = _close_and_persist  # type: ignore[method-assign]

        self._engines[sid] = engine
        while len(self._engines) > self._engine_cache_size:
            old_sid, old_eng = self._engines.popitem(last=False)
            logger.info("evicting chat engine for session %s", old_sid)
            try:
                old_eng.close()
            except Exception:  # noqa: BLE001
                logger.exception(
                    "engine close failed during eviction (%s)", old_sid)
        return engine

    def persist_agentcore_session(self, sid: str) -> None:
        """Snapshot the live engine's AgentCore session id into meta.json.

        Called after every successful chat turn so a server restart
        doesn't lose the warm sandbox id. Cheap (just rewrites
        meta.json with one extra string).
        """
        eng = self._engines.get(sid)
        if eng is None:
            return
        sb = getattr(eng, "_sandbox", None)
        if sb is None:
            return
        ac_sid = getattr(sb, "_session_id", None)
        if ac_sid is None:
            return
        try:
            self.update_meta(sid, agentcore_session_id=ac_sid)
        except Exception:  # noqa: BLE001
            logger.exception(
                "failed to persist agentcore_session_id for %s", sid)

    def close_all(self) -> None:
        for sid, eng in self._engines.items():
            try:
                eng.close()
            except Exception:  # noqa: BLE001
                logger.exception("close failed for engine %s", sid)
        self._engines.clear()

    # ---- internals ------------------------------------------------------

    def _write_meta(self, meta: SessionMeta) -> None:
        mp = self._meta_path(meta.session_id)
        mp.parent.mkdir(parents=True, exist_ok=True)
        # Atomic write so a crash mid-write doesn't corrupt the file.
        tmp = mp.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(meta.to_json(), ensure_ascii=False, indent=2))
        tmp.replace(mp)
