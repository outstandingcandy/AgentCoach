"""AgentCore Code Interpreter session manager.

Wraps StartCodeInterpreterSession / InvokeCodeInterpreter /
StopCodeInterpreterSession so the chat layer can run arbitrary Python
in a managed sandbox without minding session lifecycle.

The sandbox has a persistent filesystem across invokes, so we lazily
upload the match data files (tracks.json, ball_tracks.json,
events.json, team_assignments.json) the first time run_python is
called. Subsequent invokes inherit them. Match data is written to
/var/data/* — paths are surfaced to the LLM via the tool description.

Lifecycle:
- Session starts on first run_python call.
- Stays alive for sessionTimeoutSeconds (30 min default) of idle.
- close() is called from ChatEngine teardown / on app shutdown.
- If a session has expired (e.g. user idle > 30 min), the next call
  catches the resource-not-found error and starts a fresh one,
  re-uploading the data files.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Built-in AgentCore identifier — public, no provisioning required.
CODE_INTERPRETER_ID = "aws.codeinterpreter.v1"

# Where uploaded match data lands inside the sandbox. Surfaced to the
# LLM in the run_python tool description.
# Paths are relative to the sandbox cwd (writeFiles rejects absolute
# /var/* on aws.codeinterpreter.v1; uploads land beside the cwd).
SANDBOX_DATA_DIR = "data"
SANDBOX_OUTPUT_DIR = "output"


# Max value AgentCore Code Interpreter accepts (per service model).
# 8h is the longest sandbox you can keep alive on one StartSession call.
MAX_SESSION_TIMEOUT_S = 28800


@dataclass
class CodeSandbox:
    """Manages one AgentCore Code Interpreter session per ChatEngine.

    Construction is cheap — the AWS session only starts on first
    ``run`` call. ``close`` is idempotent.

    ``initial_session_id`` lets a caller re-attach to a still-warm
    AgentCore session (typical "Resume previous chat" path). On the
    first ``_invoke`` after construction, dead-session detection will
    transparently spawn a fresh sandbox if the inherited id has been
    reaped server-side. ``data_already_uploaded=True`` tells us not to
    re-upload data files when re-attaching to a session that already
    has them.
    """
    region: str = "us-east-1"
    # 8h is the AgentCore maximum; the previous 30-min default meant
    # any user pause longer than a coffee break burned a cold start
    # and a data re-upload on the next turn.
    session_timeout_s: int = MAX_SESSION_TIMEOUT_S
    pipeline_output_dir: Path | None = None
    initial_session_id: str | None = None
    data_already_uploaded: bool = False

    _client: Any = field(default=None, repr=False)
    _session_id: str | None = field(default=None, repr=False)
    _data_uploaded: bool = field(default=False, repr=False)

    def __post_init__(self) -> None:
        if self.initial_session_id:
            self._session_id = self.initial_session_id
            self._data_uploaded = self.data_already_uploaded

    # ------------------------------------------------------------------
    # Session lifecycle
    # ------------------------------------------------------------------

    def _ensure_client(self) -> Any:
        if self._client is None:
            import boto3
            self._client = boto3.client(
                "bedrock-agentcore", region_name=self.region,
            )
        return self._client

    def _ensure_session(self) -> str:
        if self._session_id is not None:
            self._ensure_client()
            return self._session_id
        self._ensure_client()
        resp = self._client.start_code_interpreter_session(
            codeInterpreterIdentifier=CODE_INTERPRETER_ID,
            name="goalinsight-chat",
            sessionTimeoutSeconds=self.session_timeout_s,
        )
        self._session_id = resp["sessionId"]
        logger.info("Code Interpreter session started: %s", self._session_id)
        return self._session_id

    def keepalive(self) -> bool:
        """Run a no-op invoke to confirm the session is alive.

        Returns ``True`` if the session is alive (or successfully
        revived); ``False`` only when there's no session at all yet.
        Used by SessionStore on Resume to detect — and respawn — a
        stale-but-cached AgentCore id before the user types anything.
        """
        if self._session_id is None and self.pipeline_output_dir is None:
            return False
        # _invoke handles the dead-session case for us.
        try:
            self._invoke(
                codeInterpreterIdentifier=CODE_INTERPRETER_ID,
                sessionId=self._ensure_session(),
                name="executeCode",
                arguments={"language": "python", "code": "1"},
            )
            return True
        except Exception:  # noqa: BLE001
            logger.exception("Code Interpreter keepalive failed")
            return False

    def close(self) -> None:
        if self._session_id is None or self._client is None:
            return
        try:
            self._client.stop_code_interpreter_session(
                codeInterpreterIdentifier=CODE_INTERPRETER_ID,
                sessionId=self._session_id,
            )
            logger.info("Code Interpreter session stopped: %s", self._session_id)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Failed to stop sandbox session: %s", exc)
        finally:
            self._session_id = None
            self._data_uploaded = False

    # ------------------------------------------------------------------
    # Data upload
    # ------------------------------------------------------------------

    # ``tracks.json`` and ``team_assignments.json`` are listed under
    # ``track_consolidation/`` first (the consolidated player-id-keyed
    # copies) and the raw tracker copies are the fallback. The first
    # available source wins per file. Other files are unique-source.
    DATA_FILES = [
        ("event_detection/events.json", "events.json", None),
        ("track_consolidation/tracks.json", "tracks.json",
         "tracking/tracks.json"),
        ("tracking/ball_tracks.json", "ball_tracks.json", None),
        ("track_consolidation/team_assignments.json", "team_assignments.json",
         "tracking/team_assignments.json"),
    ]

    def _ensure_data_uploaded(self) -> None:
        if self._data_uploaded or self.pipeline_output_dir is None:
            self._data_uploaded = True
            return

        sid = self._ensure_session()
        # Build content list. writeFiles supports either text or blob.
        content = []
        for entry in self.DATA_FILES:
            src_rel, sandbox_name, fallback_rel = entry
            src = self.pipeline_output_dir / src_rel
            if not src.exists() and fallback_rel:
                src = self.pipeline_output_dir / fallback_rel
            if not src.exists():
                logger.warning("data file missing, skipping upload: %s", src)
                continue
            content.append({
                "path": f"{SANDBOX_DATA_DIR}/{sandbox_name}",
                "text": src.read_text(),
            })
        if not content:
            self._data_uploaded = True
            return

        # writeFiles isn't streamed; we don't need the response body.
        resp = self._client.invoke_code_interpreter(
            codeInterpreterIdentifier=CODE_INTERPRETER_ID,
            sessionId=sid,
            name="writeFiles",
            arguments={"content": content},
        )
        for _ in resp["stream"]:
            pass

        # Make sure the output dir exists (writeFiles created data/ but
        # not output/).
        self.run_raw(
            f"import os; os.makedirs('{SANDBOX_OUTPUT_DIR}', exist_ok=True)"
        )
        self._data_uploaded = True
        logger.info(
            "Uploaded %d data files to sandbox", len(content),
        )

    # ------------------------------------------------------------------
    # Code execution
    # ------------------------------------------------------------------

    @staticmethod
    def _is_dead_session_error(exc: Exception) -> bool:
        """True when AgentCore says our session id is no longer alive.

        Sessions die when they hit ``sessionTimeoutSeconds`` of idle
        (default 30 min) — but ``_session_id`` stays cached on us, so
        the next invoke gets a ``ValidationException: Code interpreter
        session ... is not active``. We use this predicate to drop the
        cached id and start fresh on the retry.
        """
        msg = str(exc)
        return ("is not active" in msg) and ("session" in msg.lower())

    def _invoke(self, *, needs_data: bool = False, **kwargs: Any) -> Any:
        """invoke_code_interpreter with one transparent retry on dead session.

        On ``ValidationException: ... is not active`` we close the
        cached session, mark data as not-yet-uploaded (the new session
        starts with an empty FS), and run the call again. One retry —
        if the second attempt also fails the exception propagates.

        ``needs_data=True`` forces a re-upload of the match data files
        after a revival; pass it for any call that reads from
        ``data/*``. Internal housekeeping (clearing the output dir,
        listing the output dir) doesn't need the match data so leaves
        it False.
        """
        try:
            return self._client.invoke_code_interpreter(**kwargs)
        except Exception as exc:  # noqa: BLE001
            if not self._is_dead_session_error(exc):
                raise
            logger.info(
                "Code Interpreter session %s expired; starting a fresh one",
                self._session_id,
            )
            self._session_id = None
            self._data_uploaded = False
            kwargs["sessionId"] = self._ensure_session()
            if needs_data:
                self._ensure_data_uploaded()
            return self._client.invoke_code_interpreter(**kwargs)

    def run_raw(self, code: str, timeout_s: int = 60) -> dict[str, Any]:
        """Execute *code* in the sandbox without uploading data.

        Returns ``{stdout, stderr, exit_code, execution_time_s}``.
        """
        sid = self._ensure_session()
        resp = self._invoke(
            codeInterpreterIdentifier=CODE_INTERPRETER_ID,
            sessionId=sid,
            name="executeCode",
            arguments={"language": "python", "code": code},
        )
        out = {
            "stdout": "", "stderr": "",
            "exit_code": None, "execution_time_s": None,
        }
        for event in resp["stream"]:
            if "result" in event:
                sc = event["result"].get("structuredContent") or {}
                out["stdout"] = sc.get("stdout", "")
                out["stderr"] = sc.get("stderr", "")
                out["exit_code"] = sc.get("exitCode")
                out["execution_time_s"] = sc.get("executionTime")
        return out

    def run(
        self,
        code: str,
        artifact_dir: Path,
    ) -> dict[str, Any]:
        """Execute *code*, returning stdout/stderr and any new image artifacts.

        Mechanism: list /var/output before and after the code runs,
        download anything new via readFiles, save to *artifact_dir*,
        and return their relative paths in the result. Saves Claude
        from having to remember to do its own file packaging.

        Returns: ``{stdout, stderr, exit_code, execution_time_s,
        artifacts: [{path, mime_type}]}``. ``path`` is relative to
        ``artifact_dir``.
        """
        self._ensure_data_uploaded()

        # Clear output dir before execution. Earlier we diffed
        # listdir(post) - listdir(pre), but that misses files that
        # overwrite a same-named artifact from a prior call (the model
        # often re-runs a fixed-up script and re-saves
        # 'output/hist.png'). Clearing first means listdir(post) IS the
        # set of new files for this call.
        self.run_raw(
            f"import os, shutil; "
            f"shutil.rmtree('{SANDBOX_OUTPUT_DIR}', ignore_errors=True); "
            f"os.makedirs('{SANDBOX_OUTPUT_DIR}', exist_ok=True)"
        )

        # Re-resolve the session id AFTER the housekeeping call: if
        # that triggered a revival (dead session) the cached id has
        # been swapped, and using the stale local var would force a
        # second pointless revival. Also re-ensure data is uploaded
        # because the new session starts with an empty FS.
        self._ensure_data_uploaded()
        sid = self._ensure_session()

        resp = self._invoke(
            needs_data=True,
            codeInterpreterIdentifier=CODE_INTERPRETER_ID,
            sessionId=sid,
            name="executeCode",
            arguments={"language": "python", "code": code},
        )
        out: dict[str, Any] = {
            "stdout": "", "stderr": "",
            "exit_code": None, "execution_time_s": None,
            "artifacts": [],
        }
        for event in resp["stream"]:
            if "result" in event:
                sc = event["result"].get("structuredContent") or {}
                out["stdout"] = sc.get("stdout", "")
                out["stderr"] = sc.get("stderr", "")
                out["exit_code"] = sc.get("exitCode")
                out["execution_time_s"] = sc.get("executionTime")

        new_files = sorted(self._list_output_dir())
        if new_files:
            artifact_dir.mkdir(parents=True, exist_ok=True)
            artifacts = self._download_files(new_files, artifact_dir)
            out["artifacts"] = artifacts
        return out

    def _list_output_dir(self) -> list[str]:
        r = self.run_raw(
            "import os; "
            f"p='{SANDBOX_OUTPUT_DIR}'; "
            "print('\\n'.join(sorted(os.listdir(p))) "
            "if os.path.isdir(p) else '')"
        )
        return [ln for ln in (r["stdout"] or "").splitlines() if ln]

    def _download_files(
        self, names: list[str], dest_dir: Path,
    ) -> list[dict[str, str]]:
        """Pull files from sandbox output dir back to *dest_dir*."""
        sid = self._session_id
        resp = self._client.invoke_code_interpreter(
            codeInterpreterIdentifier=CODE_INTERPRETER_ID,
            sessionId=sid,
            name="readFiles",
            arguments={"paths": [f"{SANDBOX_OUTPUT_DIR}/{n}" for n in names]},
        )
        artifacts: list[dict[str, str]] = []
        for event in resp["stream"]:
            if "result" not in event:
                continue
            for item in event["result"].get("content", []):
                resource = item.get("resource") or {}
                blob = resource.get("blob")
                uri = resource.get("uri", "")
                mime = resource.get("mimeType", "application/octet-stream")
                if blob is None:
                    continue
                # uri looks like 'file:///var/output/foo.png' — keep the basename
                name = uri.rsplit("/", 1)[-1] or "artifact.bin"
                dst = dest_dir / name
                dst.write_bytes(blob)
                artifacts.append({"path": name, "mime_type": mime})
        return artifacts
