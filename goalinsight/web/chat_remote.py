"""Remote chat engine: forwards a turn to AgentCore Runtime.

Drop-in replacement for ``ChatEngine`` (`respond` / `stream` / `close`)
that calls ``bedrock-agentcore.InvokeAgentRuntime`` instead of running
the LLM loop in-process. The runtime container is the one defined under
``deploy/agentcore_runtime/`` — same SSE shape it returns is what we
yield back to the browser, so the FastAPI proxy is a straight passthrough.

The local ChatEngine stays untouched and is still selected when
``GOALINSIGHT_AGENTCORE_RUNTIME_ARN`` is unset, so this is opt-in.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
from dataclasses import dataclass, field
from typing import Any, Iterator

try:
    import boto3
    from botocore.config import Config as BotoConfig
except ImportError:
    boto3 = None  # type: ignore[assignment]
    BotoConfig = None  # type: ignore[assignment]


# Default 60s read timeout is too short — a single chat turn that uses
# run_python with several tool rounds (each = Bedrock call + Code
# Interpreter execution) routinely takes 60-120s. Bump generously.
# Retries are disabled: a retry would re-bill Bedrock + start a fresh
# Code Interpreter session, almost never what you want for a paid LLM
# call that's already streaming bytes.
_INVOKE_READ_TIMEOUT_S = int(
    os.environ.get("GOALINSIGHT_AGENTCORE_READ_TIMEOUT_S", "600"),
)
_INVOKE_CONNECT_TIMEOUT_S = 10

from ..highlights._context import MatchContext

logger = logging.getLogger(__name__)


@dataclass
class RemoteChatEngine:
    """Same shape as ``ChatEngine`` but offloads inference to AgentCore Runtime.

    The runtime is identified by *agent_runtime_arn*; the run's pipeline
    output is identified by *s3_bucket* + *s3_prefix* and pulled by the
    runtime container on first call. *run_name* doubles as the AgentCore
    session id, so repeat calls reuse the warm in-VM ChatEngine.
    """

    ctx: MatchContext
    run_name: str
    agent_runtime_arn: str
    s3_bucket: str
    s3_prefix: str
    region: str = "us-east-1"
    qualifier: str = "DEFAULT"

    _client: Any = field(default=None, repr=False)

    def __post_init__(self) -> None:
        if boto3 is None:
            raise ImportError("boto3 is required; install with `pip install boto3`")
        cfg = BotoConfig(
            connect_timeout=_INVOKE_CONNECT_TIMEOUT_S,
            read_timeout=_INVOKE_READ_TIMEOUT_S,
            retries={"max_attempts": 1, "mode": "standard"},
        )
        self._client = boto3.client(
            "bedrock-agentcore", region_name=self.region, config=cfg,
        )

    def close(self) -> None:
        # Runtime sessions expire on their own (15-min idle / 8h max).
        # Calling DELETE /sessions/{id} on the runtime is best-effort and
        # not exposed via the AWS SDK; skipping is fine.
        return

    # ------------------------------------------------------------------
    # API parity with ChatEngine
    # ------------------------------------------------------------------

    def respond(
        self,
        messages: list[dict[str, str]],
        current_time: float,
    ) -> str:
        return "".join(self.stream(messages, current_time))

    def stream(
        self,
        messages: list[dict[str, str]],
        current_time: float,
    ) -> Iterator[str]:
        payload = json.dumps({
            "run_name": self.run_name,
            "s3_bucket": self.s3_bucket,
            "s3_prefix": self.s3_prefix,
            "messages": messages,
            "current_time": current_time,
        }).encode("utf-8")

        resp = self._client.invoke_agent_runtime(
            agentRuntimeArn=self.agent_runtime_arn,
            qualifier=self.qualifier,
            runtimeSessionId=session_id_for(self.run_name),
            payload=payload,
            contentType="application/json",
            accept="text/event-stream",
        )
        yield from _parse_sse(resp["response"])


def _parse_sse(stream: Any) -> Iterator[str]:
    """Yield ``delta`` strings from an SSE byte stream.

    Frames are ``data: {json}\\n\\n``. The runtime emits
    ``{"delta": "..."}`` for tokens and ``{"done": true}`` to terminate.
    Unknown / non-JSON frames are skipped silently.
    """
    buf = b""
    for chunk in stream.iter_chunks() if hasattr(stream, "iter_chunks") else stream:
        if not chunk:
            continue
        buf += chunk if isinstance(chunk, (bytes, bytearray)) else chunk.encode()
        while b"\n\n" in buf:
            frame, buf = buf.split(b"\n\n", 1)
            for line in frame.split(b"\n"):
                if not line.startswith(b"data:"):
                    continue
                body = line[5:].strip()
                if not body:
                    continue
                try:
                    obj = json.loads(body)
                except json.JSONDecodeError:
                    continue
                if obj.get("done"):
                    return
                if "error" in obj:
                    raise RuntimeError(f"runtime error: {obj['error']}")
                delta = obj.get("delta")
                if delta:
                    yield delta


def session_id_for(run_name: str) -> str:
    """AgentCore requires ``runtimeSessionId`` length 33..100 chars and
    ``[a-zA-Z0-9_\\-]``. Some run names are shorter than 33; pad with
    a deterministic SHA-256 prefix so a given run_name always maps to
    the same session id (so prepare/warmup hits the same MicroVM as
    the subsequent chat turns).

    To bust cached MicroVMs after a runtime image roll, set
    ``GOALINSIGHT_AGENTCORE_SESSION_SALT`` (any short string). Same-salt
    requests share a MicroVM; bumping the salt forces a new one and lets
    you skip waiting 15 min for the old session to idle out.
    """
    salt = os.environ.get("GOALINSIGHT_AGENTCORE_SESSION_SALT", "")
    base = f"{run_name}|{salt}" if salt else run_name
    safe = "".join(c if c.isalnum() or c in "_-" else "_" for c in base)
    if len(safe) >= 33:
        return safe[:100]
    digest = hashlib.sha256(base.encode()).hexdigest()
    return (safe + "_" + digest)[:100]


def runtime_arn_from_env() -> str | None:
    """Return the configured AgentCore Runtime ARN, or ``None`` if unset."""
    arn = os.environ.get("GOALINSIGHT_AGENTCORE_RUNTIME_ARN", "").strip()
    return arn or None


def s3_bucket_from_env() -> str | None:
    bucket = os.environ.get("GOALINSIGHT_S3_BUCKET", "").strip()
    return bucket or None


def s3_prefix_for(run_name: str) -> str:
    """Where the runtime expects this run's pipeline output to live.

    Override the layout via ``GOALINSIGHT_S3_RUN_PREFIX_FMT`` (Python
    format string with ``{run_name}``). Default: ``runs/<run_name>``.
    """
    fmt = os.environ.get(
        "GOALINSIGHT_S3_RUN_PREFIX_FMT", "runs/{run_name}",
    )
    return fmt.format(run_name=run_name)
