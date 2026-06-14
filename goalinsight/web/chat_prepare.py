"""Chat session preparation: warm the runtime before the user types.

When the chat agent is hosted on AgentCore Runtime, the first turn pays
two costs the user shouldn't see as a long silence:

1. Upload this run's pipeline-output JSON to S3 (the runtime container
   reads from S3, not the local filesystem).
2. Invoke the runtime once to spin up the MicroVM and have it download
   the JSON files into its temp dir.

This module exposes ``stream_prepare(run_handle)`` which yields a
sequence of progress events. The FastAPI endpoint pipes those events
to the browser as SSE so the chat UI can show a "preparing" indicator
and only enable input once it's done.

When ``GOALINSIGHT_AGENTCORE_RUNTIME_ARN`` is unset (i.e. chat runs
locally in-process), preparation is a no-op — we emit a single
``ready`` event so the front-end logic is the same in either mode.
"""

from __future__ import annotations

import json
import logging
import os
from collections.abc import Iterator
from pathlib import Path
from typing import Any

try:
    import boto3
    from botocore.config import Config as BotoConfig
    from botocore.exceptions import ClientError
except ImportError:
    boto3 = None  # type: ignore[assignment]
    BotoConfig = None  # type: ignore[assignment]

from .chat_remote import (
    runtime_arn_from_env,
    s3_bucket_from_env,
    s3_prefix_for,
    session_id_for,
)

logger = logging.getLogger(__name__)


# Files the runtime container pulls back down from S3. Must stay in sync
# with deploy/agentcore_runtime/main.py:RUN_FILES.
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


def _event(stage: str, percent: int, detail: str = "") -> dict[str, Any]:
    return {"stage": stage, "percent": percent, "detail": detail}


def _upload_run_to_s3(
    run_dir: Path, bucket: str, prefix: str, region: str,
) -> Iterator[dict[str, Any]]:
    """Upload RUN_FILES from *run_dir* to ``s3://bucket/prefix/``.

    Skips files that don't exist locally or already match by ETag (size
    + checksum). Yields per-file progress events.
    """
    s3 = boto3.client("s3", region_name=region)

    targets: list[tuple[str, Path]] = []
    for rel in RUN_FILES:
        local = run_dir / rel
        if local.exists():
            targets.append((rel, local))
    total = len(targets)
    if total == 0:
        yield _event("upload", 30, "no run files to upload")
        return

    for i, (rel, local) in enumerate(targets, 1):
        key = f"{prefix.rstrip('/')}/{rel}"
        # Skip when the remote object exists and matches local size.
        # Saves us re-uploading on every page load.
        skip = False
        try:
            head = s3.head_object(Bucket=bucket, Key=key)
            if head.get("ContentLength") == local.stat().st_size:
                skip = True
        except ClientError:
            pass

        if not skip:
            s3.upload_file(str(local), bucket, key)
        # Roughly map upload progress into 5..30%; the warm-up call
        # below covers 30..95.
        pct = 5 + int(25 * i / total)
        yield _event(
            "upload", pct,
            f"{'cached' if skip else 'uploaded'} {rel} ({i}/{total})",
        )


def _warmup_runtime(
    run_name: str,
    arn: str,
    bucket: str,
    prefix: str,
    region: str,
) -> Iterator[dict[str, Any]]:
    """Trigger the runtime to download our run JSON and build a session.

    We send a sentinel message with ``__warmup__`` content; the chat
    runtime will dispatch it to Claude as a tiny no-op turn, but the
    important side effect is that ``_ensure_session`` finishes (S3 sync
    + ChatEngine construction) by the time we return.
    """
    yield _event("warmup", 35, "starting runtime session")
    # Warmup is a small request but the cold MicroVM startup + S3 sync
    # easily exceed the 60s default; mirror RemoteChatEngine's config
    # so we don't time out the prepare flow on first use.
    cfg = BotoConfig(
        connect_timeout=10,
        read_timeout=int(
            os.environ.get("GOALINSIGHT_AGENTCORE_READ_TIMEOUT_S", "600")
        ),
        retries={"max_attempts": 1, "mode": "standard"},
    )
    client = boto3.client("bedrock-agentcore", region_name=region, config=cfg)
    payload = json.dumps({
        "run_name": run_name,
        "s3_bucket": bucket,
        "s3_prefix": prefix,
        "messages": [{"role": "user", "content": "__warmup__"}],
        "current_time": 0.0,
    }).encode("utf-8")

    resp = client.invoke_agent_runtime(
        agentRuntimeArn=arn,
        qualifier="DEFAULT",
        runtimeSessionId=session_id_for(run_name),
        payload=payload,
        contentType="application/json",
        accept="text/event-stream",
    )
    # Drain the stream — we don't care about the model's reply, only
    # that _ensure_session ran to completion. As soon as we see any
    # data frame the MicroVM is up and the session is initialized.
    body = resp["response"]
    chunks = body.iter_chunks() if hasattr(body, "iter_chunks") else body
    saw_data = False
    for chunk in chunks:
        if chunk:
            saw_data = True
            break
    if not saw_data:
        yield _event(
            "warmup", 95,
            "runtime returned empty stream (continuing anyway)",
        )
    else:
        yield _event("warmup", 95, "session initialized")


def stream_prepare(
    run_name: str,
    run_dir: Path,
    region: str = "us-east-1",
) -> Iterator[dict[str, Any]]:
    """Yield progress events for preparing the chat session.

    Sequence:
        {stage:"start", percent:0}
        ...zero or more {stage:"upload", ...}
        ...zero or more {stage:"warmup", ...}
        {stage:"ready", percent:100}

    On error: ``{stage:"error", percent:0, detail:"..."}`` and stop.
    """
    arn = runtime_arn_from_env()
    if arn is None:
        # Local mode — nothing to prepare. The chat engine boots
        # in-process on the first turn, but it's fast (couple hundred
        # ms) and doesn't need progress UI.
        yield _event("ready", 100, "local chat (no runtime)")
        return

    if boto3 is None:
        yield _event("error", 0, "boto3 not installed")
        return

    bucket = s3_bucket_from_env()
    if not bucket:
        yield _event(
            "error", 0,
            "GOALINSIGHT_AGENTCORE_RUNTIME_ARN is set but "
            "GOALINSIGHT_S3_BUCKET is not — can't sync run outputs.",
        )
        return

    prefix = s3_prefix_for(run_name)
    yield _event("start", 0, f"preparing session for {run_name}")

    try:
        yield from _upload_run_to_s3(run_dir, bucket, prefix, region)
    except Exception as exc:  # noqa: BLE001
        logger.exception("upload to s3 failed")
        yield _event("error", 0, f"upload failed: {exc}")
        return

    try:
        yield from _warmup_runtime(run_name, arn, bucket, prefix, region)
    except Exception as exc:  # noqa: BLE001
        logger.exception("runtime warmup failed")
        yield _event("error", 0, f"warmup failed: {exc}")
        return

    yield _event("ready", 100, "ready")
