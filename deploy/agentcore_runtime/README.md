# GoalInsight chat → AgentCore Runtime

Optional deployment that moves the chat agent (`goalinsight/web/chat.py`)
out of the local FastAPI process and into an AWS-managed AgentCore
Runtime container. The local web app stays in charge of everything else
(annotator, pipeline jobs, library, viewer, video file serving) and
proxies chat turns to the runtime via `bedrock-agentcore.InvokeAgentRuntime`.

When `GOALINSIGHT_AGENTCORE_RUNTIME_ARN` is unset the FastAPI app falls
back to running `ChatEngine` in-process exactly like before, so this is
opt-in.

## Layout

- `Dockerfile` — ARM64 image, Python 3.12, non-root, exposes 8080.
- `main.py` — FastAPI app implementing the AgentCore HTTP contract
  (`/invocations` SSE, `/ping`). Pulls run JSON from S3 on first call
  per session, then drives `ChatEngine.stream()`.
- `chat_app/` — flat copy of the chat path
  (`chat.py`, `match_tools.py`, `code_sandbox.py`, `_context.py`).
  Keeps the runtime image free of the heavy `goalinsight` package
  imports (torch, cv2, wandb).
- `requirements.txt` — `fastapi`, `uvicorn`, `boto3`, `pydantic`. That's it.
- `setup_aws.sh` / `build_and_push.sh` / `create_runtime.sh` /
  `sync_run.sh` — deployment scripts.

## One-shot deploy

```bash
export AWS_REGION=us-east-1
export GOALINSIGHT_S3_BUCKET=goalinsight-pipeline-<account>

bash deploy/agentcore_runtime/deploy.sh
# optionally sync a run in the same step:
bash deploy/agentcore_runtime/deploy.sh --sync workspace/runs/<run_name>
```

`deploy.sh` chains the four scripts below and is safe to re-run — it
auto-detects an existing runtime and switches `create` → `update` so a
second invocation rolls a new image instead of failing. The runtime
ARN lands in `.agentcore_runtime_arn` at the repo root.

If you'd rather drive the steps yourself:

```bash
bash deploy/agentcore_runtime/setup_aws.sh        # ECR repo + IAM role
bash deploy/agentcore_runtime/build_and_push.sh   # build & push ARM64 image
bash deploy/agentcore_runtime/create_runtime.sh   # create (or 'update')
```

## Per-run

Each run's pipeline output JSON has to live in S3 before the runtime
can answer questions about it.

```bash
bash deploy/agentcore_runtime/sync_run.sh workspace/runs/<run_name>
```

This uploads only the eight JSON files the chat path reads — no video,
no weights, no vis JPGs.

## Running the FastAPI app against the runtime

```bash
export GOALINSIGHT_S3_BUCKET=goalinsight-pipeline-<account>
export GOALINSIGHT_AGENTCORE_RUNTIME_ARN="$(cat .agentcore_runtime_arn)"
export GOALINSIGHT_AGENTCORE_SESSION_SALT="$(cat .agentcore_session_salt)"

goalinsight-web --workspace ./workspace
```

Open `/insights/<run_name>` in the browser — chat is now served by the
runtime; analytics, viewer, and everything else is unchanged.

Unset `GOALINSIGHT_AGENTCORE_RUNTIME_ARN` to revert to local chat.

### Why the session salt matters

AgentCore Runtime keeps each `runtimeSessionId`'s MicroVM warm for 15
minutes after the last call. When you re-deploy a new image, in-flight
sessions from the *previous* image stick around — same session id ⇒
same (now stale) MicroVM, silently running the old code until it idles
out.

`deploy.sh` writes the runtime version into `.agentcore_session_salt`;
`session_id_for()` mixes the salt into the session id so a deploy
automatically routes you to a fresh MicroVM. Sourcing the file (as
above) is the recommended way; if you forget, the worst case is "old
code keeps running for ≤15 min after a deploy".

## Updating the runtime

Re-build, push, then:

```bash
bash deploy/agentcore_runtime/create_runtime.sh update
```

## Known limits

- **`run_python` is disabled** when chat runs inside the runtime
  container. The Code Interpreter session works, but plot artifacts
  produced by Claude have nowhere to land that the browser can fetch.
  Re-enabling requires plumbing an S3 artifact prefix through and
  serving it from the FastAPI app. See `main.py` `_ensure_session` for
  where to wire it.
- **Video** stays on the FastAPI side — the runtime never sees it.
- **Run output sync is manual** — there is no auto-upload hook.
