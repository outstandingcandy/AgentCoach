#!/usr/bin/env bash
# GoalInsight offline container entrypoint.
#
# - Seeds /workspace/configs/ with 3 starter templates (fifa / futsal /
#   children) on first launch so a brand-new workspace shows usable
#   options in the Library config dropdown without the user having to
#   ``docker cp`` anything.
# - Existing files are NEVER overwritten — users who edited their
#   templates keep their changes across container restarts.
# - Then exec's into goalinsight-web (default) or whatever the user
#   passed via ``docker run -- <cmd>``.

set -e

TEMPLATE_DIR=/opt/goalinsight/configs/templates
WORKSPACE=/workspace
WORKSPACE_CONFIGS="$WORKSPACE/configs"

# Filenames are 1:1 between the template library and the workspace
# seed, so no rename mapping is needed any more.
SEED_FILES=(fifa.yaml futsal.yaml children.yaml)

# Warn when /workspace lives inside the container's writable layer
# rather than a bind-mounted host directory. ``docker rm`` would
# throw away every pipeline output the user just produced, which is a
# common first-day footgun. ``findmnt`` returns success only when the
# path is a mount point; on the container layer it returns 2.
if command -v findmnt >/dev/null 2>&1; then
    if ! findmnt -n -T "$WORKSPACE" 2>/dev/null | grep -q "^$WORKSPACE "; then
        echo "================================================================" >&2
        echo "WARNING: /workspace is not a bind-mount. Pipeline outputs," >&2
        echo "uploaded videos, and annotations will live INSIDE the container" >&2
        echo "and disappear when you 'docker rm' it." >&2
        echo "Re-run with:  -v \"\$PWD/workspace:/workspace\"" >&2
        echo "================================================================" >&2
    fi
fi

# The Workspace.ensure() path inside goalinsight-web creates this
# dir, but it only fires AFTER exec — so we mkdir here so seeding can
# always run.
mkdir -p "$WORKSPACE_CONFIGS"
for name in "${SEED_FILES[@]}"; do
    src="$TEMPLATE_DIR/$name"
    dest="$WORKSPACE_CONFIGS/$name"
    if [ -f "$src" ] && [ ! -e "$dest" ]; then
        cp "$src" "$dest"
        echo "seeded $dest from $src"
    fi
done

# Background-launch the Qwen VL vLLM daemon on a fixed loopback port
# so the track_consolidation stage can just talk to it. First cold
# start (flashinfer JIT compile) takes 3-5 min; caching to
# ``~/.cache/flashinfer/`` means subsequent container restarts warm
# up in ~30s. Set QWEN_VLLM_DISABLE=1 to skip (e.g. no-GPU or
# no-track_consolidation deployments).
QWEN_VLLM_PORT="${QWEN_VLLM_PORT:-8100}"
QWEN_VLLM_MODEL="${QWEN_VLLM_MODEL:-Qwen/Qwen3.5-2B}"
if [ -z "${QWEN_VLLM_DISABLE:-}" ] && command -v vllm >/dev/null 2>&1; then
    mkdir -p /workspace/logs 2>/dev/null || true
    echo "starting vllm daemon: model=$QWEN_VLLM_MODEL port=$QWEN_VLLM_PORT" >&2
    vllm serve "$QWEN_VLLM_MODEL" \
        --host 127.0.0.1 \
        --port "$QWEN_VLLM_PORT" \
        --gpu-memory-utilization "${QWEN_GPU_UTIL:-0.25}" \
        --max-model-len "${QWEN_MAX_LEN:-8192}" \
        --max-num-seqs "${QWEN_MAX_NUM_SEQS:-4}" \
        > /workspace/logs/vllm.log 2>&1 &
    VLLM_PID=$!
    # Expose the base_url to the pipeline subprocess so it skips its
    # own spawn logic.
    export QWEN_VLLM_BASE_URL="http://127.0.0.1:$QWEN_VLLM_PORT/v1"
    echo "vllm pid=$VLLM_PID (log at /workspace/logs/vllm.log)" >&2
    # Reap the daemon on entrypoint exit so ``docker stop`` cleans up
    # cleanly instead of leaving zombies.
    trap 'kill -TERM "$VLLM_PID" 2>/dev/null || true' EXIT INT TERM
fi

# Hand off to goalinsight-web (default) or the user's override.
exec goalinsight-web "$@"
