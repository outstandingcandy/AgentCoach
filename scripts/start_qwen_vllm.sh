#!/usr/bin/env bash
# Launch Qwen3.6-27B-FP8 (image-text-to-text) on a local vLLM server.
#
# This script starts vLLM's OpenAI-compatible API server bound to
# localhost:8000.  The track_consolidation stage can then call it by
# setting:
#
#   track_consolidation:
#     jersey:
#       backend: qwen
#       model_id: "Qwen/Qwen3.6-27B-FP8"
#       base_url: "http://localhost:8000/v1"
#
# Requirements:
#   * vllm installed in the project's venv (or a dedicated env):
#       pip install "vllm>=0.11" "openai>=1"
#   * Enough GPU memory for a 27B FP8 MoE model — typically ≥48 GB
#     (one H100/A100-80G is comfortable; one L40S/A6000 is tight).
#   * HuggingFace token set via HF_TOKEN if the weights require gated
#     access.

set -euo pipefail

MODEL="${QWEN_MODEL:-Qwen/Qwen3.6-27B-FP8}"
HOST="${QWEN_HOST:-0.0.0.0}"
PORT="${QWEN_PORT:-8000}"
# L40S 46 GB: 27B FP8 MoE weights ~30 GB, reserve ~12 GB for KV cache.
# Higher GPU_UTIL + smaller MAX_MODEL_LEN / MAX_NUM_SEQS fit better.
GPU_UTIL="${QWEN_GPU_UTIL:-0.95}"
MAX_MODEL_LEN="${QWEN_MAX_LEN:-8192}"
MAX_NUM_SEQS="${QWEN_MAX_NUM_SEQS:-4}"
# Tensor parallel across multiple GPUs on this host (1 = single GPU).
TP="${QWEN_TP:-1}"
# Optional: cap the number of image tokens per request. Qwen3.6 uses
# dynamic patching so this is mostly a safety rail. Must be a JSON
# object string since vLLM 0.20.
LIMIT_MM="${QWEN_MM_LIMIT:-{\"image\": 30}}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

if [[ -f "$REPO_ROOT/venv/bin/activate" ]]; then
  # shellcheck disable=SC1091
  source "$REPO_ROOT/venv/bin/activate"
fi

if ! python -c "import vllm" >/dev/null 2>&1; then
  echo "ERROR: vllm is not installed in this environment."
  echo "       pip install 'vllm>=0.11' 'openai>=1'"
  exit 1
fi

echo "=== Qwen vLLM server ===" >&2
echo "  model:          $MODEL" >&2
echo "  bind:           $HOST:$PORT" >&2
echo "  tensor_parallel: $TP" >&2
echo "  max_model_len:  $MAX_MODEL_LEN" >&2
echo "  max_num_seqs:   $MAX_NUM_SEQS" >&2
echo "  gpu_memory_util: $GPU_UTIL" >&2
echo "  mm_limit:       $LIMIT_MM" >&2
echo

exec vllm serve "$MODEL" \
  --host "$HOST" \
  --port "$PORT" \
  --served-model-name "$MODEL" \
  --tensor-parallel-size "$TP" \
  --gpu-memory-utilization "$GPU_UTIL" \
  --max-model-len "$MAX_MODEL_LEN" \
  --max-num-seqs "$MAX_NUM_SEQS" \
  --limit-mm-per-prompt "$LIMIT_MM" \
  --trust-remote-code
