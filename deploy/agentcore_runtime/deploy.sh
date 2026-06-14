#!/usr/bin/env bash
# One-shot deploy of the GoalInsight chat AgentCore Runtime.
#
# Chains:
#   1. setup_aws.sh       (idempotent: ECR repo + IAM role)
#   2. build_and_push.sh  (ARM64 image to ECR)
#   3. create_runtime.sh  (create OR update the runtime)
#   4. sync_run.sh        (optional, when --sync <run_dir> is passed)
#
# Re-run any time. The first three steps are safe to repeat; step 3
# auto-detects an existing runtime by name and updates instead of
# creating.
#
# Usage:
#   bash deploy/agentcore_runtime/deploy.sh
#   bash deploy/agentcore_runtime/deploy.sh --sync workspace/runs/<run_name>
#   bash deploy/agentcore_runtime/deploy.sh --tag v2 --sync workspace/runs/foo
#
# After success the runtime ARN is in .agentcore_runtime_arn at the
# repo root; the script also prints the export commands the FastAPI
# app needs.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

# ---- args ------------------------------------------------------------------
TAG="latest"
SYNC_RUN_DIR=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --tag)   TAG="$2"; shift 2 ;;
    --sync)  SYNC_RUN_DIR="$2"; shift 2 ;;
    -h|--help)
      sed -n '2,20p' "$0" | sed 's/^# \?//'
      exit 0 ;;
    *)
      echo "Unknown flag: $1" >&2
      exit 2 ;;
  esac
done

# ---- preflight -------------------------------------------------------------
REGION="${AWS_REGION:-us-east-1}"
RUNTIME_NAME="${GOALINSIGHT_RUNTIME_NAME:-goalinsight_chat_runtime}"
ACCOUNT_ID="$(aws sts get-caller-identity --query Account --output text 2>/dev/null || true)"
if [[ -z "$ACCOUNT_ID" ]]; then
  echo "aws sts get-caller-identity failed — check your AWS credentials." >&2
  exit 1
fi
BUCKET_NAME="${GOALINSIGHT_S3_BUCKET:-goalinsight-pipeline-${ACCOUNT_ID}}"

echo "=========================================================="
echo "  GoalInsight chat → AgentCore Runtime"
echo "  region:   $REGION"
echo "  account:  $ACCOUNT_ID"
echo "  runtime:  $RUNTIME_NAME"
echo "  bucket:   $BUCKET_NAME"
echo "  tag:      $TAG"
[[ -n "$SYNC_RUN_DIR" ]] && echo "  sync:     $SYNC_RUN_DIR"
echo "=========================================================="
echo

# ---- 1. ECR + IAM ----------------------------------------------------------
echo "[1/3] AWS resources"
GOALINSIGHT_S3_BUCKET="$BUCKET_NAME" \
  bash deploy/agentcore_runtime/setup_aws.sh
echo

# ---- 2. build & push -------------------------------------------------------
echo "[2/3] build & push image (linux/arm64, tag=$TAG)"
bash deploy/agentcore_runtime/build_and_push.sh "$TAG"
echo

# ---- 3. create OR update runtime -------------------------------------------
echo "[3/3] create/update runtime"
EXISTING_ID=$(aws bedrock-agentcore-control list-agent-runtimes \
  --region "$REGION" \
  --query "agentRuntimes[?agentRuntimeName=='${RUNTIME_NAME}'].agentRuntimeId | [0]" \
  --output text 2>/dev/null || true)
if [[ -n "$EXISTING_ID" && "$EXISTING_ID" != "None" ]]; then
  ACTION="update"
else
  ACTION="create"
fi
echo "  action: $ACTION"
GOALINSIGHT_RUNTIME_IMAGE_TAG="$TAG" \
GOALINSIGHT_S3_BUCKET="$BUCKET_NAME" \
  bash deploy/agentcore_runtime/create_runtime.sh "$ACTION"

ARN=""
SALT=""
[[ -f .agentcore_runtime_arn ]] && ARN="$(cat .agentcore_runtime_arn)"
[[ -f .agentcore_session_salt ]] && SALT="$(cat .agentcore_session_salt)"

# ---- 4. optional run sync --------------------------------------------------
if [[ -n "$SYNC_RUN_DIR" ]]; then
  echo
  echo "[+] sync run outputs to S3"
  if [[ ! -d "$SYNC_RUN_DIR" ]]; then
    echo "  $SYNC_RUN_DIR is not a directory — skipping" >&2
  else
    GOALINSIGHT_S3_BUCKET="$BUCKET_NAME" \
      bash deploy/agentcore_runtime/sync_run.sh "$SYNC_RUN_DIR"
  fi
fi

echo
echo "=========================================================="
echo "Done."
[[ -n "$ARN" ]] && echo "Runtime ARN:  $ARN"
[[ -n "$SALT" ]] && echo "Session salt: $SALT  (bumps each deploy → forces fresh MicroVM)"
echo
echo "Enable runtime-backed chat in the FastAPI app:"
echo "  export GOALINSIGHT_S3_BUCKET=$BUCKET_NAME"
[[ -n "$ARN" ]] && echo "  export GOALINSIGHT_AGENTCORE_RUNTIME_ARN=\"$ARN\""
[[ -n "$SALT" ]] && echo "  export GOALINSIGHT_AGENTCORE_SESSION_SALT=\"$SALT\""
echo "  goalinsight-web --workspace ./workspace"
echo
echo "Or one-liner that picks both up automatically:"
echo "  GOALINSIGHT_S3_BUCKET=$BUCKET_NAME \\"
echo "    GOALINSIGHT_AGENTCORE_RUNTIME_ARN=\"\$(cat .agentcore_runtime_arn)\" \\"
echo "    GOALINSIGHT_AGENTCORE_SESSION_SALT=\"\$(cat .agentcore_session_salt)\" \\"
echo "    goalinsight-web --workspace ./workspace"
echo "=========================================================="
