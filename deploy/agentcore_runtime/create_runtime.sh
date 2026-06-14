#!/usr/bin/env bash
# Create (or update) the AgentCore Runtime that hosts the chat agent.
#
# Usage:
#   bash deploy/agentcore_runtime/create_runtime.sh         # create
#   bash deploy/agentcore_runtime/create_runtime.sh update  # update existing
#
# After success, exports the new ARN both to stdout and into
# .agentcore_runtime_arn at the repo root for the FastAPI app to source.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

ACTION="${1:-create}"
REGION="${AWS_REGION:-us-east-1}"
ACCOUNT_ID="$(aws sts get-caller-identity --query Account --output text)"
RUNTIME_NAME="${GOALINSIGHT_RUNTIME_NAME:-goalinsight_chat_runtime}"
ROLE_NAME="${GOALINSIGHT_RUNTIME_ROLE_NAME:-goalinsight-chat-runtime-role}"
REPO_NAME="${GOALINSIGHT_RUNTIME_ECR_REPO:-goalinsight-chat-runtime}"
TAG="${GOALINSIGHT_RUNTIME_IMAGE_TAG:-latest}"
BUCKET_NAME="${GOALINSIGHT_S3_BUCKET:-goalinsight-pipeline-${ACCOUNT_ID}}"

ROLE_ARN="arn:aws:iam::${ACCOUNT_ID}:role/${ROLE_NAME}"
IMAGE_URI="${ACCOUNT_ID}.dkr.ecr.${REGION}.amazonaws.com/${REPO_NAME}:${TAG}"

ENV_JSON=$(cat <<JSON
{
  "AWS_REGION": "${REGION}",
  "GOALINSIGHT_S3_BUCKET": "${BUCKET_NAME}"
}
JSON
)

ARTIFACT_JSON=$(cat <<JSON
{"containerConfiguration": {"containerUri": "${IMAGE_URI}"}}
JSON
)

NETWORK_JSON='{"networkMode": "PUBLIC"}'

if [[ "$ACTION" == "create" ]]; then
  echo "Creating runtime $RUNTIME_NAME with image $IMAGE_URI"
  RESP=$(aws bedrock-agentcore-control create-agent-runtime \
    --region "$REGION" \
    --agent-runtime-name "$RUNTIME_NAME" \
    --agent-runtime-artifact "$ARTIFACT_JSON" \
    --network-configuration "$NETWORK_JSON" \
    --role-arn "$ROLE_ARN" \
    --protocol-configuration '{"serverProtocol": "HTTP"}' \
    --environment-variables "$ENV_JSON")
elif [[ "$ACTION" == "update" ]]; then
  EXISTING_ID=$(aws bedrock-agentcore-control list-agent-runtimes \
    --region "$REGION" \
    --query "agentRuntimes[?agentRuntimeName=='${RUNTIME_NAME}'].agentRuntimeId | [0]" \
    --output text)
  if [[ -z "$EXISTING_ID" || "$EXISTING_ID" == "None" ]]; then
    echo "No runtime named $RUNTIME_NAME found — run with 'create' first." >&2
    exit 1
  fi
  echo "Updating runtime $RUNTIME_NAME (id=$EXISTING_ID) -> $IMAGE_URI"
  RESP=$(aws bedrock-agentcore-control update-agent-runtime \
    --region "$REGION" \
    --agent-runtime-id "$EXISTING_ID" \
    --agent-runtime-artifact "$ARTIFACT_JSON" \
    --network-configuration "$NETWORK_JSON" \
    --role-arn "$ROLE_ARN" \
    --protocol-configuration '{"serverProtocol": "HTTP"}' \
    --environment-variables "$ENV_JSON")
else
  echo "Unknown action: $ACTION (expected 'create' or 'update')" >&2
  exit 2
fi

PARSED=$(echo "$RESP" | python3 -c 'import json,sys
d = json.load(sys.stdin)
print(d.get("agentRuntimeArn", ""))
print(d.get("agentRuntimeVersion", ""))
')
ARN=$(echo "$PARSED" | sed -n '1p')
VERSION=$(echo "$PARSED" | sed -n '2p')
if [[ -z "$ARN" ]]; then
  echo "Failed to parse agentRuntimeArn from response:" >&2
  echo "$RESP" >&2
  exit 3
fi
# Session salt = runtime version. Bumps with every create/update,
# so FastAPI's session_id_for() picks a different MicroVM after a roll
# instead of sticking to the still-warm pre-update VM (which would
# silently keep running the old code until idle-out).
SALT="${VERSION:-1}"

echo "$ARN" > .agentcore_runtime_arn
echo "$SALT" > .agentcore_session_salt
echo
echo "================================================================"
echo "Runtime ARN: $ARN"
echo "Version:     $VERSION   (session salt = $SALT)"
echo "Saved to: $REPO_ROOT/.agentcore_runtime_arn"
echo "          $REPO_ROOT/.agentcore_session_salt"
echo
echo "Enable runtime-backed chat in the FastAPI app:"
echo "  export GOALINSIGHT_AGENTCORE_RUNTIME_ARN=\"$ARN\""
echo "  export GOALINSIGHT_S3_BUCKET=\"$BUCKET_NAME\""
echo "  export GOALINSIGHT_AGENTCORE_SESSION_SALT=\"$SALT\""
echo "================================================================"
