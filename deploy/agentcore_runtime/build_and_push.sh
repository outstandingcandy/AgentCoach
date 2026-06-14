#!/usr/bin/env bash
# Build the AgentCore Runtime image (linux/arm64) and push to ECR.
#
# Run from repo root:
#   bash deploy/agentcore_runtime/build_and_push.sh [tag]
#
# AgentCore Runtime requires arm64 images, so this always cross-builds
# via buildx regardless of host arch.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

REGION="${AWS_REGION:-us-east-1}"
ACCOUNT_ID="$(aws sts get-caller-identity --query Account --output text)"
REPO_NAME="${GOALINSIGHT_RUNTIME_ECR_REPO:-goalinsight-chat-runtime}"
TAG="${1:-latest}"

ECR_URI="${ACCOUNT_ID}.dkr.ecr.${REGION}.amazonaws.com/${REPO_NAME}"
FULL_IMAGE="${ECR_URI}:${TAG}"

echo "Building $FULL_IMAGE (linux/arm64)"
echo

aws ecr get-login-password --region "$REGION" \
  | docker login --username AWS --password-stdin "${ACCOUNT_ID}.dkr.ecr.${REGION}.amazonaws.com"

# Multi-arch hosts: buildx with --load works only for the host arch, so
# we push directly from the build for arm64 to keep this single-step on
# any dev box.
docker buildx build \
  --platform linux/arm64 \
  -f deploy/agentcore_runtime/Dockerfile \
  -t "$FULL_IMAGE" \
  --push \
  deploy/agentcore_runtime

echo
echo "Pushed: $FULL_IMAGE"
echo
echo "Image digest:"
aws ecr describe-images \
  --repository-name "$REPO_NAME" \
  --image-ids imageTag="$TAG" \
  --region "$REGION" \
  --query 'imageDetails[0].imageDigest' --output text
