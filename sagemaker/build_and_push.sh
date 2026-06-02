#!/usr/bin/env bash
# Build the Processing Job image and push to ECR.
#
# Run from repo root:
#   bash sagemaker/build_and_push.sh
#
# The image targets linux/amd64 — SageMaker GPU instances are all amd64,
# so on an Apple Silicon dev machine we force the platform via buildx.
# On amd64 hosts buildx still works and produces the same artifact.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

REGION="${AWS_REGION:-us-east-1}"
ACCOUNT_ID="$(aws sts get-caller-identity --query Account --output text)"
REPO_NAME="${GOALINSIGHT_ECR_REPO:-goalinsight-pipeline}"
TAG="${1:-latest}"

ECR_URI="${ACCOUNT_ID}.dkr.ecr.${REGION}.amazonaws.com/${REPO_NAME}"
FULL_IMAGE="${ECR_URI}:${TAG}"

echo "Building $FULL_IMAGE"
echo

# ECR login.
aws ecr get-login-password --region "$REGION" \
  | docker login --username AWS --password-stdin "${ACCOUNT_ID}.dkr.ecr.${REGION}.amazonaws.com"

# buildx makes amd64 cross-build painless on M-series macs and is a
# no-op on amd64 hosts. --load to put the image into the local docker
# daemon so the subsequent push works.
docker buildx build \
  --platform linux/amd64 \
  -f sagemaker/docker/Dockerfile \
  -t "$FULL_IMAGE" \
  --load \
  .

docker push "$FULL_IMAGE"
echo
echo "Pushed: $FULL_IMAGE"
