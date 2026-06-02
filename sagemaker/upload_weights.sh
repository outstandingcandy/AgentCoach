#!/usr/bin/env bash
# Upload model weights the Processing Job entrypoints download at startup.
#
# Why not bake into the image: weights change far more often than the
# code/deps (you've already iterated several PnLCalib finetunes); rebuilding
# a multi-GB image each time is slow. Image stays small and stable; weights
# evolve in S3.
#
# Usage:
#   bash sagemaker/upload_weights.sh
#   bash sagemaker/upload_weights.sh --bucket my-other-bucket
#
# Requires that setup_aws.sh has run (or that you set GOALINSIGHT_S3_BUCKET).
set -euo pipefail

REGION="${AWS_REGION:-us-east-1}"
ACCOUNT_ID="$(aws sts get-caller-identity --query Account --output text)"
BUCKET_NAME="${GOALINSIGHT_S3_BUCKET:-goalinsight-pipeline-${ACCOUNT_ID}}"
PREFIX="weights"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --bucket) BUCKET_NAME="$2"; shift 2 ;;
    --prefix) PREFIX="$2"; shift 2 ;;
    *) echo "unknown arg: $1"; exit 1 ;;
  esac
done

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
S3_ROOT="s3://${BUCKET_NAME}/${PREFIX}"

echo "Uploading to $S3_ROOT"
echo

upload() {
  local src="$1"
  local dst="$2"
  if [[ ! -e "$src" ]]; then
    echo "  SKIP (missing): $src"
    return
  fi
  aws s3 cp "$src" "${S3_ROOT}/${dst}" --only-show-errors
  echo "  uploaded: $dst"
}

# YOLOv8x (player + ball detector)
upload "${REPO_ROOT}/yolov8x.pt" "yolov8x.pt"

# PnLCalib finetuned keypoint head
upload "${REPO_ROOT}/data/finetuned_models/run_20260531_024941/models/best_model.pt" \
       "pnlcalib/keypoint_best_model.pt"

# PnLCalib finetuned line head
upload "${REPO_ROOT}/data/finetuned_line_models/run_20260601_142205/models/best_model.pt" \
       "pnlcalib/line_best_model.pt"

# OSNet — torchreid downloads on first import. We optionally cache the
# imagenet pretrain so the sandbox doesn't fetch from huggingface every
# time. Skipped silently if not present.
upload "${HOME}/.cache/torch/checkpoints/osnet_x1_0_imagenet.pth" \
       "osnet/osnet_x1_0_imagenet.pth"

echo
echo "Done. Entrypoint scripts will pull from $S3_ROOT/<name> at startup."
