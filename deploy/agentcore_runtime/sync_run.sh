#!/usr/bin/env bash
# Sync the JSON output of one pipeline run to S3 so the chat runtime
# can read it. Only the files the chat path needs — no video, no
# weights, no per-frame vis JPEGs.
#
# Usage:
#   bash deploy/agentcore_runtime/sync_run.sh <run_dir>
#   e.g.: bash deploy/agentcore_runtime/sync_run.sh workspace/runs/clip_000_finetuned
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <run_dir>" >&2
  exit 1
fi

RUN_DIR="$1"
RUN_NAME="$(basename "$RUN_DIR")"

REGION="${AWS_REGION:-us-east-1}"
ACCOUNT_ID="$(aws sts get-caller-identity --query Account --output text)"
BUCKET_NAME="${GOALINSIGHT_S3_BUCKET:-goalinsight-pipeline-${ACCOUNT_ID}}"
PREFIX_FMT="${GOALINSIGHT_S3_RUN_PREFIX_FMT:-runs/{run_name}}"
PREFIX="${PREFIX_FMT//\{run_name\}/$RUN_NAME}"

echo "Syncing $RUN_DIR -> s3://$BUCKET_NAME/$PREFIX/"

# Use sync with --exclude/--include so we ship only the JSON the chat
# path reads. aws s3 sync evaluates patterns left-to-right, so we
# exclude everything first and re-include the targets.
aws s3 sync "$RUN_DIR/" "s3://$BUCKET_NAME/$PREFIX/" \
  --region "$REGION" \
  --exclude "*" \
  --include "field_registration/calibration_metadata.json" \
  --include "tracking/tracks.json" \
  --include "tracking/ball_tracks.json" \
  --include "tracking/team_assignments.json" \
  --include "track_consolidation/tracks.json" \
  --include "track_consolidation/team_assignments.json" \
  --include "track_consolidation/players.json" \
  --include "event_detection/events.json"

echo "Done. Test with:"
echo "  aws s3 ls s3://$BUCKET_NAME/$PREFIX/ --recursive"
