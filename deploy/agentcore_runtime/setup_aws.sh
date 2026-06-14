#!/usr/bin/env bash
# One-shot AWS resource setup for the GoalInsight chat AgentCore Runtime.
#
# Creates (idempotent):
#   - ECR repository for the runtime image
#   - IAM execution role used by the runtime container
#
# Usage: bash deploy/agentcore_runtime/setup_aws.sh
#
# Requires the GOALINSIGHT_S3_BUCKET env var (or shared with the
# sagemaker setup) so the runtime role gets read access to the bucket
# holding pipeline-run JSON.
set -euo pipefail

REGION="${AWS_REGION:-us-east-1}"
ACCOUNT_ID="$(aws sts get-caller-identity --query Account --output text)"
ROLE_NAME="${GOALINSIGHT_RUNTIME_ROLE_NAME:-goalinsight-chat-runtime-role}"
REPO_NAME="${GOALINSIGHT_RUNTIME_ECR_REPO:-goalinsight-chat-runtime}"
BUCKET_NAME="${GOALINSIGHT_S3_BUCKET:-goalinsight-pipeline-${ACCOUNT_ID}}"

echo "Region:    $REGION"
echo "Account:   $ACCOUNT_ID"
echo "Role:      $ROLE_NAME"
echo "ECR repo:  $REPO_NAME"
echo "S3 bucket: $BUCKET_NAME"
echo

# ---------------------------------------------------------------------
# 1. IAM execution role
# ---------------------------------------------------------------------
echo "[1/2] IAM role"
TRUST_POLICY=$(cat <<'JSON'
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Principal": {"Service": "bedrock-agentcore.amazonaws.com"},
      "Action": "sts:AssumeRole"
    }
  ]
}
JSON
)

if aws iam get-role --role-name "$ROLE_NAME" >/dev/null 2>&1; then
  echo "  role $ROLE_NAME already exists, skipping create"
else
  aws iam create-role \
    --role-name "$ROLE_NAME" \
    --assume-role-policy-document "$TRUST_POLICY" >/dev/null
  echo "  created role $ROLE_NAME"
fi

INLINE_POLICY=$(cat <<JSON
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Sid": "InvokeBedrockClaude",
      "Effect": "Allow",
      "Action": ["bedrock:InvokeModel", "bedrock:InvokeModelWithResponseStream"],
      "Resource": "*"
    },
    {
      "Sid": "AgentCoreCodeInterpreter",
      "Effect": "Allow",
      "Action": [
        "bedrock-agentcore:StartCodeInterpreterSession",
        "bedrock-agentcore:InvokeCodeInterpreter",
        "bedrock-agentcore:StopCodeInterpreterSession"
      ],
      "Resource": "*"
    },
    {
      "Sid": "ReadRunOutputs",
      "Effect": "Allow",
      "Action": ["s3:GetObject", "s3:ListBucket"],
      "Resource": [
        "arn:aws:s3:::${BUCKET_NAME}",
        "arn:aws:s3:::${BUCKET_NAME}/*"
      ]
    },
    {
      "Sid": "WriteChatArtifacts",
      "Effect": "Allow",
      "Action": ["s3:PutObject"],
      "Resource": ["arn:aws:s3:::${BUCKET_NAME}/chat_artifacts/*"]
    },
    {
      "Sid": "ECRImagePull",
      "Effect": "Allow",
      "Action": [
        "ecr:GetAuthorizationToken",
        "ecr:BatchCheckLayerAvailability",
        "ecr:GetDownloadUrlForLayer",
        "ecr:BatchGetImage"
      ],
      "Resource": "*"
    },
    {
      "Sid": "CloudWatchLogs",
      "Effect": "Allow",
      "Action": [
        "logs:CreateLogGroup",
        "logs:CreateLogStream",
        "logs:PutLogEvents"
      ],
      "Resource": "*"
    }
  ]
}
JSON
)

aws iam put-role-policy \
  --role-name "$ROLE_NAME" \
  --policy-name "${ROLE_NAME}-inline" \
  --policy-document "$INLINE_POLICY" >/dev/null

ROLE_ARN="arn:aws:iam::${ACCOUNT_ID}:role/${ROLE_NAME}"
echo "  role arn: $ROLE_ARN"

# ---------------------------------------------------------------------
# 2. ECR repository
# ---------------------------------------------------------------------
echo
echo "[2/2] ECR repository"
if aws ecr describe-repositories --repository-names "$REPO_NAME" --region "$REGION" >/dev/null 2>&1; then
  echo "  repo $REPO_NAME already exists, skipping"
else
  aws ecr create-repository \
    --repository-name "$REPO_NAME" \
    --region "$REGION" \
    --image-scanning-configuration scanOnPush=true >/dev/null
  echo "  created repo $REPO_NAME"
fi
ECR_URI="${ACCOUNT_ID}.dkr.ecr.${REGION}.amazonaws.com/${REPO_NAME}"
echo "  ecr uri: $ECR_URI"

echo
echo "================================================================"
echo "Next steps:"
echo "  1) bash deploy/agentcore_runtime/build_and_push.sh"
echo "  2) bash deploy/agentcore_runtime/create_runtime.sh"
echo
echo "Then export to enable runtime-backed chat:"
echo "  export GOALINSIGHT_S3_BUCKET=$BUCKET_NAME"
echo "  export GOALINSIGHT_AGENTCORE_RUNTIME_ARN=<from create_runtime.sh>"
echo "================================================================"
