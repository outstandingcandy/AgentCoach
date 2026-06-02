#!/usr/bin/env bash
# One-shot AWS resource setup for running goalinsight stages as
# SageMaker Processing Jobs. Run on a workstation with admin AWS
# credentials. Idempotent: re-running is safe — it skips resources
# that already exist.
#
# Usage: bash sagemaker/setup_aws.sh
#
# Outputs the resource identifiers you need to put in the goalinsight
# config (sagemaker.role_arn, sagemaker.image_uri, sagemaker.s3_bucket).
set -euo pipefail

REGION="${AWS_REGION:-us-east-1}"
ACCOUNT_ID="$(aws sts get-caller-identity --query Account --output text)"
ROLE_NAME="${GOALINSIGHT_ROLE_NAME:-goalinsight-processing-role}"
REPO_NAME="${GOALINSIGHT_ECR_REPO:-goalinsight-pipeline}"
BUCKET_NAME="${GOALINSIGHT_S3_BUCKET:-goalinsight-pipeline-${ACCOUNT_ID}}"

echo "Region:       $REGION"
echo "Account:      $ACCOUNT_ID"
echo "Role name:    $ROLE_NAME"
echo "ECR repo:     $REPO_NAME"
echo "S3 bucket:    $BUCKET_NAME"
echo

# ---------------------------------------------------------------------
# 1. SageMaker execution role
# ---------------------------------------------------------------------
echo "[1/3] IAM role"
TRUST_POLICY=$(cat <<'JSON'
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Principal": {"Service": "sagemaker.amazonaws.com"},
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

# Attach managed policies. SageMakerFullAccess covers create_processing_job;
# AmazonS3FullAccess is broad — tighten to a bucket-scoped inline policy
# in production.
for policy in \
  arn:aws:iam::aws:policy/AmazonSageMakerFullAccess \
  arn:aws:iam::aws:policy/AmazonS3FullAccess \
  arn:aws:iam::aws:policy/AmazonEC2ContainerRegistryReadOnly; do
  aws iam attach-role-policy --role-name "$ROLE_NAME" --policy-arn "$policy" >/dev/null
done

ROLE_ARN="arn:aws:iam::${ACCOUNT_ID}:role/${ROLE_NAME}"
echo "  role arn: $ROLE_ARN"

# ---------------------------------------------------------------------
# 2. ECR repository
# ---------------------------------------------------------------------
echo
echo "[2/3] ECR repository"
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

# ---------------------------------------------------------------------
# 3. S3 bucket
# ---------------------------------------------------------------------
echo
echo "[3/3] S3 bucket"
if aws s3api head-bucket --bucket "$BUCKET_NAME" 2>/dev/null; then
  echo "  bucket $BUCKET_NAME already exists, skipping"
else
  if [[ "$REGION" == "us-east-1" ]]; then
    aws s3api create-bucket --bucket "$BUCKET_NAME" --region "$REGION" >/dev/null
  else
    aws s3api create-bucket --bucket "$BUCKET_NAME" --region "$REGION" \
      --create-bucket-configuration "LocationConstraint=$REGION" >/dev/null
  fi
  aws s3api put-bucket-versioning \
    --bucket "$BUCKET_NAME" \
    --versioning-configuration Status=Enabled >/dev/null
  echo "  created bucket $BUCKET_NAME (versioning on)"
fi

# ---------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------
echo
echo "================================================================"
echo "Add to your goalinsight config (e.g. configs/default.yaml):"
echo
cat <<YAML
sagemaker:
  region: $REGION
  role_arn: $ROLE_ARN
  image_uri: $ECR_URI:latest
  s3_bucket: $BUCKET_NAME
  s3_prefix: pipeline-runs
  weights_s3_prefix: weights
  default_instance_type: ml.g5.xlarge
  max_runtime_seconds: 3600
YAML
echo "================================================================"
