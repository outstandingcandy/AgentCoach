#!/usr/bin/env bash
#
# Tear down a goal-insight deployment.
#
# Because deploy_ec2.sh now provisions EVERYTHING inside one CloudFormation
# stack (deploy/full-stack.yaml) — EC2 instance, IAM role + profile, security
# groups, ALB, Cognito — deleting the stack removes all of it. The only
# out-of-band resource is the self-signed ACM cert (ImportCertificate isn't a
# CFN resource); it's shared/reusable across deploys, so we leave it unless
# --purge-cert is passed.
#
# Usage:
#   bash deploy/teardown.sh [--suffix <s>] [--region <r>] [--purge-cert] [--yes]
#
#   --suffix <s>    Target the parallel stack goal-insight-full<s> (must match
#                   the --suffix used at deploy time). Default: no suffix.
#   --region <r>    AWS region (env AWS_REGION, default us-east-1).
#   --purge-cert    Also delete the imported self-signed ACM cert.
#   --yes           Skip the confirmation prompt.
set -euo pipefail

REGION="${AWS_REGION:-us-east-1}"
SUFFIX="${SUFFIX:-}"
PURGE_CERT=0
ASSUME_YES=0
CERT_CN="goal-insight-viewer.elb.amazonaws.com"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --suffix)     SUFFIX="$2"; shift 2 ;;
    --region)     REGION="$2"; shift 2 ;;
    --purge-cert) PURGE_CERT=1; shift ;;
    --yes|-y)     ASSUME_YES=1; shift ;;
    -h|--help)    sed -n '2,22p' "$0" | sed 's/^# \?//'; exit 0 ;;
    *)            echo "Unknown flag: $1" >&2; exit 2 ;;
  esac
done

STACK_NAME="goal-insight-full${SUFFIX}"

# Confirm the stack exists before doing anything destructive.
if ! aws cloudformation describe-stacks --region "$REGION" \
      --stack-name "$STACK_NAME" >/dev/null 2>&1; then
  echo "Stack $STACK_NAME not found in $REGION. Nothing to delete." >&2
  echo "(If you deployed with --suffix, pass the same --suffix here.)" >&2
  exit 1
fi

echo "About to DELETE CloudFormation stack: $STACK_NAME (region $REGION)"
echo "This removes the EC2 instance, IAM role/profile, security groups,"
echo "ALB, and Cognito user pool created by that stack."
[[ "$PURGE_CERT" == 1 ]] && echo "Also purging the imported ACM cert ($CERT_CN)."
if [[ "$ASSUME_YES" != 1 ]]; then
  read -r -p "Type the stack name to confirm: " CONFIRM
  if [[ "$CONFIRM" != "$STACK_NAME" ]]; then
    echo "Confirmation mismatch; aborting." >&2
    exit 1
  fi
fi

echo "==> Deleting stack $STACK_NAME..."
aws cloudformation delete-stack --region "$REGION" --stack-name "$STACK_NAME"
echo "==> Waiting for stack deletion to complete..."
aws cloudformation wait stack-delete-complete --region "$REGION" --stack-name "$STACK_NAME"
echo "    stack deleted."

if [[ "$PURGE_CERT" == 1 ]]; then
  CERT_ARN="$(aws acm list-certificates --region "$REGION" \
    --query "CertificateSummaryList[?DomainName=='$CERT_CN'].CertificateArn | [0]" \
    --output text 2>/dev/null || true)"
  if [[ -n "$CERT_ARN" && "$CERT_ARN" != "None" ]]; then
    echo "==> Deleting ACM cert $CERT_ARN..."
    # ACM refuses to delete a cert still in use by a listener; the stack is
    # already gone above, so this should succeed.
    aws acm delete-certificate --region "$REGION" --certificate-arn "$CERT_ARN" \
      && echo "    cert deleted." \
      || echo "    WARN: cert still in use by another stack/listener; left in place."
  else
    echo "==> No imported cert found for $CERT_CN; nothing to purge."
  fi
fi

echo "==> Teardown complete."
