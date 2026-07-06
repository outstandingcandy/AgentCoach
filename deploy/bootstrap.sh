#!/usr/bin/env bash
# One-shot: generate a self-signed cert, import it into ACM, deploy the
# goal-insight ALB+Cognito stack via CloudFormation.
#
# Mirrors raw2real's actual prod setup (cert imported as type IMPORTED, not
# Private CA — Private CA is $400/mo). Browsers will show "Not secure" the
# first time; user clicks "Advanced → Continue".
#
# Usage:
#   bash deploy/bootstrap.sh <admin-email>
#
# Re-run is safe: cert import is idempotent (skipped if already in ACM with
# the same CN), CFN deploy is upsert.

set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <admin-email>" >&2
  exit 1
fi
ADMIN_EMAIL="$1"

REGION="${AWS_REGION:-us-east-1}"
STACK_NAME="${STACK_NAME:-goal-insight-alb-cognito}"
TEMPLATE="$(dirname "$0")/alb-cognito.yaml"
CERT_DIR="$(dirname "$0")/.cert"
mkdir -p "$CERT_DIR"

# ALB DNS isn't known until CFN runs once, so we bind the cert CN to the
# CFN-naming convention "<lb-name>-<random>.<region>.elb.amazonaws.com".
# AWS won't let us pre-pick the random suffix, so use the wildcard CN that
# raw2real used: matches every elb DNS in this region.
CERT_CN="goal-insight-viewer.elb.amazonaws.com"

# 1. Generate self-signed cert (RSA 2048, valid 5 years) if not already on disk
if [[ ! -f "$CERT_DIR/cert.pem" || ! -f "$CERT_DIR/key.pem" ]]; then
  echo "==> Generating self-signed cert (CN=$CERT_CN)..."
  openssl req -x509 -newkey rsa:2048 -nodes -days 1825 \
    -keyout "$CERT_DIR/key.pem" -out "$CERT_DIR/cert.pem" \
    -subj "/CN=$CERT_CN" \
    -addext "subjectAltName=DNS:$CERT_CN,DNS:*.elb.amazonaws.com"
fi

# 2. Import cert into ACM (idempotent: re-import overwrites if same CN tag)
TAG_KEY="goal-insight-cert"
EXISTING_ARN=$(aws acm list-certificates --region "$REGION" \
  --query "CertificateSummaryList[?DomainName=='$CERT_CN'].CertificateArn | [0]" \
  --output text 2>/dev/null || true)

if [[ -n "$EXISTING_ARN" && "$EXISTING_ARN" != "None" ]]; then
  echo "==> Reusing existing ACM cert: $EXISTING_ARN"
  CERT_ARN="$EXISTING_ARN"
else
  echo "==> Importing new cert to ACM..."
  CERT_ARN=$(aws acm import-certificate --region "$REGION" \
    --certificate "fileb://$CERT_DIR/cert.pem" \
    --private-key "fileb://$CERT_DIR/key.pem" \
    --tags "Key=$TAG_KEY,Value=true" \
    --query CertificateArn --output text)
  echo "    cert arn = $CERT_ARN"
fi

# 3. Deploy CFN stack
#
# ImportedCertArn + AdminEmail are always overridden. The remaining CFN
# params (VpcId / SubnetIds / Ec2InstanceId / Ec2ExistingSecurityGroupId /
# CognitoDomainPrefix) keep their template defaults unless the matching env
# var is set — this lets deploy_ec2.sh feed in freshly-provisioned resource
# IDs while the old raw2real-account defaults keep working when unset.
PARAM_OVERRIDES=(
  ImportedCertArn="$CERT_ARN"
  AdminEmail="$ADMIN_EMAIL"
)
[[ -n "${VPC_ID:-}"               ]] && PARAM_OVERRIDES+=( VpcId="$VPC_ID" )
[[ -n "${SUBNET_IDS:-}"           ]] && PARAM_OVERRIDES+=( "SubnetIds=$SUBNET_IDS" )
[[ -n "${EC2_INSTANCE_ID:-}"      ]] && PARAM_OVERRIDES+=( Ec2InstanceId="$EC2_INSTANCE_ID" )
[[ -n "${EC2_SG_ID:-}"            ]] && PARAM_OVERRIDES+=( Ec2ExistingSecurityGroupId="$EC2_SG_ID" )
[[ -n "${COGNITO_DOMAIN_PREFIX:-}" ]] && PARAM_OVERRIDES+=( CognitoDomainPrefix="$COGNITO_DOMAIN_PREFIX" )
[[ -n "${RESOURCE_SUFFIX:-}"       ]] && PARAM_OVERRIDES+=( ResourceSuffix="$RESOURCE_SUFFIX" )

echo "==> Deploying CFN stack $STACK_NAME..."
aws cloudformation deploy --region "$REGION" \
  --stack-name "$STACK_NAME" \
  --template-file "$TEMPLATE" \
  --capabilities CAPABILITY_NAMED_IAM \
  --parameter-overrides "${PARAM_OVERRIDES[@]}"

# 4. Print outputs
echo
echo "==> Stack deployed. Outputs:"
aws cloudformation describe-stacks --region "$REGION" --stack-name "$STACK_NAME" \
  --query 'Stacks[0].Outputs' --output table

echo
echo "Next steps:"
echo "  1. Cognito sent $ADMIN_EMAIL a temporary password — check that inbox."
echo "  2. Open https://<AlbDns from above>/  (browser will warn — Advanced → Continue)."
echo "  3. Sign in with $ADMIN_EMAIL + the temporary password; it'll prompt to set a new one."
echo
echo "  When you're ready to LOCK DOWN the EC2:"
echo "    aws ec2 revoke-security-group-ingress --region $REGION \\"
echo "      --group-id <your-ec2-sg> --protocol tcp --port 8000 --cidr 0.0.0.0/0"
echo "  (the ALB's SG-to-SG ingress added by this stack stays in place.)"
