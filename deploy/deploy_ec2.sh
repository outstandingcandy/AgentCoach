#!/usr/bin/env bash
#
# One-click deploy of the GoalInsight web app to a fresh GPU EC2 instance,
# fronted by an internet-facing ALB + Cognito login.
#
# EVERYTHING is provisioned by a single CloudFormation stack
# (deploy/full-stack.yaml): the EC2 instance, its IAM role + instance
# profile, its security group, the ALB, and the Cognito user pool. This
# script only does what CFN can't:
#   1. Resolve region + account.
#   2. Discover the default VPC and >=2 public subnets (unless overridden).
#   3. Generate + import a self-signed cert into ACM (ImportCertificate is
#      not a CFN resource type).
#   4. Deploy / update the stack, passing those values in.
#
# Delete everything later with: bash deploy/teardown.sh [--suffix <s>]
#
# Usage:
#   bash deploy/deploy_ec2.sh <admin-email> [options]
#
# Options (also settable via env var):
#   --region <r>          AWS region                (env AWS_REGION,    default us-east-1)
#   --instance-type <t>   GPU instance type         (env INSTANCE_TYPE, default g5.xlarge)
#   --key-name <k>        EC2 SSH key pair name     (env KEY_NAME,      default none/SSM-only)
#   --branch <b>          Repo branch to deploy     (env REPO_BRANCH,   default master)
#   --vpc-id <v>          Override VPC discovery     (env VPC_ID)
#   --subnet-ids <a,b>    Override subnet discovery  (env SUBNET_IDS, comma-separated, >=2 AZs)
#   --volume-size <gb>    Root EBS size in GiB      (env VOLUME_SIZE,   default 200)
#   --suffix <s>          Parallel-deploy suffix     (env SUFFIX) — distinct stack name + Cognito domain
#
# Cost note: g5.xlarge (A10G 24GB) is ~$1/hr on-demand — NOT free tier.
# First boot takes ~15-25 min (install Docker + NVIDIA toolkit, then
# docker build the image); the ALB target stays unhealthy until the
# container is up. That's expected.
set -euo pipefail

# ---- args -----------------------------------------------------------------
ADMIN_EMAIL=""
REGION="${AWS_REGION:-us-east-1}"
INSTANCE_TYPE="${INSTANCE_TYPE:-g5.xlarge}"
KEY_NAME="${KEY_NAME:-}"
REPO_BRANCH="${REPO_BRANCH:-master}"
REPO_URL="${REPO_URL:-https://github.com/outstandingcandy/AgentCoach.git}"
VPC_ID="${VPC_ID:-}"
SUBNET_IDS="${SUBNET_IDS:-}"
VOLUME_SIZE="${VOLUME_SIZE:-200}"
# Optional suffix for a parallel deployment alongside an existing stack.
SUFFIX="${SUFFIX:-}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TEMPLATE="$SCRIPT_DIR/full-stack.yaml"
CERT_DIR="$SCRIPT_DIR/.cert"
CERT_CN="goal-insight-viewer.elb.amazonaws.com"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --region)        REGION="$2"; shift 2 ;;
    --instance-type) INSTANCE_TYPE="$2"; shift 2 ;;
    --key-name)      KEY_NAME="$2"; shift 2 ;;
    --branch)        REPO_BRANCH="$2"; shift 2 ;;
    --vpc-id)        VPC_ID="$2"; shift 2 ;;
    --subnet-ids)    SUBNET_IDS="$2"; shift 2 ;;
    --volume-size)   VOLUME_SIZE="$2"; shift 2 ;;
    --suffix)        SUFFIX="$2"; shift 2 ;;
    -h|--help)       sed -n '2,38p' "$0" | sed 's/^# \?//'; exit 0 ;;
    -*)              echo "Unknown flag: $1" >&2; exit 2 ;;
    *)               ADMIN_EMAIL="$1"; shift ;;
  esac
done

if [[ -z "$ADMIN_EMAIL" ]]; then
  echo "Usage: $0 <admin-email> [--region ...] [--instance-type ...] [--suffix ...]" >&2
  exit 1
fi

echo "==> Region: $REGION   Instance type: $INSTANCE_TYPE   Branch: $REPO_BRANCH"
ACCOUNT_ID="$(aws sts get-caller-identity --query Account --output text)"
echo "==> Account: $ACCOUNT_ID"

STACK_NAME="goal-insight-full${SUFFIX}"
COGNITO_DOMAIN_PREFIX="goal-insight-${ACCOUNT_ID}${SUFFIX}"
[[ -n "$SUFFIX" ]] && echo "==> Parallel deploy suffix: '$SUFFIX' (stack $STACK_NAME)"

# ---- 1. Network discovery -------------------------------------------------
if [[ -z "$VPC_ID" ]]; then
  VPC_ID="$(aws ec2 describe-vpcs --region "$REGION" \
    --filters Name=isDefault,Values=true \
    --query 'Vpcs[0].VpcId' --output text)"
  if [[ -z "$VPC_ID" || "$VPC_ID" == "None" ]]; then
    echo "ERROR: no default VPC in $REGION. Pass --vpc-id and --subnet-ids." >&2
    exit 1
  fi
  echo "==> Discovered default VPC: $VPC_ID"
fi

if [[ -z "$SUBNET_IDS" ]]; then
  # Public subnets (auto-assign public IP), one per AZ, need >=2 distinct AZs.
  mapfile -t SUBNET_ROWS < <(aws ec2 describe-subnets --region "$REGION" \
    --filters "Name=vpc-id,Values=$VPC_ID" "Name=map-public-ip-on-launch,Values=true" \
    --query 'Subnets[].[SubnetId,AvailabilityZone]' --output text)
  declare -A AZ_SEEN=()
  PICKED=()
  for row in "${SUBNET_ROWS[@]}"; do
    sid="${row%%$'\t'*}"; az="${row##*$'\t'}"
    if [[ -z "${AZ_SEEN[$az]:-}" ]]; then
      AZ_SEEN[$az]=1; PICKED+=("$sid")
    fi
  done
  if [[ "${#PICKED[@]}" -lt 2 ]]; then
    echo "ERROR: need >=2 public subnets in different AZs in $VPC_ID; found ${#PICKED[@]}." >&2
    echo "       Pass --subnet-ids sub-a,sub-b explicitly." >&2
    exit 1
  fi
  SUBNET_IDS="$(IFS=,; echo "${PICKED[*]}")"
  echo "==> Discovered public subnets: $SUBNET_IDS"
fi
LAUNCH_SUBNET="${SUBNET_IDS%%,*}"

# ---- 2. Self-signed cert -> ACM (the one thing CFN can't do) --------------
mkdir -p "$CERT_DIR"
if [[ ! -f "$CERT_DIR/cert.pem" || ! -f "$CERT_DIR/key.pem" ]]; then
  echo "==> Generating self-signed cert (CN=$CERT_CN)..."
  openssl req -x509 -newkey rsa:2048 -nodes -days 1825 \
    -keyout "$CERT_DIR/key.pem" -out "$CERT_DIR/cert.pem" \
    -subj "/CN=$CERT_CN" \
    -addext "subjectAltName=DNS:$CERT_CN,DNS:*.elb.amazonaws.com"
fi
CERT_ARN="$(aws acm list-certificates --region "$REGION" \
  --query "CertificateSummaryList[?DomainName=='$CERT_CN'].CertificateArn | [0]" \
  --output text 2>/dev/null || true)"
if [[ -n "$CERT_ARN" && "$CERT_ARN" != "None" ]]; then
  echo "==> Reusing existing ACM cert: $CERT_ARN"
else
  echo "==> Importing cert to ACM..."
  CERT_ARN="$(aws acm import-certificate --region "$REGION" \
    --certificate "fileb://$CERT_DIR/cert.pem" \
    --private-key "fileb://$CERT_DIR/key.pem" \
    --tags "Key=goal-insight-cert,Value=true" \
    --query CertificateArn --output text)"
  echo "    cert arn = $CERT_ARN"
fi

# ---- 3. Resolve caller IP for SSH (optional) ------------------------------
SSH_CIDR=""
MY_IP="$(curl -s https://checkip.amazonaws.com || true)"
[[ -n "$MY_IP" ]] && SSH_CIDR="${MY_IP}/32"

# ---- 4. Deploy the full stack ---------------------------------------------
echo "==> Deploying CloudFormation stack $STACK_NAME (everything in one place)..."
aws cloudformation deploy --region "$REGION" \
  --stack-name "$STACK_NAME" \
  --template-file "$TEMPLATE" \
  --capabilities CAPABILITY_NAMED_IAM CAPABILITY_IAM \
  --parameter-overrides \
    VpcId="$VPC_ID" \
    "SubnetIds=$SUBNET_IDS" \
    LaunchSubnetId="$LAUNCH_SUBNET" \
    InstanceType="$INSTANCE_TYPE" \
    VolumeSizeGb="$VOLUME_SIZE" \
    KeyName="$KEY_NAME" \
    SshCidr="$SSH_CIDR" \
    RepoUrl="$REPO_URL" \
    RepoBranch="$REPO_BRANCH" \
    ImportedCertArn="$CERT_ARN" \
    CognitoDomainPrefix="$COGNITO_DOMAIN_PREFIX" \
    AdminEmail="$ADMIN_EMAIL"

echo
echo "==> Stack deployed. Outputs:"
aws cloudformation describe-stacks --region "$REGION" --stack-name "$STACK_NAME" \
  --query 'Stacks[0].Outputs' --output table

ALB_DNS="$(aws cloudformation describe-stacks --region "$REGION" --stack-name "$STACK_NAME" \
  --query "Stacks[0].Outputs[?OutputKey=='AlbDns'].OutputValue | [0]" --output text)"
INSTANCE_ID="$(aws cloudformation describe-stacks --region "$REGION" --stack-name "$STACK_NAME" \
  --query "Stacks[0].Outputs[?OutputKey=='InstanceId'].OutputValue | [0]" --output text)"

echo
echo "============================================================"
echo "Deploy complete — all resources are in stack $STACK_NAME."
echo "  Instance:  $INSTANCE_ID  ($INSTANCE_TYPE, $REGION)"
echo "  URL:       https://$ALB_DNS/"
echo
echo "The instance is still provisioning (install Docker + NVIDIA toolkit,"
echo "then docker build the image, ~15-25 min). The ALB target will be"
echo "UNHEALTHY until the container is up."
echo
echo "Watch progress via SSM:"
echo "  aws ssm start-session --region $REGION --target $INSTANCE_ID"
echo "  sudo tail -f /var/log/goal-insight-provision.log"
echo
echo "Sign in with $ADMIN_EMAIL + the Cognito temp password emailed to you."
echo "Tear down everything with: bash deploy/teardown.sh${SUFFIX:+ --suffix $SUFFIX}"
echo "============================================================"
