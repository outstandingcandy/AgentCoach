#!/usr/bin/env bash
#
# One-click deploy of the GoalInsight web app to a fresh GPU EC2 instance,
# fronted by an internet-facing ALB + Cognito login.
#
# What it does (all steps idempotent — safe to re-run):
#   1. Resolve region + account.
#   2. Discover the default VPC and >=2 public subnets (unless overridden).
#   3. Create-or-reuse an IAM role + instance profile granting Bedrock
#      (chat) + SSM (keyless shell) access.
#   4. Create-or-reuse the instance security group. IMPORTANT: the only
#      externally-open port is SSH 22 (locked to the caller's IP). App port
#      8000 is opened ONLY to the ALB's SG, and that rule is added by the
#      CloudFormation stack (Ec2AppPortFromAlb) — never to 0.0.0.0/0.
#   5. Resolve the latest Deep Learning GPU AMI (driver + CUDA preinstalled).
#   6. Launch the GPU instance with deploy/ec2_userdata.sh (clones repo,
#      installs the app, starts it under systemd).
#   7. Wait for the instance, then hand off to deploy/bootstrap.sh which
#      imports a self-signed cert and deploys the ALB + Cognito stack against
#      the freshly-provisioned instance/SG.
#
# Usage:
#   bash deploy/deploy_ec2.sh <admin-email> [options]
#
# Options (also settable via env var):
#   --region <r>          AWS region                (env AWS_REGION,   default us-east-1)
#   --instance-type <t>   GPU instance type         (env INSTANCE_TYPE, default g5.xlarge)
#   --key-name <k>        EC2 SSH key pair name     (env KEY_NAME,     default none/SSM-only)
#   --branch <b>          Repo branch to deploy     (env REPO_BRANCH,  default master)
#   --vpc-id <v>          Override VPC discovery     (env VPC_ID)
#   --subnet-ids <a,b>    Override subnet discovery  (env SUBNET_IDS, comma-separated, >=2 AZs)
#   --volume-size <gb>    Root EBS size in GiB      (env VOLUME_SIZE,  default 200)
#
# Cost note: g5.xlarge (A10G 24GB) is ~$1/hr on-demand — NOT free tier.
# First boot takes ~15-25 min (install Docker + NVIDIA toolkit, then
# docker build the image = torch + ML stack); the ALB target stays unhealthy
# until the container is up. That's expected.
set -euo pipefail

# ---- args -----------------------------------------------------------------
ADMIN_EMAIL=""
REGION="${AWS_REGION:-us-east-1}"
INSTANCE_TYPE="${INSTANCE_TYPE:-g5.xlarge}"
KEY_NAME="${KEY_NAME:-}"
REPO_BRANCH="${REPO_BRANCH:-master}"
VPC_ID="${VPC_ID:-}"
SUBNET_IDS="${SUBNET_IDS:-}"
VOLUME_SIZE="${VOLUME_SIZE:-200}"
# Optional suffix for a parallel deployment in a region that already has a
# goal-insight stack. It's appended to the CFN stack name, the fixed CFN
# resource names (ResourceSuffix), the Cognito domain, the instance Name tag,
# and the instance SG name — so nothing collides with an existing stack.
SUFFIX="${SUFFIX:-}"

REPO_URL="https://github.com/outstandingcandy/AgentCoach.git"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROLE_NAME="goal-insight-ec2-role"
PROFILE_NAME="goal-insight-ec2-profile"

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
    -h|--help)       sed -n '2,45p' "$0" | sed 's/^# \?//'; exit 0 ;;
    -*)              echo "Unknown flag: $1" >&2; exit 2 ;;
    *)               ADMIN_EMAIL="$1"; shift ;;
  esac
done

if [[ -z "$ADMIN_EMAIL" ]]; then
  echo "Usage: $0 <admin-email> [--region ...] [--instance-type ...] [--key-name ...]" >&2
  exit 1
fi

echo "==> Region: $REGION   Instance type: $INSTANCE_TYPE   Branch: $REPO_BRANCH"

ACCOUNT_ID="$(aws sts get-caller-identity --query Account --output text)"
echo "==> Account: $ACCOUNT_ID"

# Derive suffixed names so a parallel deploy never collides with an existing
# goal-insight stack. With no --suffix these reduce to the original names.
STACK_NAME="goal-insight-alb-cognito${SUFFIX}"
SG_NAME="goal-insight-ec2-sg${SUFFIX}"
INSTANCE_NAME="goal-insight-web${SUFFIX}"
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
  # ALB wants all AZs it can get; the instance launches into the first.
  SUBNET_IDS="$(IFS=,; echo "${PICKED[*]}")"
  echo "==> Discovered public subnets: $SUBNET_IDS"
fi
LAUNCH_SUBNET="${SUBNET_IDS%%,*}"

# ---- 2. IAM role + instance profile ---------------------------------------
echo "==> Ensuring IAM role $ROLE_NAME..."
if ! aws iam get-role --role-name "$ROLE_NAME" >/dev/null 2>&1; then
  aws iam create-role --role-name "$ROLE_NAME" \
    --assume-role-policy-document '{
      "Version":"2012-10-17",
      "Statement":[{"Effect":"Allow","Principal":{"Service":"ec2.amazonaws.com"},"Action":"sts:AssumeRole"}]
    }' >/dev/null
  echo "    created role"
else
  echo "    role exists (reuse)"
fi

# Managed policy for keyless SSM shell access.
aws iam attach-role-policy --role-name "$ROLE_NAME" \
  --policy-arn arn:aws:iam::aws:policy/AmazonSSMManagedInstanceCore >/dev/null 2>&1 || true

# Inline policy: Bedrock (chat) + AgentCore (run_python sandbox).
aws iam put-role-policy --role-name "$ROLE_NAME" \
  --policy-name goal-insight-bedrock \
  --policy-document '{
    "Version":"2012-10-17",
    "Statement":[
      {"Effect":"Allow",
       "Action":["bedrock:InvokeModel","bedrock:InvokeModelWithResponseStream"],
       "Resource":"*"},
      {"Effect":"Allow",
       "Action":["bedrock-agentcore:InvokeAgentRuntime","bedrock-agentcore:InvokeCodeInterpreter",
                 "bedrock-agentcore:StartCodeInterpreterSession","bedrock-agentcore:StopCodeInterpreterSession",
                 "bedrock-agentcore:GetCodeInterpreterSession","bedrock-agentcore:ListCodeInterpreterSessions"],
       "Resource":"*"}
    ]
  }' >/dev/null
echo "    attached SSM + Bedrock policies"

echo "==> Ensuring instance profile $PROFILE_NAME..."
if ! aws iam get-instance-profile --instance-profile-name "$PROFILE_NAME" >/dev/null 2>&1; then
  aws iam create-instance-profile --instance-profile-name "$PROFILE_NAME" >/dev/null
  aws iam add-role-to-instance-profile --instance-profile-name "$PROFILE_NAME" \
    --role-name "$ROLE_NAME" >/dev/null
  echo "    created profile + added role; waiting for propagation..."
  sleep 15
else
  echo "    profile exists (reuse)"
fi

# ---- 3. Instance security group (SSH 22 only, from caller IP) -------------
echo "==> Ensuring instance security group $SG_NAME..."
SG_ID="$(aws ec2 describe-security-groups --region "$REGION" \
  --filters "Name=group-name,Values=$SG_NAME" "Name=vpc-id,Values=$VPC_ID" \
  --query 'SecurityGroups[0].GroupId' --output text 2>/dev/null || true)"

if [[ -z "$SG_ID" || "$SG_ID" == "None" ]]; then
  SG_ID="$(aws ec2 create-security-group --region "$REGION" \
    --group-name "$SG_NAME" --vpc-id "$VPC_ID" \
    --description "goal-insight web instance: SSH only; :8000 opened to ALB SG by CFN" \
    --query 'GroupId' --output text)"
  echo "    created SG $SG_ID"
else
  echo "    SG exists (reuse): $SG_ID"
fi

# Lock SSH to the caller's current public IP only.
MY_IP="$(curl -s https://checkip.amazonaws.com || true)"
if [[ -n "$MY_IP" ]]; then
  aws ec2 authorize-security-group-ingress --region "$REGION" \
    --group-id "$SG_ID" --protocol tcp --port 22 --cidr "${MY_IP}/32" \
    >/dev/null 2>&1 && echo "    SSH 22 opened to ${MY_IP}/32" \
    || echo "    SSH 22 rule already present"
else
  echo "    WARN: could not determine caller IP; no SSH rule added (use SSM to connect)."
fi
# NOTE: we deliberately do NOT open :8000 here. The CFN stack adds an
# SG-to-SG ingress from the ALB only. No app port is ever public.

# ---- 4. Resolve Deep Learning GPU AMI -------------------------------------
echo "==> Resolving Deep Learning GPU AMI (Ubuntu 22.04)..."
AMI_ID="$(aws ssm get-parameters --region "$REGION" \
  --names /aws/service/deeplearning/ami/x86_64/base-oss-nvidia-driver-gpu-ubuntu-22.04/latest/ami-id \
  --query 'Parameters[0].Value' --output text 2>/dev/null || true)"

if [[ -z "$AMI_ID" || "$AMI_ID" == "None" ]]; then
  echo "    DLAMI lookup failed; falling back to stock Ubuntu 22.04 (userdata installs the driver)."
  AMI_ID="$(aws ssm get-parameters --region "$REGION" \
    --names /aws/service/canonical/ubuntu/server/22.04/stable/current/amd64/hvm/ebs-gp3/ami-id \
    --query 'Parameters[0].Value' --output text)"
fi
echo "    AMI: $AMI_ID"

# ---- 5. Render user-data --------------------------------------------------
USERDATA_TMP="$(mktemp)"
trap 'rm -f "$USERDATA_TMP"' EXIT
sed -e "s|__REPO_URL__|$REPO_URL|g" \
    -e "s|__REPO_BRANCH__|$REPO_BRANCH|g" \
    "$SCRIPT_DIR/ec2_userdata.sh" > "$USERDATA_TMP"

# ---- 6. Launch (or reuse) the instance ------------------------------------
# Reuse an existing running/pending instance tagged Name=$INSTANCE_NAME.
INSTANCE_ID="$(aws ec2 describe-instances --region "$REGION" \
  --filters "Name=tag:Name,Values=$INSTANCE_NAME" \
            "Name=instance-state-name,Values=pending,running,stopping,stopped" \
  --query 'Reservations[0].Instances[0].InstanceId' --output text 2>/dev/null || true)"

if [[ -n "$INSTANCE_ID" && "$INSTANCE_ID" != "None" ]]; then
  echo "==> Reusing existing instance $INSTANCE_ID (skipping run-instances)."
else
  echo "==> Launching $INSTANCE_TYPE instance..."
  KEY_ARGS=()
  [[ -n "$KEY_NAME" ]] && KEY_ARGS=(--key-name "$KEY_NAME")

  INSTANCE_ID="$(aws ec2 run-instances --region "$REGION" \
    --image-id "$AMI_ID" \
    --instance-type "$INSTANCE_TYPE" \
    --iam-instance-profile "Name=$PROFILE_NAME" \
    --security-group-ids "$SG_ID" \
    --subnet-id "$LAUNCH_SUBNET" \
    --associate-public-ip-address \
    --block-device-mappings "DeviceName=/dev/sda1,Ebs={VolumeSize=$VOLUME_SIZE,VolumeType=gp3}" \
    --user-data "file://$USERDATA_TMP" \
    "${KEY_ARGS[@]}" \
    --tag-specifications "ResourceType=instance,Tags=[{Key=Name,Value=$INSTANCE_NAME}]" \
    --query 'Instances[0].InstanceId' --output text)"
  echo "    launched $INSTANCE_ID"
fi

echo "==> Waiting for instance to reach status OK (this can take a few minutes)..."
aws ec2 wait instance-status-ok --region "$REGION" --instance-ids "$INSTANCE_ID"
echo "    instance $INSTANCE_ID is up"

# ---- 7. Hand off to bootstrap.sh (cert + ALB + Cognito CFN) ---------------
echo "==> Deploying ALB + Cognito stack ($STACK_NAME) via bootstrap.sh..."
export AWS_REGION="$REGION"
export STACK_NAME
export VPC_ID SUBNET_IDS
export EC2_INSTANCE_ID="$INSTANCE_ID"
export EC2_SG_ID="$SG_ID"
export COGNITO_DOMAIN_PREFIX
export RESOURCE_SUFFIX="$SUFFIX"
bash "$SCRIPT_DIR/bootstrap.sh" "$ADMIN_EMAIL"

echo
echo "============================================================"
echo "Deploy complete."
echo "  Instance:  $INSTANCE_ID  ($INSTANCE_TYPE, $REGION)"
echo "  App SG:    $SG_ID  (SSH 22 only; :8000 reachable from the ALB SG only)"
echo
echo "The instance is still provisioning (install Docker + NVIDIA toolkit,"
echo "then docker build the image, ~15-25 min). The ALB target will be"
echo "UNHEALTHY until 'systemctl status goal-insight-web' (the container)"
echo "is active."
echo
echo "Watch progress via SSM:"
echo "  aws ssm start-session --region $REGION --target $INSTANCE_ID"
echo "  sudo tail -f /var/log/goal-insight-provision.log"
echo
echo "Then open the AlbDns URL printed above (accept the self-signed cert),"
echo "and sign in with $ADMIN_EMAIL + the Cognito temp password emailed to you."
echo "============================================================"
