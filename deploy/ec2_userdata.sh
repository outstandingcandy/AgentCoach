#!/usr/bin/env bash
#
# EC2 provisioning for the GoalInsight web app (CONTAINERIZED).
#
# Invoked by the CloudFormation stack's UserData AFTER it has cloned the repo
# to /home/ubuntu/AgentCoach (see deploy/full-stack.yaml). This script installs
# Docker + the NVIDIA container toolkit, builds the deployment image on the
# instance, downloads model weights into the host workspace volume, and starts
# the container under systemd.
#
# Model weights are NOT baked into the image — they download into the
# bind-mounted /home/ubuntu/AgentCoach/workspace at first use (PnLCalib
# releases, ultralytics YOLO, torchreid OSNet) plus the ev_posw fine-tune
# fetched below.
#
# Assumes a Deep Learning GPU AMI (NVIDIA driver preinstalled). If the launcher
# fell back to a stock Ubuntu AMI, the driver is installed below.
#
# All output is tee'd to /var/log/goal-insight-provision.log (the UserData
# wrapper already redirects there; this is a no-op safety net when run by hand).
set -euo pipefail
echo "==> goal-insight container provisioning started at $(date -u)"

APP_DIR="/home/ubuntu/AgentCoach"
IMAGE_TAG="goalinsight:deploy"

export DEBIAN_FRONTEND=noninteractive

# ---- 1. Base packages -----------------------------------------------------
echo "==> Installing base packages..."
apt-get update -y
apt-get install -y ca-certificates curl gnupg

# ---- 1b. GPU driver fallback (only if no NVIDIA driver present) -----------
# The Deep Learning AMI already ships the driver; this is a safety net for a
# stock-Ubuntu fallback launch. nvidia-smi succeeding means we skip it.
if ! command -v nvidia-smi >/dev/null 2>&1 || ! nvidia-smi >/dev/null 2>&1; then
  echo "==> No working NVIDIA driver detected; installing ubuntu-drivers..."
  apt-get install -y ubuntu-drivers-common || true
  ubuntu-drivers autoinstall || echo "WARN: driver autoinstall failed; GPU pipeline may not work until a driver is installed + reboot."
fi

# ---- 2. Docker engine -----------------------------------------------------
# The Deep Learning AMI usually ships Docker already; install from the
# official repo only if it's missing.
if ! command -v docker >/dev/null 2>&1; then
  echo "==> Installing Docker engine..."
  install -m 0755 -d /etc/apt/keyrings
  curl -fsSL https://download.docker.com/linux/ubuntu/gpg \
    | gpg --dearmor -o /etc/apt/keyrings/docker.gpg
  chmod a+r /etc/apt/keyrings/docker.gpg
  UBUNTU_CODENAME="$(. /etc/os-release && echo "$VERSION_CODENAME")"
  echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] \
https://download.docker.com/linux/ubuntu $UBUNTU_CODENAME stable" \
    > /etc/apt/sources.list.d/docker.list
  apt-get update -y
  apt-get install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin
fi
systemctl enable --now docker
usermod -aG docker ubuntu || true

# ---- 2b. NVIDIA container toolkit (GPU passthrough into containers) -------
if ! command -v nvidia-ctk >/dev/null 2>&1; then
  echo "==> Installing NVIDIA container toolkit..."
  curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey \
    | gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
  curl -fsSL https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list \
    | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' \
    > /etc/apt/sources.list.d/nvidia-container-toolkit.list
  apt-get update -y
  apt-get install -y nvidia-container-toolkit
fi
nvidia-ctk runtime configure --runtime=docker
systemctl restart docker

# ---- 3. Workspace + build the deployment image ----------------------------
mkdir -p "$APP_DIR/workspace"
chown -R ubuntu:ubuntu "$APP_DIR"
echo "==> Building Docker image $IMAGE_TAG (~10-15 min: torch + ML stack)..."
docker build -f "$APP_DIR/deploy/Dockerfile" -t "$IMAGE_TAG" "$APP_DIR"

# ---- 4. Fetch the fine-tuned keypoint model into the host workspace -------
# Weights are too large for git (265MB) and workspace/ is gitignored, so the
# ev_posw futsal keypoint model ships as a GitHub Release asset. Download it
# into the flat layout the web picker scans (workspace/models/keypoint_*/ with
# best_model.pt + model_meta.json) so it shows up in "Pick a keypoint model".
# It lands on the host and is visible in the container via the bind mount.
# Idempotent: skip when the sha256 already matches.
echo "==> Fetching ev_posw keypoint model..."
sudo -u ubuntu bash -eux <<'KPMODEL'
cd /home/ubuntu/AgentCoach
KP_DIR="workspace/models/keypoint_ev_posw"
KP_SHA="baf423fc411c045ed7ccc20b201e366359253bf07e04666c7bbe76483ad7550b"
BASE="https://github.com/outstandingcandy/AgentCoach/releases/download/keypoint-model-ev_posw"
mkdir -p "$KP_DIR"
if [ -f "$KP_DIR/best_model.pt" ] && echo "$KP_SHA  $KP_DIR/best_model.pt" | sha256sum -c - >/dev/null 2>&1; then
  echo "keypoint model already present + verified; skipping download"
else
  curl -fSL "$BASE/best_model.pt"   -o "$KP_DIR/best_model.pt"
  curl -fSL "$BASE/model_meta.json" -o "$KP_DIR/model_meta.json"
  echo "$KP_SHA  $KP_DIR/best_model.pt" | sha256sum -c -
fi
KPMODEL

# ---- 5. systemd service (manages the container) ---------------------------
echo "==> Installing systemd unit..."
install -m 0644 "$APP_DIR/deploy/goal-insight-web.service" \
  /etc/systemd/system/goal-insight-web.service
systemctl daemon-reload
systemctl enable --now goal-insight-web.service

echo "==> goal-insight provisioning finished at $(date -u)"
echo "==> Service status:"
systemctl --no-pager status goal-insight-web.service || true
