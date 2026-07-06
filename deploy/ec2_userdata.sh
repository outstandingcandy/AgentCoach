#!/usr/bin/env bash
#
# EC2 first-boot provisioning for the GoalInsight web app.
#
# Passed as --user-data by deploy/deploy_ec2.sh. Runs once as root via
# cloud-init. Installs OS deps, clones the repo, builds the venv, installs
# the app, and starts it under systemd. Model weights are NOT baked — they
# auto-download on the first pipeline run (PnLCalib releases, ultralytics
# YOLO, torchreid OSNet).
#
# Assumes a Deep Learning GPU AMI (NVIDIA driver + CUDA preinstalled). If the
# launcher fell back to a stock Ubuntu AMI, install the driver first (see the
# GPU_DRIVER_FALLBACK block below).
#
# All output is tee'd to /var/log/goal-insight-provision.log.
set -euo pipefail
exec > >(tee -a /var/log/goal-insight-provision.log) 2>&1
echo "==> goal-insight provisioning started at $(date -u)"

# deploy_ec2.sh substitutes these two placeholders before upload.
REPO_URL="__REPO_URL__"
REPO_BRANCH="__REPO_BRANCH__"
APP_DIR="/home/ubuntu/AgentCoach"

export DEBIAN_FRONTEND=noninteractive

# ---- 1. OS packages -------------------------------------------------------
echo "==> Installing OS packages..."
apt-get update -y
apt-get install -y software-properties-common
add-apt-repository -y ppa:deadsnakes/ppa
apt-get update -y
apt-get install -y \
  git ffmpeg \
  python3.12 python3.12-venv python3.12-dev \
  build-essential \
  libgl1 libglib2.0-0

# ---- 1b. GPU driver fallback (only if no NVIDIA driver present) -----------
# The Deep Learning AMI already ships the driver; this is a safety net for a
# stock-Ubuntu fallback launch. nvidia-smi succeeding means we skip it.
if ! command -v nvidia-smi >/dev/null 2>&1 || ! nvidia-smi >/dev/null 2>&1; then
  echo "==> No working NVIDIA driver detected; installing ubuntu-drivers..."
  apt-get install -y ubuntu-drivers-common || true
  ubuntu-drivers autoinstall || echo "WARN: driver autoinstall failed; GPU pipeline may not work until a driver is installed + reboot."
fi

# ---- 2. Clone the repo ----------------------------------------------------
echo "==> Cloning $REPO_URL ($REPO_BRANCH) -> $APP_DIR"
if [[ -d "$APP_DIR/.git" ]]; then
  git -C "$APP_DIR" fetch --depth 1 origin "$REPO_BRANCH"
  git -C "$APP_DIR" checkout "$REPO_BRANCH"
  git -C "$APP_DIR" reset --hard "origin/$REPO_BRANCH"
else
  git clone --depth 1 --branch "$REPO_BRANCH" "$REPO_URL" "$APP_DIR"
fi
chown -R ubuntu:ubuntu "$APP_DIR"

# ---- 3. venv + app install (as the ubuntu user) ---------------------------
# Install the offline requirements set (deploy/offline/requirements.txt): its
# exact, known-good pins match the tested image, whereas the root
# requirements.txt uses looser ranges. It also carries the fastapi/uvicorn web
# deps this host needs. pyproject.toml declares no deps, so `-e . --no-deps`
# just installs the package (mirrors the offline Dockerfile).
echo "==> Building venv and installing the app..."
sudo -u ubuntu bash -eux <<'INSTALL'
cd /home/ubuntu/AgentCoach
python3.12 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip wheel setuptools
pip install -r deploy/offline/requirements.txt
pip install --no-deps -e .
mkdir -p workspace
INSTALL

# ---- 4. systemd service ---------------------------------------------------
echo "==> Installing systemd unit..."
install -m 0644 "$APP_DIR/deploy/goal-insight-web.service" \
  /etc/systemd/system/goal-insight-web.service
systemctl daemon-reload
systemctl enable --now goal-insight-web.service

echo "==> goal-insight provisioning finished at $(date -u)"
echo "==> Service status:"
systemctl --no-pager status goal-insight-web.service || true
