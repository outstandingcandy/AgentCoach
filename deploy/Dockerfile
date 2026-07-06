# GoalInsight web app — deployment image (chat ENABLED).
#
# Differs from deploy/offline/Dockerfile, which targets a credential-free
# offline viewer:
#   * chat is left ENABLED (no GOALINSIGHT_DISABLE_CHAT) — the deployed host
#     carries a Bedrock IAM role, so the /insights chat surface works;
#   * the chat HTML pages are NOT stripped;
#   * model weights are NOT baked in — they download at first use into the
#     bind-mounted /workspace volume (YOLO / OSNet / PnLCalib / the ev_posw
#     fine-tune), keeping the image small and the layout identical to the
#     bare-metal deploy.
#
# Built on the EC2 instance by deploy/ec2_userdata.sh (docker build) and run
# under systemd (deploy/goal-insight-web.service).

# ``-devel`` (not ``-runtime``): ships nvcc + CUDA headers that vLLM's JIT
# path needs when the qwen jersey backend spins up during track_consolidation.
FROM nvidia/cuda:12.1.0-cudnn8-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    YOLO_CONFIG_DIR=/workspace/models/yolo \
    TORCH_HOME=/workspace/models/torch \
    HF_HOME=/workspace/models/hf \
    AWS_REGION=us-east-1

# Interpreter + video/opencv runtime libs + a C compiler (vLLM/Triton JIT).
RUN apt-get update && apt-get install -y --no-install-recommends \
        software-properties-common ca-certificates curl \
    && add-apt-repository -y ppa:deadsnakes/ppa \
    && apt-get update && apt-get install -y --no-install-recommends \
        python3.12 python3.12-venv python3.12-dev \
        python3-pip \
        ffmpeg \
        libgl1 \
        libglib2.0-0 \
        gcc g++ libc6-dev \
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf /usr/bin/python3.12 /usr/local/bin/python \
    && ln -sf /usr/bin/python3.12 /usr/local/bin/python3 \
    && python -m ensurepip --upgrade \
    && python -m pip install --upgrade pip setuptools wheel

WORKDIR /opt/goalinsight

# Python deps first (layer caches across source-only changes). The offline
# requirements set has exact known-good pins and carries the fastapi/uvicorn
# web deps this host needs.
COPY deploy/offline/requirements.txt requirements.txt
RUN python -m pip install --no-cache-dir -r requirements.txt

# GoalInsight source (pyproject declares no deps → --no-deps).
COPY pyproject.toml ./
COPY goalinsight/ ./goalinsight/
COPY configs/ ./configs/
COPY scripts/ ./scripts/
RUN python -m pip install --no-cache-dir --no-deps -e .

# Seed workspace configs on first launch, then exec the web server. Reuses
# the offline entrypoint (chat stays enabled because we don't set
# GOALINSIGHT_DISABLE_CHAT). The Qwen vLLM daemon is disabled at idle — the
# track_consolidation stage spawns it on demand — so the web server doesn't
# hold GPU memory just to serve pages.
ENV QWEN_VLLM_DISABLE=1
COPY deploy/offline/entrypoint.sh /usr/local/bin/goalinsight-entrypoint
RUN chmod +x /usr/local/bin/goalinsight-entrypoint
EXPOSE 8000
ENTRYPOINT ["/usr/local/bin/goalinsight-entrypoint"]
CMD ["--workspace", "/workspace", "--host", "0.0.0.0", "--port", "8000"]
