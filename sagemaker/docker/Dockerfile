# PyTorch GPU base — provides matching CUDA+cuDNN+torch wheel without us
# having to babysit driver/lib alignment. Pin to a known good version
# instead of :latest so reproducibility is in our hands, not nvidia's.
FROM pytorch/pytorch:2.4.1-cuda12.1-cudnn9-runtime

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# OpenCV headless still needs a few system libs at runtime (libGL is the
# usual culprit). awscli is for the entrypoint to pull weights from S3.
RUN apt-get update && apt-get install -y --no-install-recommends \
        ffmpeg \
        libgl1 \
        libglib2.0-0 \
        unzip \
        curl \
    && curl -fsSL "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o awscli.zip \
    && unzip -q awscli.zip && ./aws/install && rm -rf awscli.zip aws \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /opt/goalinsight

# Install dependencies first so layer caching kicks in when only code
# changes. requirements.txt is the slim set tailored to fr+tracking;
# everything else (mmocr/sam2/qwen-vl) lives in the full requirements
# kept for local dev.
COPY sagemaker/docker/requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy source. .dockerignore (next file) keeps data/, output/, venv/
# and webapp_videos/ out so the build context stays small.
COPY pyproject.toml setup.py* ./
COPY goalinsight/ ./goalinsight/
COPY sagemaker/entrypoints/ ./entrypoints/

# Install goalinsight itself so absolute imports work without
# PYTHONPATH gymnastics.
RUN pip install --no-cache-dir --no-deps -e .

# Default command is overridden by the SageMaker container_arguments
# at submit time; this is just for ad-hoc docker-run tests.
ENTRYPOINT ["python", "/opt/goalinsight/entrypoints/run_stage.py"]
