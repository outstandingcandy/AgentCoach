# GoalInsight offline pipeline — self-contained Docker image.
#
# Build:  ./deploy/offline/build.sh
# Run:    docker run --gpus all -v $PWD/out:/output goalinsight:offline-latest \
#             --video /opt/goalinsight/example_video.mp4 \
#             --config /opt/goalinsight/configs/templates/futsal.yaml \
#             --output /output
#
# Image contents:
#   * Ubuntu 22.04 + CUDA 12.1 runtime (from nvidia/cuda base)
#   * Python 3.12 (via apt) — matches the dev venv in this repo
#   * torch 2.12 + torchvision 0.27 from PyPI (ships its own
#     CUDA 13 libraries via the nvidia-* wheels, which work fine on
#     a CUDA 12.1 driver because pip-shipped libs are self-contained)
#   * Full Python dep stack — open_clip_torch 3.x, transformers 5.x,
#     ultralytics 8.4, torchreid (with gdown + tensorboard transitive
#     deps that the PyPI fork forgets to declare)
#   * Pre-baked model weights so first run never phones home:
#       - YOLOv8x   (~250 MB)
#       - OSNet     (~40 MB,  via torchreid auto-download)
#       - PnLCalib  (SV_kp + SV_lines, ~50 MB each)
#       - CLIP-ReID ViT-L-14 fine-tuned (~1.2 GB, staged by build.sh
#         from the host's workspace/models/)
#
# PRTReID is NOT pre-baked: it ships as a git-only package (no PyPI),
# and the sample configs default to clip_reid / osnet. Users who want
# the PRTReID backend should install it themselves at runtime:
#   docker run ... --entrypoint bash goalinsight:offline-latest \
#     -c "pip install git+https://github.com/VlSomers/prtreid.git && goalinsight ..."
#   * Example configs for futsal / fifa / kids pitch types
#   * A short demo video for end-to-end smoke testing
#
# Jersey-OCR: track_consolidation is opt-in (needs a VLM backend —
# claude/gemini/qwen). Default pipeline runs field_registration →
# tracking → event_detection → player_profile, which works fully
# offline.

FROM nvidia/cuda:12.1.0-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    YOLO_CONFIG_DIR=/opt/goalinsight/models/yolo \
    TORCH_HOME=/opt/goalinsight/models/torch \
    HF_HOME=/opt/goalinsight/models/hf \
    # The web server's chat surface (LLM-powered match Q&A via
    # Bedrock / AgentCore) is disabled by default — none of the
    # bundled configs need it and enabling it would require AWS
    # credentials to be passed at runtime. Override at ``docker run``
    # time with ``-e GOALINSIGHT_DISABLE_CHAT=0`` if you ship creds.
    GOALINSIGHT_DISABLE_CHAT=1

# System libraries:
#   python3.12         — interpreter (Ubuntu 22.04 ships 3.10 by default;
#                        deadsnakes PPA provides 3.12)
#   python3.12-venv    — needed for pip's ensurepip path
#   python3-pip        — bootstraps pip; we upgrade it via pip itself
#   ffmpeg             — video decode (cv2 leans on it for non-mp4 / h.265)
#   libgl1 libglib2.0  — opencv-python runtime deps
#   ca-certificates    — for the Zenodo / GitHub model downloads
RUN apt-get update && apt-get install -y --no-install-recommends \
        software-properties-common ca-certificates curl \
    && add-apt-repository -y ppa:deadsnakes/ppa \
    && apt-get update && apt-get install -y --no-install-recommends \
        python3.12 python3.12-venv python3.12-dev \
        python3-pip \
        ffmpeg \
        libgl1 \
        libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf /usr/bin/python3.12 /usr/local/bin/python \
    && ln -sf /usr/bin/python3.12 /usr/local/bin/python3 \
    && python -m ensurepip --upgrade \
    && python -m pip install --upgrade pip setuptools wheel

WORKDIR /opt/goalinsight

# --- Layer 3: Python deps ----------------------------------------
# Copy requirements first so this layer caches across source-only
# changes. We let pip resolve torch / transformers / open_clip
# freely — this mirrors the dev venv (which has torch 2.12 + CUDA 13
# wheels working out of the box on a CUDA 12.1 driver).
COPY deploy/offline/requirements.txt requirements.txt
RUN python -m pip install --no-cache-dir -r requirements.txt

# --- Layer 4: GoalInsight source ---------------------------------
COPY pyproject.toml ./
COPY goalinsight/ ./goalinsight/
COPY configs/ ./configs/
COPY scripts/ ./scripts/
RUN python -m pip install --no-cache-dir --no-deps -e .

# Strip the chat-only HTML pages so a misconfigured deployment (e.g.
# someone flipping GOALINSIGHT_DISABLE_CHAT=0 without supplying AWS
# creds) can't accidentally serve a half-broken viewer that 500s on
# every chat call. The Python chat routes return 404 anyway thanks
# to the env flag; removing the HTML keeps the failure mode clean.
RUN rm -f \
    /opt/goalinsight/goalinsight/web/static/viewer.html \
    /opt/goalinsight/goalinsight/web/static/insights_index.html \
    /opt/goalinsight/goalinsight/web/static/index.html

# --- Layer 5: Pre-baked weights ----------------------------------
# Trigger each ReID / detector backend's auto-download path once at
# build time so the running container can work with --network none.
# Run on CPU so the build doesn't need GPU access.
RUN python -c "from ultralytics import YOLO; YOLO('yolov8x.pt')" \
 && python -c "from goalinsight.tracking.reid import OSNetExtractor; \
               e = OSNetExtractor({'device':'cpu'}); e.load_model()" \
 && python -c "from goalinsight.field_registration.pnlcalib.weight_downloader \
                   import PnLCalibWeightDownloader; \
               d = PnLCalibWeightDownloader(); \
               d.get_weight_path('SV_kp'); d.get_weight_path('SV_lines')"

# CLIP-ReID fine-tuned weights. build.sh stages these into
# deploy/offline/_build_assets/ from the host workspace.
COPY deploy/offline/_build_assets/clip_reid_weights/ \
     /opt/goalinsight/models/clip_reid/

# --- Layer 6: Examples -------------------------------------------
# The 3 user-facing scene templates already landed at
# /opt/goalinsight/configs/templates/ via the source COPY above —
# entrypoint.sh seeds workspace/configs/ from there.
COPY deploy/offline/_build_assets/example_video.mp4 \
     /opt/goalinsight/example_video.mp4
COPY deploy/offline/_build_assets/annotations/ \
     /opt/goalinsight/annotations/

# --- Layer 7: Entrypoint -----------------------------------------
# Default to the FastAPI web server so ``docker run -p 8000:8000 ...
# goalinsight:offline-latest`` drops a colleague straight into the
# library / pipeline / match / tracking / annotate viewer. The
# entrypoint script seeds /workspace/configs/ with fifa / futsal /
# children starter templates on first launch (skip if files already
# exist), then execs goalinsight-web. To run the CLI pipeline instead,
# override with ``--entrypoint goalinsight``.
COPY deploy/offline/entrypoint.sh /usr/local/bin/goalinsight-entrypoint
RUN chmod +x /usr/local/bin/goalinsight-entrypoint
EXPOSE 8000
ENTRYPOINT ["/usr/local/bin/goalinsight-entrypoint"]
CMD ["--workspace", "/workspace", "--host", "0.0.0.0", "--port", "8000"]
