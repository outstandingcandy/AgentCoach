# GoalInsight offline pipeline — self-contained Docker image.
#
# Build:  ./deploy/offline/build.sh
# Run:    docker run --gpus all -v $PWD/out:/output goalinsight:offline-latest \
#             --video /opt/goalinsight/example_video.mp4 \
#             --config /opt/goalinsight/example_configs/futsal_sample.yaml \
#             --output /output
#
# Image contents:
#   * Python 3.12 + CUDA 12.1 + PyTorch 2.4 (from base image)
#   * Full Python dep stack (see deploy/offline/requirements.txt)
#   * Pre-baked model weights so first run never phones home:
#       - YOLOv8x   (~250 MB)
#       - OSNet     (~40 MB, via torchreid auto-download)
#       - PRTReID   (~100 MB, via Zenodo auto-download)
#       - PnLCalib  (SV_kp + SV_lines, ~50 MB each)
#       - mmocr     (DBNet + SAR, ~200 MB on first inference)
#       - CLIP-ReID ViT-L-14 fine-tuned (~1.2 GB, staged by build.sh
#         from the host's workspace/models/)
#   * Example configs for futsal / fifa / kids pitch types
#   * A short demo video for end-to-end smoke testing
#
# Jersey-OCR default is **mmocr (local)** — works without any cloud
# credentials. To use Bedrock Claude or Gemini, override the backend
# in your own config and supply the AWS / GOOGLE_GENAI_API_KEY env
# vars at `docker run` time.

FROM pytorch/pytorch:2.4.1-cuda12.1-cudnn9-runtime

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    # Pin ultralytics + torch hub caches inside the image so the
    # baked weights land at predictable paths.
    YOLO_CONFIG_DIR=/opt/goalinsight/models/yolo \
    TORCH_HOME=/opt/goalinsight/models/torch \
    HF_HOME=/opt/goalinsight/models/hf

# System libraries:
#   ffmpeg   — video decode (cv2 leans on it for non-mp4 / h.265 / etc)
#   libgl1   — opencv-python runtime dep
#   libglib2.0-0 — same
RUN apt-get update && apt-get install -y --no-install-recommends \
        ffmpeg \
        libgl1 \
        libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /opt/goalinsight

# --- Layer 3: Python deps ----------------------------------------
# Copy requirements first so this layer caches across source-only
# changes. mmocr + mmcv compile against the base image's torch; pip
# can take 5-10 min the first time.
COPY deploy/offline/requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# --- Layer 4: GoalInsight source ---------------------------------
# Copy only what the offline CLI needs. ``.dockerignore`` keeps
# workspace/runs/, workspace/videos/, and the web-app static files
# out so the build context stays small.
COPY pyproject.toml ./
COPY goalinsight/ ./goalinsight/
COPY configs/ ./configs/
COPY scripts/ ./scripts/
RUN pip install --no-cache-dir --no-deps -e .

# --- Layer 5: Pre-baked weights ----------------------------------
# Trigger each ReID / detector backend's auto-download path once at
# build time so the running container can work with --network none.
# Run on CPU so the build doesn't need GPU access.
RUN python -c "from ultralytics import YOLO; YOLO('yolov8x.pt')" \
 && python -c "from goalinsight.tracking.reid import OSNetExtractor; \
               e = OSNetExtractor({'device':'cpu'}); e.load_model()" \
 && python -c "from goalinsight.tracking.reid import PRTReIDExtractor; \
               e = PRTReIDExtractor({'device':'cpu'}); e.load_model()" \
 && python -c "from goalinsight.field_registration.pnlcalib.weight_downloader \
                   import PnLCalibWeightDownloader; \
               d = PnLCalibWeightDownloader(); \
               d.download('SV_kp'); d.download('SV_lines')"

# mmocr DBNet + SAR weights — lazily downloaded on first inference.
# Pull them now via a tiny dummy run so the offline mode works.
# This is allowed to fail (mmocr's weight URLs occasionally change);
# the user can still re-fetch at first inference if the build hits
# a transient outage.
RUN python -c "import numpy as np; \
               from goalinsight.jersey.mmocr_recognizer import MMOCRJerseyRecognizer; \
               r = MMOCRJerseyRecognizer({'device':'cpu'}); \
               r._init_models()" \
    || echo "mmocr weight pre-bake failed — first inference will retry"

# CLIP-ReID fine-tuned weights. build.sh stages these into
# deploy/offline/_build_assets/ from the host workspace; absent that
# stage step the COPY would fail. The destination matches the path
# baked into example_configs/*_sample.yaml so configs work as-is.
COPY deploy/offline/_build_assets/clip_reid_weights/ \
     /opt/goalinsight/models/clip_reid/

# --- Layer 6: Examples -------------------------------------------
COPY deploy/offline/example_configs/ /opt/goalinsight/example_configs/
COPY deploy/offline/_build_assets/example_video.mp4 \
     /opt/goalinsight/example_video.mp4

# Annotation frame the futsal sample config + fixed_camera backend
# needs. Staged by build.sh from workspace/annotations/.
COPY deploy/offline/_build_assets/annotations/ \
     /opt/goalinsight/annotations/

# --- Layer 7: Entrypoint -----------------------------------------
# Run goalinsight directly. Override with --entrypoint bash if a
# user wants an interactive shell.
ENTRYPOINT ["goalinsight"]
CMD ["--help"]
