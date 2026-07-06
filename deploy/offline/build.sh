#!/usr/bin/env bash
# Build the goalinsight offline Docker image.
#
# Stages host-side artifacts (CLIP-ReID weights, demo video, annotation)
# into deploy/offline/_build_assets/ — they're too large to live in
# git, so we pull them from the local workspace at build time.
#
# Usage:
#   ./deploy/offline/build.sh                       # build with default tag
#   IMAGE_TAG=goalinsight:dev ./deploy/offline/build.sh
#   SKIP_BUILD=1 ./deploy/offline/build.sh          # stage assets only
set -euo pipefail

# Resolve repo root regardless of where the script is invoked from.
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

IMAGE_TAG="${IMAGE_TAG:-goalinsight:offline-latest}"
ASSETS_DIR="deploy/offline/_build_assets"
CLIP_REID_SRC="workspace/models/ViT-L-14_openai/Paper/weights_e4.pth"
EXAMPLE_VIDEO_SRC="workspace/videos/0626_1_part_000.mov"
EXAMPLE_ANNOTATIONS_SRC="workspace/annotations/0626_1_part_000"

echo "=== goalinsight offline image build ==="
echo "  repo: $REPO_ROOT"
echo "  tag:  $IMAGE_TAG"
echo

# Sanity-check host inputs exist before we kick off a 10+ minute
# docker build that would otherwise crash mid-COPY.
for src in "$CLIP_REID_SRC" "$EXAMPLE_VIDEO_SRC" "$EXAMPLE_ANNOTATIONS_SRC"; do
    if [[ ! -e "$src" ]]; then
        echo "ERROR: missing required input: $src" >&2
        echo "  CLIP-ReID weights come from https://github.com/KonradHabel/clip_reid" >&2
        echo "  Example video + annotation are run-time artefacts from a real annotate session." >&2
        exit 1
    fi
done

echo "Staging build assets into $ASSETS_DIR ..."
rm -rf "$ASSETS_DIR"
mkdir -p \
    "$ASSETS_DIR/clip_reid_weights/ViT-L-14_openai/Paper" \
    "$ASSETS_DIR/annotations/example_video"

# CLIP-ReID weights (~1.2 GB) — hard-link if possible so we don't
# double the disk usage during the build context tar.
ln -f "$CLIP_REID_SRC" \
    "$ASSETS_DIR/clip_reid_weights/ViT-L-14_openai/Paper/weights_e4.pth" \
    || cp "$CLIP_REID_SRC" \
       "$ASSETS_DIR/clip_reid_weights/ViT-L-14_openai/Paper/weights_e4.pth"

# Trim the 30-second 0626_1_part_000.mov down to 10s for the demo
# video. Avoids shipping the full 36 MB clip just for a smoke test.
# Falls back to a straight copy if ffmpeg isn't installed locally.
if command -v ffmpeg >/dev/null 2>&1; then
    echo "Trimming example_video.mp4 to 10s via ffmpeg ..."
    ffmpeg -y -loglevel error -ss 0 -t 10 -i "$EXAMPLE_VIDEO_SRC" \
        -c:v libx264 -crf 23 -preset veryfast -an \
        "$ASSETS_DIR/example_video.mp4"
else
    echo "ffmpeg not found — copying full $EXAMPLE_VIDEO_SRC instead."
    cp "$EXAMPLE_VIDEO_SRC" "$ASSETS_DIR/example_video.mp4"
fi

# Annotation for the fixed_camera backend. Tagged under stem
# "example_video" because that's what the bundled video is named —
# the futsal template's fixed_camera backend resolves it from
# workspace/annotations/example_video/ at runtime.
cp -r "$EXAMPLE_ANNOTATIONS_SRC"/. "$ASSETS_DIR/annotations/example_video/"

# Surface human-readable summary so the build log shows what was
# staged.
echo
echo "Staged assets:"
du -sh "$ASSETS_DIR"/*/ "$ASSETS_DIR"/*.mp4 2>/dev/null || true
echo

if [[ "${SKIP_BUILD:-0}" == "1" ]]; then
    echo "SKIP_BUILD=1 — exiting after asset stage."
    exit 0
fi

echo "Running docker build (this can take 10-15 min on first run) ..."
docker build \
    -f deploy/offline/Dockerfile \
    -t "$IMAGE_TAG" \
    .

echo
echo "=== build complete ==="
docker images --filter "reference=$IMAGE_TAG" --format \
    "  {{.Repository}}:{{.Tag}}  {{.Size}}  (id {{.ID}})"

# Warn if the image got bloated. 8 GB is roughly:
#   ~3 GB base (pytorch+cuda) + ~3 GB pip wheels (transformers +
#   vllm) + ~1.2 GB clip_reid weights + ~0.5 GB other weights.
SIZE_BYTES=$(docker image inspect "$IMAGE_TAG" --format '{{.Size}}' 2>/dev/null || echo 0)
if (( SIZE_BYTES > 8589934592 )); then  # 8 GiB
    echo
    echo "  ⚠  image is over 8 GiB — consider trimming requirements.txt"
fi
