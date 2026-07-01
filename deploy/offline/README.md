# GoalInsight offline pipeline — Docker quickstart

Self-contained image of the offline pipeline (field registration →
tracking → consolidation → events → player profile). No web app, no
AWS account required.

## Requirements

* NVIDIA GPU with **≥ 6 GB** memory (CLIP-ReID + YOLOv8x concurrently)
* NVIDIA driver **530+**
* Docker with `--gpus all` support (install `nvidia-container-toolkit`
  if `docker run --gpus all hello-world` fails)
* ~8 GB free disk for the image

## Get the image

```bash
# If a colleague shared a tarball:
docker load -i goalinsight-offline-latest.tar

# Or pull from a registry (substitute your registry path):
# docker pull <your-registry>/goalinsight:offline-latest
```

## First run — launch the web viewer

The image's default entrypoint is the FastAPI web app (library /
pipeline / match / tracking / annotate tabs). LLM chat is disabled by
default so no cloud credentials are needed.

```bash
mkdir -p workspace
docker run --rm --gpus all \
    -p 8000:8000 \
    -v "$PWD/workspace:/workspace" \
    goalinsight:offline-latest
# Then open http://localhost:8000/library in your browser.
```

The first launch creates `workspace/{videos,configs,runs,annotations}/`
on disk and seeds `workspace/configs/` with three starter templates
(`fifa.yaml`, `futsal.yaml`, `children.yaml`) you can pick from the
Library page. Drop input videos into `workspace/videos/` and start a
pipeline from the `/pipeline` tab, or trigger one via the CLI override
below.

> **Important — always bind-mount `/workspace`.** Without the
> ``-v "$PWD/workspace:/workspace"`` flag every uploaded video,
> annotation, and pipeline output lives inside the container and is
> destroyed on ``docker rm``. The entrypoint prints a warning at
> startup when it detects a non-mounted workspace.

## Run a pipeline (CLI mode)

To process a single video without the web UI, override the entrypoint:

```bash
mkdir -p out
docker run --rm --gpus all --entrypoint goalinsight \
    -v "$PWD/out:/output" \
    goalinsight:offline-latest \
    --video /opt/goalinsight/example_video.mp4 \
    --config /opt/goalinsight/configs/templates/futsal.yaml \
    --output /output \
    --run-name smoke \
    --no-timestamp
```

Expected: ~80 s wall-clock for the bundled 10-second futsal demo.
Outputs land under `out/smoke/{field_registration,tracking,event_detection}/`.
Use `--entrypoint goalinsight` for any CLI invocation; the same image
serves both modes.

## Run on your own video

```bash
docker run --rm --gpus all --entrypoint goalinsight \
    -v "$PWD/inputs:/input:ro" \
    -v "$PWD/out:/output" \
    goalinsight:offline-latest \
    --video /input/my_match.mp4 \
    --config /input/my_config.yaml \
    --output /output \
    --run-name my_match
```

Where `my_config.yaml` is one of the bundled templates copied + edited
(see `/opt/goalinsight/configs/templates/` inside the container, or
`configs/templates/` in the repo).

## Jersey-number recognition

The default templates ship with `track_consolidation` **enabled** and
`jersey.backend: qwen` (in-process Qwen3.5-2B, HuggingFace
`transformers`). Weights are pre-fetched during image build so the
first track_consolidation run works fully offline. Expect ~8 GB of
GPU memory during that stage. To disable it (e.g. for a batch-only
run where you only want tracks + events), edit the workspace yaml
and remove `- track_consolidation` + `- player_profile` from
`pipeline.stages`.

## Switching jersey-OCR backends

### Bedrock Claude (Sonnet 4.6)
1. In your config, set:
   ```yaml
   jersey_recognition:
     backend: claude
     model_id: us.anthropic.claude-sonnet-4-6
   track_consolidation:
     jersey:
       backend: claude
       model_id: us.anthropic.claude-sonnet-4-6
       ocr_backend: llm
   ```
2. Pass AWS creds at runtime:
   ```bash
   docker run --rm --gpus all \
       -e AWS_ACCESS_KEY_ID=... \
       -e AWS_SECRET_ACCESS_KEY=... \
       -e AWS_DEFAULT_REGION=us-east-1 \
       -v "$PWD/out:/output" \
       goalinsight:offline-latest \
       --video ... --config ... --output /output
   ```

### Google Gemini
Same idea: `jersey.backend: gemini` in config, pass
`-e GOOGLE_GENAI_API_KEY=...` at runtime.

## What's in the image

| Path | What |
|------|------|
| `/opt/goalinsight` | Code + `pip install -e .` |
| `/opt/goalinsight/configs/templates/` | `fifa.yaml`, `futsal.yaml`, `children.yaml` — seeded into `workspace/configs/` on first launch |
| `/opt/goalinsight/example_video.mp4` | 10s futsal demo clip |
| `/opt/goalinsight/annotations/example_video/` | Calibration annotation for the demo clip's `fixed_camera` backend |
| `/opt/goalinsight/models/clip_reid/` | CLIP-ReID ViT-L-14 fine-tuned weights |
| `/opt/goalinsight/models/yolo/` | YOLOv8x detector |
| `~/.cache/torch/` | OSNet + PRTReID |
| `~/.cache/goal-insight/pnlcalib/` | PnLCalib SV_kp + SV_lines |
| `/opt/goalinsight/models/hf/` | Qwen3.5-2B HF snapshot (~8 GB, for jersey OCR) |

## Output layout

```
out/<run_name>/
├── field_registration/
│   ├── homographies.pkl
│   ├── camera_poses.json
│   └── calibration_metadata.json
├── tracking/
│   ├── tracks.json            # per-frame bbox + track_id
│   ├── ball_tracks.json
│   ├── track_features.json    # ReID embeddings
│   └── statistics.json
├── track_consolidation/
│   ├── players.json           # consolidated player roster
│   ├── player_map.json        # raw tid → stable player_id
│   ├── tracks.json            # tracks rewritten with player_id
│   └── frames/                # per-frame JPG + JSON sidecar
├── event_detection/
│   ├── events.json            # all events (pass, shot, carry, etc.)
│   └── goals.json
└── player_profile/
    ├── players_profile.json
    ├── crops/<player_id>_front.jpg
    ├── heatmaps/<player_id>.png
    └── spotlights/<player_id>.mp4
```

## Troubleshooting

**"could not select device driver" / no NVIDIA runtime**
Install `nvidia-container-toolkit`:
```bash
sudo apt-get install -y nvidia-container-toolkit
sudo systemctl restart docker
```

**Out-of-memory on a smaller GPU**
Either drop `reid.clip_reid.batch_size` (default 16 → try 4) in your
config, or switch `reid.backend: osnet` (much lighter).

**mmocr downloading on first run**
The image pre-bakes mmocr weights at build time but the URLs are
sometimes flaky. If the first run downloads ~200 MB before starting
inference, that's normal; subsequent runs are offline.

**The output run already exists**
Drop `--no-timestamp`, or pass a new `--run-name`.

## Building from source

If you have the repo + the CLIP-ReID fine-tuned weights at
`workspace/models/ViT-L-14_openai/Paper/weights_e4.pth` and the demo
video + annotation, run:

```bash
./deploy/offline/build.sh
```

The script stages the build assets and runs `docker build`. Override
the tag with `IMAGE_TAG=my-image:dev ./deploy/offline/build.sh`.

## Licenses

The image bundles weights and code from several upstream projects:

| Component | License | Notes |
|-----------|---------|-------|
| GoalInsight | (project's license) | The code in this repo |
| PyTorch 2.4 | BSD-3 | Base image |
| Ultralytics YOLOv8 | **AGPL-3.0** | ⚠ Distribution implications — read [ultralytics.com/license](https://www.ultralytics.com/license) before sharing the image externally |
| TorchReID OSNet | MIT | |
| BPBreID / PRTReID | MIT (Zenodo) | |
| PnLCalib | MIT (GitHub releases) | |
| MMOCR | Apache-2.0 | |
| OpenCLIP | MIT | |
| CLIP-ReID | MIT | Habel et al. 2022 |
