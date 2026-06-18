# GoalInsight

[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/)
[![Code style: ruff](https://img.shields.io/badge/lint-ruff-46a2f1.svg)](https://github.com/astral-sh/ruff)

End-to-end pipeline that turns a single fixed-camera soccer video into player tracks, events, and watchable highlight clips — with an LLM chat interface on top of the resulting match data.

![Tracking demo: ball + players, team-colored boxes, IDs, and a calibrated minimap](docs/media/tracking_demo.gif)

> **Status: research-grade.** Built on weekends to make my kid's grassroots-league recordings interesting to watch. It works on the cameras I have (Veo, GoPro, phone tripods); it has not been hardened for arbitrary footage.

## Why

If you've ever recorded a youth soccer match on a fixed camera, you know the problem:

- The video is 60–90 minutes long, the action is 20% of that, and finding the goal you actually want to re-watch means scrubbing.
- Pro broadcasters get tracking data, event tags, and slow-mo replays for free; amateur footage gets none of that.
- Commercial services (Veo, Pixellot, Trace) are great but closed, expensive per-team, and don't let you ask questions like "how far did number 9 actually run today?"

GoalInsight runs entirely on your own machine (or on AWS if you want GPUs you don't own), gives you the same kinds of artifacts a broadcaster's stack produces — calibration, tracks, events, highlight clips — and exposes them through a chat interface so you can ask things like "show me every shot from team A in the second half" and get back clickable seek tags.

It's also a deliberate sandbox for putting modern Bedrock + AgentCore tool-use against a non-trivial domain dataset.

## What it does

```
Raw fixed-camera video
        │
        ▼
┌──────────────────┐    Field registration: find where the camera is in the
│ Calibration      │    world. Five backends (PnLCalib HRNet, BroadTrack,
│ (camera + pitch) │    NBJW, fixed-intrinsic Physical, plain Homography);
└──────────────────┘    pick by config based on what your footage looks like.
        │
        ▼
┌──────────────────┐    Players: YOLOv8 detection → ByteTrack/BOTSORT →
│ Tracking         │    OSNet/PRTReID re-identification → k-means or
│ (players + ball) │    tracklet team classification.
└──────────────────┘    Ball: YOLO class 32 + center-distance ByteTrack +
        │               two-pass segment classification (ground-roll vs
        ▼               airborne) and per-segment 3D fitting.
┌──────────────────┐    Possession state machine → pass / shot / carry /
│ Events           │    tackle / interception detectors. Shot detector
│ (rule-based      │    subsumes goal detection (Goal/Saved/Off_Target/
│  detectors)      │    Blocked outcomes with shooter attribution).
└──────────────────┘
        │
        ▼
┌──────────────────┐    Recipe-based: an event detector, an analyzer that
│ Highlights       │    plans 4 segments (buildup → strike → celebration →
│ (per goal)       │    replay), and a composer that crops/zooms per
└──────────────────┘    segment, draws the shooter spotlight + ball trail,
        │               and produces an MP4 with optional video2x upscaling
        ▼               + RIFE slow-motion replay.
┌──────────────────┐    FastAPI viewer + Bedrock-backed chat with five
│ Web chat         │    tools (list_events, get_player_stats,
│ (analytics on    │    get_team_stats, get_frame_snapshot, run_python in
│  the run)        │    an AgentCore Code Interpreter sandbox).
└──────────────────┘
```

Full system view, including the SageMaker remote-execution path and the Bedrock + AgentCore chat surface:

![GoalInsight on AWS — full architecture](docs/goalinsight-aws-architecture.png)

The editable source is [`docs/goalinsight-aws-architecture.drawio`](docs/goalinsight-aws-architecture.drawio) (open in [draw.io](https://app.diagrams.net)).

A still from the tracking stage (annotated player + ball detections, with team colors and IDs):

![Tracking output frame](docs/media/tracking_screenshot.png)

## Quick start

Tested on Python 3.12, Ubuntu 22.04, NVIDIA L40S/A10G. CPU-only execution works for the smaller stages but tracking+calibration are practically GPU-bound.

```bash
git clone https://github.com/outstandingcandy/AgentCoach.git
cd AgentCoach
python3.12 -m venv venv && source venv/bin/activate
pip install -e .
pip install -r requirements.txt
```

Drop a video at `data/raw_videos/<your-clip>.mp4` and run:

```bash
goalinsight \
  --video data/raw_videos/<your-clip>.mp4 \
  --output output/ \
  --config configs/clip_000_finetuned.yaml \
  --stages field_registration,tracking,event_detection,highlights
```

### Reproducing the kids_soccer demo

The 60-second `kids_soccer_clip_1250_1310` demo and its fine-tuned KP/line
models are not in git (~520 MB). They live in S3; export the bucket
name via env var and pull them down:

```bash
export GOALINSIGHT_S3_BUCKET=<your-bucket>   # see internal docs

mkdir -p workspace/videos workspace/annotations \
         data/finetuned_models/run_20260605_073045/models \
         data/finetuned_line_models/run_20260605_073744/models

# Demo video → workspace/videos/ (picked up by `goalinsight-web` Library tab)
aws s3 cp "s3://${GOALINSIGHT_S3_BUCKET}/raw_videos/kids_soccer_clip_1250_1310.mp4" \
          workspace/videos/

# Fine-tuned weights → data/finetuned_*/ (paths are hard-coded in
# configs/kids_soccer_physical.yaml, so don't relocate these)
aws s3 cp "s3://${GOALINSIGHT_S3_BUCKET}/finetuned_models/run_20260605_073045/best_model_final.pt" \
          data/finetuned_models/run_20260605_073045/models/
aws s3 cp "s3://${GOALINSIGHT_S3_BUCKET}/finetuned_line_models/run_20260605_073744/best_model_final.pt" \
          data/finetuned_line_models/run_20260605_073744/models/

goalinsight \
  --video workspace/videos/kids_soccer_clip_1250_1310.mp4 \
  --output output/kids_demo \
  --config configs/kids_soccer_physical.yaml
```

#### Annotations land in `workspace/annotations/`

`goalinsight-web` (the unified Library / Annotate / Pipeline / Insights
UI) reads annotations from `workspace/annotations/<video_stem>/`. Save
new annotations there directly — the Annotate tab does this for you when
you `goalinsight-web --workspace ./workspace`.

The 7-frame v2 finetune training set is checked into git under
`output/annotations/kids_soccer_v2/` for reproducibility. To see it in
the Annotate UI, link it into the workspace and bootstrap the index
(`AnnotationIndex` only enumerates frames listed in `index.json` —
it does not scan):

```bash
ln -s "$PWD/output/annotations/kids_soccer_v2" workspace/annotations/

python3 - <<'PY'
import json, re
from pathlib import Path
base = Path("workspace/annotations")
ann = {}
for sub in sorted(base.iterdir()):
    if not sub.is_dir():
        continue
    frames = sorted(int(m.group(1)) for p in sub.iterdir()
                    if (m := re.match(r"^frame_(\d+)\.json$", p.name)))
    if frames:
        ann[sub.name] = {"frames": frames, "last_modified": ""}
(base / "index.json").write_text(
    json.dumps({"version": "2.0", "annotations": ann}, indent=2)
)
PY
```

You can now reproduce the KP/line fine-tunes locally (or via the
**Pipeline** tab's `finetune_keypoints` / `finetune_lines` cards) without
re-annotating.

Outputs land under `output/<run-name>-<timestamp>/`:

```
field_registration/   homographies.pkl, camera_poses.pkl/json, calibration_metadata.json
tracking/             tracks.json, ball_tracks.json, team_assignments.json, tracking.mp4
event_detection/      events.json (all events), goals.json
highlights/           goal_highlight_0001.mp4, ...
annotated_video/      full match with overlays (optional)
```

## CLI flags

| Flag | Default | Description |
|------|---------|-------------|
| `--video` | required | Source video path. |
| `--output` | required | Run output directory. |
| `--config` | required | YAML config file (deep-merged onto `configs/default.yaml`). |
| `--stages` | `field_registration,tracking` | Comma-separated subset of `field_registration,tracking,track_consolidation,event_detection,player_profile,highlights,annotated_video`. `track_consolidation` must precede `event_detection` so events carry stable `A-9`-style player_ids. |
| `--no-viz` | off | Skip the tracking visualization video (saves time and memory). |
| `--keypoint-model` | from config | Override `field_registration.keypoint_model_path`. |
| `--remote-stages` | `none` | Comma-separated stages to run on SageMaker instead of locally (see [SageMaker](#sagemaker-remote-execution-optional)). |
| `--skip-existing` | off | Skip a stage if its output already exists. Useful for partial reruns. |
| `--run-name` | `<config-name>` | Name segment in the output dir. |
| `--no-timestamp` | off | Don't append a timestamp to the run dir. |

## Configuration

Configs in `configs/` override `configs/default.yaml`. The presets that ship:

- `clip_000_finetuned.yaml` — recommended starting point. PnLCalib (finetuned HRNet keypoint+line detector) on broadcast-style footage.
- `clip_000_broadtrack.yaml` — BroadTrack backend (9-parameter camera + Cauchy robust loss + arc-length line constraints). Better on heavy lens distortion.
- `clip_000_physical.yaml` — Fixed-intrinsic camera profile from `camera_profiles.yaml` (e.g. Veo). 7-DOF bounded extrinsics. The most stable backend on non-FIFA pitches.
- `clip_000_homography.yaml` — Plain ground-plane DLT homography. Fast baseline.
- `kids_soccer*.yaml` — Variants tuned for non-FIFA-spec youth pitches (smaller, different goal width).

Key knobs (full list in `configs/default.yaml`):

```yaml
field_registration:
  backend: pnlcalib       # pnlcalib | broadtrack | nbjw | physical | homography
  keypoint_threshold: 0.5
  ransac_threshold: 30    # px

video:
  process_fps: 5          # frame sample rate

detection:
  model: yolov8x.pt

reid:
  backend: osnet          # osnet | prtreid
team_classification:
  backend: kmeans         # kmeans | tracklet
jersey_recognition:
  backend: qwen           # qwen (VLM) | mmocr (DBNet+SAR)

events:
  detectors: [possession, pass, shot, carry, defensive]

highlights:
  recipes: [...]
video_enhancement:
  enabled: false          # video2x upscaling + RIFE slow-mo
  mode: docker            # binary | docker
```

## SageMaker remote execution (optional)

`field_registration` and `tracking` are GPU-bound. They can run on a SageMaker Processing Job (`ml.g5.xlarge`, ~$1.41/hr) instead of locally; the rest of the pipeline still runs on your workstation.

One-time setup:

```bash
bash sagemaker/setup_aws.sh        # creates IAM role, ECR repo, S3 bucket
bash sagemaker/upload_weights.sh   # pushes finetuned weights to S3
bash sagemaker/build_and_push.sh   # builds and pushes the container image
```

Copy the `sagemaker:` block printed by `setup_aws.sh` into `configs/default.yaml`. Then opt in per-run:

```bash
goalinsight --video clip.mp4 --output output/ --config configs/kids_soccer.yaml \
            --remote-stages field_registration,tracking
```

A 10-minute clip costs ~$0.10 wall-clock-equivalent. Full notes in [`sagemaker/README.md`](sagemaker/README.md).

## Web viewer + chat

```bash
python -m goalinsight.web --workspace ./workspace
# default: http://127.0.0.1:8000/
```

A unified workspace UI: pipeline launcher (`/pipeline`), per-stage results
viewer, single-run video viewer + chat (`/insights/<run>`), tracking
diagnostics (`/tracking/<run>`), pitch annotator (`/annotate`), and a
library of videos / runs (`/library`). Runs live under
`<workspace>/runs/<run-name>/`; videos under `<workspace>/videos/`. Use
symlinks to point an existing `output/` tree at `<workspace>/runs`.

The viewer streams the match video alongside a Bedrock-backed chat. The model has five tools:

- `list_events` — filter events by type/team/player/time window
- `get_player_stats` — per-player distance, top speed, touches, passes, shots, goals
- `get_team_stats` — possession share, pass success, shots, tackles, interceptions
- `get_frame_snapshot` — who's on screen and what's near the ball at a moment
- `run_python` — execute Python in an AgentCore Code Interpreter sandbox for ad-hoc analysis (heatmaps, custom aggregations); plots come back as inline images

Example questions that work today:

> "How far did A-9 run? When did they have their best chance?"
> "Compare B-10's first-half and second-half pass success rate."
> "Plot a heatmap of where team A's possessions ended."

### Optional: chat on AgentCore Runtime

The chat agent can also run as an AWS-managed AgentCore Runtime
container instead of inside the FastAPI process — same prompts, same
tools, same SSE shape on the wire. The local app proxies turns to the
runtime via `bedrock-agentcore.InvokeAgentRuntime` and streams the
response back to the browser unchanged. Setup, deploy, and per-run S3
sync are documented in [`deploy/agentcore_runtime/README.md`](deploy/agentcore_runtime/README.md);
toggle by setting `GOALINSIGHT_AGENTCORE_RUNTIME_ARN` (and unset to
revert to local chat).

## Repository layout

```
goalinsight/
  cli.py                     # CLI entry (-> goalinsight)
  pipeline/                  # stage framework, registry, adapters, remote
  field_registration/        # 5 calibration backends + finetune machinery
  tracking/                  # detection, ByteTrack, ReID, ball detector + 3D
  events/                    # detector framework + possession/pass/shot/...
  highlights/                # MatchContext + recipe agents
  annotation/                # Gradio-based pitch keypoint annotator
  web/                       # FastAPI viewer + Bedrock chat
  jersey/                    # OCR + VLM jersey-number recognizers
  video_enhancement/         # video2x wrapper (binary or docker mode)
  utils/, interfaces/        # factories + ABCs
configs/                     # default + per-clip + camera profiles
sagemaker/                   # setup, build, weights upload, entrypoint
deploy/agentcore_runtime/    # optional AgentCore Runtime image for chat
scripts/                     # run_full_pipeline, pipeline.sh, audit/dump tools, ...
tools/                       # make_comparison.py and other utilities
docs/                        # architecture diagrams (draw.io)
```

## Architecture quick links

- Full pipeline & AWS architecture diagram → [`docs/goalinsight-aws-architecture.drawio`](docs/goalinsight-aws-architecture.drawio)
- SageMaker-only zoom → [`docs/goalinsight-sagemaker-deployment.drawio`](docs/goalinsight-sagemaker-deployment.drawio)
- Internal architecture deep-dive → [`CLAUDE.md`](CLAUDE.md) (kept up-to-date as a guide for AI coding assistants and humans alike)

## Development

```bash
# Run individual diagnostic / experimentation scripts
python scripts/run_full_pipeline.py [same flags as goalinsight]
python scripts/run_highlights.py --run-dir output/<run>

# Per-backend quick run scripts
bash scripts/pipeline.sh             # PnLCalib finetuned
bash scripts/pipeline_broadtrack.sh
bash scripts/pipeline_physical.sh

# Finetune the PnLCalib heads (after annotating frames)
# see goalinsight/field_registration/pnlcalib/finetune_*.py
```

There is no formal test suite. Test/debug scripts at the repo root use the `test_*.py` / `debug_*.py` naming convention and are gitignored.

## Security

This is a single-user research tool. The defaults are safe; the configurable footguns are documented:

- **`python -m goalinsight.web` binds to `127.0.0.1` and ships without authentication.** Intended for local single-user use. If you pass `--host 0.0.0.0` to expose it on a network, add your own authentication layer (nginx + basic auth, SSH port-forwarding, etc.).
- **The chat `run_python` tool executes arbitrary Python in an AWS Bedrock AgentCore Code Interpreter sandbox.** The sandbox is AWS-managed, but token usage and sandbox session minutes are billed against the AWS account whose credentials are picked up from the default credential chain.
- **Pipeline calibration outputs (`homographies.pkl`, `camera_poses.pkl`) are pickle files.** Do not load pipeline output directories you didn't produce yourself — pickle deserialization is RCE-equivalent. A safer on-disk format is on the roadmap.
- **`video_enhancement.mode: docker` runs `ghcr.io/k4yt3x/video2x` with `--gpus all` and a host-directory volume mount.** Filenames and config arguments are now `shlex.quote`'d before reaching `bash -c`, but the container itself runs as root.
- **The SageMaker execution role created by `sagemaker/setup_aws.sh` uses `AmazonS3FullAccess` for ergonomic setup.** Production deployments should narrow this to a bucket-scoped inline policy; the script flags this in a comment.
- **Model weights downloaded from PnLCalib's GitHub releases are unpickled by `torch.load(weights_only=False)`.** `weight_downloader.py` now SHA-256-verifies downloads when the hash is pinned in `AVAILABLE_WEIGHTS`; the `sha256` fields ship empty so first-time use logs the hash for you to pin.

## Acknowledgements

- [PnLCalib](https://github.com/mguti97/PnLCalib) — vendored as the calibration baseline (HRNet keypoint+line detector + iterative PnP). Original paper: Gutiérrez-Pérez & Agudo, "PnLCalib: Sports Field Registration via Points and Lines Optimization", 2024.
- [BroadTrack](https://github.com/cmu-rim/broadtrack) — alternative camera model with arc-length line constraints.
- [video2x](https://github.com/k4yt3x/video2x) — Real-ESRGAN + RIFE wrapper used for highlight clip enhancement.
- [Ultralytics YOLO](https://github.com/ultralytics/ultralytics), [TorchReID](https://github.com/KaiyangZhou/deep-person-reid) for detection and re-identification.
- The clean state-machine event detector design is adapted from [SoccerNet-v3](https://www.soccer-net.org/)'s event taxonomy.

## License

[Apache License 2.0](LICENSE). Copyright © 2024–2026 Tang Jie.

The vendored PnLCalib reference port under `goalinsight/field_registration/pnlcalib/` retains its upstream license (see header in those files).

## Contributing

Issues and pull requests are welcome. The code is research-grade and the seams between stages are intentionally explicit so a contributor can swap in (say) a different ReID model or calibration backend without touching the rest of the pipeline. See `CLAUDE.md` for the architectural details.
