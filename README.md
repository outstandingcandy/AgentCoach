# GoalInsight

A soccer video analysis pipeline: field registration (camera calibration), player tracking and identification, ball tracking with 3D trajectory estimation, event detection (passes, shots, goals, tackles), and recipe-based highlight generation.

## Quick start

```bash
source venv/bin/activate
pip install -e .
pip install -r requirements.txt

goalinsight \
  --video data/raw_videos/clip.mp4 \
  --output output/ \
  --config configs/clip_000_finetuned.yaml \
  --stages field_registration,tracking,event_detection,highlights
```

For an architectural overview see [`CLAUDE.md`](./CLAUDE.md). Key subsystems live under `goalinsight/{field_registration,tracking,events,highlights,web}/`. Configs in `configs/` override `configs/default.yaml` via deep merge.

## Optional: SageMaker remote execution

`field_registration` and `tracking` (the GPU-heavy stages) can run on a SageMaker Processing Job. One-time setup:

```bash
bash sagemaker/setup_aws.sh
bash sagemaker/upload_weights.sh
bash sagemaker/build_and_push.sh
```

Then opt in per-run with `--remote-stages field_registration,tracking`. See [`sagemaker/README.md`](./sagemaker/README.md).

## Optional: Web viewer + chat

```bash
python scripts/run_web_viewer.py --run-dir output/<run-id>
```

Opens a FastAPI viewer with a Bedrock-backed chat about the match.

## Security

- **The web viewer (`scripts/run_web_viewer.py`) and annotator (`scripts/run_annotator.py`) bind to `127.0.0.1` by default and ship without authentication.** They are intended for local single-user use. If you pass `--host 0.0.0.0` to expose them on a network, add your own authentication layer (e.g., behind nginx with basic auth, or via SSH port-forwarding).
- The chat tool `run_python` executes arbitrary Python in an AWS Bedrock AgentCore Code Interpreter sandbox; cost and any prompt-driven behaviour are billed against the AWS account whose credentials are picked up from the default credential chain.
- Pipeline calibration outputs (`homographies.pkl`, `camera_poses.pkl`) are pickle files. Do not load pipeline output directories from untrusted sources — pickle deserialization is RCE-equivalent.
- The `video_enhancement.mode: docker` path runs `ghcr.io/k4yt3x/video2x` with `--gpus all` and a host-directory volume mount for output. The container runs as root inside; treat it as a trust boundary you control.
- See `sagemaker/setup_aws.sh` for the IAM policy attached to the SageMaker execution role. The default uses `AmazonS3FullAccess` for ergonomic setup; production deployments should narrow this to a bucket-scoped inline policy.

## License

(add here before publishing)
