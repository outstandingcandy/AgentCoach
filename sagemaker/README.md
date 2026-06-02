# Remote stage execution (SageMaker Processing Jobs)

`field_registration` and `tracking` can run on a managed GPU
instance instead of locally. Default behaviour is unchanged: stages
run locally unless you opt in.

## Why
- Local CPU/GPU is the bottleneck for these two stages (HRNet inference
  on every sampled frame, YOLOv8x + OSNet per detection).
- Other stages (`event_detection`, `highlights`, `track_consolidation`,
  `annotated_video`) are IO/CPU-bound and stay local — they read the
  files this remote run leaves on disk.

## One-time AWS setup

```bash
# Creates IAM role, ECR repo, S3 bucket. Idempotent.
bash sagemaker/setup_aws.sh

# Push weights once (and after every PnLCalib finetune iteration).
bash sagemaker/upload_weights.sh

# Build + push the container image (and after every requirements bump).
bash sagemaker/build_and_push.sh
```

The setup script prints a `sagemaker:` config block at the end. Copy
its values into the same block at the bottom of `configs/default.yaml`
(or your per-clip config). The four required keys are `region`,
`role_arn`, `image_uri`, `s3_bucket`.

## Running

```bash
# Local (default — unchanged from before)
goalinsight --video clip.mp4 --output output/ --config configs/kids_soccer.yaml

# Remote field_registration only (tracking still local)
goalinsight --video clip.mp4 --output output/ --config configs/kids_soccer.yaml \
            --remote-stages field_registration

# Remote both
goalinsight --video clip.mp4 --output output/ --config configs/kids_soccer.yaml \
            --remote-stages field_registration,tracking
```

The remote stage uploads inputs to S3, submits a Processing Job, polls
until completion, and downloads the products into the same
`output/<run>/<stage>/` directory the local path would have created.
Subsequent stages (`event_detection`, etc.) read those files
unchanged.

## Anatomy of a remote run

Inputs the client uploads to `s3://<bucket>/pipeline-runs/<run-id>/inputs/`:
- `video/<name>.mp4` — the raw clip
- `config/config.json` — the merged config (default + user) serialised
- `calibration/*.pkl,.json` — only for tracking; the field_registration
  outputs

Inside the container at `/opt/ml/processing/input/`:
- `video/`, `config/`, `weights/`, and `calibration/` (tracking only)

Outputs at `/opt/ml/processing/output/<stage>/` get auto-synced back to
`s3://<bucket>/pipeline-runs/<run-id>/<stage>/`. The client downloads
the small allow-list (homographies, json) and skips visualizations
unless `sagemaker.fetch_visualizations: true`.

## Operating notes

- **Image stays small (~3 GB)** because weights live in S3, not the
  image. Re-finetune a model? Just re-run `upload_weights.sh`.
- **First-job latency** is ~2-3 minutes (SageMaker provisioning + image
  pull). After the image is cached on the instance type, subsequent
  jobs start within ~1 minute.
- **Costs**: `ml.g5.xlarge` is ~\$1.41/hr on-demand. A 10-minute clip
  field_registration takes ~3-5 min wall clock = ~\$0.10. Tracking is
  similar.
- **Failures** surface as `RuntimeError` in the local pipeline. The
  failure reason from `DescribeProcessingJob` is included; full logs
  are in CloudWatch under `/aws/sagemaker/ProcessingJobs/<job-name>`.

## Troubleshooting

- `--remote-stages includes 'X' but the config lacks a complete
  sagemaker block`: setup_aws.sh hasn't been run, or its output wasn't
  copied into `configs/default.yaml`.
- Job ends with `Failed: Image pull failed`: ECR auth misconfigured or
  the image hasn't been pushed yet.
- Job ends with `Failed: ResourceLimitExceeded`: account limit on the
  requested instance type. Switch `default_instance_type` to a tier
  you have quota for, or request an increase.
- Tracking job complains about missing calibration: you likely ran
  `--remote-stages tracking` without first running field_registration
  (locally or remotely). Run that stage first; its outputs feed into
  tracking via the upload step.
