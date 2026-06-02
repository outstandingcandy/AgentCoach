"""SageMaker Processing Job runner for field_registration / tracking.

Lifts an in-tree pipeline stage and runs it on a managed GPU instance
without touching the local stage interface. The local Pipeline.run
loop calls into the same `Stage.run(ctx)` entry point as before; the
only difference is that — when configured — the stage adapter
delegates to ``run_stage_remote`` instead of the in-process runner.

Lifecycle of a remote stage call:

  1. Upload video, the merged config, and (for tracking) the
     field_registration outputs to a per-run S3 prefix.
  2. create_processing_job → it pulls our ECR image, downloads inputs
     to /opt/ml/processing/input/*, runs entrypoints/run_stage.py, and
     uploads /opt/ml/processing/output/<stage>/* back to S3.
  3. Poll until terminal status; on Completed, download the stage
     output back to the local output_dir so the rest of the local
     pipeline (event_detection, highlights, ...) reads it as if the
     stage had run locally.

Anything wider than this scope (multi-stage DAG, retry policies,
training jobs) is out of scope — keep this file thin and obvious.
"""

from __future__ import annotations

import json
import logging
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Names of files we care about when downloading stage outputs back. We
# don't pull visualizations/ because they're heavy and not consumed by
# downstream stages — set sagemaker.fetch_visualizations=true if you
# need them.
FIELD_REGISTRATION_KEEP = {
    "homographies.pkl",
    "camera_poses.pkl",
    "camera_poses.json",
    "calibration_metadata.json",
    "calibration_results.json",
}
TRACKING_KEEP = {
    "tracks.json",
    "ball_tracks.json",
    "track_features.json",
    "team_assignments.json",
    "tracking.mp4",
    "tracking_stats.json",
}


@dataclass
class SageMakerConfig:
    """Subset of the user config relevant to remote execution.

    Built from the ``sagemaker:`` block of the merged pipeline config.
    All fields are required for remote execution; if any is missing
    the stage adapter should fall back to local mode rather than
    crash.
    """
    region: str
    role_arn: str
    image_uri: str
    s3_bucket: str
    s3_prefix: str = "pipeline-runs"
    weights_s3_prefix: str = "weights"
    instance_type: str = "ml.g5.xlarge"
    instance_count: int = 1
    volume_size_gb: int = 50
    max_runtime_seconds: int = 3600
    fetch_visualizations: bool = False

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "SageMakerConfig | None":
        sm = (config or {}).get("sagemaker") or {}
        required = ("region", "role_arn", "image_uri", "s3_bucket")
        if not all(sm.get(k) for k in required):
            return None
        return cls(
            region=sm["region"],
            role_arn=sm["role_arn"],
            image_uri=sm["image_uri"],
            s3_bucket=sm["s3_bucket"],
            s3_prefix=sm.get("s3_prefix", "pipeline-runs"),
            weights_s3_prefix=sm.get("weights_s3_prefix", "weights"),
            instance_type=sm.get("default_instance_type", sm.get("instance_type", "ml.g5.xlarge")),
            instance_count=int(sm.get("instance_count", 1)),
            volume_size_gb=int(sm.get("volume_size_gb", 50)),
            max_runtime_seconds=int(sm.get("max_runtime_seconds", 3600)),
            fetch_visualizations=bool(sm.get("fetch_visualizations", False)),
        )


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def run_stage_remote(
    stage: str,
    video_path: Path,
    output_dir: Path,
    config: dict[str, Any],
    *,
    sm_config: SageMakerConfig,
    run_id: str | None = None,
) -> None:
    """Run *stage* on SageMaker, leaving products in *output_dir*/<stage>/.

    The call returns synchronously — the caller is the per-video
    pipeline loop, which already runs sequentially, so polling here
    keeps the local UX (CLI prints, error propagation) identical to
    in-process execution.
    """
    if stage not in ("field_registration", "tracking"):
        raise ValueError(
            f"remote execution not implemented for stage '{stage}'"
        )
    import boto3

    run_id = run_id or _new_run_id(video_path)
    s3_root = f"s3://{sm_config.s3_bucket}/{sm_config.s3_prefix}/{run_id}/{stage}"
    weights_root = f"s3://{sm_config.s3_bucket}/{sm_config.weights_s3_prefix}"
    logger.info("[remote/%s] run_id=%s s3=%s", stage, run_id, s3_root)

    s3 = boto3.client("s3", region_name=sm_config.region)
    sm = boto3.client("sagemaker", region_name=sm_config.region)

    # 1. Upload inputs.
    _upload_video(s3, video_path, sm_config.s3_bucket,
                  f"{sm_config.s3_prefix}/{run_id}/inputs/video")
    _upload_config(s3, config, sm_config.s3_bucket,
                   f"{sm_config.s3_prefix}/{run_id}/inputs/config")
    if stage == "tracking":
        _upload_calibration(s3, output_dir / "field_registration",
                            sm_config.s3_bucket,
                            f"{sm_config.s3_prefix}/{run_id}/inputs/calibration")

    # 2. Submit job.
    job_name = _job_name(stage, run_id)
    inputs = _build_processing_inputs(stage, run_id, sm_config, weights_root)
    outputs = _build_processing_outputs(stage, run_id, sm_config)
    _create_processing_job(
        sm,
        job_name=job_name,
        stage=stage,
        sm_config=sm_config,
        inputs=inputs,
        outputs=outputs,
    )

    # 3. Poll.
    _wait_for_completion(sm, job_name)

    # 4. Download products back to the local stage dir.
    keep = (FIELD_REGISTRATION_KEEP if stage == "field_registration"
            else TRACKING_KEEP)
    local_stage_dir = output_dir / stage
    local_stage_dir.mkdir(parents=True, exist_ok=True)
    _download_outputs(
        s3, sm_config.s3_bucket,
        f"{sm_config.s3_prefix}/{run_id}/{stage}",
        local_stage_dir,
        keep,
        fetch_visualizations=sm_config.fetch_visualizations,
    )
    logger.info("[remote/%s] products synced to %s", stage, local_stage_dir)


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------


def _new_run_id(video_path: Path) -> str:
    """Stable-ish per-call id used to namespace S3 prefixes and job names.

    Includes a short uuid suffix so re-running the same video doesn't
    overwrite a previous run's products.
    """
    stem = video_path.stem.replace("_", "-")[:32]
    return f"{stem}-{uuid.uuid4().hex[:8]}"


def _job_name(stage: str, run_id: str) -> str:
    # SageMaker requires <=63 chars and hyphens (no underscores).
    raw = f"goalinsight-{stage}-{run_id}".replace("_", "-")
    return raw[:63]


def _upload_video(s3, video_path: Path, bucket: str, key_prefix: str) -> None:
    key = f"{key_prefix}/{video_path.name}"
    logger.info("uploading video %s → s3://%s/%s", video_path.name, bucket, key)
    s3.upload_file(str(video_path), bucket, key)


def _upload_config(s3, config: dict, bucket: str, key_prefix: str) -> None:
    key = f"{key_prefix}/config.json"
    body = json.dumps(config, default=str).encode()
    s3.put_object(Bucket=bucket, Key=key, Body=body)
    logger.info("uploaded config (%d bytes) → s3://%s/%s", len(body), bucket, key)


def _upload_calibration(s3, calibration_dir: Path, bucket: str, key_prefix: str) -> None:
    if not calibration_dir.is_dir():
        raise RuntimeError(
            f"tracking remote stage requires field_registration outputs at "
            f"{calibration_dir}. Run field_registration first (locally or "
            "remotely) before invoking tracking."
        )
    uploaded = 0
    for src in calibration_dir.iterdir():
        # Skip visualizations & subdirs — tracking only reads the pkl/json.
        if not src.is_file():
            continue
        if src.name not in FIELD_REGISTRATION_KEEP:
            continue
        s3.upload_file(str(src), bucket, f"{key_prefix}/{src.name}")
        uploaded += 1
    if uploaded == 0:
        raise RuntimeError(
            f"no usable calibration files found under {calibration_dir}; "
            f"expected one of: {sorted(FIELD_REGISTRATION_KEEP)}"
        )
    logger.info("uploaded %d calibration files → s3://%s/%s",
                uploaded, bucket, key_prefix)


def _build_processing_inputs(stage, run_id, sm_config, weights_root):
    """ProcessingInput list mapping S3 prefixes to /opt/ml/processing/input/*."""
    base = f"s3://{sm_config.s3_bucket}/{sm_config.s3_prefix}/{run_id}/inputs"
    inputs = [
        {
            "InputName": "video",
            "S3Input": {
                "S3Uri": f"{base}/video",
                "LocalPath": "/opt/ml/processing/input/video",
                "S3DataType": "S3Prefix",
                "S3InputMode": "File",
                "S3DataDistributionType": "FullyReplicated",
            },
        },
        {
            "InputName": "config",
            "S3Input": {
                "S3Uri": f"{base}/config",
                "LocalPath": "/opt/ml/processing/input/config",
                "S3DataType": "S3Prefix",
                "S3InputMode": "File",
                "S3DataDistributionType": "FullyReplicated",
            },
        },
        {
            "InputName": "weights",
            "S3Input": {
                "S3Uri": weights_root,
                "LocalPath": "/opt/ml/processing/input/weights",
                "S3DataType": "S3Prefix",
                "S3InputMode": "File",
                "S3DataDistributionType": "FullyReplicated",
            },
        },
    ]
    if stage == "tracking":
        inputs.append({
            "InputName": "calibration",
            "S3Input": {
                "S3Uri": f"{base}/calibration",
                "LocalPath": "/opt/ml/processing/input/calibration",
                "S3DataType": "S3Prefix",
                "S3InputMode": "File",
                "S3DataDistributionType": "FullyReplicated",
            },
        })
    return inputs


def _build_processing_outputs(stage, run_id, sm_config):
    out_uri = f"s3://{sm_config.s3_bucket}/{sm_config.s3_prefix}/{run_id}/{stage}"
    return [{
        "OutputName": stage,
        "S3Output": {
            "S3Uri": out_uri,
            "LocalPath": f"/opt/ml/processing/output/{stage}",
            "S3UploadMode": "EndOfJob",
        },
    }]


def _create_processing_job(sm, *, job_name, stage, sm_config, inputs, outputs):
    logger.info("submitting job %s (image=%s, instance=%s)",
                job_name, sm_config.image_uri, sm_config.instance_type)
    sm.create_processing_job(
        ProcessingJobName=job_name,
        RoleArn=sm_config.role_arn,
        ProcessingResources={
            "ClusterConfig": {
                "InstanceCount": sm_config.instance_count,
                "InstanceType": sm_config.instance_type,
                "VolumeSizeInGB": sm_config.volume_size_gb,
            },
        },
        AppSpecification={
            "ImageUri": sm_config.image_uri,
            "ContainerEntrypoint": [
                "python",
                "/opt/goalinsight/entrypoints/run_stage.py",
            ],
            "ContainerArguments": ["--stage", stage],
        },
        ProcessingInputs=inputs,
        ProcessingOutputConfig={"Outputs": outputs},
        StoppingCondition={"MaxRuntimeInSeconds": sm_config.max_runtime_seconds},
    )


def _wait_for_completion(sm, job_name: str, poll_seconds: int = 20) -> None:
    """Block until the job reaches a terminal status.

    Logs a heartbeat every poll so a long-running job doesn't look hung
    in CI logs. SageMaker's own Describe* polling is the boringly
    correct way; no need to reach for waiters.
    """
    last_status = None
    while True:
        resp = sm.describe_processing_job(ProcessingJobName=job_name)
        status = resp["ProcessingJobStatus"]
        if status != last_status:
            logger.info("[remote] %s status=%s", job_name, status)
            last_status = status
        if status in ("Completed", "Failed", "Stopped"):
            if status != "Completed":
                reason = resp.get("FailureReason") or resp.get("ExitMessage") or "(no reason)"
                raise RuntimeError(
                    f"Processing job {job_name} ended with status {status}: {reason}"
                )
            return
        time.sleep(poll_seconds)


def _download_outputs(
    s3, bucket: str, key_prefix: str, dest_dir: Path,
    keep: set[str], *, fetch_visualizations: bool,
) -> None:
    """List the stage's S3 output prefix and pull files down.

    We selectively download instead of full-syncing because viz dumps
    are heavy and almost never read back by downstream stages. ``keep``
    is the allow-list of basenames; subdirectories are skipped unless
    ``fetch_visualizations`` is set.
    """
    paginator = s3.get_paginator("list_objects_v2")
    pulled = 0
    for page in paginator.paginate(Bucket=bucket, Prefix=key_prefix):
        for obj in page.get("Contents", []) or []:
            key = obj["Key"]
            rel = key[len(key_prefix):].lstrip("/")
            if "/" in rel:
                # subdirectory — typically visualizations/.
                if not fetch_visualizations:
                    continue
                local = dest_dir / rel
                local.parent.mkdir(parents=True, exist_ok=True)
            else:
                if rel not in keep:
                    continue
                local = dest_dir / rel
            s3.download_file(bucket, key, str(local))
            pulled += 1
    if pulled == 0:
        raise RuntimeError(
            f"job completed but no output files matched the keep-list at "
            f"s3://{bucket}/{key_prefix}/. Check the job logs."
        )
    logger.info("downloaded %d files from s3://%s/%s", pulled, bucket, key_prefix)
