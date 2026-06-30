#!/usr/bin/env python3
"""Fine-tune the PnLCalib keypoint or line head — locally or on SageMaker.

By default this is a thin wrapper that just dispatches to the existing
trainer ``main()``s, so all of their CLI flags (``--epochs``, ``--lr``,
``--batch_size``, ...) work unchanged. Pass ``--remote`` to upload the
annotation directory + pretrained weights and run the same training on
a SageMaker TrainingJob — the produced model.tar.gz is downloaded back
to ``--output_dir/run_<timestamp>/``.

Usage:

    # Local (default)
    python scripts/train_finetune.py --kind keypoint \\
        --annotations_dir output/annotations/kids_soccer_clip_1250_1310 \\
        --pretrained ~/.cache/goal-insight/pnlcalib/SV_kp \\
        --output_dir data/finetuned_models \\
        --batch_size 4 --epochs 100 --hflip_prob 0

    # Remote (SageMaker TrainingJob, same flags)
    python scripts/train_finetune.py --kind line --remote \\
        --config workspace/configs/kids_soccer_physical.yaml \\
        --annotations_dir output/annotations/kids_soccer_clip_1250_1310 \\
        --pretrained ~/.cache/goal-insight/pnlcalib/SV_lines \\
        --output_dir data/finetuned_line_models \\
        --batch_size 4 --epochs 200

The ``--remote`` path requires a ``sagemaker:`` block in ``--config``
with at least ``region``, ``role_arn``, ``image_uri``, ``s3_bucket``.
The container image is the same one used for inference (run_stage.py)
— SageMaker invokes ``train_entrypoint.py`` instead.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import tarfile
import time
import uuid
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)


# ----------------------------------------------------------------------
# Local mode: dispatch to the existing trainer main()s, untouched.
# ----------------------------------------------------------------------

def _run_local(kind: str, trainer_argv: list[str]) -> int:
    """Patch sys.argv and call the trainer's main() in-process.

    Calling main() directly (vs subprocess) keeps logs / tracebacks /
    keyboard interrupts behaving like the user ran the trainer module
    themselves.
    """
    if kind == "keypoint":
        from goalinsight.field_registration.pnlcalib.finetune.train_finetune import main as run
    else:
        from goalinsight.field_registration.pnlcalib.finetune.train_finetune_lines import main as run

    logger.info("local %s training, argv=%s", kind, trainer_argv)
    sys.argv = [f"train_finetune_{kind}.py"] + list(trainer_argv)
    rc = run()
    return int(rc) if isinstance(rc, int) else 0


# ----------------------------------------------------------------------
# Remote mode: SageMaker TrainingJob.
# ----------------------------------------------------------------------

def _trainer_argv_to_hparams(argv: list[str]) -> dict[str, str]:
    """Convert ['--epochs', '100', '--hflip_prob', '0', '--freeze_backbone']
    into {'epochs': '100', 'hflip_prob': '0', 'freeze_backbone': 'true'}.

    SageMaker stores hyperparameters as string-keyed strings, so the
    container-side entrypoint reverses this. ``--annotations_dir``,
    ``--pretrained`` and ``--output_dir`` are stripped because the
    container always wires those to fixed /opt/ml paths.
    """
    skip = {"annotations_dir", "pretrained", "output_dir"}
    out: dict[str, str] = {}
    i = 0
    while i < len(argv):
        token = argv[i]
        if not token.startswith("--"):
            raise ValueError(f"unexpected positional arg in trainer argv: {token!r}")
        key = token[2:]
        if i + 1 < len(argv) and not argv[i + 1].startswith("--"):
            value = argv[i + 1]
            i += 2
        else:
            value = "true"   # store_true flag
            i += 1
        if key in skip:
            continue
        out[key] = value
    return out


def _upload_dir_to_s3(s3, src: Path, bucket: str, key_prefix: str) -> int:
    """Recursively upload ``src``'s contents under ``s3://bucket/key_prefix/``."""
    n = 0
    for p in src.rglob("*"):
        if not p.is_file():
            continue
        rel = p.relative_to(src).as_posix()
        s3.upload_file(str(p), bucket, f"{key_prefix}/{rel}")
        n += 1
    if n == 0:
        raise RuntimeError(f"no files uploaded — empty source dir? {src}")
    logger.info("uploaded %d files → s3://%s/%s/", n, bucket, key_prefix)
    return n


def _wait_for_training_job(sm, job_name: str, poll_seconds: int = 20) -> dict:
    """Poll DescribeTrainingJob until terminal status; return the response."""
    last_status = None
    while True:
        resp = sm.describe_training_job(TrainingJobName=job_name)
        status = resp["TrainingJobStatus"]
        sec_status = resp.get("SecondaryStatus", "")
        compound = f"{status}/{sec_status}"
        if compound != last_status:
            logger.info("[remote] %s status=%s", job_name, compound)
            last_status = compound
        if status in ("Completed", "Failed", "Stopped"):
            if status != "Completed":
                reason = (resp.get("FailureReason")
                          or resp.get("SecondaryStatusTransitions", [{}])[-1].get("StatusMessage")
                          or "(no reason)")
                raise RuntimeError(
                    f"Training job {job_name} ended {status}: {reason}"
                )
            return resp
        time.sleep(poll_seconds)


def _download_and_extract_model(s3, model_uri: str, dest_dir: Path) -> Path:
    """Download model.tar.gz from S3 and extract under ``dest_dir``.

    Returns the path of the extracted run directory (the trainer writes
    everything under run_<timestamp>/, so the tar contains exactly that).
    """
    assert model_uri.startswith("s3://")
    bucket, key = model_uri[len("s3://"):].split("/", 1)
    dest_dir.mkdir(parents=True, exist_ok=True)
    tar_path = dest_dir / "model.tar.gz"
    logger.info("downloading %s → %s", model_uri, tar_path)
    s3.download_file(bucket, key, str(tar_path))

    logger.info("extracting %s → %s", tar_path, dest_dir)
    with tarfile.open(tar_path, "r:gz") as tf:
        tf.extractall(dest_dir)
    tar_path.unlink()
    runs = sorted(dest_dir.glob("run_*"))
    if not runs:
        # The trainer might have written directly to MODEL_DIR root
        # (some setups skip the run_<timestamp> wrapper).
        return dest_dir
    return runs[-1]


def _run_remote(kind: str, args: argparse.Namespace, trainer_argv: list[str]) -> int:
    """Submit a SageMaker TrainingJob, wait for completion, fetch model."""
    try:
        import boto3
        import yaml
    except ImportError as exc:
        raise RuntimeError(
            "boto3 + pyyaml required for --remote; install via "
            "`pip install boto3 pyyaml`"
        ) from exc

    if not args.config:
        raise RuntimeError("--remote requires --config <pipeline yaml with sagemaker block>")
    with open(args.config) as f:
        cfg = yaml.safe_load(f) or {}
    sm_cfg = cfg.get("sagemaker") or {}
    required = ("region", "role_arn", "image_uri", "s3_bucket")
    missing = [k for k in required if not sm_cfg.get(k)]
    if missing:
        raise RuntimeError(
            f"sagemaker config missing required keys: {missing}. "
            f"Add them to {args.config} under the 'sagemaker:' block."
        )

    region = sm_cfg["region"]
    s3 = boto3.client("s3", region_name=region)
    sm = boto3.client("sagemaker", region_name=region)

    # ----- 1. Stage inputs -----------------------------------------
    run_id = f"finetune-{kind}-{datetime.now().strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:6]}"
    bucket = sm_cfg["s3_bucket"]
    s3_prefix = sm_cfg.get("s3_prefix", "pipeline-runs")
    base_key = f"{s3_prefix}/training/{run_id}"

    annot_dir = Path(args.annotations_dir).resolve()
    if not annot_dir.is_dir():
        raise RuntimeError(f"--annotations_dir not found: {annot_dir}")
    _upload_dir_to_s3(s3, annot_dir, bucket, f"{base_key}/annotations")

    pretrained_path = Path(args.pretrained).expanduser().resolve()
    if not pretrained_path.is_file():
        raise RuntimeError(f"--pretrained not found: {pretrained_path}")
    pretrained_key = f"{base_key}/weights/{pretrained_path.name}"
    s3.upload_file(str(pretrained_path), bucket, pretrained_key)
    logger.info("uploaded pretrained → s3://%s/%s", bucket, pretrained_key)

    # ----- 2. Build TrainingJob spec -------------------------------
    instance_type = sm_cfg.get("training_instance_type")
    if not instance_type:
        # Reuse the inference instance type for training only when the user
        # didn't set training_instance_type. Inference profiles are usually
        # smaller (g5.xlarge) than what's optimal for training.
        instance_type = sm_cfg.get("instance_type", "ml.g5.xlarge")
    volume_size = int(sm_cfg.get("training_volume_size_gb",
                                 sm_cfg.get("volume_size_gb", 50)))
    max_runtime = int(sm_cfg.get("training_max_runtime_seconds",
                                 sm_cfg.get("max_runtime_seconds", 3600 * 6)))

    hp = _trainer_argv_to_hparams(trainer_argv)
    hp["kind"] = kind

    training_job = {
        "TrainingJobName": run_id,
        "AlgorithmSpecification": {
            "TrainingImage": sm_cfg["image_uri"],
            "TrainingInputMode": "File",
            # Override the image's default entrypoint to run our training
            # entrypoint instead of run_stage.py.
            "ContainerEntrypoint": [
                "python", "/opt/goalinsight/entrypoints/train_entrypoint.py",
            ],
        },
        "RoleArn": sm_cfg["role_arn"],
        "InputDataConfig": [
            {
                "ChannelName": "annotations",
                "DataSource": {
                    "S3DataSource": {
                        "S3DataType": "S3Prefix",
                        "S3Uri": f"s3://{bucket}/{base_key}/annotations/",
                        "S3DataDistributionType": "FullyReplicated",
                    },
                },
                "InputMode": "File",
            },
            {
                "ChannelName": "weights",
                "DataSource": {
                    "S3DataSource": {
                        "S3DataType": "S3Prefix",
                        "S3Uri": f"s3://{bucket}/{base_key}/weights/",
                        "S3DataDistributionType": "FullyReplicated",
                    },
                },
                "InputMode": "File",
            },
        ],
        "OutputDataConfig": {
            "S3OutputPath": f"s3://{bucket}/{base_key}/output/",
        },
        "ResourceConfig": {
            "InstanceType": instance_type,
            "InstanceCount": 1,
            "VolumeSizeInGB": volume_size,
        },
        "StoppingCondition": {"MaxRuntimeInSeconds": max_runtime},
        "HyperParameters": hp,
    }

    logger.info("submitting TrainingJob %s (%s, %s)",
                run_id, instance_type, sm_cfg["image_uri"])
    sm.create_training_job(**training_job)

    # ----- 3. Wait + fetch -----------------------------------------
    resp = _wait_for_training_job(sm, run_id)
    model_uri = resp["ModelArtifacts"]["S3ModelArtifacts"]
    logger.info("training complete: %s", model_uri)

    out_dir = Path(args.output_dir).resolve()
    extracted = _download_and_extract_model(s3, model_uri, out_dir)
    logger.info("model extracted at %s", extracted)
    return 0


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------

def _split_argv(argv: list[str]) -> tuple[argparse.Namespace, list[str]]:
    """Split known wrapper flags from forwarded trainer flags.

    Wrapper flags (--kind, --remote, --config) are consumed here.
    Everything else is forwarded verbatim to the underlying trainer's
    own argparse, so we don't need to maintain a duplicate flag spec.
    Trainer-side flags we do read here (--annotations_dir,
    --pretrained, --output_dir) are kept in the forwarded list so
    local mode still works the same.
    """
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--kind", required=True, choices=["keypoint", "line"],
                        help="Which head to fine-tune.")
    parser.add_argument("--remote", action="store_true",
                        help="Submit to SageMaker TrainingJob instead of local.")
    parser.add_argument("--config", default=None,
                        help="Pipeline YAML with a 'sagemaker:' block (--remote only).")
    # We also need direct access to a few trainer flags so we can stage
    # them to S3 in --remote mode. They stay in the forwarded list.
    parser.add_argument("--annotations_dir", required=True)
    parser.add_argument("--pretrained", required=True)
    parser.add_argument("--output_dir", required=True)

    args, forwarded = parser.parse_known_args(argv)

    # Re-add the three trainer-shared flags so local main() sees them.
    forwarded = (
        ["--annotations_dir", args.annotations_dir,
         "--pretrained", args.pretrained,
         "--output_dir", args.output_dir]
        + list(forwarded)
    )
    return args, forwarded


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    args, trainer_argv = _split_argv(sys.argv[1:])
    if args.remote:
        return _run_remote(args.kind, args, trainer_argv)
    return _run_local(args.kind, trainer_argv)


if __name__ == "__main__":
    sys.exit(main())
