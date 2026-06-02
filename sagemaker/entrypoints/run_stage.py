"""Entrypoint for the SageMaker Processing Job.

Lays out the on-disk view that the existing stage code expects, then
calls the same in-tree run function used by local execution. There is
no SageMaker-specific logic in field_registration / tracking — only
this glue file knows about /opt/ml/processing/* paths.

I/O contract (SageMaker Processing Job):

    /opt/ml/processing/input/video/<video.mp4>       ← uploaded by client
    /opt/ml/processing/input/config/config.json      ← merged config
    /opt/ml/processing/input/calibration/<files>     ← (tracking only)
    /opt/ml/processing/input/weights/<files>         ← all model weights

    /opt/ml/processing/output/<stage>/...            ← stage products,
                                                       auto-synced to S3

Usage:
    python run_stage.py --stage field_registration
    python run_stage.py --stage tracking

Both stages read the same env layout. The client-side code wiring up
ProcessingJob inputs/outputs (see goalinsight/pipeline/_remote.py) is
the source of truth for what lands where.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import sys
from pathlib import Path

logger = logging.getLogger(__name__)

INPUT_BASE = Path(os.environ.get("SM_INPUT_BASE", "/opt/ml/processing/input"))
OUTPUT_BASE = Path(os.environ.get("SM_OUTPUT_BASE", "/opt/ml/processing/output"))


def _find_video(input_dir: Path) -> Path:
    """Return the single video file in /opt/ml/processing/input/video/.

    The client uploads exactly one mp4/mov per job, so we just pick the
    first match. Fail loudly if there are zero or multiple — the
    Processing Job should never run without a clear input.
    """
    video_dir = input_dir / "video"
    candidates = sorted(
        p for p in video_dir.iterdir()
        if p.suffix.lower() in (".mp4", ".mov", ".mkv", ".avi")
    )
    if not candidates:
        raise RuntimeError(f"no video file found under {video_dir}")
    if len(candidates) > 1:
        raise RuntimeError(
            f"expected one video, found {len(candidates)} under {video_dir}: "
            f"{[p.name for p in candidates]}"
        )
    return candidates[0]


def _load_config(input_dir: Path) -> dict:
    cfg_path = input_dir / "config" / "config.json"
    if not cfg_path.exists():
        raise RuntimeError(f"config.json missing at {cfg_path}")
    with open(cfg_path) as f:
        return json.load(f)


def _wire_weight_paths(config: dict, input_dir: Path) -> dict:
    """Rewrite local-relative weight paths in the config to point at the
    locations where the client uploaded them.

    Goal: keep the stage code completely unaware of where it's running
    by remapping config keys *before* the runner reads them. Only paths
    explicitly listed in the upload script are mapped — anything else
    is passed through and resolved at runtime against /var/task or the
    image itself.
    """
    weights = input_dir / "weights"

    # PnLCalib finetune model paths
    fr = config.setdefault("field_registration", {})
    pnl = fr.setdefault("pnlcalib", {})
    if (weights / "pnlcalib" / "keypoint_best_model.pt").exists():
        pnl["keypoint_model_path"] = str(weights / "pnlcalib" / "keypoint_best_model.pt")
    if (weights / "pnlcalib" / "line_best_model.pt").exists():
        pnl["line_model_path"] = str(weights / "pnlcalib" / "line_best_model.pt")

    # YOLOv8x is referenced by name in detection.model. The detector
    # falls back to ultralytics' default location; if we shipped a
    # weight up, prefer that.
    yolo_path = weights / "yolov8x.pt"
    if yolo_path.exists():
        det = config.setdefault("detection", {})
        det["model_path"] = str(yolo_path)

    return config


def _stage_output_dir(stage: str) -> Path:
    """Output directory inside the container; SageMaker syncs to S3 on exit.

    SageMaker maps each declared ProcessingOutput's source to one S3 key,
    so we use a per-stage subdir to keep field_registration and tracking
    artifacts clearly separated even when both jobs upload to the same
    pipeline-run prefix.
    """
    out = OUTPUT_BASE / stage
    out.mkdir(parents=True, exist_ok=True)
    return out


def run_field_registration(input_dir: Path, video: Path, config: dict) -> None:
    out = _stage_output_dir("field_registration")
    vis_dir = out / "visualizations"
    vis_dir.mkdir(exist_ok=True)

    from goalinsight.utils.config import get_process_fps_from_config

    process_fps = get_process_fps_from_config(config)
    backend = config.get("field_registration", {}).get("backend", "pnlcalib")
    logger.info("field_registration backend=%s, process_fps=%s", backend, process_fps)

    if backend == "nbjw":
        from goalinsight.field_registration.pnlcalib_runner import _run_stage1_nbjw as runner
    elif backend == "broadtrack":
        from goalinsight.field_registration.broadtrack_runner import run_stage1_broadtrack as runner
    elif backend == "physical":
        from goalinsight.field_registration.physical_runner import run_stage1_physical as runner
    elif backend == "homography":
        from goalinsight.field_registration.homography_runner import run_stage1_homography as runner
    else:
        from goalinsight.field_registration.pnlcalib_runner import _run_stage1_pnlcalib as runner

    runner(video, out, vis_dir, config, process_fps)


def run_tracking_stage(input_dir: Path, video: Path, config: dict) -> None:
    out = _stage_output_dir("tracking")

    # The client uploads field_registration products to .../calibration/.
    # run_tracking expects a directory containing homographies.pkl /
    # camera_poses.pkl, which is exactly what we receive — no munging
    # required.
    cal_dir = input_dir / "calibration"
    if not cal_dir.is_dir() or not any(cal_dir.iterdir()):
        raise RuntimeError(
            f"tracking stage requires field_registration outputs at {cal_dir}; "
            "client must declare a ProcessingInput pointing at the "
            "field_registration output S3 prefix."
        )

    from goalinsight.tracking.orchestrator import run_tracking
    run_tracking(video, out, cal_dir, config)


STAGE_RUNNERS = {
    "field_registration": run_field_registration,
    "tracking": run_tracking_stage,
}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", required=True, choices=list(STAGE_RUNNERS))
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    logger.info("Starting stage %s", args.stage)
    logger.info("Input base: %s, Output base: %s", INPUT_BASE, OUTPUT_BASE)

    video = _find_video(INPUT_BASE)
    logger.info("Video: %s (%d bytes)", video, video.stat().st_size)

    config = _load_config(INPUT_BASE)
    config = _wire_weight_paths(config, INPUT_BASE)

    runner = STAGE_RUNNERS[args.stage]
    runner(INPUT_BASE, video, config)
    logger.info("Stage %s finished", args.stage)
    return 0


if __name__ == "__main__":
    sys.exit(main())
