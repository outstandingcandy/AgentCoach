# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

GoalInsight is a multi-stage soccer video analysis pipeline: field registration (camera calibration), player tracking/identification, and post-processing refinement.

## Environment & Running

```bash
source venv/bin/activate  # NOT .venv

# Full pipeline
python scripts/run_full_pipeline.py \
  --video data/raw_videos/football_sunday_output_000.mp4 \
  --output output/ \
  --config configs/clip_000_finetuned.yaml \
  --stages 1,2,3

# Quick run scripts
bash scripts/pipeline.sh            # PnLCalib (finetuned)
bash scripts/pipeline_broadtrack.sh # BroadTrack backend
```

Python 3.12, dependencies in `requirements.txt`.

## Architecture

### Pipeline Stages

- **Stage 0** (`stage0.py`): Shot detection & video segmentation into continuous clips
- **Stage 1** (`stage1.py`, `stage1_broadtrack.py`): Field registration — detects soccer field keypoints via HRNet, solves camera pose via PnP/PnL optimization, outputs per-frame homographies
- **Stage 2** (`stage2.py`): YOLOv8 detection + StrongSORT tracking + ReID (OSNet) + team classification (KMeans) + optional jersey recognition (Qwen VL or MMOCR)
- **Stage 3** (`stage3.py`): SAM2 segmentation for missed detections, majority voting for temporal consistency, tracklet merging

### Two Calibration Backends (Stage 1)

**PnLCalib** (default, `field_registration/pnlcalib/frame_calibrator.py`):
- Iterative PnP with multi-candidate sweep: 6 focal lengths x 3 distortion priors
- Each candidate: up to 5 iterations of LM optimize (12 params) → undistort → re-RANSAC
- 57 keypoints including 4 non-ground crossbar points (IDs {12,14,16,18} at z=-2.44m)
- Full 5-param distortion model [k1, k2, p1, p2, k3]

**BroadTrack** (`field_registration/pnlcalib/broadtrack_calibrator.py`):
- 9-parameter camera model: angleAxis(3), position(3), f, k1, k2
- Cauchy robust loss with arc-length parameterized line constraints
- Multi-distortion-prior PnP initialization

### Key Module Layout

- `goalinsight/field_registration/pnlcalib/` — Core calibration: `frame_calibrator.py`, `broadtrack_calibrator.py`, `camera.py`, `keypoint_mapping.py` (57↔115 keypoint conversion), `curve_utils.py`
- `goalinsight/tracking/` — Detection (`detector.py`), tracking (`strongsort_tracker.py`), ball detection/tracking, ReID (`reid/`), team classification (`team/`)
- `goalinsight/jersey/` — Jersey number recognition backends
- `goalinsight/utils/config.py` — Config loading and factory functions (`get_calibrator`, `get_reid_extractor`, etc.)
- `configs/default.yaml` — All configuration options with defaults

### Keypoint Format

Input uses 115 SoccerNet-GSR keypoints; internal processing uses 57 PnLCalib keypoints. `KeypointMapper` in `keypoint_mapping.py` converts between formats.

## Configuration

YAML configs in `configs/` override `configs/default.yaml`. Key settings:
- `field_registration.backend`: `pnlcalib` or `broadtrack`
- `field_registration.keypoint_threshold`, `ransac_threshold` (default 30px)
- `video.process_fps`: frame sampling rate
- `detection.model_size`: YOLOv8 variant (n/s/m/l/x)

## Testing

Ad-hoc test scripts at repo root (`test_*.py`, `debug_*.py`). No formal test suite. Run individual scripts directly:
```bash
python test_calibration.py
```
