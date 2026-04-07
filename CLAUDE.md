# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

GoalInsight is a soccer video analysis pipeline: field registration (camera calibration), player tracking/identification, ball tracking with 3D trajectory estimation, goal detection, and post-processing refinement.

## Environment & Running

```bash
source venv/bin/activate  # NOT .venv
pip install -e .           # editable install via pyproject.toml

# CLI entry point (installed via pyproject.toml)
goalinsight \
  --video data/raw_videos/football_sunday_output_000.mp4 \
  --output output/ \
  --config configs/clip_000_finetuned.yaml \
  --stages field_registration,tracking,post_processing

# Or via script
python scripts/run_full_pipeline.py [same args]

# Quick run scripts
bash scripts/pipeline.sh            # PnLCalib (finetuned)
bash scripts/pipeline_broadtrack.sh # BroadTrack backend
bash scripts/pipeline_physical.sh   # Physical calibration
```

Python 3.12, dependencies in `requirements.txt`, packaging in `pyproject.toml`.

## Architecture

### Pipeline Framework

The pipeline is config-driven via `goalinsight/pipeline/`. Stages are registered by name and executed in order:

```yaml
# configs/default.yaml
pipeline:
  stages:
    - field_registration
    - tracking
    - post_processing
```

Available stages: `shot_detection`, `field_registration`, `tracking`, `post_processing`, `goal_detection`.

```python
from goalinsight import Pipeline
pipeline = Pipeline(config)
pipeline.run(video_path, output_dir)
```

Key classes in `goalinsight/pipeline/`:
- `Stage` (ABC): base class with `run(ctx)` method
- `PipelineContext`: carries video path, config, stage output dirs, stats
- `Pipeline`: reads config, builds stage list, executes stages in order
- `STAGE_REGISTRY`: maps stage names to Stage subclasses
- Adapters in `_adapters.py`: bridge Stage interface to business modules

### Data Flow

```
shot_detection (preprocessing/runner.py)
  Output: shot_boundaries.json, segments/

field_registration (field_registration/*_runner.py)
  Output: homographies.pkl, camera_poses.pkl/.json, calibration_metadata.json
    ↓
tracking (tracking/orchestrator.py)
  Input: Video + field_registration output (homographies.pkl, camera_poses)
  Output: tracks.json, ball_tracks.json, track_features.json, team_assignments.json, tracking.mp4
    ↓
post_processing (refinement.py)
  Input: tracking/tracks.json + track_features.json
  Output: tracks_refined.json, final_track_summaries.json, statistics.json
    ↓
goal_detection (goal_detection.py)
  Input: tracking/ball_tracks.json + field_registration/camera_poses.json
  Output: goals.json
```

### Field Registration: Calibration Backends

Selected via `field_registration.backend` config. Runner files in `field_registration/`:

- **PnLCalib** (default, `pnlcalib_runner.py`): Iterative PnP with multi-candidate sweep, LM optimization, full 5-param distortion. Uses HRNet for keypoint/line detection.
- **BroadTrack** (`broadtrack_runner.py`): 9-parameter camera model with Cauchy robust loss and arc-length line constraints.
- **NBJW** (`pnlcalib_runner.py`): Alternative calibration backend (`field_registration/nbjw/`).
- **Physical** (`physical_runner.py`): Fixed camera intrinsics from `camera_profiles.yaml`.
- **Homography** (`homography_runner.py`): Direct ground-plane homography via DLT.

### Tracking: Multi-threaded Pipeline

`tracking/orchestrator.py` runs a threaded I/O pipeline: frame prefetch → YOLOv8 inference → tracking/ReID/team classification → output writing. Ball processing runs via `tracking/ball_pipeline.py`: collects detections, fits 3D parabolic trajectories (physics-based optimizer using `scipy.optimize.least_squares`), outputs pitch-projected positions.

### Factory Pattern and Interfaces

`goalinsight/utils/factories.py` provides factory functions that instantiate backends based on config (also re-exported from `utils/config.py` for backwards compatibility):
- `get_calibrator(config)` → `BaseCalibrator`
- `get_reid_extractor(config)` → `BaseReIDExtractor`
- `get_jersey_recognizer(config)` → `BaseJerseyRecognizer`
- `get_team_classifier(config)` → `BaseTeamClassifier`
- `get_visualizer(config)` → `BaseVisualizer`
- `get_side_labeler(config)` → `BaseTeamSideLabeler`

Abstract base classes live in `goalinsight/interfaces/`.

### Keypoint Format

Input uses 115 SoccerNet-GSR keypoints; internal processing uses 57 PnLCalib keypoints. `KeypointMapper` in `keypoint_mapping.py` converts between formats. 4 non-ground crossbar points (IDs {12,14,16,18}) are at z=-2.44m.

### Ball Tracking and Goal Detection

- **Ball detector** (`tracking/ball_detector.py`): YOLO class 32 (sports ball), supports SAHI sliced inference, size/pitch filtering.
- **Ball tracker** (`tracking/ball_tracker.py`): ByteTrack/BOTSORT with center-distance matching (better than IoU for tiny bboxes).
- **3D trajectory** (`tracking/ball_trajectory.py`): Parabolic physics model `P(t) = [x0+vx*dt, y0+vy*dt, z0+vz*dt-0.5*g*dt^2]`, sliding window fitting. Depth via ball pixel diameter (FIFA 0.22m) or ray-z search fallback.
- **Goal detection** (`goal_detection.py`): Detects goal-line crossings at x=+-52.5m, validates within goal width (|y|<=3.66m) and height (z<=2.44m), deduplicates within 2s.

## Configuration

YAML configs in `configs/` override `configs/default.yaml` via deep merge (`merge_configs`). Key settings:
- `pipeline.stages`: list of stages to run
- `field_registration.backend`: `pnlcalib` | `broadtrack` | `physical` | `nbjw` | `homography`
- `field_registration.keypoint_threshold`, `ransac_threshold` (default 30px)
- `video.process_fps`: frame sampling rate
- `detection.model`: YOLOv8 variant (yolov8n/s/m/l/x, yolo11*)
- `ball_detection.*`: ball detector config
- `reid.backend`: `osnet` | `prtreid`
- `team_classification.backend`: `kmeans` | `tracklet`
- `jersey_recognition.backend`: `qwen` | `mmocr`

## Testing

Ad-hoc test scripts at repo root (`test_*.py`, `debug_*.py`). No formal test suite. Run individual scripts directly:
```bash
python test_calibration.py
```
