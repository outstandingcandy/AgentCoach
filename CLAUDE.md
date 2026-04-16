# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

GoalInsight is a soccer video analysis pipeline: field registration (camera calibration), player tracking/identification, ball tracking with 3D trajectory estimation, event detection, highlight generation, and post-processing refinement.

## Environment & Running

```bash
source venv/bin/activate  # NOT .venv
pip install -e .           # editable install via pyproject.toml

# CLI entry point (installed via pyproject.toml → goalinsight.cli:main)
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

CLI flags: `--video`, `--output`, `--config`, `--keypoint-model`, `--stages` (comma-separated), `--skip-existing`, `--no-timestamp`, `--run-name`.

Python 3.12, dependencies in `requirements.txt`, packaging in `pyproject.toml`.

## Architecture

### Pipeline Framework

The pipeline is config-driven via `goalinsight/pipeline/`. Stages register themselves with `@register_stage` decorator into `STAGE_REGISTRY` and execute in order:

```yaml
# configs/default.yaml
pipeline:
  stages:
    - field_registration
    - tracking
    - post_processing
```

Available stages: `shot_detection`, `field_registration`, `tracking`, `post_processing`, `event_detection`, `highlights`, `video_enhancement`.

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
event_detection (events/)
  Input: tracking/ (ball_tracks.json, tracks.json, team_assignments.json) + field_registration/camera_poses.json
  Output: events.json (all events), goals.json (backward compat)
  Detectors: possession → pass, shot, carry, defensive (dependency-ordered)
    ↓
highlights (highlights/)
  Input: events.json + tracking/ + video
  Output: highlight clip MP4s per recipe (e.g. goal_highlight_0001.mp4)
    ↓
video_enhancement (video_enhancement/)
  Input: highlight clip MP4s
  Output: upscaled / frame-interpolated MP4s (via video2x CLI)
```

### Field Registration: Calibration Backends

Selected via `field_registration.backend` config. Runner files in `field_registration/`:

- **PnLCalib** (default, `pnlcalib_runner.py`): Iterative PnP with multi-candidate sweep, LM optimization, full 5-param distortion. Uses HRNet for keypoint/line detection.
- **BroadTrack** (`broadtrack_runner.py`): 9-parameter camera model with Cauchy robust loss and arc-length line constraints.
- **NBJW** (`pnlcalib_runner.py`): Alternative calibration backend (`field_registration/nbjw/`).
- **Physical** (`physical_runner.py`): Fixed camera intrinsics from `camera_profiles.yaml`. 7-DOF bounded optimization, 2-pass pipeline.
- **Homography** (`homography_runner.py`): Direct ground-plane homography via DLT.

### Tracking: Multi-threaded Pipeline

`tracking/orchestrator.py` runs a threaded I/O pipeline: frame prefetch → YOLOv8 inference → tracking/ReID/team classification → output writing. Ball processing runs via `tracking/ball_pipeline.py`.

### Ball Tracking and 3D Trajectory

- **Ball detector** (`tracking/ball_detector.py`): YOLO class 32 (sports ball), supports SAHI sliced inference, size/pitch filtering.
- **Ball tracker** (`tracking/ball_tracker.py`): ByteTrack/BOTSORT with center-distance matching (better than IoU for tiny bboxes).
- **3D trajectory** (`tracking/ball_trajectory.py`): Two-pass batch architecture per tracker track:
  - Pass 1: Segment at kick boundaries (sudden pixel acceleration).
  - Pass 2: Classify each segment as ground-roll vs airborne (via ground-plane speed and out-of-bounds analysis), then fit per-segment. Ground segments use Z=0 projection; airborne segments fit `P(t) = [x0+vx·dt, y0+vy·dt, z0+vz·dt-0.5·g·dt²]` with ground-contact anchors at segment boundaries and dynamic velocity bounds.
- **Goal detection** (`goal_detection.py`): DEPRECATED — delegates to `events` module. Kept for backward compatibility.

### Event Detection (`events/`)

Config-driven event detection framework. Detectors run in dependency order (possession first).

- **Possession** (`detectors/possession.py`): Foundation state machine. Tracks ball-player proximity over consecutive frames.
- **Pass** (`detectors/pass_detector.py`): Detects passes from possession transitions with ball speed jumps. Classifies successful/failed.
- **Shot** (`detectors/shot.py`): Detects shots on goal via ball speed + trajectory toward goal. Subsumes goal detection. Outcome: Goal, Saved, Off_Target, Blocked. Shooter attribution via kick-frame detection (ball speed spike) + nearest non-goalkeeper proximity. Emits both SHOT and GOAL events with `player_id`, `team_id`, `shooter_frame` in metadata.
- **Carry** (`detectors/carry.py`): Detects dribbles with forward progress during sustained possession.
- **Defensive** (`detectors/defensive.py`): Detects tackles (possession change + deflection) and interceptions (failed pass + possession gain).

Key classes:
- `EventOrchestrator`: topo-sorts detectors by `depends_on`, runs them in order
- `EventDetectionContext`: shared state passed to all detectors (ball states, possession spans)
- `MatchEvent`: universal event dataclass with `event_type`, `frame`, `player_id`, `team_id`, `metadata`
- `DETECTOR_REGISTRY`: maps detector names to classes (like pipeline's STAGE_REGISTRY)

Config: `events.detectors` lists enabled detectors; per-detector thresholds under `events.<name>`.

### Highlights System (`highlights/`)

Recipe-based agent pipeline: **Detector → Analyzer → Composer**. Recipes are defined in config under `highlights.recipes`.

- `MatchContext` (`_context.py`): Loads all pipeline output (video, tracks, ball data, calibration, events) into a queryable object.
- `HighlightOrchestrator` (`_orchestrator.py`): Reads recipes, chains the three agents per recipe.
- Agents in `highlights/agents/`:
  - `GoalEventDetector`: Reads events.json, filters to GOAL events. Requires `event_detection` stage to have run first.
  - `ScorerAnalyzer`: Uses `player_id`/`team_id` from event metadata (no duplicate attribution logic). Produces 4-segment plan: buildup (wide, follows ball) → strike (closeup, ball through shot) → celebration (medium, follows scorer, truncates when track lost) → replay (slow-motion of strike with RIFE interpolation).
  - `SegmentComposer`: Renders video with per-segment crop/zoom (closeup, medium, wide, ball-follow). Visual effects: scorer spotlight ellipse, ball trajectory trail. Holds last known crop when track is lost (no jarring wide-view jump). When `video_enhancement` is enabled in config, upscales source frames via video2x *before* composition so cropping/effects/slow-motion all operate on high-res frames. Replay slow-motion uses RIFE optical-flow interpolation (falls back to linear blending if video2x unavailable).
- `_closeup.py`: `extract_closeup()` crops and scales frames; `interpolate_bbox()` interpolates/extrapolates with a max-distance limit.
- `_temporal.py`: `find_buildup_start()`, `find_celebration_end()` compute temporal windows.

### Video Enhancement (`video_enhancement/`)

Upscaling and frame interpolation using [video2x](https://github.com/k4yt3x/video2x) (Vulkan GPU). Two integration points:

1. **Inline (preferred)**: When `video_enhancement.enabled` is set, the highlights `SegmentComposer` upscales source frames *before* composition. The pipeline adapter injects the top-level `video_enhancement` config into highlights config. This produces higher quality because cropping/effects/slow-motion operate on upscaled frames.
2. **Post-hoc stage**: The `video_enhancement` pipeline stage processes finished highlight clips. Useful for standalone upscaling or frame interpolation of any MP4.

- **Modes**: `binary` (local video2x) or `docker` (container with GPU passthrough, needed on older glibc systems).
- **Upscaling**: Real-ESRGAN, Real-CUGAN, or libplacebo (Anime4K shaders). 2x or 4x scale.
- **Frame interpolation**: RIFE. 2x+ frame rate multiplier. Also used for replay slow-motion inside `SegmentComposer`.
- **Two-pass**: If both upscale and interpolation are enabled, upscale runs first, then interpolation on the upscaled output.

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
- `events.detectors`: list of enabled detectors (`possession`, `pass`, `shot`, `carry`, `defensive`)
- `events.<detector>.*`: per-detector thresholds (distance, speed, angle, etc.)
- `highlights.recipes`: list of highlight recipes (each with detector, analyzer, composer)
- `highlights.temporal.*`: buildup/celebration durations, view types
- `highlights.effects.*`: shooter spotlight, ball trail settings
- `video_enhancement.enabled`: toggle video2x processing (requires Vulkan GPU)
- `video_enhancement.upscale.*`: upscaling processor, scale factor, model
- `video_enhancement.interpolate.*`: frame interpolation processor, multiplier
- `video_enhancement.encoder.*`: output codec and quality settings

## Tools

- `tools/make_comparison.py`: Creates picture-in-picture comparison video (enhanced full-screen + raw source as PiP in bottom-right). Reads highlight metadata JSON to reconstruct segment timing. Usage: `python tools/make_comparison.py --enhanced <highlight.mp4> --raw <source.mp4> --output <out.mp4> --label`

## Testing

Ad-hoc test scripts at repo root (`test_*.py`, `debug_*.py`). No formal test suite. Run individual scripts directly:
```bash
python test_calibration.py
```
