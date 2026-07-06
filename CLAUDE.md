# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

GoalInsight is a soccer video analysis pipeline plus an LLM chat surface
on top of the resulting match data:

- **Field registration** (camera calibration, 6 backends).
- **Tracking** (player + ball detection, ReID, team classification).
- **Track consolidation** (fragmented track_ids → stable `A-9` / `B-10` player_ids via ReID + jersey OCR).
- **Event detection** (possession-driven state machine: pass / shot / carry / tackle / interception, with shot subsuming goal).
- **Player profile** (per-player front/back crops, heatmap, distance, optional follow-cam "spotlight" clips).
- **Highlights** (recipe-based per-event clips with crop/zoom + slow-motion replay).
- **Annotated video** (full-match render with HUD overlays).
- **Web app** (FastAPI viewer + Bedrock-backed chat with five tools, optionally proxied to AgentCore Runtime).

## Environment & Running

```bash
source .venv/bin/activate
pip install -e .           # editable install via pyproject.toml

# CLI entry point (installed via pyproject.toml → goalinsight.cli:main)
goalinsight \
  --video data/raw_videos/football_sunday_output_000.mp4 \
  --output output/ \
  --config configs/clip_000_finetuned.yaml \
  --stages field_registration,tracking

# Or via script
python scripts/run_full_pipeline.py [same args]

# Quick run scripts
bash scripts/pipeline.sh            # PnLCalib (finetuned)
bash scripts/pipeline_broadtrack.sh # BroadTrack backend
bash scripts/pipeline_physical.sh   # Physical calibration
bash scripts/pipeline_homography.sh # Plain DLT homography
```

CLI flags:
- `--video`, `--output`, `--config` (required)
- `--stages` (comma-separated subset)
- `--keypoint-model` (override `field_registration.keypoint_detection.keypoint_model_path`)
- `--remote-stages` (comma-separated; offload `field_registration[,tracking]` to SageMaker)
- `--skip-existing`, `--no-timestamp`, `--run-name`
- `--no-viz` (skip tracking visualization video)

Python 3.12, dependencies in `requirements.txt`, packaging in `pyproject.toml`.

## Architecture

### Pipeline Framework

The pipeline is config-driven via `goalinsight/pipeline/`. Stages register
themselves with `@register_stage` decorator into `STAGE_REGISTRY` and execute
in the order listed under `pipeline.stages`:

```yaml
# configs/default.yaml
pipeline:
  stages:
    - field_registration
    - tracking
```

Available stages (declared in `goalinsight/pipeline/_adapters.py`):
`field_registration`, `tracking`, `track_consolidation`, `event_detection`,
`player_profile`, `highlights`, `annotated_video`.

```python
from goalinsight import Pipeline
pipeline = Pipeline(config)
pipeline.run(video_path, output_dir)
```

Key classes in `goalinsight/pipeline/`:
- `Stage` (ABC, `_base.py`): base class with `run(ctx)` method
- `PipelineContext` (`_base.py`): carries video path, config, stage output dirs, stats
- `Pipeline` (`_pipeline.py`): reads config, builds stage list, executes stages in order
- `STAGE_REGISTRY` (`_registry.py`): maps stage names to Stage subclasses
- Adapters in `_adapters.py`: bridge Stage interface to business modules
- Remote execution in `_remote.py`: SageMaker submission/polling for `field_registration` and `tracking`

### Data Flow

```
field_registration (field_registration/*_runner.py)
  Output: homographies.pkl, camera_poses.pkl/.json, calibration_metadata.json
    ↓
tracking (tracking/orchestrator.py)
  Input: Video + field_registration/ (homographies.pkl, camera_poses)
  Output: tracks.json, ball_tracks.json, track_features.json, team_assignments.json, tracking.mp4
    ↓
track_consolidation (track_consolidation/_runner.py)
  Input: tracking/ (tracks.json + features) + jersey crops
  Output: players.json, player_map.json, tracks_consolidated.json,
          jersey/<frame>.json, consolidated.mp4
  Logic: ReID-first greedy clustering → jersey vote per cluster →
         team split → orphan absorption → naming (A-9, B-10, A-GK, ...)
    ↓
event_detection (events/_orchestrator.py)
  Input: tracking/ (ball_tracks.json) + track_consolidation/ (consolidated tracks
         with stable player_ids) + field_registration/camera_poses.json
  Output: events.json (all events), goals.json (backward compat)
  Detectors: possession → pass, shot, carry, defensive (dependency-ordered)
    ↓
player_profile (player_profile/_runner.py)
  Input: tracking/ + track_consolidation/players.json + (optional) video for spotlights
  Output: players_profile.json, crops/<pid>_front.jpg, crops/<pid>_back.jpg,
          heatmaps/<pid>.png, spotlights/<pid>.mp4 (when enabled)
    ↓
highlights (highlights/_orchestrator.py)
  Input: events.json + tracking/ + track_consolidation/ + video
  Output: highlight clip MP4s per recipe (e.g. goal_highlight_0001.mp4)
    ↓
annotated_video (annotated_video/_renderer.py)
  Input: tracking/ + track_consolidation/ + events.json + video
  Output: annotated.mp4 (full match with HUD: jersey numbers, team colors,
          ball trail, event banners)
```

The graph above is the canonical full-pipeline order. `track_consolidation`
**must** run before `event_detection` so events.json carries stable
player_id strings (`"A-7"`) instead of raw int track_ids — this is enforced
by the comment in `configs/default.yaml` next to the stages list.

### Field Registration: Calibration Backends

Selected via `field_registration.backend` config. Runner files in `field_registration/`:

- **PnLCalib** (default, `pnlcalib_runner.py`): Iterative PnP with multi-candidate sweep, LM optimization, full 5-param distortion. Uses HRNet for keypoint/line detection.
- **BroadTrack** (`broadtrack_runner.py`): 9-parameter camera model with Cauchy robust loss and arc-length line constraints.
- **NBJW** (`pnlcalib_runner.py`): Alternative calibration backend (`field_registration/nbjw/`).
- **Physical** (`physical_runner.py`): Fixed camera intrinsics from `camera_profiles.yaml`. 7-DOF bounded extrinsics, 2-pass pipeline. Most stable on non-FIFA pitches. Runs per-frame HRNet keypoint estimation.
- **Fixed camera** (`fixed_camera_runner.py`): Replays a single manually-annotated pose to every frame. Loads NO keypoint model — reads one `frame_*.json` annotation, solves PnP once. Cheapest backend; correct for a truly static rig.
- **Homography** (`homography_runner.py`): Direct ground-plane homography via DLT.

Backends are independent — no runtime delegation between them. The
per-frame-detector backends (PnLCalib, BroadTrack, Physical, Homography)
all read their HRNet detector settings from ONE shared config block,
`field_registration.keypoint_detection` (`keypoint_weights`,
`keypoint_model_path`, `keypoint_threshold`, `line_weights`,
`line_threshold`), via the helper in `field_registration/_detector_config.py`.
Backend-*solver* knobs stay in each backend's own block (e.g.
`field_registration.pnlcalib` holds only `pnl_refine` / `line_weight` /
`use_lines`; `field_registration.physical` holds intrinsics/extrinsics
bounds). A fine-tuned keypoint model is wired in via
`field_registration.keypoint_detection.keypoint_model_path`.

A finetune machinery for the PnLCalib HRNet keypoint and line heads lives
under `field_registration/pnlcalib/finetune_*.py`. Frame annotations come
from the Annotate tab in `goalinsight-web` (or directly under
`workspace/annotations/<video_stem>/`).

### Tracking: Multi-threaded Pipeline

`tracking/orchestrator.py` runs a threaded I/O pipeline: frame prefetch →
YOLOv8 inference → tracking/ReID/team classification → output writing.
Ball processing runs via `tracking/ball_pipeline.py`.

**Multi-object tracker backends** (selected via `tracking.backend`):
- **StrongSORT** (default, `tracking/strongsort/` package): Cascaded matching pipeline
  (tentative-IoU → tentative-pitch → confirmed-IoU → confirmed-ReID), pitch-distance
  gating in metres (not pixels), Kalman-coast handling, stationary-track killer for
  banner/fence false positives. Decomposed into `Gate` / `MatchingStage` / `TrackLifecycle`
  abstractions; covered by unit tests under `tests/tracking/`.
- **BoT-SORT** (`tracking/botsort_tracker.py`): GMC-aware alternative; ReID flows
  through the tracker's own embedding interface.

ReID extractors are pluggable via `reid.backend`:
- **OSNet** (default): 512-dim, `osnet_x1_0` backbone via TorchReID.
- **PRTReID**: 256-dim Part-based ReID (with an albumentations 2.x compat shim).

### Ball Tracking and 3D Trajectory

- **Ball detector** (`tracking/ball_detector.py`): YOLO class 32 (sports ball), supports SAHI sliced inference, size/pitch filtering.
- **Ball tracker** (`tracking/ball_tracker.py`): ByteTrack/BOTSORT with center-distance matching (better than IoU for tiny bboxes).
- **3D trajectory** (`tracking/ball_trajectory.py`): Two-pass batch architecture per tracker track:
  - Pass 1: Segment at kick boundaries (sudden pixel acceleration).
  - Pass 2: Classify each segment as ground-roll vs airborne (via ground-plane speed and out-of-bounds analysis), then fit per-segment. Ground segments use Z=0 projection; airborne segments fit `P(t) = [x0+vx·dt, y0+vy·dt, z0+vz·dt-0.5·g·dt²]` with ground-contact anchors at segment boundaries and dynamic velocity bounds.
- **Goal detection** (`goal_detection.py`): DEPRECATED — delegates to `events` module. Kept for backward compatibility.

### Track Consolidation (`track_consolidation/`)

Merges fragmented `track_id`s emitted by the tracker into stable
`player_id`s used everywhere downstream. Five-stage greedy pipeline in
`_consolidator.py`:

- **Stage A** — ReID-first greedy clustering (cosine ≥ same_person_threshold,
  no temporal co-occurrence).
- **Stage B** — Confidence-weighted jersey vote per cluster (image-list LLM
  call to fuse redundant high-conf jerseys and rescue low-conf misreads).
- **Stage C** — Team split: same-cluster cross-team tracks get separated.
- **Stage D** — Orphan absorption against existing clusters at a looser
  threshold.
- **Stage E** — Naming: `A-9`, `B-10`, `A-GK`, `B-unk-01`, etc.

Sampling for the LLM input is in `_sampler.py`; the samples-and-votes
JSON is written under `track_consolidation/jersey/`. The vis renderer
hooked into the stage paints consolidated boxes (referees green,
unmapped-but-on-field tracks blue) and writes `consolidated.mp4`. Gated
by `output.save_visualizations`.

### Event Detection (`events/`)

Config-driven event detection framework. Detectors run in dependency order (possession first).

- **Possession** (`detectors/possession.py`): Foundation state machine. Tracks ball-player proximity over consecutive frames.
- **Pass** (`detectors/pass_detector.py`): Detects passes from possession transitions with ball speed jumps. Classifies successful/failed; catches one-touch passes (no carry between).
- **Shot** (`detectors/shot.py`): Detects shots on goal via ball speed + trajectory toward goal. Subsumes goal detection. Outcome: Goal, Saved, Off_Target, Blocked. Shooter attribution via pre-kick possession (not nearest-at-kick) so a defender briefly closer at the moment the ball leaves the foot doesn't get charged. Emits both SHOT and GOAL events with `player_id`, `team_id`, `shooter_frame` in metadata.
- **Carry** (`detectors/carry.py`): Detects dribbles with forward progress during sustained possession.
- **Defensive** (`detectors/defensive.py`): Detects tackles (possession change + deflection) and interceptions (failed pass + possession gain).

Key classes:
- `EventOrchestrator`: topo-sorts detectors by `depends_on`, runs them in order
- `EventDetectionContext`: shared state passed to all detectors (ball states, possession spans)
- `MatchEvent`: universal event dataclass with `event_type`, `frame`, `player_id`, `team_id`, `metadata`
- `DETECTOR_REGISTRY`: maps detector names to classes (like pipeline's STAGE_REGISTRY)

Config: `events.detectors` lists enabled detectors; per-detector thresholds under `events.<name>`.

### Player Profile (`player_profile/`)

Per-player artifacts for the Insights / match-detail view.

- `build_player_profiles` in `_runner.py` walks the consolidated tracks,
  picks a front-on and a back-on crop per player from the available
  detections, computes a pitch heatmap (`heatmap_bins` configurable), and
  estimates total distance run.
- **Spotlights** (`_spotlight.py`, opt-in via `player_profile.spotlights.enabled`):
  per-player follow-cam MP4s. The player is centered, scaled to ~2/3 frame
  height, with optional team-coloured ellipse, name badge (`#9 player`),
  and a pitch-trail polyline. Cuts (rather than bridging) on detection
  gaps longer than `presence_gap_seconds`. Skips players with fewer than
  `min_observations` detections.
- Outputs land under `player_profile/`: `players_profile.json`,
  `crops/<pid>_front.jpg`, `crops/<pid>_back.jpg`, `heatmaps/<pid>.png`,
  `spotlights/<pid>.mp4` (when enabled).

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

### Annotated Video (`annotated_video/`)

Full-match HUD render. `_renderer.py` reads the consolidated tracks,
events.json, ball trajectory, and calibration, and writes
`annotated.mp4` with per-frame overlays: team-coloured player boxes
with jersey numbers, ball trail, projected pitch lines, and event
banners. Optional video2x upscaling integrates the same way as the
highlights composer.

### Jersey Recognition (`jersey/`)

Per-track jersey number reading, used by `track_consolidation` to label
clusters. Multiple backends, selected via `track_consolidation.jersey.backend`:

- **claude** (`claude_recognizer.py`) — Bedrock Claude with a multi-image
  prompt. Reads role/team in phase 1 and per-crop numbers in phase 2.
- **gemini** (`gemini_recognizer.py`) — same shape via the Gemini API.
- **qwen** (`qwen_vlm_recognizer.py` / `qwen_vllm_recognizer.py`) — local
  Qwen-VL via HuggingFace transformers or a vLLM server (start with
  `bash scripts/start_qwen_vllm.sh`).
- **rapidocr** (`rapidocr.py`) — fast lightweight OCR; `ocr_backend: rapidocr`
  switches the per-crop number reader away from the LLM while keeping the
  LLM for role/team.

### Video Enhancement (`video_enhancement/`)

Upscaling and frame interpolation using [video2x](https://github.com/k4yt3x/video2x) (Vulkan GPU). Used inline by the highlights `SegmentComposer` and the `annotated_video` renderer — when `video_enhancement.enabled` is set, upscales source frames *before* composition so cropping/effects/slow-motion operate on high-res frames. Replay slow-motion uses RIFE optical-flow interpolation.

- **Modes**: `binary` (local video2x) or `docker` (container with GPU passthrough, needed on older glibc systems).
- **Upscaling**: Real-ESRGAN, Real-CUGAN, or libplacebo (Anime4K shaders). 2x or 4x scale.
- **Frame interpolation**: RIFE. 2x+ frame rate multiplier.

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

## Web app (`goalinsight/web/`)

Single FastAPI app — `python -m goalinsight.web --workspace ./workspace`
(default `http://127.0.0.1:8000/`). Tabs:

- `/library` — videos under `<workspace>/videos/` and runs under `<workspace>/runs/<name>/`.
- `/pipeline` — launch / monitor pipeline jobs (driven by `jobs.py`).
- `/insights` — index of runs; `/insights/<run>` is the per-run video viewer + chat.
- `/match/<run>` — per-run match-detail page (events, player profiles, spotlight clips).
- `/tracking/<run>` — tracking diagnostics (raw YOLO dumps, consolidation viewer).
- `/annotate` — pitch keypoint annotator (Gradio under the hood) for the
  PnLCalib finetune loop.

The viewer streams the match video alongside a Bedrock-backed chat with
five tools (defined in `match_tools.py`):

- `list_events` — filter events by type/team/player/time window
- `get_player_stats` — per-player distance, top speed, touches, passes, shots, goals
- `get_team_stats` — possession share, pass success, shots, tackles, interceptions
- `get_frame_snapshot` — who's on screen and what's near the ball at a moment
- `run_python` — execute Python in an AgentCore Code Interpreter sandbox
  (driven by `code_sandbox.py`); plots come back as inline images

The chat path streams responses live as SSE tokens with markdown
rendering on the client. Message history per session is in
`_sessions.py`.

### Chat on AgentCore Runtime (opt-in)

When `GOALINSIGHT_AGENTCORE_RUNTIME_ARN` is set, the FastAPI app routes
chat turns through `chat_remote.py` → `bedrock-agentcore.InvokeAgentRuntime`
instead of running `ChatEngine` in-process. The runtime container is
built and deployed from `deploy/agentcore_runtime/` (ARM64 image, Python
3.12, exposes 8080, implements `/invocations` SSE + `/ping`). It pulls
the run's JSON output from S3 once per session and stays warm for the
MicroVM's lifetime. Unset the env var to revert to local chat. See
`deploy/agentcore_runtime/README.md` for setup.

### Web deployment (`deploy/`)

`deploy/alb-cognito.yaml` is a CloudFormation stack that fronts the
`goal-insight-web` service (port 8000) with an ALB and Cognito-based
authentication. `deploy/bootstrap.sh` is the EC2 user-data script that
installs the app on a fresh host.

## Configuration

YAML configs in `configs/` override `configs/default.yaml` via deep merge (`merge_configs`). After merging, the CLI runs `resolve_config(merged, video_w, video_h, video_fps)` (`goalinsight/utils/config_resolver.py`) once to fill in resolution/fps-coupled values from the source video — per-video YAMLs should describe physical facts (camera position, pitch dims, sensor identity, model paths) and let the resolver handle the rest. Key settings:

- `pipeline.stages`: list of stages to run
- `field_registration.backend`: `pnlcalib` | `broadtrack` | `physical` | `nbjw` | `homography` | `fixed_camera`
- `field_registration.keypoint_detection.*`: shared HRNet detector config (weights, `keypoint_model_path`, thresholds) read by all per-frame-detector backends
- `field_registration.keypoint_detection.keypoint_threshold`, `.ransac_threshold` (default 30px)
- `video.process_fps`: frame sampling rate (auto-defaults to `min(30, source_fps)` when unset; `video.tracking_fps` overrides for tracking)
- `sample.stride`: pipeline-wide vis-frame stride (per-stage `<stage>.vis_frame_stride` overrides)
- `detection.model`: YOLOv8 variant (yolov8n/s/m/l/x, yolo11*)
- `ball_detection.*`: ball detector config
- `tracking.backend`: `strongsort` (default) | `botsort`
- `tracking.reid_iou_min`, `reid_pitch_max_m`, `pitch_gate_m`, `stationary_window`, ...
- `tracking.dump_yolo_raw`: dump raw YOLO detections to `yolo_raw/` for offline inspection
- `reid.backend`: `osnet` | `prtreid`
- `team_classification.backend`: `kmeans` | `tracklet`
- `track_consolidation.jersey.backend`: `claude` | `gemini` | `qwen`
- `track_consolidation.jersey.ocr_backend`: `llm` (LLM does numbers too) | `rapidocr` (LLM does role/team only)
- `events.detectors`: list of enabled detectors (`possession`, `pass`, `shot`, `carry`, `defensive`)
- `events.<detector>.*`: per-detector thresholds (distance, speed, angle, etc.)
- `player_profile.heatmap_bins`, `player_profile.spotlights.*` (enabled, output_size, target_player_height_frac, presence_gap_seconds, ellipse, trail, name_badge, min_observations, ...)
- `highlights.recipes`: list of highlight recipes (each with detector, analyzer, composer)
- `highlights.temporal.*`: buildup/celebration durations, view types
- `highlights.effects.*`: shooter spotlight, ball trail settings
- `video_enhancement.enabled`, `mode` (binary/docker)
- `output.save_visualizations`: gates the per-stage vis renders (track_consolidation `consolidated.mp4`, raw YOLO JPGs, etc.)

Per-video pitch + camera intrinsics can be supplied via a sparse
`overrides.yaml` next to the video file (used by the kids/youth configs
and any non-FIFA pitch).

### Auto-derived from video metadata

The resolver fills these in based on the source video's `(W, H, fps)`. Legacy explicit keys (in parentheses) always win when present, so old configs keep working unchanged.

| Key | Derived from | Legacy override |
|---|---|---|
| `video.process_fps` | `min(30, source_fps)` when unset | (explicit value) |
| `field_registration.physical.focal_bounds` | `focal_hfov_deg_bounds: [minDeg, maxDeg]` → `f = W / (2*tan(deg/2))` | `focal_bounds: [px, px]` |
| `field_registration.physical.gap_fill_chain.anchor_max_reproj_px` | `anchor_max_reproj_frac` × image width | `anchor_max_reproj_px` |
| `field_registration.physical.gap_fill_chain.overwrite_above_reproj_px` | `overwrite_above_reproj_frac` × image width | `overwrite_above_reproj_px` |
| `unified_detection.imgsz` | `min(1920, max(W, H))` when unset | (explicit value) |
| `track_consolidation.min_track_frames` | `min_track_seconds` × effective_fps | `min_track_frames` |

`tracking.{max_age, pitch_gate_m, reid_pitch_max_m, stationary_window}` are auto-scaled by `effective_fps/10` inside the tracker (`tracking/orchestrator.py:299–333`); event detector and highlight recipe thresholds already accept seconds and convert at runtime. So at process_fps=30 the tracker computes `pitch_gate_m = 4.0 * 10/30 = 1.33` m on its own — no per-video YAML override needed.

`field_registration.physical.camera_profile` is **not** auto-picked: profile naming carries sensor identity (veo vs phone vs broadcast vs kids long-focal) beyond resolution. Pick the profile that matches your sensor; the runner already scales `fx/fy` if the source's resolution differs from the profile's `image_size` (see `_load_camera_profile` in `physical_runner.py:142–156`).

## Remote stage execution (SageMaker)

`field_registration` and `tracking` can run on a SageMaker Processing Job instead of locally. Default behaviour is unchanged (local). Opt in per-run via `--remote-stages field_registration[,tracking]`. Requires the `sagemaker:` block in config to be filled (region, role_arn, image_uri, s3_bucket). One-time setup is `bash sagemaker/setup_aws.sh && bash sagemaker/upload_weights.sh && bash sagemaker/build_and_push.sh`. Full docs in `sagemaker/README.md`.

## Tools

- `tools/make_comparison.py`: Creates picture-in-picture comparison video (enhanced full-screen + raw source as PiP in bottom-right). Reads highlight metadata JSON to reconstruct segment timing. Usage: `python tools/make_comparison.py --enhanced <highlight.mp4> --raw <source.mp4> --output <out.mp4> --label`

Useful diagnostic / one-off scripts under `scripts/`:
- `run_full_pipeline.py` — same flags as `goalinsight`, kept as a script.
- `run_highlights.py --run-dir output/<run>` — re-run only the highlights agent on an existing run.
- `render_consolidated_tracking.py` — re-render `consolidated.mp4` from existing JSON (also called from the pipeline adapter).
- `dump_raw_yolo.py`, `audit_track_dropouts.py` — tracking diagnostics.
- `diagnose_calibration.py`, `eval_finetune_on_train.py` — calibration sanity checks.
- `train_finetune.py`, `select_finetune_candidates.py` — PnLCalib finetune loop.

## Testing

`tests/tracking/` covers the StrongSORT package (gates, matching stages,
lifecycle). Run with `pytest tests/`.

Beyond that, ad-hoc test/debug scripts at the repo root (`test_*.py`,
`debug_*.py`) are gitignored — they're for one-off exploration, not a
formal suite.
