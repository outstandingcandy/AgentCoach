# Config notes

Auto-extracted from `configs/*.yaml` by `scripts/extract_config_comments.py`. Comments were moved here so the yaml files themselves stay short and diff-friendly; each section is keyed by the yaml file and the dotted key path the comment was originally attached to.

## camera_profiles.yaml
### _header_
> Camera intrinsic profiles for physical calibration backend.
> Each profile provides fixed K (intrinsic matrix) and dist_coeffs (distortion)
> that are NEVER modified by the optimizer — only 6-DOF extrinsics are optimized.
> 
> To add a new camera: measure or calibrate K and dist_coeffs offline,
> then add a named entry below.

### `profiles`

- Original profile from factory calibration grid

### `profiles.veo_1080p`

- Refined via cross-frame joint optimization on football_sunday_output_000

### `profiles.veo_4k`

- 4K (3840x2160) variant of veo_1080p. Veo's 4K stream is a 2x
- bilinear upsample of the same sensor crop, so fx/fy/cx/cy all
- double exactly. Distortion coefficients are unitless (operate in
- normalized image coordinates) so they're identical to the 1080p
- profile.

### `profiles.generic_4k`

- Generic 4K sideline profile for non-Veo cameras (phones, GoPros,
- consumer cams) where we don't have a lab-calibrated K. fx=6000 is
- the LM-converged value on sunday_cup_0050-0110 (~ 31° hfov) and
- works as a starting point for similar long-focal sideline rigs.
- Distortion zeroed — most phone footage is lens-corrected upstream.
- Pair with a wide focal_bounds and lock_camera_position=false so
- the LM can refine across rigs.

### `profiles.kids_soccer_1080p`

- Long-focal sideline camera used for kids_soccer_match.mp4 / clips.
- fx=2448 (hfov ≈ 43°) recovered jointly with pitch dims via
- scripts/fit_pitch_dims.py --fit-fx (combined RMSE 5.82 px on 53 residuals).
- See configs/kids_soccer.yaml header for context.

## clip_000_broadtrack.yaml
### _header_
> Configuration for clip_000 using BroadTrack-style calibration
> BroadTrack: 8-param camera + Cauchy loss + joint keypoint/line curve constraints
> Video Processing

### `video.process_fps`

- Stage 1: Field Registration with BroadTrack algorithm

### `field_registration.pnlcalib`

- Fine-tuned keypoint model

### `field_registration.pnlcalib.keypoint_threshold`

- Line detection (always enabled for BroadTrack)

### `field_registration.pnlcalib.line_threshold`

- RANSAC threshold for PnP initialization

### `field_registration.broadtrack.cauchy_f_scale`

- Cauchy loss scale (pixels)

### `field_registration.broadtrack.line_sample_points`

- Points sampled per detected line

### `field_registration.broadtrack.line_weight`

- Line residual weight vs keypoints

### `field_registration.broadtrack.max_nfev`

- Max optimizer iterations
- Stage 2: Tracking (same as original)

### `team_classification.position_weight`

- Hardware

## clip_000_finetuned.yaml
### _header_
> Configuration for clip_000 using fine-tuned PnLCalib model
> Fine-tuned on clip_000 annotations: Recall 43.8% -> 84.4%, error 40px -> 2px
> Video Processing

### `video.process_fps`

- Process 10 fps
- Stage 1: Field Registration with fine-tuned model

### `field_registration.pnlcalib`

- Fine-tuned keypoint model (trained on clip_000 annotations)

### `field_registration.pnlcalib.keypoint_weights`

- Base weights (ignored when model_path is set)

### `field_registration.pnlcalib.keypoint_threshold`

- Lower threshold for fine-tuned model (higher recall)
- Line detection (using standard weights)

### `field_registration.pnlcalib.line_threshold`

- Calibration method: "iterative_pnp" or "h_decompose"

### `field_registration.pnlcalib.calibration_method`

- PnL optimization

### `field_registration.pnlcalib.line_weight`

- Stage 2: Tracking

### `team_classification.position_weight`

- Hardware

## clip_000_homography.yaml
### _header_
> Configuration for clip_000 using homography backend
> Direct ground-plane homography estimation via DLT (no camera model)

## clip_000_physical.yaml
### _header_
> Configuration for physical calibration backend with Veo camera
> Uses fixed camera intrinsics and 6-DOF extrinsic optimization

### `video.process_fps`

- Stage 1 calibration fps

### `video.tracking_fps`

- Stage 2 tracking fps (interpolates camera poses)

### `field_registration.physical.focal_hfov_deg_bounds`

- 1080p → [400, 3500] px

### `field_registration.physical.calibration_skip`

- Calibrate every 2nd frame, interpolate the rest

### `field_registration.pnlcalib`

- keypoint_weights: "SV_kp"

### `field_registration.pnlcalib.keypoint_model_path`

- keypoint_model_path: "data/pretrain_models/SV_kp"

### `field_registration.pnlcalib.keypoint_threshold`

- keypoint_threshold: 0.15

### `ball_detection.confidence_threshold`

- Lower threshold; tracker Kalman filter handles false positives

## default.yaml
### _header_
> Video Processing

### `video`

- Set to null/omit to let the resolver default it to min(30, source_fps).
- Override here only when you want a slower/faster sample rate than that.

### `video.resize_width`

- Resize width (null for original)

### `video.resize_height`

- Resize height (null for original)
- Stage 1: Field Registration

### `field_registration.backend`

- "pnlcalib" (default), "broadtrack", "physical", or "nbjw"
- PnLCalib backend settings (used when backend="pnlcalib")

### `field_registration.pnlcalib.keypoint_weights`

- "SV_kp", "WC14_kp", or "TSWC_kp"

### `field_registration.pnlcalib.line_weights`

- "SV_lines", "WC14_lines", or "TSWC_lines"

### `field_registration.pnlcalib.pnl_refine`

- Enable joint point+line optimization

### `field_registration.pnlcalib.line_weight`

- Relative weight of line constraints vs points
- NBJW backend settings (used when backend="nbjw")

### `field_registration.nbjw.use_prev_homography`

- Use previous frame's H on failure

### `field_registration.nbjw.weights_dir`

- Path to weights (null = auto-download)
- Physical calibrator settings (used when backend="physical")
- Uses fixed camera intrinsics from a profile, optimizes only 6-DOF extrinsics

### `field_registration.physical.camera_profile`

- Profile name from camera_profiles.yaml

### `field_registration.physical.camera_profiles_path`

- Path to camera_profiles.yaml (null = auto)

### `field_registration.physical.ransac_reproj_error`

- PnP RANSAC reprojection threshold (px)

### `field_registration.physical.line_weight`

- Weight for line residuals vs point residuals

### `field_registration.physical.line_sample_points`

- Points sampled along each line

### `field_registration.physical.max_reproj_error`

- Temporal tracker error threshold

### `field_registration.physical.use_line_model`

- Use HRNet line model (vs derive from keypoints)
- Focal length optimization bounds are derived from horizontal field
- of view (degrees) at runtime: f_px = W / (2 * tan(hfov/2)). This
- keeps them resolution-independent. Override the px values directly
- via ``focal_bounds: [px_min, px_max]`` if you have hard data on the
- sensor's true focal range — the resolver leaves any explicit
- ``focal_bounds`` untouched.

### `field_registration.physical.focal_hfov_deg_bounds`

- ~telephoto sideline to wide handheld

### `field_registration.physical.cx_bounds`

- Optical center cx offset bounds (px)

### `field_registration.physical.cy_bounds`

- Optical center cy offset bounds (px)

### `field_registration.physical.k1_bounds`

- Radial distortion k1 absolute bounds

### `field_registration.physical.k2_bounds`

- Radial distortion k2 absolute bounds

### `field_registration.physical.k3_bounds`

- Radial distortion k3 absolute bounds

### `field_registration.physical.world_residual_weight`

- Weight for world-space back-projection residuals (0=disabled)

### `field_registration.physical.world_error_threshold`

- World-error outlier rejection threshold (meters). Use .inf to disable.

### `field_registration.physical.calibration_skip`

- Calibrate every Nth frame (1=all, 2=half, etc). Skipped frames are interpolated.
- Legacy settings for backwards compatibility

### `field_registration.keypoint_detector.backend`

- "resnet50" (default) or "pnlcalib"

### `field_registration.keypoint_detector.model`

- Backbone for resnet50 backend

### `field_registration.keypoint_detector.num_keypoints`

- SoccerNet-GSR format (115 for resnet50, 58 for pnlcalib)

### `field_registration.line_detector.backend`

- "hough" (default) or "pnlcalib"

### `field_registration.pnl_solver.line_weight`

- Unified detection: single YOLO pass for both players and ball

### `unified_detection.enabled`

- Set false to use legacy separate detectors

### `unified_detection.model`

- model_path: null

### `unified_detection.confidence_threshold`

- Use ball's lower threshold; player filtering is post-hoc

### `unified_detection.iou_threshold`

- ``imgsz: 0`` → use the source frame's native resolution rounded
- up to a multiple of 32 (YOLO stride). 1920 down-samples a 47-px
- ball to ~23 px and YOLO loses confidence on small / distant
- balls (sunday_highlights frame 2738 was the smoking gun: 0
- detections at 1920, conf=0.54 at 3840). Override with a positive
- int when detection latency matters more than recall on small
- balls — inference scales O(W²), so 4K is ~4× slower than 1920.

### `unified_detection.batch_size`

- 4K input + yolov8x: 4 fits on a 46GB GPU.
- Drop to 2 if you OOM on 24GB cards.

### `unified_detection.player_confidence_threshold`

- Post-hoc filter applied to player detections
- Stage 2: Tracking and Identification

### `detection.model`

- YOLO model name (yolov8x, yolo11x, etc.)
- model_path: null       # Optional: explicit path to .pt weights file

### `detection.classes`

- Person class only

### `detection.imgsz`

- Ball Detection (sports ball class 32 in COCO)

### `ball_detection.model`

- YOLO model name (yolov8x, yolo11x, etc.)
- model_path: null          # Optional: explicit path to .pt weights file

### `ball_detection.confidence_threshold`

- Lower threshold for small object

### `ball_detection.classes`

- Sports ball class
- ``imgsz: 0`` (or null / "native") tells the ball detector to use
- the source frame's native resolution, rounded up to the nearest
- multiple of 32 (YOLO stride). This is the default because at
- 1920 a 47-px native ball was being down-sampled to ~23 px and
- YOLO confidence dropped below threshold on small / distant /
- high-arc shots (sunday_highlights frame 2738 was the smoking gun
- — model returned conf=0.54 at 3840 but missed entirely at 1920).
- Override to a fixed int (e.g. ``imgsz: 1920``) when detection
- latency matters more than recall on small balls — inference cost
- scales roughly O(W²), so 4K is ~4× slower than 1920.

### `ball_detection.min_size`

- Minimum bbox dimension in pixels

### `ball_detection.max_size`

- Maximum bbox dimension in pixels

### `ball_detection.min_aspect_ratio`

- Ball should be roughly circular

### `ball_detection.max_aspect_ratio`

- Loosened from 2.0 → 3.0 — a hard kick blurs the
- ball into a w/h≈2.7 ellipse for ~3 frames
- before/after impact. Used to drop the actual
- shot frame at goal time. The conf-skip below
- is the wider safety net.
- High-confidence detections bypass the aspect filter entirely. A
- motion-blurred ball is non-circular by definition, but YOLO still
- locks on at conf ≥ 0.6 and the downstream pitch/speed filters guard
- against false positives anyway.

### `ball_detection.aspect_filter_conf_skip`

- SAHI sliced inference for improved small ball detection

### `ball_detection.use_sahi`

- SAHI is slower and produces more false positives; disable by default

### `ball_detection.sahi_slice_size`

- Slice dimensions (pixels)

### `ball_detection.sahi_overlap_ratio`

- Overlap between adjacent slices

### `ball_detection.sahi_perform_standard_pred`

- Also run full-frame inference (catches large balls)

### `ball_detection.sahi_postprocess_type`

- "NMS", "NMM", or "GREEDYNMM"

### `ball_detection.sahi_postprocess_match_threshold`

- Two-pass detection: scan whole video first, then crop+enlarge missed frames

### `ball_detection.two_pass`

- Enable two-pass ball detection

### `ball_detection.pass1_confidence_threshold`

- High threshold for anchor detections in pass 1

### `ball_detection.crop_size`

- Crop region size around interpolated position (pixels)

### `ball_detection.crop_enlarge_to`

- Resize crop to this before detection

### `ball_detection.max_interpolation_gap`

- Max frames to interpolate across (skip if gap is larger)

### `ball_detection.pass1_batch_size`

- Batch size for pass 1 GPU inference (full-res, needs more VRAM)

### `ball_detection.pass2_batch_size`

- Batch size for pass 2 GPU inference (640x640 crops, lighter)
- Field-space trajectory filtering (applied after tracking)

### `ball_detection.field_filter.max_speed_ms`

- Max realistic ball speed (m/s)

### `ball_detection.field_filter.position_margin`

- Meters beyond pitch boundary to allow

### `ball_detection.field_filter.min_trajectory_length`

- Minimum observations per trajectory

### `ball_detection.field_filter.smoothness_threshold`

- Min average cosine of direction changes

### `ball_detection.field_filter.speed_violation_ratio`

- Max fraction of frames exceeding speed limit

### `ball_detection.field_filter.min_displacement`

- Min total displacement (meters) between first and last point

### `ball_detection.field_filter.merge_max_distance`

- Max pitch distance (m) to nearest existing frame when merging trajectories
- Ball Tracking (ByteTrack / BoT-SORT)

### `ball_tracking.tracker_type`

- "bytetrack" or "botsort"

### `ball_tracking.track_high_thresh`

- High-confidence threshold for first-stage matching

### `ball_tracking.track_low_thresh`

- Low-confidence threshold for second-stage IoU rescue

### `ball_tracking.track_buffer`

- Frames to keep lost tracks. Raised 15→60 so a
- 1s ball-detection gap during a kick (motion
- blur + small object) doesn't kill the track
- and force a new-track promotion on the next
- isolated detection — which BoT-SORT's
- unconfirmed→confirmed state machine then
- silently drops at typical 0.5-conf detections.

### `ball_tracking.match_thresh`

- Distance threshold for matching (normalized center distance)

### `ball_tracking.new_track_thresh`

- Min confidence for new track initialization (higher than track_high to avoid false positives)

### `ball_tracking.max_position_distance`

- Max pixel distance for center-distance matching

### `ball_tracking.fuse_score`

- Fuse detection score into distance matrix
- BoT-SORT specific (ignored when tracker_type=bytetrack)

### `ball_tracking.gmc_method`

- Camera motion compensation: "sparseOptFlow", "orb", "none"

### `ball_tracking.with_reid`

- Ball appearance is uniform, ReID not useful

### `ball_tracking.proximity_thresh`

- Spatial proximity threshold

### `ball_tracking.appearance_thresh`

- Appearance similarity threshold
- 3D trajectory estimation (physics-based parabolic model)

### `ball_tracking.trajectory_3d.window_size`

- Max observations in sliding window

### `ball_tracking.trajectory_3d.min_observations`

- Minimum frames to attempt 3D fit

### `ball_tracking.trajectory_3d.gravity`

- m/s²

### `ball_tracking.trajectory_3d.ball_real_diameter`

- FIFA size 5 ball diameter in meters (unused in 3D estimation)

### `ball_tracking.trajectory_3d.bbox_padding`

- Legacy: no longer used for depth estimation

### `ball_tracking.trajectory_3d.ground_contact_weight`

- Soft penalty weight for Z≈0 at trajectory endpoints

### `tracking.max_age`

- Max frames to keep lost tracks

### `tracking.min_hits`

- Min detections before track is confirmed

### `tracking.iou_threshold`

- IoU thresholds for the cascaded matcher (1 - IoU; higher = more permissive).
- Confirmed-stage stays loose so a one-frame ReID hiccup falls back to bbox
- geometry. Tentative-stage is tighter: a freshly-spawned track has no
- appearance history, so a high-IoU-but-physically-distant detection (the
- perspective failure where ~70 px vertical drift on a far player has IoU
- ≈ 0.4 yet pitch distance ≈ 3 m) was stealing the wrong identity. Reject
- via IoU here and let tentative-pitch (4 m gate) re-attach.

### `tracking.max_iou_distance_tentative`

- Confirmed-ReID stage hybrid gate: a (track, det) pair passes if
- EITHER the predicted bbox IoU ≥ reid_iou_min OR pitch distance
- ≤ reid_pitch_max_m. Without the IoU half, ReID alone can teleport
- a confirmed track to a 2 m-away same-team player whose ReID
- cosine is marginally smaller (the orig 1 frame 303→306 case in
- the kids clip). Without the pitch half, fast wingers whose bbox
- crosses its own width in one sample fail. tentative-pitch keeps
- the looser pitch_gate_m (4 m default) so newly-spawned tracks
- can still re-attach after IoU dropouts.

### `tracking.reid_pitch_max_m`

- Dump raw YOLO detections (per-frame JSON + annotated diagnostic JPGs)
- under <output>/yolo_raw/ so dropouts at conf/size/pitch filters can be
- inspected offline. Set false to skip — JSON is cheap, the JPG render
- adds ~25s/200-frame clip and is gated by output.save_visualizations.

### `tracking.dump_yolo_raw`

- Ghost-track killer: delete a track whose bbox centre has not moved
- more than `stationary_max_pixels` over the last `stationary_window`
- tracking-frame updates.  Targets the "YOLO false-positive on a
- banner / fence / spectator" failure mode where the same static
- detection keeps re-binding to an old track via ReID and produces
- a 5-15s ghost bbox.  Set stationary_window=0 to disable.

### `tracking.stationary_window`

- 30 tracking frames ≈ 3s at 10fps

### `reid.backend`

- "osnet" (default), "prtreid", or "clip_reid"

### `reid.model`

- Model for OSNet backend

### `reid.feature_dim`

- Feature dimension (512 for OSNet, 256 for PRTReID)

### `reid.batch_size`

- OSNet backend settings (used when backend="osnet")

### `reid.osnet.feature_dim`

- PRTReID backend settings (used when backend="prtreid")

### `reid.prtreid.weights_path`

- Path to weights (null = auto-download)

### `reid.clip_reid.backbone`

- OpenCLIP model name; "ViT-L-14" (default) or "ViT-B-16" (lighter).
- Embedding dim depends on `remove_proj`:
  - `remove_proj: true` (default, matches CLIP-ReID): ViT-L-14 → 1024-dim,
    ViT-B-16 → 768-dim.
  - `remove_proj: false`: ViT-L-14 → 768-dim, ViT-B-16 → 512-dim.

### `reid.clip_reid.pretrained`

- OpenCLIP pretrained tag used as the BASE before applying the
  fine-tuned checkpoint. Default "openai".

### `reid.clip_reid.weights_path`

- REQUIRED. Path to the fine-tuned CLIP-ReID checkpoint (e.g.
  `workspace/models/clip_reid/ViT-L-14_openai/weights_e4.pth`).
  Download manually from the Google Drive link in
  https://github.com/KonradHabel/clip_reid — the upstream repo
  doesn't expose a permanent HTTP URL so there's no auto-download.

### `reid.clip_reid.remove_proj`

- Mirrors upstream `OpenClipModel(remove_proj=True)`. When true the
  image encoder's joint image/text projection is dropped, so the
  embedding is the pre-projection visual feature. Default true —
  matches CLIP-ReID's published evaluation.

### `reid.clip_reid.batch_size`

- Inference batch (default 32). ViT-L-14 at 336×336 uses ~4 GB GPU
  per batch-32. Halve if you OOM.
- Team classification — consumed by the track_consolidation stage
- (tracker itself no longer does team/role assignment).

### `team_classification.position_weight`

- Weight for position features in KMeans

### `team_classification.min_samples_per_team`

- Minimum samples before classification

### `visualization.backend`

- "minimal" (default) or "step"
- Highlight Clipping (agent-based)

### `highlights.closeup.output_size`

- Close-up crop resolution (w, h)

### `highlights.closeup.padding_factor`

- How much to expand bbox for close-up

### `highlights.closeup.smooth_alpha`

- EMA alpha for crop center smoothing

### `highlights.closeup.medium_padding_factor`

- Padding for "medium" view type

### `highlights.temporal.buildup_max_seconds`

- Max lookback for build-up start

### `highlights.temporal.buildup_padding_seconds`

- Extra padding before build-up start

### `highlights.temporal.buildup_view`

- wide / medium / closeup

### `highlights.temporal.strike_pre_seconds`

- Seconds before shooter_frame (short — cut lands on the kick)

### `highlights.temporal.strike_post_seconds`

- Seconds after goal_frame (ball enters net)

### `highlights.temporal.strike_view`

- Closeup on ball during shot

### `highlights.temporal.celebration_seconds`

- Max celebration duration (or until track lost)

### `highlights.temporal.celebration_view`

- wide / medium / closeup

### `highlights.temporal.replay_enabled`

- Slow-motion replay of the strike

### `highlights.temporal.replay_speed`

- Playback speed for replay (0.4 = 2.5x slower)

### `highlights.effects.shooter_spotlight`

- Glow ellipse under scorer's feet

### `highlights.effects.spotlight_color`

- Gold (BGR)

### `highlights.effects.ball_trail`

- Trajectory trail after shot

### `highlights.effects.trail_length`

- Frames of trail history

### `highlights.effects.trail_color`

- Yellow (BGR)

### `highlights.recipes[0].composer`

- Track Consolidation (Claude Opus jersey + OSNet ReID merge)

### `track_consolidation.enabled`

- Filter tracks with fewer observations. Specify in seconds and the
- resolver converts to frames at the effective process_fps; the legacy
- ``min_track_frames`` (frame count) still wins when set.

### `track_consolidation.min_track_seconds`

- frames_per_track:
-   - 0 (default): take EVERY frame the track appears in (jersey OCR
-     fans out one parallel LLM call per crop; aggregation in Python).
-     A 100-frame track = 100 small parallel calls.
-   - >0: top-K sampling like the old behaviour — picks bbox-area
-     largest then time-spreads. Cheaper but noisier.

### `track_consolidation.frames_per_track`

- Hard ceiling on how many crops one track can spend on the LLM. A
- rare 1000-frame track is bounded so cost stays predictable. Crops
- beyond the cap are dropped lowest-bbox-area first.

### `track_consolidation.max_crops_per_track`

- Optional temporal stride: take every Nth frame in the track's
- appearance list instead of top-K or all-frames. When unset, the
- consolidator defaults to stride=10 if jersey.ocr_backend=rapidocr
- (cheap OCR per call, no API rate limit but still 50 ms each), or
- falls back to frames_per_track / max_crops_per_track for the LLM
- backend.
- frame_stride: 10

### `track_consolidation.sampler.min_bbox_height`

- upscale crops smaller than this

### `track_consolidation.sampler.upper_body_ratio`

- focus on torso (jersey number region)

### `track_consolidation.jersey`

- Backend selector. Only one of {claude, gemini, qwen} is used per run.
- Each sub-section is read only when its name matches backend.

### `track_consolidation.jersey.backend`

- Phase-2 (per-crop number reading) backend. Phase 1 (role / team
- decision) always uses the LLM ``backend`` above — only the per-
- crop OCR can be swapped.
-   - llm (default): ``SINGLE_OCR_PROMPT`` LLM call per crop. Best
-     accuracy. Capped at ~30 crops/track to control cost / rate
-     limits.
-   - rapidocr: local PaddleOCR ONNX runtime via rapidocr-onnxruntime.
-     Free and fast, so the consolidator OCRs every frame in each
-     track instead of sub-sampling. Less accurate on partial /
-     blurry digits; the per-crop voting aggregator washes the
-     noise out.

### `track_consolidation.jersey.ocr_backend`

- rapidocr-specific tuning (only read when ocr_backend == rapidocr):

### `track_consolidation.jersey.rapidocr.upscale`

- bicubic upscale before OCR (kids jerseys are tiny)

### `track_consolidation.jersey.rapidocr.min_confidence`

- drop OCR rec hits below this score
- --- Claude Opus (AWS Bedrock) -----------------------------------

### `track_consolidation.jersey.region`

- --- Gemini (Google AI Studio, uses GOOGLE_API_KEY) --------------
- Set backend: gemini and override model_id below.
- model_id: "gemini-2.5-pro"           # or "gemini-2.5-flash"
- api_key_env: "GOOGLE_API_KEY"
- --- Qwen VL via local vLLM server -------------------------------
- Set backend: qwen, launch ``bash scripts/start_qwen_vllm.sh``, then:
- model_id: "Qwen/Qwen3.6-27B-FP8"
- base_url: "http://localhost:8000/v1"
- --- Shared tuning ------------------------------------------------
- Cross-track parallelism (how many tracks resolved concurrently).
- Bedrock Claude Opus in us-east-1 typically allows 60-200 RPM. 32
- leaves headroom for the per-track OCR fan-out (each track may
- spawn ocr_per_track_concurrency=8 subcalls) — at 32 that's up to
- 256 in-flight, but most cluster fast enough that the steady state
- stays well under quota. boto3 retries 429s automatically.

### `track_consolidation.jersey.max_concurrency`

- Within-track parallelism for per-crop OCR (Phase 2 of
- recognize_multi). Each track fans out N OCR calls, one per crop;
- this caps that fan-out so a single 200-crop track doesn't
- monopolise the API quota. Total in-flight LLM calls roughly =
- max_concurrency × ocr_per_track_concurrency, so be conservative.

### `track_consolidation.jersey.ocr_per_track_concurrency`

- Crop subset size sent to the role/team decision call (Phase 1).
- 8 sharpest evenly-spaced crops is enough for "is this a player /
- what team" without burning tokens.

### `track_consolidation.jersey.jpeg_quality`

- Per-track jersey confidence threshold; below this the LLM's
- winning number is treated as unknown. With per-crop OCR over
- ALL frames in the track (frames_per_track=0), conf ranges
- roughly:
-   - winner_score / N_crops × purity, sqrt'd. So a 132-crop
-     track with 50 strong votes for #20 and 5 spurious ≈
-     sqrt(50/132 * 50/55) ≈ 0.59 → keep.
-   - 42-crop track, 12 strong votes for #18 with no rivals ≈
-     sqrt(12/42 * 1.0) ≈ 0.53 → keep.
-   - mixed-identity track: top vote outscores rivals 4×, but
-     per-N is low: sqrt(20/95 * 20/35) ≈ 0.34 → drop ✅.
- 0.25 — lowered from 0.4 to recover long tracks where the
- player's back is visible on only ~1/3 of frames (e.g. orig 90,
- 10 strong #21 votes / 48 crops → conf 0.34). The geometric mean
- punishes long tracks for having lots of back-not-visible crops
- even when the readable subset is unanimous.

### `track_consolidation.reid`

- ReID-first consolidation (see _consolidator.consolidate). Cosine
- gate for the primary clustering. 0.9 fits PRTReID's same-person
- centroid distribution on Veo kids footage; raise to 0.92+ for
- higher-fidelity broadcast where cross-team cosines are tighter.

### `track_consolidation.reid.orphan_absorb_threshold`

- second-chance absorption above this
- Two clusters that ended up with the same (team, jersey_number)
- but didn't merge in Stage A get a second-pass merge if their
- ReID cosine is at least this much. Jersey number is a second
- independent identity signal — same kit + same number + ReID
- near-miss is still very likely the same player who reappeared
- after a long-interval gap. 0.8 is well above the cross-team
- noise floor (~0.65) but below the same-team-different-player
- ceiling (~0.87), so this catches reappearances without
- collapsing two real same-team-same-number players.

### `track_consolidation.goalkeeper.enabled`

- post-consolidation GK identification

### `track_consolidation.goalkeeper.goal_area_depth`

- metres — cluster mean x must be within this of own goal line
- The ``output`` block used to control in-place rewrites of
- tracking/tracks.json + tracking/team_assignments.json. With the
- additive layout (consolidated copies live under
- track_consolidation/) those flags no longer do anything; downstream
- readers select via goalinsight.track_consolidation.load_tracks.
- Annotated Full-Video Render (HUD-style)

### `annotated_video.output_fps`

- null = source fps

### `annotated_video.carrier.outline_color`

- weak yellow (BGR)

### `annotated_video.minimap.margin_px`

- Output

### `output.format`

- Output format

### `output.save_embeddings`

- Whether to save ReID embeddings
- Hardware

### `device`

- "cuda" or "cpu"

### `batch_size`

- Event Detection

### `events.possession.distance_threshold`

- meters — max player-ball distance for possession (wider for tracking noise)

### `events.possession.min_consecutive_seconds`

- seconds — minimum duration to confirm possession (catch short touches)

### `events.possession.speed_break_threshold`

- m/s — ball speed above which possession breaks (allow fast dribbles)

### `events.pass.pass_speed_threshold`

- m/s — minimum ball speed to qualify as a pass (catch short passes)

### `events.pass.max_transit_seconds`

- seconds — max gap between possession spans (allow longer aerial passes)

### `events.shot.shot_speed_threshold`

- m/s — minimum ball speed for a shot (filter slow crosses)

### `events.shot.max_angle_to_goal`

- degrees — max angle deviation from goal

### `events.shot.min_confidence`

- minimum ball detection confidence

### `events.shot.shooter_lookback_seconds`

- seconds — how far back to look for shooter (wider window)

### `events.shot.dedup_gap_seconds`

- seconds — deduplication window

### `events.shot.approach_window_seconds`

- seconds — approach direction check window

### `events.shot.max_frame_gap_seconds`

- seconds — max gap between consecutive ball obs

### `events.shot.approach_margin`

- meters — detect shots when ball is within this distance of goal line

### `events.shot.tracking_lost_seconds`

- seconds — gap in tracking that indicates ball left the field

### `events.carry.min_duration_seconds`

- seconds — minimum possession duration (catch shorter carries)

### `events.carry.min_total_distance`

- meters — total displacement threshold

### `events.carry.min_forward_distance`

- meters — forward progress threshold

### `events.defensive.tackle_proximity`

- meters — defender must be this close to ball

### `events.defensive.tackle_deflection_angle`

- degrees — ball direction change threshold (more sensitive)

### `events.defensive.tackle_max_gap_seconds`

- seconds — max gap between possession spans for tackle

### `events.defensive.deflection_window_seconds`

- seconds — window for computing ball deflection angle
- Per-player profile artifacts (stage: player_profile).
- Front/back crops + heatmap PNGs are always produced; per-player follow-cam
- clips ("spotlights") are opt-in via spotlights.enabled.

### `player_profile.spotlights.target_player_height_frac`

- player ~2/3 of frame height after resize

### `player_profile.spotlights.presence_gap_seconds`

- gap between detections > this → cut, no bridge

### `player_profile.spotlights.trail`

- pitch-trail polyline (off by default; visually busy)

### `player_profile.spotlights.min_observations`

- skip players with fewer detections than this
- Pipeline orchestration
- Available stages: field_registration, tracking, track_consolidation, event_detection, player_profile, highlights, annotated_video
- NOTE: track_consolidation must run BEFORE event_detection so events.json
- carries stable player_id strings (e.g. "A-7") instead of raw int track_ids.

### `pipeline.stages[1]`

- Stage execution options. Remote execution is opt-in: pass
- --remote-stages=field_registration,tracking on the CLI (or set
- execution.remote_stages here) to run those stages on a SageMaker
- Processing Job. Default is local — leave this commented out.
- execution:
-   remote_stages: []
- SageMaker Processing Job settings — populate with the values printed
- by sagemaker/setup_aws.sh. Leaving any of region/role_arn/image_uri/
- s3_bucket unset disables remote execution (the stage adapter falls
- back to local with an explicit error).

### `sagemaker.fetch_visualizations`

- set true if you need vis dumps locally

## kids_soccer.yaml
### `video.process_fps`

- Camera intrinsics. Recovered jointly with pitch dims via fit_pitch_dims.py
- --fit-fx (this is a long-focal sideline cam, not the 53° default).

### `intrinsics.fx`

- hfov ≈ 43° (image width assumed 1920)
- Non-FIFA youth pitch. Dims fit jointly from 3 manual annotation frames via
- scripts/fit_pitch_dims.py --fit-fx (combined RMSE 5.82 px on 53 residuals,
- fx=2448 / hfov 43°). f64020 and f246/f338 share above-ground cameras at
- touchline_bottom side. The two unconstrained dims (no center circle /
- penalty mark in any annotation) are kept at the rough youth-pitch guess.
- Kids 7-a-side youth pitch — see configs/pitches/kids_soccer.yaml for
- the full dims. goal_line_to_penalty_mark and center_circle_radius are
- unconstrained by the kids-cup annotations and may want a revisit when
- either landmark becomes visible in footage.

### `field_registration.backend`

- Use the upstream-aligned port: it plumbs the pitch.* override above
- (via SoccerPitch.from_dict) into the world keypoint table, and uses
- heuristic_voting (18 RANSAC combos) instead of single-shot 4-point PnP.
- The in-tree pnlcalib path still hardcodes FIFA dims and 4-point PnP, so
- on this non-FIFA youth pitch it produced very loose H matrices.

### `field_registration.pnlcalib.keypoint_threshold`

- Line detection: re-enabled with finetuned line model (5 frames, 26 lines).
- Zero-shot SV_lines previously pulled rep_err 2.91 → 82px because of
- uncertain endpoints; a finetuned head on the actual kids_soccer pitch
- should give cleaner endpoints.

### `field_registration.pnlcalib.line_weight`

- Drop keypoints whose world<->image correspondence is inconsistent
- with a single planar homography. Catches mis-IDed detections (e.g.
- id 17 firing on a left-half frame) before they pull cv2.calibrateCamera
- into a phantom small-fx solution.

### `field_registration.pnlcalib.geometric_outlier_filter.min_keypoints`

- SIFT-based chain propagation between PnLCalib anchors. Fills in
- frames the keypoint detector couldn't see enough markings to solve
- — a static-camera 60s clip stays geometrically stable enough across
- 5–20 frame gaps that frame-to-frame H accumulation tracks the pitch
- well. ``max_gap_frames`` bails out of long propagations where drift
- would dominate.

## kids_soccer_physical.yaml
### `video.process_fps`

- 30 fps source = process every video frame (dense
- tracking). Smaller per-frame displacements make
- IoU/pitch gates much tighter, killing the
- "two-same-team-players-close-together" ID
- switches (e.g. orig 54 frame 249) that survived
- at 10 fps. fps-sensitive tracker defaults
- (max_age, pitch_gate_m, reid_pitch_max_m) auto-
- scale by effective_fps/10, so physical meaning
- is preserved.
- Single source of truth for "compute every Nth video frame" used both
- for the per-stage compute path and for the per-frame jpg/json vis
- sink. Keeping these in sync means the vis sample lines up exactly
- with what each stage actually processed — no missing-detection holes
- (e.g. the old vis_frame_stride=10 falling on frame 10, which was
- never sampled because process_fps gave step=3).

### `sample.stride`

- process every video frame (matches process_fps=30 on 30fps source)
- Long-focal sideline cam (fx=2448, hfov ≈ 43°). See kids_soccer.yaml.

### `intrinsics.fx`

- Non-FIFA youth pitch — see configs/pitches/kids_soccer.yaml for the
- full dims. The physical backend reads pitch_length / pitch_width from
- the resolved ``pitch:`` block; the rest of the pitch.* table is
- consumed by pnlcalib_orig and isn't used on this path, but kept for
- parity / future joint runs.

### `field_registration.physical.focal_hfov_deg_bounds`

- wide upper for late-clip zoom-in;
- resolver converts to px focal at the
- source's actual width (1920 → [2300, 4000])

### `field_registration.physical.joint_optimize`

- camera zooms during clip — keep per-frame f

### `field_registration.physical.world_error_threshold`

- Camera position. Iterative refinement via Pass 2 lock-C:
-   measured prior:    (-4.000, -21.575, 2.000)
-   iter 1 converged:  (-5.116, -22.069, 2.844)
-   iter 2 converged:  (-5.557, -21.926, 2.820)  ← current (best stable point)
-   iter 3 converged:  (-5.906, -21.714, 2.765)  — x kept drifting -0.35m
- x didn't converge (left-half KPs over-represented in train set), so we
- stop at iter 2: y / z had stabilized, reproj already at 10.85 px median.
- Diff vs measured: x ≈ -1.6, y ≈ -0.35, z +0.82 (camera is taller).

### `field_registration.physical.lock_camera_position`

- SOFT prior — frames with only 3 (often near-collinear) keypoints can't
- all reproject to zero under hard-lock + 4-DOF (only rvec + focal free),
- they settle for a ~22 px three-point compromise. With soft, the LM is
- allowed to nudge tvec away from this prior to make the visible anchors
- fit while position_weight pulls it back.

### `field_registration.physical.position_weight`

- pixels-equivalent; raise to keep tighter to prior
- Single sample stride for the whole pipeline lives at top-level
- `sample.stride`. calibration_skip=1 means physical runs PnP on
- every sampled frame (no second decimation on top of process_fps).

### `field_registration.physical.calibration_skip`

- Per-frame vis JPG sink stride (sample-frame indices, not video).
- 1 = save a calibration JPG for every sampled frame.

### `field_registration.physical.vis_interval`

- SIFT-based gap fill: replace linearly-interpolated frames with chained
- H propagation from high-quality anchors. Much more accurate when the
- camera is panning/zooming during a multi-second gap.

### `field_registration.physical.gap_fill_chain.anchor_max_reproj_px`

- only frames with reproj<=30px become anchors

### `field_registration.physical.gap_fill_chain.overwrite_above_reproj_px`

- don't overwrite frames already <=30px

### `field_registration.physical.gap_fill_chain.max_anchor_distance_frames`

- extended further to cover the 192-frame gap between anchors 30 and 222

### `field_registration.physical.gap_fill_chain.sift_top_skip_frac`

- mask top 25% (OSD + sky)
- Reuse the finetuned kids_soccer keypoint + line heads. The physical
- runner pulls model paths from this block.

### `field_registration.pnlcalib`

- v2 KP best after full 100-epoch run on kids_soccer_v2 (clean manual
- annotations). 83.3% recall (30/36) / 0 phantoms, mean err 2.8 px on
- the v2 train set; baseline run_20260531_024941 was 71% / 3 phantoms.
- Left-goal z=-G_H channels (11 / 12 / 16) still don't activate above
- conf=0.05 — limitation of SV_kp pretrained head, not training data.

### `field_registration.pnlcalib.use_lines`

- v2 line head from kids_soccer_v2 freeze_backbone training (200 ep).

### `device`

- PRTReID is fine-tuned on SoccerNet (player-level), 256-dim with HRNet32.
- Our same-jersey discrimination test on OSNet showed cross-player cosine
- at 0.85 — too close to true-continuation cosine (0.85) for Hungarian to
- pick correctly under team-uniform conditions. PRTReID's part-based
- embedding (separate body parts) should give better same-jersey
- discrimination on the kids clip.

### `reid.backend`

- 30fps tracking — both pitch_gate_m and reid_pitch_max_m auto-derive to
- the values this clip was tuned at (4.0 * 10/30 = 1.33 m,
- 1.5 * 10/30 = 0.5 m). No per-clip overrides needed.
- Per-stage vis sinks. mp4 outputs are full-rate; the per-frame jpg/json
- stride matches sample.stride (set at top-level) so vis frames line up
- exactly with sampled frames — no overlay gaps from picking a frame the
- stage never actually processed. Override per stage if you want a
- different cadence.

### `track_consolidation`

- 1 s at process_fps=30 ≈ 30 frames. Filters genuine 1–2 frame YOLO
- false positives while keeping fragmented player tids. Resolver
- converts seconds → frames at the effective process_fps.

### `track_consolidation.min_track_seconds`

- Phase-2 jersey OCR runs through the LLM (Bedrock). RapidOCR is
- fast but misses partial / occluded numbers (e.g. orig 11 all 51
- crops returned "no text detected"); the LLM still reads them
- via Phase-1 colour / partial-digit reasoning.

### `track_consolidation.jersey.ocr_backend`

- stride=10 keeps OCR cost bounded at ~60 LLM calls per long track.
- On Bedrock that's ~$0.05/track; the bottleneck is rate limiting.

## kids_soccer_physical_reid08.yaml
### `video.process_fps`

- 30 fps source = process every video frame (dense
- tracking). Smaller per-frame displacements make
- IoU/pitch gates much tighter, killing the
- "two-same-team-players-close-together" ID
- switches (e.g. orig 54 frame 249) that survived
- at 10 fps. fps-sensitive tracker defaults
- (max_age, pitch_gate_m, reid_pitch_max_m) auto-
- scale by effective_fps/10, so physical meaning
- is preserved.
- Single source of truth for "compute every Nth video frame" used both
- for the per-stage compute path and for the per-frame jpg/json vis
- sink. Keeping these in sync means the vis sample lines up exactly
- with what each stage actually processed — no missing-detection holes
- (e.g. the old vis_frame_stride=10 falling on frame 10, which was
- never sampled because process_fps gave step=3).

### `sample.stride`

- process every video frame (matches process_fps=30 on 30fps source)
- Long-focal sideline cam (fx=2448, hfov ≈ 43°). See kids_soccer.yaml.

### `intrinsics.fx`

- Non-FIFA youth pitch — see configs/pitches/kids_soccer.yaml for the
- full dims. The physical backend reads pitch_length / pitch_width from
- the resolved ``pitch:`` block; the rest of the pitch.* table is
- consumed by pnlcalib_orig and isn't used on this path, but kept for
- parity / future joint runs.

### `field_registration.physical.focal_hfov_deg_bounds`

- wide upper for late-clip zoom-in;
- resolver converts to px focal at the
- source's actual width (1920 → [2300, 4000])

### `field_registration.physical.joint_optimize`

- camera zooms during clip — keep per-frame f

### `field_registration.physical.world_error_threshold`

- Camera position. Iterative refinement via Pass 2 lock-C:
-   measured prior:    (-4.000, -21.575, 2.000)
-   iter 1 converged:  (-5.116, -22.069, 2.844)
-   iter 2 converged:  (-5.557, -21.926, 2.820)  ← current (best stable point)
-   iter 3 converged:  (-5.906, -21.714, 2.765)  — x kept drifting -0.35m
- x didn't converge (left-half KPs over-represented in train set), so we
- stop at iter 2: y / z had stabilized, reproj already at 10.85 px median.
- Diff vs measured: x ≈ -1.6, y ≈ -0.35, z +0.82 (camera is taller).

### `field_registration.physical.lock_camera_position`

- SOFT prior — frames with only 3 (often near-collinear) keypoints can't
- all reproject to zero under hard-lock + 4-DOF (only rvec + focal free),
- they settle for a ~22 px three-point compromise. With soft, the LM is
- allowed to nudge tvec away from this prior to make the visible anchors
- fit while position_weight pulls it back.

### `field_registration.physical.position_weight`

- pixels-equivalent; raise to keep tighter to prior
- Single sample stride for the whole pipeline lives at top-level
- `sample.stride`. calibration_skip=1 means physical runs PnP on
- every sampled frame (no second decimation on top of process_fps).

### `field_registration.physical.calibration_skip`

- Per-frame vis JPG sink stride (sample-frame indices, not video).
- 1 = save a calibration JPG for every sampled frame.

### `field_registration.physical.vis_interval`

- SIFT-based gap fill: replace linearly-interpolated frames with chained
- H propagation from high-quality anchors. Much more accurate when the
- camera is panning/zooming during a multi-second gap.

### `field_registration.physical.gap_fill_chain.anchor_max_reproj_px`

- only frames with reproj<=30px become anchors

### `field_registration.physical.gap_fill_chain.overwrite_above_reproj_px`

- don't overwrite frames already <=30px

### `field_registration.physical.gap_fill_chain.max_anchor_distance_frames`

- extended further to cover the 192-frame gap between anchors 30 and 222

### `field_registration.physical.gap_fill_chain.sift_top_skip_frac`

- mask top 25% (OSD + sky)
- Reuse the finetuned kids_soccer keypoint + line heads. The physical
- runner pulls model paths from this block.

### `field_registration.pnlcalib`

- v2 KP best after full 100-epoch run on kids_soccer_v2 (clean manual
- annotations). 83.3% recall (30/36) / 0 phantoms, mean err 2.8 px on
- the v2 train set; baseline run_20260531_024941 was 71% / 3 phantoms.
- Left-goal z=-G_H channels (11 / 12 / 16) still don't activate above
- conf=0.05 — limitation of SV_kp pretrained head, not training data.

### `field_registration.pnlcalib.use_lines`

- v2 line head from kids_soccer_v2 freeze_backbone training (200 ep).

### `device`

- PRTReID is fine-tuned on SoccerNet (player-level), 256-dim with HRNet32.
- Our same-jersey discrimination test on OSNet showed cross-player cosine
- at 0.85 — too close to true-continuation cosine (0.85) for Hungarian to
- pick correctly under team-uniform conditions. PRTReID's part-based
- embedding (separate body parts) should give better same-jersey
- discrimination on the kids clip.

### `reid.backend`

- 30fps tracking — pitch_gate_m and reid_pitch_max_m auto-derive to the
- tuned values (1.33 m and 0.5 m). max_cosine_distance is the only knob
- left; merge into the duplicate tracking: block below.
- Per-stage vis sinks. mp4 outputs are full-rate; the per-frame jpg/json
- stride matches sample.stride (set at top-level) so vis frames line up
- exactly with sampled frames — no overlay gaps from picking a frame the
- stage never actually processed. Override per stage if you want a
- different cadence.

### `track_consolidation`

- 1 s at process_fps=30 ≈ 30 frames. Filters genuine 1–2 frame YOLO
- false positives. Resolver converts seconds → frames at effective fps.

### `track_consolidation.min_track_seconds`

- Phase-2 jersey OCR runs through the LLM (Bedrock). RapidOCR is
- fast but misses partial / occluded numbers (e.g. orig 11 all 51
- crops returned "no text detected"); the LLM still reads them
- via Phase-1 colour / partial-digit reasoning.

### `track_consolidation.jersey.ocr_backend`

- stride=10 keeps OCR cost bounded at ~60 LLM calls per long track.
- On Bedrock that's ~$0.05/track; the bottleneck is rate limiting.

### `pipeline.stages[5]`

- Override: stricter ReID threshold (cosine >= 0.8 instead of 0.7)

### `tracking.max_cosine_distance`

- was 0.3 → cosine threshold lifted from 0.7 to 0.8

## sunday_soccer.yaml
### `video.process_fps`

- 60 fps source → sample every 2nd frame for
- 30 fps processing (dense tracking). Smaller
- per-frame displacements make IoU/pitch gates
- much tighter, killing the "two-same-team-
- players-close-together" ID switches. fps-
- sensitive tracker defaults (max_age,
- pitch_gate_m, reid_pitch_max_m) auto-scale
- by effective_fps/10, so physical meaning is
- preserved.
- Single source of truth for "compute every Nth video frame" used both
- for the per-stage compute path and for the per-frame jpg/json vis
- sink. Keeping these in sync means the vis sample lines up exactly
- with what each stage actually processed — no missing-detection holes.

### `sample.stride`

- leave per-stage sampling fps-driven
- (process_fps=30 on a 60fps source ⇒ every
- 2nd frame is processed). Setting stride>1
- here would compound on top of process_fps.
- Pitch geometry — see configs/pitches/fifa.yaml for the full dims
- (105 × 68 m, FIFA spec). Used by the physical backend's PnP + the
- rest of the pipeline (events, highlights, pitch labels, etc.).

### `field_registration.backend`

- Calibration changes slowly across frames (camera doesn't pan/zoom
- at 30 fps). Run field_registration at 10 fps to cut wall-clock
- ~3× without affecting downstream stages — tracking still consumes
- the full 30 fps stream because video.process_fps stays at 30.

### `field_registration.physical`

- Generic 4K profile (3500 fx, no distortion) — matches the
- 3840x2160 Sunday-cup footage which is from a non-Veo camera
- (phone / GoPro). Wide focal_bounds let LM refine the rough
- initial fx; camera_position is a soft prior, not locked.

### `field_registration.physical.camera_profile`

- Detection batch size: 16 fits comfortably in 30GB of L40S VRAM
- alongside the web server's own model copy. ~1.7× faster than
- the project default of 8. Push to 32 only when the GPU is
- exclusive to this run.

### `field_registration.physical.ransac_reproj_error`

- line_weight=0 disables the LM line residual term entirely (the
- line model is buggy on this footage; use_line_model already
- turns off the dedicated detector, but lines derived from KP
- pairs still feed the LM unless we zero the weight here).

### `field_registration.physical.use_line_model`

- Camera zooms during the clip — frame 681 converges to fx≈6000
- (~36° hfov on 4K), frame 301 to fx≈9000 (~24°). The bounds need to
- cover the most-zoomed frame; otherwise PnP RANSAC can't find a
- focal that fits. Expressed as HFOV degrees so the same range
- transfers to a 1080p downsample without edits — the resolver
- converts to px focal at the source's actual width.

### `field_registration.physical.world_error_threshold`

- Camera position — two-pass solve.
- Pass 1: lock=false, the LM picks per-frame cam_pos freely under
- a soft prior + wide bounds. Pass 2 (in physical_runner): take
- the median across well-fit Pass 1 frames, hard-lock that, and
- re-solve every frame at 4-DOF (rvec + fx). The prior here is
- only the cold-start guess; the run will refine it. Previous
- human-estimated (7, -37, 5) was off by several meters; the LM
- consistently lands near (0.6, -41, 6.1) on this clip.

### `field_registration.physical.position_weight`

- Per-axis hard bounds in metres around the prior. Wide enough
- for LM to find the true cam_pos when the prior is rough; tight
- enough that tvec can't drift to absurd places (e.g. inside the
- pitch).

### `field_registration.physical.pitch_bounds_deg`

- Camera tilt below horizon (degrees). Sunday-cup rig sits low
- (~5m tall, ~35m behind goal line), so the camera looks nearly
- horizontally — typical pitch_deg is 3–10°, never near broadcast
- truck levels (20–30°). Bound the LM to a generous [2, 15] band:
- this catches the runaway-into-near-flat (≤2°) and inverted
- (>15° looks-down) failure modes without rejecting legitimate
- frames. Barrier: 100 px / degree of violation, fixed-length.

### `field_registration.physical.calibration_skip`

- Per-frame calibration JPG audit. Interval=30 (~1/sec on 30fps
- processed stream) keeps the overlay legible without exploding
- disk on long clips — a 10 min run drops from ~14 GB calibration
- JPGs to ~470 MB.

### `field_registration.physical.vis_interval`

- SIFT chain gap-fill — re-enabled now that FeatureMatcher
- downscales 4K input to 1080p before SIFT (max_long_side=1920),
- roughly a 4× speedup with the same descriptors. Keypoint coords
- are rescaled back to native 4K resolution before homography
- matching, so downstream PnP sees full-res points.

### `field_registration.physical.gap_fill_chain.enabled`

- Targets are sampled every Nth frame; remaining tracker frames
- get filled by ``_interpolate_camera_poses`` in tracking. step=3
- gives ~20 gap-fill anchors per second of 60fps footage — dense
- enough that 1/3-second linear pose interpolation introduces
- sub-pixel drift on a slow-panning sideline rig, and 3× faster
- than step=1 (which on 36k-frame 1080p clips ran into both
- wall-clock and OOM walls).

### `field_registration.physical.gap_fill_chain.frame_step`

- Use the GPU LightGlue+SuperPoint matcher (~20× faster than
- the CPU SIFT+FLANN backend on 4K, with comparable inlier
- quality). Falls back to SIFT if lightglue isn't installed.

### `field_registration.physical.gap_fill_chain.matcher_backend`

- Anchor gate at 30px rejected ~22% of stage1 frames where the
- KP detector found 3 line constraints instead of 6 (mid-zoom
- frames where the touchline isn't framed). The ~1% of image
- width gate brings anchor coverage from 78 % to 86 % — well
- within "good enough to seed SIFT propagation" tolerance.
- Expressed as a fraction so the same gate transfers across
- resolutions; resolver multiplies by image width.

### `field_registration.physical.gap_fill_chain.max_gap_frames`

- Anchors farther than this many frames from the target are
- skipped. 60 frames ≈ 1 s on 60fps footage — within that window
- the camera barely moves, and SIFT/LightGlue matches are nearly
- all true positives. Larger windows (was 120) added 2× the SIFT
- tries per target with diminishing returns: most extras failed
- the inlier threshold or got rejected at PnP.

### `field_registration.physical.gap_fill_chain.sift_top_skip_frac`

- KP fine-tuned on 4 sunday_cup annotations (frames 0/301/681/1103,
- re-annotated 2026-06-15). Best val loss 0.000032 @ epoch 100.
- The line head is permanently disabled on this footage — the
- finetuned line model produces noisy detections that hurt LM
- convergence rather than helping. Both ``use_lines`` (controls the
- KP-derived line residual at solve time) and ``use_line_model``
- (above, in the physical block) are off.

### `device`

- 4K source — letterbox to 1920 long edge (vs default 1280) so far-end
- players keep enough pixels for YOLO to detect them. Cuts the
- YOLO-miss rate that drove the orig 64 / orig 73 frame-897 swap
- (right-side player not detected at imgsz=1280 → tracker had only
- one detection in the area → wrong assignment cascaded).
- 
- iou_threshold raised from 0.45 → 0.55 so YOLO's NMS keeps both
- bboxes when two players overlap heavily (frame 897 cluster: #8 and
- the player hugging him have IoU≈0.50; at 0.45 NMS suppressed the
- lower-conf #8). 0.55 keeps both; 0.7+ would keep way too many
- duplicates of single players.
- 
- imgsz / iou_threshold MUST live under ``unified_detection`` (the
- default fused player+ball pass uses that block); putting them under
- ``detection`` is now a hard error. ``detection.confidence_threshold``
- below is the only ``detection`` key still honoured in unified mode —
- it gates the post-hoc PlayerDetector filter.

### `unified_detection.imgsz`

- native (longer side rounded to multiple of
- 32). 1920 down-samples a 47-px ball to
- ~23 px and YOLO loses confidence on small
- / distant balls (frame 2738 was the
- smoking gun). 4K runs ~4× slower than
- 1920 but recall on small balls is much
- better.

### `unified_detection.iou_threshold`

- PRTReID is fine-tuned on SoccerNet (player-level), 256-dim with HRNet32.
- Same choice as the kids config — better same-team discrimination than OSNet.

### `reid.backend`

- 30fps tracking — pitch_gate_m auto-derives to 4.0 * 10/30 = 1.33 m,
- which is exactly what this clip needs at process_fps=30. Only override
- what's *not* the auto-scaled default.
- 
- reid_pitch_max_m raised from auto default (0.5 m) → 0.8 m: the strict
- 0.5×scale gate sat right at the borderline for legitimate 1-sample
- reattachments (e.g. orig 6 at f=103: pitch jump ~0.9m vs limit 1.0m,
- jitter pushed it across). 0.8 gives ~1.6m budget at tsu=1, still half
- the per-frame sprint distance, while letting routine reattachments
- through.

### `tracking.reid_pitch_max_m`

- Per-frame tracking JPG audit: every 60 video frames ≈ 1/sec on a
- 60fps source. Default (sample.stride=1) writes a JPG per processed
- frame, which cost ~2 GB on the 10 min clip; this keeps per-second
- diagnostic coverage at ~30 MB.

### `tracking.vis_frame_stride`

- Diagnostic dumps for ball detection are large (~9 GB each on this
- clip) and only useful when actively debugging the ball pipeline.
- Off by default; flip to true on demand.

### `tracking.dump_ball_diag`

- Jersey number recognition backend. Tried Qwen3.6-27B-FP8 (local
- vLLM) and Gemini 2.5 Flash; both committed to single-digit reads
- more often than Claude Opus, dropping ~3 jerseys vs the Claude
- baseline (15/20 vs 18/20 with jersey). Sticking with Claude Opus
- 4.7 via Bedrock for now.

### `track_consolidation`

- 1 s of observations — filters genuine 1–2 frame YOLO false positives
- but keeps fragmented player tids. Resolver converts to frames at the
- effective process_fps (30 fps × 1 s = 30 frames here).

### `track_consolidation.min_track_seconds`

- ``ocr_batch_size: 4`` packs 4 crops into ONE LLM call as separate
- images (image-list mode, no montage tile), so each crop occupies
- its own image slot. This avoids the multi-player-bleed issue
- montage cells have — when bbox padding catches a teammate
- standing close, the LLM saw a 160×160 cell with two backs and
- often committed to whichever number was clearer. Independent
- images each contain just one bbox crop with the target centred,
- so adjacent players appear as the small bbox-edge slivers they
- actually are.

### `track_consolidation.jersey.ocr_image_list`

- pass crops as image-list, not as a montage
- stride=8: ~3.75 fps sampling per track (60 fps source × 1/2 tracker
- decimation × 1/8 jersey decimation). At 60-frame tracks that's
- 22 crops / track ≈ 6 LLM calls — enough voting redundancy for
- OCR to commit on a clean angle without paying the per-call image
- premium that image-list mode adds (each crop = 1568 token min).

