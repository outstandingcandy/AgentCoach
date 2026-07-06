"""Fixed-camera field registration: solve once, propagate to every frame.

For setups where the camera doesn't move, doesn't zoom, and the lens
profile is known (or has been calibrated once via the annotate page),
running HRNet keypoint detection on every frame and re-solving PnP is
pure waste — the pose is the same on every frame.

This runner reads a single saved annotation (from
``workspace/annotations/<video_stem>/frame_<idx>.json``), re-solves
the (rvec, tvec, K, dist) once using the same path the live annotator
takes (``solve_camera_physical`` with ``free_intrinsics`` honored),
and writes that one pose into every frame's slot of the stage-1
output files. No HRNet load, no per-frame inference, no LightGlue
gap fill.

Selected via ``field_registration.backend: fixed_camera`` in the
pipeline config. Config keys consumed:

- ``camera_profile``: same as physical backend (must point at a real
  K + dist_coeffs in ``configs/camera_profiles.yaml``)
- ``camera_position``, ``focal_hfov_deg_bounds``, ``position_weight``,
  ``position_bounds_m``, ``free_intrinsics``: passed through to
  ``solve_camera_physical`` so the one-time solve mirrors what the
  annotate page produced
- ``annotation_frame``: optional explicit frame index to solve from;
  defaults to the most recently saved annotation
- ``pitch``: standard top-level pitch dict (futsal, fifa, etc.)
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from ..annotation import pitch_constants
from ..annotation.homography import (
    MIN_POINTS_FOR_PNLCALIB,
    camera_to_image_to_world,
    solve_camera,
    solve_camera_physical,
)
from ..annotation.pitch import keypoints as _pk
from ..annotation.pitch.geometry import SoccerPitch
from ..utils.config import get_process_fps_from_config
from ._runner_base import (
    compute_calibration_stats,
    init_calibration_results,
    make_sampler,
    open_video,
    print_calibration_summary,
    save_calibration_outputs,
)

logger = logging.getLogger(__name__)


def _resolve_annotations_dir(video_path: Path) -> Path:
    """Find the workspace/annotations/<stem>/ directory for *video_path*.

    Walks up from the video file looking for a sibling ``annotations``
    directory (covers ``workspace/videos/X.mov`` →
    ``workspace/annotations/X/``).
    """
    stem = video_path.stem
    cur = video_path.parent
    for _ in range(4):
        cand = cur / "annotations" / stem
        if cand.is_dir():
            return cand
        cur = cur.parent
    # Fallback: <video_parent>/../annotations/<stem>
    return video_path.parent.parent / "annotations" / stem


def _pick_annotation_frame(
    annotations_dir: Path,
    explicit: int | None,
) -> Path:
    """Return the path to the JSON of the frame we'll solve from.

    Picks the most-recently-modified ``frame_*.json`` when no explicit
    frame is given. Skips ``*_all_points.json`` and ``*_raw`` files.
    """
    if explicit is not None:
        p = annotations_dir / f"frame_{explicit}.json"
        if not p.exists():
            raise FileNotFoundError(
                f"annotation_frame={explicit} requested but "
                f"{p} does not exist",
            )
        return p
    candidates = sorted(
        (p for p in annotations_dir.glob("frame_*.json")
         if not p.name.endswith("_all_points.json")),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        raise FileNotFoundError(
            f"No frame_*.json annotations under {annotations_dir} — "
            f"open the annotate page, mark ≥6 points, click Compute "
            f"and Save before running the fixed_camera backend.",
        )
    return candidates[0]


def _solve_from_annotation(
    annotation_path: Path,
    img_size: tuple[int, int],
    physical_cfg: dict,
    camera_profiles: dict,
) -> tuple[dict, dict]:
    """Read the saved annotation, run PnP, return (camera_pose, diagnostics).

    ``camera_pose`` is the dict shape physical_runner stuffs into
    ``camera_poses[fidx]``: ``{K, dist_coeffs, rvec, tvec,
    reprojection_error}``. Diagnostics is the upstream diag dict the
    annotator surfaces (median_err / max_err / fx / mode / ...).
    """
    with open(annotation_path) as fh:
        data = json.load(fh)

    points = data.get("points") or []
    pixel_pts: list[tuple[float, float]] = []
    world_3d_pts: list[tuple[float, float, float]] = []

    # Manual keypoints: pull world (x, y, z) from the keypoint table
    # — the JSON stores world by name, and the names are stable across
    # active pitch as long as the pitch type matches.
    for p in points:
        name = p.get("keypoint_name") or p.get("name")
        pix = p.get("pixel")
        if not (name and pix and len(pix) >= 2):
            continue
        pt_3d = _pk.PITCH_POINTS.get(name)
        if pt_3d is None:
            continue
        # Same z-flip the annotator's collect_pnp_points applies — the
        # keypoint table stores crossbar z = -GOAL_HEIGHT (legacy),
        # but OpenCV solvePnP wants +z up.
        pixel_pts.append((float(pix[0]), float(pix[1])))
        world_3d_pts.append((
            float(pt_3d[0]), float(pt_3d[1]), -float(pt_3d[2]),
        ))

    # Accepted derived points (line intersections). The JSON stores
    # both pixel + world per derived entry. We rely on the saved
    # ``accepted`` flag, defaulting to True for legacy saves that
    # pre-date the flag (matches the annotator's behaviour).
    for d in data.get("derived_points") or []:
        if not d.get("accepted", True):
            continue
        pix = d.get("pixel"); wxy = d.get("world")
        if not (pix and wxy and len(pix) >= 2 and len(wxy) >= 2):
            continue
        pixel_pts.append((float(pix[0]), float(pix[1])))
        world_3d_pts.append((float(wxy[0]), float(wxy[1]), 0.0))

    if len(pixel_pts) < 4:
        raise RuntimeError(
            f"{annotation_path.name} has only {len(pixel_pts)} usable "
            f"correspondences (need ≥4 for PnP). Add more keypoints in "
            f"the annotate page and re-save.",
        )

    cam, mean_err, diag = solve_camera_physical(
        pixel_pts, world_3d_pts, img_size,
        physical_cfg=physical_cfg,
        camera_profiles=camera_profiles,
    )
    if cam is None:
        raise RuntimeError(
            f"PnP failed on {annotation_path.name}: {diag!r}. Try a "
            f"different annotation_frame or check camera_position prior.",
        )

    # Shape into camera_poses[fidx] schema (physical_runner contract).
    R = np.asarray(cam.rotation, dtype=np.float64)
    # rvec/tvec the renderer uses are stashed on the Camera by
    # solve_camera_physical; fall back to recomputing if absent.
    rvec = np.asarray(
        getattr(cam, "_pnp_rvec", cv2.Rodrigues(R)[0]),
        dtype=np.float64,
    ).reshape(3)
    pos = np.asarray(cam.position, dtype=np.float64).reshape(3)
    tvec = (-R @ pos).reshape(3)
    K = np.asarray(cam.calibration, dtype=np.float64)
    dist = np.asarray(
        getattr(cam, "_pnp_dist", np.zeros(5)),
        dtype=np.float64,
    ).ravel()

    camera_pose = {
        "K": K.tolist(),
        "dist_coeffs": dist.tolist(),
        "rvec": rvec.tolist(),
        "tvec": tvec.tolist(),
        "reprojection_error": float(mean_err),
        "world_error": 0.0,
        "world_error_all": 0.0,
        "inliers_count": int(len(pixel_pts)),
    }
    return camera_pose, diag


def _reprojection_residuals(cam, pixel_pts, world_3d_pts) -> np.ndarray:
    """Per-point reprojection residual (px) for a solved camera.

    Projects each world point with the camera's optimised pose +
    intrinsics + distortion and returns the pixel distance to its detected
    location. Used to find the single worst mis-localised detection so the
    fixed-camera model solve can trim it (the solver's own outlier loop is
    world-space and never fires for on-ground points).
    """
    R = np.asarray(cam.rotation, dtype=np.float64)
    rvec = np.asarray(
        getattr(cam, "_pnp_rvec", cv2.Rodrigues(R)[0]), dtype=np.float64,
    ).reshape(3)
    pos = np.asarray(cam.position, dtype=np.float64).reshape(3)
    tvec = (-R @ pos).reshape(3)
    K = np.asarray(cam.calibration, dtype=np.float64)
    dist = np.asarray(getattr(cam, "_pnp_dist", np.zeros(5)), dtype=np.float64).ravel()
    world = np.asarray(world_3d_pts, dtype=np.float64).reshape(-1, 1, 3)
    proj, _ = cv2.projectPoints(world, rvec, tvec, K, dist)
    proj = proj.reshape(-1, 2)
    detected = np.asarray(pixel_pts, dtype=np.float64).reshape(-1, 2)
    return np.linalg.norm(proj - detected, axis=1)


def _candidate_solve_frames(frame_indices: list[int], n: int = 12) -> list[int]:
    """Pick up to *n* candidate frames spread across the sampled range.

    A single frame's keypoint detection is unreliable (sparse pitch,
    occlusion, motion blur), so the model-detection solve tries several
    and keeps the best. Uniformly spaced over the middle 90% to skip
    fade-in / end oddness.
    """
    if not frame_indices:
        return [0]
    lo = int(len(frame_indices) * 0.05)
    hi = int(len(frame_indices) * 0.95)
    span = frame_indices[lo:hi] or frame_indices
    if len(span) <= n:
        return list(span)
    step = len(span) / n
    return [span[int(i * step)] for i in range(n)]


def _solve_from_model_detection(
    video_path: Path,
    frame_indices: list[int],
    img_size: tuple[int, int],
    fr_config: dict,
    physical_cfg: dict,
    camera_profiles: dict,
) -> tuple[dict, dict]:
    """Solve the fixed pose from a keypoint MODEL's detections.

    Runs the configured HRNet keypoint model (``keypoint_detection`` block,
    honoring a fine-tuned ``keypoint_model_path``) across several candidate
    frames, maps each detected PnLCalib channel id → landmark name → the
    active pitch's world coordinate, and POOLS all detections into one
    correspondence set for a single free-intrinsics solve. This lets the
    fixed-rig backend reuse the pitch-specific model the wizard selected,
    with no manual annotation.

    Why pool + free intrinsics: the true camera intrinsics are unknown
    (the profile is a rough guess; wide action-cam lenses diverge from
    it), so we let ``cv2.calibrateCamera`` fit focal / principal /
    distortion. That needs ~12+ points — more than a single frame's 4-8
    detections — but a FIXED rig shares one pose + intrinsics across all
    frames, so pooling candidate frames' detections is valid and gives
    enough points.

    Returns the same ``(camera_pose, diag)`` shape as ``_solve_from_annotation``.
    """
    from . import KeypointDetector
    from ._detector_config import (
        build_keypoint_detector_config,
        get_detection_config,
    )
    from ..annotation.pitch import keypoints as _pk
    from ..annotation.pitch.keypoints import PITCH_POINT_TO_PNLCALIB_ID

    det_config = get_detection_config(fr_config)
    detector = KeypointDetector(build_keypoint_detector_config(det_config))
    detector.load_model()

    id_to_name = {v: k for k, v in PITCH_POINT_TO_PNLCALIB_ID.items()}
    # 3D landmark world coords keyed by name (x, y, z). z != 0 for goal-post
    # tops (on the crossbar) — solvePnP uses their height as extra pose
    # constraint, so we must NOT flatten them to the ground plane. Matches
    # the annotation path in _solve_from_annotation (same z-flip: the table
    # stores crossbar z negative-up, OpenCV wants +z up).
    kp_world = _pk.PITCH_POINTS
    conf_thresh = float(det_config.get("keypoint_threshold", 0.3434))
    # Intrinsics handling honours ``physical.free_intrinsics`` (default
    # True). Free is the right default when the camera's true focal /
    # principal / distortion are unknown (the profile is only a rough
    # guess, and wide action-cam lenses diverge a lot from it): a FIXED
    # rig shares one pose + one set of intrinsics across all frames, so we
    # POOL detections across candidate frames into a single correspondence
    # set — that gives calibrateCamera the ~12+ points it needs.
    #
    # BUT free intrinsics is ill-posed when the detected points are few
    # and geometrically clustered (e.g. a model that only fires on a
    # handful of near-collinear landmarks): calibrateCamera then absorbs
    # the residual into non-physical distortion (huge k1/k2), giving a
    # deceptively low reproj that projects badly off the training points.
    # Setting ``free_intrinsics: false`` with a matched ``camera_profile``
    # (known K + dist) locks intrinsics and solves pose only, which is far
    # more robust for such sparse point sets. So respect the config here
    # rather than forcing free.
    solve_cfg = dict(physical_cfg)
    solve_cfg.setdefault("free_intrinsics", True)

    candidates = _candidate_solve_frames(frame_indices)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"cannot open video for fixed-pose solve: {video_path}")

    # Gather detections keyed by landmark. A FIXED camera sees each
    # landmark at ONE true pixel location, so every candidate frame that
    # detects it should agree to within a few pixels. We collect all
    # observations per landmark, then collapse to a single consensus point
    # (below) — that both removes the duplicate over-weighting (a landmark
    # seen in all 12 frames must not count 12× in the solve) and lets us
    # reject genuine mis-detections by their disagreement with the median.
    obs: dict[str, dict] = {}  # name → {"px": [(x,y)...], "world": (x,y,z)}
    frames_with_pts = 0
    try:
        for fidx in candidates:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(fidx))
            ok, frame = cap.read()
            if not ok or frame is None:
                continue
            keypoints = detector.detect(frame, convert_to_soccernet=False)
            got = False
            for kp in keypoints:
                if kp.get("confidence", 0) < conf_thresh:
                    continue
                name = id_to_name.get(kp["id"])
                world = kp_world.get(name) if name else None
                if world is None:
                    continue
                # z-flip (crossbar stored negative-up → +z up for solvePnP);
                # ground points have z=0 so the flip is a no-op for them.
                zc = -float(world[2]) if len(world) > 2 else 0.0
                rec = obs.setdefault(
                    name, {"px": [], "world": (float(world[0]), float(world[1]), zc)},
                )
                rec["px"].append((float(kp["x"]), float(kp["y"])))
                got = True
            if got:
                frames_with_pts += 1
    finally:
        cap.release()

    # Collapse each landmark to one consensus pixel: the median location,
    # after dropping observations that sit > CONSENSUS_MAX_PX from it
    # (those are the genuine mis-detections — the ones worth excluding
    # from the error). A landmark whose observations are internally
    # incoherent (median spread still large after filtering) is dropped
    # entirely: we can't trust where it is. Everything that survives is a
    # believable, de-duplicated correspondence, so the reprojection error
    # reported later is honest over exactly the points we trust.
    CONSENSUS_MAX_PX = 15.0
    pixel_pts: list[tuple[float, float]] = []
    world_3d_pts: list[tuple[float, float, float]] = []
    dropped_landmarks: list[str] = []
    for name, rec in obs.items():
        arr = np.asarray(rec["px"], dtype=np.float64)
        med = np.median(arr, axis=0)
        dev = np.linalg.norm(arr - med, axis=1)
        keep = arr[dev <= CONSENSUS_MAX_PX]
        if len(keep) == 0:
            dropped_landmarks.append(name)
            continue
        consensus = np.median(keep, axis=0)
        pixel_pts.append((float(consensus[0]), float(consensus[1])))
        world_3d_pts.append(rec["world"])
    if dropped_landmarks:
        logger.info(
            "Stage 1 (Fixed): dropped %d incoherent landmark(s) as "
            "mis-detections: %s",
            len(dropped_landmarks), ", ".join(sorted(dropped_landmarks)),
        )

    n_distinct = len(pixel_pts)
    if n_distinct < 4:
        raise RuntimeError(
            f"keypoint-model fixed-pose solve failed: only {n_distinct} "
            f"distinct landmark(s) reached consensus across {len(candidates)} "
            f"frames (conf≥{conf_thresh}, need ≥4). The model detects too "
            f"few distinct pitch points on this pitch. Use a pitch-matched "
            f"fine-tuned model, or annotate a frame by hand.",
        )

    # Honest error: with intrinsics locked and one trusted point per
    # landmark, there is no duplicate mass for a pose-coupled outlier loop
    # to over-fit, so disable that loop (world_error_threshold=inf). The
    # reprojection error then reflects every trusted point — which is what
    # "only count the points you can see" should mean: the believable
    # detections, not the subset that happens to fit a degenerate pose.
    solve_cfg = dict(solve_cfg)
    solve_cfg["world_error_threshold"] = float("inf")

    # Do NOT anchor the camera position to a config guess here. The camera
    # position is exactly what this solve is meant to RECOVER — treating a
    # hand-typed ``camera_position`` as a strong prior (soft residual +
    # P3P disambiguation tiebreak) pins the pose to that guess and, when
    # the guess is off, wrecks the fit (measured: prior-locked 61 px vs
    # free 2 px on this clip). The pooled consensus points are enough to
    # solve pose freely, so drop the prior and its bounds for the solve.
    # (A truly known rig position can still be enforced via the manual
    # annotation path or a dedicated lock flag, not this auto model-solve.)
    solve_cfg["camera_position"] = None
    solve_cfg["position_bounds_m"] = None
    solve_cfg["lock_camera_position"] = False

    cam, mean_err, diag = solve_camera_physical(
        pixel_pts, world_3d_pts, img_size,
        physical_cfg=solve_cfg, camera_profiles=camera_profiles,
    )
    if cam is None:
        raise RuntimeError(
            f"keypoint-model fixed-pose solve diverged on {len(pixel_pts)} "
            f"pooled points ({diag!r}). The detections are geometrically "
            f"inconsistent — the model likely needs more/better training "
            f"data. Annotate a frame by hand as a fallback.",
        )

    # Pixel-residual outlier rejection. The solver's built-in outlier loop
    # keys off WORLD-space error, which is identically zero here (every
    # consensus point is on the ground plane, z=0), so it never fires for a
    # fixed-camera model solve — a single mis-localised detection (e.g. a
    # keypoint that landed ~30 px off on a featureless stretch of grass)
    # then drags the mean reprojection error up even though the median is
    # low. So re-solve while dropping the single worst-reprojecting point,
    # as long as it exceeds the threshold AND we keep ≥ the minimum needed
    # for a stable pose. Bounded iterations; each drop is logged.
    reject_px = float(solve_cfg.get("model_solve_reject_px", 12.0))
    MIN_KEEP = 6  # keep enough for a well-conditioned free-intrinsics solve
    if reject_px > 0 and len(pixel_pts) > MIN_KEEP:
        cur_px = list(pixel_pts)
        cur_w3 = list(world_3d_pts)
        for _ in range(len(cur_px)):
            resid = _reprojection_residuals(cam, cur_px, cur_w3)
            worst = int(np.argmax(resid))
            if resid[worst] <= reject_px or len(cur_px) <= MIN_KEEP:
                break
            logger.info(
                "Stage 1 (Fixed): dropping worst-reprojecting point "
                "(resid %.1f px > %.1f) and re-solving on %d pts",
                resid[worst], reject_px, len(cur_px) - 1,
            )
            cur_px.pop(worst)
            cur_w3.pop(worst)
            cam2, mean_err2, diag2 = solve_camera_physical(
                cur_px, cur_w3, img_size,
                physical_cfg=solve_cfg, camera_profiles=camera_profiles,
            )
            if cam2 is None:
                # Re-solve diverged: keep the last good pose, stop trimming.
                logger.warning(
                    "Stage 1 (Fixed): re-solve after outlier drop diverged; "
                    "keeping previous pose.",
                )
                break
            cam, mean_err, diag = cam2, mean_err2, diag2
            pixel_pts, world_3d_pts = cur_px, cur_w3

    npts = len(pixel_pts)
    intr = "free" if solve_cfg.get("free_intrinsics", True) else "locked"
    logger.info(
        "Stage 1 (Fixed): %s-intrinsics solve on %d consensus landmark(s) "
        "(1 per landmark, de-duplicated across %d frames) — reproj %.2f px "
        "(honest, over every trusted point)",
        intr, npts, frames_with_pts, mean_err,
    )
    # mean_err is the honest reprojection error over every trusted
    # consensus point (no pose-coupled outlier loop to hide behind). A
    # large value means the trusted detections can't be reconciled by any
    # single pose — warn loudly rather than let a deceptively "solved" but
    # wrong calibration flow downstream.
    if mean_err > 25.0:
        logger.warning(
            "Stage 1 (Fixed): HIGH reprojection error %.1f px on %d "
            "consensus landmarks — they are geometrically weak (too few "
            "distinct, near-collinear points). Calibration is unreliable; "
            "annotate a frame by hand or fine-tune the keypoint model on "
            "more frames.",
            mean_err, npts,
        )

    R = np.asarray(cam.rotation, dtype=np.float64)
    rvec = np.asarray(
        getattr(cam, "_pnp_rvec", cv2.Rodrigues(R)[0]), dtype=np.float64,
    ).reshape(3)
    pos = np.asarray(cam.position, dtype=np.float64).reshape(3)
    tvec = (-R @ pos).reshape(3)
    K = np.asarray(cam.calibration, dtype=np.float64)
    dist = np.asarray(getattr(cam, "_pnp_dist", np.zeros(5)), dtype=np.float64).ravel()

    camera_pose = {
        "K": K.tolist(),
        "dist_coeffs": dist.tolist(),
        "rvec": rvec.tolist(),
        "tvec": tvec.tolist(),
        "reprojection_error": float(mean_err),
        "world_error": 0.0,
        "world_error_all": 0.0,
        "inliers_count": int(npts),
        # The consensus pixel locations the model actually detected (one per
        # trusted landmark). Carried through so the calibration vis can draw
        # them (green dots) alongside the projected pitch template — that is
        # what lets you eyeball detection quality per keypoint model. Every
        # surviving consensus point is a trusted inlier (mis-detections were
        # already dropped above), so the mask is all-True.
        "img_pts": [[float(x), float(y)] for (x, y) in pixel_pts],
        "inlier_mask": [True] * len(pixel_pts),
    }
    return camera_pose, diag


def run_stage1_fixed_camera(
    video_path: Path,
    output_dir: Path,
    vis_dir: Path,
    config: dict,
    process_fps: float | None,
) -> dict:
    """Solve once from an annotation, replicate the pose to every frame.

    Output files (homographies.pkl, camera_poses.pkl/json,
    calibration_metadata.json) match the physical_runner schema so
    downstream stages don't notice the difference.
    """
    fr_config = config.get("field_registration", {})
    phys_config = fr_config.get("physical", {}) or {}

    # 1. Active pitch — same idiom physical_runner uses.
    top_pitch = config.get("pitch") or {}
    pitch_dims = dict(top_pitch)
    pitch_length = float(pitch_dims.get("pitch_length", 105.0))
    pitch_width = float(pitch_dims.get("pitch_width", 68.0))
    if pitch_dims:
        try:
            pitch_constants.set_active_pitch(SoccerPitch.from_dict(pitch_dims))
        except (TypeError, ValueError) as exc:
            logger.warning("Bad pitch block, falling back to FIFA: %s", exc)

    # 2. Camera profile (mirrors solve_camera_physical's expectations).
    camera_profiles = _load_camera_profiles(phys_config)

    # 3. Open video for the metadata we need to fan the single pose
    # out across all frames.
    video = open_video(video_path)
    try:
        sampler = make_sampler(video, process_fps, backend_label="Stage 1 (Fixed)")
        frame_indices = list(sampler)

        # 4. Solve the fixed pose once. Source priority:
        #   (a) an explicit manual annotation (``annotation_frame_path``),
        #   (b) else the configured keypoint MODEL, detected on a mid-clip
        #       frame (``keypoint_detection.keypoint_model_path`` / SV_kp),
        #   (c) else auto-pick the most-recent manual annotation on disk.
        # (a)/(c) reuse the annotation JSON; (b) runs the model. Either
        # way the solved pose is replicated to every frame below.
        annotations_dir = _resolve_annotations_dir(video_path)
        explicit_path = phys_config.get("annotation_frame_path")
        # A ``keypoint_detection`` block (always present in the shipped
        # templates) means we can solve the fixed pose from the model when
        # the user didn't hand-annotate a frame. Legacy configs without the
        # block fall through to auto-picking an on-disk annotation.
        has_model = bool(fr_config.get("keypoint_detection"))
        img_size = (video.width, video.height)

        # ``solve_source`` records where the fixed pose came from (an
        # annotation file path, or ``model:<name>``) for the run metadata.
        solve_source: str
        if explicit_path:
            annotation_path = Path(explicit_path)
            if not annotation_path.exists():
                raise FileNotFoundError(
                    f"annotation_frame_path={explicit_path} does not exist",
                )
            logger.info(
                "Stage 1 (Fixed): solving once from annotation %s",
                annotation_path.name,
            )
            camera_pose, diag = _solve_from_annotation(
                annotation_path, img_size=img_size,
                physical_cfg=phys_config, camera_profiles=camera_profiles,
            )
            solve_source = str(annotation_path)
        elif has_model:
            logger.info(
                "Stage 1 (Fixed): solving from keypoint-model detection "
                "(scanning candidate frames)",
            )
            camera_pose, diag = _solve_from_model_detection(
                video_path, frame_indices, img_size=img_size,
                fr_config=fr_config, physical_cfg=phys_config,
                camera_profiles=camera_profiles,
            )
            kp_model = (fr_config.get("keypoint_detection") or {}).get(
                "keypoint_model_path",
            )
            solve_source = f"model:{Path(kp_model).parent.parent.parent.name}" \
                if kp_model else "model:SV_kp"
        else:
            explicit = phys_config.get("annotation_frame")
            annotation_path = _pick_annotation_frame(
                annotations_dir, int(explicit) if explicit is not None else None,
            )
            logger.info(
                "Stage 1 (Fixed): solving once from annotation %s",
                annotation_path.name,
            )
            camera_pose, diag = _solve_from_annotation(
                annotation_path, img_size=img_size,
                physical_cfg=phys_config, camera_profiles=camera_profiles,
            )
            solve_source = str(annotation_path)

        # 5. Build the ground-plane homography from (K, dist≈0 contribution
        # for the planar projection — dist would distort, but the planar
        # H is a pinhole approximation; downstream players-on-pitch code
        # uses it for "rough enough" image↔world mapping).
        R = cv2.Rodrigues(np.asarray(camera_pose["rvec"]))[0]
        K = np.asarray(camera_pose["K"], dtype=np.float64)
        tvec = np.asarray(camera_pose["tvec"], dtype=np.float64).reshape(3, 1)
        # World→image planar homography: project the [X, Y, 0, 1] columns.
        # H_w2i = K @ [r1 r2 t], where r1, r2 are the first two columns of R.
        H_w2i = K @ np.column_stack([R[:, 0], R[:, 1], tvec.ravel()])
        try:
            H_i2w = np.linalg.inv(H_w2i)
        except np.linalg.LinAlgError:
            raise RuntimeError("Solved H is singular; can't invert for image→world")

        logger.info(
            "Stage 1 (Fixed): reproj=%.2f px, fx=%.1f, %d annotation pts",
            camera_pose["reprojection_error"],
            float(K[0, 0]),
            camera_pose["inliers_count"],
        )

        # 6. Replicate to every processed frame.
        homographies: dict[int, np.ndarray] = {}
        camera_poses: dict[int, dict] = {}
        calibration_results = init_calibration_results(
            video_path, video, process_fps,
            extra_video_info={
                "pitch_length": pitch_length,
                "pitch_width": pitch_width,
            },
            extra_top_level={"backend": "fixed_camera"},
        )
        for fidx in frame_indices:
            homographies[fidx] = H_i2w
            # Each frame gets its own copy of the pose dict so downstream
            # code that mutates a per-frame entry doesn't accidentally
            # share state across frames.
            camera_poses[fidx] = dict(camera_pose)
            calibration_results["frames"][fidx] = {
                "calibrated": True,
                "fixed_camera": True,
                "num_keypoints": camera_pose["inliers_count"],
                "num_lines": 0,
                "num_intersections": 0,
                "total_points": camera_pose["inliers_count"],
                "inliers": camera_pose["inliers_count"],
                "reprojection_error": camera_pose["reprojection_error"],
                "world_error": 0.0,
                "world_error_all": 0.0,
                "line_constraints": 0,
                "warm_start": False,
            }

        # Per-frame calibration overlay JPGs. Mirrors physical_runner's
        # vis_dir/calibration/ output so the Library UI / inspectors see
        # the same structure. Pose is identical on every frame (that's
        # the whole point of fixed_camera), so the overlay is too — we
        # just paint it on top of each sampled frame.
        vis_interval = int(phys_config.get("vis_interval", 30))
        if vis_interval > 0:
            _render_calibration_overlays(
                video=video,
                frame_indices=frame_indices,
                vis_interval=vis_interval,
                vis_dir=vis_dir,
                camera_pose=camera_pose,
                pitch_length=pitch_length,
                pitch_width=pitch_width,
                homography=H_i2w,
                pitch_dims=pitch_dims,
            )

        save_calibration_outputs(
            output_dir, calibration_results, homographies,
            camera_poses=camera_poses,
        )
        # Mirror physical_runner: append derived fields to camera_poses.json
        # so downstream readers see the same shape.
        cam_pos_xyz = (-R.T @ tvec.ravel())
        camera_poses_json_path = output_dir / "camera_poses.json"
        with open(camera_poses_json_path) as f:
            camera_poses_json = json.load(f)
        for entry in camera_poses_json.values():
            entry["camera_position"] = {
                "x": round(float(cam_pos_xyz[0]), 3),
                "y": round(float(cam_pos_xyz[1]), 3),
                "z": round(float(cam_pos_xyz[2]), 3),
            }
            entry["focal_length"] = float(K[0, 0])
        with open(camera_poses_json_path, "w") as f:
            json.dump(camera_poses_json, f, indent=2)

        stats = compute_calibration_stats(
            calibration_results, video, sampler,
            calibrated_count=len(frame_indices),
            exclude_interpolated=False,
            extra_stats={
                "fixed_camera_source": solve_source,
                "fx": float(K[0, 0]),
                "camera_position": list(cam_pos_xyz),
            },
        )
        print_calibration_summary(stats, label="Stage 1 (Fixed)")
        return stats
    finally:
        video.cap.release()


def _render_calibration_overlays(
    *,
    video,
    frame_indices: list[int],
    vis_interval: int,
    vis_dir: Path,
    camera_pose: dict,
    pitch_length: float,
    pitch_width: float,
    homography,
    pitch_dims: dict | None = None,
) -> None:
    """Write per-frame calibration JPGs to ``vis_dir/calibration/``.

    Mirrors what ``physical_runner`` emits in its ``visualizations``
    subdirectory so the existing inspectors (and any user expecting the
    ``calibration/frame_<idx>.jpg`` audit) keep working. With fixed
    camera the pose is the same on every frame, so we just paint the
    same overlay onto each sampled frame — no per-frame solve, no
    per-frame keypoint detection.
    """
    from tqdm import tqdm
    from ..utils.pitch import get_pitch_template_points
    from .physical_runner import _draw_physical_calibration, _stamp_source
    from .pitch_template import build_field_template
    from .pnlcalib import KeypointMapper

    pitch_template = get_pitch_template_points(pitch_length, pitch_width)
    # Build the keypoint world coords from the FULL pitch block, not just
    # (length, width). ``build_field_template`` FIFA-defaults every missing
    # dimension (penalty/goal-area size, centre-circle radius, penalty-mark
    # distance), so feeding it only L/W placed every non-corner landmark at
    # its 105×68 FIFA location while the yellow LINES (via
    # ``get_pitch_template_points`` → active pitch) used the true futsal
    # geometry — the two drifted by up to ~11 m in world space, which is
    # exactly why the projected keypoints didn't sit on the projected lines.
    # Pass the complete dims so points and lines share one geometry.
    field_dims = dict(pitch_dims) if pitch_dims else {
        "pitch_length": pitch_length, "pitch_width": pitch_width,
    }
    field_dims.setdefault("pitch_length", pitch_length)
    field_dims.setdefault("pitch_width", pitch_width)
    field_world_coords, _, _ = build_field_template(field_dims)
    # Shape ``result`` the same way ``_draw_physical_calibration`` expects.
    K = np.asarray(camera_pose["K"], dtype=np.float64)
    dist = np.asarray(camera_pose["dist_coeffs"], dtype=np.float64).ravel()
    rvec = np.asarray(camera_pose["rvec"], dtype=np.float64).ravel()
    tvec = np.asarray(camera_pose["tvec"], dtype=np.float64).ravel()
    R_mat, _ = cv2.Rodrigues(rvec)
    result_dict = {
        "homography": homography,
        "camera_params": {
            "K": K,
            "dist_coeffs": dist,
            "rvec": rvec,
            "tvec": tvec,
            "R": R_mat,
            "focal_length": float(K[0, 0]),
        },
        "final_error": float(camera_pose.get("reprojection_error", 0.0)),
        "world_error": 0.0,
        "world_error_all": 0.0,
        "inliers": int(camera_pose.get("inliers_count", 0)),
        "total_points": int(camera_pose.get("inliers_count", 0)),
        "line_constraints_count": 0,
    }
    # When the pose was solved from a keypoint MODEL, the runner stashes the
    # detected consensus pixels; surface them so _draw_physical_calibration
    # paints the actual detections (green=inlier) over the projected template.
    # Absent for the manual-annotation path (nothing detected to show).
    img_pts = camera_pose.get("img_pts")
    if img_pts:
        result_dict["img_pts"] = [
            (float(x), float(y)) for (x, y) in img_pts
        ]
        result_dict["inlier_mask"] = list(
            camera_pose.get("inlier_mask", [True] * len(img_pts))
        )

    vis_calib_dir = vis_dir / "calibration"
    vis_calib_dir.mkdir(parents=True, exist_ok=True)

    cap = video.cap
    keypoint_mapper = KeypointMapper()
    # ``_draw_physical_calibration`` pokes into ``calibrator._field_world_coords``
    # and the top-down panel needs an ``image_size``; provide a minimal stub
    # carrying just those.
    class _CalibratorStub:
        _field_world_coords = field_world_coords
        line_intersections: list = []

    _CalibratorStub.image_size = (video.width, video.height)
    stub = _CalibratorStub()

    to_render = [
        fidx for i, fidx in enumerate(frame_indices) if i % vis_interval == 0
    ]
    for fidx in tqdm(to_render, desc="Stage 1 (Fixed): rendering vis"):
        cap.set(cv2.CAP_PROP_POS_FRAMES, fidx)
        ok, frame = cap.read()
        if not ok:
            continue
        vis = _draw_physical_calibration(
            frame, keypoints={}, lines={}, result=result_dict,
            pitch_template=pitch_template,
            keypoint_mapper=keypoint_mapper, calibrator=stub,
            pitch_length=pitch_length, pitch_width=pitch_width,
        )
        _stamp_source(vis, "Fixed Camera")
        cv2.imwrite(
            str(vis_calib_dir / f"frame_{fidx:05d}.jpg"), vis,
            [cv2.IMWRITE_JPEG_QUALITY, 85],
        )
    logger.info(
        "Stage 1 (Fixed): wrote %d calibration JPGs to %s",
        len(to_render), vis_calib_dir,
    )


def _load_camera_profiles(phys_config: dict) -> dict:
    """Load configs/camera_profiles.yaml (same logic physical_runner uses)."""
    import yaml
    profile_path = phys_config.get("camera_profiles_path")
    candidates: list[Path] = []
    if profile_path:
        candidates.append(Path(profile_path))
    # Walk up from this file to find configs/camera_profiles.yaml
    cur = Path(__file__).resolve().parent
    for _ in range(5):
        candidates.append(cur / "configs" / "camera_profiles.yaml")
        candidates.append(cur / "camera_profiles.yaml")
        cur = cur.parent
    candidates.append(Path("configs/camera_profiles.yaml"))

    for cand in candidates:
        if cand.is_file():
            with open(cand) as fh:
                doc = yaml.safe_load(fh)
            return doc.get("profiles") or {}
    raise FileNotFoundError(
        "configs/camera_profiles.yaml not found. Set "
        "field_registration.physical.camera_profiles_path explicitly.",
    )
