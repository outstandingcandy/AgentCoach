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

        # 4. Pick an annotation frame and solve once.
        # ``annotation_frame_path`` (a full path, set by the scene-setup
        # wizard after the user saves their first calibration frame) wins
        # over ``annotation_frame`` (a bare integer index, legacy). Falls
        # back to auto-picking the most recent annotation in the standard
        # workspace/annotations/<stem>/ dir when neither is set.
        annotations_dir = _resolve_annotations_dir(video_path)
        explicit_path = phys_config.get("annotation_frame_path")
        if explicit_path:
            annotation_path = Path(explicit_path)
            if not annotation_path.exists():
                raise FileNotFoundError(
                    f"annotation_frame_path={explicit_path} does not exist",
                )
        else:
            explicit = phys_config.get("annotation_frame")
            annotation_path = _pick_annotation_frame(
                annotations_dir, int(explicit) if explicit is not None else None,
            )
        logger.info(
            "Stage 1 (Fixed): solving once from %s",
            annotation_path.relative_to(annotation_path.parents[2])
            if len(annotation_path.parents) >= 3 else annotation_path,
        )

        camera_pose, diag = _solve_from_annotation(
            annotation_path,
            img_size=(video.width, video.height),
            physical_cfg=phys_config,
            camera_profiles=camera_profiles,
        )

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
                "fixed_camera_source": str(annotation_path),
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
    pitch_dims = {
        "pitch_length": pitch_length, "pitch_width": pitch_width,
    }
    field_world_coords, _, _ = build_field_template(pitch_dims)
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
