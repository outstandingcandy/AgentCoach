#!/usr/bin/env python3
"""Recover physical pitch dimensions from manual annotation JSONs.

Reads one or more annotation JSONs (produced by the workspace annotator at
``/annotate`` — ``python -m goalinsight.web``), all assumed to be on the
same physical field, and jointly fits per-frame camera pose and global
pitch dimensions via robust nonlinear least squares.

Output is printed to the console — the user reviews the recovered dims and
manually updates ``configs/kids_soccer.yaml``. Nothing is written to disk.

Usage:
    python scripts/fit_pitch_dims.py \
        -c configs/kids_soccer.yaml \
        output/annotations/kids_soccer_match/frame_64020.json \
        output/annotations/kids_soccer_clip_1250_1310/frame_246.json \
        output/annotations/kids_soccer_clip_1250_1310/frame_338.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import yaml
from scipy.optimize import least_squares

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from goalinsight.annotation.homography import (
    line_intersection,
    solve_camera,
)
from goalinsight.annotation.pitch.geometry import FIFA_DEFAULTS, SoccerPitch
from goalinsight.annotation.pitch_constants import _build_lines_for_pitch


# Generous physical bounds for an arbitrary outdoor pitch.
DIM_BOUNDS: dict[str, tuple[float, float]] = {
    "pitch_length": (30.0, 110.0),
    "pitch_width": (20.0, 70.0),
    "goal_length": (4.0, 12.0),
    "goal_height": (1.5, 3.0),
    "penalty_area_length": (5.0, 18.0),
    "penalty_area_width": (15.0, 45.0),
    "goal_area_length": (2.0, 8.0),
    "goal_area_width": (8.0, 22.0),
}


def _line_constrains(line_name: str) -> set[str]:
    """Which pitch dims a given line endpoint depends on."""
    if line_name in {"touchline_top", "touchline_bottom"}:
        return {"pitch_length", "pitch_width"}
    if line_name in {"goal_line_left", "goal_line_right"}:
        return {"pitch_length", "pitch_width"}
    if line_name == "center_line":
        return {"pitch_width"}
    if line_name.startswith("penalty_") and line_name.endswith("_front"):
        return {"pitch_length", "penalty_area_length", "penalty_area_width"}
    if line_name.startswith("penalty_") and line_name.endswith(("_top", "_bottom")):
        return {"pitch_length", "penalty_area_length", "penalty_area_width"}
    if line_name.startswith("goal_area_") and line_name.endswith("_front"):
        return {"pitch_length", "goal_area_length", "goal_area_width"}
    if line_name.startswith("goal_area_") and line_name.endswith(("_top", "_bottom")):
        return {"pitch_length", "goal_area_length", "goal_area_width"}
    return set()


def _point_constrains(name: str) -> set[str]:
    """Which dims a given keypoint constrains (best-effort)."""
    if name.endswith("_PITCH_CORNER"):
        return {"pitch_length", "pitch_width"}
    if "_GOAL_TL_POST" in name or "_GOAL_TR_POST" in name:
        return {"pitch_length", "goal_length", "goal_height"}
    if "_GOAL_BL_POST" in name or "_GOAL_BR_POST" in name:
        return {"pitch_length", "goal_length"}
    if "PENALTY_AREA" in name:
        return {"pitch_length", "penalty_area_length", "penalty_area_width"}
    if "GOAL_AREA" in name:
        return {"pitch_length", "goal_area_length", "goal_area_width"}
    if name == "CENTER_MARK":
        return set()  # always at origin
    if "PENALTY_MARK" in name:
        return {"pitch_length", "goal_line_to_penalty_mark"}
    if "TOUCH_AND_HALFWAY" in name:
        return {"pitch_width"}
    return set()


def _detect_constrained_dims(frames: list[dict]) -> list[str]:
    """Auto-detect which dims have at least one observation across all frames."""
    constrained: set[str] = set()
    for f in frames:
        for line in f["lines"]:
            constrained.update(_line_constrains(line["name"]))
        for pt in f["points"]:
            constrained.update(_point_constrains(pt["keypoint_name"]))
    return [k for k in DIM_BOUNDS if k in constrained]


def _cam_z_from_pose(rvec: np.ndarray, tvec: np.ndarray) -> float:
    R, _ = cv2.Rodrigues(rvec.reshape(3, 1))
    cam = -R.T @ tvec.reshape(3, 1)
    return float(cam[2, 0])


def _lookat_pose(
    cam_world: tuple[float, float, float],
    target_world: tuple[float, float, float],
) -> tuple[np.ndarray, np.ndarray]:
    """OpenCV camera (z fwd, y down) looking at target. World z<0 = up.

    R rows = [right_world, down_world, fwd_world]; R takes world->cam.
    """
    cam = np.asarray(cam_world, dtype=np.float64)
    tgt = np.asarray(target_world, dtype=np.float64)
    fwd = tgt - cam
    fwd /= np.linalg.norm(fwd)
    world_down = np.array([0.0, 0.0, 1.0])  # z>0 is below ground = world-down
    right = np.cross(fwd, world_down)
    if np.linalg.norm(right) < 1e-6:
        right = np.array([1.0, 0.0, 0.0])
    right /= np.linalg.norm(right)
    down = np.cross(right, fwd)
    R = np.stack([right, down, fwd], axis=0)
    rvec, _ = cv2.Rodrigues(R)
    tvec = -R @ cam
    return rvec.reshape(3), tvec.reshape(3)


def _init_pose_for_frame(
    frame: dict,
    pitch: SoccerPitch,
    img_size: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray] | None:
    """PnP init using manual 3D points + 2D line-line intersections.

    If the direct PnP solution puts the camera below ground (cam_z >= 0 in our
    z-down-positive convention), fall back to a look-at sweep over plausible
    sideline positions to break the mirror ambiguity.
    """
    pixel_pts: list[tuple[float, float]] = []
    world_3d_pts: list[tuple[float, float, float]] = []

    # Manual keypoints (may be 3D — goal posts at z<0).
    for pt in frame["points"]:
        name = pt["keypoint_name"]
        wp = pitch.point_dict.get(name)
        if wp is None or not np.all(np.isfinite(wp)):
            continue
        pixel_pts.append(tuple(pt["pixel"]))
        world_3d_pts.append((float(wp[0]), float(wp[1]), float(wp[2])))

    # Line-line intersections: pixel from user clicks, world from current pitch.
    lines2d = _build_lines_for_pitch(pitch)
    line_pixels = {l["name"]: l["pixels"] for l in frame["lines"]}
    line_names = list(line_pixels.keys())
    for i, n1 in enumerate(line_names):
        for n2 in line_names[i + 1:]:
            if n1 not in lines2d or n2 not in lines2d:
                continue
            (a1, a2) = line_pixels[n1]
            (b1, b2) = line_pixels[n2]
            px = line_intersection((tuple(a1), tuple(a2)), (tuple(b1), tuple(b2)))
            if px is None:
                continue
            wx = line_intersection(lines2d[n1], lines2d[n2])
            if wx is None:
                continue
            if abs(wx[0]) > pitch.PITCH_LENGTH or abs(wx[1]) > pitch.PITCH_WIDTH:
                continue
            # Skip if the intersection isn't roughly on the user-clicked segments
            # (parallel-ish lines yield far-flung pixel intersections).
            if not (-img_size[0] * 2 < px[0] < img_size[0] * 3
                    and -img_size[1] * 2 < px[1] < img_size[1] * 3):
                continue
            pixel_pts.append(px)
            world_3d_pts.append((float(wx[0]), float(wx[1]), 0.0))

    if len(pixel_pts) < 4:
        return None

    # Direct PnP first.
    K = _build_intrinsics(img_size)
    obj = np.array(world_3d_pts, dtype=np.float64).reshape(-1, 1, 3)
    img_arr = np.array(pixel_pts, dtype=np.float64).reshape(-1, 1, 2)

    cam_direct, _, _ = solve_camera(pixel_pts, world_3d_pts, img_size)
    rvec_direct: np.ndarray | None = None
    tvec_direct: np.ndarray | None = None
    if cam_direct is not None:
        rv, _ = cv2.Rodrigues(cam_direct.rotation)
        tv = -cam_direct.rotation @ cam_direct.position
        rvec_direct, tvec_direct = rv.reshape(3, 1), tv.reshape(3, 1)
        if _cam_z_from_pose(rvec_direct, tvec_direct) < 0:
            return rvec_direct.reshape(3), tvec_direct.reshape(3)

    # Direct solve is underground — mirror-flip + look-at sweep to escape the
    # planar mirror basin. solve_camera already runs a sweep internally, but
    # this one is parameterized by the *current trial pitch* (varies during
    # joint dim+pose fitting), so we run it here too.
    candidates: list[tuple[float, np.ndarray, np.ndarray]] = []

    def _try_init(rvec0: np.ndarray, tvec0: np.ndarray) -> None:
        try:
            success, rv, tv = cv2.solvePnP(
                obj, img_arr, K, None,
                rvec=rvec0.reshape(3, 1).copy(),
                tvec=tvec0.reshape(3, 1).copy(),
                useExtrinsicGuess=True,
                flags=cv2.SOLVEPNP_ITERATIVE,
            )
        except cv2.error:
            return
        if not success or _cam_z_from_pose(rv, tv) >= 0:
            return
        proj, _ = cv2.projectPoints(obj, rv, tv, K, None)
        err = np.linalg.norm(
            proj.reshape(-1, 2) - np.array(pixel_pts), axis=1,
        ).mean()
        candidates.append((err, rv.reshape(3), tv.reshape(3)))

    if rvec_direct is not None and tvec_direct is not None:
        R_bad, _ = cv2.Rodrigues(rvec_direct)
        cam_bad = -R_bad.T @ tvec_direct.reshape(3, 1)
        cam_mirror = np.array([cam_bad[0, 0], cam_bad[1, 0], -cam_bad[2, 0]])
        rvec0, tvec0 = _lookat_pose(tuple(cam_mirror), (0.0, 0.0, 0.0))
        _try_init(rvec0, tvec0)

    hL = pitch.PITCH_LENGTH / 2.0
    hW = pitch.PITCH_WIDTH / 2.0
    for cam_y in (-hW - 5.0, -hW - 1.0, hW + 1.0, hW + 5.0):
        for cam_x in (-hL * 0.5, 0.0, hL * 0.5):
            for tgt_x in (-hL * 0.7, 0.0, hL * 0.7):
                rvec0, tvec0 = _lookat_pose(
                    (cam_x, cam_y, -3.0), (tgt_x, 0.0, 0.0),
                )
                _try_init(rvec0, tvec0)

    if not candidates:
        return None
    candidates.sort(key=lambda c: c[0])
    return candidates[0][1], candidates[0][2]


def _build_intrinsics_from_fx(
    fx: float, img_size: tuple[int, int],
) -> np.ndarray:
    cx, cy = img_size[0] / 2.0, img_size[1] / 2.0
    return np.array([[fx, 0, cx], [0, fx, cy], [0, 0, 1]], dtype=np.float64)


def _build_intrinsics(img_size: tuple[int, int]) -> np.ndarray:
    return _build_intrinsics_from_fx(float(img_size[0]), img_size)


def _build_residuals(
    x: np.ndarray,
    frames: list[dict],
    fixed_dims: dict[str, float],
    dim_keys: list[str],
    img_size: tuple[int, int],
    fit_fx: bool,
    frame_to_group: list[int],
) -> np.ndarray:
    n = len(dim_keys)
    dims = dict(fixed_dims)
    for i, k in enumerate(dim_keys):
        dims[k] = float(x[i])
    pitch = SoccerPitch(**dims)
    lines2d = _build_lines_for_pitch(pitch)
    if fit_fx:
        fx = float(x[n])
        K = _build_intrinsics_from_fx(fx, img_size)
        pose_off = n + 1
    else:
        K = _build_intrinsics(img_size)
        pose_off = n

    res: list[float] = []
    for fi, frame in enumerate(frames):
        gi = frame_to_group[fi]
        rvec = x[pose_off + 6 * gi: pose_off + 6 * gi + 3].reshape(3, 1)
        tvec = x[pose_off + 6 * gi + 3: pose_off + 6 * gi + 6].reshape(3, 1)

        # Mirror penalty: cam_z >= 0 means camera underground (z-down-positive
        # convention with z<0 = up). Always emit one residual per frame for
        # fixed length; it's 0 when cam is above ground.
        R, _ = cv2.Rodrigues(rvec)
        cam_w = -R.T @ tvec
        res.append(500.0 * max(0.0, cam_w[2, 0] + 0.3))

        # Point residuals (2D pixel error).
        for pt in frame["points"]:
            wp = pitch.point_dict.get(pt["keypoint_name"])
            if wp is None or not np.all(np.isfinite(wp)):
                continue
            obj = np.array([[wp[0], wp[1], wp[2]]], dtype=np.float64).reshape(-1, 1, 3)
            proj, _ = cv2.projectPoints(obj, rvec, tvec, K, None)
            px = np.asarray(pt["pixel"], dtype=np.float64)
            err = proj.reshape(-1) - px
            res.extend(err.tolist())

        # Line residuals (perpendicular pixel distance).
        for line in frame["lines"]:
            seg = lines2d.get(line["name"])
            if seg is None:
                continue
            (wx1, wy1), (wx2, wy2) = seg
            obj = np.array([[wx1, wy1, 0.0], [wx2, wy2, 0.0]], dtype=np.float64).reshape(-1, 1, 3)
            proj, _ = cv2.projectPoints(obj, rvec, tvec, K, None)
            p1 = proj[0, 0]
            p2 = proj[1, 0]
            d = p2 - p1
            d_len = float(np.hypot(d[0], d[1]))
            if d_len < 1e-6:
                # Degenerate: line collapses on image plane; emit large penalty.
                res.extend([1e3, 1e3])
                continue
            nrm = np.array([-d[1], d[0]]) / d_len
            for click in line["pixels"]:
                v = np.asarray(click, dtype=np.float64) - p1
                res.append(float(np.dot(v, nrm)))
    return np.asarray(res, dtype=np.float64)


def _residual_labels(frames: list[dict]) -> list[str]:
    labels: list[str] = []
    for frame in frames:
        tag = f"{frame['video_name']}/f{frame['frame_idx']}"
        labels.append(f"{tag} mirror_penalty")
        for pt in frame["points"]:
            labels.append(f"{tag} pt:{pt['keypoint_name']}.x")
            labels.append(f"{tag} pt:{pt['keypoint_name']}.y")
        for line in frame["lines"]:
            for click_idx, _ in enumerate(line["pixels"]):
                labels.append(f"{tag} line:{line['name']}#{click_idx}")
    return labels


def _frame_rmse(
    frame: dict,
    rvec: np.ndarray,
    tvec: np.ndarray,
    pitch: SoccerPitch,
    K: np.ndarray,
) -> tuple[float, int]:
    lines2d = _build_lines_for_pitch(pitch)
    sq: list[float] = []

    for pt in frame["points"]:
        wp = pitch.point_dict.get(pt["keypoint_name"])
        if wp is None or not np.all(np.isfinite(wp)):
            continue
        obj = np.array([[wp[0], wp[1], wp[2]]], dtype=np.float64).reshape(-1, 1, 3)
        proj, _ = cv2.projectPoints(obj, rvec.reshape(3, 1), tvec.reshape(3, 1), K, None)
        px = np.asarray(pt["pixel"], dtype=np.float64)
        sq.append(float(np.sum((proj.reshape(-1) - px) ** 2)))

    for line in frame["lines"]:
        seg = lines2d.get(line["name"])
        if seg is None:
            continue
        (wx1, wy1), (wx2, wy2) = seg
        obj = np.array([[wx1, wy1, 0.0], [wx2, wy2, 0.0]], dtype=np.float64).reshape(-1, 1, 3)
        proj, _ = cv2.projectPoints(obj, rvec.reshape(3, 1), tvec.reshape(3, 1), K, None)
        p1 = proj[0, 0]
        p2 = proj[1, 0]
        d = p2 - p1
        d_len = float(np.hypot(d[0], d[1]))
        if d_len < 1e-6:
            continue
        nrm = np.array([-d[1], d[0]]) / d_len
        for click in line["pixels"]:
            v = np.asarray(click, dtype=np.float64) - p1
            sq.append(float(np.dot(v, nrm)) ** 2)

    if not sq:
        return float("nan"), 0
    return float(np.sqrt(np.mean(sq))), len(sq)


def _load_yaml_pitch(cfg_path: Path) -> dict[str, float]:
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f) or {}
    pitch_cfg = cfg.get("pitch") or {}
    out = dict(FIFA_DEFAULTS)
    for k, v in pitch_cfg.items():
        if k in FIFA_DEFAULTS:
            out[k] = float(v)
    return out


def _load_annotation(path: Path) -> dict:
    with open(path) as f:
        data = json.load(f)
    return {
        "frame_idx": data["frame_idx"],
        "video_name": data["video_name"],
        "points": data.get("points", []),
        "lines": data.get("lines", []),
    }


def _parse_img_size(s: str) -> tuple[int, int]:
    w, h = s.lower().split("x")
    return int(w), int(h)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n", 1)[0])
    parser.add_argument("annotations", nargs="+", type=Path,
                        help="One or more annotation JSON files")
    parser.add_argument("-c", "--config", type=Path, required=True,
                        help="YAML config providing initial pitch dims")
    parser.add_argument("--img-size", type=_parse_img_size, default=(1920, 1080),
                        help="WxH (default 1920x1080)")
    parser.add_argument("--max-iter", type=int, default=200)
    parser.add_argument("--fit-fx", action="store_true",
                        help="Also fit fx (shared focal length across all frames). "
                             "Default fx = img_width.")
    parser.add_argument("--fx-bounds", type=str, default="800,4000",
                        help="lo,hi pixel bounds for fx (default 800,4000)")
    args = parser.parse_args()
    fx_lo, fx_hi = (float(v) for v in args.fx_bounds.split(","))

    init_dims = _load_yaml_pitch(args.config)
    frames = [_load_annotation(p) for p in args.annotations]
    print(f"Loaded {len(frames)} annotation frame(s):")
    for f in frames:
        print(f"  {f['video_name']}/f{f['frame_idx']}: "
              f"{len(f['points'])} pts, {len(f['lines'])} lines")
    print()

    dim_keys = _detect_constrained_dims(frames)
    print(f"Free pitch dims (constrained by annotations): {dim_keys}")
    fixed_dims = {k: v for k, v in init_dims.items() if k not in dim_keys}
    print(f"Frozen pitch dims (no constraint, kept at YAML init): "
          f"{sorted(fixed_dims.keys())}")
    print()

    # Group frames by video — same video means same physical camera pose.
    group_for_video: dict[str, int] = {}
    frame_to_group: list[int] = []
    for f in frames:
        v = f["video_name"]
        if v not in group_for_video:
            group_for_video[v] = len(group_for_video)
        frame_to_group.append(group_for_video[v])
    n_groups = len(group_for_video)
    print(f"Pose groups: {n_groups} ({list(group_for_video.keys())})")
    print()

    K = _build_intrinsics(args.img_size)
    init_pitch = SoccerPitch(**init_dims)
    init_poses_per_frame: list[tuple[np.ndarray, np.ndarray]] = []
    print("Per-frame initial pose (after mirror-resolve):")
    for frame in frames:
        pose = _init_pose_for_frame(frame, init_pitch, args.img_size)
        if pose is None:
            print(f"ERROR: PnP init failed for "
                  f"{frame['video_name']}/f{frame['frame_idx']}")
            return 1
        init_poses_per_frame.append(pose)
        rvec, tvec = pose
        R0, _ = cv2.Rodrigues(rvec.reshape(3, 1))
        cam0 = -R0.T @ tvec.reshape(3, 1)
        print(f"  {frame['video_name']}/f{frame['frame_idx']}: "
              f"cam=({cam0[0,0]:+.1f},{cam0[1,0]:+.1f},{cam0[2,0]:+.1f})")
    print()

    # Per-group initial pose: pick the frame in this group with lowest
    # reprojection error under its own init.
    init_group_poses: list[tuple[np.ndarray, np.ndarray]] = []
    for gi in range(n_groups):
        best_rmse = float("inf")
        best_pose = None
        for fi in range(len(frames)):
            if frame_to_group[fi] != gi:
                continue
            rvec, tvec = init_poses_per_frame[fi]
            rmse, _ = _frame_rmse(frames[fi], rvec, tvec, init_pitch, K)
            if rmse < best_rmse:
                best_rmse = rmse
                best_pose = (rvec, tvec)
        assert best_pose is not None
        init_group_poses.append(best_pose)

    # x = [dim_keys..., (fx if fit_fx), (rvec, tvec) per pose-group].
    n_dims = len(dim_keys)
    n_extra = 1 if args.fit_fx else 0
    x0 = np.zeros(n_dims + n_extra + 6 * n_groups, dtype=np.float64)
    lo = np.full_like(x0, -np.inf)
    hi = np.full_like(x0, np.inf)
    for i, k in enumerate(dim_keys):
        x0[i] = init_dims[k]
        lo[i], hi[i] = DIM_BOUNDS[k]
    if args.fit_fx:
        x0[n_dims] = float(args.img_size[0])
        lo[n_dims], hi[n_dims] = fx_lo, fx_hi
    pose_off = n_dims + n_extra
    for gi, (rvec, tvec) in enumerate(init_group_poses):
        x0[pose_off + 6 * gi: pose_off + 6 * gi + 3] = rvec
        x0[pose_off + 6 * gi + 3: pose_off + 6 * gi + 6] = tvec

    # Per-frame initial RMSE.
    print("Initial reprojection RMSE (with YAML dims + PnP-init pose):")
    init_residual = _build_residuals(x0, frames, fixed_dims, dim_keys, args.img_size, args.fit_fx, frame_to_group)
    init_rmse = float(np.sqrt(np.mean(init_residual ** 2))) if init_residual.size else 0.0
    print(f"  Combined: {init_rmse:.3f} px ({len(init_residual)} residuals)")
    for fi, frame in enumerate(frames):
        gi = frame_to_group[fi]
        rvec, tvec = init_group_poses[gi]
        rmse, n_obs = _frame_rmse(frame, rvec, tvec, init_pitch, K)
        print(f"  {frame['video_name']}/f{frame['frame_idx']}: "
              f"{rmse:.3f} px ({n_obs} residuals)")
    print()

    label = "pose+dims+fx" if args.fit_fx else "pose+dims"
    print(f"Solving joint {label} (TRF / Cauchy / f_scale=5)...")
    result = least_squares(
        _build_residuals,
        x0,
        args=(frames, fixed_dims, dim_keys, args.img_size, args.fit_fx, frame_to_group),
        bounds=(lo, hi),
        method="trf",
        loss="cauchy",
        f_scale=5.0,
        max_nfev=args.max_iter * (len(x0) + 1),
        verbose=0,
    )
    print(f"  status:    {result.status} ({result.message})")
    print(f"  nfev:      {result.nfev}")
    print(f"  final cost (0.5*||r||^2 with cauchy): {result.cost:.4f}")
    print()

    final_residual = _build_residuals(
        result.x, frames, fixed_dims, dim_keys, args.img_size, args.fit_fx, frame_to_group,
    )
    final_rmse = float(np.sqrt(np.mean(final_residual ** 2)))
    print(f"Final unweighted reprojection RMSE (combined): {final_rmse:.3f} px")
    print()

    recovered_dims = dict(fixed_dims)
    for i, k in enumerate(dim_keys):
        recovered_dims[k] = float(result.x[i])
    final_pitch = SoccerPitch(**recovered_dims)

    if args.fit_fx:
        fx_recovered = float(result.x[n_dims])
        K_final = _build_intrinsics_from_fx(fx_recovered, args.img_size)
        hfov = 2.0 * np.degrees(np.arctan(0.5 * args.img_size[0] / fx_recovered))
        flag = ""
        if abs(fx_recovered - fx_lo) < 1e-3 or abs(fx_recovered - fx_hi) < 1e-3:
            flag = "  (at bound!)"
        print(f"Recovered fx: {fx_recovered:.2f} px  "
              f"(initial {args.img_size[0]:.0f}, hfov {hfov:.1f}°){flag}")
        print()
    else:
        K_final = K

    print("Per-frame final RMSE:")
    for fi, frame in enumerate(frames):
        gi = frame_to_group[fi]
        rvec = result.x[pose_off + 6 * gi: pose_off + 6 * gi + 3]
        tvec = result.x[pose_off + 6 * gi + 3: pose_off + 6 * gi + 6]
        rmse, n_obs = _frame_rmse(frame, rvec, tvec, final_pitch, K_final)
        cam_pos = -np.linalg.inv(cv2.Rodrigues(rvec.reshape(3, 1))[0]) @ tvec.reshape(3, 1)
        print(f"  {frame['video_name']}/f{frame['frame_idx']}: "
              f"{rmse:.3f} px ({n_obs} residuals)  "
              f"cam=({cam_pos[0,0]:+.1f},{cam_pos[1,0]:+.1f},{cam_pos[2,0]:+.1f})")
    print()

    print(f"{'DIMENSION':<32}{'INITIAL':>10}{'RECOVERED':>14}{'DELTA':>12}")
    print("-" * 68)
    for k in DIM_BOUNDS:
        if k not in dim_keys:
            continue
        v0 = init_dims[k]
        v1 = recovered_dims[k]
        flag = ""
        lo_b, hi_b = DIM_BOUNDS[k]
        if abs(v1 - lo_b) < 1e-3 or abs(v1 - hi_b) < 1e-3:
            flag = "  (at bound!)"
        print(f"{k:<32}{v0:>10.3f}{v1:>14.3f}{v1 - v0:>+12.3f}{flag}")
    if fixed_dims:
        print()
        print("Frozen (kept at YAML init):")
        for k, v in sorted(fixed_dims.items()):
            print(f"  {k:<32}{v:>10.3f}")
    print()

    # Worst per-residual list.
    labels = _residual_labels(frames)
    if len(labels) == final_residual.size:
        order = np.argsort(np.abs(final_residual))[::-1][:8]
        print("Top residuals (sorted by |error|):")
        for idx in order:
            print(f"  {labels[idx]:<60} {final_residual[idx]:>+8.2f} px")
        print()

    print("Suggested YAML block (drop into configs/kids_soccer.yaml):")
    print("pitch:")
    for k in [
        "pitch_length", "pitch_width",
        "penalty_area_width", "penalty_area_length",
        "goal_area_width", "goal_area_length",
        "goal_line_to_penalty_mark", "center_circle_radius",
        "goal_height", "goal_length",
    ]:
        v = recovered_dims[k]
        marker = " " if k in dim_keys else " #"
        print(f" {marker}{k}: {v:.2f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
