"""Diagnose whether principal point offset (cx/cy) or radial distortion (k1/k2)
improves calibration, bypassing PhysicalCalibrator's rejection threshold.

Runs on multiple frames and compares:
  - 7-DOF: rvec(3), tvec(3), f
  - 9-DOF-CX: rvec(3), tvec(3), f, dcx, dcy
  - 9-DOF-K:  rvec(3), tvec(3), f, k1, k2
  - 11-DOF:   rvec(3), tvec(3), f, dcx, dcy, k1, k2
"""

import sys
import os
import yaml
import cv2
import numpy as np
from scipy.optimize import least_squares

sys.path.insert(0, os.path.dirname(__file__))

from goalinsight.field_registration.pnlcalib import KeypointMapper
from goalinsight.field_registration.keypoint_detector import KeypointDetector
from goalinsight.field_registration.physical_calibrator import build_field_template


def load_config():
    with open("configs/clip_000_physical.yaml") as f:
        cfg = yaml.safe_load(f)
    with open("configs/default.yaml") as f:
        default = yaml.safe_load(f)
    # Merge
    for k, v in default.items():
        if k not in cfg:
            cfg[k] = v
        elif isinstance(v, dict) and isinstance(cfg[k], dict):
            for kk, vv in v.items():
                if kk not in cfg[k]:
                    cfg[k][kk] = vv
    return cfg


def make_detector(cfg):
    """Create and load keypoint detector."""
    pnl_cfg = cfg["field_registration"]["pnlcalib"]
    kp_config = {
        "backend": "pnlcalib",
        "pnlcalib": {
            "weights": pnl_cfg.get("keypoint_weights", "SV_kp"),
            "model_path": pnl_cfg.get("keypoint_model_path"),
            "confidence_threshold": pnl_cfg.get("keypoint_threshold", 0.3434),
        }
    }
    detector = KeypointDetector(kp_config)
    detector.load_model()
    return detector


def detect_keypoints(frame, detector):
    """Detect keypoints using the finetuned model."""
    return detector.detect(frame, convert_to_soccernet=False)


def get_correspondences(keypoints, min_confidence=0.3):
    """Build 2D-3D correspondences from detected keypoints."""
    mapper = KeypointMapper()
    img_pts, world_pts, kp_ids = mapper.build_3d_correspondence_matrix(
        keypoints, filter_by_confidence=min_confidence, exclude_non_ground=False,
    )
    # Override with field template (standard 105x68)
    field_coords, _, _ = build_field_template(105.0, 68.0)
    if len(world_pts) > 0:
        world_pts = world_pts.copy()
        for idx, kid in enumerate(kp_ids):
            if kid < len(field_coords):
                wx, wy = field_coords[kid]
                z = world_pts[idx, 2]
                world_pts[idx] = [wx, wy, z]
    return img_pts.astype(np.float64), world_pts.astype(np.float64), kp_ids


def pnp_ransac_init(world_pts, img_pts, f_init, cx, cy, reproj_thresh=30.0):
    """Multi-focal PnP RANSAC initialization."""
    focal_candidates = [400, 800, 1200, 1600, 2000, 2500, 3000, 3500]
    dist = np.zeros(5, dtype=np.float64)
    best = None
    best_n = -1

    for f in focal_candidates:
        K = np.array([[f, 0, cx], [0, f, cy], [0, 0, 1]], dtype=np.float64)
        for solver in [cv2.SOLVEPNP_SQPNP, cv2.SOLVEPNP_EPNP]:
            ok, rvec, tvec, inliers = cv2.solvePnPRansac(
                world_pts.reshape(-1, 1, 3), img_pts.reshape(-1, 1, 2),
                K, dist, reprojectionError=reproj_thresh, iterationsCount=2000,
                flags=solver,
            )
            if ok and inliers is not None and len(inliers) > best_n:
                best_n = len(inliers)
                best = (rvec.ravel(), tvec.ravel(), f, inliers.ravel())

    if best is None:
        return None
    return best


def optimize_ndof(rvec0, tvec0, f0, img_pts, world_pts, cx0, cy0, mode="7dof",
                  focal_bounds=(400.0, 3500.0), cam_pos=None, pos_weight=200.0):
    """Optimize camera parameters with different DOF configurations.

    Modes:
      7dof:   [rvec(3), tvec(3), f]
      9dof_cx: [rvec(3), tvec(3), f, dcx, dcy]
      9dof_k:  [rvec(3), tvec(3), f, k1, k2]
      11dof:   [rvec(3), tvec(3), f, dcx, dcy, k1, k2]
    """
    x0 = list(rvec0.ravel()) + list(tvec0.ravel()) + [f0]
    lb = [-np.inf]*6 + [focal_bounds[0]]
    ub = [np.inf]*6 + [focal_bounds[1]]

    if mode in ("9dof_cx", "11dof"):
        x0 += [0.0, 0.0]  # dcx, dcy
        lb += [-200.0, -200.0]
        ub += [200.0, 200.0]
    if mode in ("9dof_k", "11dof"):
        x0 += [0.0, 0.0]  # k1, k2
        lb += [-1.0, -1.0]
        ub += [1.0, 1.0]

    x0 = np.array(x0, dtype=np.float64)
    lb = np.array(lb)
    ub = np.array(ub)

    def cost_fn(x):
        rvec = x[:3].reshape(3, 1)
        tvec = x[3:6].reshape(3, 1)
        f = x[6]
        cx = cx0
        cy = cy0
        k1 = k2 = 0.0

        idx = 7
        if mode in ("9dof_cx", "11dof"):
            cx = cx0 + x[idx]
            cy = cy0 + x[idx+1]
            idx += 2
        if mode in ("9dof_k", "11dof"):
            k1 = x[idx]
            k2 = x[idx+1]

        K = np.array([[f, 0, cx], [0, f, cy], [0, 0, 1]], dtype=np.float64)
        dist = np.array([k1, k2, 0, 0, 0], dtype=np.float64)

        proj, _ = cv2.projectPoints(
            world_pts.reshape(-1, 1, 3), rvec, tvec, K, dist,
        )
        residuals = (proj.reshape(-1, 2) - img_pts).ravel()

        all_res = [residuals]

        # Camera position constraint
        if cam_pos is not None:
            R, _ = cv2.Rodrigues(rvec)
            cam_center = -R.T @ tvec.ravel()
            pos_target = np.array(cam_pos)
            n_copies = 50
            w = pos_weight / np.sqrt(n_copies)
            all_res.append(np.tile((cam_center - pos_target) * w, n_copies))

        return np.concatenate(all_res)

    result = least_squares(
        cost_fn, x0, method="trf", bounds=(lb, ub),
        loss="cauchy", f_scale=15.0, max_nfev=500,
    )

    # Extract results
    rvec = result.x[:3]
    tvec = result.x[3:6]
    f = result.x[6]
    cx = cx0
    cy = cy0
    k1 = k2 = 0.0
    idx = 7
    if mode in ("9dof_cx", "11dof"):
        cx = cx0 + result.x[idx]
        cy = cy0 + result.x[idx+1]
        idx += 2
    if mode in ("9dof_k", "11dof"):
        k1 = result.x[idx]
        k2 = result.x[idx+1]

    K = np.array([[f, 0, cx], [0, f, cy], [0, 0, 1]], dtype=np.float64)
    dist = np.array([k1, k2, 0, 0, 0], dtype=np.float64)

    # Compute reprojection error
    proj, _ = cv2.projectPoints(
        world_pts.reshape(-1, 1, 3),
        rvec.reshape(3, 1), tvec.reshape(3, 1), K, dist,
    )
    errors = np.linalg.norm(proj.reshape(-1, 2) - img_pts, axis=1)

    return {
        "mode": mode,
        "f": f, "cx": cx, "cy": cy, "k1": k1, "k2": k2,
        "rvec": rvec, "tvec": tvec,
        "K": K, "dist": dist,
        "mean_err": float(np.mean(errors)),
        "median_err": float(np.median(errors)),
        "max_err": float(np.max(errors)),
        "per_point_errors": errors,
        "cost": result.cost,
        "nfev": result.nfev,
    }


def main():
    cfg = load_config()
    video_path = "data/raw_videos/football_sunday_output_000.mp4"
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cx0 = w / 2.0
    cy0 = h / 2.0

    cam_pos = cfg["field_registration"]["physical"].get("camera_position")

    print("Loading keypoint detector...")
    detector = make_detector(cfg)

    # Test frames: include 1860 plus some frames that typically calibrate well
    test_frames = [60, 300, 600, 900, 1200, 1500, 1860]
    modes = ["7dof", "9dof_cx", "9dof_k", "11dof"]

    print(f"Video: {w}x{h} @ {fps:.1f}fps, {total} frames")
    print(f"Image center: ({cx0}, {cy0})")
    print(f"Camera position constraint: {cam_pos}")
    print(f"Testing frames: {test_frames}")
    print(f"Modes: {modes}")
    print("=" * 100)

    all_results = {}

    for frame_idx in test_frames:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret:
            print(f"\nFrame {frame_idx}: FAILED TO READ")
            continue

        print(f"\n{'='*100}")
        print(f"Frame {frame_idx}")
        print(f"{'='*100}")

        # Detect keypoints
        keypoints = detect_keypoints(frame, detector)
        n_detected = len([k for k in keypoints if k.get("confidence", 0) >= 0.3])
        print(f"  Detected keypoints (conf>=0.3): {n_detected}")

        if n_detected < 6:
            print(f"  SKIP: too few keypoints")
            continue

        # Get correspondences
        img_pts, world_pts, kp_ids = get_correspondences(keypoints)
        print(f"  Correspondences: {len(kp_ids)} (kp_ids: {sorted(kp_ids)[:10]}...)")

        # PnP RANSAC init
        init = pnp_ransac_init(world_pts, img_pts, 1500, cx0, cy0)
        if init is None:
            print(f"  SKIP: PnP RANSAC failed")
            continue

        rvec0, tvec0, f_best, inlier_idx = init
        print(f"  PnP RANSAC: f={f_best:.0f}, {len(inlier_idx)}/{len(kp_ids)} inliers")

        # Use only RANSAC inliers for optimization
        inlier_mask = np.zeros(len(kp_ids), dtype=bool)
        inlier_mask[inlier_idx] = True
        img_in = img_pts[inlier_mask]
        world_in = world_pts[inlier_mask]
        kp_in = [kp_ids[i] for i in range(len(kp_ids)) if inlier_mask[i]]

        frame_results = {}
        for mode in modes:
            res = optimize_ndof(
                rvec0, tvec0, f_best, img_in, world_in, cx0, cy0,
                mode=mode, cam_pos=cam_pos,
            )
            frame_results[mode] = res

        # Print comparison table
        print(f"\n  {'Mode':<12} {'MeanErr':>8} {'MedErr':>8} {'MaxErr':>8}  "
              f"{'f':>7} {'dcx':>7} {'dcy':>7} {'k1':>9} {'k2':>9}  {'Cost':>10}")
        print(f"  {'-'*12} {'-'*8} {'-'*8} {'-'*8}  "
              f"{'-'*7} {'-'*7} {'-'*7} {'-'*9} {'-'*9}  {'-'*10}")
        for mode in modes:
            r = frame_results[mode]
            dcx = r["cx"] - cx0
            dcy = r["cy"] - cy0
            print(f"  {mode:<12} {r['mean_err']:8.2f} {r['median_err']:8.2f} {r['max_err']:8.2f}  "
                  f"{r['f']:7.1f} {dcx:+7.1f} {dcy:+7.1f} {r['k1']:+9.5f} {r['k2']:+9.5f}  "
                  f"{r['cost']:10.1f}")

        # Camera position comparison
        print(f"\n  Camera positions:")
        for mode in modes:
            r = frame_results[mode]
            R, _ = cv2.Rodrigues(r["rvec"].reshape(3, 1))
            cam = -R.T @ r["tvec"].ravel()
            print(f"    {mode:<12}: ({cam[0]:+7.2f}, {cam[1]:+7.2f}, {cam[2]:+7.2f})")

        all_results[frame_idx] = frame_results

    cap.release()

    # Summary across all frames
    print(f"\n\n{'='*100}")
    print("SUMMARY: Mean reprojection error across frames")
    print(f"{'='*100}")
    print(f"  {'Frame':<8}", end="")
    for mode in modes:
        print(f" {mode:>12}", end="")
    print(f"  {'Best':>12}")

    for fidx in sorted(all_results.keys()):
        fr = all_results[fidx]
        errs = {m: fr[m]["median_err"] for m in modes}
        best = min(errs, key=errs.get)
        print(f"  {fidx:<8}", end="")
        for mode in modes:
            marker = " *" if mode == best else "  "
            print(f" {errs[mode]:10.2f}{marker}", end="")
        print(f"  {best:>12}")

    # Aggregate dcx, dcy, k1, k2 across successful frames
    print(f"\n  Mean parameter offsets across frames:")
    for mode in ["9dof_cx", "9dof_k", "11dof"]:
        dcxs, dcys, k1s, k2s = [], [], [], []
        for fidx in sorted(all_results.keys()):
            r = all_results[fidx][mode]
            if mode in ("9dof_cx", "11dof"):
                dcxs.append(r["cx"] - cx0)
                dcys.append(r["cy"] - cy0)
            if mode in ("9dof_k", "11dof"):
                k1s.append(r["k1"])
                k2s.append(r["k2"])
        parts = [f"{mode:<12}:"]
        if dcxs:
            parts.append(f"dcx={np.mean(dcxs):+.2f}±{np.std(dcxs):.2f}")
            parts.append(f"dcy={np.mean(dcys):+.2f}±{np.std(dcys):.2f}")
        if k1s:
            parts.append(f"k1={np.mean(k1s):+.6f}±{np.std(k1s):.6f}")
            parts.append(f"k2={np.mean(k2s):+.6f}±{np.std(k2s):.6f}")
        print(f"    {' '.join(parts)}")


if __name__ == "__main__":
    main()
