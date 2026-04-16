"""Find optimal field dimensions across ALL frames.

For each (length, width) candidate, run 7-DOF optimization on every frame
and compute aggregate error. The dimensions that minimize total error across
all frames are the most likely true field size.

Uses weak position constraint (weight=20) to avoid over-constraining.
"""

import sys, os, yaml, cv2
import numpy as np
from scipy.optimize import least_squares

sys.path.insert(0, os.path.dirname(__file__))

from goalinsight.field_registration.keypoint_detector import KeypointDetector
from goalinsight.field_registration.pnlcalib import KeypointMapper
from goalinsight.field_registration.physical_calibrator import build_field_template


def load_config():
    with open("configs/clip_000_physical.yaml") as f:
        cfg = yaml.safe_load(f)
    with open("configs/default.yaml") as f:
        default = yaml.safe_load(f)
    for k, v in default.items():
        if k not in cfg:
            cfg[k] = v
        elif isinstance(v, dict) and isinstance(cfg[k], dict):
            for kk, vv in v.items():
                if kk not in cfg[k]:
                    cfg[k][kk] = vv
    return cfg


def make_detector(cfg):
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


def get_correspondences(keypoints, pitch_length, pitch_width, min_conf=0.3):
    mapper = KeypointMapper()
    img_pts, world_pts, kp_ids = mapper.build_3d_correspondence_matrix(
        keypoints, filter_by_confidence=min_conf, exclude_non_ground=False,
    )
    field_coords, _, _ = build_field_template(pitch_length, pitch_width)
    if len(world_pts) > 0:
        world_pts = world_pts.copy()
        for idx, kid in enumerate(kp_ids):
            if kid < len(field_coords):
                wx, wy = field_coords[kid]
                z = world_pts[idx, 2]
                world_pts[idx] = [wx, wy, z]
    return img_pts.astype(np.float64), world_pts.astype(np.float64), kp_ids


def optimize_7dof(img_pts, world_pts, cx, cy, focal_bounds=(400, 3500),
                  cam_pos=None, pos_weight=20.0):
    dist = np.zeros(5, dtype=np.float64)
    best = None
    best_n = -1
    for f in [400, 800, 1200, 1600, 2000, 2500, 3000, 3500]:
        K = np.array([[f, 0, cx], [0, f, cy], [0, 0, 1]], dtype=np.float64)
        for solver in [cv2.SOLVEPNP_SQPNP, cv2.SOLVEPNP_EPNP]:
            ok, rv, tv, inl = cv2.solvePnPRansac(
                world_pts.reshape(-1, 1, 3), img_pts.reshape(-1, 1, 2),
                K, dist, reprojectionError=50.0, iterationsCount=3000, flags=solver,
            )
            if ok and inl is not None and len(inl) > best_n:
                best_n = len(inl)
                best = (rv.ravel(), tv.ravel(), f)
    if best is None:
        return None

    rv0, tv0, f0 = best
    x0 = np.concatenate([rv0, tv0, [f0]])

    def cost(x):
        rv = x[:3].reshape(3, 1)
        tv = x[3:6].reshape(3, 1)
        f = x[6]
        K = np.array([[f, 0, cx], [0, f, cy], [0, 0, 1]], dtype=np.float64)
        proj, _ = cv2.projectPoints(world_pts.reshape(-1, 1, 3), rv, tv, K, dist)
        res = [(proj.reshape(-1, 2) - img_pts).ravel()]
        if cam_pos is not None:
            R, _ = cv2.Rodrigues(rv)
            cc = -R.T @ tv.ravel()
            n_copies = 50
            w = pos_weight / np.sqrt(n_copies)
            res.append(np.tile((cc - np.array(cam_pos)) * w, n_copies))
        return np.concatenate(res)

    result = least_squares(cost, x0, method="trf",
                           bounds=([-np.inf]*6 + [focal_bounds[0]], [np.inf]*6 + [focal_bounds[1]]),
                           loss="cauchy", f_scale=15.0, max_nfev=500)

    rv_opt = result.x[:3]
    tv_opt = result.x[3:6]
    f_opt = result.x[6]
    K_opt = np.array([[f_opt, 0, cx], [0, f_opt, cy], [0, 0, 1]], dtype=np.float64)
    proj, _ = cv2.projectPoints(world_pts.reshape(-1, 1, 3),
                                rv_opt.reshape(3, 1), tv_opt.reshape(3, 1), K_opt, dist)
    errs = np.linalg.norm(proj.reshape(-1, 2) - img_pts, axis=1)
    return rv_opt, tv_opt, f_opt, errs


def main():
    cfg = load_config()
    print("Loading detector...")
    detector = make_detector(cfg)

    video_path = "data/raw_videos/football_sunday_output_000.mp4"
    cap = cv2.VideoCapture(video_path)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cx0, cy0 = w / 2.0, h / 2.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    cam_pos = [0.0, -32.0, 6.3]

    # Sample more frames for robust estimation
    test_frames = list(range(0, min(total_frames, 1870), 60))
    print(f"Video: {w}x{h}, {total_frames} frames")
    print(f"Sampling {len(test_frames)} frames (every 60)")
    print(f"Weak position constraint: {cam_pos}, weight=20")

    # Pre-detect keypoints for all frames
    print("\nDetecting keypoints...")
    frame_keypoints = {}
    for fidx in test_frames:
        cap.set(cv2.CAP_PROP_POS_FRAMES, fidx)
        ret, frame = cap.read()
        if not ret:
            continue
        kps = detector.detect(frame, convert_to_soccernet=False)
        n_good = len([k for k in kps if k.get("confidence", 0) >= 0.3])
        if n_good >= 6:
            frame_keypoints[fidx] = kps
    cap.release()
    print(f"  Frames with >=6 keypoints: {len(frame_keypoints)}/{len(test_frames)}")

    # Coarse grid search
    print("\n" + "="*80)
    print("Phase 1: Coarse grid (2m steps)")
    print("="*80)

    coarse_results = {}
    for pl in range(88, 108, 2):
        for pw in range(55, 70, 2):
            frame_errors = []
            frame_focals = []
            n_ok = 0
            for fidx, kps in frame_keypoints.items():
                ip, wp, ki = get_correspondences(kps, pl, pw)
                if len(ki) < 6:
                    continue
                res = optimize_7dof(ip, wp, cx0, cy0, cam_pos=cam_pos)
                if res is None:
                    continue
                _, _, f_opt, errs = res
                # Skip catastrophic failures
                if np.median(errs) > 200:
                    continue
                frame_errors.append(np.median(errs))
                frame_focals.append(f_opt)
                n_ok += 1

            if n_ok >= 5:
                me = np.median(frame_errors)
                coarse_results[(pl, pw)] = {
                    "median_err": me,
                    "mean_err": np.mean(frame_errors),
                    "n_ok": n_ok,
                    "f_median": np.median(frame_focals),
                    "f_std": np.std(frame_focals),
                }

    # Sort and show top results
    sorted_results = sorted(coarse_results.items(), key=lambda x: x[1]["median_err"])
    print(f"\n  {'L×W':>7} {'MedErr':>8} {'MeanErr':>8} {'N_OK':>5} {'f_med':>7} {'f_std':>7}")
    for (pl, pw), r in sorted_results[:15]:
        print(f"  {pl:3d}×{pw:<3d} {r['median_err']:8.2f} {r['mean_err']:8.2f} "
              f"{r['n_ok']:5d} {r['f_median']:7.1f} {r['f_std']:7.1f}")

    best_pl, best_pw = sorted_results[0][0]
    print(f"\n  Coarse best: {best_pl}×{best_pw}m")

    # Fine grid around best
    print("\n" + "="*80)
    print(f"Phase 2: Fine grid around {best_pl}×{best_pw} (1m steps)")
    print("="*80)

    fine_results = {}
    for pl in range(best_pl - 3, best_pl + 4):
        for pw in range(best_pw - 3, best_pw + 4):
            frame_errors = []
            frame_focals = []
            n_ok = 0
            for fidx, kps in frame_keypoints.items():
                ip, wp, ki = get_correspondences(kps, pl, pw)
                if len(ki) < 6:
                    continue
                res = optimize_7dof(ip, wp, cx0, cy0, cam_pos=cam_pos)
                if res is None:
                    continue
                _, _, f_opt, errs = res
                if np.median(errs) > 200:
                    continue
                frame_errors.append(np.median(errs))
                frame_focals.append(f_opt)
                n_ok += 1

            if n_ok >= 5:
                fine_results[(pl, pw)] = {
                    "median_err": np.median(frame_errors),
                    "mean_err": np.mean(frame_errors),
                    "n_ok": n_ok,
                    "f_median": np.median(frame_focals),
                    "f_std": np.std(frame_focals),
                }

    sorted_fine = sorted(fine_results.items(), key=lambda x: x[1]["median_err"])
    print(f"\n  {'L×W':>7} {'MedErr':>8} {'MeanErr':>8} {'N_OK':>5} {'f_med':>7} {'f_std':>7}")
    for (pl, pw), r in sorted_fine[:15]:
        print(f"  {pl:3d}×{pw:<3d} {r['median_err']:8.2f} {r['mean_err']:8.2f} "
              f"{r['n_ok']:5d} {r['f_median']:7.1f} {r['f_std']:7.1f}")

    opt_pl, opt_pw = sorted_fine[0][0]
    opt_r = sorted_fine[0][1]
    print(f"\n  Optimal: {opt_pl}×{opt_pw}m (median_err={opt_r['median_err']:.2f}px, "
          f"f={opt_r['f_median']:.0f}±{opt_r['f_std']:.0f})")

    # Per-frame comparison: optimal vs 105x68
    print("\n" + "="*80)
    print(f"Phase 3: Per-frame comparison ({opt_pl}×{opt_pw} vs 105×68)")
    print("="*80)
    print(f"  {'Frame':>6} {'105×68':>8} {f'{opt_pl}×{opt_pw}':>8} {'Δ':>8} {'f_105':>7} {'f_opt':>7}")

    cap = cv2.VideoCapture(video_path)
    wins_opt = 0
    wins_105 = 0
    sample_frames = list(frame_keypoints.keys())[::3]  # every 3rd for speed
    for fidx in sample_frames:
        kps = frame_keypoints[fidx]
        ip1, wp1, ki1 = get_correspondences(kps, 105, 68)
        ip2, wp2, ki2 = get_correspondences(kps, opt_pl, opt_pw)
        if len(ki1) < 6 or len(ki2) < 6:
            continue

        r1 = optimize_7dof(ip1, wp1, cx0, cy0, cam_pos=cam_pos)
        r2 = optimize_7dof(ip2, wp2, cx0, cy0, cam_pos=cam_pos)
        if r1 is None or r2 is None:
            continue

        me1 = np.median(r1[3])
        me2 = np.median(r2[3])
        delta = me2 - me1
        if me1 < 200 or me2 < 200:
            if me2 < me1:
                wins_opt += 1
            else:
                wins_105 += 1
            print(f"  {fidx:6d} {me1:8.2f} {me2:8.2f} {delta:+8.2f} {r1[2]:7.1f} {r2[2]:7.1f}")

    cap.release()
    print(f"\n  {opt_pl}×{opt_pw} wins: {wins_opt}, 105×68 wins: {wins_105}")


if __name__ == "__main__":
    main()
