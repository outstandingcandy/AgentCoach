"""Fine-grained field dimension search around 91×55m.
Uses multiprocessing to parallelize across grid points (16 cores)."""

import sys, os, yaml, cv2
import numpy as np
from scipy.optimize import least_squares
from concurrent.futures import ProcessPoolExecutor, as_completed
import time

sys.path.insert(0, os.path.dirname(__file__))


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


def get_correspondences(keypoints, pitch_length, pitch_width, min_conf=0.3):
    from goalinsight.field_registration.pnlcalib import KeypointMapper
    from goalinsight.field_registration.physical_calibrator import build_field_template
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
                  cam_pos=None, pos_weight=200.0):
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


def eval_single(args):
    """Worker function for one (pl, pw) combination across all frames."""
    pl, pw, frame_keypoints_list, cx0, cy0, cam_pos, pos_weight = args
    frame_errors = []
    frame_focals = []
    for kps in frame_keypoints_list:
        ip, wp, ki = get_correspondences(kps, pl, pw)
        if len(ki) < 6:
            continue
        res = optimize_7dof(ip, wp, cx0, cy0, cam_pos=cam_pos, pos_weight=pos_weight)
        if res is None:
            continue
        _, _, f_opt, errs = res
        if np.median(errs) > 200:
            continue
        frame_errors.append(np.median(errs))
        frame_focals.append(f_opt)
    if len(frame_errors) < 5:
        return pl, pw, None
    return pl, pw, {
        "median_err": float(np.median(frame_errors)),
        "mean_err": float(np.mean(frame_errors)),
        "n_ok": len(frame_errors),
        "f_median": float(np.median(frame_focals)),
        "f_std": float(np.std(frame_focals)),
    }


def run_grid(lengths, widths, frame_kps_list, cx0, cy0, cam_pos, pos_weight, n_workers=14):
    """Run grid search in parallel."""
    tasks = []
    for pl in lengths:
        for pw in widths:
            tasks.append((round(pl, 1), round(pw, 1), frame_kps_list, cx0, cy0, cam_pos, pos_weight))

    results = {}
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=n_workers) as pool:
        futures = {pool.submit(eval_single, t): t[:2] for t in tasks}
        done = 0
        for fut in as_completed(futures):
            pl, pw, r = fut.result()
            done += 1
            if r is not None:
                results[(pl, pw)] = r
            if done % 50 == 0 or done == len(tasks):
                elapsed = time.time() - t0
                print(f"  {done}/{len(tasks)} done ({elapsed:.0f}s)")

    return results


def main():
    cfg = load_config()

    # Detect keypoints (must be done in main process due to GPU model)
    from goalinsight.field_registration.keypoint_detector import KeypointDetector
    pnl_cfg = cfg["field_registration"]["pnlcalib"]
    kp_config = {
        "backend": "pnlcalib",
        "pnlcalib": {
            "weights": pnl_cfg.get("keypoint_weights", "SV_kp"),
            "model_path": pnl_cfg.get("keypoint_model_path"),
            "confidence_threshold": pnl_cfg.get("keypoint_threshold", 0.3434),
        }
    }
    print("Loading detector...")
    detector = KeypointDetector(kp_config)
    detector.load_model()

    video_path = "data/raw_videos/football_sunday_output_000.mp4"
    cap = cv2.VideoCapture(video_path)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cx0, cy0 = w / 2.0, h / 2.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    cam_pos = [0.0, -32.0, 6.3]
    pos_weight = 200.0

    test_frames = list(range(0, min(total_frames, 1870), 60))
    print(f"Detecting keypoints on {len(test_frames)} frames...")
    frame_kps_list = []
    for fidx in test_frames:
        cap.set(cv2.CAP_PROP_POS_FRAMES, fidx)
        ret, frame = cap.read()
        if not ret:
            continue
        kps = detector.detect(frame, convert_to_soccernet=False)
        n_good = len([k for k in kps if k.get("confidence", 0) >= 0.3])
        if n_good >= 6:
            frame_kps_list.append(kps)
    cap.release()
    print(f"  {len(frame_kps_list)} frames ready, using 14 workers")

    # Free GPU memory before forking
    del detector
    import torch
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Phase 1: 0.5m grid
    print("\n" + "="*80)
    print("Phase 1: 0.5m grid (L: 88-94, W: 52-58)")
    print("="*80)

    lengths1 = np.arange(88.0, 94.5, 0.5)
    widths1 = np.arange(52.0, 58.5, 0.5)
    print(f"  {len(lengths1)}×{len(widths1)} = {len(lengths1)*len(widths1)} combinations")

    results1 = run_grid(lengths1, widths1, frame_kps_list, cx0, cy0, cam_pos, pos_weight)
    sorted1 = sorted(results1.items(), key=lambda x: x[1]["median_err"])

    print(f"\n  {'L×W':>10} {'MedErr':>8} {'MeanErr':>8} {'N':>4} {'f_med':>7} {'f_std':>7}")
    for (pl, pw), r in sorted1[:15]:
        print(f"  {pl:5.1f}×{pw:<5.1f} {r['median_err']:8.2f} {r['mean_err']:8.2f} "
              f"{r['n_ok']:4d} {r['f_median']:7.1f} {r['f_std']:7.1f}")

    best_pl, best_pw = sorted1[0][0]
    print(f"\n  Best (0.5m): {best_pl}×{best_pw}m, err={sorted1[0][1]['median_err']:.2f}px")

    # Phase 2: 0.1m grid around best
    print("\n" + "="*80)
    print(f"Phase 2: 0.1m grid around {best_pl}×{best_pw}")
    print("="*80)

    lengths2 = np.arange(best_pl - 1.0, best_pl + 1.05, 0.1)
    widths2 = np.arange(best_pw - 1.0, best_pw + 1.05, 0.1)
    print(f"  {len(lengths2)}×{len(widths2)} = {len(lengths2)*len(widths2)} combinations")

    results2 = run_grid(lengths2, widths2, frame_kps_list, cx0, cy0, cam_pos, pos_weight)
    sorted2 = sorted(results2.items(), key=lambda x: x[1]["median_err"])

    print(f"\n  {'L×W':>11} {'MedErr':>8} {'MeanErr':>8} {'N':>4} {'f_med':>7} {'f_std':>7}")
    for (pl, pw), r in sorted2[:20]:
        print(f"  {pl:5.1f}×{pw:<5.1f} {r['median_err']:8.2f} {r['mean_err']:8.2f} "
              f"{r['n_ok']:4d} {r['f_median']:7.1f} {r['f_std']:7.1f}")

    opt_pl, opt_pw = sorted2[0][0]
    opt_r = sorted2[0][1]
    print(f"\n  OPTIMAL: {opt_pl}×{opt_pw}m")
    print(f"  median_err={opt_r['median_err']:.2f}px, f={opt_r['f_median']:.0f}±{opt_r['f_std']:.0f}")


if __name__ == "__main__":
    main()
