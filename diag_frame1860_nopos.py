"""Same analysis on frame 1860 but WITHOUT camera position constraint,
and test finer field dimension grid around the sweet spot found (95x60)."""

import sys, os, yaml, cv2
import numpy as np
from scipy.optimize import least_squares

sys.path.insert(0, os.path.dirname(__file__))

from goalinsight.field_registration.keypoint_detector import KeypointDetector
from goalinsight.field_registration.pnlcalib import KeypointMapper
from goalinsight.field_registration.physical_calibrator import build_field_template


def load_all():
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

    cap = cv2.VideoCapture("data/raw_videos/football_sunday_output_000.mp4")
    cap.set(cv2.CAP_PROP_POS_FRAMES, 1860)
    ret, frame = cap.read()
    cap.release()
    h, w = frame.shape[:2]
    return cfg, frame, detector, keypoints_from(detector, frame), w, h


def keypoints_from(detector, frame):
    return detector.detect(frame, convert_to_soccernet=False)


def get_correspondences(keypoints, pitch_length=105.0, pitch_width=68.0, min_conf=0.3):
    mapper = KeypointMapper()
    img_pts, world_pts, kp_ids = mapper.build_3d_correspondence_matrix(
        keypoints, filter_by_confidence=min_conf, exclude_non_ground=False,
    )
    field_coords, _, _ = build_field_template(pitch_length, pitch_width)
    world_pts = world_pts.copy()
    for idx, kid in enumerate(kp_ids):
        if kid < len(field_coords):
            wx, wy = field_coords[kid]
            z = world_pts[idx, 2]
            world_pts[idx] = [wx, wy, z]
    return img_pts.astype(np.float64), world_pts.astype(np.float64), kp_ids


def optimize_7dof(img_pts, world_pts, cx, cy, f_init=1500, focal_bounds=(400, 3500),
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
    res_vecs = proj.reshape(-1, 2) - img_pts
    return rv_opt, tv_opt, f_opt, errs, res_vecs


def main():
    cfg, frame, detector, keypoints, w, h = load_all()
    cx0, cy0 = w / 2.0, h / 2.0

    img_pts, world_pts, kp_ids = get_correspondences(keypoints)
    print(f"\nFrame 1860: {len(kp_ids)} keypoints, kp_ids={kp_ids}")

    # ========== Compare WITH vs WITHOUT position constraint ==========
    print("\n" + "="*80)
    print("A. WITH camera position constraint (0, -32, 6.3), weight=200")
    print("="*80)
    res_pos = optimize_7dof(img_pts, world_pts, cx0, cy0,
                            cam_pos=[0, -32, 6.3], pos_weight=200)
    rv, tv, f, errs, vecs = res_pos
    R, _ = cv2.Rodrigues(rv.reshape(3, 1))
    cam = -R.T @ tv
    print(f"  f={f:.1f}, cam=({cam[0]:+.2f}, {cam[1]:+.2f}, {cam[2]:+.2f})")
    print(f"  mean={np.mean(errs):.1f}, median={np.median(errs):.1f}, max={np.max(errs):.1f}")
    for i in range(len(kp_ids)):
        print(f"    kp{kp_ids[i]:2d}: err={errs[i]:6.1f}  dXY=({vecs[i,0]:+7.1f},{vecs[i,1]:+7.1f})")

    print("\n" + "="*80)
    print("B. WITHOUT camera position constraint")
    print("="*80)
    res_free = optimize_7dof(img_pts, world_pts, cx0, cy0, cam_pos=None)
    rv2, tv2, f2, errs2, vecs2 = res_free
    R2, _ = cv2.Rodrigues(rv2.reshape(3, 1))
    cam2 = -R2.T @ tv2
    print(f"  f={f2:.1f}, cam=({cam2[0]:+.2f}, {cam2[1]:+.2f}, {cam2[2]:+.2f})")
    print(f"  mean={np.mean(errs2):.1f}, median={np.median(errs2):.1f}, max={np.max(errs2):.1f}")
    for i in range(len(kp_ids)):
        print(f"    kp{kp_ids[i]:2d}: err={errs2[i]:6.1f}  dXY=({vecs2[i,0]:+7.1f},{vecs2[i,1]:+7.1f})")

    # ========== Fine field dimension grid WITHOUT position constraint ==========
    print("\n" + "="*80)
    print("C. Field dimension grid (NO position constraint)")
    print("="*80)
    print(f"  {'L×W':>7} {'MeanErr':>8} {'MedErr':>8} {'MaxErr':>8} {'f':>7} {'CamXYZ':>30}")

    best_err = 999
    best_dim = None
    for pl in range(88, 108, 2):
        for pw in range(55, 70, 2):
            img_p, world_p, _ = get_correspondences(keypoints, pl, pw)
            res = optimize_7dof(img_p, world_p, cx0, cy0, cam_pos=None)
            if res is None:
                continue
            rv_d, tv_d, f_d, errs_d, _ = res
            R_d, _ = cv2.Rodrigues(rv_d.reshape(3, 1))
            cam_d = -R_d.T @ tv_d
            me = np.mean(errs_d)
            if me < best_err:
                best_err = me
                best_dim = (pl, pw)
            if me < 30:  # only print promising ones
                print(f"  {pl:3d}×{pw:<3d} {me:8.2f} {np.median(errs_d):8.2f} "
                      f"{np.max(errs_d):8.2f} {f_d:7.1f} ({cam_d[0]:+7.2f},{cam_d[1]:+7.2f},{cam_d[2]:+5.2f})")

    print(f"\n  Best: {best_dim[0]}×{best_dim[1]}m → mean_err={best_err:.2f}px")

    # Fine grid around best
    pl_best, pw_best = best_dim
    print(f"\n  Fine grid around {pl_best}×{pw_best}:")
    print(f"  {'L×W':>7} {'MeanErr':>8} {'MedErr':>8} {'MaxErr':>8} {'f':>7} {'CamXYZ':>30}")
    for pl in range(pl_best - 4, pl_best + 5):
        for pw in range(pw_best - 4, pw_best + 5):
            img_p, world_p, _ = get_correspondences(keypoints, pl, pw)
            res = optimize_7dof(img_p, world_p, cx0, cy0, cam_pos=None)
            if res is None:
                continue
            rv_d, tv_d, f_d, errs_d, _ = res
            R_d, _ = cv2.Rodrigues(rv_d.reshape(3, 1))
            cam_d = -R_d.T @ tv_d
            me = np.mean(errs_d)
            md = np.median(errs_d)
            mx = np.max(errs_d)
            if me < best_err + 3:
                print(f"  {pl:3d}×{pw:<3d} {me:8.2f} {md:8.2f} {mx:8.2f} {f_d:7.1f} "
                      f"({cam_d[0]:+7.2f},{cam_d[1]:+7.2f},{cam_d[2]:+5.2f})")

    # ========== D. Verify on multiple frames with best dim ==========
    print(f"\n" + "="*80)
    print(f"D. Verify best dimensions ({pl_best}×{pw_best}) on other frames")
    print("="*80)

    cap = cv2.VideoCapture("data/raw_videos/football_sunday_output_000.mp4")
    test_frames = [60, 300, 600, 900, 1200, 1500, 1860]

    print(f"  {'Frame':>6} {'105×68 MeanErr':>15} {'Best MeanErr':>13} {'105×68 f':>9} {'Best f':>7} {'Best CamXYZ':>30}")
    for fidx in test_frames:
        cap.set(cv2.CAP_PROP_POS_FRAMES, fidx)
        ret, fr = cap.read()
        if not ret:
            continue
        kps = detector.detect(fr, convert_to_soccernet=False)

        # 105x68
        ip1, wp1, ki1 = get_correspondences(kps, 105, 68)
        if len(ki1) < 6:
            continue
        r1 = optimize_7dof(ip1, wp1, cx0, cy0, cam_pos=None)

        # Best dim
        ip2, wp2, ki2 = get_correspondences(kps, pl_best, pw_best)
        r2 = optimize_7dof(ip2, wp2, cx0, cy0, cam_pos=None)

        if r1 and r2:
            R_1, _ = cv2.Rodrigues(r1[0].reshape(3, 1))
            c1 = -R_1.T @ r1[1]
            R_2, _ = cv2.Rodrigues(r2[0].reshape(3, 1))
            c2 = -R_2.T @ r2[1]
            print(f"  {fidx:6d} {np.mean(r1[3]):15.2f} {np.mean(r2[3]):13.2f} "
                  f"{r1[2]:9.1f} {r2[2]:7.1f} ({c2[0]:+7.2f},{c2[1]:+7.2f},{c2[2]:+5.2f})")
    cap.release()


if __name__ == "__main__":
    main()
