"""If all keypoints are correctly detected on frame 1860, the high error must come
from the geometric model. Analyze residual patterns to find the root cause:
- Field dimensions mismatch?
- Lens distortion not modeled?
- Principal point offset?

Approach:
1. Optimize 7-DOF on ALL 12 points (no RANSAC filtering)
2. Show residual vectors (direction matters, not just magnitude)
3. Leave-one-out analysis
4. Try different field sizes to see error sensitivity
"""

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

    keypoints = detector.detect(frame, convert_to_soccernet=False)
    return cfg, frame, keypoints, w, h


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
    """Optimize 7-DOF on given points. Returns (rvec, tvec, f, per_point_errors)."""
    dist = np.zeros(5, dtype=np.float64)

    # PnP init (multi-focal)
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
        res = [((proj.reshape(-1, 2) - img_pts).ravel())]
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
    residual_vecs = proj.reshape(-1, 2) - img_pts  # direction matters!
    return rv_opt, tv_opt, f_opt, errs, residual_vecs


def main():
    cfg, frame, keypoints, w, h = load_all()
    cx0, cy0 = w / 2.0, h / 2.0
    cam_pos = cfg["field_registration"]["physical"].get("camera_position")

    img_pts, world_pts, kp_ids = get_correspondences(keypoints)
    print(f"\nFrame 1860: {len(kp_ids)} keypoints")
    print(f"kp_ids: {kp_ids}")

    # ========== 1. Optimize ALL points (no RANSAC filter) ==========
    print("\n" + "="*80)
    print("1. 7-DOF on ALL 12 points (no filtering)")
    print("="*80)

    res = optimize_7dof(img_pts, world_pts, cx0, cy0, cam_pos=cam_pos)
    if res is None:
        print("  FAILED")
        return
    rv, tv, f, errs, res_vecs = res
    R, _ = cv2.Rodrigues(rv.reshape(3, 1))
    cam = -R.T @ tv

    print(f"  f={f:.1f}, cam=({cam[0]:+.2f}, {cam[1]:+.2f}, {cam[2]:+.2f})")
    print(f"  mean={np.mean(errs):.1f}px, median={np.median(errs):.1f}px, max={np.max(errs):.1f}px")
    print(f"\n  {'KP':>4} {'ImgXY':>14} {'Err':>6} {'dX':>7} {'dY':>7}  {'WorldXY':>16}  Note")
    print(f"  {'-'*4} {'-'*14} {'-'*6} {'-'*7} {'-'*7}  {'-'*16}  {'-'*20}")
    for i in range(len(kp_ids)):
        kid = kp_ids[i]
        ix, iy = img_pts[i]
        dx, dy = res_vecs[i]
        err = errs[i]
        wx, wy = world_pts[i, :2]
        note = ""
        if err > 30: note = "*** BAD"
        elif err > 15: note = "* marginal"
        print(f"  {kid:4d} ({ix:6.1f},{iy:6.1f}) {err:6.1f} {dx:+7.1f} {dy:+7.1f}  ({wx:+6.1f},{wy:+5.1f})  {note}")

    # ========== 2. Residual direction analysis ==========
    print("\n" + "="*80)
    print("2. Residual direction analysis (systematic pattern = model deficiency)")
    print("="*80)
    # If residuals all point in similar direction → principal point offset
    # If residuals radiate from center → radial distortion
    # If residuals correlate with field position → wrong field dimensions
    angles = np.arctan2(res_vecs[:, 1], res_vecs[:, 0]) * 180 / np.pi
    print(f"  Residual angles (degrees from +X axis):")
    for i in range(len(kp_ids)):
        if errs[i] > 5:
            print(f"    kp{kp_ids[i]:2d}: angle={angles[i]:+6.1f}°, mag={errs[i]:.1f}px, "
                  f"img=({img_pts[i,0]:.0f},{img_pts[i,1]:.0f})")

    # Check radial pattern: correlation between distance-from-center and error
    r_from_center = np.sqrt((img_pts[:, 0] - cx0)**2 + (img_pts[:, 1] - cy0)**2)
    corr = np.corrcoef(r_from_center, errs)[0, 1]
    print(f"\n  Correlation(distance_from_center, error) = {corr:.3f}")
    if abs(corr) > 0.5:
        print(f"  → Strong radial pattern: suggests lens distortion")
    else:
        print(f"  → No clear radial pattern")

    # Check if residuals point away from center (barrel distortion) or towards (pincushion)
    radial_dirs = img_pts - np.array([cx0, cy0])
    radial_dirs_norm = radial_dirs / (np.linalg.norm(radial_dirs, axis=1, keepdims=True) + 1e-10)
    radial_component = np.sum(res_vecs * radial_dirs_norm, axis=1)  # positive = away from center
    print(f"  Radial component of residuals (positive=away from center):")
    for i in range(len(kp_ids)):
        if errs[i] > 5:
            print(f"    kp{kp_ids[i]:2d}: radial={radial_component[i]:+7.1f}px (of {errs[i]:.1f}px total)")

    # ========== 3. Leave-one-out analysis ==========
    print("\n" + "="*80)
    print("3. Leave-one-out analysis")
    print("="*80)
    print(f"  {'Removed':>8} {'RemainingErr':>12} {'RemovedPtErr':>13} {'f':>7} {'CamXYZ':>25}")

    for skip_i in range(len(kp_ids)):
        mask = np.ones(len(kp_ids), dtype=bool)
        mask[skip_i] = False
        img_sub = img_pts[mask]
        world_sub = world_pts[mask]

        res_sub = optimize_7dof(img_sub, world_sub, cx0, cy0, cam_pos=cam_pos)
        if res_sub is None:
            print(f"  kp{kp_ids[skip_i]:4d}     FAILED")
            continue
        rv_s, tv_s, f_s, errs_s, _ = res_sub

        # Error on the removed point
        K_s = np.array([[f_s, 0, cx0], [0, f_s, cy0], [0, 0, 1]], dtype=np.float64)
        proj_rem, _ = cv2.projectPoints(
            world_pts[skip_i:skip_i+1].reshape(-1, 1, 3),
            rv_s.reshape(3, 1), tv_s.reshape(3, 1), K_s, np.zeros(5))
        err_rem = np.linalg.norm(proj_rem.reshape(-1, 2) - img_pts[skip_i:skip_i+1])

        R_s, _ = cv2.Rodrigues(rv_s.reshape(3, 1))
        cam_s = -R_s.T @ tv_s
        print(f"  kp{kp_ids[skip_i]:4d} {np.mean(errs_s):12.2f} {err_rem:13.2f} {f_s:7.1f} "
              f"({cam_s[0]:+7.2f},{cam_s[1]:+7.2f},{cam_s[2]:+5.2f})")

    # ========== 4. Field dimension sensitivity ==========
    print("\n" + "="*80)
    print("4. Field dimension sensitivity")
    print("="*80)
    print(f"  {'Length':>6} {'Width':>5} {'MeanErr':>8} {'MedErr':>8} {'MaxErr':>8} {'f':>7} {'CamXYZ':>25}")

    for pl, pw in [(105, 68), (100, 65), (100, 68), (105, 65), (95, 60), (98, 64), (102, 66)]:
        img_p, world_p, kp_i = get_correspondences(keypoints, pl, pw)
        res_d = optimize_7dof(img_p, world_p, cx0, cy0, cam_pos=cam_pos)
        if res_d is None:
            print(f"  {pl:6d} {pw:5d}   FAILED")
            continue
        rv_d, tv_d, f_d, errs_d, _ = res_d
        R_d, _ = cv2.Rodrigues(rv_d.reshape(3, 1))
        cam_d = -R_d.T @ tv_d
        print(f"  {pl:6d} {pw:5d} {np.mean(errs_d):8.2f} {np.median(errs_d):8.2f} "
              f"{np.max(errs_d):8.2f} {f_d:7.1f} ({cam_d[0]:+7.2f},{cam_d[1]:+7.2f},{cam_d[2]:+5.2f})")

    # ========== 5. Visualization ==========
    vis = frame.copy()
    # Re-run full optimization for vis
    res = optimize_7dof(img_pts, world_pts, cx0, cy0, cam_pos=cam_pos)
    rv, tv, f, errs, res_vecs = res
    K_vis = np.array([[f, 0, cx0], [0, f, cy0], [0, 0, 1]], dtype=np.float64)

    # Project ALL field template points for overlay
    field_coords, line_defs, _ = build_field_template(105.0, 68.0)
    for lid, ld in line_defs.items():
        p1 = np.array(ld["p1"], dtype=np.float64).reshape(1, 1, 3)
        p2 = np.array(ld["p2"], dtype=np.float64).reshape(1, 1, 3)
        pr1, _ = cv2.projectPoints(p1, rv.reshape(3, 1), tv.reshape(3, 1), K_vis, np.zeros(5))
        pr2, _ = cv2.projectPoints(p2, rv.reshape(3, 1), tv.reshape(3, 1), K_vis, np.zeros(5))
        pt1 = tuple(pr1.reshape(2).astype(int))
        pt2 = tuple(pr2.reshape(2).astype(int))
        if all(-500 < c < w+500 for c in pt1+pt2):
            cv2.line(vis, pt1, pt2, (255, 255, 0), 1, cv2.LINE_AA)

    for i in range(len(kp_ids)):
        kid = kp_ids[i]
        ix, iy = int(img_pts[i, 0]), int(img_pts[i, 1])
        err = errs[i]
        dx, dy = res_vecs[i]

        color = (0, 0, 255) if err > 30 else ((0, 165, 255) if err > 15 else (0, 255, 0))

        # Detected (filled circle)
        cv2.circle(vis, (ix, iy), 6, color, -1)
        # Residual arrow (amplified 3x for visibility)
        end_x = int(ix + dx * 3)
        end_y = int(iy + dy * 3)
        cv2.arrowedLine(vis, (ix, iy), (end_x, end_y), color, 2, tipLength=0.3)
        # Label
        cv2.putText(vis, f"kp{kid}({err:.0f}px)", (ix+10, iy-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)

    cv2.imwrite("diag_frame1860_residuals.jpg", vis)
    print(f"\nVisualization saved to diag_frame1860_residuals.jpg")


if __name__ == "__main__":
    main()
