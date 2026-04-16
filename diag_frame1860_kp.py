"""Diagnose which keypoints are misidentified on frame 1860.
Shows per-keypoint reprojection error and visualizes on the frame."""

import sys, os, yaml, cv2
import numpy as np
from scipy.optimize import least_squares

sys.path.insert(0, os.path.dirname(__file__))

from goalinsight.field_registration.keypoint_detector import KeypointDetector
from goalinsight.field_registration.pnlcalib import KeypointMapper
from goalinsight.field_registration.physical_calibrator import build_field_template

# Load config
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

# Read frame
cap = cv2.VideoCapture("data/raw_videos/football_sunday_output_000.mp4")
cap.set(cv2.CAP_PROP_POS_FRAMES, 1860)
ret, frame = cap.read()
cap.release()
h, w = frame.shape[:2]
cx0, cy0 = w / 2.0, h / 2.0

# Detect
keypoints = detector.detect(frame, convert_to_soccernet=False)
mapper = KeypointMapper()
img_pts, world_pts, kp_ids = mapper.build_3d_correspondence_matrix(
    keypoints, filter_by_confidence=0.3, exclude_non_ground=False,
)

# Override world coords
field_coords, _, _ = build_field_template(105.0, 68.0)
world_pts = world_pts.copy()
for idx, kid in enumerate(kp_ids):
    if kid < len(field_coords):
        wx, wy = field_coords[kid]
        z = world_pts[idx, 2]
        world_pts[idx] = [wx, wy, z]

img_pts = img_pts.astype(np.float64)
world_pts = world_pts.astype(np.float64)

# Get confidences
conf_map = {kp["id"]: kp.get("confidence", 0) for kp in keypoints}

print(f"\nFrame 1860: {len(kp_ids)} keypoints detected (conf>=0.3)")
print(f"  kp_ids: {kp_ids}")

# PnP RANSAC
focal_candidates = [400, 800, 1200, 1600, 2000, 2500, 3000, 3500]
dist = np.zeros(5, dtype=np.float64)
best = None
best_n = -1
for f in focal_candidates:
    K = np.array([[f, 0, cx0], [0, f, cy0], [0, 0, 1]], dtype=np.float64)
    for solver in [cv2.SOLVEPNP_SQPNP, cv2.SOLVEPNP_EPNP]:
        ok, rvec, tvec, inliers = cv2.solvePnPRansac(
            world_pts.reshape(-1, 1, 3), img_pts.reshape(-1, 1, 2),
            K, dist, reprojectionError=30.0, iterationsCount=2000, flags=solver,
        )
        if ok and inliers is not None and len(inliers) > best_n:
            best_n = len(inliers)
            best = (rvec.ravel(), tvec.ravel(), f, inliers.ravel())

rvec0, tvec0, f0, inlier_idx = best
print(f"  PnP RANSAC: f={f0}, {len(inlier_idx)}/{len(kp_ids)} inliers")
print(f"  RANSAC inlier kp_ids: {[kp_ids[i] for i in inlier_idx]}")
print(f"  RANSAC outlier kp_ids: {[kp_ids[i] for i in range(len(kp_ids)) if i not in inlier_idx]}")

# Optimize 7-DOF on RANSAC inliers
inlier_mask = np.zeros(len(kp_ids), dtype=bool)
inlier_mask[inlier_idx] = True
img_in = img_pts[inlier_mask]
world_in = world_pts[inlier_mask]
kp_in = [kp_ids[i] for i in range(len(kp_ids)) if inlier_mask[i]]

x0 = np.concatenate([rvec0, tvec0, [f0]])
def cost_fn(x):
    rv = x[:3].reshape(3, 1)
    tv = x[3:6].reshape(3, 1)
    f = x[6]
    K = np.array([[f, 0, cx0], [0, f, cy0], [0, 0, 1]], dtype=np.float64)
    proj, _ = cv2.projectPoints(world_in.reshape(-1, 1, 3), rv, tv, K, dist)
    return (proj.reshape(-1, 2) - img_in).ravel()

result = least_squares(cost_fn, x0, method="trf",
                       bounds=([-np.inf]*6 + [400], [np.inf]*6 + [3500]),
                       loss="cauchy", f_scale=15.0, max_nfev=500)
rvec_opt = result.x[:3]
tvec_opt = result.x[3:6]
f_opt = result.x[6]
K_opt = np.array([[f_opt, 0, cx0], [0, f_opt, cy0], [0, 0, 1]], dtype=np.float64)

# Compute per-keypoint errors on ALL points (including RANSAC outliers)
proj_all, _ = cv2.projectPoints(
    world_pts.reshape(-1, 1, 3),
    rvec_opt.reshape(3, 1), tvec_opt.reshape(3, 1), K_opt, dist,
)
proj_all = proj_all.reshape(-1, 2)
errors = np.linalg.norm(proj_all - img_pts, axis=1)

# Also compute world back-projection error
R, _ = cv2.Rodrigues(rvec_opt.reshape(3, 1))
cam_center = -R.T @ tvec_opt
pts_undist = cv2.undistortPoints(
    img_pts.reshape(-1, 1, 2), K_opt, dist
).reshape(-1, 2)
rays_cam = np.hstack([pts_undist, np.ones((len(pts_undist), 1))])
rays_world = (R.T @ rays_cam.T).T
t_params = -cam_center[2] / rays_world[:, 2]
wp_back = cam_center[np.newaxis, :] + t_params[:, np.newaxis] * rays_world
world_errs = np.linalg.norm(wp_back[:, :2] - world_pts[:, :2], axis=1)

print(f"\n  Optimized: f={f_opt:.1f}")
R_cam, _ = cv2.Rodrigues(rvec_opt.reshape(3, 1))
cam_pos = -R_cam.T @ tvec_opt
print(f"  Camera position: ({cam_pos[0]:+.2f}, {cam_pos[1]:+.2f}, {cam_pos[2]:+.2f})")

print(f"\n  {'KP_ID':>5} {'Conf':>6} {'RANSAC':>7} {'ImgX':>7} {'ImgY':>7} "
      f"{'ProjX':>7} {'ProjY':>7} {'RepErr':>8} {'WorldErr':>9} {'WorldXY':>20}")
print(f"  {'-'*5} {'-'*6} {'-'*7} {'-'*7} {'-'*7} {'-'*7} {'-'*7} {'-'*8} {'-'*9} {'-'*20}")

for i in range(len(kp_ids)):
    kid = kp_ids[i]
    conf = conf_map.get(kid, 0)
    is_inlier = "inlier" if inlier_mask[i] else "OUTLIER"
    ix, iy = img_pts[i]
    px, py = proj_all[i]
    err = errors[i]
    werr = world_errs[i]
    wx, wy = world_pts[i, :2]
    marker = " ***" if err > 50 else (" **" if err > 30 else (" *" if err > 15 else ""))
    print(f"  {kid:5d} {conf:6.3f} {is_inlier:>7} {ix:7.1f} {iy:7.1f} "
          f"{px:7.1f} {py:7.1f} {err:8.1f} {werr:9.2f}  ({wx:+.1f},{wy:+.1f}){marker}")

# Draw visualization
vis = frame.copy()
for i in range(len(kp_ids)):
    kid = kp_ids[i]
    ix, iy = int(img_pts[i, 0]), int(img_pts[i, 1])
    px, py = int(proj_all[i, 0]), int(proj_all[i, 1])
    err = errors[i]

    # Color: green=good (<15px), yellow=medium, red=bad (>50px)
    if err > 50:
        color = (0, 0, 255)  # red
    elif err > 15:
        color = (0, 165, 255)  # orange
    else:
        color = (0, 255, 0)  # green

    # Draw detected position (circle)
    cv2.circle(vis, (ix, iy), 8, color, 2)
    # Draw projected position (cross)
    cv2.drawMarker(vis, (px, py), color, cv2.MARKER_CROSS, 15, 2)
    # Draw line between them
    cv2.line(vis, (ix, iy), (px, py), color, 1)
    # Label
    cv2.putText(vis, f"kp{kid} ({err:.0f}px)", (ix+10, iy-10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

cv2.imwrite("diag_frame1860_keypoints.jpg", vis)
print(f"\nVisualization saved to diag_frame1860_keypoints.jpg")
