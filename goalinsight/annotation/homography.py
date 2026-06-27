"""Homography and PnP helpers for manual annotations.

The annotator's solve+project path delegates to ``pnlcalib_orig`` — the
faithful port of upstream PnLCalib (``cv2.calibrateCamera`` Zhang +
3-plane reparameterization for crossbars + 18-combo heuristic voting).
``solve_camera`` returns a ``Camera`` populated from the upstream result,
converted from upstream's y-down centered world frame to the project's
y-up centered frame so existing project/visualize callers keep working.
"""

import cv2
import numpy as np

from goalinsight.annotation.pitch import keypoints as _pk
from goalinsight.field_registration.pnlcalib.camera import Camera
from goalinsight.field_registration.pnlcalib_orig import (
    FramebyFrameCalib,
    PnLCalibIdMap,
)


def line_intersection(
    line1: tuple[tuple[float, float], tuple[float, float]],
    line2: tuple[tuple[float, float], tuple[float, float]],
) -> tuple[float, float] | None:
    """Intersection of two lines, or None if parallel."""
    (x1, y1), (x2, y2) = line1
    (x3, y3), (x4, y4) = line2

    denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    if abs(denom) < 1e-10:
        return None

    t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / denom
    return (x1 + t * (x2 - x1), y1 + t * (y2 - y1))


# ---------------------------------------------------------------------------
# pnlcalib_orig-backed solve + project
# ---------------------------------------------------------------------------

# Minimum point count for upstream's ``cv2.calibrateCamera`` path (Zhang
# multi-plane). Mirrors the project's prior 6-point floor.
MIN_POINTS_FOR_PNLCALIB = 6

# Y-axis flip between upstream's y-down centered world frame and the
# project's y-up centered frame. Conjugating by J swaps the sign of world
# Y while leaving the optical axis untouched, so a pose solved in y-down
# coords becomes a y-up pose via ``R_up = R_dn @ J``, ``t_up = t_dn``.
_J_FLIP_Y = np.diag([1.0, -1.0, 1.0])


def _camera_from_upstream_result(
    result: dict, img_size: tuple[int, int],
) -> Camera:
    """Wrap a ``heuristic_voting`` result as a populated project ``Camera``.

    Converts the upstream y-down centered world frame into the project's
    y-up centered frame by post-multiplying R with ``diag(1, -1, 1)`` and
    recomputing the camera position from the new (R, t).
    """
    cp = result["cam_params"]
    fx = float(cp["x_focal_length"])
    fy = float(cp["y_focal_length"])
    px, py = float(cp["principal_point"][0]), float(cp["principal_point"][1])

    R_dn = np.asarray(cp["rotation_matrix"], dtype=np.float64).reshape(3, 3)
    pos_dn = np.asarray(cp["position_meters"], dtype=np.float64).reshape(3)
    t_dn = -R_dn @ pos_dn

    R_up = R_dn @ _J_FLIP_Y
    t_up = t_dn
    pos_up = -R_up.T @ t_up

    K = np.array([[fx, 0.0, px], [0.0, fy, py], [0.0, 0.0, 1.0]], dtype=np.float64)

    w, h = int(img_size[0]), int(img_size[1])
    cam = Camera(iwidth=w, iheight=h)
    cam.calibration = K
    cam.xfocal_length = fx
    cam.yfocal_length = fy
    cam.principal_point = (px, py)
    cam.rotation = R_up
    cam.position = pos_up
    cam.distortion = np.zeros(5, dtype=np.float64)
    # Stash full PnP state (rvec/tvec/dist) for the renderer. pnlcalib's
    # heuristic_voting is pinhole-only, so dist is zero — but the
    # interface is the same as the physical path so the consumer doesn't
    # have to branch.
    rvec_up, _ = cv2.Rodrigues(R_up)
    cam._pnp_rvec = rvec_up.reshape(3, 1)
    cam._pnp_tvec = t_up.reshape(3, 1)
    cam._pnp_dist = np.zeros(5, dtype=np.float64)
    return cam


def _project_world_to_pixel(
    cam: Camera, pt_3d: np.ndarray,
) -> tuple[float, float] | None:
    """Manual projection: K @ R @ (P - C). Returns None for points behind cam.

    Bypasses ``cv2.projectPoints`` because the y-up world frame stored on
    the project ``Camera`` produces a reflection rotation (det = -1) —
    the upstream solver works in y-down, and conjugating by ``diag(1,-1,
    1)`` to swap conventions makes ``R_up = R_dn @ J``. cv2.Rodrigues /
    projectPoints expect a proper rotation, but the linear algebra
    ``K @ R_up @ (P - C_up)`` still produces the correct pixel.
    """
    p = cam.rotation @ (pt_3d - cam.position)
    if p[2] <= 1e-3:
        return None
    x = p[0] / p[2] * cam.xfocal_length + cam.principal_point[0]
    y = p[1] / p[2] * cam.yfocal_length + cam.principal_point[1]
    return float(x), float(y)


def _active_pitch_dims() -> dict:
    """Read the active SoccerPitch as the dict ``FramebyFrameCalib`` expects."""
    pitch = _pk.get_active_pitch()
    return {
        "pitch_length": pitch.PITCH_LENGTH,
        "pitch_width": pitch.PITCH_WIDTH,
        "penalty_area_width": pitch.PENALTY_AREA_WIDTH,
        "penalty_area_length": pitch.PENALTY_AREA_LENGTH,
        "goal_area_width": pitch.GOAL_AREA_WIDTH,
        "goal_area_length": pitch.GOAL_AREA_LENGTH,
        "goal_line_to_penalty_mark": pitch.GOAL_LINE_TO_PENALTY_MARK,
        "center_circle_radius": pitch.CENTER_CIRCLE_RADIUS,
        "goal_height": pitch.GOAL_HEIGHT,
        "goal_length": pitch.GOAL_LENGTH,
    }


def solve_camera(
    pixel_pts: list[tuple[float, float]],
    world_3d_pts: list[tuple[float, float, float]],
    img_size: tuple[int, int],
) -> tuple[Camera | None, float, dict]:
    """Solve a ``Camera`` from 3D-2D point correspondences via pnlcalib_orig.

    Delegates to upstream's ``FramebyFrameCalib.update`` +
    ``heuristic_voting`` (18 mode×ransac combos, ``cv2.calibrateCamera``
    Zhang). World points are matched against the upstream-indexed pitch
    table (``world_xyz_to_upstream_id``) so the upstream solver sees the
    same correspondences in its own keypoint id space.

    Returns ``(camera, mean_error_px, diagnostics)`` on success, ``(None,
    inf, {...})`` on failure (diagnostics still carries ``unresolved`` so
    the caller can tell apart "too few resolvable pts" from "PnP failed").
    Diagnostics keys: ``median_err``, ``max_err``, ``back_facing``,
    ``fx``, ``mode``, ``ransac``, ``calib_plane``, ``unresolved``.
    """
    if len(pixel_pts) < MIN_POINTS_FOR_PNLCALIB:
        return None, float("inf"), {"unresolved": 0}

    pitch_dims = _active_pitch_dims()
    pitch = _pk.get_active_pitch()
    id_map = PnLCalibIdMap(pitch=pitch)

    w, h = int(img_size[0]), int(img_size[1])
    calib = FramebyFrameCalib(
        iwidth=w, iheight=h, denormalize=False, pitch_dims=pitch_dims,
    )

    keypoints_dict: dict[int, dict[str, float]] = {}
    unresolved = 0
    for (px, py), w3d in zip(pixel_pts, world_3d_pts, strict=True):
        up_id = id_map.world_xyz_to_upstream_id(
            (float(w3d[0]), float(w3d[1]), float(w3d[2])),
        )
        if up_id is None:
            unresolved += 1
            continue
        # Multiple project pixels may collapse onto the same upstream id
        # (e.g. derived & raw at the same corner). Keep the highest-prob
        # one — the upstream solver only reads one obs per id anyway.
        prev = keypoints_dict.get(up_id)
        if prev is None:
            keypoints_dict[up_id] = {"x": float(px), "y": float(py), "p": 1.0}

    if len(keypoints_dict) < MIN_POINTS_FOR_PNLCALIB:
        return None, float("inf"), {"unresolved": unresolved}

    calib.update(keypoints_dict, {})
    result = calib.heuristic_voting(refine=True, refine_lines=False, th=5.0)
    if result is None:
        return None, float("inf"), {"unresolved": unresolved}

    cam = _camera_from_upstream_result(result, img_size)

    errs = []
    for (px, py), w3d in zip(pixel_pts, world_3d_pts, strict=True):
        proj = _project_world_to_pixel(cam, np.asarray(w3d, dtype=np.float64))
        if proj is None:
            errs.append(float("inf"))
            continue
        errs.append(float(np.hypot(proj[0] - px, proj[1] - py)))
    err_arr = np.asarray(errs, dtype=np.float64)
    finite = err_arr[np.isfinite(err_arr)]
    median_err = float(np.median(finite)) if finite.size else float("inf")
    max_err = float(err_arr.max())
    mean_err = float(np.mean(err_arr))

    # Camera looks along +Z in *its own* frame, so the world's depth at
    # the camera's principal axis is just t_z. Reuse the upstream
    # ``position_meters`` (y-down): if its z-component (height) is < 0,
    # the camera is below the pitch — same physical check as before.
    pos_dn_z = float(result["cam_params"]["position_meters"][2])

    diagnostics = {
        "median_err": median_err,
        "max_err": max_err,
        "back_facing": bool(pos_dn_z >= 0),
        "fx": float(cam.calibration[0, 0]),
        "mode": result.get("mode"),
        "ransac": result.get("use_ransac"),
        "calib_plane": result.get("calib_plane"),
        "unresolved": unresolved,
    }
    return cam, mean_err, diagnostics


def _calibrate_free_intrinsics(
    pixel_pts: list[tuple[float, float]],
    world_3d_pts: list[tuple[float, float, float]],
    img_size: tuple[int, int],
    K_init: np.ndarray,
    profile_name: str,
) -> tuple[Camera | None, float, dict]:
    """Run cv2.calibrateCamera with intrinsics fully free, return Camera + diag.

    Uses fy = fx (aspect lock — futsal phones / standard sensors don't
    have anamorphic distortion) and k3 = 0 (the 6th-order radial term
    is virtually always over-fit on small annotation sets), but lets
    cx, cy, k1, k2, p1, p2 all move. With ≥ ~12 points (mostly
    ground-plane plus one or two z != 0) this typically converges to
    physically reasonable values — see scripts/probe_chain_gapfill.py
    for the exploration.
    """
    w, h = int(img_size[0]), int(img_size[1])
    img_arr = np.asarray(pixel_pts, dtype=np.float32).reshape(-1, 1, 2)
    world_arr = np.asarray(world_3d_pts, dtype=np.float32).reshape(-1, 3)

    flags = (
        cv2.CALIB_USE_INTRINSIC_GUESS
        | cv2.CALIB_FIX_ASPECT_RATIO  # fy = fx
        | cv2.CALIB_FIX_K3             # zero out 6th-order radial term
    )
    try:
        ret, K, dist, rvecs, tvecs = cv2.calibrateCamera(
            [world_arr], [img_arr], (w, h),
            K_init.astype(np.float64).copy(),
            np.zeros(5, dtype=np.float64),
            flags=flags,
        )
    except cv2.error as exc:
        return None, float("inf"), {
            "unresolved": 0, "mode": "physical_free",
            "fail": str(exc).splitlines()[0][:120],
            "profile": profile_name,
        }

    rvec = rvecs[0].reshape(3)
    tvec = tvecs[0].reshape(3)

    # Robust refinement: cv2.calibrateCamera optimises L2 loss, so one or
    # two badly-clickable points (center-circle / halfway-line crossings
    # at the far end of the field) drag the whole pose to compromise
    # between outlier and inlier. Re-run with Cauchy loss to down-weight
    # those naturally — typically drops the median by ~30% and rescues
    # the well-clicked ground-plane corners. Falls back silently to the
    # L2 result if scipy isn't available or the refinement diverges.
    try:
        from scipy.optimize import least_squares as _ls
    except ImportError:
        _ls = None
    if _ls is not None:
        img_pix64 = np.asarray(pixel_pts, dtype=np.float64)
        world64 = np.asarray(world_3d_pts, dtype=np.float64)

        def _unpack(x):
            K_ = np.array(
                [[x[0], 0, x[1]], [0, x[0], x[2]], [0, 0, 1]],
                dtype=np.float64,
            )
            dist_ = np.array([x[3], x[4], x[5], x[6], 0.0], dtype=np.float64)
            return K_, dist_, x[7:10].reshape(3, 1), x[10:13].reshape(3, 1)

        def _residuals(x):
            K_, dist_, rv_, tv_ = _unpack(x)
            try:
                proj_, _ = cv2.projectPoints(world64, rv_, tv_, K_, dist_)
            except cv2.error:
                return np.full(2 * len(img_pix64), 1e6)
            return (proj_.reshape(-1, 2) - img_pix64).ravel()

        dist_flat = np.asarray(dist).reshape(-1)
        x0 = np.r_[
            K[0, 0], K[0, 2], K[1, 2],
            dist_flat[:4],  # k1, k2, p1, p2 (k3 stays zero)
            rvec, tvec,
        ]
        try:
            result = _ls(
                _residuals, x0, loss="cauchy", f_scale=5.0,
                method="trf", max_nfev=2000,
            )
            K_new, dist_new, rvec_new, tvec_new = _unpack(result.x)
            # Sanity: only accept if camera is in front of pitch and the
            # median residual is no worse than the L2 seed (Cauchy tries
            # to balance robustness, but with bad data it can wander).
            proj_new, _ = cv2.projectPoints(
                world_arr, rvec_new, tvec_new, K_new, dist_new,
            )
            err_new = np.linalg.norm(
                proj_new.reshape(-1, 2) - np.asarray(pixel_pts), axis=1,
            )
            err_l2 = np.linalg.norm(
                cv2.projectPoints(world_arr, rvec, tvec, K, dist)[0]
                .reshape(-1, 2) - np.asarray(pixel_pts), axis=1,
            )
            if (np.median(err_new) <= np.median(err_l2)
                    and tvec_new.ravel()[2] > 0):
                K = K_new
                dist = dist_new
                rvec = rvec_new.ravel()
                tvec = tvec_new.ravel()
        except (ValueError, np.linalg.LinAlgError):
            pass

    proj, _ = cv2.projectPoints(world_arr, rvec, tvec, K, dist)
    err = np.linalg.norm(proj.reshape(-1, 2) - np.asarray(pixel_pts), axis=1)
    mean_err = float(np.mean(err))
    median_err = float(np.median(err))
    max_err = float(err.max())

    cam = Camera(iwidth=w, iheight=h)
    cam.calibration = K
    cam.xfocal_length = float(K[0, 0])
    cam.yfocal_length = float(K[1, 1])
    cam.principal_point = (float(K[0, 2]), float(K[1, 2]))
    R, _ = cv2.Rodrigues(rvec)
    cam.rotation = R
    cam.position = (-R.T @ tvec).ravel()
    # Stash full PnP state for the renderer (matches the other paths).
    cam._pnp_rvec = rvec.reshape(3, 1)
    cam._pnp_tvec = tvec.reshape(3, 1)
    cam._pnp_dist = dist.ravel()

    diagnostics = {
        "median_err": median_err,
        "max_err": max_err,
        "back_facing": bool(tvec[2] <= 0),
        "fx": float(K[0, 0]),
        "mode": "physical_free",
        "inliers": int(len(pixel_pts)),
        "total": int(len(pixel_pts)),
        "profile": profile_name,
        "unresolved": 0,
    }
    return cam, mean_err, diagnostics


def solve_camera_physical(
    pixel_pts: list[tuple[float, float]],
    world_3d_pts: list[tuple[float, float, float]],
    img_size: tuple[int, int],
    physical_cfg: dict,
    camera_profiles: dict,
) -> tuple[Camera | None, float, dict]:
    """Solve camera with the pipeline's physical-backend code.

    Thin adapter that constructs a ``PhysicalCalibrator`` from the
    config / profile and delegates to its
    :meth:`calibrate_correspondences` method, then unpacks the
    pipeline-shaped result dict back into the ``(Camera, mean_err,
    diagnostics)`` shape expected by the annotator.
    """
    import cv2

    from goalinsight.field_registration.physical_calibrator import (
        PhysicalCalibrator,
    )

    if len(pixel_pts) < 4:
        return None, float("inf"), {"unresolved": 0, "mode": "physical"}

    profile_name = physical_cfg.get("camera_profile", "veo_1080p")
    profile = camera_profiles.get(profile_name)
    if profile is None:
        return None, float("inf"), {
            "unresolved": 0, "mode": "physical",
            "error": f"camera_profile '{profile_name}' not found",
        }

    K_init = np.asarray(profile["K"], dtype=np.float64)
    # Brown-Conrady [k1, k2, p1, p2, k3]. profile yamls may omit it
    # (defaults to a pinhole), or supply 4-5 entries — pass straight
    # through; PhysicalCalibrator pads to 5.
    dist_init = profile.get("dist_coeffs")
    w, h = int(img_size[0]), int(img_size[1])
    focal_bounds = tuple(physical_cfg.get(
        "focal_bounds", [K_init[0, 0] * 0.5, K_init[0, 0] * 2.0],
    ))
    cam_pos = physical_cfg.get("camera_position")
    if cam_pos is not None:
        cam_pos = tuple(float(v) for v in cam_pos)

    pos_bounds = physical_cfg.get("position_bounds_m")
    if pos_bounds is not None:
        pos_bounds = tuple(pos_bounds)

    pl = float(physical_cfg.get("pitch_length", 105.0))
    pw = float(physical_cfg.get("pitch_width", 68.0))

    # Opt-in: skip PhysicalCalibrator's RANSAC + 7-DOF LM and run
    # cv2.calibrateCamera directly with full intrinsics free. PhysicalCalibrator
    # locks cx, cy at the image centre and stays on a fixed dist; for cameras
    # with off-centre principal points (lens not exactly aligned with sensor,
    # or video downstream of an asymmetric crop) the locked-centre fit can
    # land on absurd dist values to compensate. This path lets cx, cy, k1, k2,
    # p1, p2 all move — at the cost of a heavier model and the usual
    # over-fit risk on small annotation sets (< 12-15 points).
    if bool(physical_cfg.get("free_intrinsics", False)):
        cam, mean_err, diag = _calibrate_free_intrinsics(
            pixel_pts, world_3d_pts, (w, h), K_init,
            profile_name=profile_name,
        )
        return cam, mean_err, diag

    calibrator = PhysicalCalibrator(
        K=K_init,
        image_size=(w, h),
        ransac_reproj_error=float(physical_cfg.get("ransac_reproj_error", 50.0)),
        line_weight=float(physical_cfg.get("line_weight", 1.0)),
        line_sample_points=int(physical_cfg.get("line_sample_points", 20)),
        focal_bounds=focal_bounds,
        world_residual_weight=float(physical_cfg.get("world_residual_weight", 0.0)),
        world_error_threshold=float(physical_cfg.get("world_error_threshold", 5.0)),
        camera_position=cam_pos,
        position_weight=float(physical_cfg.get("position_weight", 50.0)),
        lock_camera_position=bool(physical_cfg.get("lock_camera_position", False)),
        position_bounds_m=pos_bounds,
        pitch_length=pl,
        pitch_width=pw,
        dist_coeffs=dist_init,
    )
    img_arr = np.asarray(pixel_pts, dtype=np.float64).reshape(-1, 2)
    world_arr = np.asarray(world_3d_pts, dtype=np.float64).reshape(-1, 3)

    result = calibrator.calibrate_correspondences(
        img_pts=img_arr,
        world_pts=world_arr,
    )
    if result is None:
        debug = getattr(calibrator, "_last_debug", None) or {}
        return None, float("inf"), {
            "unresolved": 0,
            "mode": "physical",
            "fail": debug.get("failure_reason", "calibrate_correspondences returned None"),
            "profile": profile_name,
        }

    cam_params = result["camera_params"]
    K_best = np.asarray(cam_params["K"], dtype=np.float64)
    rvec = np.asarray(cam_params["rvec"], dtype=np.float64).reshape(3)
    tvec = np.asarray(cam_params["tvec"], dtype=np.float64).reshape(3)
    f_best = float(cam_params["focal_length"])

    cam = Camera(iwidth=w, iheight=h)
    cam.calibration = K_best
    cam.xfocal_length = f_best
    cam.yfocal_length = f_best
    cam.principal_point = (float(K_best[0, 2]), float(K_best[1, 2]))
    R, _ = cv2.Rodrigues(rvec)
    cam.rotation = R
    cam.position = (-R.T @ tvec).ravel()
    # Stash full PnP state on the Camera object so the annotator can
    # later project pitch lines through the same (K, dist, R, t) pair
    # used during solve — bypassing the planar H approximation that
    # ignores distortion.
    cam._pnp_rvec = np.asarray(rvec, dtype=np.float64).reshape(3, 1)
    cam._pnp_tvec = np.asarray(tvec, dtype=np.float64).reshape(3, 1)
    cam._pnp_dist = np.asarray(
        cam_params.get("dist_coeffs", calibrator.dist_coeffs),
        dtype=np.float64,
    ).ravel()

    diagnostics = {
        "median_err": float(np.median(np.linalg.norm(
            result["img_pts"] - cv2.projectPoints(
                result["world_pts"].reshape(-1, 1, 3), rvec, tvec,
                K_best, calibrator.dist_coeffs,
            )[0].reshape(-1, 2),
            axis=1,
        ))),
        "max_err": float(result.get("final_error", 0.0)) * 2.0,  # rough proxy
        "back_facing": bool(tvec[2] <= 0),
        "fx": f_best,
        "mode": "physical",
        "inliers": int(result.get("inliers", 0)),
        "total": int(result.get("total_points", len(pixel_pts))),
        "profile": profile_name,
        "unresolved": 0,
    }
    mean_err = float(result["final_error"])
    return cam, mean_err, diagnostics


def camera_to_image_to_world(cam: Camera) -> np.ndarray | None:
    """Invert Camera.to_homography() into the image->world H the annotator stores."""
    H_w2i = cam.to_homography()
    try:
        return np.linalg.inv(H_w2i).astype(np.float32)
    except np.linalg.LinAlgError:
        return None


def project_camera_point(
    cam: Camera, pt_3d: tuple[float, float, float],
) -> tuple[float, float] | None:
    """Project a 3D world point through ``cam`` to pixel coordinates."""
    return _project_world_to_pixel(cam, np.asarray(pt_3d, dtype=np.float64))
