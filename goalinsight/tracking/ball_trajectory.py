"""3D ball trajectory estimation using physics-based parabolic model.

Two-pass batch architecture (replaces the old sliding-window approach):

  Pass 1 — Flight segmentation:
    Split each tracker track into motion segments at kick events (sudden
    pixel acceleration).  Within each segment, classify as ground-roll vs
    airborne using ground-plane projected speeds.

  Pass 2 — Per-segment parabola fit:
    Ground segments:  Z = 0 everywhere, use ground-plane projection for (x, y).
    Airborne segments:  Fit P(t) = [x0+vx·dt, y0+vy·dt, Vz0·dt - 0.5·g·dt²]
      - Ground-contact anchors at segment boundaries: Z(0) = 0, Z(T) = 0
      - Back-solve Vz0 = 0.5·g·T from flight duration T
      - Minimize 2D reprojection error (angular constraint)
      - NO bbox-size depth estimation (too noisy for 8-15px balls)

The flight interval is determined by physical events (kick boundaries +
ground-plane speed analysis), not by arbitrary window edges.
"""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np
from scipy.optimize import least_squares


# ---------------------------------------------------------------------------
# Projection utilities
# ---------------------------------------------------------------------------

def project_3d_to_pixel(point_3d: np.ndarray, pose: dict) -> np.ndarray | None:
    """Project a 3D world point to 2D pixel coordinates."""
    rvec = np.array(pose["rvec"], dtype=np.float64).reshape(3, 1)
    tvec = np.array(pose["tvec"], dtype=np.float64).reshape(3, 1)
    K = np.array(pose["K"], dtype=np.float64)
    dist = np.array(pose["dist_coeffs"], dtype=np.float64)
    pt = np.array(point_3d, dtype=np.float64).reshape(1, 3)
    pts_2d, _ = cv2.projectPoints(pt, rvec, tvec, K, dist)
    return pts_2d.reshape(2)


def project_to_ground(pixel_center: tuple[float, float], pose: dict) -> list[float] | None:
    """Project a pixel point to the ground plane (z=0) using camera pose."""
    K = np.array(pose["K"], dtype=np.float64)
    dist = np.array(pose["dist_coeffs"], dtype=np.float64)
    pts = np.array([[pixel_center[0], pixel_center[1]]], dtype=np.float64).reshape(-1, 1, 2)
    pts_undist = cv2.undistortPoints(pts, K, dist, P=K)
    R, _ = cv2.Rodrigues(np.array(pose["rvec"], dtype=np.float64))
    tvec = np.array(pose["tvec"], dtype=np.float64).flatten()
    H = K @ np.column_stack([R[:, 0], R[:, 1], tvec])
    try:
        H_inv = np.linalg.inv(H)
    except np.linalg.LinAlgError:
        return None
    pt = pts_undist.reshape(-1, 2)[0]
    ph = H_inv @ np.array([pt[0], pt[1], 1.0])
    if abs(ph[2]) > 1e-6:
        return [float(ph[0] / ph[2]), float(ph[1] / ph[2])]
    return None


def pixel_to_ray(
    pixel: tuple[float, float], pose: dict
) -> tuple[np.ndarray, np.ndarray] | None:
    """Compute camera origin and ray direction for a pixel observation."""
    K = np.array(pose["K"], dtype=np.float64)
    dist = np.array(pose["dist_coeffs"], dtype=np.float64)
    R, _ = cv2.Rodrigues(np.array(pose["rvec"], dtype=np.float64))
    tvec = np.array(pose["tvec"], dtype=np.float64).flatten()
    pts = np.array([[pixel[0], pixel[1]]], dtype=np.float64).reshape(-1, 1, 2)
    pts_norm = cv2.undistortPoints(pts, K, dist).reshape(2)
    C = -R.T @ tvec
    d_cam = np.array([pts_norm[0], pts_norm[1], 1.0])
    d_world = R.T @ d_cam
    norm = np.linalg.norm(d_world)
    if norm < 1e-10:
        return None
    d_world /= norm
    return C, d_world


def ray_intersect_z(
    origin: np.ndarray, direction: np.ndarray, z: float
) -> tuple[float, float] | None:
    """Intersect a ray with a horizontal plane at height z."""
    if abs(direction[2]) < 1e-8:
        return None
    t = (z - origin[2]) / direction[2]
    if t < 0:
        return None
    return (float(origin[0] + t * direction[0]), float(origin[1] + t * direction[1]))


# ---------------------------------------------------------------------------
# Kick detection (shared between trajectory fitting and event detection)
# ---------------------------------------------------------------------------

def detect_kick_frames(
    observations: list[tuple[int, tuple[float, float]]],
    accel_threshold: float = 50.0,
) -> list[int]:
    """Detect kick events from pixel-space acceleration.

    A kick is a sudden change in the ball's pixel velocity — the frame
    where it was struck.  This is more reliable than pitch-coordinate
    speed for two reasons:

    1. Pixel centers are direct observations, not derived from 3D fitting
       (which can drift for airborne balls).
    2. The same algorithm is used by the trajectory fitter to segment
       flight phases, so kick detection is consistent across the pipeline.

    Args:
        observations: list of (frame_idx, (px_x, px_y)) sorted by frame.
        accel_threshold: Pixel acceleration magnitude to trigger a kick.

    Returns:
        Sorted list of frame indices where kicks were detected (the last
        frame before the ball's velocity changed).
    """
    if len(observations) < 3:
        return []

    kick_frames: list[int] = []
    for i in range(2, len(observations)):
        _, prev2_px = observations[i - 2]
        f_prev1, prev1_px = observations[i - 1]
        _, curr_px = observations[i]

        v_old = (prev1_px[0] - prev2_px[0], prev1_px[1] - prev2_px[1])
        v_new = (curr_px[0] - prev1_px[0], curr_px[1] - prev1_px[1])
        accel = ((v_new[0] - v_old[0]) ** 2 + (v_new[1] - v_old[1]) ** 2) ** 0.5

        if accel > accel_threshold:
            kick_frames.append(f_prev1)

    return kick_frames


# ---------------------------------------------------------------------------
# Observation type used throughout
# ---------------------------------------------------------------------------
# (frame_idx, time_sec, pixel_center, camera_pose)
Obs = tuple[int, float, tuple[float, float], dict]


# ---------------------------------------------------------------------------
# BallTrajectory3D — batch-fit per track
# ---------------------------------------------------------------------------

class BallTrajectory3D:
    """Batch 3D trajectory estimator for a single ball tracker track.

    Usage (from orchestrator):
        traj3d = BallTrajectory3D(config)
        results = traj3d.fit_track(observations, fps)
        for fidx, result in results.items():
            all_ball_tracks[fidx].update(result)
    """

    def __init__(self, config: dict[str, Any] | None = None):
        config = config or {}
        self.gravity = config.get("gravity", 9.81)
        self.min_segment_length = config.get("min_observations", 4)
        self.ground_threshold = config.get("ground_threshold", 0.3)
        self.pitch_half_length = config.get("pitch_half_length", 52.5)
        self.pitch_half_width = config.get("pitch_half_width", 34.0)
        self.max_ball_speed = config.get("max_ball_speed", 50.0)
        # Ground-plane speed above which a segment is classified airborne.
        # 40 m/s is a hard shot; if ground-projected speed exceeds this,
        # the projection is wrong → ball is above ground.
        self.airborne_speed_threshold = config.get("airborne_speed_threshold", 35.0)
        # Pixel acceleration threshold to detect kicks (segment boundaries)
        self.kick_accel_threshold = config.get("kick_accel_threshold", 50.0)

    def fit_track(
        self,
        observations: list[tuple[int, tuple[float, float], dict]],
        fps: float,
    ) -> dict[int, dict]:
        """Fit 3D trajectory for an entire tracker track.

        Args:
            observations: list of (frame_idx, pixel_center, camera_pose)
                sorted by frame_idx, with predicted frames already filtered out.
            fps: Video frame rate.

        Returns:
            dict of frame_idx → {position_3d, velocity_3d, pitch_position,
            height, on_ground}. Frames where estimation fails are omitted.
        """
        if len(observations) < 2:
            return self._ground_project_all(observations)

        # Build Obs tuples with time
        obs_list: list[Obs] = [
            (fidx, fidx / fps if fps > 0 else 0.0, center, pose)
            for fidx, center, pose in observations
        ]

        # Pass 1: segment at kick boundaries
        segments = self._segment_at_kicks(obs_list)

        # Pass 2: classify and fit each segment
        results: dict[int, dict] = {}
        for seg in segments:
            if len(seg) < 2:
                results.update(self._ground_project_all_obs(seg))
                continue

            if self._is_airborne(seg, fps):
                seg_results = self._fit_airborne_segment(seg)
            else:
                seg_results = self._ground_project_all_obs(seg)
            results.update(seg_results)

        return results

    # Keep old interface for backward compatibility (orchestrator still calls
    # add_observation/estimate in a loop).  Redirect to batch fit internally.
    def clear(self) -> None:
        """Reset state (backward-compat, now a no-op)."""
        self._compat_obs: list[tuple[int, tuple[float, float], dict]] = []
        self._compat_fps: float = 0.0

    def add_observation(
        self,
        time_sec: float,
        pixel_center: tuple[float, float],
        camera_pose: dict,
        bbox: tuple[float, float, float, float] | None = None,
    ) -> None:
        """Accumulate observations (backward-compat shim).

        The bbox parameter is accepted but ignored — depth is no longer
        estimated from pixel diameter.
        """
        if not hasattr(self, "_compat_obs"):
            self._compat_obs = []
            self._compat_fps = 0.0
        # Derive frame index from time_sec (approximate)
        fidx = round(time_sec * 10) if time_sec > 0 else 0
        # Recover fps from first two observations
        if len(self._compat_obs) == 1:
            prev_t = self._compat_obs[0][0]
            dt = time_sec - prev_t / (self._compat_fps if self._compat_fps > 0 else 10.0)
            # Store actual fps for later use
        self._compat_obs.append((time_sec, pixel_center, camera_pose))

    def estimate(self) -> dict | None:
        """Backward-compat: run batch fit on accumulated observations, return latest."""
        if not hasattr(self, "_compat_obs") or len(self._compat_obs) < 1:
            return None

        obs = self._compat_obs
        # Build observations with frame indices derived from time
        fps_guess = 10.0
        if len(obs) >= 2:
            dt = obs[-1][0] - obs[-2][0]
            if dt > 0:
                fps_guess = 1.0 / dt

        obs_for_fit = []
        for t, center, pose in obs:
            fidx = round(t * fps_guess)
            obs_for_fit.append((fidx, center, pose))

        results = self.fit_track(obs_for_fit, fps_guess)

        # Return result for latest frame
        if not results:
            return None
        latest_fidx = obs_for_fit[-1][0]
        if latest_fidx in results:
            return results[latest_fidx]
        # Return any result
        return next(iter(results.values())) if results else None

    # ------------------------------------------------------------------
    # Pass 1: Segment at kick boundaries
    # ------------------------------------------------------------------

    def _segment_at_kicks(self, obs_list: list[Obs]) -> list[list[Obs]]:
        """Split observation sequence at kick boundaries.

        Delegates kick detection to the module-level `detect_kick_frames`
        (pixel-space acceleration), then splits the observation list so
        each segment contains one continuous motion phase where a single
        parabola model is valid.
        """
        if len(obs_list) < 3:
            return [obs_list]

        pixel_obs = [(fidx, px) for fidx, _, px, _ in obs_list]
        kick_set = set(detect_kick_frames(pixel_obs, self.kick_accel_threshold))

        segments: list[list[Obs]] = []
        current: list[Obs] = [obs_list[0], obs_list[1]]

        for i in range(2, len(obs_list)):
            prev1_frame = obs_list[i - 1][0]
            if prev1_frame in kick_set:
                if len(current) >= 2:
                    segments.append(current)
                current = [obs_list[i - 1]]  # overlap: kick frame starts new segment

            current.append(obs_list[i])

        if len(current) >= 2:
            segments.append(current)

        return segments if segments else [obs_list]

    # ------------------------------------------------------------------
    # Pass 1.5: Classify ground vs airborne
    # ------------------------------------------------------------------

    def _is_airborne(self, segment: list[Obs], fps: float) -> bool:
        """Detect if a segment contains airborne flight.

        Two independent signals (either one triggers airborne):

        1. Speed signal: ground-projected inter-frame speeds exceed the
           airborne threshold (35 m/s). A ground ball rarely exceeds this;
           speeds of 60+ m/s are telltale of altitude.

        2. Out-of-bounds signal: ground-projected positions fall outside
           the pitch boundaries. A ball on the ground can't be at x=-72m
           on a 91m pitch — the Z=0 projection is wrong because the ball
           is above ground, causing the ray-ground intersection to overshoot.
        """
        ground_positions = []
        for fidx, t, center, pose in segment:
            gp = project_to_ground(center, pose)
            if gp is not None:
                ground_positions.append((fidx, t, gp))

        if len(ground_positions) < 2:
            return False

        margin = 3.0  # small margin for projection noise near boundary
        hl = self.pitch_half_length + margin
        hw = self.pitch_half_width + margin

        speed_violations = 0
        oob_count = 0
        n_pairs = len(ground_positions) - 1

        for i, (fidx, t, gp) in enumerate(ground_positions):
            # Check out-of-bounds
            if abs(gp[0]) > hl or abs(gp[1]) > hw:
                oob_count += 1

            # Check speed
            if i > 0:
                _, t0, p0 = ground_positions[i - 1]
                dt = t - t0
                if dt > 1e-6:
                    dist = ((gp[0] - p0[0]) ** 2 + (gp[1] - p0[1]) ** 2) ** 0.5
                    if dist / dt > self.airborne_speed_threshold:
                        speed_violations += 1

        # Signal 1: >20% of frame pairs exceed speed threshold
        if n_pairs > 0 and speed_violations / n_pairs > 0.2:
            return True

        # Signal 2: >30% of positions are outside pitch bounds
        if oob_count / len(ground_positions) > 0.3:
            return True

        return False

    # ------------------------------------------------------------------
    # Pass 2a: Ground segment — simple projection
    # ------------------------------------------------------------------

    def _ground_project_all(
        self, observations: list[tuple[int, tuple[float, float], dict]]
    ) -> dict[int, dict]:
        """Ground-project raw observations (no Obs wrapper)."""
        results = {}
        for fidx, center, pose in observations:
            gp = project_to_ground(center, pose)
            if gp is None:
                continue
            results[fidx] = {
                "position_3d": [gp[0], gp[1], 0.0],
                "velocity_3d": [0.0, 0.0, 0.0],
                "pitch_position": [gp[0], gp[1]],
                "height": 0.0,
                "on_ground": True,
            }
        return results

    def _ground_project_all_obs(self, segment: list[Obs]) -> dict[int, dict]:
        """Ground-project Obs-wrapped observations."""
        results = {}
        for fidx, t, center, pose in segment:
            gp = project_to_ground(center, pose)
            if gp is None:
                continue
            results[fidx] = {
                "position_3d": [gp[0], gp[1], 0.0],
                "velocity_3d": [0.0, 0.0, 0.0],
                "pitch_position": [gp[0], gp[1]],
                "height": 0.0,
                "on_ground": True,
            }
        return results

    # ------------------------------------------------------------------
    # Pass 2b: Airborne segment — global parabola fit
    # ------------------------------------------------------------------

    def _fit_airborne_segment(self, segment: list[Obs]) -> dict[int, dict]:
        """Fit a single parabola to an airborne segment.

        Initial estimate uses ONLY the first 2–3 frames (near launch, Z≈0
        so ground projection is still reliable) to compute (x0, y0, vx, vy).
        The last frame's ground projection is NOT used — it's wildly wrong
        for an airborne ball.

        Velocity bounds are tightened dynamically based on segment duration
        so that positions stay within 2× pitch dimensions throughout.
        """
        if len(segment) < self.min_segment_length:
            return self._ground_project_all_obs(segment)

        t_ref = segment[0][1]
        T = segment[-1][1] - t_ref
        if T < 1e-6:
            return self._ground_project_all_obs(segment)

        # --- Initial estimate from first 2–3 frames (near ground) ---
        # Ground projection is reliable only near Z≈0 (launch).
        # Use the first two frames with valid ground projection.
        early_gps: list[tuple[float, float, float]] = []  # (dt, x, y)
        for fidx, t, center, pose in segment[:min(4, len(segment))]:
            gp = project_to_ground(center, pose)
            if gp is not None:
                early_gps.append((t - t_ref, gp[0], gp[1]))
            if len(early_gps) >= 2:
                break

        if len(early_gps) < 2:
            return self._ground_project_all_obs(segment)

        x0_est = early_gps[0][1]
        y0_est = early_gps[0][2]
        dt01 = early_gps[1][0] - early_gps[0][0]
        if dt01 < 1e-6:
            return self._ground_project_all_obs(segment)

        vx_est = (early_gps[1][1] - early_gps[0][1]) / dt01
        vy_est = (early_gps[1][2] - early_gps[0][2]) / dt01

        # Clamp horizontal velocity to max ball speed
        max_v = self.max_ball_speed  # 50 m/s
        vx_est = float(np.clip(vx_est, -max_v, max_v))
        vy_est = float(np.clip(vy_est, -max_v, max_v))

        # Vz0 from Z(T) = 0 constraint, capped for reasonable apex height
        # Max apex = vz²/(2g); cap at 10m → vz_max = sqrt(2*g*10) ≈ 14 m/s
        vz_est = min(0.5 * self.gravity * T, 14.0)

        x0_params = np.array([x0_est, y0_est, 0.0, vx_est, vy_est, vz_est])

        # --- Dynamic velocity bounds ---
        # Ensure positions stay within 2× pitch dimensions throughout segment.
        # |x0 + vx*T| < 2*hl  →  vx ∈ [-(2*hl + x0)/T, (2*hl - x0)/T]
        margin = 10.0
        hl = self.pitch_half_length + margin
        hw = self.pitch_half_width + margin
        if T > 0.1:
            vx_lo = max(-max_v, -(2 * hl + abs(x0_est)) / T)
            vx_hi = min(max_v, (2 * hl + abs(x0_est)) / T)
            vy_lo = max(-max_v, -(2 * hw + abs(y0_est)) / T)
            vy_hi = min(max_v, (2 * hw + abs(y0_est)) / T)
        else:
            vx_lo, vx_hi = -max_v, max_v
            vy_lo, vy_hi = -max_v, max_v

        lower = [-hl, -hw, -0.5, vx_lo, vy_lo, 0.0]
        upper = [hl, hw, 2.0, vx_hi, vy_hi, 15.0]
        x0_params = np.clip(x0_params, lower, upper)

        try:
            result = least_squares(
                self._airborne_residuals,
                x0_params,
                args=(t_ref, segment),
                method="trf",
                loss="cauchy",
                f_scale=5.0,
                bounds=(lower, upper),
                max_nfev=200,
            )
            params = result.x
        except Exception:
            return self._ground_project_all_obs(segment)

        # Evaluate parabola at each frame
        x0, y0, z0, vx, vy, vz = params
        results: dict[int, dict] = {}

        for fidx, t, center, pose in segment:
            dt = t - t_ref
            px = x0 + vx * dt
            py = y0 + vy * dt
            pz = z0 + vz * dt - 0.5 * self.gravity * dt * dt
            pz = max(0.0, pz)

            results[fidx] = {
                "position_3d": [float(px), float(py), float(pz)],
                "velocity_3d": [float(vx), float(vy), float(vz - self.gravity * dt)],
                "pitch_position": [float(px), float(py)],
                "height": float(pz),
                "on_ground": bool(pz < self.ground_threshold),
            }

        return results

    def _airborne_residuals(
        self,
        params: np.ndarray,
        t_ref: float,
        segment: list[Obs],
    ) -> np.ndarray:
        """Residuals for airborne parabola fit.

        Three constraint types:
          1. 2D reprojection: angular position must match pixel observations
          2. Ground-contact: Z ≈ 0 at segment start and end
          3. Position range: penalize positions outside pitch boundaries
        """
        x0, y0, z0, vx, vy, vz = params
        T = segment[-1][1] - t_ref
        hl = self.pitch_half_length + 5.0  # small margin
        hw = self.pitch_half_width + 5.0
        residuals = []

        # 2D reprojection residuals
        for _, t, pixel_center, pose in segment:
            dt = t - t_ref
            px = x0 + vx * dt
            py = y0 + vy * dt
            pz = z0 + vz * dt - 0.5 * self.gravity * dt * dt

            pred_2d = project_3d_to_pixel(np.array([px, py, pz]), pose)
            if pred_2d is None:
                residuals.extend([0.0, 0.0])
                continue
            residuals.append(pred_2d[0] - pixel_center[0])
            residuals.append(pred_2d[1] - pixel_center[1])

        # Ground-contact constraints at segment boundaries
        gc_weight = 3.0

        # Launch: Z(0) = z0 ≈ 0
        residuals.append(gc_weight * z0)

        # Landing: Z(T) ≈ 0
        z_end = z0 + vz * T - 0.5 * self.gravity * T * T
        residuals.append(gc_weight * z_end)

        # Non-negativity at apex
        if self.gravity > 0:
            t_apex = vz / self.gravity
            if 0 < t_apex < T:
                z_apex = z0 + vz * t_apex - 0.5 * self.gravity * t_apex * t_apex
                residuals.append(gc_weight * max(0.0, -z_apex))

        # Position range: penalize trajectory leaving the pitch at each frame
        range_weight = 2.0
        for _, t, _, _ in segment:
            dt = t - t_ref
            px = x0 + vx * dt
            py = y0 + vy * dt
            # Soft penalty: only activates outside pitch + margin
            residuals.append(range_weight * max(0.0, abs(px) - hl))
            residuals.append(range_weight * max(0.0, abs(py) - hw))

        return np.array(residuals)
