"""3D ball trajectory estimation using physics-based parabolic model.

Uses a sliding window of 2D ball detections + camera poses to fit a
parabolic 3D trajectory (constant horizontal velocity, gravity in z).
This resolves the depth ambiguity inherent in single-camera setups by
leveraging the constraint that a ball in flight follows a predictable arc.
"""

from collections import deque
from typing import Any

import cv2
import numpy as np
from scipy.optimize import least_squares


def project_3d_to_pixel(point_3d: np.ndarray, pose: dict) -> np.ndarray | None:
    """Project a 3D world point to 2D pixel coordinates.

    Args:
        point_3d: (3,) array [x, y, z] in world coordinates.
        pose: Camera pose dict with K, dist_coeffs, rvec, tvec.

    Returns:
        (2,) pixel coordinates [u, v], or None if behind camera.
    """
    rvec = np.array(pose["rvec"], dtype=np.float64).reshape(3, 1)
    tvec = np.array(pose["tvec"], dtype=np.float64).reshape(3, 1)
    K = np.array(pose["K"], dtype=np.float64)
    dist = np.array(pose["dist_coeffs"], dtype=np.float64)

    pt = np.array(point_3d, dtype=np.float64).reshape(1, 3)
    pts_2d, _ = cv2.projectPoints(pt, rvec, tvec, K, dist)
    return pts_2d.reshape(2)


def project_to_ground(pixel_center: tuple[float, float], pose: dict) -> list[float] | None:
    """Project a pixel point to the ground plane (z=0) using camera pose.

    Args:
        pixel_center: (u, v) pixel coordinates.
        pose: Camera pose dict with K, dist_coeffs, rvec, tvec.

    Returns:
        [x, y] world coordinates on ground plane, or None on failure.
    """
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
    """Compute camera origin and ray direction for a pixel observation.

    Args:
        pixel: (u, v) pixel coordinates.
        pose: Camera pose dict with K, dist_coeffs, rvec, tvec.

    Returns:
        (origin, direction) where origin is the camera center in world coords
        and direction is the normalized ray direction, or None on failure.
    """
    K = np.array(pose["K"], dtype=np.float64)
    dist = np.array(pose["dist_coeffs"], dtype=np.float64)
    R, _ = cv2.Rodrigues(np.array(pose["rvec"], dtype=np.float64))
    tvec = np.array(pose["tvec"], dtype=np.float64).flatten()

    # Undistort to normalized camera coordinates (no P matrix)
    pts = np.array([[pixel[0], pixel[1]]], dtype=np.float64).reshape(-1, 1, 2)
    pts_norm = cv2.undistortPoints(pts, K, dist).reshape(2)

    # Camera origin in world coordinates
    C = -R.T @ tvec

    # Ray direction in world coordinates (normalized)
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
    """Intersect a ray with a horizontal plane at height z.

    Args:
        origin: (3,) ray origin.
        direction: (3,) normalized ray direction.
        z: Height of the horizontal plane.

    Returns:
        (x, y) intersection point, or None if ray is parallel or behind camera.
    """
    if abs(direction[2]) < 1e-8:
        return None
    t = (z - origin[2]) / direction[2]
    if t < 0:
        return None  # intersection behind camera
    return (float(origin[0] + t * direction[0]), float(origin[1] + t * direction[1]))


def estimate_distance_from_size(
    pixel_diameter: float, pose: dict, real_diameter: float = 0.22
) -> float | None:
    """Estimate distance to ball from its apparent pixel size.

    Args:
        pixel_diameter: Ball diameter in pixels (from YOLO bbox).
        pose: Camera pose dict with K matrix.
        real_diameter: Real ball diameter in meters (default: FIFA size 5).

    Returns:
        Estimated distance in meters, or None if pixel_diameter too small.
    """
    if pixel_diameter < 1.0:
        return None
    K = np.array(pose["K"], dtype=np.float64)
    fx = K[0, 0]
    return float(real_diameter * fx / pixel_diameter)


class BallTrajectory3D:
    """Estimates 3D ball position from multi-frame observations using physics.

    Fits a parabolic trajectory model to a sliding window of (pixel, camera_pose)
    observations by minimizing reprojection error with scipy least_squares.

    Model: P(t) = [x0 + vx*dt, y0 + vy*dt, z0 + vz*dt - 0.5*g*dt²]
    where dt = t - t_ref, t_ref is the earliest observation in the window.
    """

    def __init__(self, config: dict[str, Any] | None = None):
        config = config or {}
        self.gravity = config.get("gravity", 9.81)
        self.window_size = config.get("window_size", 15)
        self.min_observations = config.get("min_observations", 4)
        self.ground_threshold = config.get("ground_threshold", 0.3)  # meters

        self.pitch_half_length = config.get("pitch_half_length", 52.5)
        self.pitch_half_width = config.get("pitch_half_width", 34.0)
        self.ball_real_diameter = config.get("ball_real_diameter", 0.22)

        self.observations: deque[tuple[float, tuple[float, float], dict, tuple | None]] = deque(
            maxlen=self.window_size
        )
        self._last_result: dict | None = None

    def add_observation(
        self,
        time_sec: float,
        pixel_center: tuple[float, float],
        camera_pose: dict,
        bbox: tuple[float, float, float, float] | None = None,
    ) -> None:
        """Add a ball observation to the sliding window.

        Detects sudden pixel acceleration (e.g. ball kicked) and resets the
        window so the optimizer doesn't try to fit across different motion
        segments (ground roll + flight in a single parabola).
        """
        if len(self.observations) >= 2:
            _, prev1, _, _ = self.observations[-1]
            _, prev2, _, _ = self.observations[-2]
            # Pixel velocities (current vs previous)
            v_old = (prev1[0] - prev2[0], prev1[1] - prev2[1])
            v_new = (pixel_center[0] - prev1[0], pixel_center[1] - prev1[1])
            # Acceleration magnitude
            accel = ((v_new[0] - v_old[0]) ** 2 + (v_new[1] - v_old[1]) ** 2) ** 0.5
            if accel > 50.0:  # Large acceleration → ball kicked, reset window
                self.observations.clear()

        self.observations.append((time_sec, pixel_center, camera_pose, bbox))

    def estimate(self) -> dict | None:
        """Fit 3D parabolic trajectory to current observations.

        Returns:
            Dict with position_3d, velocity_3d, pitch_position, height, on_ground,
            or None if insufficient observations.
        """
        if len(self.observations) < self.min_observations:
            return self._fallback_projection()

        obs_list = list(self.observations)
        t_ref = obs_list[0][0]

        x0 = self._get_initial_estimate(obs_list, t_ref)
        if x0 is None:
            return self._fallback_projection()

        # Tighten bounds to pitch dimensions
        margin = 10.0
        half_l = self.pitch_half_length + margin
        half_w = self.pitch_half_width + margin
        lower = [-half_l, -half_w, 0.0, -40, -40, -25]
        upper = [half_l, half_w, 15.0, 40, 40, 25]

        # Clamp initial estimate to bounds
        x0 = np.clip(x0, lower, upper)

        try:
            result = least_squares(
                self._residuals,
                x0,
                args=(t_ref, obs_list),
                method="trf",
                loss="cauchy",
                f_scale=5.0,
                bounds=(lower, upper),
                max_nfev=100,
            )
        except Exception:
            return self._fallback_projection()

        params = result.x

        # Always use size-based position for the latest frame.
        # The optimizer is poorly constrained with a single fixed camera and
        # produces noisy positions; size-based distance estimation uses the
        # ball's known physical diameter (22cm) to determine depth directly.
        _, latest_pixel, latest_pose, latest_bbox = obs_list[-1]
        ray_pos = self._size_estimate_position(latest_pixel, latest_bbox, latest_pose)
        if ray_pos is not None:
            x, y, z = ray_pos
        else:
            # Fallback to optimizer result
            t_latest = obs_list[-1][0]
            dt = t_latest - t_ref
            x = params[0] + params[3] * dt
            y = params[1] + params[4] * dt
            z = max(0.0, params[2] + params[5] * dt - 0.5 * self.gravity * dt * dt)

        self._last_result = {
            "position_3d": [float(x), float(y), float(z)],
            "velocity_3d": [float(params[3]), float(params[4]), float(params[5])],
            "pitch_position": [float(x), float(y)],
            "height": float(z),
            "on_ground": bool(z < self.ground_threshold),
        }
        return self._last_result

    def clear(self) -> None:
        """Clear observation window (e.g. when ball track changes)."""
        self.observations.clear()
        self._last_result = None

    def _residuals(
        self,
        params: np.ndarray,
        t_ref: float,
        obs_list: list,
    ) -> np.ndarray:
        """Compute reprojection residuals for all observations."""
        x0, y0, z0, vx, vy, vz = params
        residuals = []

        for t, pixel_center, pose, _bbox in obs_list:
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

        return np.array(residuals)

    def _size_estimate_position(
        self,
        pixel: tuple[float, float],
        bbox: tuple[float, float, float, float] | None,
        pose: dict,
    ) -> tuple[float, float, float] | None:
        """Estimate 3D position using ball's apparent pixel size.

        Uses the known real ball diameter and the YOLO bbox to compute
        distance along the camera ray. Falls back to ray-z search if
        bbox is unavailable or the result fails validation.
        """
        ray = pixel_to_ray(pixel, pose)
        if ray is None:
            return None

        origin, direction = ray
        margin = 5.0
        x_lim = self.pitch_half_length + margin
        y_lim = self.pitch_half_width + margin

        if bbox is not None:
            w = bbox[2] - bbox[0]
            h = bbox[3] - bbox[1]
            # Use min dimension — less affected by motion blur / YOLO padding
            pixel_diameter = min(w, h)
            dist = estimate_distance_from_size(
                pixel_diameter, pose, self.ball_real_diameter
            )
            if dist is not None:
                pos = origin + dist * direction
                x, y, z = float(pos[0]), float(pos[1]), float(pos[2])
                # Validate: within pitch bounds and reasonable height
                if (
                    abs(x) <= x_lim
                    and abs(y) <= y_lim
                    and -0.5 <= z <= 15.0
                ):
                    return (x, y, max(0.0, z))

        # Fall back to ray-z search
        return self._ray_estimate_position(pixel, pose)

    def _ray_estimate_position(
        self, pixel: tuple[float, float], pose: dict
    ) -> tuple[float, float, float] | None:
        """Estimate 3D position by searching for the lowest z where the
        ray intersection falls within the pitch bounds.

        Args:
            pixel: (u, v) pixel coordinates.
            pose: Camera pose dict.

        Returns:
            (x, y, z) estimated position, or None on failure.
        """
        ray = pixel_to_ray(pixel, pose)
        if ray is None:
            return None

        origin, direction = ray
        margin = 5.0
        x_lim = self.pitch_half_length + margin
        y_lim = self.pitch_half_width + margin

        # Search from ground up — take the first z where (x,y) is in bounds
        for z in [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0]:
            xy = ray_intersect_z(origin, direction, z)
            if xy is None:
                continue
            if abs(xy[0]) <= x_lim and abs(xy[1]) <= y_lim:
                return (xy[0], xy[1], z)

        # No z gives in-bounds result — clamp z=0 intersection to pitch bounds
        xy = ray_intersect_z(origin, direction, 0.0)
        if xy is not None:
            cx = float(np.clip(xy[0], -x_lim, x_lim))
            cy = float(np.clip(xy[1], -y_lim, y_lim))
            return (cx, cy, 0.0)

        return None

    def _get_initial_estimate(
        self,
        obs_list: list,
        t_ref: float,
    ) -> np.ndarray | None:
        """Compute initial parameter estimate using size-based distance."""
        t0, center0, pose0, bbox0 = obs_list[0]
        t1, center1, pose1, bbox1 = obs_list[-1]

        p0 = self._size_estimate_position(center0, bbox0, pose0)
        p1 = self._size_estimate_position(center1, bbox1, pose1)

        if p0 is None or p1 is None:
            return None

        dt = t1 - t0
        if dt < 1e-6:
            return np.array([p0[0], p0[1], p0[2], 0.0, 0.0, 0.0])

        vx = (p1[0] - p0[0]) / dt
        vy = (p1[1] - p0[1]) / dt
        # Invert parabolic model: z1 = z0 + vz*dt - 0.5*g*dt² → vz = (z1-z0)/dt + 0.5*g*dt
        vz = (p1[2] - p0[2]) / dt + 0.5 * self.gravity * dt

        return np.array([p0[0], p0[1], p0[2], vx, vy, vz])

    def _fallback_projection(self) -> dict | None:
        """Fall back to size-based position estimate of latest observation."""
        if not self.observations:
            return None

        _, pixel_center, pose, bbox = self.observations[-1]

        pos = self._size_estimate_position(pixel_center, bbox, pose)
        if pos is None:
            # Last resort: ground projection
            ground = project_to_ground(pixel_center, pose)
            if ground is None:
                return None
            return {
                "position_3d": [ground[0], ground[1], 0.0],
                "velocity_3d": [0.0, 0.0, 0.0],
                "pitch_position": ground,
                "height": 0.0,
                "on_ground": True,
            }

        return {
            "position_3d": [pos[0], pos[1], pos[2]],
            "velocity_3d": [0.0, 0.0, 0.0],
            "pitch_position": [pos[0], pos[1]],
            "height": pos[2],
            "on_ground": bool(pos[2] < self.ground_threshold),
        }
