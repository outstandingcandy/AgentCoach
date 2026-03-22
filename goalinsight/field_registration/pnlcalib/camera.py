"""Camera model for physically-grounded calibration.

Ported from soccer-redpanda/pitch/camera.py. Provides Camera class that
extracts intrinsics from homography (Algorithm 8.2, MVG), then solves PnP
with 3D points (including goal crossbars at z=-2.44) and refines with LM.

Z convention: z-negative = up (crossbars at z=-2.44, camera z < 0).
"""

from __future__ import annotations

import logging

import cv2
import numpy as np

logger = logging.getLogger(__name__)


class Camera:
    """Camera model for soccer pitch calibration and projection."""

    def __init__(self, iwidth: int = 960, iheight: int = 540):
        self.position = np.zeros(3)
        self.rotation = np.eye(3)
        self.calibration = np.eye(3)
        self.image_width = iwidth
        self.image_height = iheight
        self.xfocal_length = 1.0
        self.yfocal_length = 1.0
        self.principal_point = (self.image_width / 2, self.image_height / 2)

    def estimate_calibration_matrix_from_plane_homography(
        self, homography: np.ndarray
    ) -> tuple[bool, np.ndarray]:
        """Extract intrinsic matrix K from plane homography.

        Based on Algorithm 8.2 of Multiple View Geometry (p225).
        Principal point is kept centered (extracted estimate is noisy).

        Returns:
            (success, K matrix)
        """
        H = np.reshape(homography, (9,))
        A = np.zeros((5, 6))
        A[0, 1] = 1.0
        A[1, 0] = 1.0
        A[1, 2] = -1.0
        A[2, 3] = self.principal_point[1] / self.principal_point[0]
        A[2, 4] = -1.0
        A[3, 0] = H[0] * H[1]
        A[3, 1] = H[0] * H[4] + H[1] * H[3]
        A[3, 2] = H[3] * H[4]
        A[3, 3] = H[0] * H[7] + H[1] * H[6]
        A[3, 4] = H[3] * H[7] + H[4] * H[6]
        A[3, 5] = H[6] * H[7]
        A[4, 0] = H[0] * H[0] - H[1] * H[1]
        A[4, 1] = 2 * H[0] * H[3] - 2 * H[1] * H[4]
        A[4, 2] = H[3] * H[3] - H[4] * H[4]
        A[4, 3] = 2 * H[0] * H[6] - 2 * H[1] * H[7]
        A[4, 4] = 2 * H[3] * H[6] - 2 * H[4] * H[7]
        A[4, 5] = H[6] * H[6] - H[7] * H[7]

        u, s, vh = np.linalg.svd(A)
        w = vh[-1]
        W = np.zeros((3, 3))
        W[0, 0] = w[0] / w[5]
        W[0, 1] = w[1] / w[5]
        W[0, 2] = w[3] / w[5]
        W[1, 0] = w[1] / w[5]
        W[1, 1] = w[2] / w[5]
        W[1, 2] = w[4] / w[5]
        W[2, 0] = w[3] / w[5]
        W[2, 1] = w[4] / w[5]
        W[2, 2] = w[5] / w[5]

        try:
            Ktinv = np.linalg.cholesky(W)
        except np.linalg.LinAlgError:
            return False, np.eye(3)

        K = np.linalg.inv(np.transpose(Ktinv))
        K /= K[2, 2]

        self.xfocal_length = K[0, 0]
        self.yfocal_length = K[1, 1]
        self.principal_point = (self.image_width / 2, self.image_height / 2)
        self.calibration = np.array([
            [self.xfocal_length, 0, self.principal_point[0]],
            [0, self.yfocal_length, self.principal_point[1]],
            [0, 0, 1],
        ], dtype="float")
        return True, K

    def from_homography(self, homography: np.ndarray) -> bool:
        """Initialize camera parameters from homography.

        Extracts K via Algorithm 8.2 (MVG), then R and t.

        Returns:
            True if successful.
        """
        success, _ = self.estimate_calibration_matrix_from_plane_homography(homography)
        if not success:
            return False

        hprim = np.linalg.inv(self.calibration) @ homography
        lambda1 = 1 / np.linalg.norm(hprim[:, 0])
        lambda2 = 1 / np.linalg.norm(hprim[:, 1])
        lambda3 = np.sqrt(lambda1 * lambda2)

        r0 = hprim[:, 0] * lambda1
        r1 = hprim[:, 1] * lambda2
        r2 = np.cross(r0, r1)

        R = np.column_stack((r0, r1, r2))
        u, s, vh = np.linalg.svd(R)
        R = u @ vh
        if np.linalg.det(R) < 0:
            u[:, 2] *= -1
            R = u @ vh
        self.rotation = R
        t = hprim[:, 2] * lambda3
        self.position = -np.transpose(R) @ t
        return True

    def solve_pnp(self, point_matches: list[tuple[np.ndarray, np.ndarray]]) -> bool:
        """Solve PnP with known K to get R, t from 3D-2D correspondences.

        Tries PnP RANSAC first (with relaxed threshold for noisy keypoints),
        falls back to SQPNP if RANSAC fails.

        Args:
            point_matches: List of (world_3d, image_2d) pairs.

        Returns:
            True if successful.
        """
        target_pts = np.array([pt[0] for pt in point_matches], dtype=np.float64)
        src_pts = np.array([pt[1] for pt in point_matches], dtype=np.float64)

        # Try RANSAC with relaxed threshold (keypoints may have 10-20px noise)
        success, rvec, t, inliers = cv2.solvePnPRansac(
            target_pts, src_pts, self.calibration, None,
            reprojectionError=20.0, iterationsCount=2000)

        if not success:
            # Fallback: SQPNP handles noisy data better than RANSAC with few points
            success, rvec, t = cv2.solvePnP(
                target_pts, src_pts, self.calibration, None,
                flags=cv2.SOLVEPNP_SQPNP)
            if not success:
                return False

        self.rotation, _ = cv2.Rodrigues(rvec)
        self.position = -np.transpose(self.rotation) @ t.flatten()
        return True

    def refine_camera(self, point_matches: list[tuple[np.ndarray, np.ndarray]]) -> None:
        """Refine camera parameters using Levenberg-Marquardt.

        Args:
            point_matches: List of (world_3d, image_2d) pairs.
        """
        rvec, _ = cv2.Rodrigues(self.rotation)
        target_pts = np.array([pt[0] for pt in point_matches], dtype=np.float64)
        src_pts = np.array([pt[1] for pt in point_matches], dtype=np.float64)

        rvec, t = cv2.solvePnPRefineLM(
            target_pts, src_pts, self.calibration, None, rvec,
            -self.rotation @ self.position,
            (cv2.TERM_CRITERIA_MAX_ITER + cv2.TERM_CRITERIA_EPS, 20000, 0.00001))
        self.rotation, _ = cv2.Rodrigues(rvec)
        self.position = -np.transpose(self.rotation) @ t

    def project_point(self, point3D: np.ndarray) -> np.ndarray:
        """Project 3D point to 2D image coordinates (no distortion).

        Returns:
            [x, y, 1] or [0, 0, 0] if behind camera.
        """
        point = point3D - self.position
        rotated_point = self.rotation @ np.transpose(point)
        if rotated_point[2] <= 1e-3:
            return np.zeros(3)
        rotated_point = rotated_point / rotated_point[2]
        x = rotated_point[0] * self.xfocal_length + self.principal_point[0]
        y = rotated_point[1] * self.yfocal_length + self.principal_point[1]
        return np.array([x, y, 1])

    def projection_rmse(self, matched_points: list[tuple[np.ndarray, np.ndarray]]) -> float:
        """Compute mean L2 projection error.

        Args:
            matched_points: List of (world_3d, image_2d) pairs.

        Returns:
            Mean L2 error in pixels.
        """
        target_pts = np.array([pt[0] for pt in matched_points])
        img_pts = np.array([pt[1] for pt in matched_points])
        projected_points = np.array([self.project_point(p3d)[:2] for p3d in target_pts])
        l2 = np.linalg.norm(img_pts - projected_points, ord=2.0, axis=-1)
        return float(np.mean(l2))

    def to_homography(self) -> np.ndarray:
        """Derive ground-plane homography from camera parameters.

        H = K @ [R[:,0] | R[:,1] | t] for z=0 plane.
        """
        t = -self.rotation @ self.position
        H = self.calibration @ np.column_stack([self.rotation[:, 0], self.rotation[:, 1], t])
        return H


def is_good_camera(cam: Camera) -> bool:
    """Validate camera parameters are physically plausible.

    Checks focal length range, position bounds, and center projection.
    Accepts z > 0 or z < 0 for camera height (supports both z-up and z-down
    conventions — our PnLCalib world coords use right-handed z-up).
    """
    focus = 10 <= cam.calibration[0, 0] <= 20000
    pos = cam.position
    pos_x = -250 < pos[0] < 250
    pos_y = -250 < pos[1] < 250
    pos_z = 0 < abs(pos[2]) < 100  # Camera must be above ground (either convention)

    if not (focus and pos_x and pos_y and pos_z):
        return False

    try:
        K = cam.calibration
        R = cam.rotation
        t = -R @ cam.position

        H_world_to_pixel = K @ np.column_stack([R[:, 0], R[:, 1], t])
        det = np.linalg.det(H_world_to_pixel)
        if abs(det) < 1e-10:
            return False
        H_pixel_to_world = np.linalg.inv(H_world_to_pixel)

        center_pixel = np.array([cam.image_width / 2, cam.image_height / 2, 1], dtype=np.float64)
        world = H_pixel_to_world @ center_pixel
        if abs(world[2]) < 1e-10:
            return False
        world = world / world[2]
        cx, cy = world[0], world[1]

        if not (np.isfinite(cx) and np.isfinite(cy)):
            return False
        if abs(cx) > 100 or abs(cy) > 100:
            return False
    except Exception:
        pass  # Don't reject solely on projection failure

    return True
