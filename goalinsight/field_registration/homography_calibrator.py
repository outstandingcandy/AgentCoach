"""Homography-based calibrator for ground-plane projection.

Computes a 3×3 homography directly from 2D image ↔ 2D world correspondences
via DLT (cv2.findHomography), without decomposing into intrinsic/extrinsic
camera parameters. Only ground-plane keypoints (z=0) are used.

Inspired by StatsBomb's approach:
https://blogarchive.statsbomb.com/articles/football/creating-better-data-how-to-map-homography/
"""

from __future__ import annotations

import logging
from typing import Any

import cv2
import numpy as np

from .physical_calibrator import (
    LINE_KEYPOINTS,
    build_field_template,
)
from .pnlcalib.curve_utils import sample_points_on_image_line

logger = logging.getLogger(__name__)

# Non-ground keypoint IDs (crossbar/goal post tops at z=-2.44m)
NON_GROUND_KP_IDS = {12, 14, 16, 18}


class HomographyCalibrator:
    """Ground-plane homography calibrator using DLT.

    Estimates a 3×3 world→image homography from ground-plane keypoints.
    No camera intrinsics or 3D pose are computed.
    """

    def __init__(
        self,
        image_size: tuple[int, int],
        ransac_reproj_error: float = 10.0,
        max_reproj_error: float = 15.0,
        world_error_threshold: float = 5.0,
        pitch_length: float = 105.0,
        pitch_width: float = 68.0,
    ):
        self.width, self.height = image_size
        self.ransac_reproj_error = ransac_reproj_error
        self.max_reproj_error = max_reproj_error
        self.world_error_threshold = world_error_threshold
        self.pitch_length = pitch_length
        self.pitch_width = pitch_width

        self._field_world_coords, self._field_line_defs, self._field_intersections = \
            build_field_template(pitch_length, pitch_width)

        self._keypoints: list[dict] = []
        self._lines: list[dict] = []

    def update(self, keypoints: list[dict], lines: list[dict]):
        """Update with detected keypoints and lines for current frame."""
        self._keypoints = keypoints
        self._lines = lines

    def calibrate(
        self,
        keypoint_mapper,
        line_mapper=None,
        min_confidence: float = 0.3,
        initial_rvec: np.ndarray | None = None,
        initial_tvec: np.ndarray | None = None,
    ) -> dict[str, Any] | None:
        """Estimate ground-plane homography from keypoint correspondences.

        Args:
            keypoint_mapper: KeypointMapper for 2D-3D correspondences.
            line_mapper: Unused, kept for interface compatibility.
            min_confidence: Minimum keypoint confidence threshold.
            initial_rvec: Unused, kept for interface compatibility.
            initial_tvec: Unused, kept for interface compatibility.

        Returns:
            Result dict with homography and stats, or None if calibration failed.
        """
        # Step 1: Extract ground-plane correspondences
        img_pts, world_pts_2d, kp_ids = self._prepare_ground_correspondences(
            keypoint_mapper, min_confidence
        )

        # Step 2: Add line-line intersection points
        isect_img, isect_world, isect_ids = self._compute_line_intersections(
            kp_ids, img_pts, keypoint_mapper
        )
        n_intersections = len(isect_ids)
        if n_intersections > 0:
            img_pts = np.vstack([img_pts, isect_img])
            world_pts_2d = np.vstack([world_pts_2d, isect_world])
            kp_ids = list(kp_ids) + isect_ids

        if len(kp_ids) < 4:
            logger.debug("Too few ground keypoints (%d < 4), skipping", len(kp_ids))
            return None

        # Step 3: Compute initial homography via LMEDS (robust median estimator)
        # LMEDS minimizes the median error and handles up to 50% outliers without
        # needing a reproj threshold — avoids biased fits from clustered point subsets.
        H, mask = cv2.findHomography(
            world_pts_2d.reshape(-1, 1, 2),
            img_pts.reshape(-1, 1, 2),
            cv2.LMEDS,
        )
        if H is None:
            logger.debug("findHomography (LMEDS) failed")
            return None

        inlier_mask = mask.ravel().astype(bool)

        # Step 4: Iterative world-error outlier rejection with adaptive threshold.
        # Points near the vanishing point have high H_inv magnification, so a small
        # pixel error becomes a large world error. Scale the threshold per-point by
        # the local magnification factor to avoid rejecting valid far-field points.
        if np.isfinite(self.world_error_threshold):
            for _ in range(3):
                world_errs = self._per_point_world_error(H, img_pts, world_pts_2d)
                magnification = self._local_magnification(H, img_pts)
                median_mag = max(np.median(magnification), 1e-6)
                adaptive_thresh = self.world_error_threshold * np.maximum(
                    magnification / median_mag, 1.0
                )
                keep = world_errs < adaptive_thresh
                n_removed = int(np.sum(~keep))
                if n_removed == 0:
                    break
                if int(np.sum(keep)) < 4:
                    break
                img_pts = img_pts[keep]
                world_pts_2d = world_pts_2d[keep]
                kp_ids = [kp_ids[i] for i in range(len(kp_ids)) if keep[i]]

                H_new, mask_new = cv2.findHomography(
                    world_pts_2d.reshape(-1, 1, 2),
                    img_pts.reshape(-1, 1, 2),
                    cv2.LMEDS,
                )
                if H_new is None:
                    break
                H = H_new
                inlier_mask = mask_new.ravel().astype(bool)

        # Final least-squares refinement using ALL cleaned points (no filtering)
        if len(img_pts) >= 4:
            H_refined, _ = cv2.findHomography(
                world_pts_2d.reshape(-1, 1, 2),
                img_pts.reshape(-1, 1, 2),
                0,  # method=0: regular DLT least-squares, uses all points
            )
            if H_refined is not None:
                H = H_refined
                inlier_mask = np.ones(len(img_pts), dtype=bool)

        # Normalize homography
        if abs(H[2, 2]) > 1e-10:
            H = H / H[2, 2]

        # Step 5: Compute stats
        reproj_err = self._reprojection_error(H, img_pts, world_pts_2d)
        world_err = self._mean_world_error(H, img_pts, world_pts_2d)
        per_pt_world_err = self._per_point_world_error(H, img_pts, world_pts_2d)

        # Sanity check
        if reproj_err > 200:
            logger.warning("Homography rejected (reproj_err=%.0fpx)", reproj_err)
            return None

        # Build world_pts with z=0 column for compatibility
        world_pts_3d = np.column_stack([world_pts_2d, np.zeros(len(world_pts_2d))])

        return {
            "homography": H,
            "final_error": float(reproj_err),
            "world_error": float(world_err),
            "world_error_all": float(world_err),
            "per_point_world_errors": per_pt_world_err,
            "per_kp_world_errors": {},
            "num_keypoints": len(self._keypoints),
            "num_lines": 0,
            "num_intersections": n_intersections,
            "total_points": len(img_pts),
            "inliers": int(np.sum(inlier_mask)),
            "intrinsics_init": None,
            "camera_params": None,
            "camera_pose": None,
            "img_pts": img_pts,
            "world_pts": world_pts_3d,
            "kp_ids": kp_ids,
            "inlier_mask": inlier_mask,
            "line_constraints_count": 0,
            "line_constraints": [],
            "ransac_info": None,
        }

    def _prepare_ground_correspondences(self, keypoint_mapper, min_confidence):
        """Extract ground-plane (z=0) keypoint correspondences."""
        img_pts, world_pts, kp_ids = keypoint_mapper.build_3d_correspondence_matrix(
            self._keypoints,
            filter_by_confidence=min_confidence,
            exclude_non_ground=False,
        )

        if len(world_pts) == 0:
            return (
                np.zeros((0, 2), dtype=np.float64),
                np.zeros((0, 2), dtype=np.float64),
                [],
            )

        # Override world coordinates with field template
        world_pts = world_pts.copy()
        for idx, kid in enumerate(kp_ids):
            if kid < len(self._field_world_coords):
                wx, wy = self._field_world_coords[kid]
                z = world_pts[idx, 2]
                world_pts[idx] = [wx, wy, z]

        # Filter to ground-plane only (z == 0)
        ground_mask = np.abs(world_pts[:, 2]) < 0.01
        img_pts = img_pts[ground_mask].astype(np.float64)
        world_pts_2d = world_pts[ground_mask, :2].astype(np.float64)
        kp_ids = [kp_ids[i] for i in range(len(kp_ids)) if ground_mask[i]]

        return img_pts, world_pts_2d, kp_ids

    def _compute_line_intersections(self, existing_kp_ids, existing_img_pts, keypoint_mapper):
        """Compute line-line intersection points as additional correspondences."""
        # Build detected keypoint lookup
        detected = {}
        for i, kid in enumerate(existing_kp_ids):
            detected[kid] = existing_img_pts[i]

        non_ground = keypoint_mapper.NON_GROUND_KEYPOINTS

        # Derive image lines from detected keypoints
        line_endpoints = {}
        for line_id, kp_list in LINE_KEYPOINTS.items():
            found = [(kid, detected[kid]) for kid in kp_list
                     if kid in detected and kid not in non_ground]
            if len(found) < 2:
                continue
            line_endpoints[line_id] = (found[0][1], found[-1][1])

        existing_set = set(existing_kp_ids)
        margin = 50

        int_img = []
        int_world = []
        int_ids = []

        for (lid_a, lid_b), (wx, wy, wz, kp_id) in self._field_intersections.items():
            if lid_a not in line_endpoints or lid_b not in line_endpoints:
                continue
            if kp_id in existing_set:
                continue

            p1a, p2a = line_endpoints[lid_a]
            p1b, p2b = line_endpoints[lid_b]

            x1, y1 = float(p1a[0]), float(p1a[1])
            x2, y2 = float(p2a[0]), float(p2a[1])
            x3, y3 = float(p1b[0]), float(p1b[1])
            x4, y4 = float(p2b[0]), float(p2b[1])

            denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
            if abs(denom) < 1e-6:
                continue

            t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / denom
            ix = x1 + t * (x2 - x1)
            iy = y1 + t * (y2 - y1)

            if ix < -margin or ix > self.width + margin:
                continue
            if iy < -margin or iy > self.height + margin:
                continue

            int_img.append([ix, iy])
            int_world.append([wx, wy])
            int_ids.append(f"isect_{lid_a}_{lid_b}")

        if not int_img:
            return np.zeros((0, 2), dtype=np.float64), np.zeros((0, 2), dtype=np.float64), []

        return (
            np.array(int_img, dtype=np.float64),
            np.array(int_world, dtype=np.float64),
            int_ids,
        )

    @staticmethod
    def _local_magnification(H, img_pts):
        """Compute local meters-per-pixel magnification of H_inv at each image point.

        Returns an array of magnification factors (one per point). Points near the
        vanishing point will have large values, meaning small pixel errors produce
        large world-coordinate errors.
        """
        H_inv = np.linalg.inv(H)
        h = H_inv
        mags = np.empty(len(img_pts))
        for i, (u, v) in enumerate(img_pts):
            d = h[2, 0] * u + h[2, 1] * v + h[2, 2]
            if abs(d) < 1e-10:
                mags[i] = 1e6
                continue
            d2 = d * d
            # Jacobian of H_inv: d(world)/d(pixel)
            # J = [[dx/du, dx/dv], [dy/du, dy/dv]]
            dxdu = (h[0, 0] * d - h[2, 0] * (h[0, 0] * u + h[0, 1] * v + h[0, 2])) / d2
            dxdv = (h[0, 1] * d - h[2, 1] * (h[0, 0] * u + h[0, 1] * v + h[0, 2])) / d2
            dydu = (h[1, 0] * d - h[2, 0] * (h[1, 0] * u + h[1, 1] * v + h[1, 2])) / d2
            dydv = (h[1, 1] * d - h[2, 1] * (h[1, 0] * u + h[1, 1] * v + h[1, 2])) / d2
            # Magnification = sqrt(|det(J)|), i.e. local area scale
            det_j = abs(dxdu * dydv - dxdv * dydu)
            mags[i] = max(det_j ** 0.5, 1e-6)
        return mags

    @staticmethod
    def _reprojection_error(H, img_pts, world_pts_2d):
        """Mean reprojection error (world→image) in pixels."""
        pts_h = np.column_stack([world_pts_2d, np.ones(len(world_pts_2d))])
        proj = (H @ pts_h.T).T
        proj_2d = proj[:, :2] / proj[:, 2:3]
        return float(np.mean(np.linalg.norm(proj_2d - img_pts, axis=1)))

    @staticmethod
    def _mean_world_error(H, img_pts, world_pts_2d):
        """Mean world-space error (image→world) in meters."""
        H_inv = np.linalg.inv(H)
        pts_h = np.column_stack([img_pts, np.ones(len(img_pts))])
        proj = (H_inv @ pts_h.T).T
        proj_2d = proj[:, :2] / proj[:, 2:3]
        return float(np.mean(np.linalg.norm(proj_2d - world_pts_2d, axis=1)))

    @staticmethod
    def _per_point_world_error(H, img_pts, world_pts_2d):
        """Per-point world-space error (image→world) in meters."""
        H_inv = np.linalg.inv(H)
        pts_h = np.column_stack([img_pts, np.ones(len(img_pts))])
        proj = (H_inv @ pts_h.T).T
        proj_2d = proj[:, :2] / proj[:, 2:3]
        return np.linalg.norm(proj_2d - world_pts_2d, axis=1)
