"""Automatic anchor frame selection for homography chain calibration.

This module automatically detects high-quality anchor frames from video
by scoring frames based on pitch feature visibility and calibration quality.

Usage:
    from goalinsight.field_registration.homography_chain import AutoAnchorSelector

    selector = AutoAnchorSelector()
    selector.load_models()
    scores = selector.scan_video(video_path, start_frame, end_frame)
    anchors = selector.select_anchors(scores)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import cv2
import numpy as np

from ..distortion import DistortionCorrector

logger = logging.getLogger(__name__)


@dataclass
class FrameScore:
    """Quality score for a single frame."""

    frame_idx: int
    num_keypoints: int = 0
    num_lines: int = 0
    reprojection_error: float = float("inf")
    line_iou: float = 0.0  # Line IoU score (BroadTrack-style)
    quality_score: float = 0.0
    homography: np.ndarray | None = None
    camera_params: dict | None = None
    mean_kp_confidence: float = 0.0
    mean_line_confidence: float = 0.0
    calibration_success: bool = False


class AnchorSelectionError(Exception):
    """Raised when anchor selection fails."""

    pass


class AutoAnchorSelector:
    """Automatic anchor frame selection using pitch feature detection.

    Scans video frames and scores them based on:
    - Number of detected keypoints (0-58)
    - Number of detected lines (0-23)
    - Reprojection error from per-frame calibration
    - Mean detection confidence

    Then selects optimal anchor frames using a greedy algorithm
    that ensures spatial diversity.
    """

    def __init__(
        self,
        keypoint_backend: str = "pnlcalib",
        line_backend: str = "hough",
        keypoint_threshold: float = 0.3434,
        line_threshold: float = 0.15,
        device: str | None = None,
    ):
        """Initialize anchor selector.

        Args:
            keypoint_backend: Backend for keypoint detection ("pnlcalib" or "resnet50").
            line_backend: Backend for line detection ("pnlcalib" or "hough").
            keypoint_threshold: Confidence threshold for keypoint detection.
            line_threshold: Confidence threshold for line detection.
            device: Device for inference ("cuda", "cpu", or None for auto).
        """
        self.keypoint_backend = keypoint_backend
        self.line_backend = line_backend
        self.keypoint_threshold = keypoint_threshold
        self.line_threshold = line_threshold
        self.device = device

        self._keypoint_detector = None
        self._line_detector = None
        self._keypoint_mapper = None
        self._line_mapper = None
        self._frame_calibrator = None
        self._hough_matcher = None

        self._models_loaded = False
        self._vis_output_dir: Path | None = None
        # Hough refinement uses line endpoints as additional keypoints
        # This method is more robust than line constraint optimization
        self._enable_hough_refinement = True
        self._hough_beta = 0.15  # Weight for Hough constraints (used by line optimizer)

        # Lens distortion correction
        self._distortion_corrector = DistortionCorrector()
        self._distortion_k: float | None = None
        self._enable_distortion_correction = True

    def enable_visualization(self, output_dir: str | Path) -> None:
        """Enable visualization of anchor detection process.

        Args:
            output_dir: Directory to save visualization images.
        """
        self._vis_output_dir = Path(output_dir)
        self._vis_output_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Anchor detection visualization enabled: {self._vis_output_dir}")

    def configure_hough_refinement(
        self,
        enable: bool = True,
        beta: float = 0.3,
    ) -> None:
        """Configure Hough line refinement.

        Args:
            enable: Whether to enable Hough line refinement.
            beta: Weight for Hough constraints in optimization (0-1).
                  Higher values give more weight to Hough line alignment.
        """
        self._enable_hough_refinement = enable
        self._hough_beta = beta
        logger.info(
            f"Hough refinement {'enabled' if enable else 'disabled'}, beta={beta}"
        )

    def configure_distortion_correction(
        self,
        enable: bool = True,
        k_range: tuple[float, float] = (-1e-5, 1e-5),
        k_steps: int = 200,
    ) -> None:
        """Configure lens distortion correction.

        Args:
            enable: Whether to enable distortion correction.
            k_range: Range of k values to search (min, max).
            k_steps: Number of steps in grid search.
        """
        self._enable_distortion_correction = enable
        self._distortion_corrector = DistortionCorrector({
            "k_range": k_range,
            "k_steps": k_steps,
        })
        self._distortion_k = None  # Reset cached k
        logger.info(
            f"Distortion correction {'enabled' if enable else 'disabled'}, "
            f"k_range={k_range}, steps={k_steps}"
        )

    def load_models(self) -> None:
        """Load detection models.

        This must be called before scanning video.
        """
        from ..keypoint_detector import KeypointDetector
        from ..line_detector import LineDetector
        from ..pnlcalib import FramebyFrameCalib, HoughLineMatcher, KeypointMapper, LineMapper

        # Initialize keypoint detector
        kp_config = {
            "backend": self.keypoint_backend,
            "pnlcalib": {
                "confidence_threshold": self.keypoint_threshold,
                "weights": "SV_kp",
            },
        }
        if self.device:
            kp_config["device"] = self.device

        self._keypoint_detector = KeypointDetector(config=kp_config)
        self._keypoint_detector.load_model()

        # Initialize line detector
        line_config = {
            "backend": self.line_backend,
            "pnlcalib": {
                "confidence_threshold": self.line_threshold,
                "weights": "SV_lines",
            },
        }
        if self.device:
            line_config["device"] = self.device

        self._line_detector = LineDetector(config=line_config)
        self._line_detector.load_model()

        # Initialize mappers and calibrator
        self._keypoint_mapper = KeypointMapper()
        self._line_mapper = LineMapper()
        self._frame_calibrator = FramebyFrameCalib(image_size=(960, 540))
        self._hough_matcher = HoughLineMatcher(
            angle_tolerance=20.0,
            distance_tolerance=50.0,
        )

        self._models_loaded = True
        logger.info("Anchor selector models loaded successfully")

    def score_frame(
        self,
        frame: np.ndarray,
        frame_idx: int,
        visualize: bool = True,
    ) -> FrameScore:
        """Score a single frame for anchor quality.

        Detects features, attempts calibration, and computes quality score.

        Args:
            frame: Input frame (BGR format).
            frame_idx: Frame index.
            visualize: Whether to save visualization (if enabled).

        Returns:
            FrameScore with detection results and quality score.
        """
        if not self._models_loaded:
            raise RuntimeError("Models not loaded. Call load_models() first.")

        score = FrameScore(frame_idx=frame_idx)

        # Detect keypoints (raw PnLCalib format for calibration)
        keypoints = self._keypoint_detector.detect_pnlcalib_raw(frame)
        score.num_keypoints = len(keypoints)
        if keypoints:
            score.mean_kp_confidence = np.mean([kp.get("confidence", 0) for kp in keypoints])

        # Detect lines
        lines = self._line_detector.detect(frame)
        score.num_lines = len(lines)
        if lines:
            score.mean_line_confidence = np.mean([ln.get("confidence", 0) for ln in lines])

        # Estimate lens distortion from lines (only once, on first frame with enough lines)
        h, w = frame.shape[:2]
        if (self._enable_distortion_correction and
            self._distortion_k is None and
            len(lines) >= 3):
            self._distortion_k = self._distortion_corrector.estimate_distortion(
                lines, (w, h)
            )
            if self._distortion_k != 0:
                logger.info(f"Estimated lens distortion k={self._distortion_k:.2e}")

        # Apply distortion correction to keypoints
        if (self._enable_distortion_correction and
            self._distortion_k is not None and
            self._distortion_k != 0):
            keypoints = self._distortion_corrector.undistort_keypoints(
                keypoints, (w, h), self._distortion_k
            )

        # Attempt per-frame calibration
        # Use exclude_edge=True to reduce lens distortion effects on wide-angle cameras
        if score.num_keypoints >= 4:
            try:
                # Only use line refinement for PnLCalib lines (they have semantic labels)
                # Hough lines are only used for quality scoring
                use_lines_for_calib = self.line_backend == "pnlcalib" and len(lines) > 0
                calib_lines = lines if use_lines_for_calib else []

                self._frame_calibrator.update(keypoints, calib_lines)
                result = self._frame_calibrator.calibrate(
                    self._keypoint_mapper,
                    self._line_mapper,
                    min_confidence=0.3,
                    use_line_refinement=use_lines_for_calib,
                    exclude_edge=True,  # Exclude edge keypoints affected by lens distortion
                )

                if result is not None and result["final_error"] < 100:
                    score.homography = result["homography"]
                    score.reprojection_error = result["final_error"]
                    score.calibration_success = True

                    # Refine with Hough lines if enabled and using Hough backend
                    if (self._enable_hough_refinement and
                        self.line_backend == "hough" and
                        len(lines) > 0):
                        refined_H, hough_matches = self._refine_with_hough_lines(
                            frame, score.homography, lines
                        )
                        if refined_H is not None:
                            # Recompute error with refined homography
                            img_pts, world_pts = self._frame_calibrator.get_all_correspondences(
                                self._keypoint_mapper, 0.3, exclude_non_ground=True, exclude_edge=True
                            )
                            if len(img_pts) >= 4:
                                new_error = self._frame_calibrator._compute_reprojection_error(
                                    img_pts, world_pts, refined_H
                                )
                                # Only use refined H if error is reasonable
                                if new_error < score.reprojection_error * 1.5:
                                    score.homography = refined_H
                                    score.reprojection_error = new_error
                                    logger.debug(
                                        f"Frame {frame_idx}: Hough refinement improved, "
                                        f"{len(hough_matches)} matches"
                                    )

                    # Compute Line IoU (BroadTrack-style quality metric)
                    score.line_iou = self._compute_line_iou(frame, score.homography)

            except Exception as e:
                logger.debug(f"Frame {frame_idx} calibration failed: {e}")

        # Compute quality score
        score.quality_score = self._compute_quality_score(score)

        # Visualize if enabled
        if visualize and self._vis_output_dir is not None:
            self._visualize_frame_detection(frame, frame_idx, keypoints, lines, score)

        return score

    def _compute_line_iou(
        self,
        frame: np.ndarray,
        H: np.ndarray,
        line_width: int = 10,
    ) -> float:
        """Compute Line IoU score (BroadTrack-style).

        Projects the pitch wireframe using homography and computes IoU
        with detected lines in the frame using edge detection.

        Args:
            frame: Input frame (BGR).
            H: World->image homography (3x3).
            line_width: Width of projected lines for IoU computation.

        Returns:
            IoU score in [0, 1]. Higher is better.
        """
        h, w = frame.shape[:2]

        # Create grass mask using HSV color space (filter out sky/players)
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        # Green grass: H=35-85, S=30-255, V=30-255
        grass_mask = cv2.inRange(hsv, (35, 30, 30), (85, 255, 255))
        # Dilate grass mask to include nearby line pixels
        grass_mask = cv2.dilate(grass_mask, np.ones((15, 15), np.uint8))

        # Detect lines using Canny edge detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)

        # Also detect bright lines (white/light colored pitch markings)
        _, bright_mask = cv2.threshold(gray, 170, 255, cv2.THRESH_BINARY)

        # Combine edge and bright line detection
        detected_mask = cv2.bitwise_or(edges, bright_mask)

        # Apply grass mask to focus on pitch area only
        detected_mask = cv2.bitwise_and(detected_mask, grass_mask)

        # Dilate detected lines to improve overlap with projected lines
        detected_mask = cv2.dilate(detected_mask, np.ones((3, 3), np.uint8))

        # Create projected pitch mask
        projected_mask = np.zeros((h, w), dtype=np.uint8)

        def project(world_pt):
            """Project world point to image."""
            pt = np.array([world_pt[0], world_pt[1], 1.0])
            img_pt = H @ pt
            if abs(img_pt[2]) < 1e-6:
                return None
            return (int(img_pt[0] / img_pt[2]), int(img_pt[1] / img_pt[2]))

        def in_bounds(pt):
            if pt is None:
                return False
            return -500 < pt[0] < w + 500 and -500 < pt[1] < h + 500

        # Draw pitch lines on projected_mask
        # Center circle
        r = 9.15
        prev_pt = None
        for i in range(37):
            a = 2 * np.pi * i / 36
            pt = project((r * np.cos(a), r * np.sin(a)))
            if prev_pt and in_bounds(prev_pt) and in_bounds(pt):
                cv2.line(projected_mask, prev_pt, pt, 255, line_width)
            prev_pt = pt

        # Center line
        p1 = project((0, -34))
        p2 = project((0, 34))
        if p1 and p2 and in_bounds(p1) and in_bounds(p2):
            cv2.line(projected_mask, p1, p2, 255, line_width)

        # Touchlines
        for y in [-34, 34]:
            p1 = project((-52.5, y))
            p2 = project((52.5, y))
            if p1 and p2 and in_bounds(p1) and in_bounds(p2):
                cv2.line(projected_mask, p1, p2, 255, line_width)

        # Goal lines
        for x in [-52.5, 52.5]:
            p1 = project((x, -34))
            p2 = project((x, 34))
            if p1 and p2 and in_bounds(p1) and in_bounds(p2):
                cv2.line(projected_mask, p1, p2, 255, line_width)

        # Penalty areas
        for sign in [-1, 1]:
            x = sign * 52.5
            pts = [
                (x, -20.16), (x - sign * 16.5, -20.16),
                (x - sign * 16.5, 20.16), (x, 20.16)
            ]
            for i in range(len(pts)):
                p1 = project(pts[i])
                p2 = project(pts[(i + 1) % len(pts)])
                if p1 and p2 and in_bounds(p1) and in_bounds(p2):
                    cv2.line(projected_mask, p1, p2, 255, line_width)

        # Goal areas
        for sign in [-1, 1]:
            x = sign * 52.5
            pts = [
                (x, -9.16), (x - sign * 5.5, -9.16),
                (x - sign * 5.5, 9.16), (x, 9.16)
            ]
            for i in range(len(pts)):
                p1 = project(pts[i])
                p2 = project(pts[(i + 1) % len(pts)])
                if p1 and p2 and in_bounds(p1) and in_bounds(p2):
                    cv2.line(projected_mask, p1, p2, 255, line_width)

        # Compute IoU
        intersection = np.logical_and(detected_mask > 0, projected_mask > 0).sum()
        union = np.logical_or(detected_mask > 0, projected_mask > 0).sum()

        if union == 0:
            return 0.0

        return float(intersection / union)

    def _refine_with_hough_lines(
        self,
        frame: np.ndarray,
        H_init: np.ndarray,
        hough_lines: list[dict],
    ) -> tuple[np.ndarray | None, list[dict]]:
        """Refine homography using Hough line endpoints as additional keypoints.

        Matches Hough-detected lines to pitch template lines and uses
        the endpoints as pseudo-keypoints for refinement.

        Args:
            frame: Input frame (BGR).
            H_init: Initial homography.
            hough_lines: List of Hough-detected lines with orientation.

        Returns:
            Tuple of (refined_homography, matches).
            Returns (None, []) if refinement fails or no matches found.
        """
        if self._hough_matcher is None:
            return None, []

        h, w = frame.shape[:2]

        # Match Hough lines to template lines (unique match per template)
        matches = self._hough_matcher.match_lines(
            hough_lines, H_init, image_size=(w, h), unique_template=True
        )

        if len(matches) < 1:
            logger.debug(f"No Hough line matches found")
            return None, []

        # Extract line endpoints as pseudo-keypoints (only from close matches)
        endpoint_img_pts, endpoint_world_pts = self._hough_matcher.get_endpoint_keypoints(
            matches, H_init, max_distance=15.0
        )

        if len(endpoint_img_pts) < 2:
            logger.debug(f"Not enough line endpoints: {len(endpoint_img_pts)}")
            return None, []

        # Refine homography by adding endpoints as keypoints
        H_refined = self._frame_calibrator.refine_with_line_endpoints(
            H_init,
            self._keypoint_mapper,
            endpoint_img_pts,
            endpoint_world_pts,
            min_confidence=0.3,
            exclude_edge=True,
        )

        return H_refined, matches

    def _compute_quality_score(self, score: FrameScore) -> float:
        """Compute quality score from frame score components.

        Balanced formula combining Line IoU with keypoint metrics:
            0.30 * line_iou +                          # Line IoU (edge-based)
            0.25 * min(1.0, num_keypoints / 12.0) +   # Keypoint count
            0.20 * max(0.0, 1.0 - error / 20) +       # Reprojection error
            0.15 * mean_kp_confidence +                # Keypoint confidence
            0.10 * min(1.0, num_lines / 6.0)          # Line count

        Args:
            score: FrameScore with detection results.

        Returns:
            Quality score in [0, 1].
        """
        # Line IoU component (edge-based detection)
        iou_score = score.line_iou

        # Keypoint count component (saturates at 12 keypoints)
        kp_score = min(1.0, score.num_keypoints / 12.0)

        # Reprojection error component
        # Perfect (0px) = 1.0, bad (>=20px) = 0.0
        if score.reprojection_error < float("inf"):
            error_score = max(0.0, 1.0 - score.reprojection_error / 20)
        else:
            error_score = 0.0

        # Confidence component
        kp_conf = score.mean_kp_confidence

        # Line count component
        line_count_score = min(1.0, score.num_lines / 6.0)

        # Weighted sum - balanced between Line IoU and keypoints
        quality = (
            0.30 * iou_score
            + 0.25 * kp_score
            + 0.20 * error_score
            + 0.15 * kp_conf
            + 0.10 * line_count_score
        )

        return quality

    def scan_video(
        self,
        video_path: str | Path,
        start_frame: int | None = None,
        end_frame: int | None = None,
        sample_interval: int = 30,
        progress_callback: Callable[[int, int], None] | None = None,
    ) -> list[FrameScore]:
        """Scan video and score frames at regular intervals.

        Args:
            video_path: Path to video file.
            start_frame: Start frame (inclusive). If None, starts at 0.
            end_frame: End frame (inclusive). If None, goes to end of video.
            sample_interval: Frame sampling interval (default: 30 = 1fps for 30fps video).
            progress_callback: Optional callback(current, total) for progress updates.

        Returns:
            List of FrameScore objects for sampled frames.
        """
        if not self._models_loaded:
            raise RuntimeError("Models not loaded. Call load_models() first.")

        video_path = Path(video_path)
        if not video_path.exists():
            raise FileNotFoundError(f"Video not found: {video_path}")

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise RuntimeError(f"Failed to open video: {video_path}")

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if start_frame is None:
            start_frame = 0
        if end_frame is None:
            end_frame = total_frames - 1

        # Build frame list
        sample_frames = list(range(start_frame, end_frame + 1, sample_interval))
        if end_frame not in sample_frames:
            sample_frames.append(end_frame)

        logger.info(
            f"Scanning {len(sample_frames)} frames from {start_frame} to {end_frame} "
            f"(interval={sample_interval})"
        )

        scores = []
        for i, frame_idx in enumerate(sample_frames):
            if progress_callback:
                progress_callback(i + 1, len(sample_frames))

            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if not ret:
                logger.warning(f"Failed to read frame {frame_idx}")
                continue

            score = self.score_frame(frame, frame_idx)
            scores.append(score)

            logger.debug(
                f"Frame {frame_idx}: kp={score.num_keypoints}, lines={score.num_lines}, "
                f"error={score.reprojection_error:.1f}, quality={score.quality_score:.3f}"
            )

        cap.release()

        # Log summary
        successful = [s for s in scores if s.calibration_success]
        logger.info(
            f"Scanned {len(scores)} frames, {len(successful)} calibrated successfully"
        )

        return scores

    def select_anchors(
        self,
        scores: list[FrameScore],
        min_anchors: int = 2,
        max_anchors: int = 5,
        min_quality: float = 0.5,
        min_spacing: int = 100,
        prefer_edges: bool = True,
        edge_ratio: float = 0.1,
    ) -> list[FrameScore]:
        """Select best anchor frames using greedy algorithm.

        Selection strategy:
        1. Filter frames with quality >= min_quality
        2. Sort by quality (descending)
        3. Prefer frames near start/end of range (first/last edge_ratio)
        4. Greedy selection with min_spacing constraint
        5. Fallback: progressively lower threshold if insufficient anchors

        Args:
            scores: List of FrameScore from scan_video().
            min_anchors: Minimum number of anchors to select.
            max_anchors: Maximum number of anchors to select.
            min_quality: Minimum quality score threshold.
            min_spacing: Minimum frame spacing between anchors.
            prefer_edges: If True, prefer frames near start/end of range.
            edge_ratio: Fraction of range considered "edge" (default 10%).

        Returns:
            List of selected anchor FrameScore objects, sorted by frame index.

        Raises:
            AnchorSelectionError: If no anchors could be selected.
        """
        if not scores:
            raise AnchorSelectionError("No frames to select from")

        # Get frame range
        frame_indices = [s.frame_idx for s in scores]
        range_start = min(frame_indices)
        range_end = max(frame_indices)
        range_len = range_end - range_start + 1

        # Edge boundaries
        edge_size = int(range_len * edge_ratio)
        start_edge = range_start + edge_size
        end_edge = range_end - edge_size

        def is_edge_frame(frame_idx: int) -> bool:
            return frame_idx <= start_edge or frame_idx >= end_edge

        # Try progressively lower thresholds
        thresholds = [min_quality, 0.4, 0.3, 0.2, 0.1, 0.0]

        for threshold in thresholds:
            # Filter by quality and calibration success
            candidates = [
                s for s in scores
                if s.quality_score >= threshold and s.calibration_success
            ]

            if len(candidates) < min_anchors:
                continue

            # Sort by quality (descending), with edge preference
            def sort_key(s: FrameScore) -> tuple[int, float]:
                edge_bonus = 1 if prefer_edges and is_edge_frame(s.frame_idx) else 0
                return (edge_bonus, s.quality_score)

            candidates.sort(key=sort_key, reverse=True)

            # Greedy selection with spacing constraint
            selected = []
            for candidate in candidates:
                if len(selected) >= max_anchors:
                    break

                # Check spacing with existing selections
                too_close = False
                for sel in selected:
                    if abs(candidate.frame_idx - sel.frame_idx) < min_spacing:
                        too_close = True
                        break

                if not too_close:
                    selected.append(candidate)

            if len(selected) >= min_anchors:
                # Sort by frame index for return
                selected.sort(key=lambda s: s.frame_idx)
                logger.info(
                    f"Selected {len(selected)} anchors at threshold {threshold}: "
                    f"{[s.frame_idx for s in selected]}"
                )
                return selected

        # Fallback: use N best frames regardless of constraints
        best = sorted(
            [s for s in scores if s.calibration_success],
            key=lambda s: s.quality_score,
            reverse=True,
        )[:min_anchors]

        if best:
            best.sort(key=lambda s: s.frame_idx)
            logger.warning(
                f"Fallback: selected {len(best)} anchors (ignoring constraints): "
                f"{[s.frame_idx for s in best]}"
            )
            return best

        raise AnchorSelectionError(
            f"Could not select {min_anchors} anchor frames. "
            f"No frames with successful calibration found."
        )

    def _visualize_frame_detection(
        self,
        frame: np.ndarray,
        frame_idx: int,
        keypoints: list[dict],
        lines: list[dict],
        score: FrameScore,
    ) -> None:
        """Visualize keypoint and line detection for a single frame.

        Args:
            frame: Original frame (BGR).
            frame_idx: Frame index.
            keypoints: Detected keypoints.
            lines: Detected lines.
            score: Computed frame score.
        """
        if self._vis_output_dir is None:
            return

        vis = frame.copy()
        h, w = vis.shape[:2]

        # Draw keypoints
        for kp in keypoints:
            x, y = int(kp["x"]), int(kp["y"])
            conf = kp.get("confidence", 0)
            # Color based on confidence: green=high, yellow=medium, red=low
            if conf > 0.7:
                color = (0, 255, 0)
            elif conf > 0.4:
                color = (0, 255, 255)
            else:
                color = (0, 0, 255)
            cv2.circle(vis, (x, y), 6, color, -1)
            cv2.circle(vis, (x, y), 8, (255, 255, 255), 1)
            # Draw keypoint ID
            cv2.putText(vis, str(kp.get("id", "")), (x + 10, y - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

        # Draw lines
        for line in lines:
            x1, y1 = int(line["x1"]), int(line["y1"])
            x2, y2 = int(line["x2"]), int(line["y2"])
            conf = line.get("confidence", 0)
            # Color based on confidence
            if conf > 0.8:
                color = (255, 0, 0)  # Blue for high confidence
            else:
                color = (255, 128, 0)  # Orange for lower confidence
            cv2.line(vis, (x1, y1), (x2, y2), color, 2)
            # Draw line class name
            class_name = line.get("class_name", f"L{line.get('id', '')}")
            mid_x, mid_y = (x1 + x2) // 2, (y1 + y2) // 2
            cv2.putText(vis, class_name[:15], (mid_x, mid_y - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

        # Draw reprojection if calibration succeeded
        if score.homography is not None:
            self._draw_reprojection(vis, score.homography, color=(0, 255, 255))

        # Draw info panel
        panel_h = 185
        cv2.rectangle(vis, (10, 10), (400, panel_h), (0, 0, 0), -1)
        cv2.rectangle(vis, (10, 10), (400, panel_h), (255, 255, 255), 1)

        y_offset = 35
        cv2.putText(vis, f"Frame {frame_idx}", (20, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # Show distortion correction status
        if self._distortion_k is not None and self._distortion_k != 0:
            cv2.putText(vis, f"k={self._distortion_k:.1e}", (200, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 200), 1)

        y_offset += 25
        cv2.putText(vis, f"Keypoints: {score.num_keypoints}", (20, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        cv2.putText(vis, f"Lines: {score.num_lines}", (180, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 128, 0), 1)

        y_offset += 22
        cv2.putText(vis, f"KP Conf: {score.mean_kp_confidence:.2f}", (20, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        cv2.putText(vis, f"Line Conf: {score.mean_line_confidence:.2f}", (180, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        y_offset += 22
        if score.reprojection_error < float("inf"):
            cv2.putText(vis, f"Reproj Error: {score.reprojection_error:.1f}px", (20, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        else:
            cv2.putText(vis, "Reproj Error: N/A", (20, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 100, 100), 1)

        # Line IoU (BroadTrack-style)
        iou_color = (0, 255, 0) if score.line_iou >= 0.1 else (0, 255, 255) if score.line_iou > 0 else (100, 100, 100)
        cv2.putText(vis, f"Line IoU: {score.line_iou:.3f}", (220, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, iou_color, 1)

        y_offset += 25
        quality_color = (0, 255, 0) if score.quality_score >= 0.5 else (0, 255, 255) if score.quality_score >= 0.3 else (0, 0, 255)
        cv2.putText(vis, f"Quality Score: {score.quality_score:.3f}", (20, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, quality_color, 2)

        status = "CALIBRATED" if score.calibration_success else "FAILED"
        status_color = (0, 255, 0) if score.calibration_success else (0, 0, 255)
        cv2.putText(vis, status, (280, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 2)

        # Save
        out_path = self._vis_output_dir / f"scan_frame_{frame_idx:06d}.jpg"
        cv2.imwrite(str(out_path), vis)

    def _draw_reprojection(
        self,
        frame: np.ndarray,
        H: np.ndarray,
        color: tuple = (0, 255, 255),
    ) -> None:
        """Draw pitch reprojection using homography.

        Args:
            frame: Frame to draw on.
            H: World->image homography (3x3).
            color: Line color.
        """
        def project(world_pt):
            """Project world point to image."""
            pt = np.array([world_pt[0], world_pt[1], 1.0])
            img_pt = H @ pt
            if abs(img_pt[2]) < 1e-6:
                return None
            return (int(img_pt[0] / img_pt[2]), int(img_pt[1] / img_pt[2]))

        h, w = frame.shape[:2]

        def in_bounds(pt):
            if pt is None:
                return False
            return -100 < pt[0] < w + 100 and -100 < pt[1] < h + 100

        # Draw center circle
        r = 9.15
        prev_pt = None
        for i in range(37):
            a = 2 * np.pi * i / 36
            pt = project((r * np.cos(a), r * np.sin(a)))
            if prev_pt and in_bounds(prev_pt) and in_bounds(pt):
                cv2.line(frame, prev_pt, pt, color, 2)
            prev_pt = pt

        # Draw center line
        p1 = project((0, -34))
        p2 = project((0, 34))
        if p1 and p2 and in_bounds(p1) and in_bounds(p2):
            cv2.line(frame, p1, p2, color, 2)

        # Draw penalty areas
        for sign in [-1, 1]:
            x = sign * 52.5
            pts = [
                (x, -20.16), (x - sign * 16.5, -20.16),
                (x - sign * 16.5, 20.16), (x, 20.16)
            ]
            for i in range(len(pts)):
                p1 = project(pts[i])
                p2 = project(pts[(i + 1) % len(pts)])
                if p1 and p2 and in_bounds(p1) and in_bounds(p2):
                    cv2.line(frame, p1, p2, color, 2)

    def visualize_scores(
        self,
        scores: list[FrameScore],
        output_path: str | Path,
        selected_anchors: list[FrameScore] | None = None,
    ) -> None:
        """Generate visualization of frame scores.

        Args:
            scores: List of FrameScore from scan_video().
            output_path: Path to save plot image.
            selected_anchors: Optional list of selected anchor frames to highlight.
        """
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)

            frames = [s.frame_idx for s in scores]

            # Plot 1: Quality score
            ax1 = axes[0]
            qualities = [s.quality_score for s in scores]
            ax1.plot(frames, qualities, "b-o", markersize=3, label="Quality Score")
            ax1.axhline(0.5, color="g", linestyle="--", alpha=0.5, label="Threshold 0.5")
            if selected_anchors:
                anchor_frames = [a.frame_idx for a in selected_anchors]
                anchor_qualities = [a.quality_score for a in selected_anchors]
                ax1.scatter(anchor_frames, anchor_qualities, c="red", s=100, zorder=5,
                           marker="*", label="Selected Anchors")
            ax1.set_ylabel("Quality Score")
            ax1.set_ylim(0, 1)
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            ax1.set_title("Frame Quality Scores")

            # Plot 2: Detection counts
            ax2 = axes[1]
            keypoints = [s.num_keypoints for s in scores]
            lines = [s.num_lines for s in scores]
            ax2.plot(frames, keypoints, "g-", label="Keypoints")
            ax2.plot(frames, lines, "m-", label="Lines")
            ax2.set_ylabel("Count")
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            ax2.set_title("Detection Counts")

            # Plot 3: Reprojection error
            ax3 = axes[2]
            errors = [s.reprojection_error if s.reprojection_error < 100 else np.nan
                     for s in scores]
            ax3.plot(frames, errors, "r-o", markersize=3, label="Reprojection Error")
            ax3.set_ylabel("Error (pixels)")
            ax3.set_xlabel("Frame Index")
            ax3.set_ylim(0, 20)
            ax3.legend()
            ax3.grid(True, alpha=0.3)
            ax3.set_title("Calibration Reprojection Error")

            plt.tight_layout()
            plt.savefig(output_path, dpi=150)
            plt.close()

            logger.info(f"Saved score visualization to {output_path}")

        except ImportError:
            logger.warning("matplotlib not available, skipping visualization")

    def visualize_distortion_correction(
        self,
        frame: np.ndarray,
        lines: list[dict],
        output_path: str | Path,
    ) -> None:
        """Visualize lens distortion correction before/after comparison.

        Args:
            frame: Input frame (BGR format).
            lines: Detected lines from the frame.
            output_path: Path to save the visualization.
        """
        if self._distortion_k is None:
            logger.warning("No distortion coefficient estimated. Run score_frame() first.")
            return

        from ..distortion import DistortionVisualizer

        visualizer = DistortionVisualizer(self._distortion_corrector)
        visualizer.visualize_correction(
            frame, lines, self._distortion_k, output_path
        )

    def get_distortion_coefficient(self) -> float | None:
        """Get the estimated distortion coefficient.

        Returns:
            Distortion coefficient k, or None if not estimated.
        """
        return self._distortion_k

    def set_distortion_coefficient(self, k: float) -> None:
        """Manually set the distortion coefficient.

        Args:
            k: Distortion coefficient to use.
        """
        self._distortion_k = k
        logger.info(f"Manually set distortion coefficient k={k:.2e}")
