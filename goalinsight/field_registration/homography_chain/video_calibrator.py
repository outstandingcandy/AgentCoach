"""High-level video calibration API with automatic anchor detection.

This module provides a simple interface for calibrating video segments
without requiring pre-computed calibration data.

Usage:
    from goalinsight.field_registration.homography_chain import VideoCalibrator

    calibrator = VideoCalibrator()
    results = calibrator.calibrate_video(
        video_path="video.mp4",
        start_frame=680,
        end_frame=1128,
    )
    calibrator.export_results("calibration.json")
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Callable

import cv2
import numpy as np

from .auto_anchor_selector import AnchorSelectionError, AutoAnchorSelector, FrameScore
from .chain_calibrator import ChainCalibrator

logger = logging.getLogger(__name__)


class VideoCalibrator:
    """High-level video calibration with automatic anchor detection.

    Pipeline:
    1. Load keypoint and line detection models
    2. Scan video with AutoAnchorSelector
    3. Select high-quality anchor frames
    4. Create ChainCalibrator and add anchors
    5. Run calibrate_range() for all frames

    This class provides a complete solution for calibrating video segments
    where direct calibration (BroadTrack/PnLCalib) fails.
    """

    def __init__(
        self,
        keypoint_backend: str = "pnlcalib",
        line_backend: str = "pnlcalib",
        chain_method: str = "improved",
        frame_step: int = 1,
        device: str | None = None,
    ):
        """Initialize video calibrator.

        Args:
            keypoint_backend: Backend for keypoint detection.
            line_backend: Backend for line detection.
            chain_method: Method for chain calibration ("improved" or "legacy").
            frame_step: Frame step for calibration (1=every frame, 30=1fps).
            device: Device for inference ("cuda", "cpu", or None for auto).
        """
        self.keypoint_backend = keypoint_backend
        self.line_backend = line_backend
        self.chain_method = chain_method
        self.frame_step = frame_step
        self.device = device

        self._anchor_selector: AutoAnchorSelector | None = None
        self._chain_calibrator: ChainCalibrator | None = None

        self._models_loaded = False
        self._results: dict[int, dict[str, Any]] = {}
        self._selected_anchors: list[FrameScore] = []
        self._frame_scores: list[FrameScore] = []

    def load_models(self) -> None:
        """Load detection models.

        This is called automatically by calibrate_video() if not already loaded.
        """
        if self._models_loaded:
            return

        self._anchor_selector = AutoAnchorSelector(
            keypoint_backend=self.keypoint_backend,
            line_backend=self.line_backend,
            device=self.device,
        )
        self._anchor_selector.load_models()

        self._models_loaded = True
        logger.info("VideoCalibrator models loaded")

    def calibrate_video(
        self,
        video_path: str | Path,
        start_frame: int | None = None,
        end_frame: int | None = None,
        sample_interval: int = 30,
        min_anchors: int = 2,
        max_anchors: int = 5,
        min_anchor_quality: float = 0.5,
        min_anchor_spacing: int = 100,
        progress_callback: Callable[[str, int, int], None] | None = None,
    ) -> dict[int, dict[str, Any]]:
        """Calibrate a video segment with automatic anchor detection.

        Args:
            video_path: Path to video file.
            start_frame: Start frame (inclusive). If None, starts at 0.
            end_frame: End frame (inclusive). If None, goes to end of video.
            sample_interval: Frame interval for anchor scanning (default: 30).
            min_anchors: Minimum number of anchor frames to select.
            max_anchors: Maximum number of anchor frames to select.
            min_anchor_quality: Minimum quality score for anchor selection.
            min_anchor_spacing: Minimum frame spacing between anchors.
            progress_callback: Optional callback(stage, current, total).
                              Stages: "scanning", "calibrating"

        Returns:
            Dictionary mapping frame_idx to calibration results.
            Each result contains:
                - camera_params: Camera parameters dict
                - confidence: Calibration confidence [0, 1]
                - source: "anchor" or propagation type

        Raises:
            FileNotFoundError: If video file not found.
            AnchorSelectionError: If anchor selection fails.
            RuntimeError: If calibration fails.
        """
        video_path = Path(video_path)
        if not video_path.exists():
            raise FileNotFoundError(f"Video not found: {video_path}")

        # Load models if needed
        self.load_models()

        # Enable anchor selector visualization if vis_dir was set before load_models
        if hasattr(self, "_vis_output_dir") and self._vis_output_dir and self._anchor_selector:
            self._anchor_selector.enable_visualization(self._vis_output_dir / "anchor_scan")

        # Get video info
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise RuntimeError(f"Failed to open video: {video_path}")

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()

        if start_frame is None:
            start_frame = 0
        if end_frame is None:
            end_frame = total_frames - 1

        logger.info(
            f"Calibrating video: {video_path.name}, "
            f"frames {start_frame}-{end_frame}, size={width}x{height}"
        )

        # Step 1: Scan for anchors
        logger.info("Step 1: Scanning for anchor frames...")

        def scan_progress(current, total):
            if progress_callback:
                progress_callback("scanning", current, total)

        self._frame_scores = self._anchor_selector.scan_video(
            video_path,
            start_frame=start_frame,
            end_frame=end_frame,
            sample_interval=sample_interval,
            progress_callback=scan_progress,
        )

        # Step 2: Select anchors
        logger.info("Step 2: Selecting anchor frames...")

        self._selected_anchors = self._anchor_selector.select_anchors(
            self._frame_scores,
            min_anchors=min_anchors,
            max_anchors=max_anchors,
            min_quality=min_anchor_quality,
            min_spacing=min_anchor_spacing,
        )

        logger.info(
            f"Selected {len(self._selected_anchors)} anchors: "
            f"{[a.frame_idx for a in self._selected_anchors]}"
        )

        # Step 3: Build camera params from anchors
        # For auto-detected anchors, we need to convert homography to camera params
        anchor_params = self._build_anchor_camera_params(width, height)

        # Step 4: Create chain calibrator and add anchors
        logger.info("Step 3: Running homography chain calibration...")

        self._chain_calibrator = ChainCalibrator(
            image_width=width,
            image_height=height,
            mode="offline",
            method=self.chain_method,
            frame_step=self.frame_step,
        )

        # Enable visualization if vis_dir was set
        if hasattr(self, "_vis_output_dir") and self._vis_output_dir:
            self._chain_calibrator.enable_visualization(self._vis_output_dir)

        for anchor in self._selected_anchors:
            if anchor.frame_idx in anchor_params:
                self._chain_calibrator.add_anchor(
                    frame_idx=anchor.frame_idx,
                    camera_params=anchor_params[anchor.frame_idx],
                    homography=anchor.homography,  # Use original homography for accurate projection
                    confidence=anchor.quality_score,
                )

        # Step 5: Run calibration
        def calib_progress(current, total):
            if progress_callback:
                progress_callback("calibrating", current, total)

        self._results = self._chain_calibrator.calibrate_range(
            video_path=video_path,
            start_frame=start_frame,
            end_frame=end_frame,
            progress_callback=calib_progress,
        )

        logger.info(f"Calibration complete: {len(self._results)} frames calibrated")

        return self._results

    def _build_anchor_camera_params(
        self,
        image_width: int,
        image_height: int,
    ) -> dict[int, dict[str, Any]]:
        """Convert anchor homographies to camera parameters.

        Uses homography decomposition to estimate camera parameters
        from the detected homography at each anchor frame.

        Args:
            image_width: Image width in pixels.
            image_height: Image height in pixels.

        Returns:
            Dictionary mapping frame_idx to camera params.
        """
        from .camera_param_converter import CameraParamConverter

        converter = CameraParamConverter(
            image_width=image_width,
            image_height=image_height,
        )

        anchor_params = {}

        # Use default camera params as reference (BroadTrack convention)
        # Note: Z is negative for elevated camera in BroadTrack coordinate system
        default_params = {
            "panDegrees": 0.0,
            "tiltDegrees": 80.0,  # Looking down steeply (0=horizontal, 90=straight down)
            "rollDegrees": 0.0,
            "horizontalFieldOfViewDegrees": 100.0,  # Wide angle lens
            "positionXMeters": 0.0,
            "positionYMeters": 80.0,  # Behind far touchline
            "positionZMeters": -15.0,  # Elevated (negative Z in BroadTrack convention)
            "sensorResolutionWidthPixels": image_width,
            "sensorResolutionHeightPixels": image_height,
        }

        for anchor in self._selected_anchors:
            if anchor.homography is None:
                continue

            try:
                # Convert homography to camera params
                params = converter.homography_to_camera_params(
                    anchor.homography,
                    reference_params=default_params,
                )

                if params is not None:
                    anchor_params[anchor.frame_idx] = params
                    logger.debug(
                        f"Anchor {anchor.frame_idx}: pan={params['panDegrees']:.2f}, "
                        f"tilt={params['tiltDegrees']:.2f}"
                    )
            except Exception as e:
                logger.warning(
                    f"Failed to convert anchor {anchor.frame_idx} params: {e}"
                )

        # Fallback: use default params if conversion failed
        for anchor in self._selected_anchors:
            if anchor.frame_idx not in anchor_params:
                logger.warning(
                    f"Using default params for anchor {anchor.frame_idx}"
                )
                params = default_params.copy()
                # Estimate pan from homography if possible
                if anchor.homography is not None:
                    try:
                        pan = self._estimate_pan_from_homography(
                            anchor.homography, image_width, image_height
                        )
                        params["panDegrees"] = pan
                    except Exception:
                        pass
                anchor_params[anchor.frame_idx] = params

        return anchor_params

    def _estimate_pan_from_homography(
        self,
        H: np.ndarray,
        image_width: int,
        image_height: int,
    ) -> float:
        """Estimate pan angle from homography.

        Uses the horizontal displacement at image center to estimate pan.

        Args:
            H: 3x3 homography matrix (world -> image).
            image_width: Image width.
            image_height: Image height.

        Returns:
            Estimated pan angle in degrees.
        """
        # Project center point (0, 0) in world coords
        center_world = np.array([0, 0, 1], dtype=np.float64)
        center_img = H @ center_world
        center_img = center_img[:2] / center_img[2]

        # Pan is related to horizontal offset from image center
        dx = center_img[0] - image_width / 2

        # Rough conversion: assume 50 deg FOV
        fov_rad = np.radians(50)
        pixels_per_degree = image_width / np.degrees(fov_rad)
        pan = -dx / pixels_per_degree

        return float(np.clip(pan, -45, 45))

    def export_results(
        self,
        output_path: str | Path,
        frame_path_template: str | None = None,
    ) -> None:
        """Export calibration results to JSON.

        Args:
            output_path: Path to output JSON file.
            frame_path_template: Optional template for frame keys
                               (e.g., "frame_{:06d}.jpg").
        """
        if self._chain_calibrator:
            self._chain_calibrator.export_results(output_path, frame_path_template)
        else:
            # Export results directly
            output = {}
            for frame_idx, result in sorted(self._results.items()):
                if frame_path_template:
                    key = frame_path_template.format(frame_idx)
                else:
                    key = str(frame_idx)

                entry = {
                    "confidence": result.get("confidence", 0.0),
                    "source": result.get("source", "auto"),
                }
                if "camera_params" in result:
                    entry["cp"] = result["camera_params"]

                output[key] = entry

            with open(output_path, "w") as f:
                json.dump(output, f, indent=2)

            logger.info(f"Exported {len(output)} calibrations to {output_path}")

    def get_camera_params(self, frame_idx: int) -> dict | None:
        """Get camera parameters for a frame.

        Args:
            frame_idx: Frame index.

        Returns:
            Camera parameters dict or None if not calibrated.
        """
        result = self._results.get(frame_idx)
        if result:
            return result.get("camera_params")
        return None

    def get_selected_anchors(self) -> list[FrameScore]:
        """Get the selected anchor frames.

        Returns:
            List of FrameScore objects for selected anchors.
        """
        return self._selected_anchors

    def get_frame_scores(self) -> list[FrameScore]:
        """Get all scanned frame scores.

        Returns:
            List of FrameScore objects from scanning.
        """
        return self._frame_scores

    def visualize_scores(self, output_path: str | Path) -> None:
        """Generate visualization of frame scores and selected anchors.

        Args:
            output_path: Path to save plot image.
        """
        if self._anchor_selector and self._frame_scores:
            self._anchor_selector.visualize_scores(
                self._frame_scores,
                output_path,
                selected_anchors=self._selected_anchors,
            )

    def enable_visualization(self, output_dir: str | Path) -> None:
        """Enable visualization output for anchor detection and chain calibration.

        Args:
            output_dir: Directory to save visualization images.
        """
        self._vis_output_dir = Path(output_dir)
        self._vis_output_dir.mkdir(parents=True, exist_ok=True)
        # Enable anchor selector visualization
        if self._anchor_selector:
            self._anchor_selector.enable_visualization(self._vis_output_dir / "anchor_scan")
        if self._chain_calibrator:
            self._chain_calibrator.enable_visualization(output_dir)
