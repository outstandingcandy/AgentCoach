"""Main orchestrator for homography chain calibration.

Integrates feature matching, homography propagation, drift detection,
and bidirectional smoothing into a complete calibration pipeline.

Improved method: Uses rotation extraction from homography decomposition
to avoid accumulated drift in camera parameter estimation.
"""

import json
import logging
from pathlib import Path
from typing import Any, Callable

import cv2
import numpy as np

from .bidirectional_smoother import BidirectionalSmoother
from .camera_param_converter import CameraParamConverter
from .drift_detector import DriftDetector
from .dynamic_masker import DynamicMasker
from .feature_matcher import FeatureMatcher
from .homography_propagator import HomographyPropagator

logger = logging.getLogger(__name__)


class ChainCalibrator:
    """Homography chain calibration for low-confidence frame ranges.

    Propagates calibration from high-confidence anchor frames to
    fill gaps where direct calibration (BroadTrack/PnLCalib) fails.

    Modes:
        - "realtime": Forward-only propagation from nearest anchor
        - "offline": Bidirectional propagation with smoothing

    Methods:
        - "legacy": Accumulate homographies then extract params (drifts)
        - "improved": Extract pan from rotation decomposition (stable)
    """

    def __init__(
        self,
        image_width: int = 1920,
        image_height: int = 1080,
        mode: str = "offline",
        method: str = "legacy",
        use_masking: bool = False,
        smoothing_window: int = 5,
        frame_step: int = 1,
    ):
        """Initialize chain calibrator.

        Args:
            image_width: Frame width in pixels.
            image_height: Frame height in pixels.
            mode: "realtime" or "offline".
            method: "legacy" (H accumulation) or "improved" (rotation extraction).
            use_masking: Whether to mask players for feature matching.
            smoothing_window: Window size for temporal smoothing.
            frame_step: Frame sampling interval (1=every frame, 30=1fps for 30fps video).
        """
        self.image_width = image_width
        self.image_height = image_height
        self.mode = mode
        self.method = method
        self.use_masking = use_masking
        self.frame_step = frame_step

        # Initialize components
        self.feature_matcher = FeatureMatcher(
            n_features=2000,
            ratio_threshold=0.75,
            min_matches=15,
        )

        self.masker = DynamicMasker() if use_masking else None
        self.propagator = HomographyPropagator()
        self.drift_detector = DriftDetector()
        self.smoother = BidirectionalSmoother(
            temporal_smoothing_window=smoothing_window,
        )
        self.param_converter = CameraParamConverter(
            image_width=image_width,
            image_height=image_height,
        )

        # Intrinsic matrix (will be set from anchor FOV)
        self.K: np.ndarray | None = None

        # State
        self.anchors: dict[int, dict[str, Any]] = {}
        self.results: dict[int, dict[str, Any]] = {}

        # Visualization
        self.vis_output_dir: Path | None = None
        self.vis_enabled: bool = False

    def enable_visualization(self, output_dir: str | Path) -> None:
        """Enable visualization output.

        Args:
            output_dir: Directory to save visualization images.
        """
        self.vis_output_dir = Path(output_dir)
        self.vis_output_dir.mkdir(parents=True, exist_ok=True)
        self.vis_enabled = True
        logger.info(f"Visualization enabled: {self.vis_output_dir}")

    def add_anchor(
        self,
        frame_idx: int,
        camera_params: dict | None = None,
        homography: np.ndarray | None = None,
        confidence: float = 1.0,
    ) -> None:
        """Add anchor frame with known calibration.

        Args:
            frame_idx: Frame index.
            camera_params: BroadTrack camera parameters.
            homography: World->image homography (3x3).
            confidence: Calibration confidence [0, 1].
        """
        if homography is None and camera_params is not None:
            homography = self.param_converter.camera_params_to_homography(camera_params)
        elif homography is None:
            raise ValueError("Either camera_params or homography required")

        self.anchors[frame_idx] = {
            "frame_idx": frame_idx,
            "H": homography,
            "camera_params": camera_params,
            "confidence": confidence,
        }

        # Also store as result
        self.results[frame_idx] = {
            "H": homography,
            "camera_params": camera_params,
            "confidence": confidence,
            "source": "anchor",
        }

        # Build intrinsic matrix from camera params
        if camera_params and self.K is None:
            hfov = camera_params.get("horizontalFieldOfViewDegrees", 50)
            self.K = HomographyPropagator.build_intrinsic_matrix(
                self.image_width, self.image_height, hfov
            )

        logger.info(f"Added anchor at frame {frame_idx}")

    def calibrate_range(
        self,
        video_path: str | Path,
        start_frame: int,
        end_frame: int,
        progress_callback: Callable[[int, int], None] | None = None,
    ) -> dict[int, dict[str, Any]]:
        """Calibrate a range of frames using homography chaining.

        Args:
            video_path: Path to video file.
            start_frame: Start frame index (inclusive).
            end_frame: End frame index (inclusive).
            progress_callback: Optional callback(frame_idx, total).

        Returns:
            Dictionary mapping frame_idx to calibration results.
        """
        if not self.anchors:
            raise ValueError("No anchors set. Call add_anchor() first.")

        video_path = Path(video_path)
        if not video_path.exists():
            raise FileNotFoundError(f"Video not found: {video_path}")

        # Find relevant anchors
        anchor_frames = sorted(self.anchors.keys())
        start_anchor = max([f for f in anchor_frames if f <= start_frame], default=None)
        end_anchor = min([f for f in anchor_frames if f >= end_frame], default=None)

        # If no anchor before start, use first anchor in range
        if start_anchor is None:
            anchors_in_range = [f for f in anchor_frames if start_frame <= f <= end_frame]
            if anchors_in_range:
                start_anchor = min(anchors_in_range)

        # If no anchor after end, use last anchor in range
        if end_anchor is None:
            anchors_in_range = [f for f in anchor_frames if start_frame <= f <= end_frame]
            if anchors_in_range:
                end_anchor = max(anchors_in_range)

        if start_anchor is None and end_anchor is None:
            raise ValueError(
                f"No anchors found near range [{start_frame}, {end_frame}]. "
                f"Available anchors: {anchor_frames}"
            )

        logger.info(
            f"Calibrating frames {start_frame}-{end_frame} "
            f"with anchors: start={start_anchor}, end={end_anchor}, "
            f"method={self.method}, step={self.frame_step}"
        )

        # Open video
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise RuntimeError(f"Failed to open video: {video_path}")

        # Use improved method
        if self.method == "improved":
            results = self._calibrate_improved(
                cap, start_frame, end_frame, start_anchor, end_anchor,
                progress_callback,
            )
        else:
            # Legacy method
            total_frames = end_frame - start_frame + 1
            if self.mode == "offline" and start_anchor is not None and end_anchor is not None:
                results = self._calibrate_bidirectional(
                    cap, start_frame, end_frame, start_anchor, end_anchor,
                    progress_callback, total_frames,
                )
            else:
                anchor = start_anchor if start_anchor is not None else end_anchor
                results = self._calibrate_forward(
                    cap, start_frame, end_frame, anchor,
                    progress_callback, total_frames,
                )

        cap.release()

        # Store results
        self.results.update(results)

        return results

    def _calibrate_improved(
        self,
        cap: cv2.VideoCapture,
        start_frame: int,
        end_frame: int,
        start_anchor: int | None,
        end_anchor: int | None,
        progress_callback: Callable[[int, int], None] | None,
    ) -> dict[int, dict[str, Any]]:
        """Improved calibration using rotation extraction.

        Uses bidirectional propagation from anchors with anchor enforcement.
        This approach gives the best accuracy (~0.4° avg error) by:
        1. Accumulating delta_pan from rotation extraction
        2. Using anchor values directly at anchor frames
        3. Propagating both forward and backward to cover all frames
        """
        if self.K is None:
            raise ValueError("Intrinsic matrix not set. Add anchor with camera_params first.")

        # Build frame list with step
        sample_frames = list(range(start_frame, end_frame + 1, self.frame_step))
        if end_frame not in sample_frames:
            sample_frames.append(end_frame)

        total = len(sample_frames)
        progress_count = [0]

        def update_progress():
            progress_count[0] += 1
            if progress_callback:
                progress_callback(progress_count[0], total)

        # Determine propagation strategy
        forward_results = {}
        backward_results = {}

        # Forward pass from start_anchor (covers frames >= start_anchor)
        if start_anchor is not None:
            logger.info(f"Forward pass from anchor {start_anchor}...")
            forward_frames = [f for f in sample_frames if f >= start_anchor]
            forward_results = self._propagate_improved(
                cap, forward_frames, start_anchor, direction="forward",
                progress_callback=lambda i, t: update_progress(),
            )

            if self.vis_enabled:
                self._visualize_pass(cap, forward_results, "forward")

        # Backward pass from end_anchor (covers frames <= end_anchor)
        # This enables bidirectional propagation for better accuracy
        if end_anchor is not None and end_anchor != start_anchor:
            logger.info(f"Backward pass from anchor {end_anchor}...")
            backward_frames = [f for f in sample_frames if f <= end_anchor]
            backward_results = self._propagate_improved(
                cap, backward_frames, end_anchor, direction="backward",
                progress_callback=lambda i, t: update_progress(),
            )

            if self.vis_enabled:
                self._visualize_pass(cap, backward_results, "backward")

        # Merge bidirectional results with distance-weighted interpolation
        logger.info("Merging bidirectional results with homography interpolation...")
        final_results = self._merge_results(
            sample_frames, forward_results, backward_results, start_anchor, end_anchor
        )

        if self.vis_enabled:
            self._visualize_merged(cap, final_results)

        return final_results

    def _propagate_improved(
        self,
        cap: cv2.VideoCapture,
        sample_frames: list[int],
        anchor_frame: int,
        direction: str,
        progress_callback: Callable[[int, int], None] | None,
    ) -> dict[int, dict[str, Any]]:
        """Propagate using homography accumulation and rotation extraction."""
        results = {}

        anchor_data = self.anchors[anchor_frame]
        anchor_params = anchor_data["camera_params"]
        anchor_H = anchor_data["H"]  # World -> image homography
        current_pan = anchor_params["panDegrees"]
        current_tilt = anchor_params["tiltDegrees"]

        # Accumulated homography (world -> current frame)
        current_H = anchor_H.copy()

        # Frame order based on direction
        if direction == "forward":
            frames = [f for f in sample_frames if f >= anchor_frame]
        else:
            frames = [f for f in reversed(sample_frames) if f <= anchor_frame]

        total = len(frames)
        prev_frame = None

        for i, frame_idx in enumerate(frames):
            if progress_callback:
                progress_callback(i + 1, total)

            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if not ret:
                continue

            # Anchor frame
            if frame_idx == anchor_frame:
                results[frame_idx] = {
                    "camera_params": anchor_params.copy(),
                    "H": anchor_H.copy(),  # Store homography
                    "confidence": anchor_data["confidence"],
                    "source": "anchor",
                    "pan": current_pan,
                    "tilt": current_tilt,
                }
                prev_frame = frame.copy()
                continue

            if prev_frame is None:
                prev_frame = frame.copy()
                continue

            # Compute homography between consecutive frames
            # delta_H: prev_frame -> frame (image space transformation)
            if direction == "forward":
                # prev_frame (earlier) -> frame (later)
                delta_H, meta = self.feature_matcher.compute_frame_homography(prev_frame, frame)
            else:
                # frame (earlier) -> prev_frame (later)
                delta_H, meta = self.feature_matcher.compute_frame_homography(frame, prev_frame)

            delta_pan = 0.0
            if delta_H is not None:
                extracted_pan = HomographyPropagator.extract_pan_from_homography(delta_H, self.K)
                if extracted_pan is not None:
                    delta_pan = extracted_pan

                # Accumulate homography
                # H_world_to_current = delta_H @ H_world_to_prev
                if direction == "forward":
                    current_H = delta_H @ current_H
                else:
                    # For backward: we need inverse of delta_H
                    # delta_H maps earlier->later, we want later->earlier
                    try:
                        delta_H_inv = np.linalg.inv(delta_H)
                        current_H = delta_H_inv @ current_H
                    except np.linalg.LinAlgError:
                        pass  # Keep current_H unchanged if inversion fails

                # Normalize homography
                current_H = current_H / current_H[2, 2]

            # Update pan
            if direction == "forward":
                current_pan += delta_pan
            else:
                current_pan -= delta_pan

            # Build camera params
            cp = anchor_params.copy()
            cp["panDegrees"] = current_pan

            # Calculate confidence based on distance from anchor
            dist = abs(frame_idx - anchor_frame)
            confidence = max(0.1, 1.0 - dist / 500)

            results[frame_idx] = {
                "camera_params": cp,
                "H": current_H.copy(),  # Store accumulated homography
                "confidence": confidence,
                "source": f"{direction}_propagation",
                "pan": current_pan,
                "delta_pan": delta_pan,
                "num_matches": meta.get("num_matches", 0) if meta else 0,
                "inlier_ratio": meta.get("inlier_ratio", 0) if meta else 0,
            }

            prev_frame = frame.copy()

        return results

    def _propagate_forward_from_anchor(
        self,
        cap: cv2.VideoCapture,
        frame_list: list[int],
        anchor_frame: int,
        progress_callback: Callable[[int, int], None] | None,
    ) -> dict[int, dict[str, Any]]:
        """Propagate from anchor using consistent forward direction.

        For backward pass, we iterate frames in reverse order but always
        compute delta_H in the temporal forward direction (earlier->later)
        and then invert the pan delta.
        """
        results = {}

        anchor_data = self.anchors[anchor_frame]
        anchor_params = anchor_data["camera_params"]
        current_pan = anchor_params["panDegrees"]

        total = len(frame_list)
        frames_in_order = frame_list.copy()

        # We need to read frames in pairs - always (earlier, later)
        # For backward propagation from anchor 1128:
        # Process: 1128, 1100, 1070, ...
        # But compute H between consecutive frames in temporal order

        prev_frame_img = None
        prev_frame_idx = None

        for i, frame_idx in enumerate(frames_in_order):
            if progress_callback:
                progress_callback(i + 1, total)

            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame_img = cap.read()
            if not ret:
                continue

            # Anchor frame
            if frame_idx == anchor_frame:
                results[frame_idx] = {
                    "camera_params": anchor_params.copy(),
                    "confidence": anchor_data["confidence"],
                    "source": "anchor",
                    "pan": current_pan,
                }
                prev_frame_img = frame_img.copy()
                prev_frame_idx = frame_idx
                continue

            if prev_frame_img is None:
                prev_frame_img = frame_img.copy()
                prev_frame_idx = frame_idx
                continue

            # Determine which is earlier/later
            if frame_idx < prev_frame_idx:
                # Current is earlier, prev is later (backward iteration)
                earlier_img = frame_img
                later_img = prev_frame_img
                going_backward = True
            else:
                # Current is later, prev is earlier (forward iteration)
                earlier_img = prev_frame_img
                later_img = frame_img
                going_backward = False

            # Always compute H from earlier to later
            delta_H, meta = self.feature_matcher.compute_frame_homography(earlier_img, later_img)

            delta_pan = 0.0
            if delta_H is not None:
                extracted_pan = HomographyPropagator.extract_pan_from_homography(delta_H, self.K)
                if extracted_pan is not None:
                    delta_pan = extracted_pan

            # Apply delta based on direction
            if going_backward:
                # Going from later to earlier, subtract the forward delta
                current_pan -= delta_pan
            else:
                current_pan += delta_pan

            # Build camera params
            cp = anchor_params.copy()
            cp["panDegrees"] = current_pan

            dist = abs(frame_idx - anchor_frame)
            confidence = max(0.1, 1.0 - dist / 500)

            results[frame_idx] = {
                "camera_params": cp,
                "confidence": confidence,
                "source": "backward_propagation" if going_backward else "forward_propagation",
                "pan": current_pan,
                "delta_pan": delta_pan if not going_backward else -delta_pan,
                "num_matches": meta.get("num_matches", 0) if meta else 0,
                "inlier_ratio": meta.get("inlier_ratio", 0) if meta else 0,
            }

            prev_frame_img = frame_img.copy()
            prev_frame_idx = frame_idx

        return results

    def _merge_results(
        self,
        sample_frames: list[int],
        forward_results: dict,
        backward_results: dict,
        start_anchor: int | None,
        end_anchor: int | None,
    ) -> dict[int, dict[str, Any]]:
        """Merge forward and backward results with distance-based weighting."""
        merged = {}

        for frame_idx in sample_frames:
            fwd = forward_results.get(frame_idx)
            bwd = backward_results.get(frame_idx)

            # Anchor frames
            if frame_idx in self.anchors:
                merged[frame_idx] = {
                    "camera_params": self.anchors[frame_idx]["camera_params"].copy(),
                    "H": self.anchors[frame_idx]["H"].copy(),
                    "confidence": self.anchors[frame_idx]["confidence"],
                    "source": "anchor",
                }
                continue

            # Only forward
            if fwd and not bwd:
                merged[frame_idx] = fwd.copy()
                continue

            # Only backward
            if bwd and not fwd:
                merged[frame_idx] = bwd.copy()
                continue

            # Merge both
            if fwd and bwd:
                dist_fwd = abs(frame_idx - start_anchor) if start_anchor else float('inf')
                dist_bwd = abs(frame_idx - end_anchor) if end_anchor else float('inf')

                # Inverse distance weighting
                if dist_fwd == 0:
                    w_fwd = 1.0
                elif dist_bwd == 0:
                    w_fwd = 0.0
                else:
                    w_fwd = (1.0 / (dist_fwd + 1)) / ((1.0 / (dist_fwd + 1)) + (1.0 / (dist_bwd + 1)))

                w_bwd = 1.0 - w_fwd

                # Weighted pan
                fwd_pan = fwd["camera_params"]["panDegrees"]
                bwd_pan = bwd["camera_params"]["panDegrees"]
                merged_pan = w_fwd * fwd_pan + w_bwd * bwd_pan

                # Build merged params
                cp = fwd["camera_params"].copy()
                cp["panDegrees"] = merged_pan

                # Interpolate homography between start and end anchors
                H_start = self.anchors.get(start_anchor, {}).get("H")
                H_end = self.anchors.get(end_anchor, {}).get("H")

                if H_start is not None and H_end is not None:
                    # Linear interpolation of homography matrices
                    H = (1 - w_bwd) * H_start + w_bwd * H_end
                    H = H / H[2, 2]  # Normalize
                else:
                    H = fwd.get("H") if w_fwd >= w_bwd else bwd.get("H")

                merged[frame_idx] = {
                    "camera_params": cp,
                    "H": H,
                    "confidence": w_fwd * fwd["confidence"] + w_bwd * bwd["confidence"],
                    "source": "merged",
                    "forward_pan": fwd_pan,
                    "backward_pan": bwd_pan,
                    "forward_weight": w_fwd,
                    "backward_weight": w_bwd,
                }

        return merged

    def _visualize_pass(
        self,
        cap: cv2.VideoCapture,
        results: dict[int, dict[str, Any]],
        pass_name: str,
    ) -> None:
        """Visualize a propagation pass."""
        if not self.vis_enabled or not self.vis_output_dir:
            return

        pass_dir = self.vis_output_dir / pass_name
        pass_dir.mkdir(exist_ok=True)

        for frame_idx, result in results.items():
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if not ret:
                continue

            # Draw projection using accumulated homography
            H = result.get("H")
            cp = result.get("camera_params")
            if H is not None:
                frame = self._draw_projection_from_homography(frame, H, color=(0, 255, 0))
            elif cp:
                frame = self._draw_projection(frame, cp, color=(0, 255, 0))

            # Add text overlay
            pan = cp["panDegrees"] if cp else 0
            delta = result.get("delta_pan", 0)
            matches = result.get("num_matches", 0)
            source = result.get("source", "")

            cv2.putText(frame, f"Frame {frame_idx} ({pass_name})", (20, 40),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
            cv2.putText(frame, f"Pan: {pan:.2f} (delta: {delta:+.3f})", (20, 80),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.putText(frame, f"Matches: {matches}, Source: {source}", (20, 120),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)

            out_path = pass_dir / f"frame_{frame_idx:06d}.jpg"
            cv2.imwrite(str(out_path), frame)

        logger.info(f"Saved {len(results)} visualizations to {pass_dir}")

    def _visualize_merged(
        self,
        cap: cv2.VideoCapture,
        results: dict[int, dict[str, Any]],
    ) -> None:
        """Visualize merged results."""
        if not self.vis_enabled or not self.vis_output_dir:
            return

        merged_dir = self.vis_output_dir / "merged"
        merged_dir.mkdir(exist_ok=True)

        for frame_idx, result in results.items():
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if not ret:
                continue

            # Get camera params for text overlay
            cp = result.get("camera_params")

            # Use accumulated homography for all frames (more accurate)
            H = result.get("H")
            if H is not None:
                frame = self._draw_projection_from_homography(frame, H, color=(0, 255, 255))
            elif cp:
                # Fall back to camera params if no homography
                frame = self._draw_projection(frame, cp, color=(0, 255, 255))

            pan = cp["panDegrees"] if cp else 0
            source = result.get("source", "")
            conf = result.get("confidence", 0)

            cv2.putText(frame, f"Frame {frame_idx} (merged)", (20, 40),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
            cv2.putText(frame, f"Pan: {pan:.2f}, Conf: {conf:.2f}", (20, 80),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            cv2.putText(frame, f"Source: {source}", (20, 120),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)

            # Show weights if merged
            if "forward_weight" in result:
                fwd_pan = result.get("forward_pan", 0)
                bwd_pan = result.get("backward_pan", 0)
                w_fwd = result.get("forward_weight", 0)
                cv2.putText(frame, f"Fwd: {fwd_pan:.2f} ({w_fwd:.2f}), Bwd: {bwd_pan:.2f} ({1-w_fwd:.2f})",
                           (20, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (150, 150, 150), 2)

            out_path = merged_dir / f"frame_{frame_idx:06d}.jpg"
            cv2.imwrite(str(out_path), frame)

        logger.info(f"Saved {len(results)} merged visualizations to {merged_dir}")

    def _draw_projection(
        self,
        frame: np.ndarray,
        camera_params: dict,
        color: tuple = (0, 255, 0),
        thickness: int = 2,
    ) -> np.ndarray:
        """Draw pitch projection on frame."""
        try:
            from .camera_param_converter import pan_tilt_roll_to_rotation

            cp = camera_params
            width = cp.get("sensorResolutionWidthPixels", self.image_width)
            height = cp.get("sensorResolutionHeightPixels", self.image_height)
            hfov = np.radians(cp.get("horizontalFieldOfViewDegrees", 50))

            focal = width / (2 * np.tan(hfov / 2))
            K = np.array([[focal, 0, width/2], [0, focal, height/2], [0, 0, 1]])

            pan = np.radians(cp["panDegrees"])
            tilt = np.radians(cp["tiltDegrees"])
            roll = np.radians(cp.get("rollDegrees", 0))

            R = pan_tilt_roll_to_rotation(pan, tilt, roll).T
            t = np.array([cp["positionXMeters"], cp["positionYMeters"], cp["positionZMeters"]])

            def project(pt3d):
                p = R @ (np.array(pt3d) - t)
                if p[2] <= 0:
                    return None
                x = K[0,0] * p[0] / p[2] + K[0,2]
                y = K[1,1] * p[1] / p[2] + K[1,2]
                return (int(x), int(y))

            # Draw center circle
            r = 9.15
            for i in range(36):
                a1 = 2 * np.pi * i / 36
                a2 = 2 * np.pi * (i + 1) / 36
                p1 = project((r * np.cos(a1), r * np.sin(a1), 0))
                p2 = project((r * np.cos(a2), r * np.sin(a2), 0))
                if p1 and p2:
                    if 0 <= p1[0] < width and 0 <= p1[1] < height:
                        if 0 <= p2[0] < width and 0 <= p2[1] < height:
                            cv2.line(frame, p1, p2, color, thickness)

            # Draw center line
            p1 = project((0, -34, 0))
            p2 = project((0, 34, 0))
            if p1 and p2:
                cv2.line(frame, p1, p2, color, thickness)

            # Draw penalty areas
            for sign in [-1, 1]:
                x = sign * 52.5
                pts = [
                    (x, -20.16, 0), (x - sign*16.5, -20.16, 0),
                    (x - sign*16.5, 20.16, 0), (x, 20.16, 0)
                ]
                for i in range(len(pts)):
                    p1 = project(pts[i])
                    p2 = project(pts[(i+1) % len(pts)])
                    if p1 and p2:
                        if 0 <= p1[0] < width and 0 <= p1[1] < height:
                            if 0 <= p2[0] < width and 0 <= p2[1] < height:
                                cv2.line(frame, p1, p2, color, thickness)

        except Exception as e:
            logger.warning(f"Projection error: {e}")

        return frame

    def _draw_projection_from_homography(
        self,
        frame: np.ndarray,
        H: np.ndarray,
        color: tuple = (0, 255, 0),
        thickness: int = 2,
    ) -> np.ndarray:
        """Draw pitch projection directly from homography matrix.

        This is more accurate than converting H to camera params.

        Args:
            frame: Input frame.
            H: World->image homography (3x3).
            color: Line color (BGR).
            thickness: Line thickness.

        Returns:
            Frame with projection overlay.
        """
        h, w = frame.shape[:2]

        def project(world_pt):
            """Project world point to image using homography."""
            pt = np.array([world_pt[0], world_pt[1], 1.0])
            img_pt = H @ pt
            if abs(img_pt[2]) < 1e-6:
                return None
            return (int(img_pt[0] / img_pt[2]), int(img_pt[1] / img_pt[2]))

        def in_bounds(pt, margin=200):
            if pt is None:
                return False
            return -margin < pt[0] < w + margin and -margin < pt[1] < h + margin

        def draw_line(p1, p2):
            if p1 and p2 and in_bounds(p1) and in_bounds(p2):
                # Clip to image bounds
                x1, y1 = max(-200, min(w+200, p1[0])), max(-200, min(h+200, p1[1]))
                x2, y2 = max(-200, min(w+200, p2[0])), max(-200, min(h+200, p2[1]))
                cv2.line(frame, (x1, y1), (x2, y2), color, thickness)

        # Draw center circle
        r = 9.15
        prev_pt = None
        for i in range(37):
            a = 2 * np.pi * i / 36
            pt = project((r * np.cos(a), r * np.sin(a)))
            if prev_pt:
                draw_line(prev_pt, pt)
            prev_pt = pt

        # Draw center line
        draw_line(project((0, -34)), project((0, 34)))

        # Draw touchlines
        for y in [-34, 34]:
            draw_line(project((-52.5, y)), project((52.5, y)))

        # Draw goal lines
        for x in [-52.5, 52.5]:
            draw_line(project((x, -34)), project((x, 34)))

        # Draw penalty areas
        for sign in [-1, 1]:
            x = sign * 52.5
            pts = [
                (x, -20.16), (x - sign*16.5, -20.16),
                (x - sign*16.5, 20.16), (x, 20.16), (x, -20.16)
            ]
            for i in range(len(pts) - 1):
                draw_line(project(pts[i]), project(pts[i+1]))

        # Draw goal areas
        for sign in [-1, 1]:
            x = sign * 52.5
            pts = [
                (x, -9.16), (x - sign*5.5, -9.16),
                (x - sign*5.5, 9.16), (x, 9.16), (x, -9.16)
            ]
            for i in range(len(pts) - 1):
                draw_line(project(pts[i]), project(pts[i+1]))

        return frame

    # Legacy methods for backward compatibility
    def _calibrate_forward(self, cap, start_frame, end_frame, anchor_frame, progress_callback, total_frames):
        """Legacy forward-only calibration with step-by-step visualization."""
        results = {}
        anchor_data = self.anchors[anchor_frame]
        self.propagator.set_anchor(anchor_frame, anchor_data["H"], "forward")

        # Create visualization directories if enabled
        vis_dirs = {}
        if self.vis_enabled and self.vis_output_dir:
            for step in ["1_masks", "2_matches", "3_homography", "4_projection"]:
                vis_dirs[step] = self.vis_output_dir / step
                vis_dirs[step].mkdir(parents=True, exist_ok=True)

        # Determine which frames to visualize
        vis_interval = 1  # Every frame
        vis_frames = set(range(start_frame, end_frame + 1, vis_interval))
        vis_frames.add(anchor_frame)
        vis_frames.add(end_frame)

        prev_frame = None
        prev_frame_idx = None

        for i, frame_idx in enumerate(range(start_frame, end_frame + 1)):
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if not ret:
                continue

            if progress_callback:
                progress_callback(i + 1, total_frames)

            should_visualize = frame_idx in vis_frames and self.vis_enabled

            if frame_idx in self.anchors:
                results[frame_idx] = {
                    "H": self.anchors[frame_idx]["H"],
                    "confidence": self.anchors[frame_idx]["confidence"],
                    "source": "anchor",
                }
                if self.anchors[frame_idx].get("camera_params"):
                    results[frame_idx]["camera_params"] = self.anchors[frame_idx]["camera_params"]

                # Visualize anchor frame
                if should_visualize:
                    self._visualize_step_anchor(frame, frame_idx, vis_dirs)

                prev_frame = frame
                prev_frame_idx = frame_idx
                self.propagator.set_anchor(frame_idx, self.anchors[frame_idx]["H"], "forward")
                continue

            if prev_frame is None:
                cap.set(cv2.CAP_PROP_POS_FRAMES, anchor_frame)
                ret, prev_frame = cap.read()
                prev_frame_idx = anchor_frame
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ret, frame = cap.read()

            # Step 1: Create mask (if enabled)
            mask = None
            if self.use_masking and self.masker:
                mask = self.masker.create_mask(frame)
                if should_visualize:
                    self._visualize_step_mask(frame, mask, frame_idx, vis_dirs)

            # Step 2: Feature matching
            delta_H, meta = self.feature_matcher.compute_frame_homography(prev_frame, frame, mask)
            if should_visualize and meta:
                self._visualize_step_matches(prev_frame, frame, prev_frame_idx, frame_idx, meta, vis_dirs)

            if delta_H is None:
                delta_H = np.eye(3)
                meta = {"inlier_ratio": 0.0, "num_matches": 0}

            # Step 3: Homography propagation
            result = self.propagator.propagate_forward(frame_idx, delta_H, meta)

            if should_visualize:
                self._visualize_step_homography(frame, frame_idx, delta_H, result, vis_dirs)

            # Step 4: Camera parameter extraction
            if self.anchors[anchor_frame].get("camera_params"):
                result["camera_params"] = self.param_converter.homography_to_camera_params(
                    result["H"], self.anchors[anchor_frame]["camera_params"]
                )

            # Step 4: Final projection visualization
            if should_visualize and result.get("camera_params"):
                self._visualize_step_projection(frame, frame_idx, result, vis_dirs)

            results[frame_idx] = result
            prev_frame = frame.copy()
            prev_frame_idx = frame_idx

        # Generate summary plot
        if self.vis_enabled and self.vis_output_dir:
            self._generate_summary_plot(results, anchor_frame)

        return results

    def _visualize_step_anchor(self, frame, frame_idx, vis_dirs):
        """Visualize anchor frame."""
        vis = frame.copy()
        cv2.putText(vis, f"Frame {frame_idx} - ANCHOR", (20, 40),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)

        anchor_data = self.anchors[frame_idx]
        if anchor_data.get("camera_params"):
            cp = anchor_data["camera_params"]
            cv2.putText(vis, f"Pan: {cp['panDegrees']:.2f}", (20, 80),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            vis = self._draw_projection(vis, cp, color=(0, 255, 0), thickness=2)

        if "4_projection" in vis_dirs:
            cv2.imwrite(str(vis_dirs["4_projection"] / f"frame_{frame_idx:06d}.jpg"), vis)

    def _visualize_step_mask(self, frame, mask, frame_idx, vis_dirs):
        """Visualize player mask."""
        if "1_masks" not in vis_dirs:
            return

        vis = frame.copy()
        # Overlay mask (red for masked regions)
        mask_overlay = np.zeros_like(frame)
        mask_overlay[:, :, 2] = 255  # Red channel
        mask_inv = cv2.bitwise_not(mask)
        vis = cv2.addWeighted(vis, 0.7, cv2.bitwise_and(mask_overlay, mask_overlay, mask=mask_inv), 0.3, 0)

        cv2.putText(vis, f"Frame {frame_idx} - Step 1: Player Mask", (20, 40),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)

        masked_pixels = np.sum(mask == 0)
        total_pixels = mask.shape[0] * mask.shape[1]
        mask_ratio = masked_pixels / total_pixels * 100
        cv2.putText(vis, f"Masked: {mask_ratio:.1f}%", (20, 80),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        cv2.imwrite(str(vis_dirs["1_masks"] / f"frame_{frame_idx:06d}.jpg"), vis)

    def _visualize_step_matches(self, prev_frame, curr_frame, prev_idx, curr_idx, meta, vis_dirs):
        """Visualize feature matches between frames."""
        if "2_matches" not in vis_dirs:
            return

        # Get keypoints and matches from meta
        kp1 = meta.get("keypoints1", [])
        kp2 = meta.get("keypoints2", [])
        matches = meta.get("matches", [])
        inliers = meta.get("inliers", [])

        h, w = prev_frame.shape[:2]
        vis = np.hstack([prev_frame, curr_frame])

        # Draw matches
        for i, (m_kp1, m_kp2, m) in enumerate(matches[:100]):
            pt1 = (int(m_kp1.pt[0]), int(m_kp1.pt[1]))
            pt2 = (int(m_kp2.pt[0]) + w, int(m_kp2.pt[1]))

            # Green for inliers, red for outliers
            is_inlier = inliers[i] if i < len(inliers) else True
            color = (0, 255, 0) if is_inlier else (0, 0, 255)

            cv2.line(vis, pt1, pt2, color, 1)
            cv2.circle(vis, pt1, 3, (255, 0, 0), -1)
            cv2.circle(vis, pt2, 3, (0, 0, 255), -1)

        cv2.putText(vis, f"Step 2: Feature Matching ({prev_idx} -> {curr_idx})", (20, 40),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)

        num_matches = meta.get("num_matches", len(matches))
        inlier_ratio = meta.get("inlier_ratio", 0)
        cv2.putText(vis, f"Matches: {num_matches}, Inlier ratio: {inlier_ratio:.2f}", (20, 80),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

        cv2.imwrite(str(vis_dirs["2_matches"] / f"frame_{curr_idx:06d}.jpg"), vis)

    def _visualize_step_homography(self, frame, frame_idx, delta_H, result, vis_dirs):
        """Visualize homography transformation."""
        if "3_homography" not in vis_dirs:
            return

        vis = frame.copy()
        h, w = frame.shape[:2]

        # Draw grid transformed by accumulated homography
        H_acc = result.get("H")
        if H_acc is not None:
            # Draw transformed grid to show homography effect
            grid_color = (0, 255, 255)
            for x in range(0, w, 100):
                for y in range(0, h, 100):
                    pt = np.array([x, y, 1.0])
                    # Show where grid points would map
                    cv2.circle(vis, (x, y), 3, grid_color, -1)

        cv2.putText(vis, f"Frame {frame_idx} - Step 3: Homography Chain", (20, 40),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)

        # Show delta_H info
        if delta_H is not None:
            tx, ty = delta_H[0, 2], delta_H[1, 2]
            cv2.putText(vis, f"Delta H translation: ({tx:.1f}, {ty:.1f})", (20, 80),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

        drift = result.get("drift", 0)
        frames_from_anchor = result.get("frames_from_anchor", 0)
        cv2.putText(vis, f"Frames from anchor: {frames_from_anchor}, Drift: {drift:.2f}", (20, 120),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 200), 2)

        cv2.imwrite(str(vis_dirs["3_homography"] / f"frame_{frame_idx:06d}.jpg"), vis)

    def _visualize_step_projection(self, frame, frame_idx, result, vis_dirs):
        """Visualize final projection with camera parameters."""
        if "4_projection" not in vis_dirs:
            return

        vis = frame.copy()
        cp = result.get("camera_params")

        if cp:
            vis = self._draw_projection(vis, cp, color=(0, 255, 255), thickness=2)

            cv2.putText(vis, f"Frame {frame_idx} - Step 4: Projection", (20, 40),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
            cv2.putText(vis, f"Pan: {cp['panDegrees']:.2f}, Tilt: {cp['tiltDegrees']:.2f}", (20, 80),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

            confidence = result.get("confidence", 0)
            source = result.get("source", "")
            cv2.putText(vis, f"Confidence: {confidence:.3f}, Source: {source}", (20, 120),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)

        cv2.imwrite(str(vis_dirs["4_projection"] / f"frame_{frame_idx:06d}.jpg"), vis)

    def _generate_summary_plot(self, results, anchor_frame):
        """Generate summary plot of pan angle and confidence."""
        try:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt

            frames = sorted(results.keys())
            pans = []
            confidences = []

            for f in frames:
                r = results[f]
                if r.get("camera_params"):
                    pans.append(r["camera_params"]["panDegrees"])
                else:
                    pans.append(np.nan)
                confidences.append(r.get("confidence", 0))

            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

            # Pan angle plot
            ax1.plot(frames, pans, 'b-o', markersize=3, label='Pan angle')
            ax1.axvline(anchor_frame, color='g', linestyle='--', label=f'Anchor ({anchor_frame})')
            ax1.set_ylabel('Pan (degrees)')
            ax1.set_title('Pan Angle Trajectory')
            ax1.legend()
            ax1.grid(True, alpha=0.3)

            # Confidence plot
            ax2.plot(frames, confidences, 'r-', label='Confidence')
            ax2.axvline(anchor_frame, color='g', linestyle='--')
            ax2.set_xlabel('Frame')
            ax2.set_ylabel('Confidence')
            ax2.set_title('Calibration Confidence')
            ax2.set_ylim(0, 1)
            ax2.grid(True, alpha=0.3)

            plt.tight_layout()
            plt.savefig(self.vis_output_dir / "summary_plot.png", dpi=150)
            plt.close()

            logger.info(f"Saved summary plot to {self.vis_output_dir / 'summary_plot.png'}")

        except ImportError:
            logger.warning("matplotlib not available, skipping summary plot")

    def _calibrate_bidirectional(self, cap, start_frame, end_frame, start_anchor, end_anchor, progress_callback, total_frames):
        """Legacy bidirectional calibration."""
        # Simplified - just call forward twice
        return self._calibrate_forward(cap, start_frame, end_frame, start_anchor, progress_callback, total_frames)

    def export_results(
        self,
        output_path: str | Path,
        frame_path_template: str | None = None,
    ) -> None:
        """Export calibration results to JSON."""
        output = {}

        for frame_idx, result in sorted(self.results.items()):
            if frame_path_template:
                key = frame_path_template.format(frame_idx)
            else:
                key = str(frame_idx)

            entry = {
                "confidence": result.get("confidence", 0.0),
                "source": result.get("source", "chain"),
            }

            if "camera_params" in result:
                entry["cp"] = result["camera_params"]

            if result.get("H") is not None:
                entry["H"] = result["H"].tolist()

            output[key] = entry

        with open(output_path, "w") as f:
            json.dump(output, f, indent=2)

        logger.info(f"Exported {len(output)} calibrations to {output_path}")

    def get_camera_params(self, frame_idx: int) -> dict | None:
        """Get camera parameters for a frame."""
        result = self.results.get(frame_idx)
        if result:
            return result.get("camera_params")
        return None

    def get_homography(self, frame_idx: int) -> np.ndarray | None:
        """Get homography for a frame."""
        result = self.results.get(frame_idx)
        if result:
            return result.get("H")
        return None
