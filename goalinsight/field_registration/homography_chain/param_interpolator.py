"""Direct camera parameter interpolation between anchors.

Piecewise linear interpolation of pan/tilt/roll/FOV values between
anchor frames. Supports multiple anchors including turning points.
More stable than homography chaining for long gaps.
"""

from typing import Any

import numpy as np


class ParamInterpolator:
    """Interpolate camera parameters directly between anchors.

    This is a simpler and more stable approach than homography chaining
    for filling calibration gaps. Supports:
    - Multiple anchors (not just two endpoints)
    - Piecewise linear interpolation between consecutive anchors
    - Rate-based extrapolation beyond anchor range

    Works best when camera motion segments are approximately linear.
    """

    def __init__(
        self,
        extrapolation_frames: int = 30,
        easing: str = "linear",
    ):
        """Initialize interpolator.

        Args:
            extrapolation_frames: Max frames to extrapolate beyond anchors.
            easing: Easing function for interpolation
                    ("linear", "ease_out", "ease_in", "ease_in_out").
        """
        self.anchors: dict[int, dict] = {}
        self.extrapolation_frames = extrapolation_frames
        self.easing = easing

    def add_anchor(
        self,
        frame_idx: int,
        camera_params: dict,
        confidence: float = 1.0,
    ) -> None:
        """Add anchor frame with known calibration.

        Args:
            frame_idx: Frame index.
            camera_params: BroadTrack format camera parameters.
            confidence: Calibration confidence.
        """
        self.anchors[frame_idx] = {
            "camera_params": camera_params.copy(),
            "confidence": confidence,
        }

    def interpolate(
        self,
        frame_idx: int,
    ) -> dict[str, Any] | None:
        """Interpolate camera parameters for a frame.

        Args:
            frame_idx: Target frame index.

        Returns:
            Interpolated result with camera_params and confidence.
        """
        if not self.anchors:
            return None

        anchor_frames = sorted(self.anchors.keys())

        # Check if frame is an anchor
        if frame_idx in self.anchors:
            return {
                "camera_params": self.anchors[frame_idx]["camera_params"].copy(),
                "confidence": self.anchors[frame_idx]["confidence"],
                "source": "anchor",
            }

        # Find surrounding anchors
        before = [f for f in anchor_frames if f < frame_idx]
        after = [f for f in anchor_frames if f > frame_idx]

        if not before and not after:
            return None

        if not before:
            # Extrapolate backward from first anchor using rate
            anchor = after[0]
            frames_back = anchor - frame_idx
            if frames_back > self.extrapolation_frames:
                # Too far, just use anchor values
                return {
                    "camera_params": self.anchors[anchor]["camera_params"].copy(),
                    "confidence": 0.1,
                    "source": "extrapolate_forward_clamped",
                }

            # Try to get rate from anchor to next anchor
            rate = self._estimate_rate_at_anchor(anchor, direction="forward")
            extrapolated = self._extrapolate_params(
                self.anchors[anchor]["camera_params"],
                rate,
                -frames_back,  # Negative because going backward
            )
            confidence = 0.3 * (1 - frames_back / self.extrapolation_frames)
            return {
                "camera_params": extrapolated,
                "confidence": max(0.1, confidence),
                "source": "extrapolate_backward",
                "frames_from_anchor": frames_back,
            }

        if not after:
            # Extrapolate forward from last anchor using rate
            anchor = before[-1]
            frames_ahead = frame_idx - anchor
            if frames_ahead > self.extrapolation_frames:
                return {
                    "camera_params": self.anchors[anchor]["camera_params"].copy(),
                    "confidence": 0.1,
                    "source": "extrapolate_backward_clamped",
                }

            rate = self._estimate_rate_at_anchor(anchor, direction="backward")
            extrapolated = self._extrapolate_params(
                self.anchors[anchor]["camera_params"],
                rate,
                frames_ahead,
            )
            confidence = 0.3 * (1 - frames_ahead / self.extrapolation_frames)
            return {
                "camera_params": extrapolated,
                "confidence": max(0.1, confidence),
                "source": "extrapolate_forward",
                "frames_from_anchor": frames_ahead,
            }

        # Interpolate between two anchors
        anchor1 = before[-1]
        anchor2 = after[0]

        t = (frame_idx - anchor1) / (anchor2 - anchor1)

        params1 = self.anchors[anchor1]["camera_params"]
        params2 = self.anchors[anchor2]["camera_params"]

        # Interpolation with easing
        interpolated = self._lerp_params(params1, params2, t, easing=self.easing)

        # Confidence is highest in middle, lower at edges
        # Use parabolic curve: conf = 1 - 4*(t-0.5)^2 = 4*t*(1-t)
        base_conf = 4 * t * (1 - t)  # Max 1.0 at t=0.5
        conf1 = self.anchors[anchor1]["confidence"]
        conf2 = self.anchors[anchor2]["confidence"]
        confidence = base_conf * min(conf1, conf2)

        return {
            "camera_params": interpolated,
            "confidence": confidence,
            "source": "interpolate",
            "anchor1": anchor1,
            "anchor2": anchor2,
            "t": t,
        }

    def _lerp_params(
        self,
        params1: dict,
        params2: dict,
        t: float,
        easing: str = "linear",
    ) -> dict:
        """Interpolate camera parameters with optional easing.

        Args:
            params1: First anchor parameters.
            params2: Second anchor parameters.
            t: Interpolation factor [0, 1].
            easing: Easing function ("linear", "ease_out", "ease_in_out").

        Returns:
            Interpolated parameters.
        """
        # Apply easing to t
        t_eased = self._apply_easing(t, easing)

        def lerp(a, b, t):
            return a + t * (b - a)

        return {
            "panDegrees": lerp(params1["panDegrees"], params2["panDegrees"], t_eased),
            "tiltDegrees": lerp(params1["tiltDegrees"], params2["tiltDegrees"], t_eased),
            "rollDegrees": lerp(params1["rollDegrees"], params2["rollDegrees"], t_eased),
            "horizontalFieldOfViewDegrees": lerp(
                params1["horizontalFieldOfViewDegrees"],
                params2["horizontalFieldOfViewDegrees"],
                t_eased,
            ),
            # Keep position fixed (from first anchor)
            "positionXMeters": params1["positionXMeters"],
            "positionYMeters": params1["positionYMeters"],
            "positionZMeters": params1["positionZMeters"],
            "sensorResolutionWidthPixels": params1["sensorResolutionWidthPixels"],
            "sensorResolutionHeightPixels": params1["sensorResolutionHeightPixels"],
        }

    def _apply_easing(self, t: float, easing: str) -> float:
        """Apply easing function to interpolation factor.

        Args:
            t: Linear interpolation factor [0, 1].
            easing: Easing type.

        Returns:
            Eased interpolation factor.
        """
        if easing == "linear":
            return t
        elif easing == "ease_out":
            # Fast start, slow end (like camera settling)
            return 1 - (1 - t) ** 2
        elif easing == "ease_in":
            # Slow start, fast end
            return t ** 2
        elif easing == "ease_in_out":
            # Smooth acceleration and deceleration
            if t < 0.5:
                return 2 * t ** 2
            else:
                return 1 - (-2 * t + 2) ** 2 / 2
        else:
            return t

    def interpolate_range(
        self,
        start_frame: int,
        end_frame: int,
    ) -> dict[int, dict[str, Any]]:
        """Interpolate all frames in a range.

        Args:
            start_frame: Start frame (inclusive).
            end_frame: End frame (inclusive).

        Returns:
            Dictionary mapping frame_idx to interpolated results.
        """
        results = {}
        for frame_idx in range(start_frame, end_frame + 1):
            result = self.interpolate(frame_idx)
            if result:
                results[frame_idx] = result
        return results

    def _estimate_rate_at_anchor(
        self,
        anchor_frame: int,
        direction: str = "forward",
    ) -> dict[str, float]:
        """Estimate parameter rate of change at an anchor.

        Uses the slope to the adjacent anchor in the specified direction.

        Args:
            anchor_frame: The anchor frame index.
            direction: "forward" (use next anchor) or "backward" (use prev anchor).

        Returns:
            Dictionary of parameter rates (degrees/frame).
        """
        anchor_frames = sorted(self.anchors.keys())
        idx = anchor_frames.index(anchor_frame)

        # Default zero rate
        default_rate = {
            "panDegrees": 0.0,
            "tiltDegrees": 0.0,
            "rollDegrees": 0.0,
            "horizontalFieldOfViewDegrees": 0.0,
        }

        if direction == "forward" and idx < len(anchor_frames) - 1:
            next_anchor = anchor_frames[idx + 1]
            delta_frames = next_anchor - anchor_frame
            if delta_frames > 0:
                p1 = self.anchors[anchor_frame]["camera_params"]
                p2 = self.anchors[next_anchor]["camera_params"]
                return {
                    "panDegrees": (p2["panDegrees"] - p1["panDegrees"]) / delta_frames,
                    "tiltDegrees": (p2["tiltDegrees"] - p1["tiltDegrees"]) / delta_frames,
                    "rollDegrees": (p2["rollDegrees"] - p1["rollDegrees"]) / delta_frames,
                    "horizontalFieldOfViewDegrees": (
                        p2["horizontalFieldOfViewDegrees"] - p1["horizontalFieldOfViewDegrees"]
                    ) / delta_frames,
                }

        elif direction == "backward" and idx > 0:
            prev_anchor = anchor_frames[idx - 1]
            delta_frames = anchor_frame - prev_anchor
            if delta_frames > 0:
                p1 = self.anchors[prev_anchor]["camera_params"]
                p2 = self.anchors[anchor_frame]["camera_params"]
                return {
                    "panDegrees": (p2["panDegrees"] - p1["panDegrees"]) / delta_frames,
                    "tiltDegrees": (p2["tiltDegrees"] - p1["tiltDegrees"]) / delta_frames,
                    "rollDegrees": (p2["rollDegrees"] - p1["rollDegrees"]) / delta_frames,
                    "horizontalFieldOfViewDegrees": (
                        p2["horizontalFieldOfViewDegrees"] - p1["horizontalFieldOfViewDegrees"]
                    ) / delta_frames,
                }

        return default_rate

    def _extrapolate_params(
        self,
        base_params: dict,
        rate: dict[str, float],
        delta_frames: int,
    ) -> dict:
        """Extrapolate parameters using rate.

        Args:
            base_params: Base camera parameters.
            rate: Rate of change per frame.
            delta_frames: Number of frames to extrapolate.

        Returns:
            Extrapolated parameters.
        """
        return {
            "panDegrees": base_params["panDegrees"] + rate["panDegrees"] * delta_frames,
            "tiltDegrees": base_params["tiltDegrees"] + rate["tiltDegrees"] * delta_frames,
            "rollDegrees": base_params["rollDegrees"] + rate["rollDegrees"] * delta_frames,
            "horizontalFieldOfViewDegrees": (
                base_params["horizontalFieldOfViewDegrees"]
                + rate["horizontalFieldOfViewDegrees"] * delta_frames
            ),
            "positionXMeters": base_params["positionXMeters"],
            "positionYMeters": base_params["positionYMeters"],
            "positionZMeters": base_params["positionZMeters"],
            "sensorResolutionWidthPixels": base_params["sensorResolutionWidthPixels"],
            "sensorResolutionHeightPixels": base_params["sensorResolutionHeightPixels"],
        }

    def add_turning_point(
        self,
        frame_idx: int,
        pan_offset: float = 0.0,
        method: str = "midpoint",
    ) -> None:
        """Add a turning point anchor between existing anchors.

        Useful for capturing pan direction changes (e.g., camera pans left
        then right). The turning point values are estimated from adjacent anchors.

        Args:
            frame_idx: Frame index for turning point.
            pan_offset: Additional pan offset at turning point.
            method: "midpoint" (average of adjacent) or "extremum" (use offset).
        """
        if frame_idx in self.anchors:
            return  # Already an anchor

        anchor_frames = sorted(self.anchors.keys())
        before = [f for f in anchor_frames if f < frame_idx]
        after = [f for f in anchor_frames if f > frame_idx]

        if not before or not after:
            return  # Can't estimate without both sides

        prev_anchor = before[-1]
        next_anchor = after[0]

        p1 = self.anchors[prev_anchor]["camera_params"]
        p2 = self.anchors[next_anchor]["camera_params"]

        # Compute interpolation factor
        t = (frame_idx - prev_anchor) / (next_anchor - prev_anchor)

        # Base interpolated values
        interpolated = self._lerp_params(p1, p2, t)

        if method == "extremum":
            # Apply pan offset for turning point
            interpolated["panDegrees"] += pan_offset

        self.add_anchor(frame_idx, interpolated, confidence=0.8)

    def estimate_turning_point(
        self,
        start_frame: int,
        end_frame: int,
        pan_rate_per_second: float = -0.5,
        fps: float = 30.0,
    ) -> int | None:
        """Estimate where a turning point should be based on pan rate.

        For camera pan following action, estimates where the pan direction
        reverses based on expected pan rate and endpoint values.

        Args:
            start_frame: Start anchor frame.
            end_frame: End anchor frame.
            pan_rate_per_second: Expected pan rate in degrees/second (negative=left).
            fps: Video frame rate.

        Returns:
            Estimated turning point frame, or None if not applicable.
        """
        if start_frame not in self.anchors or end_frame not in self.anchors:
            return None

        p1 = self.anchors[start_frame]["camera_params"]
        p2 = self.anchors[end_frame]["camera_params"]

        pan1 = p1["panDegrees"]
        pan2 = p2["panDegrees"]

        # Total frame span
        total_frames = end_frame - start_frame
        duration_sec = total_frames / fps

        # Expected pan change if camera pans at constant rate then reverses
        # pan1 -> pan_min -> pan2
        # Let t be the turning point fraction
        # pan_min = pan1 + rate * t * duration
        # pan2 = pan_min + (-rate) * (1-t) * duration
        # Solving: pan2 = pan1 + rate*t*duration - rate*(1-t)*duration
        #        = pan1 + rate*duration*(2t - 1)
        # t = (pan2 - pan1) / (2 * rate * duration) + 0.5

        rate_per_frame = pan_rate_per_second / fps
        rate_total = pan_rate_per_second * duration_sec

        if abs(rate_total) < 0.1:
            return None

        t = (pan2 - pan1) / (2 * rate_total) + 0.5

        # Clamp to valid range
        if t < 0.2 or t > 0.8:
            return None  # Turning point outside reasonable range

        turning_frame = int(start_frame + t * total_frames)
        return turning_frame

    def setup_with_turning_point(
        self,
        start_frame: int,
        start_params: dict,
        end_frame: int,
        end_params: dict,
        turning_frame: int | None = None,
        pan_extremum: float | None = None,
    ) -> None:
        """Set up interpolation with automatic turning point.

        This replicates V15-style piecewise interpolation where the camera
        pans in one direction, reaches an extremum, then reverses.

        Args:
            start_frame: Start anchor frame.
            start_params: Start camera parameters.
            end_frame: End anchor frame.
            end_params: End camera parameters.
            turning_frame: Turning point frame (auto-estimated if None).
            pan_extremum: Pan value at turning point (auto-estimated if None).
        """
        self.clear()
        self.add_anchor(start_frame, start_params, confidence=1.0)
        self.add_anchor(end_frame, end_params, confidence=1.0)

        if turning_frame is None:
            turning_frame = self.estimate_turning_point(start_frame, end_frame)

        if turning_frame is None:
            return  # No turning point needed

        # Estimate pan at turning point
        if pan_extremum is None:
            # Use linear extrapolation from start
            pan1 = start_params["panDegrees"]
            pan2 = end_params["panDegrees"]

            # Estimate rate from start to turning point
            # Assume symmetric: same rate magnitude in both directions
            total_frames = end_frame - start_frame
            t = (turning_frame - start_frame) / total_frames

            # pan_extremum = pan1 + (pan_diff * t / (2*t - 1)) if t != 0.5
            # For t=0.5, pan_extremum = (pan1 + pan2) / 2 + offset
            if abs(t - 0.5) < 0.01:
                # Turning point at midpoint
                pan_extremum = (pan1 + pan2) / 2
            else:
                # Calculate extremum that gives symmetric rates
                # Rate from start to turn = (pan_extremum - pan1) / t
                # Rate from turn to end = (pan2 - pan_extremum) / (1-t)
                # For smooth motion, magnitudes should be similar
                # Solving: pan_extremum = pan1 + t * (pan2 - pan1) / (2*t - 1)
                if abs(2 * t - 1) > 0.01:
                    pan_extremum = pan1 + t * (pan2 - pan1) / (2 * t - 1)
                else:
                    pan_extremum = (pan1 + pan2) / 2

        # Create turning point params
        turning_params = self._lerp_params(
            start_params, end_params,
            (turning_frame - start_frame) / (end_frame - start_frame)
        )
        turning_params["panDegrees"] = pan_extremum

        self.add_anchor(turning_frame, turning_params, confidence=0.8)

    def load_from_broadtrack(
        self,
        calib_path: str,
        score_threshold: float = 0.35,
    ) -> list[int]:
        """Load high-confidence anchors from BroadTrack calibration file.

        Args:
            calib_path: Path to BroadTrack JSON calibration file.
            score_threshold: Minimum score for anchor selection.

        Returns:
            List of anchor frame indices that were loaded.
        """
        import json
        from pathlib import Path

        with open(calib_path) as f:
            calib = json.load(f)

        loaded_frames = []
        for key, data in calib.items():
            score = data.get("score", 0)
            if score >= score_threshold:
                # Extract frame number from key (supports both formats)
                if "/" in key:
                    # Path format: .../000680.jpg
                    frame_str = Path(key).stem
                    frame_idx = int(frame_str)
                else:
                    # Direct frame number
                    frame_idx = int(key)

                self.add_anchor(frame_idx, data["cp"], confidence=score)
                loaded_frames.append(frame_idx)

        return sorted(loaded_frames)

    def get_anchor_gaps(self, min_gap: int = 30) -> list[tuple[int, int]]:
        """Find gaps between anchors that need interpolation.

        Args:
            min_gap: Minimum gap size to report.

        Returns:
            List of (start_frame, end_frame) tuples for gaps.
        """
        anchor_frames = sorted(self.anchors.keys())
        gaps = []

        for i in range(len(anchor_frames) - 1):
            gap_size = anchor_frames[i + 1] - anchor_frames[i]
            if gap_size >= min_gap:
                gaps.append((anchor_frames[i], anchor_frames[i + 1]))

        return gaps

    def export_calibration(
        self,
        output_path: str,
        frame_range: tuple[int, int] | None = None,
        base_calib_path: str | None = None,
    ) -> None:
        """Export interpolated calibration to JSON file.

        Args:
            output_path: Output JSON path.
            frame_range: Optional (start, end) frame range.
            base_calib_path: Base calibration file to merge with.
        """
        import json
        from pathlib import Path

        # Load base calibration if provided
        if base_calib_path:
            with open(base_calib_path) as f:
                output = json.load(f)
        else:
            output = {}

        # Determine frame range
        if frame_range:
            start, end = frame_range
        else:
            anchor_frames = sorted(self.anchors.keys())
            if len(anchor_frames) < 2:
                return
            start, end = anchor_frames[0], anchor_frames[-1]

        # Interpolate all frames
        for frame_idx in range(start, end + 1):
            result = self.interpolate(frame_idx)
            if result is None:
                continue

            # Create output entry
            key = str(frame_idx)
            if key not in output:
                output[key] = {"cp": {}, "score": 0.0}

            output[key]["cp"] = result["camera_params"]
            output[key]["confidence"] = result["confidence"]
            output[key]["source"] = result.get("source", "interpolate")
            output[key]["interpolated"] = result.get("source") != "anchor"

        # Save
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(output, f, indent=2)

    def clear(self) -> None:
        """Clear all anchors."""
        self.anchors.clear()
