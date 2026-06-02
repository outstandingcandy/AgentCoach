"""Fill homography gaps via SIFT-based frame-to-frame chaining.

Adapter around ``homography_chain.ChainCalibrator``. The motivating use
case: PnLCalib only solves the frames where the keypoint detector saw
enough field markings. On a 60-second kids-soccer clip that's ~71/200
sampled frames, leaving 129 with ``None`` homography. Adjacent frames
share the same camera ~99% of the time, so we can fill the gaps by
SIFT-tracking the pitch through the missing frames using the surrounding
PnLCalib solves as anchors.

Called from ``_run_stage1_pnlcalib_orig`` after the per-frame solve loop
and before output is saved. Mutates the ``homographies`` dict and
``calibration_results["frames"]`` in place — downstream stages then see
a denser ``homographies.pkl`` with no schema change.

Design choices:

- We feed homography-only anchors (no upstream camera_params). The
  underlying ``_calibrate_improved`` requires ``camera_params`` for pan
  extraction; we attach a dummy ``panDegrees=0`` and an explicit ``K``
  built from the config's ``intrinsics.fx`` / image size. Pan accuracy
  doesn't matter — we only consume the propagated ``H`` field.

- Gaps split at the calibrated bracket edges. Frames before the first
  anchor / after the last get one-sided extrapolation via
  ``_propagate_improved`` directly — ``calibrate_range`` would set
  ``end_anchor = start_anchor`` and silently skip the backward pass,
  emitting only the anchor frame.

- ``max_gap_frames`` lets the caller bail out of long propagations
  (drift accumulates linearly with hops). When set, runs longer than
  the threshold pass through unchanged.

- Newly-filled frames carry ``"interpolated": true`` so
  ``compute_calibration_stats(..., exclude_interpolated=True)`` can keep
  real-vs-interpolated rates separable.
"""

from __future__ import annotations

import logging
from pathlib import Path

import cv2
import numpy as np

from .homography_chain.chain_calibrator import ChainCalibrator
from .homography_chain.homography_propagator import HomographyPropagator


logger = logging.getLogger(__name__)


_DUMMY_CP = {
    "panDegrees": 0.0,
    "tiltDegrees": 0.0,
    "rollDegrees": 0.0,
    # horizontalFieldOfViewDegrees is filled in per-call from intrinsics.
    "sensorResolutionWidthPixels": 1920,
    "sensorResolutionHeightPixels": 1080,
    "positionXMeters": 0.0,
    "positionYMeters": 0.0,
    "positionZMeters": 0.0,
}


def _hfov_from_fx(fx: float, image_width: int) -> float:
    """Horizontal field of view in degrees, given focal length in
    pixels and image width."""
    return float(np.degrees(2.0 * np.arctan(image_width / (2.0 * fx))))


def _build_calibrator(
    image_width: int,
    image_height: int,
    hfov_deg: float,
    method: str,
    smoothing_window: int,
    use_masking: bool,
    frame_step: int,
) -> ChainCalibrator:
    cc = ChainCalibrator(
        image_width=image_width,
        image_height=image_height,
        mode="offline",
        method=method,
        use_masking=use_masking,
        smoothing_window=smoothing_window,
        frame_step=frame_step,
    )
    # _calibrate_improved derives K from the first anchor's
    # horizontalFieldOfViewDegrees, but homography-only anchors don't
    # carry one — set K explicitly so it's never None when the loop
    # checks self.K.
    cc.K = HomographyPropagator.build_intrinsic_matrix(
        image_width, image_height, hfov_deg,
    )
    return cc


def _attach_anchor(
    cc: ChainCalibrator,
    frame_idx: int,
    H: np.ndarray,
    hfov_deg: float,
) -> None:
    """Add an anchor with a dummy ``camera_params`` so
    ``_propagate_improved`` can read panDegrees / FOV without crashing."""
    cc.add_anchor(frame_idx=frame_idx, homography=H, confidence=1.0)
    cp = dict(_DUMMY_CP)
    cp["horizontalFieldOfViewDegrees"] = hfov_deg
    cp["sensorResolutionWidthPixels"] = cc.image_width
    cp["sensorResolutionHeightPixels"] = cc.image_height
    cc.anchors[frame_idx]["camera_params"] = cp


def _split_runs(
    sampled_frames: list[int],
    anchor_set: set[int],
) -> list[dict]:
    """Group consecutive uncalibrated sampled frames into runs.

    Each run is dict(start, end, left_anchor, right_anchor); the anchor
    fields are None when the run is one-sided (before first / after last
    anchor).
    """
    runs: list[dict] = []
    current_start: int | None = None
    sorted_anchors = sorted(anchor_set)

    def left_anchor_of(frame: int) -> int | None:
        prev = [a for a in sorted_anchors if a < frame]
        return prev[-1] if prev else None

    def right_anchor_of(frame: int) -> int | None:
        nxt = [a for a in sorted_anchors if a > frame]
        return nxt[0] if nxt else None

    for f in sampled_frames:
        if f in anchor_set:
            if current_start is not None:
                runs.append({
                    "start": current_start,
                    "end": prev_uncal,
                    "left_anchor": left_anchor_of(current_start),
                    "right_anchor": right_anchor_of(prev_uncal),
                })
                current_start = None
            continue
        if current_start is None:
            current_start = f
        prev_uncal = f

    if current_start is not None:
        runs.append({
            "start": current_start,
            "end": prev_uncal,
            "left_anchor": left_anchor_of(current_start),
            "right_anchor": right_anchor_of(prev_uncal),
        })

    return runs


def fill_gaps_with_chain(
    *,
    video_path: Path,
    sampled_frames: list[int],
    image_width: int,
    image_height: int,
    homographies: dict[int, np.ndarray],
    calibration_results: dict,
    fx: float,
    config: dict,
) -> dict:
    """Fill missing entries in ``homographies`` via SIFT chaining.

    Args:
        video_path: Source video.
        sampled_frames: All frame indices the runner asked PnLCalib to
            solve. These are the candidates for gap-filling.
        image_width / image_height: Frame dimensions.
        homographies: ``{frame_idx: H}`` mutated in place.
        calibration_results: The metadata dict whose ``frames`` map will
            also get ``calibrated=True, interpolated=True`` entries for
            newly-filled frames.
        fx: Focal length in pixels (from ``intrinsics.fx``). Used only
            to derive the K that ``ChainCalibrator`` needs for rotation
            extraction.
        config: The ``field_registration.gap_filling`` block.

    Returns:
        Stats dict ``{filled, skipped_too_long, failed}``.
    """
    method = config.get("method", "improved")
    smoothing_window = int(config.get("smoothing_window", 5))
    use_masking = bool(config.get("use_masking", False))
    frame_step = int(config.get("frame_step", 1))
    max_gap_frames = config.get("max_gap_frames")

    hfov_deg = _hfov_from_fx(fx, image_width)
    logger.info(
        "Gap-fill: fx=%.1f → hfov=%.2f° on %dx%d, method=%s, "
        "smoothing=%d, masking=%s, max_gap=%s",
        fx, hfov_deg, image_width, image_height,
        method, smoothing_window, use_masking, max_gap_frames,
    )

    anchor_set = set(homographies.keys())
    sampled_frames = sorted(sampled_frames)
    runs = _split_runs(sampled_frames, anchor_set)

    stats = {"filled": 0, "skipped_too_long": 0, "failed": 0, "runs": len(runs)}

    for run in runs:
        run_start = run["start"]
        run_end = run["end"]
        left = run["left_anchor"]
        right = run["right_anchor"]

        if left is None and right is None:
            stats["failed"] += 1
            continue

        # Length: distance the chain has to propagate. For two-sided
        # gaps it's the anchor span; for one-sided extrapolation it's
        # how far the run extends past the only anchor.
        if left is not None and right is not None:
            span = right - left
            sub_start, sub_end = run_start, run_end
        elif right is not None:
            # backward extrapolation: only fill frames within
            # max_gap_frames of the right anchor; frames further out
            # would only accumulate drift.
            span = right - run_start
            if max_gap_frames is not None and span > int(max_gap_frames):
                sub_start = right - int(max_gap_frames)
                sub_end = run_end
                logger.info(
                    "  clamping backward run [%d-%d] -> [%d-%d] "
                    "(anchor=%d, max_gap=%d)",
                    run_start, run_end, sub_start, sub_end,
                    right, max_gap_frames,
                )
                span = right - sub_start
            else:
                sub_start, sub_end = run_start, run_end
        else:  # left is not None, right is None
            span = run_end - left
            if max_gap_frames is not None and span > int(max_gap_frames):
                sub_start = run_start
                sub_end = left + int(max_gap_frames)
                logger.info(
                    "  clamping forward run [%d-%d] -> [%d-%d] "
                    "(anchor=%d, max_gap=%d)",
                    run_start, run_end, sub_start, sub_end,
                    left, max_gap_frames,
                )
                span = sub_end - left
            else:
                sub_start, sub_end = run_start, run_end
        if max_gap_frames is not None and span > int(max_gap_frames):
            logger.info(
                "  skipping run [%d-%d] span=%d > max_gap=%d",
                run_start, run_end, span, max_gap_frames,
            )
            stats["skipped_too_long"] += 1
            continue
        run_start, run_end = sub_start, sub_end

        cc = _build_calibrator(
            image_width, image_height, hfov_deg,
            method, smoothing_window, use_masking, frame_step,
        )
        if left is not None:
            _attach_anchor(cc, left, homographies[left], hfov_deg)
        if right is not None and right != left:
            _attach_anchor(cc, right, homographies[right], hfov_deg)

        try:
            if left is not None and right is not None:
                results = cc.calibrate_range(
                    video_path, start_frame=run_start, end_frame=run_end,
                )
            else:
                # One-sided: bypass calibrate_range's bidirectional
                # gating that would skip the only meaningful direction.
                cap = cv2.VideoCapture(str(video_path))
                try:
                    sample_frames_run = list(
                        range(run_start, run_end + 1, frame_step)
                    )
                    if run_end not in sample_frames_run:
                        sample_frames_run.append(run_end)
                    direction = "backward" if right is not None else "forward"
                    anchor = right if right is not None else left
                    results = cc._propagate_improved(
                        cap, sample_frames=sample_frames_run,
                        anchor_frame=anchor,
                        direction=direction,
                        progress_callback=None,
                    )
                finally:
                    cap.release()
        except Exception as e:  # noqa: BLE001 — defensive; log and move on
            logger.warning(
                "  gap-fill failed on run [%d-%d]: %s", run_start, run_end, e,
            )
            stats["failed"] += 1
            continue

        # Write back, but only for frames the runner originally sampled.
        # ChainCalibrator may emit extras at frame_step boundaries that
        # PnLCalib never had in its frames map.
        for fidx, res in results.items():
            if fidx not in sampled_frames:
                continue
            if fidx in homographies:
                continue  # don't overwrite a real anchor
            H = res.get("H")
            if H is None:
                continue
            homographies[fidx] = H
            calibration_results["frames"][fidx] = {
                "calibrated": True,
                "interpolated": True,
                "source": res.get("source", "chain"),
                "confidence": float(res.get("confidence", 0.0)),
                "left_anchor": left,
                "right_anchor": right,
            }
            stats["filled"] += 1

    logger.info(
        "Gap-fill done: filled=%d, skipped_too_long=%d, failed=%d, runs=%d",
        stats["filled"], stats["skipped_too_long"],
        stats["failed"], stats["runs"],
    )
    return stats
