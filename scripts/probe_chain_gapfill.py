"""Feasibility probe: chain-propagate homography across uncalibrated frames.

Loads the existing kids_soccer run's homographies + metadata and runs
ChainCalibrator over selected gaps, writing visualizations so we can
eyeball whether SIFT-based H-chaining tracks the pitch through frames
where direct PnLCalib failed.

Run:
  source venv/bin/activate
  python scripts/probe_chain_gapfill.py
"""

from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path

import numpy as np


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import cv2

from goalinsight.field_registration.homography_chain.chain_calibrator import (
    ChainCalibrator,
)
from goalinsight.field_registration.homography_chain.homography_propagator import (
    HomographyPropagator,
)


VIDEO = REPO / "data/raw_videos/kids_soccer_clip_1250_1310.mp4"
RUN = REPO / "output/kids_soccer_clip_1250_1310_run20260526/field_registration"
PROBE_OUT = REPO / "output/kids_soccer_clip_1250_1310_run20260526/_probe_chain_gapfill"

# kids_soccer.yaml: long-focal sideline cam, fx=2448.68 (hfov ~43°)
IMG_W, IMG_H = 1920, 1080
FX = 2448.68
HFOV_DEG = float(np.degrees(2 * np.arctan(IMG_W / (2 * FX))))

# PnLCalib was solved against FIFA dims even when configs/kids_soccer.yaml
# overrides pitch.* (the override doesn't flow into pnl_solver). Use FIFA
# dims for the overlay so projections match the anchor frames.
PITCH_LENGTH = 105.0
PITCH_WIDTH = 68.0
PEN_AREA_W = 40.32  # 2 * 20.16
PEN_AREA_L = 16.5
GOAL_AREA_W = 18.32  # 2 * 9.16
GOAL_AREA_L = 5.5
CENTER_CIRCLE_R = 9.15


def draw_kids_pitch_overlay(frame: np.ndarray, H: np.ndarray, color=(0, 255, 255), thickness=2) -> np.ndarray:
    """Project FIFA pitch through H (world->image)."""
    out = frame.copy()
    h_img, w_img = out.shape[:2]
    half_l = PITCH_LENGTH / 2
    half_w = PITCH_WIDTH / 2

    def proj(pt):
        v = np.array([pt[0], pt[1], 1.0])
        u = H @ v
        if abs(u[2]) < 1e-9:
            return None
        return (int(u[0] / u[2]), int(u[1] / u[2]))

    def line(p1, p2):
        if p1 is None or p2 is None:
            return
        # Skip if both endpoints are far outside frame
        margin = 400
        if (p1[0] < -margin and p2[0] < -margin) or (p1[0] > w_img + margin and p2[0] > w_img + margin):
            return
        if (p1[1] < -margin and p2[1] < -margin) or (p1[1] > h_img + margin and p2[1] > h_img + margin):
            return
        cv2.line(out, p1, p2, color, thickness)

    # Touchlines + goal lines
    line(proj((-half_l, -half_w)), proj((half_l, -half_w)))
    line(proj((-half_l, half_w)), proj((half_l, half_w)))
    line(proj((-half_l, -half_w)), proj((-half_l, half_w)))
    line(proj((half_l, -half_w)), proj((half_l, half_w)))
    # Halfway line
    line(proj((0, -half_w)), proj((0, half_w)))
    # Center circle
    prev = None
    for i in range(73):
        a = 2 * np.pi * i / 72
        pt = proj((CENTER_CIRCLE_R * np.cos(a), CENTER_CIRCLE_R * np.sin(a)))
        if prev is not None:
            line(prev, pt)
        prev = pt
    # Penalty + goal areas
    for sign in (-1, 1):
        x_goal = sign * half_l
        x_pen = x_goal - sign * PEN_AREA_L
        x_ga = x_goal - sign * GOAL_AREA_L
        line(proj((x_goal, -PEN_AREA_W / 2)), proj((x_pen, -PEN_AREA_W / 2)))
        line(proj((x_goal, PEN_AREA_W / 2)), proj((x_pen, PEN_AREA_W / 2)))
        line(proj((x_pen, -PEN_AREA_W / 2)), proj((x_pen, PEN_AREA_W / 2)))
        line(proj((x_goal, -GOAL_AREA_W / 2)), proj((x_ga, -GOAL_AREA_W / 2)))
        line(proj((x_goal, GOAL_AREA_W / 2)), proj((x_ga, GOAL_AREA_W / 2)))
        line(proj((x_ga, -GOAL_AREA_W / 2)), proj((x_ga, GOAL_AREA_W / 2)))
    return out


def render_kids_overlays(label_dir: Path, results: dict, video_path: Path) -> None:
    out_dir = label_dir / "kids_overlay"
    out_dir.mkdir(exist_ok=True)
    cap = cv2.VideoCapture(str(video_path))
    for fidx in sorted(results.keys()):
        H = results[fidx].get("H")
        if H is None:
            continue
        cap.set(cv2.CAP_PROP_POS_FRAMES, fidx)
        ret, frame = cap.read()
        if not ret:
            continue
        ovr = draw_kids_pitch_overlay(frame, H)
        src = results[fidx].get("source", "?")
        conf = results[fidx].get("confidence", 0.0)
        cv2.putText(ovr, f"frame {fidx}  src={src}  conf={conf:.2f}", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
        cv2.imwrite(str(out_dir / f"frame_{fidx:06d}.jpg"), ovr)
    cap.release()


def load_run() -> tuple[dict, dict]:
    with open(RUN / "homographies.pkl", "rb") as f:
        homographies = pickle.load(f)
    with open(RUN / "calibration_metadata.json") as f:
        meta = json.load(f)
    return homographies, meta


def patch_calibrator_with_K(cc: ChainCalibrator) -> None:
    """Inject K so _calibrate_improved doesn't require camera_params."""
    cc.K = HomographyPropagator.build_intrinsic_matrix(IMG_W, IMG_H, HFOV_DEG)


def run_single_anchor_probe(
    label: str,
    anchor: int,
    direction: str,
    homographies: dict[int, np.ndarray],
    sample_step: int,
    range_start: int,
    range_end: int,
) -> dict:
    """Single-sided propagation from one anchor (works around the
    _calibrate_improved gating that skips backward when start==end anchor).
    """
    print(f"\n=== Probe: {label} ===")
    print(f"  anchor: {anchor}  direction: {direction}")
    print(f"  range = [{range_start}, {range_end}]")

    out_dir = PROBE_OUT / label
    out_dir.mkdir(parents=True, exist_ok=True)

    cc = ChainCalibrator(
        image_width=IMG_W, image_height=IMG_H,
        mode="offline", method="improved",
        smoothing_window=5, frame_step=sample_step,
    )
    patch_calibrator_with_K(cc)

    dummy_cp = {
        "panDegrees": 0.0, "tiltDegrees": 0.0, "rollDegrees": 0.0,
        "horizontalFieldOfViewDegrees": HFOV_DEG,
        "sensorResolutionWidthPixels": IMG_W, "sensorResolutionHeightPixels": IMG_H,
        "positionXMeters": 0.0, "positionYMeters": 0.0, "positionZMeters": 0.0,
    }
    cc.add_anchor(frame_idx=anchor, homography=homographies[anchor], confidence=1.0)
    cc.anchors[anchor]["camera_params"] = dict(dummy_cp)

    sample_frames = list(range(range_start, range_end + 1, sample_step))
    if range_end not in sample_frames:
        sample_frames.append(range_end)

    cap = cv2.VideoCapture(str(VIDEO))
    try:
        results = cc._propagate_improved(
            cap, sample_frames=sample_frames,
            anchor_frame=anchor, direction=direction,
            progress_callback=None,
        )
    finally:
        cap.release()

    n_with_H = sum(1 for r in results.values() if r.get("H") is not None)
    print(f"  produced H for {n_with_H}/{len(results)} sampled frames")
    render_kids_overlays(out_dir, results, VIDEO)
    return results


def run_probe(
    label: str,
    start_anchor: int,
    end_anchor: int,
    homographies: dict[int, np.ndarray],
    sample_step: int,
    range_start: int | None = None,
    range_end: int | None = None,
) -> dict:
    """Propagate from start_anchor..end_anchor using only those two anchors.

    range_start/range_end (defaults to start_anchor/end_anchor) let us probe
    extrapolation beyond the anchors.
    """
    if range_start is None:
        range_start = start_anchor
    if range_end is None:
        range_end = end_anchor

    print(f"\n=== Probe: {label} ===")
    print(f"  anchors: {start_anchor}, {end_anchor}")
    print(f"  range = [{range_start}, {range_end}]  ({range_end - range_start + 1} frames)")

    out_dir = PROBE_OUT / label
    out_dir.mkdir(parents=True, exist_ok=True)

    cc = ChainCalibrator(
        image_width=IMG_W,
        image_height=IMG_H,
        mode="offline",
        method="improved",
        smoothing_window=5,
        frame_step=sample_step,
    )
    patch_calibrator_with_K(cc)

    # add_anchor without camera_params (homography-only branch)
    cc.add_anchor(
        frame_idx=start_anchor,
        homography=homographies[start_anchor],
        confidence=1.0,
    )
    if end_anchor != start_anchor:
        cc.add_anchor(
            frame_idx=end_anchor,
            homography=homographies[end_anchor],
            confidence=1.0,
        )

    cc.enable_visualization(out_dir)

    # _calibrate_improved currently requires anchor_params["panDegrees"]; since
    # we're feeding homography-only anchors, attach a dummy camera_params dict
    # so the existing logic doesn't crash. We don't care about pan accuracy
    # here — we'll inspect the H-based projection overlay.
    dummy_cp = {
        "panDegrees": 0.0,
        "tiltDegrees": 0.0,
        "rollDegrees": 0.0,
        "horizontalFieldOfViewDegrees": HFOV_DEG,
        "sensorResolutionWidthPixels": IMG_W,
        "sensorResolutionHeightPixels": IMG_H,
        "positionXMeters": 0.0,
        "positionYMeters": 0.0,
        "positionZMeters": 0.0,
    }
    cc.anchors[start_anchor]["camera_params"] = dict(dummy_cp)
    if end_anchor != start_anchor:
        cc.anchors[end_anchor]["camera_params"] = dict(dummy_cp)

    results = cc.calibrate_range(
        VIDEO,
        start_frame=range_start,
        end_frame=range_end,
    )

    n_total = len(results)
    n_with_H = sum(1 for r in results.values() if r.get("H") is not None)
    print(f"  produced H for {n_with_H}/{n_total} sampled frames")
    print(f"  visualizations -> {out_dir}/merged/")

    # Quick sanity: drift of accumulated H vs anchors at endpoints
    if start_anchor in results and "H" in results[start_anchor]:
        H_start = results[start_anchor]["H"]
        diff = np.linalg.norm(H_start - homographies[start_anchor])
        print(f"  H[start]==anchor? diff norm={diff:.3e}")
    if end_anchor in results and "H" in results[end_anchor]:
        H_end = results[end_anchor]["H"]
        diff = np.linalg.norm(H_end - homographies[end_anchor])
        print(f"  H[end]==anchor?   diff norm={diff:.3e}")

    render_kids_overlays(out_dir, results, VIDEO)

    return results


def main() -> int:
    homographies, meta = load_run()
    cal_frames = sorted(homographies.keys())
    print(f"Loaded {len(cal_frames)} calibrated anchors")
    print(f"  range: [{cal_frames[0]}, {cal_frames[-1]}]")

    PROBE_OUT.mkdir(parents=True, exist_ok=True)

    # Probe A: backward extrapolation from earliest anchor 198 toward frame 0.
    # No anchor exists before 198, so this exercises pure single-sided
    # propagation across 66 uncalibrated sampled frames — the worst case.
    run_single_anchor_probe(
        label="A_extrapolate_back_from_198",
        anchor=198,
        direction="backward",
        homographies=homographies,
        sample_step=3,
        range_start=0,
        range_end=198,
    )

    # Probe B: bridge a real intra-range gap. The largest within the
    # calibrated bracket is (375, 390) — 4 uncalibrated sampled frames
    # between two solid anchors (16 frames span at sample_step=3).
    run_probe(
        label="B_bracketed_375_to_390",
        start_anchor=375,
        end_anchor=390,
        homographies=homographies,
        sample_step=3,
    )

    # Probe C: forward extrapolation from latest anchor 405 toward frame 597.
    run_single_anchor_probe(
        label="C_extrapolate_forward_from_405",
        anchor=405,
        direction="forward",
        homographies=homographies,
        sample_step=3,
        range_start=405,
        range_end=597,
    )

    print(f"\nAll probes done. Inspect: {PROBE_OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
