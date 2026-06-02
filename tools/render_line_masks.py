"""Render pitch-line segmentation masks from manual annotations OR HRNet detections.

Three modes:

* ``--source manual`` (default) — read ``frame_<idx>.json.lines`` from the
  annotation directory and draw each clicked line segment. The endpoints
  are pixel-accurate because a human picked them on the white line.

* ``--source hrnet-points`` — run the PnLCalib HRNet keypoint model on
  each ``frame_*_raw.jpg`` to recover PnLCalib keypoint pixels, then
  connect them along the known pitch topology in
  ``utils/pitch.PITCH_LINE_KEYPOINTS``. No homography solve.

* ``--source hrnet-lines`` — run the PnLCalib HRNet *line* model. Each of
  the 23 line classes outputs a heatmap whose top-2 peaks are the line's
  two endpoints; we draw each class as a single segment between those
  peaks. This is closer in spirit to 1810.10658's edge-image target.

Output is one PNG per frame:

- ``frame_<idx>_mask.png``        binary 0/255 line mask
- ``frame_<idx>_overlay.png``     (with ``--debug``) raw frame + mask overlay
- ``frame_<idx>_mask_classes.png`` (with ``--multi-class``) class-id mask

This produces the same kind of "field-marking edge image" 1810.10658 trains
its 2-GAN model to predict — but directly from labels or detections, so we
can train a supervised line-mask network and skip the GAN.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

# Resolve project root before relative imports.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from goalinsight.utils.pitch import PITCH_LINE_KEYPOINTS  # noqa: E402


# Class id per topology chain in PITCH_LINE_KEYPOINTS, in order.
CHAIN_CLASS_IDS: list[int] = [
    1, 1,                  # top + bottom boundary
    1, 1,                  # sidelines
    2,                     # center line
    4, 4, 4,               # left penalty rectangle
    4, 4, 4,               # right penalty rectangle
    5, 5, 5,               # left goal area
    5, 5, 5,               # right goal area
    3,                     # center circle (closed loop)
    6, 6,                  # left + right penalty arcs
]
assert len(CHAIN_CLASS_IDS) == len(PITCH_LINE_KEYPOINTS)


# Class id per HRNetLineModel.LINE_CLASSES index (--source hrnet-lines).
# Mirrors the manual-line bucketing below so all three sources produce
# class ids in the same coordinate system.
LINE_CLASS_TO_CLASS_ID: dict[int, int] = {
    # Big rect (penalty area): class 4
    0: 4, 1: 4, 2: 4,
    3: 4, 4: 4, 5: 4,
    # Goal posts / crossbars: class 7 (not in manual mode, kept distinct)
    6: 7, 7: 7, 8: 7,
    9: 7, 10: 7, 11: 7,
    # Middle line: 2
    12: 2,
    # Touchlines / goal lines (boundary): 1
    13: 1, 14: 1, 15: 1, 16: 1,
    # Small rect (goal area): 5
    17: 5, 18: 5, 19: 5,
    20: 5, 21: 5, 22: 5,
}


# Map each annotator line name to a class id (for --source manual + multi-class).
LINE_NAME_TO_CLASS: dict[str, int] = {
    "touchline_top": 1,
    "touchline_bottom": 1,
    "goal_line_left": 1,
    "goal_line_right": 1,
    "center_line": 2,
    "penalty_left_top": 4,
    "penalty_left_bottom": 4,
    "penalty_left_front": 4,
    "penalty_right_top": 4,
    "penalty_right_bottom": 4,
    "penalty_right_front": 4,
    "goal_area_left_top": 5,
    "goal_area_left_bottom": 5,
    "goal_area_left_front": 5,
    "goal_area_right_top": 5,
    "goal_area_right_bottom": 5,
    "goal_area_right_front": 5,
}


def _draw_polyline(canvas, pts, *, color, thickness):
    pts = [p for p in pts if p is not None]
    if len(pts) < 2:
        return
    arr = np.array(pts, dtype=np.int32).reshape(-1, 1, 2)
    cv2.polylines(canvas, [arr], isClosed=False, color=color,
                  thickness=thickness, lineType=cv2.LINE_AA)


# ---- manual-source rendering ----------------------------------------------

def render_mask_from_lines(
    annotated_lines: list[dict],
    image_shape: tuple[int, int],
    *,
    line_width: int,
    multi_class: bool,
) -> np.ndarray:
    h, w = image_shape
    canvas = np.zeros((h, w), dtype=np.uint8)
    for ln in annotated_lines:
        (x1, y1), (x2, y2) = ln["pixels"][0], ln["pixels"][1]
        cls = LINE_NAME_TO_CLASS.get(ln.get("name", ""), 1)
        color = int(cls) if multi_class else 255
        cv2.line(
            canvas,
            (int(round(x1)), int(round(y1))),
            (int(round(x2)), int(round(y2))),
            color,
            line_width,
            lineType=cv2.LINE_AA,
        )
    return canvas


# ---- hrnet-source rendering -----------------------------------------------

def render_mask_from_keypoints(
    keypoints: list[dict],
    image_shape: tuple[int, int],
    *,
    line_width: int,
    multi_class: bool,
    confidence_threshold: float,
) -> np.ndarray:
    """Connect HRNet keypoints along PITCH_LINE_KEYPOINTS topology.

    ``keypoints`` is the raw list returned by KeypointDetector.detect(...,
    convert_to_soccernet=False) — each entry has ``id`` (pnlcalib id),
    ``x``, ``y``, ``confidence``.
    """
    h, w = image_shape

    pix_by_id: dict[int, tuple[int, int]] = {}
    for kp in keypoints:
        if kp.get("confidence", 0.0) < confidence_threshold:
            continue
        kp_id = kp.get("id")
        if kp_id is None or kp_id < 0:
            continue
        pix_by_id[int(kp_id)] = (
            int(round(float(kp["x"]))),
            int(round(float(kp["y"]))),
        )

    canvas = np.zeros((h, w), dtype=np.uint8)
    for chain, cls in zip(PITCH_LINE_KEYPOINTS, CHAIN_CLASS_IDS):
        pts = [pix_by_id.get(kp_id) for kp_id in chain]
        color = int(cls) if multi_class else 255
        _draw_polyline(canvas, pts, color=color, thickness=line_width)
    return canvas


def _build_hrnet_detector(
    model_path: str,
    weights: str,
    confidence_threshold: float,
):
    from goalinsight.field_registration.keypoint_detector import KeypointDetector
    detector = KeypointDetector({
        "backend": "pnlcalib",
        "pnlcalib": {
            "weights": weights,
            "model_path": model_path,
            "confidence_threshold": confidence_threshold,
        },
    })
    detector.load_model(model_path)
    return detector


# ---- hrnet-lines source rendering -----------------------------------------

class _LineInferencer:
    """Run the finetuned PnLCalib line model on a frame.

    The line model outputs (1, 24, H', W') = 23 class channels + 1 background.
    For each class channel we take the top-2 peaks via maxpool NMS — those
    are the line's endpoints.
    """

    def __init__(self, ckpt_path: str, device: str | None = None):
        import torch
        from goalinsight.field_registration.pnlcalib import HRNetLineModel
        self._torch = torch
        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.model = HRNetLineModel(num_line_classes=23)
        self.model.load_pretrained(ckpt_path)
        self.model = self.model.to(self.device).eval()

    def detect(self, frame_bgr, *, threshold: float) -> list[dict]:
        from goalinsight.field_registration.pnlcalib import (
            get_lines_from_heatmap_maxpool,
        )
        torch = self._torch

        h, w = frame_bgr.shape[:2]
        resized = cv2.resize(frame_bgr, (960, 540))
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        tensor = (
            torch.from_numpy(rgb).float().permute(2, 0, 1).div(255.0)
            .unsqueeze(0).to(self.device)
        )
        with torch.no_grad():
            heatmaps = self.model(tensor)
        # Drop background channel.
        line_hm = heatmaps[:, :-1, :, :]
        scale_x = w / heatmaps.shape[3]
        scale_y = h / heatmaps.shape[2]
        # scale=1 returns raw heatmap pixel coords; we then rescale per-axis.
        lines = get_lines_from_heatmap_maxpool(
            line_hm, scale=1, threshold=threshold,
        )
        for ln in lines:
            ln["x1"] *= scale_x
            ln["y1"] *= scale_y
            ln["x2"] *= scale_x
            ln["y2"] *= scale_y
        return lines


def render_mask_from_line_model(
    lines: list[dict],
    image_shape: tuple[int, int],
    *,
    line_width: int,
    multi_class: bool,
) -> np.ndarray:
    h, w = image_shape
    canvas = np.zeros((h, w), dtype=np.uint8)
    for ln in lines:
        cls_idx = int(ln["id"])
        cls = LINE_CLASS_TO_CLASS_ID.get(cls_idx, 1)
        color = int(cls) if multi_class else 255
        cv2.line(
            canvas,
            (int(round(ln["x1"])), int(round(ln["y1"]))),
            (int(round(ln["x2"])), int(round(ln["y2"]))),
            color,
            line_width,
            lineType=cv2.LINE_AA,
        )
    return canvas


# ---- shared I/O helpers ---------------------------------------------------

def _save_overlay(img: np.ndarray, mask: np.ndarray, out_path: Path) -> None:
    if img.shape[:2] != mask.shape[:2]:
        mask = cv2.resize(mask, (img.shape[1], img.shape[0]),
                          interpolation=cv2.INTER_NEAREST)
    overlay = img.copy()
    overlay[mask > 0] = (0, 255, 255)
    blended = cv2.addWeighted(img, 0.55, overlay, 0.45, 0)
    cv2.imwrite(str(out_path), blended)


def _resolve_raw_image(annotations_dir: Path, frame_idx: int) -> Path | None:
    for cand in (
        annotations_dir / f"frame_{frame_idx}_raw.jpg",
        annotations_dir / f"frame_{frame_idx}.jpg",
    ):
        if cand.exists():
            return cand
    return None


def _iter_manual_frames(annotations_dir: Path):
    for ap in sorted(annotations_dir.glob("frame_*.json")):
        if ap.name.endswith("_all_points.json"):
            continue
        frame_idx = int(ap.stem.split("_")[1])
        yield frame_idx, ap


def _iter_image_frames(annotations_dir: Path):
    """For HRNet mode: walk every frame_*_raw.jpg under the dir."""
    seen: set[int] = set()
    for ip in sorted(annotations_dir.glob("frame_*_raw.jpg")):
        try:
            frame_idx = int(ip.stem.split("_")[1])
        except ValueError:
            continue
        if frame_idx in seen:
            continue
        seen.add(frame_idx)
        yield frame_idx, ip


def _iter_video_frames(video_path: Path, sample_fps: float):
    """Yield (frame_idx, bgr_frame) sampled at ~``sample_fps`` from a video.

    We compute a frame stride from the source FPS and grab every Nth frame.
    For typical 25/30 fps inputs and sample_fps=1, the stride is exact; for
    non-integer ratios the stride rounds — at 1 fps that's a sub-frame
    drift and not worth chasing.
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if src_fps <= 0:
        cap.release()
        raise RuntimeError(f"Video has no fps metadata: {video_path}")
    stride = max(1, int(round(src_fps / sample_fps)))
    print(f"Video: {total} frames @ {src_fps:.2f} fps, stride={stride} "
          f"(~{src_fps / stride:.2f} fps sampled)")

    try:
        for frame_idx in range(0, total, stride):
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ok, frame = cap.read()
            if not ok or frame is None:
                continue
            yield frame_idx, frame
    finally:
        cap.release()


# ---- main ------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations", type=Path,
                        help="Directory containing frame_*_raw.jpg "
                             "(and frame_<idx>.json for --source manual). "
                             "Mutually exclusive with --video.")
    parser.add_argument("--video", type=Path,
                        help="Video file to sample frames from. Implies "
                             "--source hrnet-lines (or hrnet-points).")
    parser.add_argument("--sample-fps", type=float, default=1.0,
                        help="Sampling rate when using --video (default: 1)")
    parser.add_argument("--output", required=True, type=Path,
                        help="Directory to write mask PNGs into")
    parser.add_argument(
        "--source",
        choices=("manual", "hrnet-points", "hrnet-lines"),
        default="manual",
        help="manual=draw frame_<idx>.json clicked lines; "
             "hrnet-points=run keypoint HRNet and connect detected "
             "keypoints along pitch topology; "
             "hrnet-lines=run line HRNet and draw segment per class.",
    )
    parser.add_argument("--ckpt", type=str,
                        default="data/finetuned_models/run_20260528_134416/models/best_model.pt",
                        help="Keypoint HRNet checkpoint (--source hrnet-points)")
    parser.add_argument("--line-ckpt", type=str,
                        default="data/finetuned_line_models/run_20260528_104928/models/best_model.pt",
                        help="Line HRNet checkpoint (--source hrnet-lines)")
    parser.add_argument("--weights", default="SV_kp",
                        help="Backbone weights tag (SV_kp / WC14_kp / TSWC_kp)")
    parser.add_argument("--kp-threshold", type=float, default=0.15,
                        help="Per-keypoint confidence threshold")
    parser.add_argument("--line-threshold", type=float, default=0.05,
                        help="Per-line endpoint confidence threshold "
                             "(--source hrnet-lines)")
    parser.add_argument("--line-width", type=int, default=3)
    parser.add_argument("--multi-class", action="store_true")
    parser.add_argument("--debug", action="store_true",
                        help="Save mask overlaid on the raw frame")
    args = parser.parse_args()

    if (args.annotations is None) == (args.video is None):
        print("ERROR: pass exactly one of --annotations or --video",
              file=sys.stderr)
        return 2
    if args.video is not None:
        if args.source == "manual":
            print("ERROR: --video requires --source hrnet-lines (or hrnet-points)",
                  file=sys.stderr)
            return 2
        if not args.video.is_file():
            print(f"ERROR: video not found: {args.video}", file=sys.stderr)
            return 2
    elif not args.annotations.is_dir():
        print(f"ERROR: dir not found: {args.annotations}", file=sys.stderr)
        return 2
    args.output.mkdir(parents=True, exist_ok=True)

    detector = None
    line_inferencer = None
    use_video = args.video is not None

    if args.source == "hrnet-points":
        detector = _build_hrnet_detector(
            args.ckpt, args.weights, args.kp_threshold,
        )
    elif args.source == "hrnet-lines":
        line_inferencer = _LineInferencer(args.line_ckpt)

    if use_video:
        frame_iter = _iter_video_frames(args.video, args.sample_fps)
        what = "video frames"
    elif args.source == "hrnet-points" or args.source == "hrnet-lines":
        frame_iter = _iter_image_frames(args.annotations)
        what = "raw images"
    else:
        frame_iter = _iter_manual_frames(args.annotations)
        what = "manual annotation JSONs"

    n_ok, n_skip = 0, 0
    for frame_idx, item in frame_iter:
        if use_video:
            img = item
            raw_path = None
        else:
            raw_path = _resolve_raw_image(args.annotations, frame_idx)
            if raw_path is None:
                print(f"  SKIP frame {frame_idx}: raw image not found")
                n_skip += 1
                continue
            img = cv2.imread(str(raw_path))
            if img is None:
                print(f"  SKIP frame {frame_idx}: raw image unreadable")
                n_skip += 1
                continue

        if args.source == "manual":
            with open(item) as f:
                data = json.load(f)
            annotated_lines = data.get("lines") or []
            if not annotated_lines:
                print(f"  SKIP frame {frame_idx}: no manual lines in {item.name}")
                n_skip += 1
                continue
            mask = render_mask_from_lines(
                annotated_lines, img.shape[:2],
                line_width=args.line_width, multi_class=args.multi_class,
            )
            detail = f"{len(annotated_lines)} manual lines"
        elif args.source == "hrnet-points":
            keypoints = detector.detect(img, convert_to_soccernet=False)
            kept = sum(
                1 for kp in keypoints
                if kp.get("confidence", 0.0) >= args.kp_threshold
            )
            mask = render_mask_from_keypoints(
                keypoints, img.shape[:2],
                line_width=args.line_width,
                multi_class=args.multi_class,
                confidence_threshold=args.kp_threshold,
            )
            detail = f"{kept}/{len(keypoints)} HRNet kps >= {args.kp_threshold}"
        else:  # hrnet-lines
            lines = line_inferencer.detect(img, threshold=args.line_threshold)
            mask = render_mask_from_line_model(
                lines, img.shape[:2],
                line_width=args.line_width, multi_class=args.multi_class,
            )
            detail = f"{len(lines)}/23 line classes >= {args.line_threshold}"

        out_name = "mask_classes.png" if args.multi_class else "mask.png"
        out_path = args.output / f"frame_{frame_idx}_{out_name}"
        cv2.imwrite(str(out_path), mask)
        if args.debug:
            _save_overlay(
                img,
                (mask > 0).astype(np.uint8) * 255,
                args.output / f"frame_{frame_idx}_overlay.png",
            )
        n_ok += 1
        print(f"  OK   frame {frame_idx}: {detail} -> {out_path.name}")

    if n_ok == 0 and n_skip == 0:
        print(f"No {what} found under {args.annotations}", file=sys.stderr)
        return 1
    print(f"Done: {n_ok} masks, {n_skip} skipped.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
