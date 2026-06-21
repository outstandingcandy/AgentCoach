"""DataLoader for line annotations, generating endpoint heatmaps.

Companion to ``point_dataloader.py``. Loads ``frame_*.json`` files (NOT
``_all_points.json`` — those only carry points, not lines) and produces the
24-channel target heatmap PnLCalib's line model expects:

- 23 line-class channels, each with two Gaussian peaks marking the line's
  two endpoints (the upstream decoder uses ``topk(k=2)`` per channel — see
  ``heatmap_utils.get_lines_from_heatmap_maxpool``).
- 1 trailing background channel = ``1 - sum(line channels)``, clipped to
  [0, 1]. Mirrors the keypoint trainer's convention.

Mask is per-class (length 23): 1 if the line is annotated in the frame, 0
otherwise. ``MaskedMSELoss`` (the keypoint trainer's loss) only penalizes
present classes, so unseen classes don't get pushed to zero — important
because our 3-frame kids dataset only covers ~13 / 23 classes.
"""

from __future__ import annotations

import glob
import json
import os
from pathlib import Path
from typing import Callable

import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from .point_dataloader import (
    KeypointRandomZoomCrop,
    grass_hue_shift,
    gaussian_blur,
    gaussian_noise,
    gamma_correction,
    generate_gaussian_heatmap,
)


# annotation name (pitch_constants.PITCH_LINES key) → upstream LINE_CLASSES idx
# Goal-post / crossbar classes (6-11) are intentionally absent: the manual
# annotator does not produce them, so the network never sees a positive
# example. ``MaskedMSELoss`` keeps gradient zero on those channels.
LINE_NAME_TO_UPSTREAM_IDX: dict[str, int] = {
    "touchline_top": 13,
    "touchline_bottom": 16,
    "goal_line_left": 14,
    "goal_line_right": 15,
    "center_line": 12,
    "penalty_left_top": 0,
    "penalty_left_bottom": 2,
    "penalty_left_front": 1,
    "penalty_right_top": 3,
    "penalty_right_bottom": 5,
    "penalty_right_front": 4,
    "goal_area_left_top": 17,
    "goal_area_left_bottom": 19,
    "goal_area_left_front": 18,
    "goal_area_right_top": 20,
    "goal_area_right_bottom": 22,
    "goal_area_right_front": 21,
}


# Horizontal-flip mirror map for line-class indices. Pairs that swap when
# the image is mirrored about x = pitch_center.
LINE_MIRROR_MAP: dict[int, int] = {
    # Big rect (penalty area) left ↔ right
    0: 3, 3: 0,
    1: 4, 4: 1,
    2: 5, 5: 2,
    # Small rect (goal area) left ↔ right
    17: 20, 20: 17,
    18: 21, 21: 18,
    19: 22, 22: 19,
    # Goal posts / crossbar left ↔ right
    6: 9, 9: 6,
    7: 11, 11: 7,
    8: 10, 10: 8,
    # Side lines
    14: 15, 15: 14,
    13: 13,  # top touchline maps to itself under horizontal flip (still top)
    16: 16,
    # Middle line maps to itself
    12: 12,
}


def _line_endpoints_from_annotation(
    annotation: dict,
) -> dict[int, tuple[tuple[float, float], tuple[float, float]]]:
    """Extract {upstream_idx: ((x1,y1), (x2,y2))} in pixel coords.

    Skips annotations whose ``name`` we don't recognise rather than failing
    — the annotator can grow new line names over time.
    """
    out: dict[int, tuple[tuple[float, float], tuple[float, float]]] = {}
    for ln in annotation.get("lines", []):
        name = ln.get("name")
        idx = LINE_NAME_TO_UPSTREAM_IDX.get(name)
        if idx is None:
            continue
        (x1, y1), (x2, y2) = ln["pixels"][0], ln["pixels"][1]
        out[idx] = ((float(x1), float(y1)), (float(x2), float(y2)))
    return out


def generate_line_endpoint_heatmaps(
    num_classes: int,
    lines: dict[int, tuple[tuple[float, float], tuple[float, float]]],
    image_size: tuple[int, int],
    down_ratio: int = 2,
    sigma: float = 2.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Build upstream-format (num_classes + 1, H/dr, W/dr) target + class mask.

    Matches ``generate_gaussian_array_vectorized_l`` in upstream PnLCalib
    ``utils/utils_heatmap.py``:

    - Per-class channel = SUM of the two endpoint Gaussians
      (``hm1 + hm2``, NOT ``np.maximum`` — overlap intentionally exceeds
      1 so the loss latches onto endpoint coincidence).
    - Channel ``num_classes`` ("border") = densely-sampled Gaussians along
      every line, summed across all classes, then clipped to [0, 1].
      This is what ``SV_lines`` pretrained weights were trained against,
      so deviating from it (e.g. using ``1 - sum`` like the keypoint head)
      breaks the pretrain transfer.

    Lines whose endpoints both fall well outside the frame are dropped
    from the mask (treated as "not present in this view").
    """
    width, height = image_size
    nw = width // down_ratio
    nh = height // down_ratio

    heatmaps = np.zeros((num_classes, nh, nw), dtype=np.float64)
    border = np.zeros((nh, nw), dtype=np.float64)
    mask = np.zeros(num_classes, dtype=np.int32)

    for cls_idx, ((x1, y1), (x2, y2)) in lines.items():
        if not (0 <= cls_idx < num_classes):
            continue
        # Scale to heatmap resolution.
        x1s = x1 * nw / width
        y1s = y1 * nh / height
        x2s = x2 * nw / width
        y2s = y2 * nh / height
        # If both endpoints are way outside the frame, skip (line not visible).
        ep_in = lambda x, y: -nw * 0.1 < x < nw * 1.1 and -nh * 0.1 < y < nh * 1.1
        if not (ep_in(x1s, y1s) or ep_in(x2s, y2s)):
            continue
        hm1 = generate_gaussian_heatmap(nw, nh, x1s, y1s, sigma)
        hm2 = generate_gaussian_heatmap(nw, nh, x2s, y2s, sigma)
        heatmaps[cls_idx] = hm1 + hm2
        mask[cls_idx] = 1

        # Sample Gaussians densely along the segment for the border channel.
        pixel_dist = float(np.hypot(x2s - x1s, y2s - y1s))
        n_samples = max(1, int(pixel_dist / sigma))
        if n_samples == 1:
            xc = abs(x2s - x1s) / 2.0
            yc = abs(y2s - y1s) / 2.0
            border += generate_gaussian_heatmap(nw, nh, xc, yc, sigma)
        else:
            for i in range(n_samples):
                alpha = i / (n_samples - 1)
                xs = x1s + alpha * (x2s - x1s)
                ys = y1s + alpha * (y2s - y1s)
                border += generate_gaussian_heatmap(nw, nh, xs, ys, sigma)

    border = np.clip(border, 0, 1)
    full = np.concatenate([heatmaps, border[None, :, :]], axis=0)
    return full, mask


def _crop_lines(
    lines: dict[int, tuple[tuple[float, float], tuple[float, float]]],
    crop_x: float, crop_y: float, crop_w: float, crop_h: float,
    out_w: int, out_h: int,
) -> dict[int, tuple[tuple[float, float], tuple[float, float]]]:
    """Apply the same crop+resize used by ``KeypointRandomZoomCrop`` to lines.

    Endpoints outside the crop are not clipped to the boundary — clipping
    would invent fake endpoints that don't exist on the real line. Instead
    we let endpoints land outside the heatmap and rely on
    ``generate_line_endpoint_heatmaps`` to drop classes whose both
    endpoints are far out of frame.
    """
    out: dict = {}
    for cls_idx, ((x1, y1), (x2, y2)) in lines.items():
        nx1 = (x1 - crop_x) / crop_w * out_w
        ny1 = (y1 - crop_y) / crop_h * out_h
        nx2 = (x2 - crop_x) / crop_w * out_w
        ny2 = (y2 - crop_y) / crop_h * out_h
        out[cls_idx] = ((nx1, ny1), (nx2, ny2))
    return out


class LineAnnotationDataset(Dataset):
    """Loads ``frame_*.json`` (line annotations) + ``frame_*_raw.jpg``.

    Looks for ``frame_<idx>.json`` first; if absent, tries ``<base>.json``
    derived from the matching ``_all_points.json`` so we can reuse the
    same annotation directory layout.
    """

    def __init__(
        self,
        annotations_dir: str | list[str],
        image_size: tuple[int, int] = (960, 540),
        num_classes: int = 23,
        down_ratio: int = 2,
        sigma: float = 2.0,
    ):
        dirs = (
            [Path(d) for d in annotations_dir]
            if isinstance(annotations_dir, (list, tuple))
            else [Path(annotations_dir)]
        )
        self.annotations_dir = dirs[0]  # legacy single-dir attribute
        self.annotations_dirs = dirs
        self.image_size = image_size
        self.num_classes = num_classes
        self.down_ratio = down_ratio
        self.sigma = sigma

        # Glob ``frame_*.json`` (line annotations, NOT the ``*_all_points``
        # variant) across every directory and keep files that actually
        # carry a non-empty ``lines`` field. Multi-dir input is used by
        # the annotate-page "Train this group" button to combine every
        # sibling video's annotations into one training set.
        candidates: list[str] = []
        for d in dirs:
            ds_jsons = sorted(glob.glob(str(d / "frame_*.json")))
            candidates.extend(p for p in ds_jsons if not p.endswith("_all_points.json"))

        self.annotation_files: list[str] = []
        for path in candidates:
            with open(path) as f:
                ann = json.load(f)
            if _line_endpoints_from_annotation(ann):
                self.annotation_files.append(path)

        if not self.annotation_files:
            raise ValueError(
                f"No line annotations found in {[str(d) for d in dirs]}. "
                "Expected frame_<idx>.json with a non-empty 'lines' field."
            )
        if len(dirs) > 1:
            print(
                f"[lines] Found {len(self.annotation_files)} annotation files "
                f"across {len(dirs)} directories"
            )
        else:
            print(f"[lines] Found {len(self.annotation_files)} annotation files")

    def __len__(self) -> int:
        return len(self.annotation_files)

    def _load_image(self, annotation_path: str) -> Image.Image:
        base = annotation_path[:-len(".json")]
        for cand in (base + "_raw.jpg", base + ".jpg"):
            if os.path.exists(cand):
                return Image.open(cand).convert("RGB")
        raise FileNotFoundError(f"Image not found for {annotation_path}")

    def __getitem__(self, idx: int):
        annotation_path = self.annotation_files[idx]
        with open(annotation_path) as f:
            annotation = json.load(f)

        image = self._load_image(annotation_path)
        orig_w, orig_h = image.size
        image_resized = image.resize(self.image_size, Image.BILINEAR)

        lines = _line_endpoints_from_annotation(annotation)
        # Scale endpoints to resized image.
        sx = self.image_size[0] / orig_w
        sy = self.image_size[1] / orig_h
        lines = {
            i: ((x1 * sx, y1 * sy), (x2 * sx, y2 * sy))
            for i, ((x1, y1), (x2, y2)) in lines.items()
        }

        heatmaps, mask = generate_line_endpoint_heatmaps(
            num_classes=self.num_classes,
            lines=lines,
            image_size=self.image_size,
            down_ratio=self.down_ratio,
            sigma=self.sigma,
        )

        image_np = np.array(image_resized).astype(np.float32) / 255.0
        image_tensor = torch.from_numpy(image_np).permute(2, 0, 1)
        return (
            image_tensor,
            torch.from_numpy(heatmaps).float(),
            torch.from_numpy(mask).float(),
        )


class AugmentedLineDataset(LineAnnotationDataset):
    """Apply zoom-crop / color jitter / hflip / noise to line annotations.

    Mirrors ``AugmentedPointDataset``. The zoom-crop uses the line
    endpoints as proxy keypoints when checking visibility (so we don't
    crop into a region that contains zero annotated lines).
    """

    def __init__(
        self,
        annotations_dir: str | list[str],
        augment_factor: int = 10,
        zoom_range: tuple[float, float] = (1.0, 1.5),
        zoom_prob: float = 0.5,
        min_endpoints: int = 4,
        seed: int | None = None,
        **kwargs,
    ):
        super().__init__(annotations_dir, **kwargs)
        self.augment_factor = augment_factor
        self.rng = np.random.default_rng(seed)
        self.zoom_prob = zoom_prob
        self.zoom_range = zoom_range
        self.min_endpoints = min_endpoints

    def __len__(self) -> int:
        return len(self.annotation_files) * self.augment_factor

    def _zoom_crop(self, image_np, lines, width, height):
        """Try to find a crop that keeps at least ``min_endpoints``
        annotated endpoints inside the frame. Falls back to identity."""
        # Flatten endpoints for visibility check.
        eps = []
        for ((x1, y1), (x2, y2)) in lines.values():
            eps.append((x1, y1))
            eps.append((x2, y2))

        for _ in range(50):
            zoom = self.rng.uniform(*self.zoom_range)
            cw = width / zoom
            ch = height / zoom
            cx = self.rng.uniform(0, max(0.001, width - cw))
            cy = self.rng.uniform(0, max(0.001, height - ch))
            inside = sum(
                1 for (x, y) in eps
                if cx <= x < cx + cw and cy <= y < cy + ch
            )
            if inside >= self.min_endpoints:
                px_x = max(0, int(cx))
                px_y = max(0, int(cy))
                px_w = max(1, min(int(cw), width - px_x))
                px_h = max(1, min(int(ch), height - px_y))
                cropped = image_np[px_y:px_y + px_h, px_x:px_x + px_w]
                resized = cv2.resize(cropped, (width, height))
                lines = _crop_lines(
                    lines, cx, cy, cw, ch, width, height,
                )
                return resized, lines
        return image_np, lines

    def _augment(self, image_np, lines, width, height):
        if self.rng.random() < self.zoom_prob:
            image_np, lines = self._zoom_crop(image_np, lines, width, height)

        if self.rng.random() > 0.5:
            factor = self.rng.uniform(0.8, 1.2)
            image_np = np.clip(image_np * factor, 0, 255).astype(np.uint8)

        if self.rng.random() > 0.5:
            mean = image_np.mean()
            factor = self.rng.uniform(0.8, 1.2)
            image_np = np.clip((image_np - mean) * factor + mean, 0, 255).astype(np.uint8)

        if self.rng.random() > 0.5:
            image_np = grass_hue_shift(image_np, self.rng)
        if self.rng.random() > 0.5:
            image_np = gaussian_blur(image_np, self.rng)
        if self.rng.random() > 0.5:
            image_np = gaussian_noise(image_np, self.rng)
        if self.rng.random() > 0.5:
            image_np = gamma_correction(image_np, self.rng)

        # Horizontal flip — must swap class indices via mirror map.
        if self.rng.random() > 0.5:
            image_np = np.fliplr(image_np).copy()
            new_lines: dict = {}
            for cls_idx, ((x1, y1), (x2, y2)) in lines.items():
                mirror = LINE_MIRROR_MAP.get(cls_idx, cls_idx)
                new_lines[mirror] = (
                    (width - x1, y1),
                    (width - x2, y2),
                )
            lines = new_lines

        # Small endpoint noise.
        if self.rng.random() > 0.5:
            noisy: dict = {}
            for cls_idx, ((x1, y1), (x2, y2)) in lines.items():
                noisy[cls_idx] = (
                    (x1 + self.rng.normal(0, 2.0), y1 + self.rng.normal(0, 2.0)),
                    (x2 + self.rng.normal(0, 2.0), y2 + self.rng.normal(0, 2.0)),
                )
            lines = noisy

        return image_np, lines

    def __getitem__(self, idx: int):
        orig_idx = idx % len(self.annotation_files)
        aug_idx = idx // len(self.annotation_files)

        annotation_path = self.annotation_files[orig_idx]
        with open(annotation_path) as f:
            annotation = json.load(f)

        image = self._load_image(annotation_path)
        orig_w, orig_h = image.size
        image_np = np.array(image)

        lines = _line_endpoints_from_annotation(annotation)

        if aug_idx > 0:
            image_np, lines = self._augment(image_np, lines, orig_w, orig_h)

        image_resized = cv2.resize(image_np, self.image_size)
        sx = self.image_size[0] / orig_w
        sy = self.image_size[1] / orig_h
        lines = {
            i: ((x1 * sx, y1 * sy), (x2 * sx, y2 * sy))
            for i, ((x1, y1), (x2, y2)) in lines.items()
        }

        heatmaps, mask = generate_line_endpoint_heatmaps(
            num_classes=self.num_classes,
            lines=lines,
            image_size=self.image_size,
            down_ratio=self.down_ratio,
            sigma=self.sigma,
        )

        image_tensor = torch.from_numpy(
            image_resized.astype(np.float32) / 255.0
        ).permute(2, 0, 1)
        return (
            image_tensor,
            torch.from_numpy(heatmaps).float(),
            torch.from_numpy(mask).float(),
        )


class CachedAugmentedLineDataset(Dataset):
    """Pre-cached version of ``AugmentedLineDataset`` for stable validation."""

    def __init__(self, **kwargs):
        # Reuse the augmented dataset with a fixed seed, then materialise.
        seed = kwargs.pop("seed", 42)
        ds = AugmentedLineDataset(seed=seed, **kwargs)
        self.samples = [ds[i] for i in range(len(ds))]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        return self.samples[idx]
