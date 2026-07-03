"""DataLoader for point annotations, directly generating heatmaps.

This module provides a simplified dataloader that:
1. Loads point annotations from JSON files
2. Converts HRNet indices to PnLCalib indices via world coordinate matching
3. Generates Gaussian heatmaps directly from point coordinates
4. Creates masks to indicate which keypoints are visible
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

from .index_mapper import HRNetToPnLCalibMapper

# Keypoint ID mirror mapping for horizontal flip (1-indexed PnLCalib IDs)
# Based on pitch symmetry about x=52.5m center line
KEYPOINT_MIRROR_MAP = {
    # Corner flags
    1: 3, 3: 1,      # 0↔2 in 0-indexed
    # Bottom penalty area
    4: 7, 7: 4,      # 3↔6
    5: 6, 6: 5,      # 4↔5
    # Bottom goal area
    8: 11, 11: 8,    # 7↔10
    9: 10, 10: 9,    # 8↔9
    # Goal posts (bottom)
    12: 14, 14: 12,  # 11↔13
    13: 15, 15: 13,  # 12↔14
    # Goal posts (top)
    16: 18, 18: 16,  # 15↔17
    17: 19, 19: 17,  # 16↔18
    # Top goal area
    20: 23, 23: 20,  # 19↔22
    21: 22, 22: 21,  # 20↔21
    # Top penalty area
    24: 27, 27: 24,  # 23↔26
    25: 26, 26: 25,  # 24↔25
    # Top corner flags
    28: 30, 30: 28,  # 27↔29
    # Penalty arcs
    31: 33, 33: 31,  # 30↔32
    34: 36, 36: 34,  # 33↔35
    # Center circle outer
    37: 40, 40: 37,  # 36↔39
    38: 39, 39: 38,  # 37↔38
    41: 44, 44: 41,  # 40↔43
    42: 43, 43: 42,  # 41↔42
    # Penalty spots and arcs
    45: 57, 57: 45,  # 44↔56
    46: 56, 56: 46,  # 45↔55
    47: 55, 55: 47,  # 46↔54
    # Center circle inner
    48: 49, 49: 48,  # 47↔48
    50: 52, 52: 50,  # 49↔51
    53: 54, 54: 53,  # 52↔53
    # Center line points (map to themselves)
    2: 2,    # 1 - center bottom
    29: 29,  # 28 - center top
    32: 32,  # 31 - center circle bottom
    35: 35,  # 34 - center circle top
    51: 51,  # 50 - center spot
}


class KeypointRandomZoomCrop:
    """Simulate camera zoom by cropping and resizing.

    Crops a random region of the image and resizes back to original dimensions.
    Ensures at least min_keypoints remain visible in the cropped region.
    Adapted from sportlight's implementation for our pixel coordinate format.
    """

    def __init__(
        self,
        zoom_range: tuple[float, float] = (1.0, 1.5),
        min_keypoints: int = 4,
        max_attempts: int = 50,
        center_crop: bool = False,
        rng: np.random.Generator | None = None,
    ):
        """Initialize zoom crop augmentation.

        Args:
            zoom_range: (min_zoom, max_zoom) - zoom factor range.
                1.0 = no zoom, 2.0 = 2x zoom (crop half the image).
            min_keypoints: Minimum keypoints that must remain visible.
            max_attempts: Maximum attempts to find valid crop region.
            center_crop: If True, always crop from center instead of random.
            rng: Random number generator for reproducibility.
        """
        self.zoom_range = zoom_range
        self.min_keypoints = min_keypoints
        self.max_attempts = max_attempts
        self.center_crop = center_crop
        self.rng = rng or np.random.default_rng()

    def _get_valid_keypoints(
        self, keypoints: dict[int, dict]
    ) -> list[tuple[int, float, float]]:
        """Extract visible keypoint positions."""
        valid_kpts = []
        for kp_id, kp in keypoints.items():
            if kp.get('in_frame', False):
                valid_kpts.append((kp_id, kp['x'], kp['y']))
        return valid_kpts

    def _keypoints_in_crop(
        self,
        valid_kpts: list[tuple[int, float, float]],
        crop_x: float,
        crop_y: float,
        crop_w: float,
        crop_h: float,
    ) -> list[tuple[int, float, float]]:
        """Return keypoints that fall within the crop region (pixel coords)."""
        inside = []
        for kp_id, x, y in valid_kpts:
            if crop_x <= x < crop_x + crop_w and crop_y <= y < crop_y + crop_h:
                inside.append((kp_id, x, y))
        return inside

    def __call__(
        self,
        image: np.ndarray,
        keypoints: dict[int, dict],
        width: int,
        height: int,
    ) -> tuple[np.ndarray, dict[int, dict]]:
        """Apply zoom crop augmentation.

        Args:
            image: Input image (H, W, C).
            keypoints: Dict mapping kp_id to {'x', 'y', 'in_frame'} in pixel coords.
            width: Original image width.
            height: Original image height.

        Returns:
            Tuple of (cropped_resized_image, updated_keypoints).
        """
        # Get valid keypoints
        valid_kpts = self._get_valid_keypoints(keypoints)

        if len(valid_kpts) < self.min_keypoints:
            # Not enough keypoints, skip augmentation
            return image, keypoints

        # Try to find a valid crop region
        for _ in range(self.max_attempts):
            # Random zoom factor
            zoom = self.rng.uniform(self.zoom_range[0], self.zoom_range[1])

            # Crop size in pixels
            crop_w = width / zoom
            crop_h = height / zoom

            # Crop position
            max_x = width - crop_w
            max_y = height - crop_h

            if self.center_crop:
                crop_x = max_x / 2
                crop_y = max_y / 2
            else:
                crop_x = self.rng.uniform(0, max(0.001, max_x))
                crop_y = self.rng.uniform(0, max(0.001, max_y))

            # Check how many keypoints fall in crop region
            kpts_in_crop = self._keypoints_in_crop(
                valid_kpts, crop_x, crop_y, crop_w, crop_h
            )

            if len(kpts_in_crop) >= self.min_keypoints:
                # Valid crop found - apply it
                px_x = int(crop_x)
                px_y = int(crop_y)
                px_w = int(crop_w)
                px_h = int(crop_h)

                # Ensure valid dimensions
                px_w = max(1, min(px_w, width - px_x))
                px_h = max(1, min(px_h, height - px_y))

                # Crop image
                cropped = image[px_y:px_y + px_h, px_x:px_x + px_w]

                # Resize back to original dimensions
                resized = cv2.resize(cropped, (width, height))

                # Update keypoint coordinates
                new_keypoints = {}
                for kp_id, kp in keypoints.items():
                    old_x, old_y = kp['x'], kp['y']

                    # Check if in crop
                    if (crop_x <= old_x < crop_x + crop_w and
                            crop_y <= old_y < crop_y + crop_h):
                        # Transform to new coords: (old - crop_origin) * scale
                        new_x = (old_x - crop_x) / crop_w * width
                        new_y = (old_y - crop_y) / crop_h * height
                        new_keypoints[kp_id] = {
                            'x': new_x,
                            'y': new_y,
                            'in_frame': True,
                        }
                    else:
                        # Keypoint outside crop
                        new_keypoints[kp_id] = {
                            'x': old_x,
                            'y': old_y,
                            'in_frame': False,
                        }

                return resized, new_keypoints

        # No valid crop found, return original
        return image, keypoints


def grass_hue_shift(image: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Shift hue of grass (green) regions to simulate different field colors."""
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV).astype(np.float32)
    # Green hue range in OpenCV HSV (0-179 scale)
    green_mask = (hsv[:, :, 0] >= 35) & (hsv[:, :, 0] <= 85)
    hue_shift = rng.uniform(-15, 15)
    hsv[:, :, 0] = np.where(green_mask, hsv[:, :, 0] + hue_shift, hsv[:, :, 0])
    hsv[:, :, 0] = np.clip(hsv[:, :, 0], 0, 179)
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)


def gaussian_blur(image: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Apply random Gaussian blur to simulate focus variation."""
    ksize = int(rng.choice([3, 5, 7]))
    return cv2.GaussianBlur(image, (ksize, ksize), 0)


def gaussian_noise(image: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Add Gaussian noise to simulate camera noise or compression artifacts."""
    noise = rng.normal(0, 10, image.shape).astype(np.float32)
    noisy = image.astype(np.float32) + noise
    return np.clip(noisy, 0, 255).astype(np.uint8)


def gamma_correction(image: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Apply random gamma correction to simulate different exposure/lighting."""
    gamma = rng.uniform(0.7, 1.5)
    inv_gamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** inv_gamma) * 255
                      for i in range(256)]).astype(np.uint8)
    return cv2.LUT(image, table)


def generate_gaussian_heatmap(
    h: int,
    w: int,
    px: float,
    py: float,
    sigma: float = 2.0,
) -> np.ndarray:
    """Generate a single Gaussian heatmap.

    Matches PnLCalib's generate_gaussian_matrix_vectorized convention:
    - h, w are the output dimensions (h=width, w=height in PnLCalib convention)
    - px, py are the center coordinates
    - Output shape is (w, h) = (height, width) for (H, W) format

    Args:
        h: First dimension of output (corresponds to x-axis / width).
        w: Second dimension of output (corresponds to y-axis / height).
        px: X coordinate of Gaussian center.
        py: Y coordinate of Gaussian center.
        sigma: Gaussian standard deviation.

    Returns:
        2D numpy array of shape (w, h) with Gaussian centered at (px, py).
    """
    x, y = np.meshgrid(np.arange(h), np.arange(w))
    array = (-((x - px) ** 2 + (y - py) ** 2) / (2 * sigma ** 2)).astype(float)
    return np.exp(array)


def generate_heatmaps_from_keypoints(
    num_keypoints: int,
    keypoints: dict[int, dict],
    image_size: tuple[int, int],
    down_ratio: int = 2,
    sigma: float = 2.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate heatmap tensor and mask from keypoint dict.

    Args:
        num_keypoints: Number of keypoint channels (57 for PnLCalib).
        keypoints: Dict mapping keypoint ID (1-indexed) to {'x', 'y', 'in_frame'}.
        image_size: (width, height) of the original image.
        down_ratio: Downsampling ratio for heatmaps.
        sigma: Gaussian standard deviation.

    Returns:
        Tuple of (heatmaps, mask):
        - heatmaps: (num_keypoints + 1, H/down_ratio, W/down_ratio) array
        - mask: (num_keypoints,) binary array indicating visible keypoints
    """
    width, height = image_size
    new_width = width // down_ratio
    new_height = height // down_ratio

    # Scale keypoint coordinates
    scaled_keypoints = {}
    for kp_id, kp_info in keypoints.items():
        scaled_keypoints[kp_id] = {
            'x': kp_info['x'] * new_width / width,
            'y': kp_info['y'] * new_height / height,
            'in_frame': kp_info['in_frame'],
        }

    # Generate heatmaps for each keypoint channel
    # Following PnLCalib convention: pass (width, height) to get output shape (height, width)
    heatmaps = []
    for kp_id in range(1, num_keypoints + 1):
        if kp_id in scaled_keypoints and scaled_keypoints[kp_id]['in_frame']:
            x = scaled_keypoints[kp_id]['x']
            y = scaled_keypoints[kp_id]['y']
            heatmap = generate_gaussian_heatmap(new_width, new_height, x, y, sigma)
        else:
            # Empty heatmap for invisible keypoints, shape (H, W)
            heatmap = np.zeros((new_height, new_width))
        heatmaps.append(heatmap)

    heatmaps = np.array(heatmaps)

    # Add background channel (1 - sum of all keypoint channels)
    background = 1 - heatmaps.sum(axis=0, keepdims=True)
    heatmaps = np.concatenate([heatmaps, background], axis=0)
    heatmaps = np.clip(heatmaps, 0, 1)

    # Create mask: 1 for visible keypoints, 0 for invisible
    mask = np.zeros(num_keypoints, dtype=np.int32)
    for kp_id in keypoints:
        if 1 <= kp_id <= num_keypoints and keypoints[kp_id]['in_frame']:
            mask[kp_id - 1] = 1

    return heatmaps, mask


def compute_negative_channels(
    annotation_files: list[str], num_keypoints: int,
) -> set[int]:
    """1-indexed channels never annotated anywhere in the dataset.

    The model has ``num_keypoints`` output channels but a pitch only uses
    a subset, and the user annotates only some of those. Channels that
    appear in NO annotation across the whole training set are supervised
    with an all-zero target (see ``apply_negative_channels``) so the model
    learns a low response there instead of firing per its FIFA pretraining.
    Dataset-level, NOT per-frame: a channel annotated somewhere keeps its
    normal per-frame mask (partial/incomplete labelling isn't punished).
    """
    annotated: set[int] = set()
    for f in annotation_files:
        try:
            with open(f) as fh:
                data = json.load(fh)
        except (OSError, json.JSONDecodeError):
            continue
        for p in data.get("all_points", []):
            pnl = p.get("pnlcalib_id")
            # Stored 0-indexed; heatmap channel is pnl + 1 (matches
            # convert_annotation's ``pnlcalib_id + 1``).
            if pnl is not None and pnl >= 0:
                annotated.add(int(pnl) + 1)
    return {c for c in range(1, num_keypoints + 1) if c not in annotated}


def apply_negative_channels(
    mask: np.ndarray, negative_channels: set[int] | None,
) -> np.ndarray:
    """Set mask=1 on never-annotated channels (their target is all-zero).

    Turns them into negative samples so the model learns a low response.
    No-op when ``negative_channels`` is empty/None.
    """
    if not negative_channels:
        return mask
    for c in negative_channels:  # 1-indexed; mask index is c-1
        if 1 <= c <= len(mask):
            mask[c - 1] = 1
    return mask


class PointAnnotationDataset(Dataset):
    """Dataset for point annotations with direct heatmap generation.

    This dataset loads annotation files (*_all_points.json) and their
    corresponding images (*_raw.jpg), converts keypoint indices, and
    generates heatmaps for training.
    """

    def __init__(
        self,
        annotations_dir: str | list[str],
        transform: Callable | None = None,
        image_size: tuple[int, int] = (960, 540),
        num_keypoints: int = 57,
        down_ratio: int = 2,
        sigma: float = 2.0,
        tolerance: float = 0.5,
    ):
        """Initialize the dataset.

        Args:
            annotations_dir: Directory containing *_all_points.json and *_raw.jpg files.
                Can also be a list of directories — every file from every dir
                is concatenated into one training set. Used by the
                ``annotate``-page "Train this group" button so finetune sees
                annotations from every video sharing a name prefix.
            transform: Optional transform to apply to images.
            image_size: Target image size (width, height) for resizing.
            num_keypoints: Number of keypoint channels (57 for PnLCalib).
            down_ratio: Downsampling ratio for heatmaps.
            sigma: Gaussian standard deviation for heatmaps.
            tolerance: World coordinate matching tolerance (meters).
        """
        dirs = (
            [Path(d) for d in annotations_dir]
            if isinstance(annotations_dir, (list, tuple))
            else [Path(annotations_dir)]
        )
        self.annotations_dir = dirs[0]  # legacy single-dir attribute
        self.annotations_dirs = dirs
        self.transform = transform
        self.image_size = image_size
        self.num_keypoints = num_keypoints
        self.down_ratio = down_ratio
        self.sigma = sigma

        # Initialize mapper
        self.mapper = HRNetToPnLCalibMapper(tolerance=tolerance)

        # Find all annotation files across every directory.
        self.annotation_files: list[str] = []
        for d in dirs:
            self.annotation_files.extend(
                sorted(glob.glob(str(d / "*_all_points.json")))
            )

        if not self.annotation_files:
            raise ValueError(
                f"No annotation files found in {[str(d) for d in dirs]}"
            )

        if len(dirs) > 1:
            print(
                f"Found {len(self.annotation_files)} annotation files "
                f"across {len(dirs)} directories"
            )
        else:
            print(f"Found {len(self.annotation_files)} annotation files")

        # Dataset-level negative channels. The model has ``num_keypoints``
        # output channels, but a given pitch only uses a subset, and the
        # user only annotates some of those. Channels that appear in NO
        # annotation across the whole training set are "never used": we
        # supervise them with an all-zero target (mask=1, target already
        # zero) so the model learns to output a low response there instead
        # of leaving the FIFA-pretrained weights to fire spuriously.
        #
        # This is dataset-level, NOT per-frame: a channel the user DID
        # annotate somewhere is left to the normal per-frame mask (so
        # frames where they merely didn't label it stay ignored, mask=0 —
        # respecting incomplete/partial annotation without punishing it).
        self._negative_channels = compute_negative_channels(
            self.annotation_files, self.num_keypoints,
        )
        if self._negative_channels:
            print(
                f"Negative-supervised channels (never annotated, forced "
                f"low-response): {len(self._negative_channels)} of "
                f"{self.num_keypoints}"
            )

    def __len__(self) -> int:
        return len(self.annotation_files)

    def _load_image(self, annotation_path: str) -> Image.Image:
        """Load the corresponding image for an annotation."""
        # Replace _all_points.json with _raw.jpg
        base = annotation_path.replace("_all_points.json", "")
        image_path = base + "_raw.jpg"

        if not os.path.exists(image_path):
            # Fallback: try just .jpg
            image_path = base + ".jpg"

        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found for {annotation_path}")

        image = Image.open(image_path).convert("RGB")
        return image

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Get a training sample.

        Returns:
            Tuple of (image, heatmaps, mask):
            - image: (3, H, W) tensor
            - heatmaps: (num_keypoints + 1, H/down_ratio, W/down_ratio) tensor
            - mask: (num_keypoints,) tensor
        """
        annotation_path = self.annotation_files[idx]

        # Load annotation
        with open(annotation_path, 'r') as f:
            annotation = json.load(f)

        # Load image
        image = self._load_image(annotation_path)
        orig_width, orig_height = image.size

        # Resize image
        image = image.resize(self.image_size, Image.BILINEAR)

        # Convert keypoints
        keypoints = self.mapper.convert_annotation(
            annotation,
            image_width=orig_width,
            image_height=orig_height,
        )

        # Scale keypoint coordinates to resized image
        scale_x = self.image_size[0] / orig_width
        scale_y = self.image_size[1] / orig_height
        for kp_id in keypoints:
            keypoints[kp_id]['x'] *= scale_x
            keypoints[kp_id]['y'] *= scale_y

        # Generate heatmaps
        heatmaps, mask = generate_heatmaps_from_keypoints(
            num_keypoints=self.num_keypoints,
            keypoints=keypoints,
            image_size=self.image_size,
            down_ratio=self.down_ratio,
            sigma=self.sigma,
        )

        # Force negative supervision on never-annotated channels: mask=1
        # with an all-zero target (heatmap channel is already zero for
        # any channel not in ``keypoints``) teaches the model to output a
        # low response there, instead of firing per its FIFA pretraining.
        mask = self._apply_negative_channels(mask)

        # Convert image to tensor
        image_np = np.array(image).astype(np.float32) / 255.0
        image_tensor = torch.from_numpy(image_np).permute(2, 0, 1)  # (H, W, C) -> (C, H, W)

        # Apply transforms if any
        if self.transform:
            sample = self.transform({
                'image': image_tensor,
                'target': heatmaps,
                'mask': mask,
            })
            image_tensor = sample['image']
            heatmaps = sample['target']
            mask = sample['mask']

        return (
            image_tensor,
            torch.from_numpy(heatmaps).float(),
            torch.from_numpy(mask).float(),
        )

    def _apply_negative_channels(self, mask: np.ndarray) -> np.ndarray:
        """Instance wrapper around ``apply_negative_channels``."""
        return apply_negative_channels(
            mask, getattr(self, "_negative_channels", None),
        )

    def get_frame_info(self, idx: int) -> dict:
        """Get metadata for a frame."""
        annotation_path = self.annotation_files[idx]
        with open(annotation_path, 'r') as f:
            annotation = json.load(f)
        return {
            'path': annotation_path,
            'frame_idx': annotation.get('frame_idx'),
            'total_points': annotation.get('total_points'),
            'reprojection_error': annotation.get('reprojection_error'),
        }


class AugmentedPointDataset(PointAnnotationDataset):
    """Dataset with data augmentation for small datasets.

    Applies random augmentations to increase effective dataset size:
    - Zoom crop (simulates camera zoom)
    - Horizontal flip
    - Color jitter
    - Random keypoint position noise
    """

    def __init__(
        self,
        annotations_dir: str | list[str],
        augment_factor: int = 10,
        zoom_range: tuple[float, float] = (1.0, 1.5),
        zoom_prob: float = 0.5,
        min_keypoints: int = 4,
        hflip_prob: float = 0.5,
        **kwargs,
    ):
        """Initialize with augmentation.

        Args:
            annotations_dir: Directory containing annotations.
            augment_factor: Number of augmented versions per original sample.
            zoom_range: (min_zoom, max_zoom) for zoom crop augmentation.
            zoom_prob: Probability of applying zoom crop.
            min_keypoints: Minimum keypoints required after zoom crop.
            hflip_prob: Probability of horizontal flip + mirror-id swap. Set
                to 0 when the dataset comes from a fixed-side camera and
                hflipped images don't reflect any real inference-time view —
                otherwise the mirror-id swap synthesises positives for the
                opposite-half keypoint channels and the model double-fires
                left/right pairs at inference (e.g. id 15 + id 17 firing on
                the same goalpost).
            **kwargs: Additional arguments for PointAnnotationDataset.
        """
        super().__init__(annotations_dir, **kwargs)
        self.augment_factor = augment_factor
        self.rng = np.random.default_rng()

        # Zoom augmentation
        self.zoom_prob = zoom_prob
        self.hflip_prob = hflip_prob
        self.zoom_crop = KeypointRandomZoomCrop(
            zoom_range=zoom_range,
            min_keypoints=min_keypoints,
            rng=self.rng,
        )

    def __len__(self) -> int:
        return len(self.annotation_files) * self.augment_factor

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Get an augmented training sample."""
        # Map augmented index to original index
        orig_idx = idx % len(self.annotation_files)
        aug_idx = idx // len(self.annotation_files)

        annotation_path = self.annotation_files[orig_idx]

        # Load annotation
        with open(annotation_path, 'r') as f:
            annotation = json.load(f)

        # Load image
        image = self._load_image(annotation_path)
        orig_width, orig_height = image.size

        # Convert to numpy for augmentation
        image_np = np.array(image)

        # Convert keypoints
        keypoints = self.mapper.convert_annotation(
            annotation,
            image_width=orig_width,
            image_height=orig_height,
        )

        # Apply augmentations (except for aug_idx == 0, keep original)
        if aug_idx > 0:
            image_np, keypoints = self._apply_augmentation(
                image_np, keypoints, orig_width, orig_height
            )

        # Resize image
        image_resized = cv2.resize(image_np, self.image_size)

        # Scale keypoint coordinates
        scale_x = self.image_size[0] / orig_width
        scale_y = self.image_size[1] / orig_height
        for kp_id in keypoints:
            keypoints[kp_id]['x'] *= scale_x
            keypoints[kp_id]['y'] *= scale_y

        # Generate heatmaps
        heatmaps, mask = generate_heatmaps_from_keypoints(
            num_keypoints=self.num_keypoints,
            keypoints=keypoints,
            image_size=self.image_size,
            down_ratio=self.down_ratio,
            sigma=self.sigma,
        )
        mask = self._apply_negative_channels(mask)

        # Convert to tensor
        image_tensor = torch.from_numpy(
            image_resized.astype(np.float32) / 255.0
        ).permute(2, 0, 1)

        return (
            image_tensor,
            torch.from_numpy(heatmaps).float(),
            torch.from_numpy(mask).float(),
        )

    def _apply_augmentation(
        self,
        image: np.ndarray,
        keypoints: dict[int, dict],
        width: int,
        height: int,
    ) -> tuple[np.ndarray, dict[int, dict]]:
        """Apply random augmentation to image and keypoints."""
        # Zoom crop (first, before other augmentations)
        if self.rng.random() < self.zoom_prob:
            image, keypoints = self.zoom_crop(image, keypoints, width, height)

        # Color jitter (doesn't affect keypoints)
        if self.rng.random() > 0.5:
            # Brightness
            factor = self.rng.uniform(0.8, 1.2)
            image = np.clip(image * factor, 0, 255).astype(np.uint8)

        if self.rng.random() > 0.5:
            # Contrast
            mean = image.mean()
            factor = self.rng.uniform(0.8, 1.2)
            image = np.clip((image - mean) * factor + mean, 0, 255).astype(np.uint8)

        # Grass hue shift
        if self.rng.random() > 0.5:
            image = grass_hue_shift(image, self.rng)

        # Gaussian blur
        if self.rng.random() > 0.5:
            image = gaussian_blur(image, self.rng)

        # Gaussian noise
        if self.rng.random() > 0.5:
            image = gaussian_noise(image, self.rng)

        # Gamma correction
        if self.rng.random() > 0.5:
            image = gamma_correction(image, self.rng)

        # Horizontal flip with keypoint ID swapping. See __init__ for why
        # this is gated rather than always-on.
        if self.hflip_prob > 0 and self.rng.random() < self.hflip_prob:
            image = np.fliplr(image).copy()
            new_keypoints = {}
            for kp_id, kp in keypoints.items():
                # Flip x coordinate
                new_x = width - kp['x']
                # Check if still in frame
                in_frame = kp.get('in_frame', False) and 0 <= new_x < width
                # Swap keypoint ID to its mirror counterpart
                mirror_id = KEYPOINT_MIRROR_MAP.get(kp_id, kp_id)
                new_keypoints[mirror_id] = {
                    'x': new_x,
                    'y': kp['y'],
                    'in_frame': in_frame,
                }
            keypoints = new_keypoints

        # Add small random noise to keypoint positions
        if self.rng.random() > 0.5:
            noise_std = 2.0  # pixels
            for kp_id in keypoints:
                keypoints[kp_id]['x'] += self.rng.normal(0, noise_std)
                keypoints[kp_id]['y'] += self.rng.normal(0, noise_std)
                # Clamp to image bounds
                keypoints[kp_id]['x'] = np.clip(keypoints[kp_id]['x'], 0, width - 1)
                keypoints[kp_id]['y'] = np.clip(keypoints[kp_id]['y'], 0, height - 1)

        return image, keypoints


class CachedAugmentedDataset(Dataset):
    """Dataset that pre-generates and caches all augmented samples.

    Use this for validation to ensure consistent data across epochs.
    Generates augmented samples once during initialization and caches them.
    """

    def __init__(
        self,
        annotations_dir: str | list[str],
        augment_factor: int = 10,
        zoom_range: tuple[float, float] = (1.0, 1.5),
        zoom_prob: float = 0.5,
        min_keypoints: int = 4,
        image_size: tuple[int, int] = (960, 540),
        num_keypoints: int = 57,
        down_ratio: int = 2,
        sigma: float = 2.0,
        tolerance: float = 0.5,
        seed: int = 42,
        hflip_prob: float = 0.5,
    ):
        """Initialize and pre-generate all augmented samples.

        Args:
            annotations_dir: Directory containing annotations.
            augment_factor: Number of augmented versions per original sample.
            zoom_range: (min_zoom, max_zoom) for zoom crop augmentation.
            zoom_prob: Probability of applying zoom crop.
            min_keypoints: Minimum keypoints required after zoom crop.
            image_size: Target image size (width, height).
            num_keypoints: Number of keypoint channels.
            down_ratio: Downsampling ratio for heatmaps.
            sigma: Gaussian standard deviation for heatmaps.
            tolerance: World coordinate matching tolerance.
            seed: Random seed for reproducibility.
            hflip_prob: Probability of horizontal flip (see
                ``AugmentedPointDataset.__init__`` for why this is gated).
        """
        self.image_size = image_size
        self.num_keypoints = num_keypoints
        self.down_ratio = down_ratio
        self.sigma = sigma

        # Initialize mapper
        self.mapper = HRNetToPnLCalibMapper(tolerance=tolerance)

        # Fixed random generator for reproducibility
        self.rng = np.random.default_rng(seed)

        # Zoom augmentation
        self.zoom_prob = zoom_prob
        self.hflip_prob = hflip_prob
        self.zoom_crop = KeypointRandomZoomCrop(
            zoom_range=zoom_range,
            min_keypoints=min_keypoints,
            rng=self.rng,
        )

        # Find annotation files across one or more directories.
        dirs = (
            [Path(d) for d in annotations_dir]
            if isinstance(annotations_dir, (list, tuple))
            else [Path(annotations_dir)]
        )
        annotation_files: list[str] = []
        for d in dirs:
            annotation_files.extend(
                sorted(glob.glob(str(d / "*_all_points.json")))
            )

        if not annotation_files:
            raise ValueError(
                f"No annotation files found in {[str(d) for d in dirs]}"
            )

        # Same dataset-level negative channels as the training set, so
        # validation loss is measured on the same supervision signal.
        self._negative_channels = compute_negative_channels(
            annotation_files, num_keypoints,
        )

        print(f"Pre-generating {len(annotation_files) * augment_factor} cached validation samples...")

        # Pre-generate all samples
        self.cached_samples = []
        for annot_path in annotation_files:
            # Load annotation
            with open(annot_path, 'r') as f:
                annotation = json.load(f)

            # Load image
            base = annot_path.replace("_all_points.json", "")
            image_path = base + "_raw.jpg"
            if not os.path.exists(image_path):
                image_path = base + ".jpg"
            if not os.path.exists(image_path):
                print(f"  Warning: Image not found for {annot_path}, skipping")
                continue

            image = Image.open(image_path).convert("RGB")
            orig_width, orig_height = image.size
            image_np = np.array(image)

            # Convert keypoints once
            base_keypoints = self.mapper.convert_annotation(
                annotation,
                image_width=orig_width,
                image_height=orig_height,
            )

            # Generate augmented versions
            for aug_idx in range(augment_factor):
                # Deep copy keypoints
                keypoints = {
                    kp_id: {
                        'x': kp['x'],
                        'y': kp['y'],
                        'in_frame': kp['in_frame'],
                    }
                    for kp_id, kp in base_keypoints.items()
                }

                img = image_np.copy()

                # Apply augmentation (except first sample)
                if aug_idx > 0:
                    img, keypoints = self._apply_augmentation(
                        img, keypoints, orig_width, orig_height
                    )

                # Resize image
                img_resized = cv2.resize(img, self.image_size)

                # Scale keypoint coordinates
                scale_x = self.image_size[0] / orig_width
                scale_y = self.image_size[1] / orig_height
                for kp_id in keypoints:
                    keypoints[kp_id]['x'] *= scale_x
                    keypoints[kp_id]['y'] *= scale_y

                # Generate heatmaps
                heatmaps, mask = generate_heatmaps_from_keypoints(
                    num_keypoints=self.num_keypoints,
                    keypoints=keypoints,
                    image_size=self.image_size,
                    down_ratio=self.down_ratio,
                    sigma=self.sigma,
                )
                mask = apply_negative_channels(mask, self._negative_channels)

                # Convert to tensors
                image_tensor = torch.from_numpy(
                    img_resized.astype(np.float32) / 255.0
                ).permute(2, 0, 1)
                heatmaps_tensor = torch.from_numpy(heatmaps).float()
                mask_tensor = torch.from_numpy(mask).float()

                self.cached_samples.append((image_tensor, heatmaps_tensor, mask_tensor))

        print(f"  Cached {len(self.cached_samples)} validation samples")

    def __len__(self) -> int:
        return len(self.cached_samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return pre-cached sample."""
        return self.cached_samples[idx]

    def _apply_augmentation(
        self,
        image: np.ndarray,
        keypoints: dict[int, dict],
        width: int,
        height: int,
    ) -> tuple[np.ndarray, dict[int, dict]]:
        """Apply random augmentation to image and keypoints."""
        # Zoom crop
        if self.rng.random() < self.zoom_prob:
            image, keypoints = self.zoom_crop(image, keypoints, width, height)

        # Color jitter
        if self.rng.random() > 0.5:
            factor = self.rng.uniform(0.8, 1.2)
            image = np.clip(image * factor, 0, 255).astype(np.uint8)

        if self.rng.random() > 0.5:
            mean = image.mean()
            factor = self.rng.uniform(0.8, 1.2)
            image = np.clip((image - mean) * factor + mean, 0, 255).astype(np.uint8)

        # Grass hue shift
        if self.rng.random() > 0.5:
            image = grass_hue_shift(image, self.rng)

        # Gaussian blur
        if self.rng.random() > 0.5:
            image = gaussian_blur(image, self.rng)

        # Gaussian noise
        if self.rng.random() > 0.5:
            image = gaussian_noise(image, self.rng)

        # Gamma correction
        if self.rng.random() > 0.5:
            image = gamma_correction(image, self.rng)

        # Horizontal flip with keypoint ID swapping. See __init__ for why
        # this is gated rather than always-on.
        if self.hflip_prob > 0 and self.rng.random() < self.hflip_prob:
            image = np.fliplr(image).copy()
            new_keypoints = {}
            for kp_id, kp in keypoints.items():
                new_x = width - kp['x']
                in_frame = kp.get('in_frame', False) and 0 <= new_x < width
                # Swap keypoint ID to its mirror counterpart
                mirror_id = KEYPOINT_MIRROR_MAP.get(kp_id, kp_id)
                new_keypoints[mirror_id] = {
                    'x': new_x,
                    'y': kp['y'],
                    'in_frame': in_frame,
                }
            keypoints = new_keypoints

        # Position noise
        if self.rng.random() > 0.5:
            noise_std = 2.0
            for kp_id in keypoints:
                keypoints[kp_id]['x'] += self.rng.normal(0, noise_std)
                keypoints[kp_id]['y'] += self.rng.normal(0, noise_std)
                keypoints[kp_id]['x'] = np.clip(keypoints[kp_id]['x'], 0, width - 1)
                keypoints[kp_id]['y'] = np.clip(keypoints[kp_id]['y'], 0, height - 1)

        return image, keypoints
