"""NBJW Calibration backend using HRNet for keypoint detection.

Implements BaseCalibrator interface.
Based on sn-gamestate's NBJW_Calib implementation.
"""

import os
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image

from ...interfaces import BaseCalibrator

# Add nbjw_calib plugin path (for running with sn-gamestate plugins)
NBJW_PLUGIN_PATH = Path(__file__).parent.parent.parent.parent.parent / "sn-gamestate" / "plugins" / "calibration" / "nbjw_calib"
if str(NBJW_PLUGIN_PATH) not in sys.path:
    sys.path.insert(0, str(NBJW_PLUGIN_PATH))

# Weights URLs from Zenodo
WEIGHTS_URL_KP = "https://zenodo.org/records/12626395/files/SV_kp?download=1"
WEIGHTS_URL_LINES = "https://zenodo.org/records/12626395/files/SV_lines?download=1"

# Keypoint world coordinates (centered on pitch)
KEYPOINT_WORLD_COORDS_2D = [
    [0., 0.], [52.5, 0.], [105., 0.], [0., 13.84], [16.5, 13.84], [88.5, 13.84], [105., 13.84],
    [0., 24.84], [5.5, 24.84], [99.5, 24.84], [105., 24.84], [0., 30.34], [0., 30.34],
    [105., 30.34], [105., 30.34], [0., 37.66], [0., 37.66], [105., 37.66], [105., 37.66],
    [0., 43.16], [5.5, 43.16], [99.5, 43.16], [105., 43.16], [0., 54.16], [16.5, 54.16],
    [88.5, 54.16], [105., 54.16], [0., 68.], [52.5, 68.], [105., 68.], [16.5, 26.68],
    [52.5, 24.85], [88.5, 26.68], [16.5, 41.31], [52.5, 43.15], [88.5, 41.31], [19.99, 32.29],
    [43.68, 31.53], [61.31, 31.53], [85., 32.29], [19.99, 35.7], [43.68, 36.46], [61.31, 36.46],
    [85., 35.7], [11., 34.], [16.5, 34.], [20.15, 34.], [46.03, 27.53], [58.97, 27.53],
    [43.35, 34.], [52.5, 34.], [61.5, 34.], [46.03, 40.47], [58.97, 40.47], [84.85, 34.],
    [88.5, 34.], [94., 34.]
]
# Center coordinates (pitch center at origin)
KEYPOINT_WORLD_COORDS_2D = [[x - 52.5, y - 34] for x, y in KEYPOINT_WORLD_COORDS_2D]

# Line definitions for keypoint-to-line mapping
LINE_KEYPOINTS_MATCH = {
    "Big rect. left bottom": [24, 68, 25],
    "Big rect. left main": [5, 64, 31, 46, 34, 66, 25],
    "Big rect. left top": [4, 62, 5],
    "Big rect. right bottom": [26, 69, 27],
    "Big rect. right main": [6, 65, 33, 56, 36, 67, 26],
    "Big rect. right top": [6, 63, 7],
    "Circle central": [32, 48, 38, 50, 42, 53, 35, 54, 43, 52, 39, 49],
    "Circle left": [31, 37, 47, 41, 34],
    "Circle right": [33, 40, 55, 44, 36],
    "Goal left crossbar": [16, 12],
    "Goal left post left": [16, 17],
    "Goal left post right": [12, 13],
    "Goal right crossbar": [15, 19],
    "Goal right post left": [15, 14],
    "Goal right post right": [19, 18],
    "Middle line": [2, 32, 51, 35, 29],
    "Side line bottom": [28, 70, 71, 29, 72, 73, 30],
    "Side line left": [1, 4, 8, 13, 17, 20, 24, 28],
    "Side line right": [3, 7, 11, 14, 18, 23, 27, 30],
    "Side line top": [1, 58, 59, 2, 60, 61, 3],
    "Small rect. left bottom": [20, 21],
    "Small rect. left main": [9, 21],
    "Small rect. left top": [8, 9],
    "Small rect. right bottom": [22, 23],
    "Small rect. right main": [10, 22],
    "Small rect. right top": [10, 11],
}


def kp_to_lines(keypoints: dict) -> dict:
    """Convert keypoints to line segments for visualization."""
    lines = {}
    for line_name, kp_indices in LINE_KEYPOINTS_MATCH.items():
        line = []
        for idx in kp_indices:
            if idx in keypoints:
                line.append({'x': keypoints[idx]['x'], 'y': keypoints[idx]['y']})
        if line:
            lines[line_name] = line
    return lines


def download_file(url: str, dest_path: str) -> None:
    """Download file from URL to destination path."""
    import urllib.request

    print(f"Downloading {url} to {dest_path}")
    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    urllib.request.urlretrieve(url, dest_path)
    print(f"Download complete: {dest_path}")


class NbjwCalibrator(BaseCalibrator):
    """NBJW Calibration using HRNet for keypoint detection.

    Provides:
    1. Pitch keypoint detection using HRNet models
    2. Homography computation from detected keypoints
    3. Frame-to-world coordinate transformation
    """

    def __init__(self, config: dict[str, Any] | None = None):
        """Initialize NBJW calibration.

        Args:
            config: Configuration dictionary with optional keys:
                - device: torch device (default: cuda if available)
                - weights_dir: Directory for model weights
                - use_prev_homography: Use previous frame's H on failure
        """
        self.config = config or {}
        self.device = self.config.get("device", "cuda" if torch.cuda.is_available() else "cpu")
        weights_dir = self.config.get("weights_dir")
        self.weights_dir = Path(weights_dir) if weights_dir else Path.home() / ".cache" / "nbjw_calib"
        self.use_prev_homography = self.config.get("use_prev_homography", True)

        # Model input size
        self.input_height = 540
        self.input_width = 960

        # Models
        self.model_kp = None
        self.model_lines = None

        # Transform for input preprocessing
        self.transform = T.Compose([
            T.Resize((self.input_height, self.input_width)),
            T.ToTensor()
        ])

        # Thresholds
        self.kp_threshold = 0.1449
        self.line_threshold = 0.2983

        # State for temporal smoothing
        self.last_homography = None

    def load_models(self) -> None:
        """Load HRNet models for keypoint and line detection."""
        try:
            from model.cls_hrnet import get_cls_net
            from model.cls_hrnet_l import get_cls_net as get_cls_net_l
        except ImportError as e:
            raise ImportError(f"Failed to import nbjw_calib models: {e}")

        # Ensure weights exist
        kp_weights = self.weights_dir / "SV_kp"
        line_weights = self.weights_dir / "SV_lines"

        if not kp_weights.exists():
            download_file(WEIGHTS_URL_KP, str(kp_weights))
        if not line_weights.exists():
            download_file(WEIGHTS_URL_LINES, str(line_weights))

        # Load configs
        import yaml
        config_dir = NBJW_PLUGIN_PATH / "config"

        with open(config_dir / "hrnetv2_w48.yaml") as f:
            cfg_kp = yaml.safe_load(f)
        with open(config_dir / "hrnetv2_w48_l.yaml") as f:
            cfg_lines = yaml.safe_load(f)

        # Load keypoint model
        self.model_kp = get_cls_net(cfg_kp)
        state_dict = torch.load(str(kp_weights), map_location=self.device, weights_only=False)
        self.model_kp.load_state_dict(state_dict)
        self.model_kp.to(self.device)
        self.model_kp.eval()

        # Load lines model
        self.model_lines = get_cls_net_l(cfg_lines)
        state_dict = torch.load(str(line_weights), map_location=self.device, weights_only=False)
        self.model_lines.load_state_dict(state_dict)
        self.model_lines.to(self.device)
        self.model_lines.eval()

        print(f"Loaded NBJW calibration models on {self.device}")

    def detect_keypoints(self, frame: np.ndarray) -> dict:
        """Detect pitch keypoints in a frame.

        Args:
            frame: Input frame (BGR format from OpenCV).

        Returns:
            Dictionary with detected keypoints and lines.
        """
        if self.model_kp is None:
            self.load_models()

        try:
            from utils.utils_heatmap import (
                get_keypoints_from_heatmap_batch_maxpool,
                get_keypoints_from_heatmap_batch_maxpool_l,
                complete_keypoints,
                coords_to_dict
            )
        except ImportError as e:
            raise ImportError(f"Failed to import nbjw_calib utils: {e}")

        # Preprocess
        image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        tensor = self.transform(image).unsqueeze(0).to(self.device)

        # Detect keypoints and lines
        with torch.no_grad():
            heatmaps_kp = self.model_kp(tensor)
            heatmaps_lines = self.model_lines(tensor)

        # Extract coordinates from heatmaps
        kp_coords = get_keypoints_from_heatmap_batch_maxpool(heatmaps_kp[:, :-1, :, :])
        line_coords = get_keypoints_from_heatmap_batch_maxpool_l(heatmaps_lines[:, :-1, :, :])

        # Convert to dictionaries
        kp_dict = coords_to_dict(kp_coords, threshold=self.kp_threshold)
        lines_dict = coords_to_dict(line_coords, threshold=self.line_threshold)

        # Complete missing keypoints using line intersections
        complete_dict = complete_keypoints(
            kp_dict, lines_dict,
            w=self.input_width, h=self.input_height,
            normalize=True
        )

        keypoints = complete_dict[0] if complete_dict else {}
        lines = kp_to_lines(keypoints)

        return {
            'keypoints': keypoints,
            'lines': lines,
            'raw_kp_dict': kp_dict[0] if kp_dict else {},
            'raw_lines_dict': lines_dict[0] if lines_dict else {},
        }

    def compute_homography(
        self,
        keypoints: dict,
        image_width: int,
        image_height: int,
    ) -> np.ndarray | None:
        """Compute homography from detected keypoints.

        Args:
            keypoints: Detected keypoints (normalized 0-1 coordinates).
            image_width: Original image width.
            image_height: Original image height.

        Returns:
            3x3 homography matrix (image -> world), or None if failed.
        """
        try:
            from utils.utils_calib import FramebyFrameCalib
        except ImportError:
            return self._compute_homography_simple(keypoints, image_width, image_height)

        if not keypoints:
            if self.use_prev_homography and self.last_homography is not None:
                return self.last_homography
            return None

        # Denormalize keypoints
        denorm_kp = {}
        for idx, kp in keypoints.items():
            denorm_kp[idx] = {
                'x': kp['x'] * image_width,
                'y': kp['y'] * image_height,
            }
            if 'p' in kp:
                denorm_kp[idx]['p'] = kp['p']

        # Use FramebyFrameCalib
        calib = FramebyFrameCalib(iwidth=image_width, iheight=image_height, denormalize=False)
        calib.update(denorm_kp)

        # Get homography (image -> world)
        homography = calib.get_homography_from_ground_plane(use_ransac=50, inverse=True)

        if homography is not None:
            self.last_homography = homography
            return homography

        if self.use_prev_homography and self.last_homography is not None:
            return self.last_homography

        return None

    def _compute_homography_simple(
        self,
        keypoints: dict,
        image_width: int,
        image_height: int,
    ) -> np.ndarray | None:
        """Simple homography computation without FramebyFrameCalib."""
        if len(keypoints) < 4:
            if self.use_prev_homography and self.last_homography is not None:
                return self.last_homography
            return None

        # Collect image and world point correspondences
        img_pts = []
        world_pts = []

        for idx, kp in keypoints.items():
            if idx > 57:
                continue  # Skip auxiliary keypoints

            # Image coordinates
            img_x = kp['x'] * image_width
            img_y = kp['y'] * image_height

            # World coordinates
            world_coord = KEYPOINT_WORLD_COORDS_2D[idx - 1]

            img_pts.append([img_x, img_y])
            world_pts.append([world_coord[0], world_coord[1]])

        if len(img_pts) < 4:
            if self.use_prev_homography and self.last_homography is not None:
                return self.last_homography
            return None

        img_pts = np.array(img_pts, dtype=np.float32)
        world_pts = np.array(world_pts, dtype=np.float32)

        # Compute homography (world -> image)
        H, mask = cv2.findHomography(world_pts, img_pts, cv2.RANSAC, 5.0)

        if H is None:
            if self.use_prev_homography and self.last_homography is not None:
                return self.last_homography
            return None

        # Invert for image -> world
        det = np.linalg.det(H)
        if np.isclose(det, 0):
            if self.use_prev_homography and self.last_homography is not None:
                return self.last_homography
            return None

        H_inv = np.linalg.inv(H)
        homography = H_inv / H_inv[-1, -1]

        self.last_homography = homography
        return homography

    def project_to_pitch(
        self,
        image_point: tuple[float, float],
        homography: np.ndarray,
    ) -> tuple[float, float] | None:
        """Project image point to pitch coordinates.

        Args:
            image_point: (x, y) in image coordinates.
            homography: Image -> world homography matrix.

        Returns:
            (x, y) in pitch coordinates (meters), or None if failed.
        """
        if homography is None:
            return None

        pt = np.array([image_point[0], image_point[1], 1.0])
        world_pt = homography @ pt

        if abs(world_pt[2]) < 1e-6:
            return None

        world_x = world_pt[0] / world_pt[2]
        world_y = world_pt[1] / world_pt[2]

        return (float(world_x), float(world_y))

    def reset(self) -> None:
        """Reset calibration state."""
        self.last_homography = None


# Backwards compatibility alias
NbjwCalibration = NbjwCalibrator
