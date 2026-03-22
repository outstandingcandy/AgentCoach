"""Weight downloader for PnLCalib pretrained models.

Downloads and caches pretrained weights from PnLCalib GitHub releases.
"""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path
from typing import Any

import httpx

logger = logging.getLogger(__name__)


class PnLCalibWeightDownloader:
    """Download and cache PnLCalib pretrained weights.

    Available weights from PnLCalib GitHub releases v1.0.0:
    - SV_kp.pth: Single-view keypoint detector
    - SV_lines.pth: Single-view line detector
    - WC14_kp.pth: WorldCup 2014 fine-tuned keypoint detector
    - WC14_lines.pth: WorldCup 2014 fine-tuned line detector
    - TSWC_kp.pth: TS World Cup fine-tuned keypoint detector
    - TSWC_lines.pth: TS World Cup fine-tuned line detector

    Attributes:
        CACHE_DIR: Default cache directory for downloaded weights.
        BASE_URL: Base URL for downloading weights.
    """

    CACHE_DIR = Path.home() / ".cache" / "goal-insight" / "pnlcalib"
    BASE_URL = "https://github.com/mguti97/PnLCalib/releases/download/v1.0.0/"

    # Available weight files and their expected sizes (for validation)
    # Note: PnLCalib release files don't have .pth extension
    AVAILABLE_WEIGHTS: dict[str, dict[str, Any]] = {
        "SV_kp": {
            "filename": "SV_kp",
            "description": "Single-view keypoint detector (recommended)",
            "type": "keypoint",
        },
        "SV_lines": {
            "filename": "SV_lines",
            "description": "Single-view line detector (recommended)",
            "type": "line",
        },
        "SV_FT_WC14_kp": {
            "filename": "SV_FT_WC14_kp",
            "description": "WorldCup 2014 fine-tuned keypoint detector",
            "type": "keypoint",
        },
        "SV_FT_WC14_lines": {
            "filename": "SV_FT_WC14_lines",
            "description": "WorldCup 2014 fine-tuned line detector",
            "type": "line",
        },
        "SV_FT_TSWC_kp": {
            "filename": "SV_FT_TSWC_kp",
            "description": "TS World Cup fine-tuned keypoint detector",
            "type": "keypoint",
        },
        "SV_FT_TSWC_lines": {
            "filename": "SV_FT_TSWC_lines",
            "description": "TS World Cup fine-tuned line detector",
            "type": "line",
        },
        "MV_kp": {
            "filename": "MV_kp",
            "description": "Multi-view keypoint detector",
            "type": "keypoint",
        },
        "MV_lines": {
            "filename": "MV_lines",
            "description": "Multi-view line detector",
            "type": "line",
        },
    }

    def __init__(self, cache_dir: str | Path | None = None):
        """Initialize weight downloader.

        Args:
            cache_dir: Directory to cache downloaded weights.
                      Defaults to ~/.cache/goal-insight/pnlcalib/
        """
        self.cache_dir = Path(cache_dir) if cache_dir else self.CACHE_DIR
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def get_weight_path(self, weight_name: str, download: bool = True) -> Path:
        """Get path to weight file, downloading if necessary.

        Args:
            weight_name: Name of the weights (e.g., "SV_kp", "SV_lines").
            download: Whether to download if not cached.

        Returns:
            Path to the weight file.

        Raises:
            ValueError: If weight_name is not recognized.
            RuntimeError: If download fails.
        """
        if weight_name not in self.AVAILABLE_WEIGHTS:
            available = ", ".join(self.AVAILABLE_WEIGHTS.keys())
            raise ValueError(f"Unknown weight name: {weight_name}. Available: {available}")

        weight_info = self.AVAILABLE_WEIGHTS[weight_name]
        filename = weight_info["filename"]
        local_path = self.cache_dir / filename

        if local_path.exists():
            logger.info(f"Using cached weights: {local_path}")
            return local_path

        if not download:
            raise FileNotFoundError(f"Weight file not found: {local_path}")

        # Download weights
        url = self.BASE_URL + filename
        self._download_file(url, local_path)

        return local_path

    def _download_file(
        self,
        url: str,
        dest_path: Path,
        timeout: float = 300.0,
    ) -> None:
        """Download a file from URL to local path.

        Args:
            url: URL to download from.
            dest_path: Local destination path.
            timeout: Download timeout in seconds.

        Raises:
            RuntimeError: If download fails.
        """
        logger.info(f"Downloading weights from {url}")
        logger.info(f"Saving to {dest_path}")

        try:
            with httpx.stream("GET", url, timeout=timeout, follow_redirects=True) as response:
                response.raise_for_status()

                total_size = int(response.headers.get("content-length", 0))
                downloaded = 0

                # Download to temp file first
                temp_path = dest_path.with_suffix(".tmp")
                with open(temp_path, "wb") as f:
                    for chunk in response.iter_bytes(chunk_size=8192):
                        f.write(chunk)
                        downloaded += len(chunk)
                        if total_size > 0:
                            progress = downloaded / total_size * 100
                            if downloaded % (1024 * 1024 * 10) < 8192:  # Log every ~10MB
                                logger.info(f"Download progress: {progress:.1f}%")

                # Move to final location
                temp_path.rename(dest_path)
                logger.info(f"Download complete: {dest_path}")

        except httpx.HTTPError as e:
            if dest_path.with_suffix(".tmp").exists():
                dest_path.with_suffix(".tmp").unlink()
            raise RuntimeError(f"Failed to download weights from {url}: {e}")

    def get_keypoint_weights(
        self,
        variant: str = "SV",
        download: bool = True,
    ) -> Path:
        """Get keypoint detector weights.

        Args:
            variant: Weight variant - "SV", "SV_FT_WC14", "SV_FT_TSWC", or "MV".
            download: Whether to download if not cached.

        Returns:
            Path to the weight file.
        """
        weight_name = f"{variant}_kp"
        return self.get_weight_path(weight_name, download=download)

    def get_line_weights(
        self,
        variant: str = "SV",
        download: bool = True,
    ) -> Path:
        """Get line detector weights.

        Args:
            variant: Weight variant - "SV", "SV_FT_WC14", "SV_FT_TSWC", or "MV".
            download: Whether to download if not cached.

        Returns:
            Path to the weight file.
        """
        weight_name = f"{variant}_lines"
        return self.get_weight_path(weight_name, download=download)

    def list_available_weights(self) -> list[dict[str, Any]]:
        """List all available weight files.

        Returns:
            List of weight information dictionaries.
        """
        result = []
        for name, info in self.AVAILABLE_WEIGHTS.items():
            local_path = self.cache_dir / info["filename"]
            result.append({
                "name": name,
                "filename": info["filename"],
                "description": info["description"],
                "type": info["type"],
                "cached": local_path.exists(),
                "path": str(local_path) if local_path.exists() else None,
            })
        return result

    def clear_cache(self) -> int:
        """Clear all cached weight files.

        Returns:
            Number of files deleted.
        """
        count = 0
        for weight_info in self.AVAILABLE_WEIGHTS.values():
            local_path = self.cache_dir / weight_info["filename"]
            if local_path.exists():
                local_path.unlink()
                count += 1
                logger.info(f"Deleted cached weights: {local_path}")
        return count

    @classmethod
    def verify_weights(cls, weight_path: Path) -> bool:
        """Verify that a weight file is valid PyTorch weights.

        Args:
            weight_path: Path to the weight file.

        Returns:
            True if weights appear valid.
        """
        import torch

        try:
            state_dict = torch.load(weight_path, map_location="cpu")
            # Check if it's a valid state dict
            if isinstance(state_dict, dict):
                if "state_dict" in state_dict:
                    state_dict = state_dict["state_dict"]
                return len(state_dict) > 0
            return False
        except Exception as e:
            logger.warning(f"Failed to verify weights {weight_path}: {e}")
            return False
