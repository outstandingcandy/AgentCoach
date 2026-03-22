"""PRTReID feature extraction with role prediction.

Implements BaseReIDExtractor interface.
Based on SoccerNet sn-gamestate's PRTReID implementation.
"""

from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from ...interfaces import BaseReIDExtractor

# PRTReID weights URL (from sn-gamestate)
WEIGHTS_URL = "https://zenodo.org/records/10653453/files/prtreid-soccernet-baseline.pth.tar?download=1"
WEIGHTS_MD5 = "9633825232bc89f23a94522c5561650e"


class PRTReIDExtractor(BaseReIDExtractor):
    """PRTReID feature extraction with role prediction.

    Based on BPBreID with HRNet32 backbone, outputs:
    - 256-dim embeddings for ReID
    - Role classification (ball, goalkeeper, player, referee, other)
    """

    def __init__(self, config: dict[str, Any] | None = None):
        """Initialize PRTReID extractor.

        Args:
            config: Configuration dictionary with optional keys:
                - device: "cuda" or "cpu"
                - weights_path: Path to PRTReID weights
                - referee_confidence_threshold: Threshold for referee detection (default: 0.85)
                - goalkeeper_confidence_threshold: Threshold for GK detection (default: 0.7)
        """
        self.config = config or {}
        self.input_size = (256, 128)  # Height x Width
        self._feature_dim = 256  # PRTReID outputs 256-dim embeddings
        self.device = self.config.get("device", "cuda" if torch.cuda.is_available() else "cpu")

        # Role mapping (from sn-gamestate)
        self.role_map = {
            0: 'ball',
            1: 'goalkeeper',
            2: 'other',
            3: 'player',
            4: 'referee'
        }

        # Image preprocessing parameters
        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]

        # Role classification confidence thresholds
        self.referee_confidence_threshold = self.config.get("referee_confidence_threshold", 0.85)
        self.goalkeeper_confidence_threshold = self.config.get("goalkeeper_confidence_threshold", 0.7)

        self.model = None
        self._prtreid_available = None

    def _check_prtreid_available(self) -> bool:
        """Check if PRTReID is available."""
        if self._prtreid_available is not None:
            return self._prtreid_available

        try:
            import prtreid  # noqa: F401
            self._prtreid_available = True
        except ImportError:
            self._prtreid_available = False

        return self._prtreid_available

    def load_model(self, model_path: str | Path | None = None) -> None:
        """Load PRTReID model.

        Args:
            model_path: Path to PRTReID weights. If None, attempts to download.
        """
        model_path = model_path or self.config.get("weights_path")

        if not self._check_prtreid_available():
            raise ImportError(
                "PRTReID is not available. Install with:\n"
                "uv pip install git+https://github.com/VlSomers/prtreid.git"
            )

        self._load_prtreid(model_path)

    def _load_prtreid(self, weights_path: str | Path | None = None) -> None:
        """Load PRTReID model using checkpoint config."""
        from yacs.config import CfgNode as CN
        from prtreid.scripts.default_config import get_default_config
        from prtreid.models import build_model

        if weights_path is None:
            weights_path = self._download_weights()

        weights_path = Path(weights_path)

        # Load checkpoint with weights_only=False for PyTorch 2.6+
        print(f"Loading PRTReID checkpoint from {weights_path}")
        checkpoint = torch.load(weights_path, map_location='cpu', weights_only=False)

        # Get config from checkpoint
        saved_config = checkpoint['config']

        # Build config using default + saved
        cfg = get_default_config()

        # Update config from checkpoint
        def update_cfg(cfg_node, saved_dict):
            for k, v in saved_dict.items():
                if isinstance(v, dict) and k in cfg_node:
                    update_cfg(cfg_node[k], v)
                elif k in cfg_node:
                    cfg_node[k] = v

        update_cfg(cfg, saved_config)

        # Set device
        cfg.use_gpu = self.device.startswith('cuda')

        # Fix paths - set HRNet weights path
        hrnet_path = self._ensure_hrnet_weights(cfg.model.bpbreid.hrnet_pretrained_path)
        cfg.model.bpbreid.hrnet_pretrained_path = str(hrnet_path)

        # Freeze config
        cfg.freeze()

        # Build model
        print(f"Building PRTReID model: {cfg.model.name} with {cfg.model.bpbreid.backbone} backbone")
        self.model = build_model(
            name=cfg.model.name,
            loss=cfg.loss.name,
            pretrained=False,
            num_classes=1,
            use_gpu=cfg.use_gpu,
            pooling=cfg.model.bpbreid.pooling,
            normalization=cfg.model.bpbreid.normalization,
            last_stride=cfg.model.bpbreid.last_stride,
            config=cfg
        )

        # Load weights manually
        state_dict = checkpoint['state_dict']
        new_state_dict = {}
        skipped_keys = []
        for k, v in state_dict.items():
            key = k[7:] if k.startswith('module.') else k

            # Skip identity_classifier layers (different num_classes)
            if 'identity_classifier' in key:
                skipped_keys.append(key)
                continue

            new_state_dict[key] = v

        if skipped_keys:
            print(f"  Skipped {len(skipped_keys)} identity classifier layers")

        # Load state dict
        missing, unexpected = self.model.load_state_dict(new_state_dict, strict=False)
        if missing:
            missing = [k for k in missing if 'identity_classifier' not in k]
            if missing:
                print(f"  Missing keys: {missing[:5]}...")
        if unexpected:
            print(f"  Unexpected keys: {len(unexpected)}")

        self.model = self.model.to(self.device)
        self.model.eval()

        self.cfg = cfg
        self._feature_dim = cfg.model.bpbreid.dim_reduce_output
        print(f"PRTReID model loaded successfully (feature_dim={self._feature_dim})")

    def _ensure_hrnet_weights(self, original_path: str) -> Path:
        """Get HRNet pretrained weights path."""
        cache_dir = Path.home() / ".cache" / "prtreid"
        cache_dir.mkdir(parents=True, exist_ok=True)
        return cache_dir

    def _download_weights(self) -> str:
        """Download pretrained weights from Zenodo."""
        cache_dir = Path.home() / ".cache" / "prtreid"
        cache_dir.mkdir(parents=True, exist_ok=True)
        weights_path = cache_dir / "prtreid-soccernet-baseline.pth.tar"

        if not weights_path.exists():
            print(f"Downloading PRTReID weights from Zenodo...")
            import urllib.request
            urllib.request.urlretrieve(WEIGHTS_URL, weights_path)
            print(f"Downloaded to {weights_path}")

        return str(weights_path)

    def preprocess(self, image: np.ndarray) -> torch.Tensor:
        """Preprocess image for model input."""
        # Convert BGR to RGB
        if len(image.shape) == 3 and image.shape[2] == 3:
            image = image[:, :, ::-1].copy()

        # Convert to PIL and resize
        pil_image = Image.fromarray(image)
        pil_image = pil_image.resize((self.input_size[1], self.input_size[0]))

        # Convert to tensor and normalize
        img_tensor = torch.from_numpy(np.array(pil_image)).float()
        img_tensor = img_tensor.permute(2, 0, 1) / 255.0

        # Normalize
        for c in range(3):
            img_tensor[c] = (img_tensor[c] - self.mean[c]) / self.std[c]

        return img_tensor

    def extract(self, crops: list[np.ndarray]) -> np.ndarray:
        """Extract ReID features from image crops.

        Args:
            crops: List of person image crops (BGR format).

        Returns:
            Array of feature vectors, shape (N, feature_dim).
        """
        result = self.extract_with_roles(crops)
        return result['embeddings']

    def extract_with_roles(
        self,
        crops: list[np.ndarray],
    ) -> dict[str, Any]:
        """Extract features and predict roles.

        Args:
            crops: List of person image crops (BGR format).

        Returns:
            Dictionary with:
                - embeddings: np.ndarray (N, feature_dim)
                - roles: list[str] - predicted roles
                - role_confidences: np.ndarray (N,)
                - visibility_scores: np.ndarray (N,)
        """
        if self.model is None:
            self.load_model()

        if not crops:
            return {
                'embeddings': np.array([]).reshape(0, self._feature_dim),
                'roles': [],
                'role_confidences': np.array([]),
                'visibility_scores': np.array([])
            }

        return self._extract_prtreid(crops)

    def _extract_prtreid(self, crops: list[np.ndarray]) -> dict[str, Any]:
        """Extract features using PRTReID."""
        # Preprocess all crops
        tensors = [self.preprocess(crop) for crop in crops]
        batch = torch.stack(tensors).to(self.device)

        # Forward pass
        batch_size = 32
        all_embeddings = []
        all_role_logits = []
        all_visibility = []

        with torch.no_grad():
            for i in range(0, len(batch), batch_size):
                batch_slice = batch[i:i + batch_size]

                output = self.model(batch_slice)

                # Output[0] is embeddings dict
                embeddings_dict = output[0]

                # Get global embeddings (256-dim, normalized)
                if 'bn_globl' in embeddings_dict:
                    emb = embeddings_dict['bn_globl']
                else:
                    emb = embeddings_dict['globl']

                # Normalize embeddings
                emb = F.normalize(emb, p=2, dim=1)
                all_embeddings.append(emb.cpu())

                # Get role predictions
                if hasattr(self.model, 'global_Role_classifier'):
                    role_output = self.model.global_Role_classifier(embeddings_dict['globl'])
                    role_logits = role_output[1]
                    all_role_logits.append(role_logits.cpu())

                # Get visibility scores if available
                if len(output) > 4 and isinstance(output[4], dict):
                    vis_dict = output[4]
                    if 'globl' in vis_dict:
                        vis = vis_dict['globl']
                        if isinstance(vis, torch.Tensor):
                            all_visibility.append(vis.cpu())

        # Concatenate results
        embeddings = torch.cat(all_embeddings, dim=0).numpy()

        # Handle roles
        if all_role_logits:
            role_logits = torch.cat(all_role_logits, dim=0)
            role_probs = torch.softmax(role_logits, dim=1)
            role_indices = role_probs.argmax(dim=1).numpy()
            role_confidences = role_probs.max(dim=1)[0].numpy()

            roles = []
            for idx, conf in zip(role_indices, role_confidences):
                role = self.role_map.get(int(idx), 'player')

                # Apply confidence thresholds
                if role == 'referee' and conf < self.referee_confidence_threshold:
                    role = 'player'
                elif role == 'goalkeeper' and conf < self.goalkeeper_confidence_threshold:
                    role = 'player'

                roles.append(role)
        else:
            roles = ['player'] * len(crops)
            role_confidences = np.ones(len(crops))

        # Handle visibility
        if all_visibility:
            visibility = torch.cat(all_visibility, dim=0).numpy()
            if visibility.ndim > 1:
                visibility = visibility.mean(axis=1)
        else:
            visibility = np.ones(len(crops))

        return {
            'embeddings': embeddings,
            'roles': roles,
            'role_confidences': role_confidences,
            'visibility_scores': visibility
        }

    @property
    def feature_dim(self) -> int:
        """Return feature dimension (256 for PRTReID)."""
        return self._feature_dim
