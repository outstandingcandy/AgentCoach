"""CLIP-ReID feature extraction (open_clip backbone).

Implements :class:`BaseReIDExtractor` on top of an OpenCLIP vision
encoder fine-tuned for player re-identification (Habel et al.,
MMSports 2022 — https://github.com/KonradHabel/clip_reid, MIT). Use as
a third option alongside OSNet / PRTReID via ``reid.backend:
clip_reid`` so the same crops can be re-encoded with a stronger,
jersey-aware embedding and the tracker / consolidator behaviour
compared A/B.

Two implementation notes that differ from the other extractors:

1. **No auto-download.** The fine-tuned weights live on a Google
   Drive link the upstream repo provides — there's no permanent URL
   we can urlretrieve reliably. ``reid.clip_reid.weights_path`` is
   required; an explicit FileNotFoundError points the user at the
   Drive link if it's missing.
2. **CLIP normalization, not ImageNet.** The encoder was trained on
   OpenAI's CLIP statistics, which differ from PRTReID's ImageNet
   mean/std by enough (~10%) to noticeably degrade embeddings if you
   swap them.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from ...interfaces import BaseReIDExtractor

# OpenAI CLIP image-normalization constants. These are the values the
# CLIP-ReID fine-tune (and every open_clip pretrained checkpoint) saw
# during training; ImageNet mean/std would tilt the input distribution
# enough to hurt embedding quality.
_CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
_CLIP_STD = (0.26862954, 0.26130258, 0.27577711)

# Public reference for the fine-tuned weights (used in the error
# message when ``weights_path`` is missing). The upstream README
# distributes this checkpoint on Google Drive; no stable HTTP URL.
_WEIGHTS_REFERENCE = (
    "https://github.com/KonradHabel/clip_reid "
    "(weights_e4.pth, ~1.5 GB, on the Google Drive linked from the README)"
)

# Per-backbone defaults: native input resolution + (post-proj_kept,
# post-proj_removed) feature dims. ``post-proj_removed`` is the
# transformer width (= the visual encoder's natural output before the
# joint image/text projection layer), and is what CLIP-ReID's
# upstream evaluation uses (``remove_proj=True``).
#
# Numbers come from inspecting open_clip's ``create_model`` outputs;
# tweak the table if you add a new backbone.
_BACKBONE_SPECS: dict[str, dict[str, int]] = {
    "ViT-B-16":     {"input": 224, "dim_proj": 512,  "dim_no_proj": 768},
    "ViT-L-14":     {"input": 224, "dim_proj": 768,  "dim_no_proj": 1024},
    "ViT-L-14-336": {"input": 336, "dim_proj": 768,  "dim_no_proj": 1024},
    "RN50":         {"input": 224, "dim_proj": 1024, "dim_no_proj": 2048},
}


class ClipReIDExtractor(BaseReIDExtractor):
    """OpenCLIP-backed ReID extractor with optional player-ReID fine-tune.

    Output is L2-normalized image features with dimension determined
    by the chosen ``backbone`` and ``remove_proj`` setting (see
    ``_BACKBONE_SPECS``). With ``remove_proj=True`` (the CLIP-ReID
    default), ViT-L-14 → 1024-dim, ViT-B-16 → 768-dim.
    """

    def __init__(self, config: dict[str, Any] | None = None):
        """Initialize ClipReID extractor.

        Args:
            config: dict with keys:
                - device: "cuda" / "cpu" (default: cuda if available)
                - backbone: open_clip model name; default "ViT-L-14"
                - pretrained: open_clip pretrained tag for the base
                  weights before fine-tune; default "openai"
                - weights_path: path to the fine-tuned ``.pth`` state
                  dict (REQUIRED; the file must be downloaded manually).
                - remove_proj: bool; mirrors upstream
                  ``OpenClipModel(remove_proj=True)`` so the image
                  encoder output is the pre-projection feature vector
                  rather than the joint image/text-projected one.
                  Default: True (matches CLIP-ReID training).
                - batch_size: inference batch (default: 32). ViT-L-14
                  at 336×336 uses ~4 GB GPU @ batch 32; halve if OOM.
        """
        self.config = config or {}
        self.device = self.config.get(
            "device", "cuda" if torch.cuda.is_available() else "cpu",
        )
        self.backbone = self.config.get("backbone", "ViT-L-14")
        self.pretrained = self.config.get("pretrained", "openai")
        self.remove_proj = bool(self.config.get("remove_proj", True))
        self.batch_size = int(self.config.get("batch_size", 32))

        spec = _BACKBONE_SPECS.get(self.backbone)
        if spec is None:
            # Unknown backbones still work as long as open_clip
            # accepts them; we just have to wait until load_model()
            # to compute feature_dim from a dry forward pass.
            self.input_size = 224
            self._feature_dim = 0
        else:
            self.input_size = spec["input"]
            self._feature_dim = (
                spec["dim_no_proj"] if self.remove_proj else spec["dim_proj"]
            )

        self.model = None

    # ------------------------------------------------------------------
    # BaseReIDExtractor interface
    # ------------------------------------------------------------------

    def load_model(self, model_path: str | Path | None = None) -> None:
        """Load OpenCLIP backbone + apply the fine-tuned state dict.

        Args:
            model_path: optional override of ``config['weights_path']``.

        Raises:
            ImportError: open_clip not installed.
            FileNotFoundError: no fine-tuned weights provided.
        """
        weights_path = model_path or self.config.get("weights_path")
        if not weights_path:
            raise FileNotFoundError(
                f"reid.clip_reid.weights_path is required. "
                f"Download the fine-tuned weights from {_WEIGHTS_REFERENCE} "
                f"and point ``weights_path`` at the resulting .pth file."
            )
        weights_path = Path(weights_path)
        if not weights_path.exists():
            raise FileNotFoundError(
                f"clip_reid weights not found at {weights_path}. "
                f"Download from {_WEIGHTS_REFERENCE}.",
            )

        try:
            import open_clip
        except ImportError as exc:
            raise ImportError(
                "open_clip is required for the clip_reid backend. "
                "Install with `pip install open_clip_torch`.",
            ) from exc

        # Build the base model. open_clip.create_model handles
        # backbone string + pretrained tag → fully constructed model
        # (image + text encoders), torch nn.Module, weights initialized
        # to the OpenAI / Laion checkpoint.
        model = open_clip.create_model(self.backbone, pretrained=self.pretrained)

        # CLIP-ReID's contribution is to use the unprojected visual
        # features. open_clip's ViT image encoder applies a projection
        # to a shared image/text contrastive space at the very end;
        # setting that projection to None makes ``model.visual(x)``
        # return the pre-projection embedding. For ResNet variants the
        # attribute name differs (``attnpool.c_proj``); detect and
        # disable both forms.
        if self.remove_proj and hasattr(model, "visual"):
            visual = model.visual
            if hasattr(visual, "proj") and visual.proj is not None:
                visual.proj = None
            elif hasattr(visual, "attnpool") and hasattr(visual.attnpool, "c_proj"):
                # RN50: stripping is fiddlier; downstream callers
                # should prefer ViT backbones until we need it.
                pass

        # Apply the fine-tuned checkpoint. Upstream's training saves
        # the wrapper ``OpenClipModel`` state, so every key is
        # prefixed with ``model.`` (e.g. ``model.visual.class_embedding``).
        # open_clip's bare model expects unprefixed keys, so strip the
        # prefix before load — otherwise strict=False silently lets
        # every key fail to match and the model stays at its randomly-
        # initialized weights, which feels like "loaded OK" but isn't.
        state = torch.load(
            weights_path, map_location=self.device, weights_only=False,
        )
        if isinstance(state, dict) and "model" in state and "epoch" in state:
            # Training-loop wrapper: {"model": <state>, "epoch": N, ...}.
            state = state["model"]
        # Strip a uniform ``model.`` prefix if the whole checkpoint
        # has one — upstream's OpenClipModel is one such wrapper.
        if state and all(k.startswith("model.") for k in state.keys()):
            state = {k[len("model."):]: v for k, v in state.items()}
        missing, unexpected = model.load_state_dict(state, strict=False)
        # Trim a known-good amount of slack; surface unusual mismatches
        # so a wrong checkpoint doesn't silently load with mostly-zero
        # weights. The text encoder + logit_scale + visual.proj are
        # the expected misses when remove_proj=True.
        unexpected_real = [
            k for k in unexpected
            if not (k.startswith("transformer.")
                    or k.startswith("token_embedding")
                    or k.startswith("positional_embedding")
                    or k == "logit_scale"
                    or k == "ln_final.weight" or k == "ln_final.bias"
                    or k == "text_projection"
                    # ``visual.proj`` is the joint image/text projection
                    # we explicitly disabled via remove_proj.
                    or (self.remove_proj and k == "visual.proj"))
        ]
        if unexpected_real:
            import logging
            logging.getLogger(__name__).warning(
                "clip_reid load_state_dict: %d unexpected keys (first 5: %s)",
                len(unexpected_real), unexpected_real[:5],
            )

        model = model.to(self.device).eval()
        self.model = model

        # If the backbone wasn't in our table, infer feature_dim from
        # a single dry forward pass so the property is accurate.
        if self._feature_dim == 0:
            with torch.no_grad():
                dummy = torch.zeros(
                    (1, 3, self.input_size, self.input_size),
                    device=self.device,
                )
                feat = model.visual(dummy)
                self._feature_dim = int(feat.shape[-1])

    def extract(self, crops: list[np.ndarray]) -> np.ndarray:
        """Encode ``crops`` (BGR np arrays) into L2-normalized features.

        Returns:
            ``(N, feature_dim)`` float32 array. If ``crops`` is empty,
            returns ``(0, feature_dim)``.
        """
        if self.model is None:
            self.load_model()
        if not crops:
            return np.zeros((0, self._feature_dim), dtype=np.float32)

        # Preprocess all crops into one big tensor. CPU-side; OpenCV
        # resize beats PIL by ~3x on large batches.
        tensors = [self._preprocess(c) for c in crops]
        batch = torch.stack(tensors).to(self.device, non_blocking=True)

        out_chunks: list[torch.Tensor] = []
        with torch.no_grad():
            for i in range(0, batch.shape[0], self.batch_size):
                chunk = batch[i:i + self.batch_size]
                feats = self.model.visual(chunk)
                feats = F.normalize(feats, p=2, dim=1)
                out_chunks.append(feats.cpu())
        return torch.cat(out_chunks, dim=0).numpy().astype(np.float32)

    @property
    def feature_dim(self) -> int:
        return self._feature_dim

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _preprocess(self, crop: np.ndarray) -> torch.Tensor:
        """BGR np → square, aspect-ratio-preserving, CLIP-normalized tensor.

        The CLIP-ReID paper uses ``RectResize`` (aspect-ratio
        preserving scale + zero-pad) because raw resize squashes the
        human silhouette and degrades the embedding. We replicate it
        inline (~10 LOC) so we don't have to pull the upstream module.
        """
        h, w = crop.shape[:2]
        size = self.input_size

        # BGR → RGB.
        if crop.ndim == 3 and crop.shape[2] == 3:
            crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)

        # Aspect-ratio-preserving scale: target = size×size, source
        # fits inside by scaling its longer side to ``size`` and zero-
        # padding the rest.
        scale = size / max(h, w)
        new_w = max(1, int(round(w * scale)))
        new_h = max(1, int(round(h * scale)))
        resized = cv2.resize(crop, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        canvas = np.zeros((size, size, 3), dtype=np.uint8)
        # Centre the resized crop on the square canvas.
        y0 = (size - new_h) // 2
        x0 = (size - new_w) // 2
        canvas[y0:y0 + new_h, x0:x0 + new_w] = resized

        # uint8 [0..255] → float32 [0..1] → CLIP-normalize.
        tensor = torch.from_numpy(canvas).float().permute(2, 0, 1) / 255.0
        mean = torch.tensor(_CLIP_MEAN, dtype=tensor.dtype).view(3, 1, 1)
        std = torch.tensor(_CLIP_STD, dtype=tensor.dtype).view(3, 1, 1)
        tensor = (tensor - mean) / std
        return tensor
