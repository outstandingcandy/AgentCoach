"""Jersey/role recognition via Google Gemini (AI Studio).

Subclasses :class:`BaseVLMRecognizer` — only the google-genai payload
format and the ``generate_content`` call live here.

Requires ``google-genai`` and the ``GOOGLE_API_KEY`` env var.
"""

from __future__ import annotations

import logging
import os
import re
import time
from typing import Any

import numpy as np

from ._vlm_base import BaseVLMRecognizer, Block

try:
    from google import genai
    from google.genai import types as _genai_types
except ImportError:
    genai = None
    _genai_types = None

try:
    import cv2
    from PIL import Image
except ImportError:
    cv2 = None
    Image = None

logger = logging.getLogger(__name__)


class GeminiJerseyRecognizer(BaseVLMRecognizer):
    """Google Gemini backend (AI Studio API key)."""

    def __init__(self, config: dict[str, Any] | None = None):
        if genai is None:
            raise ImportError(
                "google-genai is required; install with "
                "`pip install google-genai`")
        super().__init__(config)
        self.model_id = self.config.get("model_id", "gemini-2.5-pro")

        api_key = os.environ.get(
            self.config.get("api_key_env", "GOOGLE_API_KEY")
        )
        if not api_key:
            raise RuntimeError(
                "GOOGLE_API_KEY not set (or the env var named in "
                "jersey.api_key_env).")
        self._client = genai.Client(api_key=api_key)

    # ------------------------------------------------------------------
    # BaseVLMRecognizer hook
    # ------------------------------------------------------------------

    def _call(
        self,
        blocks: list[Block],
        system: str | None = None,
        max_tokens: int | None = None,
    ) -> str | None:
        # Gemini's ``contents`` is a flat parts list (mixed strings and
        # images). The system prompt — when given — is inlined as a text
        # part *just before* the trailing user-task prompt, which the base
        # always emits as the final block. This matches the historic
        # Gemini call shape (system text close to the question).
        if system and blocks:
            blocks = blocks[:-1] + [("text", system)] + blocks[-1:]

        parts: list[Any] = []
        for kind, value in blocks:
            if kind == "text":
                parts.append(value)
            else:  # "image" or "image_wide"
                img = self._pil_from_crop(value, is_wide=(kind == "image_wide"))
                if img is not None:
                    parts.append(img)
        if not parts:
            return None
        return self._invoke_with_retry(parts, max_tokens or self.max_tokens)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _pil_from_crop(
        self, crop: np.ndarray, *, is_wide: bool = False,
    ) -> "Image.Image | None":
        if cv2 is None or Image is None:
            raise ImportError("opencv + Pillow are required")
        if crop is None or crop.size == 0:
            return None
        h, w = crop.shape[:2]
        max_dim = 1280 if is_wide else self.max_crop_dim
        if max(h, w) > max_dim:
            scale = max_dim / max(h, w)
            crop = cv2.resize(
                crop, (int(w * scale), int(h * scale)),
                interpolation=cv2.INTER_AREA,
            )
        rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        return Image.fromarray(rgb)

    def _invoke_with_retry(
        self, parts: list[Any], max_tokens: int,
    ) -> str | None:
        gen_config = _genai_types.GenerateContentConfig(
            max_output_tokens=max_tokens,
            temperature=0.0,
            thinking_config=_genai_types.ThinkingConfig(thinking_budget=0),
        )
        last_err: Exception | None = None
        for attempt in range(self.max_retries + 1):
            try:
                resp = self._client.models.generate_content(
                    model=self.model_id,
                    contents=parts,
                    config=gen_config,
                )
                return (resp.text or "").strip()
            except Exception as exc:  # noqa: BLE001
                last_err = exc
                wait = self._retry_delay_for(exc, attempt)
                logger.warning(
                    "Gemini invoke failed (attempt %d/%d): %s — retrying in %.1fs",
                    attempt + 1, self.max_retries + 1, exc, wait,
                )
                time.sleep(wait)
        logger.error("Gemini invoke permanently failed: %s", last_err)
        return None

    @staticmethod
    def _retry_delay_for(exc: Exception, attempt: int) -> float:
        """Honour server-suggested retry delays for 429s.

        Gemini's RESOURCE_EXHAUSTED responses include
        ``RetryInfo.retryDelay`` (e.g. ``"9s"``); if we wait less than
        that, the next attempt fails immediately. Fall back to
        exponential backoff for non-429 errors.
        """
        msg = str(exc)
        if "RESOURCE_EXHAUSTED" in msg or "429" in msg:
            m = re.search(r"retryDelay['\"]?:\s*['\"]?([0-9.]+)s", msg)
            if m:
                return float(m.group(1)) + 1.0
            return 12.0
        return 0.5 * (2 ** attempt)
