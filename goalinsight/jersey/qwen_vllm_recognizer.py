"""Jersey/role recognition via local Qwen VL served by vLLM.

Subclasses :class:`BaseVLMRecognizer` — only the OpenAI-compatible
payload format and the ``chat.completions.create`` call live here.

Prerequisites:
  1. Launch the server:  ``bash scripts/start_qwen_vllm.sh``
  2. Install the client:  ``pip install openai``

The vLLM server ignores the API key but the SDK requires any non-empty
string.
"""

from __future__ import annotations

import base64
import logging
import time
from typing import Any

import numpy as np

from ._vlm_base import BaseVLMRecognizer, Block

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None

try:
    import cv2
except ImportError:
    cv2 = None

logger = logging.getLogger(__name__)


class QwenVLLMRecognizer(BaseVLMRecognizer):
    """Local Qwen-VL model exposed via vLLM's OpenAI-compatible API."""

    def __init__(self, config: dict[str, Any] | None = None):
        if OpenAI is None:
            raise ImportError(
                "openai is required; install with `pip install openai`")
        super().__init__(config)
        self.model_id = self.config.get("model_id", "Qwen/Qwen3.6-27B-FP8")
        self.base_url = self.config.get("base_url", "http://localhost:8000/v1")
        self.timeout = float(self.config.get("timeout", 120))

        self._client = OpenAI(
            base_url=self.base_url,
            api_key=self.config.get("api_key", "EMPTY"),
            timeout=self.timeout,
        )

    # ------------------------------------------------------------------
    # BaseVLMRecognizer hook
    # ------------------------------------------------------------------

    def _call(
        self,
        blocks: list[Block],
        system: str | None = None,
        max_tokens: int | None = None,
    ) -> str | None:
        content: list[dict[str, Any]] = []
        for kind, value in blocks:
            if kind == "text":
                content.append({"type": "text", "text": value})
            else:  # "image" or "image_wide"
                blk = self._image_block(value, is_wide=(kind == "image_wide"))
                if blk is not None:
                    content.append(blk)
        if not content:
            return None
        messages: list[dict[str, Any]] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": content})
        return self._invoke_with_retry(messages, max_tokens or self.max_tokens)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _image_block(
        self, crop: np.ndarray, *, is_wide: bool = False,
    ) -> dict[str, Any] | None:
        """OpenAI-compatible image block using a base64 data URL."""
        if cv2 is None:
            raise ImportError("opencv is required")
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
        ok, buf = cv2.imencode(
            ".jpg", crop, [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality],
        )
        if not ok:
            return None
        b64 = base64.b64encode(buf.tobytes()).decode("ascii")
        return {
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{b64}"},
        }

    def _invoke_with_retry(
        self, messages: list[dict[str, Any]], max_tokens: int,
    ) -> str | None:
        # Qwen3 "thinking" chain-of-thought is on by default and eats into
        # our max_tokens budget. Disable it so the model answers directly.
        extra_body: dict[str, Any] = {}
        if self.config.get("disable_thinking", True):
            extra_body["chat_template_kwargs"] = {"enable_thinking": False}
        last_err: Exception | None = None
        for attempt in range(self.max_retries + 1):
            try:
                resp = self._client.chat.completions.create(
                    model=self.model_id,
                    messages=messages,
                    max_tokens=max_tokens,
                    temperature=0.0,
                    extra_body=extra_body or None,
                )
                return (resp.choices[0].message.content or "").strip()
            except Exception as exc:  # noqa: BLE001
                last_err = exc
                wait = 0.5 * (2 ** attempt)
                logger.warning(
                    "vLLM invoke failed (attempt %d/%d): %s — retrying in %.1fs",
                    attempt + 1, self.max_retries + 1, exc, wait,
                )
                time.sleep(wait)
        logger.error("vLLM invoke permanently failed: %s", last_err)
        return None
