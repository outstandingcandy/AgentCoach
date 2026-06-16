"""Jersey/role recognition via a local Qwen VL model exposed through
vLLM's OpenAI-compatible API.

Subclasses :class:`BaseVLMRecognizer` — only the OpenAI chat-completion
payload format lives here. Reuses every prompt + parser the Bedrock
(Claude) and Gemini backends use, including the BATCHED_OCR_PROMPT
montage flow and the single-digit defensive guard.

Launch the server with::

    bash scripts/start_qwen_vllm.sh   # serves on :8000 by default

Then point the config at it::

    jersey_recognition:
      backend: qwen_vl
      mode: api
      api:
        base_url: "http://localhost:8000/v1"
        model: "Qwen/Qwen3.6-27B-FP8"
"""

from __future__ import annotations

import base64
import logging
import time
from typing import Any

import numpy as np

from ._vlm_base import BaseVLMRecognizer, Block

try:
    import requests
except ImportError:
    requests = None

try:
    import cv2
except ImportError:
    cv2 = None

logger = logging.getLogger(__name__)


class QwenVLMRecognizer(BaseVLMRecognizer):
    """Qwen VL backend reachable via OpenAI-compatible HTTP API.

    All prompt / parser / batching machinery is inherited from
    :class:`BaseVLMRecognizer`. The only backend-specific bit is
    :meth:`_call`, which packages the ``Block`` list into an OpenAI
    chat-completion request body and POSTs it to the vLLM server.
    """

    def __init__(self, config: dict[str, Any] | None = None):
        if requests is None:
            raise ImportError("requests is required for the Qwen VLM backend")
        super().__init__(config)
        # Two layouts are accepted so this works whether the config
        # came from the top-level ``jersey_recognition`` block (api
        # nested) or directly from a flattened sub-dict the factory
        # built.
        api_cfg = self.config.get("api") or {}
        self.base_url = (
            api_cfg.get("base_url")
            or self.config.get("base_url")
            or "http://localhost:8000/v1"
        ).rstrip("/")
        self.model_id = (
            api_cfg.get("model")
            or self.config.get("model")
            or self.config.get("model_id")
            or "Qwen/Qwen3.6-27B-FP8"
        )
        self.timeout = float(
            api_cfg.get("timeout", self.config.get("timeout", 120.0))
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
        content = self._blocks_to_openai_content(blocks)
        if not content:
            return None
        messages: list[dict[str, Any]] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": content})

        payload = {
            "model": self.model_id,
            "messages": messages,
            "temperature": 0.0,
            "max_tokens": int(max_tokens or self.max_tokens),
        }
        return self._post_with_retry(payload)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _blocks_to_openai_content(
        self, blocks: list[Block],
    ) -> list[dict[str, Any]]:
        """Translate the project's Block format into OpenAI chat content."""
        content: list[dict[str, Any]] = []
        for kind, value in blocks:
            if kind == "text":
                content.append({"type": "text", "text": value})
            else:  # "image" or "image_wide"
                b64 = self._encode_crop(value, is_wide=(kind == "image_wide"))
                if b64 is None:
                    continue
                content.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{b64}"},
                })
        return content

    def _encode_crop(
        self, crop: np.ndarray, *, is_wide: bool = False,
    ) -> str | None:
        if cv2 is None:
            raise ImportError("opencv-python is required")
        if crop is None or getattr(crop, "size", 0) == 0:
            return None
        h, w = crop.shape[:2]
        max_dim = 1280 if is_wide else self.max_crop_dim
        if max(h, w) > max_dim:
            scale = max_dim / max(h, w)
            crop = cv2.resize(
                crop, (int(w * scale), int(h * scale)),
                interpolation=cv2.INTER_AREA,
            )
        ok, buf = cv2.imencode(".jpg", crop, [cv2.IMWRITE_JPEG_QUALITY, 88])
        if not ok:
            return None
        return base64.b64encode(buf).decode("ascii")

    def _post_with_retry(self, payload: dict[str, Any]) -> str | None:
        url = f"{self.base_url}/chat/completions"
        last_err: Exception | None = None
        for attempt in range(self.max_retries + 1):
            try:
                resp = requests.post(
                    url, json=payload, timeout=self.timeout,
                )
                resp.raise_for_status()
                data = resp.json()
                msg = (
                    data.get("choices", [{}])[0]
                    .get("message", {})
                    .get("content", "")
                )
                return (msg or "").strip()
            except Exception as exc:  # noqa: BLE001
                last_err = exc
                wait = self._retry_delay_for(exc, attempt)
                logger.warning(
                    "Qwen VL call failed (attempt %d/%d): %s — sleeping %.1fs",
                    attempt + 1, self.max_retries + 1, exc, wait,
                )
                if attempt >= self.max_retries:
                    break
                time.sleep(wait)
        logger.error("Qwen VL call gave up after %d attempts: %s",
                     self.max_retries + 1, last_err)
        return None
