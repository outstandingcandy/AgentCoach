"""Jersey/role recognition via Claude (Bedrock).

Subclasses :class:`BaseVLMRecognizer` — only the Bedrock-specific
payload format and the ``invoke_model`` call live here.

Uses ``us.anthropic.claude-opus-4-7`` directly as ``modelId`` (no
inference-profile call site).
"""

from __future__ import annotations

import base64
import json
import logging
import time
from typing import Any

import numpy as np

from ._vlm_base import BaseVLMRecognizer, Block

try:
    import boto3
    from botocore.config import Config as BotoConfig
except ImportError:
    boto3 = None
    BotoConfig = None

try:
    import cv2
except ImportError:
    cv2 = None

logger = logging.getLogger(__name__)


class ClaudeJerseyRecognizer(BaseVLMRecognizer):
    """Bedrock-backed recognizer using Claude Opus (multi-modal)."""

    def __init__(self, config: dict[str, Any] | None = None):
        if boto3 is None:
            raise ImportError("boto3 is required; install with `pip install boto3`")
        super().__init__(config)
        self.model_id = self.config.get("model_id", "us.anthropic.claude-opus-4-7")
        self.region = self.config.get("region", "us-east-1")

        boto_cfg = BotoConfig(
            retries={"max_attempts": 1, "mode": "standard"},
            read_timeout=60,
            connect_timeout=10,
        )
        self._client = boto3.client(
            "bedrock-runtime", region_name=self.region, config=boto_cfg,
        )
        # Token accounting: surfaces actual Bedrock spend per
        # consolidation run so the per-video cost is reportable.
        # Thread-safe — only mutated under the GIL via successful
        # invoke_model returns; no concurrency primitives needed.
        self._total_input_tokens = 0
        self._total_output_tokens = 0
        self._total_calls = 0

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
        body: dict[str, Any] = {
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": max_tokens or self.max_tokens,
            "messages": [{"role": "user", "content": content}],
        }
        if system:
            body["system"] = system
        return self._invoke_with_retry(body)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _image_block(
        self, crop: np.ndarray, *, is_wide: bool = False,
    ) -> dict[str, Any] | None:
        """Bedrock image block (base64 JPEG)."""
        if cv2 is None:
            raise ImportError("opencv is required")
        if crop is None or crop.size == 0:
            return None
        # NOTE: Claude has historically used the same cap for wide frames
        # and per-person crops (Gemini/Qwen use 1280 for wide). Keep that
        # parity to avoid changing per-call token counts here; bump if/
        # when scene understanding quality regresses.
        del is_wide
        h, w = crop.shape[:2]
        if max(h, w) > self.max_crop_dim:
            scale = self.max_crop_dim / max(h, w)
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
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": "image/jpeg",
                "data": b64,
            },
        }

    def _invoke_with_retry(self, body: dict[str, Any]) -> str | None:
        last_err: Exception | None = None
        for attempt in range(self.max_retries + 1):
            try:
                resp = self._client.invoke_model(
                    modelId=self.model_id,
                    body=json.dumps(body),
                    contentType="application/json",
                )
                payload = json.loads(resp["body"].read())
                # Tally per-call token usage so the runner can surface
                # the total Bedrock cost for the consolidation pass.
                usage = payload.get("usage") or {}
                self._total_input_tokens += int(usage.get("input_tokens", 0))
                self._total_output_tokens += int(usage.get("output_tokens", 0))
                self._total_calls += 1
                for p in payload.get("content", []):
                    if p.get("type") == "text":
                        return p["text"]
                return ""
            except Exception as exc:  # noqa: BLE001
                last_err = exc
                wait = 0.5 * (2 ** attempt)
                logger.warning(
                    "Bedrock invoke failed (attempt %d/%d): %s — retrying in %.1fs",
                    attempt + 1, self.max_retries + 1, exc, wait,
                )
                time.sleep(wait)
        logger.error("Bedrock invoke permanently failed: %s", last_err)
        return None

    def usage_stats(self) -> dict[str, int]:
        """Return per-recognizer Bedrock token tally.

        Reported by the track_consolidation runner at end-of-stage so
        each run logs how much Sonnet/Opus it spent.
        """
        return {
            "model_id": self.model_id,
            "calls": int(self._total_calls),
            "input_tokens": int(self._total_input_tokens),
            "output_tokens": int(self._total_output_tokens),
        }
