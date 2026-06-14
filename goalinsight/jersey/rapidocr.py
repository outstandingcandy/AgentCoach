"""Local RapidOCR back end for jersey-number reading (Phase 2 only).

RapidOCR runs the full PaddleOCR pipeline (det + cls + rec) via
ONNX Runtime on CPU/GPU. It's free, fast (~30 ms / crop), and
processes hundreds of crops per track without hitting API rate limits
— so the consolidator can OCR every frame in a track instead of
sub-sampling for cost.

Caveats:
- Single-digit / partial numbers ("0" with the tens digit occluded)
  are read as the visible digit. Caller-side filtering (ratio of
  full-vs-partial reads) must reject those if you want to avoid
  guessing 10 / 20 / 30. We surface the raw single-digit reading and
  rely on per-crop voting to wash it out.
- The bundled Chinese model confuses "8" with the Han character "八"
  / "本" / "公". We post-map a small set of common confusions back to
  digits before regex-extracting 1–2 digit jersey numbers.

Used as Phase 2 for ``recognize_multi`` when the consolidator config
sets ``jersey.ocr_backend: rapidocr``. Phase 1 (role / team decision
+ scene context) still goes through the LLM because OCR cannot
classify role / team.
"""

from __future__ import annotations

import logging
import re
from typing import Any

import cv2
import numpy as np

logger = logging.getLogger(__name__)


# Han characters / lookalikes the Chinese RapidOCR model emits when it
# sees Latin digits on a blurry / low-contrast jersey. Mapping is
# conservative — only entries we have *seen* misfire on actual player
# crops. Anything not in this table is dropped by the regex below.
_LOOKALIKE_TO_DIGIT = {
    "一": "1", "二": "2", "三": "3", "四": "4", "五": "5",
    "六": "6", "七": "7", "八": "8", "九": "9", "〇": "0", "零": "0",
    "公": "8", "本": "8",  # observed on kids_clip jersey #88
    "O": "0", "o": "0", "Q": "0", "D": "0",
    "l": "1", "I": "1", "i": "1",
    "Z": "2", "z": "2",
    "B": "8", "S": "5", "s": "5",
    "T": "7",
}


def _normalise(text: str) -> str:
    """Map common OCR confusions to digits and drop everything else."""
    out: list[str] = []
    for ch in text or "":
        if ch.isdigit():
            out.append(ch)
        elif ch in _LOOKALIKE_TO_DIGIT:
            out.append(_LOOKALIKE_TO_DIGIT[ch])
    return "".join(out)


class RapidOCRJerseyReader:
    """Thin wrapper around ``rapidocr_onnxruntime.RapidOCR`` for per-crop
    jersey-number reads.

    The class is intentionally stateless beyond the ONNX session — one
    instance is shared across all crops of one track and across tracks
    if the caller wants to (RapidOCR is thread-safe enough that fan-out
    via ThreadPoolExecutor is fine; see ``_vlm_base._ocr_crops_parallel``
    for the calling site).
    """

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        config = config or {}
        # 4× upscale by default — kids' jerseys at our render distance
        # are typically 30-60 px tall pre-upscale; PaddleOCR rec wants
        # ~32 px text height to be confident.
        self.upscale = float(config.get("upscale", 4.0))
        self.min_confidence = float(config.get("min_confidence", 0.5))
        # Lazy import so the dep stays optional.
        try:
            from rapidocr_onnxruntime import RapidOCR  # noqa: PLC0415
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError(
                "rapidocr_onnxruntime is not installed. "
                "Install with `pip install rapidocr-onnxruntime` "
                "to use the rapidocr jersey backend."
            ) from exc
        self._ocr = RapidOCR()

    def ocr_crop(
        self,
        crop: np.ndarray,
    ) -> tuple[int | None, float, str]:
        """Read the jersey number off a single crop.

        Returns ``(reading, crop_confidence, visible_digits)`` matching
        :func:`goalinsight.jersey._vlm_common.parse_single_ocr_response`'s
        contract so the existing per-crop voting aggregator works
        unchanged.

        - ``reading``: 1-99 if a clean 1-2 digit token was extracted,
          else ``None``.
        - ``crop_confidence``: the underlying OCR rec score, ∈ [0, 1].
          0 when ``reading`` is ``None``.
        - ``visible_digits``: short factual description for debugging;
          mirrors what the LLM-OCR path emits.
        """
        if crop is None or crop.size == 0:
            return None, 0.0, "empty crop"

        img = crop
        if self.upscale and self.upscale != 1.0:
            img = cv2.resize(
                img, None,
                fx=self.upscale, fy=self.upscale,
                interpolation=cv2.INTER_CUBIC,
            )

        try:
            result, _ = self._ocr(img)
        except Exception as exc:  # noqa: BLE001
            logger.warning("RapidOCR call failed: %s", exc)
            return None, 0.0, f"ocr error: {exc}"

        if not result:
            return None, 0.0, "no text detected"

        # result entries: [box_4pts, text, rec_score]. Vote across all
        # detected text fragments by mapping each to digits and keeping
        # the highest-confidence 1–2 digit token in [1, 99].
        best_num: int | None = None
        best_score = 0.0
        best_visible = ""
        all_seen: list[str] = []
        for entry in result:
            try:
                _box, text, score = entry[0], entry[1], float(entry[2])
            except (IndexError, TypeError, ValueError):
                continue
            if score < self.min_confidence:
                continue
            digits = _normalise(text)
            all_seen.append(text if not digits else digits)
            if not digits:
                continue
            m = re.fullmatch(r"\d{1,2}", digits)
            if not m:
                # Multi-digit garbage (e.g. timestamp, jersey + sock):
                # skip rather than guess.
                continue
            num = int(m.group())
            if not 1 <= num <= 99:
                continue
            if score > best_score:
                best_score = score
                best_num = num
                best_visible = digits
        if best_num is not None:
            return best_num, max(0.0, min(1.0, best_score)), best_visible
        # Got OCR hits but none parsed as a clean jersey number.
        sample = ",".join(all_seen[:3])[:60]
        return None, 0.0, f"no digit token (saw: {sample})" if sample else "no digit token"
