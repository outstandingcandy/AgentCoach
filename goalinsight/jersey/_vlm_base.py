"""Shared orchestration for VLM-backed jersey/role recognizers.

Claude (Bedrock), Gemini (AI Studio), and Qwen (local vLLM) speak the
same structured protocol — ``recognize``, ``recognize_batch``,
``describe_scene``, ``recognize_team_seeds``, ``recognize_multi`` — and
build their request bodies by interleaving the same text+image content
in the same order.  The only backend-specific bits are:

  1. how an image becomes a request payload element (Bedrock dict vs
     OpenAI ``image_url`` dict vs PIL image),
  2. how text/image elements are wrapped into the final request
     (system role vs in-line system text), and
  3. which SDK call actually fires the request.

All three are folded into the abstract ``_call`` hook.  Everything else
lives here so each backend file is just ``__init__`` + ``_call``.
"""

from __future__ import annotations

import abc
import logging
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import numpy as np

from ..interfaces import BaseJerseyRecognizer
from ._vlm_common import (
    BATCHED_OCR_PROMPT,
    BATCHED_OCR_PROMPT_IMAGE_LIST,
    MULTI_SYSTEM,
    SCENE_TASK_TEXT,
    SINGLE_OCR_PROMPT,
    SINGLE_PROMPT,
    TEAM_SEEDS_INTRO,
    TEAM_SEEDS_TASK_TEXT,
    aggregate_crop_verdicts,
    build_multi_user_text,
    build_scene_context_block,
    burn_id_banner,
    parse_batched_ocr_response,
    parse_multi_response,
    parse_scene_response,
    parse_single_ocr_response,
    parse_single_response,
    parse_team_seeds_response,
)

logger = logging.getLogger(__name__)


# A single content element to send to the VLM.
# - ("text", str)            : a text snippet
# - ("image", np.ndarray)    : a per-person crop (downscaled to max_crop_dim)
# - ("image_wide", np.ndarray): a full-frame image (downscaled to a larger cap)
Block = tuple[str, Any]


class BaseVLMRecognizer(BaseJerseyRecognizer):
    """Common scaffolding for VLM jersey/role recognizers."""

    def __init__(self, config: dict[str, Any] | None = None):
        self.config = config or {}
        self.max_retries = int(self.config.get("max_retries", 2))
        # 128 was just barely enough for a 7-field JSON reply; longer
        # reasoning strings or multi-line target_kit_description got
        # truncated mid-sentence and hit the parser's "no json" fallback,
        # so the track came back as role/team=unknown despite the model
        # having actually identified the kit. 256 leaves headroom and
        # is still cheap (each call is ~$0.001).
        self.max_tokens = int(self.config.get("max_tokens", 256))
        self.jpeg_quality = int(self.config.get("jpeg_quality", 85))
        self.max_crop_dim = int(self.config.get("max_crop_dim", 384))
        self.max_concurrency = int(self.config.get("max_concurrency", 4))
        # OCR backend for Phase 2 (per-crop jersey-number reading).
        #   - "llm" (default): one ``SINGLE_OCR_PROMPT`` LLM call per crop.
        #     Accurate but expensive ($ + rate limit) so the consolidator
        #     normally caps at ~30 crops/track.
        #   - "rapidocr": local RapidOCR / PaddleOCR ONNX. Free + fast,
        #     can OCR every frame without rate-limit concerns. Less
        #     accurate (Chinese model confuses some Latin digits with
        #     Han characters; a small lookalike mapping recovers the
        #     common cases). Phase 1 (role / team / scene) still goes
        #     through the LLM regardless.
        self.ocr_backend = str(
            self.config.get("ocr_backend", "llm")).lower()
        self._rapid_reader = None  # lazy on first call

    # ------------------------------------------------------------------
    # BaseJerseyRecognizer surface
    # ------------------------------------------------------------------

    def recognize(self, crop: np.ndarray) -> tuple[int | None, float]:
        if crop is None or crop.size == 0:
            return (None, 0.0)
        text = self._call(
            [("text", SINGLE_PROMPT), ("image", crop)],
            system=None,
        )
        if text is None:
            return (None, 0.0)
        num = parse_single_response(text)
        return (num, 0.7 if num is not None else 0.0)

    def recognize_batch(
        self, crops: list[np.ndarray],
    ) -> list[tuple[int | None, float]]:
        results: list[tuple[int | None, float]] = [(None, 0.0)] * len(crops)
        with ThreadPoolExecutor(max_workers=self.max_concurrency) as pool:
            futs = {pool.submit(self.recognize, c): i for i, c in enumerate(crops)}
            for fut in futs:
                idx = futs[fut]
                try:
                    results[idx] = fut.result()
                except Exception as exc:  # noqa: BLE001
                    logger.warning("recognize failed for crop %d: %s", idx, exc)
        return results

    # ------------------------------------------------------------------
    # Multi-image endpoints
    # ------------------------------------------------------------------

    def describe_scene(
        self,
        wide_frame: np.ndarray,
        person_crops: list[tuple[int, np.ndarray]],
    ) -> dict[str, Any]:
        blocks: list[Block] = [
            ("text", "=== Wide-view frame of the match ==="),
            ("image_wide", wide_frame),
            ("text",
             "\n=== Individual person crops from the same match "
             "(each crop has its person id burned onto the top banner) ==="),
        ]
        for tid, crop in person_crops:
            blocks.append(("text", f"Person #{tid}:"))
            blocks.append(("image", burn_id_banner(crop, tid)))
        blocks.append(("text", SCENE_TASK_TEXT))
        raw = self._call(
            blocks, system=None,
            max_tokens=max(self.max_tokens, 800),
        )
        if raw is None:
            return {}
        return parse_scene_response(raw)

    def recognize_team_seeds(
        self,
        crops: list[tuple[int, list[np.ndarray]]] | list[tuple[int, np.ndarray]],
    ) -> dict[int, str]:
        if not crops:
            return {}
        blocks: list[Block] = [("text", TEAM_SEEDS_INTRO)]
        n_total = 0
        for tid, crop_or_list in crops:
            crop_list = (
                crop_or_list if isinstance(crop_or_list, list) else [crop_or_list]
            )
            if not crop_list:
                continue
            blocks.append(("text", f"Person #{tid}:"))
            for crop in crop_list:
                blocks.append(("image", crop))
            n_total += 1
        blocks.append(("text", TEAM_SEEDS_TASK_TEXT))
        raw = self._call(
            blocks, system=None,
            max_tokens=max(self.max_tokens, 60 * n_total + 200),
        )
        if raw is None:
            return {}
        return parse_team_seeds_response(raw)

    def recognize_multi(
        self,
        crops: list[np.ndarray],
        position_label: str = "unknown",
        team_exemplars: dict[str, list[np.ndarray]] | None = None,
        scene_description: dict[str, str] | None = None,
        movement_description: str = "",
    ) -> tuple[
        int | None, float, str, str, str,
        list[tuple[int, float]], list[dict[str, Any]],
    ]:
        """Two-phase recognition.

        Phase 1 (1 LLM call): role/team decision over a representative
        subset of crops + scene + team exemplars. The LLM returns
        ``role / team / target_kit_description / reasoning`` plus a
        legacy ``crop_verdicts`` field that we ignore here.

        Phase 2 (N LLM calls in parallel, only if role=='player'):
        per-crop OCR. Each call sees ONE crop and returns
        ``(reading, crop_confidence, visible_digits)``. Python
        aggregates via :func:`aggregate_crop_verdicts`.

        Splitting the work means a 200-crop track (long lifespan) gets
        200 small parallel OCR calls instead of one huge prompt that
        the model can't reason carefully over.
        """
        if not crops:
            return (None, 0.0, "no crops", "unknown", "unknown", [], [])

        # ---- Phase 1: role/team decision -----------------------------
        # Use a representative subset (large bbox + sharp) to keep the
        # decision call cheap. The full crop list still feeds the per-
        # crop OCR phase.
        decision_n = min(len(crops), int(self.config.get(
            "decision_subset_size", 8)))
        # Crops are already ordered by the sampler (frame_id); take an
        # evenly-spaced subset to cover the track's duration.
        if decision_n >= len(crops):
            decision_crops = crops
        else:
            step = len(crops) / decision_n
            decision_crops = [crops[int(i * step)] for i in range(decision_n)]

        blocks: list[Block] = []
        ctx = build_scene_context_block(scene_description)
        if ctx:
            blocks.append(("text", ctx))
        if team_exemplars:
            for team_label in ("team_A", "team_B"):
                exemplars = team_exemplars.get(team_label, []) or []
                if not exemplars:
                    continue
                blocks.append((
                    "text",
                    f"=== {team_label} match kit — reference images "
                    f"({len(exemplars)} crops of different {team_label} "
                    "players in the same match) ===",
                ))
                for i, ex in enumerate(exemplars):
                    blocks.append(("text", f"{team_label} ref {i + 1}:"))
                    blocks.append(("image", ex))
            blocks.append((
                "text",
                "=== Target person — decide role / team below "
                "based on the reference images above ===",
            ))
        for i, crop in enumerate(decision_crops):
            blocks.append(("text", f"Image {i + 1}:"))
            blocks.append(("image", crop))
        blocks.append((
            "text",
            build_multi_user_text(
                len(decision_crops), position_label, movement_description),
        ))

        decision_budget = max(self.max_tokens, 80 + 60 * len(decision_crops))
        raw = self._call(
            blocks, system=MULTI_SYSTEM, max_tokens=decision_budget,
        )
        if raw is None:
            return (None, 0.0, "api error", "unknown", "unknown", [], [])
        # ``parse_multi_response`` returns the LLM's own crop_verdicts;
        # we ignore those (the decision-call subset is too small for
        # robust voting) and let Phase 2 redo the OCR over ALL crops.
        _legacy_num, _legacy_conf, reasoning, role, team, _legacy_breakdown = (
            parse_multi_response(raw)
        )

        # ---- Phase 2: per-crop OCR (only for players) ---------------
        if role != "player":
            return (None, 0.0, reasoning, role, team, [], [])

        verdicts = self._ocr_crops_parallel(crops)
        num, conf, breakdown = aggregate_crop_verdicts(verdicts)
        return num, conf, reasoning, role, team, breakdown, verdicts

    def _ocr_crops_parallel(
        self,
        crops: list[np.ndarray],
    ) -> list[dict[str, Any]]:
        """Read jersey numbers across ``crops``.

        For the LLM backend, we montage crops in groups of
        ``ocr_batch_size`` (default 16) into a grid and ask the model
        to vote across the montage and return ONE number per call.
        That number becomes one "verdict" for the whole batch, so a
        300-crop track produces 300/16 ≈ 19 verdicts (one per batch)
        instead of 300, while still letting the higher-level
        ``aggregate_crop_verdicts`` re-vote across batches in case the
        model commits to different numbers in different batches.
        Groups are processed in parallel up to
        ``ocr_per_track_concurrency`` (defaults to
        ``max_concurrency``).

        For RapidOCR (local CPU) we keep the per-crop path — there's
        no API cost to amortise and montaging would only blur digits.

        Returns a list of verdict dicts (one per LLM call for the LLM
        path, one per crop for the RapidOCR path) — they share the
        ``{reading, crop_confidence, visible_digits}`` shape so the
        caller can pass either through ``aggregate_crop_verdicts``.
        """
        n = len(crops)
        per_track_workers = int(self.config.get(
            "ocr_per_track_concurrency", self.max_concurrency))

        if self.ocr_backend == "rapidocr":
            verdicts: list[dict[str, Any]] = [{}] * n
            with ThreadPoolExecutor(max_workers=per_track_workers) as pool:
                futs = [pool.submit(self._ocr_one_rapidocr, crops[i])
                        for i in range(n)]
                for i, fut in enumerate(futs):
                    try:
                        verdicts[i] = fut.result()
                    except Exception as exc:  # noqa: BLE001
                        logger.warning("OCR failed for crop %d: %s", i, exc)
                        verdicts[i] = {
                            "reading": None,
                            "crop_confidence": 0.0,
                            "visible_digits": f"error: {exc}",
                        }
            return verdicts

        # LLM path — group crops into montages and vote per-montage.
        # Each verdict carries the montage image (``_montage``) and the
        # source-crop indices it covered (``_crop_indices``) so the
        # caller can persist the *exact* image that fed the LLM next to
        # its reading. These under-prefixed keys are stripped before
        # the verdict is fed to ``aggregate_crop_verdicts``.
        batch_size = max(1, int(self.config.get("ocr_batch_size", 16)))
        groups = [list(range(i, min(i + batch_size, n)))
                  for i in range(0, n, batch_size)]

        def _run_group(idxs: list[int]) -> dict[str, Any]:
            batch = [crops[i] for i in idxs]
            try:
                verdict = self._ocr_batch_llm(batch)
            except Exception as exc:  # noqa: BLE001
                logger.warning("Batched OCR failed (n=%d): %s", len(idxs), exc)
                verdict = {
                    "reading": None, "crop_confidence": 0.0,
                    "visible_digits": f"error: {exc}",
                }
            # Always attach the montage + indices; even on failure it's
            # useful context for debugging the page.
            verdict["_montage"] = _build_crop_montage(batch)
            verdict["_crop_indices"] = list(idxs)
            return verdict

        batch_verdicts: list[dict[str, Any]] = [{} for _ in groups]
        with ThreadPoolExecutor(max_workers=per_track_workers) as pool:
            futs = {pool.submit(_run_group, g): gi
                    for gi, g in enumerate(groups)}
            for fut, gi in futs.items():
                batch_verdicts[gi] = fut.result()
        return batch_verdicts

    def _ocr_one_llm(self, crop: np.ndarray) -> dict[str, Any]:
        raw = self._call(
            [("text", SINGLE_OCR_PROMPT), ("image", crop)],
            system=None,
            max_tokens=80,
        )
        if raw is None:
            return {
                "reading": None,
                "crop_confidence": 0.0,
                "visible_digits": "api error",
            }
        reading, cc, visible = parse_single_ocr_response(raw)
        return {
            "reading": reading,
            "crop_confidence": cc,
            "visible_digits": visible,
        }

    def _ocr_batch_llm(
        self, crops: list[np.ndarray],
    ) -> dict[str, Any]:
        """Single LLM call over up to ``ocr_batch_size`` crops.

        Two transport modes (``ocr_image_list`` flag):
        - ``False`` (default): tile crops into a montage grid and send
          ONE image. Cheap (1× image-token minimum per call) but a
          tightly cropped player + a teammate standing nearby end up
          in the same 160×160 cell, which the model can't always
          isolate.
        - ``True``: send N independent images in the user message. Each
          image is just one bbox crop with the target centred, so
          adjacent players appear only as bbox-edge slivers — the
          model never has to disambiguate "two backs in one cell".
          Costs N× the image-token minimum (~1568 tokens per image
          on Anthropic), so pair with a smaller ``ocr_batch_size``
          (e.g. 4) and a larger ``frame_stride`` to keep total token
          spend reasonable.

        Returns ``{reading, crop_confidence, visible_digits}``;
        ``crop_confidence`` is named that way (not ``confidence``)
        so the verdict is interchangeable with single-crop verdicts
        when fed to ``aggregate_crop_verdicts``.
        """
        n = len(crops)
        if n == 0:
            return {"reading": None, "crop_confidence": 0.0,
                    "visible_digits": "empty batch"}
        if n == 1:
            return self._ocr_one_llm(crops[0])

        # 1024 leaves headroom for chain-of-thought-style models
        # (Qwen3 VL emits a paragraph of reasoning before the JSON
        # by default; 200 truncated mid-thought and the JSON parser
        # got an empty reply). Claude / Gemini still respond in <100
        # tokens so the larger ceiling has no cost there.
        max_tok = max(self.max_tokens, 1024)
        if bool(self.config.get("ocr_image_list", False)):
            # Image-list mode: prompt + each crop as a separate image.
            prompt = BATCHED_OCR_PROMPT_IMAGE_LIST.format(N=n)
            blocks: list[Block] = [("text", prompt)]
            for c in crops:
                blocks.append(("image", c))
            raw = self._call(blocks, system=None, max_tokens=max_tok)
        else:
            prompt = BATCHED_OCR_PROMPT.format(N=n)
            montage = _build_crop_montage(crops)
            raw = self._call(
                [("text", prompt), ("image", montage)],
                system=None, max_tokens=max_tok,
            )
        if raw is None:
            return {"reading": None, "crop_confidence": 0.0,
                    "visible_digits": "api error"}
        reading, conf, visible = parse_batched_ocr_response(raw)
        return {"reading": reading, "crop_confidence": conf,
                "visible_digits": visible}

    def _ocr_one_rapidocr(self, crop: np.ndarray) -> dict[str, Any]:
        if self._rapid_reader is None:
            from .rapidocr import RapidOCRJerseyReader  # noqa: PLC0415
            self._rapid_reader = RapidOCRJerseyReader(
                config=self.config.get("rapidocr", {}))
        reading, cc, visible = self._rapid_reader.ocr_crop(crop)
        return {
            "reading": reading,
            "crop_confidence": cc,
            "visible_digits": visible,
        }

    # ------------------------------------------------------------------
    # Subclass hooks
    # ------------------------------------------------------------------

    @abc.abstractmethod
    def _call(
        self,
        blocks: list[Block],
        system: str | None = None,
        max_tokens: int | None = None,
    ) -> str | None:
        """Translate ``blocks`` to a backend-native request and execute.

        Returns the raw response text (str) or ``None`` on failure.
        ``max_tokens`` defaults to ``self.max_tokens`` when unset.
        """


def _build_crop_montage(
    crops: list[np.ndarray],
    cell_size: tuple[int, int] = (160, 160),
) -> np.ndarray:
    """Tile ``crops`` into a square-ish row-major grid for batched OCR.

    Each crop is letterboxed (preserve aspect, pad black) into a
    uniform ``cell_size`` so the model sees a predictable layout. The
    grid uses ``cols = ceil(sqrt(N))`` rows then fills row-major so
    the prompt's "row-major order" instruction matches the visual.
    Empty cells (when N isn't a perfect square) stay black.
    """
    import cv2 as _cv2  # local import — avoid pulling cv2 at module load

    n = len(crops)
    if n == 0:
        return np.zeros((cell_size[1], cell_size[0], 3), dtype=np.uint8)
    cols = int(np.ceil(np.sqrt(n)))
    rows = int(np.ceil(n / cols))
    cw, ch = cell_size
    canvas = np.zeros((rows * ch, cols * cw, 3), dtype=np.uint8)
    for i, crop in enumerate(crops):
        r, c = divmod(i, cols)
        h, w = crop.shape[:2]
        # Letterbox into cell_size preserving aspect.
        scale = min(cw / max(w, 1), ch / max(h, 1))
        nh, nw = max(1, int(round(h * scale))), max(1, int(round(w * scale)))
        resized = _cv2.resize(crop, (nw, nh), interpolation=_cv2.INTER_AREA)
        if resized.ndim == 2:
            resized = _cv2.cvtColor(resized, _cv2.COLOR_GRAY2BGR)
        elif resized.shape[2] == 4:
            resized = _cv2.cvtColor(resized, _cv2.COLOR_BGRA2BGR)
        x = c * cw + (cw - nw) // 2
        y = r * ch + (ch - nh) // 2
        canvas[y:y + nh, x:x + nw] = resized
        # Thin separator/border so the model can see cell boundaries.
        _cv2.rectangle(
            canvas,
            (c * cw, r * ch),
            (c * cw + cw - 1, r * ch + ch - 1),
            (255, 255, 255),
            1,
        )
    return canvas
