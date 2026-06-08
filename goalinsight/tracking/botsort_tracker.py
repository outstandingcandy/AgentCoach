"""BoT-SORT adapter — wraps Ultralytics' built-in BoT-SORT into the same
``update / reset / get_track_features`` interface as :class:`StrongSORTTracker`
so the orchestrator can swap backends with one config flag.

Why a dedicated wrapper instead of using Ultralytics' ``model.track()``
high-level API: the orchestrator already runs YOLO inference itself
(in batches, with a separate ReID extractor), so we only need to feed
detections + embeddings into a tracker. ``BOTSORT.update()`` accepts an
arbitrary "results" object as long as it exposes ``conf``, ``xyxy``,
``xywh``, ``cls``, ``__len__``, and boolean indexing — :class:`_DetResults`
below is a thin numpy-backed shim.

BoT-SORT vs StrongSORT for this project:

- StrongSORT (custom impl in this repo): Kalman + IoU + ReID cosine.
- BoT-SORT: same backbone but adds **camera motion compensation** via
  ECC / sparse optical flow ("gmc"), which compensates for moving /
  panning cameras before running Kalman gating. On Veo / phone-tripod
  amateur footage where the operator pans, this measurably reduces id
  switches when the camera moves faster than the player.

The wrapper exposes the same API surface the orchestrator already uses
(see ``orchestrator.py``); switching is via
``tracking.backend: "botsort"`` in YAML config.
"""

from __future__ import annotations

from typing import Any

import numpy as np

try:
    from ultralytics.trackers.bot_sort import BOTSORT
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "BoT-SORT backend requires ``ultralytics>=8`` (already a project "
        "dep). The import failed; check your venv."
    ) from exc


class _DetResults:
    """Numpy-backed shim that quacks enough like ``ultralytics.engine.results.Boxes``
    for ``BOTSORT.update / init_track`` to consume.

    Implements only the attributes those code paths read: ``conf``,
    ``cls``, ``xyxy``, ``xywh``, ``__len__``, and boolean indexing /
    masking.
    """

    def __init__(
        self,
        xyxy: np.ndarray,
        conf: np.ndarray,
        cls: np.ndarray,
    ) -> None:
        self.xyxy = xyxy.astype(np.float32) if xyxy.size else xyxy
        self.conf = conf.astype(np.float32)
        self.cls = cls.astype(np.float32)

    def __len__(self) -> int:
        return len(self.conf)

    def __getitem__(self, key: Any) -> "_DetResults":
        return _DetResults(self.xyxy[key], self.conf[key], self.cls[key])

    @property
    def xywh(self) -> np.ndarray:
        if self.xyxy.size == 0:
            return np.zeros((0, 4), dtype=np.float32)
        x1, y1, x2, y2 = self.xyxy.T
        return np.stack([(x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1], axis=1)


class _Args:
    """Plain holder for BoT-SORT's ``args`` namespace — the upstream
    code uses ``args.<field>`` attribute access throughout.
    """

    def __init__(self, **kw: Any) -> None:
        for k, v in kw.items():
            setattr(self, k, v)


class BoTSORTTracker:
    """Thin facade matching :class:`StrongSORTTracker`'s API.

    Maintained methods:

    - :meth:`update(detections, embeddings)` — main per-frame entry point
    - :meth:`reset()`
    - :meth:`get_track_features()` — returns ``{tid: smooth_feature}``
      so downstream consolidation can reuse OSNet centroids
    """

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        cfg = config or {}
        # Map our config keys onto Ultralytics' BoT-SORT args. Defaults
        # mirror upstream botsort.yaml; override per-key from YAML when
        # the user has tuned them.
        args = _Args(
            tracker_type="botsort",
            track_high_thresh=cfg.get("track_high_thresh", 0.25),
            track_low_thresh=cfg.get("track_low_thresh", 0.1),
            new_track_thresh=cfg.get("new_track_thresh", 0.25),
            track_buffer=cfg.get("max_age", 30),
            match_thresh=cfg.get("match_thresh", 0.8),
            fuse_score=cfg.get("fuse_score", True),
            # Default OFF: GMC needs the raw frame each call, which the
            # current orchestrator doesn't pass into update(). Set
            # tracking.botsort.gmc_method to "sparseOptFlow"/"orb"/etc
            # AND wire frame data through orchestrator.update(img=...)
            # to enable camera motion compensation.
            gmc_method=cfg.get("gmc_method", "none"),
            proximity_thresh=cfg.get("proximity_thresh", 0.5),
            appearance_thresh=cfg.get(
                "appearance_thresh",
                # Match StrongSORT's cosine_distance default when
                # users didn't override — 1 - 0.3 = 0.7.
                1.0 - cfg.get("max_cosine_distance", 0.3),
            ),
            # We always supply external embeddings, so 'auto' is fine —
            # BOTSORT only triggers internal ReID when 'with_reid' is True.
            with_reid=False,
            model="auto",
        )
        frame_rate = int(round(cfg.get("frame_rate", 30)))
        self._tracker = BOTSORT(args=args, frame_rate=frame_rate)
        self._args = args

        # Per-tid running ReID centroid, exponentially smoothed.
        self._features: dict[int, np.ndarray] = {}
        self._feature_alpha = float(cfg.get("feature_alpha", 0.9))

        # img_w / img_h are set by the orchestrator after construction;
        # BoT-SORT's GMC reads frame size from the image we hand it
        # directly, so these are unused but kept for interface parity.
        self.img_w: int | None = None
        self.img_h: int | None = None

    # ------------------------------------------------------------------
    # Public API matching StrongSORTTracker
    # ------------------------------------------------------------------
    def reset(self) -> None:
        """Drop all tracks and start over (e.g. between videos)."""
        self.__init__({  # type: ignore[misc]
            "track_high_thresh": self._args.track_high_thresh,
            "track_low_thresh": self._args.track_low_thresh,
            "new_track_thresh": self._args.new_track_thresh,
            "max_age": self._args.track_buffer,
            "match_thresh": self._args.match_thresh,
            "fuse_score": self._args.fuse_score,
            "gmc_method": self._args.gmc_method,
            "proximity_thresh": self._args.proximity_thresh,
            "appearance_thresh": self._args.appearance_thresh,
            "feature_alpha": self._feature_alpha,
        })

    def update(
        self,
        detections: list[dict[str, Any]],
        embeddings: np.ndarray | None = None,
        img: np.ndarray | None = None,
    ) -> list[dict[str, Any]]:
        """Run one tracker step and return per-track dicts.

        ``img`` enables GMC; the orchestrator should pass the current
        frame when available. When ``img`` is None GMC is skipped (the
        upstream code wraps gmc.apply in try/except, but skipping the
        call entirely is faster).

        Output schema matches StrongSORTTracker.update() so the
        orchestrator's downstream code (pitch projection, embedding
        bookkeeping, vis) works unchanged.
        """
        if not detections:
            xyxy = np.zeros((0, 4), dtype=np.float32)
            conf = np.zeros((0,), dtype=np.float32)
            cls = np.zeros((0,), dtype=np.float32)
        else:
            xyxy = np.asarray(
                [d["bbox"] for d in detections], dtype=np.float32
            )
            conf = np.asarray(
                [float(d.get("confidence", 1.0)) for d in detections],
                dtype=np.float32,
            )
            cls = np.asarray(
                [int(d.get("class", 0)) for d in detections],
                dtype=np.float32,
            )

        results = _DetResults(xyxy=xyxy, conf=conf, cls=cls)

        # Feed embeddings only when the upstream code path can use them
        # — currently with_reid=False, so feats=None. We instead bind
        # embeddings to detections by IoU-overlap below.
        feats = None
        out = self._tracker.update(results, img=img, feats=feats)

        # ``out`` is shape (M, 7+): [x1, y1, x2, y2, track_id, conf, cls, ...]
        tracks: list[dict[str, Any]] = []
        if out is None or len(out) == 0:
            return tracks
        for row in out:
            x1, y1, x2, y2 = row[:4].tolist()
            tid = int(row[4])
            track_conf = float(row[5])
            tracks.append({
                "track_id": tid,
                "bbox": [x1, y1, x2, y2],
                "confidence": track_conf,
                "class_id": int(row[6]) if len(row) > 6 else 0,
                # Filled by orchestrator after consolidation; kept here
                # for output-schema parity with StrongSORTTracker.
                "team": "unknown",
                "jersey_number": None,
                "role": "player",
                "confirmed": True,
            })

        # External-embedding fold-in: BoT-SORT itself isn't using ReID
        # (with_reid=False), but the orchestrator still relies on
        # smoothed feature vectors per tid for track_consolidation. We
        # match each track's bbox to its detection by IoU and update
        # the running centroid here.
        if embeddings is not None and len(detections) > 0 and tracks:
            self._fold_in_embeddings(tracks, detections, embeddings)

        return tracks

    def get_track_features(self) -> dict[int, np.ndarray]:
        """Return ``{tid: smoothed_reid_centroid}`` so consolidation
        can reuse the same OSNet features StrongSORTTracker exposed.
        """
        return dict(self._features)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------
    def _fold_in_embeddings(
        self,
        tracks: list[dict[str, Any]],
        detections: list[dict[str, Any]],
        embeddings: np.ndarray,
    ) -> None:
        """For each emitted track, find the detection it came from
        (highest IoU) and fold that detection's ReID embedding into
        the track's smoothed centroid via exponential moving average.

        Tracks driven by Kalman extrapolation (no detection match this
        frame) won't be in ``tracks`` at all under StrongSORTTracker's
        time_since_update guard; under BoT-SORT they only appear when
        a detection actually matched, so the embedding is always real.
        """
        if not tracks:
            return
        det_xyxy = np.asarray([d["bbox"] for d in detections], dtype=np.float32)
        for t in tracks:
            tx = np.asarray(t["bbox"], dtype=np.float32)
            ious = _iou_xyxy(tx, det_xyxy)
            if ious.size == 0:
                continue
            j = int(ious.argmax())
            if ious[j] < 0.3:
                continue  # bbox doesn't really match any det; skip update
            emb = embeddings[j]
            if emb is None:
                continue
            tid = t["track_id"]
            prev = self._features.get(tid)
            if prev is None:
                self._features[tid] = emb / (np.linalg.norm(emb) + 1e-9)
            else:
                a = self._feature_alpha
                merged = a * prev + (1 - a) * emb
                self._features[tid] = merged / (np.linalg.norm(merged) + 1e-9)


def _iou_xyxy(box: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    """IoU of one box against an (N,4) array — vectorized."""
    if boxes.size == 0:
        return np.zeros((0,), dtype=np.float32)
    x1 = np.maximum(box[0], boxes[:, 0])
    y1 = np.maximum(box[1], boxes[:, 1])
    x2 = np.minimum(box[2], boxes[:, 2])
    y2 = np.minimum(box[3], boxes[:, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    a1 = max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])
    a2 = np.clip(boxes[:, 2] - boxes[:, 0], 0, None) * \
        np.clip(boxes[:, 3] - boxes[:, 1], 0, None)
    return inter / (a1 + a2 - inter + 1e-9)
