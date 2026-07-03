"""Shared reader for the ``field_registration.keypoint_detection`` block.

Every calibration backend that runs the HRNet keypoint/line detectors
(``pnlcalib``, ``physical``, ``broadtrack``, ``homography``) reads its
detector settings from ONE shared config block:
``field_registration.keypoint_detection``. This block is intentionally
NOT named after any single backend — it configures the detector, which
several backends reuse.

Backend-*solver* settings stay under their own block (e.g. the
``pnlcalib`` block holds only PnLCalib-solver knobs like ``pnl_refine``,
``use_lines``, ``voting_threshold``). Keeping the two apart is why this
helper exists: so a backend swap doesn't drag solver-specific keys into
the detector config, and so ``keypoint_model_path`` has exactly one home.
"""

from __future__ import annotations

from typing import Any

# Config sub-key that every detector-using backend reads.
KEYPOINT_DETECTION_KEY = "keypoint_detection"


def get_detection_config(fr_config: dict[str, Any]) -> dict[str, Any]:
    """Return the ``field_registration.keypoint_detection`` sub-dict."""
    return fr_config.get(KEYPOINT_DETECTION_KEY, {}) or {}


def build_keypoint_detector_config(det_config: dict[str, Any]) -> dict[str, Any]:
    """Build a ``KeypointDetector`` config from the detection block.

    ``det_config`` is the ``keypoint_detection`` sub-dict. Honors a custom
    fine-tuned ``keypoint_model_path`` when present (falls back to the
    auto-downloaded ``keypoint_weights``, default ``SV_kp``).
    """
    return {
        "backend": "pnlcalib",
        "pnlcalib": {
            "weights": det_config.get("keypoint_weights", "SV_kp"),
            "model_path": det_config.get("keypoint_model_path"),
            "confidence_threshold": det_config.get("keypoint_threshold", 0.3434),
        },
    }


def build_line_detector_config(det_config: dict[str, Any]) -> dict[str, Any]:
    """Build a ``LineDetector`` config from the detection block."""
    return {
        "backend": "pnlcalib",
        "pnlcalib": {
            "weights": det_config.get("line_weights", "SV_lines"),
            "confidence_threshold": det_config.get("line_threshold", 0.15),
        },
    }
