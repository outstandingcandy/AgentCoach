"""Backward-compat shim. AnchorAnnotator lives in ``annotator_state``."""

from .annotator_state import AnchorAnnotator

__all__ = ["AnchorAnnotator"]
