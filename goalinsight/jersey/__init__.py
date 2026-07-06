"""Jersey number recognition backends."""

from .qwen_recognizer import QwenJerseyRecognizer, JerseyNumberAggregator

__all__ = [
    "QwenJerseyRecognizer",
    "JerseyNumberAggregator",
]
