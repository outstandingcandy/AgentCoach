"""Jersey number recognition backends."""

from .qwen_recognizer import QwenJerseyRecognizer, JerseyNumberAggregator
from .mmocr_recognizer import MMOCRJerseyRecognizer, TrackJerseyNumberVoting

__all__ = [
    "QwenJerseyRecognizer",
    "MMOCRJerseyRecognizer",
    "JerseyNumberAggregator",
    "TrackJerseyNumberVoting",
]
