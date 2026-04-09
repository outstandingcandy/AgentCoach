"""Detector registry: maps detector names to BaseEventDetector subclasses."""

from __future__ import annotations

from typing import Type

from ._base import BaseEventDetector

DETECTOR_REGISTRY: dict[str, Type[BaseEventDetector]] = {}


def register_detector(cls: Type[BaseEventDetector]) -> Type[BaseEventDetector]:
    """Class decorator that registers a detector by its .name attribute."""
    if not cls.name:
        raise ValueError(f"{cls.__name__} must define a non-empty 'name'")
    DETECTOR_REGISTRY[cls.name] = cls
    return cls
