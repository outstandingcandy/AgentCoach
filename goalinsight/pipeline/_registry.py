"""Stage registry: maps stage names to Stage subclasses."""

from typing import Type

from ._base import Stage

STAGE_REGISTRY: dict[str, Type[Stage]] = {}


def register_stage(cls: Type[Stage]) -> Type[Stage]:
    """Class decorator that registers a Stage subclass by its .name attribute."""
    if not cls.name:
        raise ValueError(f"{cls.__name__} must define a non-empty 'name' class attribute")
    STAGE_REGISTRY[cls.name] = cls
    return cls
