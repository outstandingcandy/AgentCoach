"""Config-driven pipeline orchestration for GoalInsight."""

from ._base import Stage, PipelineContext
from ._pipeline import Pipeline
from ._registry import STAGE_REGISTRY, register_stage

# Import adapters to trigger registration
from . import _adapters  # noqa: F401

__all__ = ["Stage", "PipelineContext", "Pipeline", "STAGE_REGISTRY", "register_stage"]
