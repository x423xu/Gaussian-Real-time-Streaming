from __future__ import annotations

from typing import Any

from ..types import DatasetStage
from .base import StepTracker, ViewSampler
from .samplers import AllSampler, ArbitrarySampler, BoundedSampler, BoundedV2Sampler, EvaluationSampler, UnboundedSampler

SAMPLERS = {
    "all": AllSampler,
    "arbitrary": ArbitrarySampler,
    "bounded": BoundedSampler,
    "boundedv2": BoundedV2Sampler,
    "evaluation": EvaluationSampler,
    "unbounded": UnboundedSampler,
}


def build_view_sampler(cfg: dict[str, Any], stage: DatasetStage | str, overfit: bool, cameras_are_circular: bool, step_tracker: StepTracker | None = None) -> ViewSampler:
    name = cfg["name"]
    return SAMPLERS[name](cfg, DatasetStage(stage), overfit, cameras_are_circular, step_tracker)

__all__ = ["StepTracker", "ViewSampler", "build_view_sampler"]
