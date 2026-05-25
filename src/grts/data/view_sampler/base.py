from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor

from ..types import DatasetStage


@dataclass
class StepTracker:
    step: int = 0

    def get_step(self) -> int:
        return self.step


class ViewSampler(ABC):
    def __init__(self, cfg: dict[str, Any], stage: DatasetStage, overfit: bool, cameras_are_circular: bool, step_tracker: StepTracker | None = None):
        self.cfg = cfg
        self.stage = DatasetStage(stage)
        self.overfit = overfit
        self.cameras_are_circular = cameras_are_circular
        self.step_tracker = step_tracker

    @abstractmethod
    def sample(self, scene: str, extrinsics: Tensor, intrinsics: Tensor, device: torch.device = torch.device("cpu"), **kwargs):
        raise NotImplementedError

    @property
    def global_step(self) -> int:
        return 0 if self.step_tracker is None else self.step_tracker.get_step()

    @property
    def num_context_views(self) -> int:
        return int(self.cfg.get("num_context_views", 0))

    @property
    def num_target_views(self) -> int:
        return int(self.cfg.get("num_target_views", 0))
