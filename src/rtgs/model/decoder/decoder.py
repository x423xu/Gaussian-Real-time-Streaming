from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Literal, Mapping

from torch import Tensor, nn


DepthRenderingMode = Literal["depth", "log", "disparity", "relative_disparity"]
GaussianDict = Mapping[str, Tensor]


@dataclass(slots=True)
class DecoderOutput:
    color: Tensor
    depth: Tensor | None


class Decoder(nn.Module, ABC):
    @abstractmethod
    def forward(
        self,
        gaussians: GaussianDict,
        extrinsics: Tensor,
        intrinsics: Tensor,
        near: Tensor,
        far: Tensor,
        image_shape: tuple[int, int],
        depth_mode: DepthRenderingMode | None = None,
    ) -> DecoderOutput:
        pass
