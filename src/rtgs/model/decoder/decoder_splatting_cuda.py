from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch
from torch import Tensor

from .cuda_splatting import DepthRenderingMode, render_cuda, render_depth_cuda
from .decoder import Decoder, DecoderOutput, GaussianDict


@dataclass(slots=True)
class DecoderSplattingCUDACfg:
    name: Literal["splatting_cuda"] = "splatting_cuda"
    background_color: tuple[float, float, float] = (0.0, 0.0, 0.0)


class DecoderSplattingCUDA(Decoder):
    def __init__(self, cfg: DecoderSplattingCUDACfg | None = None) -> None:
        super().__init__()
        self.cfg = cfg or DecoderSplattingCUDACfg()
        self.register_buffer("background_color", torch.tensor(self.cfg.background_color, dtype=torch.float32), persistent=False)

    def forward(
        self,
        gaussians: GaussianDict,
        extrinsics: Tensor,
        intrinsics: Tensor,
        near: Tensor,
        far: Tensor,
        image_shape: tuple[int, int],
        depth_mode: DepthRenderingMode | None = None,
        scale_invariant: bool = False,
    ) -> DecoderOutput:
        batch, views = extrinsics.shape[:2]
        flat_shape = (batch * views,)
        color = render_cuda(
            extrinsics.reshape(*flat_shape, 4, 4),
            intrinsics.reshape(*flat_shape, 3, 3),
            near.reshape(*flat_shape),
            far.reshape(*flat_shape),
            image_shape,
            self.background_color.to(device=extrinsics.device, dtype=extrinsics.dtype).reshape(1, 3).expand(batch * views, 3),
            gaussians["means"].unsqueeze(1).expand(batch, views, -1, -1).reshape(batch * views, -1, 3).contiguous(),
            gaussians["covariances"].unsqueeze(1).expand(batch, views, -1, -1, -1).reshape(batch * views, -1, 3, 3).contiguous(),
            gaussians["harmonics"].unsqueeze(1).expand(batch, views, -1, -1, -1).reshape(batch * views, -1, 3, gaussians["harmonics"].shape[-1]).contiguous(),
            gaussians["opacities"].squeeze(-1).unsqueeze(1).expand(batch, views, -1).reshape(batch * views, -1).contiguous(),
            scale_invariant=scale_invariant,
        ).reshape(batch, views, 3, *image_shape)
        depth = None
        if depth_mode is not None:
            depth = self.render_depth(gaussians, extrinsics, intrinsics, near, far, image_shape, depth_mode)
        return DecoderOutput(color=color, depth=depth)

    def render_depth(
        self,
        gaussians: GaussianDict,
        extrinsics: Tensor,
        intrinsics: Tensor,
        near: Tensor,
        far: Tensor,
        image_shape: tuple[int, int],
        mode: DepthRenderingMode = "depth",
    ) -> Tensor:
        batch, views = extrinsics.shape[:2]
        flat_shape = (batch * views,)
        depth = render_depth_cuda(
            extrinsics.reshape(*flat_shape, 4, 4),
            intrinsics.reshape(*flat_shape, 3, 3),
            near.reshape(*flat_shape),
            far.reshape(*flat_shape),
            image_shape,
            gaussians["means"].unsqueeze(1).expand(batch, views, -1, -1).reshape(batch * views, -1, 3),
            gaussians["covariances"].unsqueeze(1).expand(batch, views, -1, -1, -1).reshape(batch * views, -1, 3, 3),
            gaussians["opacities"].squeeze(-1).unsqueeze(1).expand(batch, views, -1).reshape(batch * views, -1),
            mode=mode,
        )
        return depth.reshape(batch, views, *image_shape)
