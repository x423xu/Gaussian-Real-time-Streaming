from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn
import torch.nn.functional as F


@dataclass(slots=True)
class RTGSModelConfig:
    name: str = "rtgs_model"
    hidden_channels: int = 16


class RTGSModel(nn.Module):
    """Tiny feedforward placeholder: two conv layers from context RGB to RGB + opacity."""

    def __init__(self, cfg: RTGSModelConfig | None = None) -> None:
        super().__init__()
        self.cfg = cfg or RTGSModelConfig()
        self.conv1 = nn.Conv2d(3, self.cfg.hidden_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(self.cfg.hidden_channels, 4, kernel_size=3, padding=1)

    def forward(self, batch: dict) -> dict[str, Tensor | dict[str, Tensor]]:
        context = batch["context"]["image"]
        if context.ndim == 4:
            context = context.unsqueeze(0)
        if context.ndim != 5:
            raise ValueError(f"Expected context image shape (B,V,3,H,W), got {tuple(context.shape)}")
        features = context.mean(dim=1)
        raw = self.conv2(F.relu(self.conv1(features)))
        rgb = torch.sigmoid(raw[:, :3])
        opacity_map = torch.sigmoid(raw[:, 3:4])
        return {"rgb": rgb, "gaussians": self._image_proxy_to_gaussians(rgb, opacity_map)}

    def _image_proxy_to_gaussians(self, rgb: Tensor, opacity_map: Tensor) -> dict[str, Tensor]:
        batch, _, height, width = rgb.shape
        y, x = torch.meshgrid(
            torch.linspace(-1.0, 1.0, height, device=rgb.device, dtype=rgb.dtype),
            torch.linspace(-1.0, 1.0, width, device=rgb.device, dtype=rgb.dtype),
            indexing="ij",
        )
        z = opacity_map[:, 0]
        means = torch.stack([x.expand(batch, -1, -1), y.expand(batch, -1, -1), z], dim=-1)
        return {
            "means": means.reshape(batch, height * width, 3),
            "colors": rgb.permute(0, 2, 3, 1).reshape(batch, height * width, 3),
            "opacities": opacity_map.permute(0, 2, 3, 1).reshape(batch, height * width, 1),
        }