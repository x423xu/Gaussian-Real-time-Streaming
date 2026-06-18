from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import torch
from torch import Tensor, nn


@dataclass(slots=True)
class IntrinsicEmbeddingConfig:
    enabled: bool = False
    dim: int = 32
    hidden_dim: int = 64


def coerce_intrinsic_embedding_config(value: IntrinsicEmbeddingConfig | Mapping[str, Any] | None) -> IntrinsicEmbeddingConfig:
    if value is None:
        return IntrinsicEmbeddingConfig()
    if isinstance(value, IntrinsicEmbeddingConfig):
        return value
    raw = dict(value)
    return IntrinsicEmbeddingConfig(
        enabled=bool(raw.get("enabled", False)),
        dim=int(raw.get("dim", 32)),
        hidden_dim=int(raw.get("hidden_dim", 64)),
    )


class IntrinsicEmbedding(nn.Module):
    """Small per-view embedding for camera intrinsics at the current image scale."""

    def __init__(self, dim: int = 32, hidden_dim: int = 64) -> None:
        super().__init__()
        if dim <= 0:
            raise ValueError(f"Intrinsic embedding dim must be positive, got {dim}")
        self.dim = int(dim)
        self.net = nn.Sequential(
            nn.Linear(6, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim),
        )

    def forward(self, intrinsics: Tensor, image_shape: tuple[int, int]) -> Tensor:
        if intrinsics.ndim != 4 or intrinsics.shape[-2:] != (3, 3):
            raise ValueError(f"Expected intrinsics shape (B,V,3,3), got {tuple(intrinsics.shape)}")
        height, width = image_shape
        size = intrinsics.new_tensor(
            [max(float(width), 1.0), max(float(height), 1.0)],
        )
        fx = intrinsics[..., 0, 0] / size[0]
        fy = intrinsics[..., 1, 1] / size[1]
        cx = intrinsics[..., 0, 2] / size[0]
        cy = intrinsics[..., 1, 2] / size[1]
        log_w = intrinsics.new_full(fx.shape, float(width)).clamp_min(1.0).log()
        log_h = intrinsics.new_full(fx.shape, float(height)).clamp_min(1.0).log()
        values = torch.stack((fx, fy, cx, cy, log_w, log_h), dim=-1)
        return self.net(values)

    def broadcast(self, embedding: Tensor, image_shape: tuple[int, int]) -> Tensor:
        if embedding.ndim != 3:
            raise ValueError(f"Expected embedding shape (B,V,C), got {tuple(embedding.shape)}")
        height, width = image_shape
        return embedding[..., None, None].expand(*embedding.shape, height, width)
