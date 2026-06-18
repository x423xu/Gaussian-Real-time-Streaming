from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Mapping

import torch
from torch import Tensor, nn


@dataclass(slots=True)
class CameraRefinementConfig:
    enabled: bool = False
    hidden_dim: int = 64
    max_rotation_deg: float = 1.0
    max_translation_ratio: float = 0.02
    anchor_first_context: bool = True
    lambda_delta: float = 0.0


def coerce_camera_refinement_config(value: CameraRefinementConfig | Mapping[str, Any] | None) -> CameraRefinementConfig:
    if value is None:
        return CameraRefinementConfig()
    if isinstance(value, CameraRefinementConfig):
        return value
    raw = dict(value)
    return CameraRefinementConfig(
        enabled=bool(raw.get("enabled", False)),
        hidden_dim=int(raw.get("hidden_dim", 64)),
        max_rotation_deg=float(raw.get("max_rotation_deg", 1.0)),
        max_translation_ratio=float(raw.get("max_translation_ratio", 0.02)),
        anchor_first_context=bool(raw.get("anchor_first_context", True)),
        lambda_delta=float(raw.get("lambda_delta", 0.0)),
    )


class CameraPoseRefiner(nn.Module):
    """Bounded, zero-initialized SE(3) residual refiner for DA3 camera poses."""

    def __init__(self, cfg: CameraRefinementConfig | Mapping[str, Any] | None = None, intrinsic_embedding_dim: int = 0) -> None:
        super().__init__()
        self.cfg = coerce_camera_refinement_config(cfg)
        self.intrinsic_embedding_dim = int(intrinsic_embedding_dim)
        input_dim = 2 + self.intrinsic_embedding_dim
        self.net = nn.Sequential(
            nn.Linear(input_dim, self.cfg.hidden_dim),
            nn.GELU(),
            nn.Linear(self.cfg.hidden_dim, 6),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(
        self,
        extrinsics: Tensor,
        depth: Tensor,
        intrinsic_embedding: Tensor | None = None,
        context_views: int | None = None,
    ) -> dict[str, Tensor | dict[str, Tensor]]:
        if extrinsics.ndim != 4 or extrinsics.shape[-2:] != (4, 4):
            raise ValueError(f"Expected extrinsics shape (B,V,4,4), got {tuple(extrinsics.shape)}")
        if depth.shape[:2] != extrinsics.shape[:2]:
            raise ValueError(f"Expected depth leading shape {tuple(extrinsics.shape[:2])}, got {tuple(depth.shape[:2])}")
        depth_stats = self._depth_stats(depth)
        if self.intrinsic_embedding_dim > 0:
            if intrinsic_embedding is None:
                raise ValueError("CameraPoseRefiner requires intrinsic_embedding when intrinsic_embedding_dim > 0")
            features = torch.cat([depth_stats, intrinsic_embedding], dim=-1)
        else:
            features = depth_stats
        raw_delta = self.net(features)
        if self.cfg.anchor_first_context and context_views is not None and context_views > 0:
            raw_delta = raw_delta.clone()
            raw_delta[:, 0] = 0.0
        refined = self._apply_delta(extrinsics, raw_delta, depth)
        losses = {
            "camera_delta_regularization": raw_delta.square().mean() * self.cfg.lambda_delta,
        }
        diagnostics = {
            "camera_delta_abs_mean": raw_delta.detach().abs().mean(),
        }
        return {"extrinsics": refined, "losses": losses, "diagnostics": diagnostics}

    def _depth_stats(self, depth: Tensor) -> Tensor:
        flat = depth.clamp_min(1.0e-6).log().reshape(*depth.shape[:2], -1)
        return torch.stack((flat.mean(dim=-1), flat.std(dim=-1, unbiased=False)), dim=-1)

    def _apply_delta(self, extrinsics: Tensor, raw_delta: Tensor, depth: Tensor) -> Tensor:
        max_angle = math.radians(self.cfg.max_rotation_deg)
        rotvec = torch.tanh(raw_delta[..., :3]) * max_angle
        rotation = self._axis_angle_to_matrix(rotvec)
        scene_scale = depth.detach().reshape(depth.shape[0], -1).median(dim=-1).values.clamp_min(1.0e-6)
        translation = torch.tanh(raw_delta[..., 3:]) * (scene_scale[:, None, None] * self.cfg.max_translation_ratio)
        delta = torch.eye(4, device=extrinsics.device, dtype=extrinsics.dtype).reshape(1, 1, 4, 4).repeat(*extrinsics.shape[:2], 1, 1)
        delta[..., :3, :3] = rotation
        delta[..., :3, 3] = translation
        return extrinsics @ delta

    def _axis_angle_to_matrix(self, rotvec: Tensor) -> Tensor:
        angle = rotvec.norm(dim=-1, keepdim=True).clamp_min(1.0e-8)
        axis = rotvec / angle
        x, y, z = axis.unbind(dim=-1)
        zero = torch.zeros_like(x)
        skew = torch.stack(
            (
                zero,
                -z,
                y,
                z,
                zero,
                -x,
                -y,
                x,
                zero,
            ),
            dim=-1,
        ).reshape(*rotvec.shape[:-1], 3, 3)
        eye = torch.eye(3, device=rotvec.device, dtype=rotvec.dtype).reshape(1, 1, 3, 3)
        angle = angle[..., None]
        return eye + torch.sin(angle) * skew + (1.0 - torch.cos(angle)) * (skew @ skew)
