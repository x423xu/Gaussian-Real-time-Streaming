from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

import torch
from torch import Tensor, nn
import torch.nn.functional as F


@dataclass(slots=True)
class CostVolumeDepthRefinementConfig:
    enabled: bool = False
    type: str = "cost_volume"
    num_depth_bins: int = 128
    feature_scale: int = 4
    depth_sampling: str = "log"
    bound_source: str = "context"
    prior_sigma: float = 0.03
    temperature: float = 1.0
    max_log_depth_shift: float = 0.25
    detach_da3_depth: bool = True
    da3_feature_layers: list[int] = field(default_factory=lambda: [5, 7, 9, 11])
    hidden_channels: int = 16
    lambda_kl: float = 0.01
    lambda_smooth: float = 0.0


def coerce_depth_refinement_config(
    value: CostVolumeDepthRefinementConfig | Mapping[str, Any] | None,
) -> CostVolumeDepthRefinementConfig:
    if value is None:
        return CostVolumeDepthRefinementConfig()
    if isinstance(value, CostVolumeDepthRefinementConfig):
        return value
    raw = dict(value)
    layers = raw.get("da3_feature_layers", [5, 7, 9, 11])
    return CostVolumeDepthRefinementConfig(
        enabled=bool(raw.get("enabled", False)),
        type=str(raw.get("type", "cost_volume")),
        num_depth_bins=int(raw.get("num_depth_bins", 128)),
        feature_scale=int(raw.get("feature_scale", 4)),
        depth_sampling=str(raw.get("depth_sampling", "log")),
        bound_source=str(raw.get("bound_source", "context")),
        prior_sigma=float(raw.get("prior_sigma", 0.03)),
        temperature=float(raw.get("temperature", 1.0)),
        max_log_depth_shift=float(raw.get("max_log_depth_shift", 0.25)),
        detach_da3_depth=bool(raw.get("detach_da3_depth", True)),
        da3_feature_layers=[int(layer) for layer in layers],
        hidden_channels=int(raw.get("hidden_channels", 16)),
        lambda_kl=float(raw.get("lambda_kl", 0.01)),
        lambda_smooth=float(raw.get("lambda_smooth", 0.0)),
    )


class CostVolumeDepthRefiner(nn.Module):
    """Refine frozen DA3 depth by shifting a bounded distribution from cross-view feature costs."""

    def __init__(
        self,
        cfg: CostVolumeDepthRefinementConfig | Mapping[str, Any] | None = None,
        intrinsic_embedding_dim: int = 0,
    ) -> None:
        super().__init__()
        self.cfg = coerce_depth_refinement_config(cfg)
        self.intrinsic_embedding_dim = int(intrinsic_embedding_dim)
        if self.cfg.num_depth_bins <= 1:
            raise ValueError("num_depth_bins must be greater than 1")
        if self.cfg.feature_scale <= 0:
            raise ValueError("feature_scale must be positive")
        if self.cfg.depth_sampling not in {"log", "linear"}:
            raise ValueError("depth_sampling must be 'log' or 'linear'")
        if self.cfg.bound_source != "context":
            raise ValueError("Only bound_source='context' is supported for RTGS context-depth refinement")

        input_channels = self.cfg.num_depth_bins + self.intrinsic_embedding_dim
        self.logit_head = nn.Sequential(
            nn.Conv2d(input_channels, self.cfg.hidden_channels, kernel_size=3, padding=1, padding_mode="replicate"),
            nn.GELU(),
            nn.Conv2d(self.cfg.hidden_channels, self.cfg.num_depth_bins, kernel_size=3, padding=1, padding_mode="replicate"),
        )
        nn.init.zeros_(self.logit_head[-1].weight)
        nn.init.zeros_(self.logit_head[-1].bias)

    def forward(
        self,
        images: Tensor,
        depth: Tensor,
        intrinsics: Tensor,
        extrinsics: Tensor,
        intrinsic_embedding: Tensor | None = None,
        rtgs_features: Tensor | Mapping[str, Tensor] | None = None,
        da3_features: Tensor | list[Tensor] | tuple[Tensor, ...] | Mapping[str, Tensor] | None = None,
    ) -> dict[str, Tensor | dict[str, Tensor]]:
        self._validate_inputs(images, depth, intrinsics, extrinsics)
        batch, views, _, height, width = images.shape
        low_shape = (max(1, height // self.cfg.feature_scale), max(1, width // self.cfg.feature_scale))

        features = self._compose_features(rtgs_features, da3_features, batch, views, low_shape, depth.device, depth.dtype)
        depth_for_prior = depth.detach() if self.cfg.detach_da3_depth else depth
        low_depth = F.interpolate(
            depth_for_prior.reshape(batch * views, 1, height, width),
            size=low_shape,
            mode="bilinear",
            align_corners=False,
        ).reshape(batch, views, *low_shape)
        near, far = self._compute_bounds(depth_for_prior)
        depth_bins = self._make_depth_bins(near, far, depth.device, depth.dtype)

        prior_logits = self._make_prior_logits(depth_bins, low_depth)
        matching_cost = self._build_matching_cost(features, depth_bins, intrinsics, extrinsics, (height, width))
        logit_input = matching_cost.reshape(batch * views, self.cfg.num_depth_bins, *low_shape)
        if self.intrinsic_embedding_dim > 0:
            if intrinsic_embedding is None:
                raise ValueError("CostVolumeDepthRefiner requires intrinsic_embedding when intrinsic_embedding_dim > 0")
            expected_shape = (batch, views, self.intrinsic_embedding_dim)
            if intrinsic_embedding.shape != expected_shape:
                raise ValueError(f"Expected intrinsic_embedding shape {expected_shape}, got {tuple(intrinsic_embedding.shape)}")
            conditioning = intrinsic_embedding.reshape(batch * views, self.intrinsic_embedding_dim, 1, 1)
            conditioning = conditioning.expand(-1, -1, *low_shape)
            logit_input = torch.cat([logit_input, conditioning], dim=1)
        delta_logits = self.logit_head(logit_input).reshape(batch, views, self.cfg.num_depth_bins, *low_shape)

        prior_prob = prior_logits.softmax(dim=2)
        logits = prior_logits + delta_logits / max(self.cfg.temperature, 1.0e-6)
        probability = logits.softmax(dim=2)
        low_prior_depth = (prior_prob * depth_bins[:, None, :, None, None]).sum(dim=2)
        low_refined_depth = (probability * depth_bins[:, None, :, None, None]).sum(dim=2)
        low_correction = low_refined_depth - low_prior_depth
        correction = F.interpolate(
            low_correction.reshape(batch * views, 1, *low_shape),
            size=(height, width),
            mode="bilinear",
            align_corners=False,
        ).reshape(batch, views, height, width)
        refined = self._clamp_log_shift(depth + correction, depth_for_prior)

        losses = {
            "depth_refinement_kl": self._kl_loss(probability, prior_prob),
            "depth_refinement_smoothness": self._smoothness_loss(refined, images),
        }
        diagnostics = {
            "depth_refinement_entropy": self._entropy(probability).detach(),
            "depth_refinement_feature_channels": features.new_tensor(float(features.shape[2])).detach(),
            "depth_refinement_mean_abs_log_shift": (refined.clamp_min(1.0e-6).log() - depth_for_prior.clamp_min(1.0e-6).log()).abs().mean().detach(),
        }
        return {
            "depth": refined,
            "probability": probability,
            "near": near,
            "far": far,
            "losses": losses,
            "diagnostics": diagnostics,
        }

    def _validate_inputs(self, images: Tensor, depth: Tensor, intrinsics: Tensor, extrinsics: Tensor) -> None:
        if images.ndim != 5:
            raise ValueError(f"Expected images shape (B,V,3,H,W), got {tuple(images.shape)}")
        if depth.shape != images.shape[:2] + images.shape[-2:]:
            raise ValueError(f"Expected depth shape {(images.shape[:2] + images.shape[-2:])}, got {tuple(depth.shape)}")
        if intrinsics.shape != images.shape[:2] + (3, 3):
            raise ValueError(f"Expected intrinsics shape {(images.shape[:2] + (3, 3))}, got {tuple(intrinsics.shape)}")
        if extrinsics.shape != images.shape[:2] + (4, 4):
            raise ValueError(f"Expected extrinsics shape {(images.shape[:2] + (4, 4))}, got {tuple(extrinsics.shape)}")

    def _compose_features(
        self,
        rtgs_features: Tensor | Mapping[str, Tensor] | None,
        da3_features: Tensor | list[Tensor] | tuple[Tensor, ...] | Mapping[str, Tensor] | None,
        batch: int,
        views: int,
        output_shape: tuple[int, int],
        device: torch.device,
        dtype: torch.dtype,
    ) -> Tensor:
        feature_maps: list[Tensor] = []
        feature_maps.extend(self._flatten_feature_collection(rtgs_features))
        feature_maps.extend(self._flatten_feature_collection(da3_features))
        if not feature_maps:
            raise ValueError("CostVolumeDepthRefiner requires RTGS and/or DA3 feature maps; image-only fallback is intentionally disabled")
        prepared = [
            self._prepare_feature_map(feature, batch, views, output_shape, device, dtype)
            for feature in feature_maps
        ]
        return torch.cat(prepared, dim=2)

    def _flatten_feature_collection(self, value: Any) -> list[Tensor]:
        if value is None:
            return []
        if torch.is_tensor(value):
            return [value]
        if isinstance(value, Mapping):
            result: list[Tensor] = []
            for item in value.values():
                result.extend(self._flatten_feature_collection(item))
            return result
        if isinstance(value, (list, tuple)):
            result = []
            for item in value:
                result.extend(self._flatten_feature_collection(item))
            return result
        return []

    def _prepare_feature_map(
        self,
        feature: Tensor,
        batch: int,
        views: int,
        output_shape: tuple[int, int],
        device: torch.device,
        dtype: torch.dtype,
    ) -> Tensor:
        feature = feature.to(device=device, dtype=dtype)
        if feature.ndim == 4:
            if feature.shape[0] != batch * views:
                raise ValueError(f"Expected flattened feature batch {batch * views}, got {tuple(feature.shape)}")
            channels = feature.shape[1]
            feature = feature.reshape(batch, views, channels, *feature.shape[-2:])
        elif feature.ndim == 5:
            if feature.shape[:2] != (batch, views):
                raise ValueError(f"Expected feature leading shape {(batch, views)}, got {tuple(feature.shape[:2])}")
        else:
            raise ValueError(f"Expected feature map shape (B,V,C,H,W) or (B*V,C,H,W), got {tuple(feature.shape)}")
        feature = feature.reshape(batch * views, feature.shape[2], *feature.shape[-2:])
        if feature.shape[-2:] != output_shape:
            feature = F.interpolate(feature, size=output_shape, mode="bilinear", align_corners=False)
        feature = feature.reshape(batch, views, feature.shape[1], *output_shape)
        return F.normalize(feature, dim=2)

    def _compute_bounds(self, depth: Tensor) -> tuple[Tensor, Tensor]:
        flat = depth.reshape(depth.shape[0], -1).clamp_min(1.0e-6)
        near = flat.amin(dim=1)
        far = flat.amax(dim=1)
        far = torch.maximum(far, near * 1.001)
        return near, far

    def _make_depth_bins(self, near: Tensor, far: Tensor, device: torch.device, dtype: torch.dtype) -> Tensor:
        t = torch.linspace(0.0, 1.0, self.cfg.num_depth_bins, device=device, dtype=dtype)
        if self.cfg.depth_sampling == "log":
            log_near = near.clamp_min(1.0e-6).log()
            log_far = far.clamp_min(1.0e-6).log()
            return (log_near[:, None] * (1.0 - t[None]) + log_far[:, None] * t[None]).exp()
        return near[:, None] * (1.0 - t[None]) + far[:, None] * t[None]

    def _make_prior_logits(self, depth_bins: Tensor, low_depth: Tensor) -> Tensor:
        if self.cfg.depth_sampling == "log":
            bin_values = depth_bins.clamp_min(1.0e-6).log()
            center = low_depth.clamp_min(1.0e-6).log()
        else:
            bin_values = depth_bins
            center = low_depth
        sigma = max(self.cfg.prior_sigma, 1.0e-6)
        return -0.5 * ((bin_values[:, None, :, None, None] - center[:, :, None]) / sigma).square()

    def _build_matching_cost(
        self,
        features: Tensor,
        depth_bins: Tensor,
        intrinsics: Tensor,
        extrinsics: Tensor,
        image_shape: tuple[int, int],
    ) -> Tensor:
        batch, views, channels, height, width = features.shape
        if views == 1:
            return features.new_zeros(batch, views, self.cfg.num_depth_bins, height, width)
        scaled_intrinsics = self._scale_intrinsics(intrinsics, image_shape, (height, width))
        cost = features.new_zeros(batch, views, self.cfg.num_depth_bins, height, width)
        counts = features.new_zeros(batch, views, 1, 1, 1)
        for ref_idx in range(views):
            ref_feature = features[:, ref_idx].permute(0, 2, 3, 1).reshape(batch, height * width, channels)
            for src_idx in range(views):
                if src_idx == ref_idx:
                    continue
                warped, valid = self._warp_source_features(
                    features[:, src_idx],
                    depth_bins,
                    scaled_intrinsics[:, ref_idx],
                    scaled_intrinsics[:, src_idx],
                    extrinsics[:, ref_idx],
                    extrinsics[:, src_idx],
                )
                diff = (warped - ref_feature[:, None]).square().mean(dim=-1).reshape(batch, self.cfg.num_depth_bins, height, width)
                invalid_penalty = diff.detach().amax(dim=(1, 2, 3), keepdim=True).clamp_min(1.0) + 1.0
                diff = torch.where(valid.reshape(batch, self.cfg.num_depth_bins, height, width), diff, invalid_penalty)
                cost[:, ref_idx] = cost[:, ref_idx] + diff
                counts[:, ref_idx] = counts[:, ref_idx] + 1.0
        return cost / counts.clamp_min(1.0)

    def _warp_source_features(
        self,
        source_features: Tensor,
        depth_bins: Tensor,
        ref_intrinsics: Tensor,
        src_intrinsics: Tensor,
        ref_c2w: Tensor,
        src_c2w: Tensor,
    ) -> tuple[Tensor, Tensor]:
        batch, channels, height, width = source_features.shape
        yy, xx = torch.meshgrid(
            torch.arange(height, device=source_features.device, dtype=source_features.dtype),
            torch.arange(width, device=source_features.device, dtype=source_features.dtype),
            indexing="ij",
        )
        pixels = torch.stack((xx, yy), dim=-1).reshape(1, height * width, 2)
        z = depth_bins[:, :, None].expand(batch, self.cfg.num_depth_bins, height * width)
        fx = ref_intrinsics[:, 0, 0].clamp_min(1.0e-6)[:, None, None]
        fy = ref_intrinsics[:, 1, 1].clamp_min(1.0e-6)[:, None, None]
        cx = ref_intrinsics[:, 0, 2][:, None, None]
        cy = ref_intrinsics[:, 1, 2][:, None, None]
        x = (pixels[..., 0][:, None] - cx) / fx * z
        y = (pixels[..., 1][:, None] - cy) / fy * z
        points_cam = torch.stack((x, y, z), dim=-1)
        world = torch.einsum("bij,bdpj->bdpi", ref_c2w[:, :3, :3], points_cam) + ref_c2w[:, None, None, :3, 3]
        src_w2c = torch.linalg.inv(src_c2w)
        src_cam = torch.einsum("bij,bdpj->bdpi", src_w2c[:, :3, :3], world) + src_w2c[:, None, None, :3, 3]
        src_z = src_cam[..., 2].clamp_min(1.0e-6)
        u = src_intrinsics[:, 0, 0][:, None, None] * src_cam[..., 0] / src_z + src_intrinsics[:, 0, 2][:, None, None]
        v = src_intrinsics[:, 1, 1][:, None, None] * src_cam[..., 1] / src_z + src_intrinsics[:, 1, 2][:, None, None]
        grid_x = 2.0 * u / max(width - 1, 1) - 1.0
        grid_y = 2.0 * v / max(height - 1, 1) - 1.0
        grid = torch.stack((grid_x, grid_y), dim=-1)
        valid = (src_cam[..., 2] > 1.0e-6) & (grid_x >= -1.0) & (grid_x <= 1.0) & (grid_y >= -1.0) & (grid_y <= 1.0)
        sampled = F.grid_sample(
            source_features,
            grid.reshape(batch, self.cfg.num_depth_bins * height * width, 1, 2),
            mode="bilinear",
            padding_mode="zeros",
            align_corners=True,
        )
        sampled = sampled.reshape(batch, channels, self.cfg.num_depth_bins, height * width).permute(0, 2, 3, 1)
        return sampled, valid

    def _scale_intrinsics(self, intrinsics: Tensor, source_shape: tuple[int, int], target_shape: tuple[int, int]) -> Tensor:
        source_h, source_w = source_shape
        target_h, target_w = target_shape
        scaled = intrinsics.clone()
        scaled[..., 0, :] *= float(target_w) / float(source_w)
        scaled[..., 1, :] *= float(target_h) / float(source_h)
        return scaled

    def _clamp_log_shift(self, refined: Tensor, base_depth: Tensor) -> Tensor:
        if self.cfg.max_log_depth_shift <= 0.0:
            return refined.clamp_min(1.0e-6)
        base_log = base_depth.clamp_min(1.0e-6).log()
        refined_log = refined.clamp_min(1.0e-6).log()
        refined_log = torch.minimum(refined_log, base_log + self.cfg.max_log_depth_shift)
        refined_log = torch.maximum(refined_log, base_log - self.cfg.max_log_depth_shift)
        return refined_log.exp()

    def _kl_loss(self, probability: Tensor, prior_probability: Tensor) -> Tensor:
        kl = probability * (probability.clamp_min(1.0e-8).log() - prior_probability.clamp_min(1.0e-8).log())
        return kl.sum(dim=2).mean() * self.cfg.lambda_kl

    def _smoothness_loss(self, depth: Tensor, images: Tensor) -> Tensor:
        if self.cfg.lambda_smooth <= 0.0:
            return depth.new_zeros(())
        log_depth = depth.clamp_min(1.0e-6).log()
        dx = (log_depth[..., :, 1:] - log_depth[..., :, :-1]).abs()
        dy = (log_depth[..., 1:, :] - log_depth[..., :-1, :]).abs()
        gray = images.mean(dim=2)
        wx = torch.exp(-10.0 * (gray[..., :, 1:] - gray[..., :, :-1]).abs().mean(dim=1, keepdim=True))
        wy = torch.exp(-10.0 * (gray[..., 1:, :] - gray[..., :-1, :]).abs().mean(dim=1, keepdim=True))
        return (dx * wx).mean() * self.cfg.lambda_smooth + (dy * wy).mean() * self.cfg.lambda_smooth

    def _entropy(self, probability: Tensor) -> Tensor:
        return -(probability * probability.clamp_min(1.0e-8).log()).sum(dim=2).mean()
