from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import sys
from typing import Any

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from .camera_refinement import CameraPoseRefiner, CameraRefinementConfig, coerce_camera_refinement_config
from .decoder import DecoderOutput, DecoderSplattingCUDA, DecoderSplattingCUDACfg
from .depth_refinement import CostVolumeDepthRefiner, CostVolumeDepthRefinementConfig, coerce_depth_refinement_config
from .intrinsic_embedding import IntrinsicEmbedding, IntrinsicEmbeddingConfig, coerce_intrinsic_embedding_config


C0 = 0.28209479177387814


@dataclass(slots=True)
class RTGSModelConfig:
    name: str = "rtgs_model"
    hidden_channels: int = 16
    vit_type: str = "vit-b"
    vit_pretrained: bool = True
    vit_image_size: int = 252
    dpt_feature_channels: int = 128
    da3_model_name: str = "depth-anything/DA3-BASE"
    da3_ref_view_strategy: str = "middle"
    gaussian_scale_min: float = 1.0e-4
    gaussian_scale_max: float = 1.0e-2
    sh_degree: int = 3
    decoder_background_color: tuple[float, float, float] = (0.0, 0.0, 0.0)
    intrinsic_embedding: IntrinsicEmbeddingConfig | dict[str, Any] = field(default_factory=IntrinsicEmbeddingConfig)
    depth_refinement: CostVolumeDepthRefinementConfig | dict[str, Any] = field(default_factory=CostVolumeDepthRefinementConfig)
    camera_refinement: CameraRefinementConfig | dict[str, Any] = field(default_factory=CameraRefinementConfig)


def _ensure_da3_src_on_path() -> None:
    repo_root = Path(__file__).resolve().parents[3]
    da3_src = repo_root / "submodules" / "Depth-Anything-3" / "src"
    if str(da3_src) not in sys.path:
        sys.path.insert(0, str(da3_src))


def _normalize_vit_type(vit_type: str) -> str:
    aliases = {
        "s": "vit-s",
        "small": "vit-s",
        "vits": "vit-s",
        "vit_s": "vit-s",
        "vit-s": "vit-s",
        "b": "vit-b",
        "base": "vit-b",
        "vitb": "vit-b",
        "vit_b": "vit-b",
        "vit-b": "vit-b",
        "l": "vit-l",
        "large": "vit-l",
        "vitl": "vit-l",
        "vit_l": "vit-l",
        "vit-l": "vit-l",
    }
    normalized = aliases.get(vit_type.lower())
    if normalized is None:
        raise ValueError(f"Unsupported vit_type {vit_type!r}. Expected vit-s, vit-b, or vit-l.")
    return normalized


def _vit_spec(vit_type: str) -> dict[str, Any]:
    normalized = _normalize_vit_type(vit_type)
    specs = {
        "vit-s": {
            "timm_name": "vit_small_patch14_dinov2",
            "embed_dim": 384,
            "layers": [2, 5, 8, 11],
            "out_channels": [48, 96, 192, 384],
        },
        "vit-b": {
            "timm_name": "vit_base_patch14_dinov2",
            "embed_dim": 768,
            "layers": [5, 7, 9, 11],
            "out_channels": [96, 192, 384, 768],
        },
        "vit-l": {
            "timm_name": "vit_large_patch14_dinov2",
            "embed_dim": 1024,
            "layers": [11, 15, 19, 23],
            "out_channels": [128, 256, 512, 1024],
        },
    }
    return specs[normalized]


class DA3ViewMetaExtractor(nn.Module):
    """Run DA3 on context images and return GS-resolution depth, intrinsics, and C2W poses."""

    def __init__(
        self,
        model_name: str,
        ref_view_strategy: str = "middle",
        da3_model: Any | None = None,
        export_feat_layers: list[int] | None = None,
    ) -> None:
        super().__init__()
        self.model_name = model_name
        self.ref_view_strategy = ref_view_strategy
        self.export_feat_layers = list(export_feat_layers or [])
        self.da3_model = da3_model if da3_model is not None else self._load_da3_model()
        if isinstance(self.da3_model, nn.Module):
            self.da3_model.eval()
            for parameter in self.da3_model.parameters():
                parameter.requires_grad_(False)

    def forward(self, context: dict, image: Tensor) -> dict[str, Any]:
        return self._infer_da3(context, image)

    def _load_da3_model(self):
        _ensure_da3_src_on_path()
        from depth_anything_3.api import DepthAnything3

        return DepthAnything3.from_pretrained(self.model_name)

    @torch.no_grad()
    def _infer_da3(self, context: dict, image: Tensor) -> dict[str, Any]:
        if self.da3_model is None:
            raise RuntimeError("DA3 is enabled, but no DA3 model is initialized.")

        da3_input = context.get("da3_input")
        if da3_input is None:
            raise KeyError("Expected context['da3_input'] from the dataloader. DA3 image preprocessing must not run in RTGSModel.")
        if da3_input.ndim == 4:
            da3_input = da3_input.unsqueeze(0)
        if da3_input.ndim != 5:
            raise ValueError(f"Expected DA3 input shape (B,V,3,H,W), got {tuple(da3_input.shape)}")

        batch, views = da3_input.shape[:2]
        gs_shape = tuple(image.shape[-2:])
        raw_output = self.da3_model(
            da3_input,
            None,
            None,
            self.export_feat_layers,
            False,
            False,
            self.ref_view_strategy,
        )
        depth_da3 = self._extract_batched_depth(raw_output, image)
        intrinsics_da3 = self._extract_batched_intrinsics(raw_output, image, batch, views)
        extrinsics_c2w = self._da3_extrinsics_to_c2w(self._extract_batched_extrinsics(raw_output, image, batch, views))
        source_shape = tuple(depth_da3.shape[-2:])

        result: dict[str, Tensor | list[Tensor]] = {
            "depth": self._resize_depth(depth_da3, gs_shape),
            "intrinsics": self._scale_intrinsics(intrinsics_da3, source_shape, gs_shape),
            "extrinsics": extrinsics_c2w,
        }
        features = self._extract_batched_features(raw_output, image, batch, views)
        if features:
            result["features"] = features
        return result

    def _resize_depth(self, depth: Tensor, image_shape: tuple[int, int]) -> Tensor:
        batch, views = depth.shape[:2]
        return F.interpolate(
            depth.reshape(batch * views, 1, *depth.shape[-2:]),
            size=image_shape,
            mode="bilinear",
            align_corners=False,
        ).reshape(batch, views, *image_shape)

    def _scale_intrinsics(self, intrinsics: Tensor, source_shape: tuple[int, int], target_shape: tuple[int, int]) -> Tensor:
        scaled = intrinsics.clone()
        scaled[..., 0, :] *= float(target_shape[1]) / float(source_shape[1])
        scaled[..., 1, :] *= float(target_shape[0]) / float(source_shape[0])
        return scaled

    def _extract_batched_depth(self, raw_output: dict[str, Tensor], image: Tensor) -> Tensor:
        depth = raw_output["depth"].to(device=image.device, dtype=image.dtype)
        if depth.ndim == 5 and depth.shape[-1] == 1:
            depth = depth[..., 0]
        elif depth.ndim == 5 and depth.shape[2] == 1:
            depth = depth[:, :, 0]
        if depth.ndim != 4:
            raise ValueError(f"Expected batched DA3 depth shape (B,V,H,W), got {tuple(depth.shape)}")
        return depth

    def _extract_batched_intrinsics(self, raw_output: dict[str, Tensor], image: Tensor, batch: int, views: int) -> Tensor:
        intrinsics = raw_output.get("intrinsics")
        if intrinsics is None:
            raise KeyError("DA3 raw output did not contain intrinsics.")
        intrinsics = intrinsics.to(device=image.device, dtype=image.dtype)
        if intrinsics.shape != (batch, views, 3, 3):
            raise ValueError(f"Expected batched DA3 intrinsics shape {(batch, views, 3, 3)}, got {tuple(intrinsics.shape)}")
        return intrinsics

    def _extract_batched_extrinsics(self, raw_output: dict[str, Tensor], image: Tensor, batch: int, views: int) -> Tensor:
        extrinsics = raw_output.get("extrinsics")
        if extrinsics is None:
            raise KeyError("DA3 raw output did not contain extrinsics.")
        extrinsics = extrinsics.to(device=image.device, dtype=image.dtype)
        if extrinsics.shape[-2:] not in ((3, 4), (4, 4)) or extrinsics.shape[:2] != (batch, views):
            raise ValueError(f"Expected batched DA3 extrinsics shape (B,V,3,4) or (B,V,4,4), got {tuple(extrinsics.shape)}")
        return extrinsics

    def _extract_batched_features(self, raw_output: dict[str, Any], image: Tensor, batch: int, views: int) -> list[Tensor]:
        features = None
        for key in ("features", "feature_maps", "intermediate_features", "intermediate_feats", "auxiliary_features", "da3_features"):
            if key in raw_output:
                features = raw_output[key]
                break
        if features is None:
            return []
        flattened = self._flatten_feature_collection(features)
        result = []
        for feature in flattened:
            feature = feature.to(device=image.device, dtype=image.dtype)
            if feature.ndim == 4 and feature.shape[0] == batch * views:
                result.append(feature)
            elif feature.ndim == 5 and feature.shape[:2] == (batch, views):
                result.append(feature)
            else:
                raise ValueError(
                    "Expected DA3 feature map shape (B,V,C,H,W) or (B*V,C,H,W), "
                    f"got {tuple(feature.shape)}"
                )
        return result

    def _flatten_feature_collection(self, value: Any) -> list[Tensor]:
        if torch.is_tensor(value):
            return [value]
        if isinstance(value, dict):
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

    def _da3_extrinsics_to_c2w(self, extrinsics: Tensor) -> Tensor:
        if extrinsics.shape[-2:] == (3, 4):
            padded = torch.eye(4, device=extrinsics.device, dtype=extrinsics.dtype).repeat(*extrinsics.shape[:-2], 1, 1)
            padded[..., :3, :4] = extrinsics
            extrinsics = padded
        if extrinsics.shape[-2:] != (4, 4):
            raise ValueError(f"Expected DA3 extrinsics shape (V,3,4) or (V,4,4), got {tuple(extrinsics.shape)}")
        return torch.linalg.inv(extrinsics)


class SimpleGaussianAdapter(nn.Module):
    """DepthSplat-style adapter from raw per-pixel predictions to world-space Gaussians."""

    def __init__(self, scale_min: float, scale_max: float, sh_degree: int = 0) -> None:
        super().__init__()
        self.scale_min = scale_min
        self.scale_max = scale_max
        self.sh_degree = sh_degree
        self.register_buffer("sh_mask", self._build_sh_mask(sh_degree), persistent=False)

    @property
    def d_sh(self) -> int:
        return (self.sh_degree + 1) ** 2

    @property
    def d_in(self) -> int:
        return 7 + 3 * self.d_sh

    def forward(
        self,
        extrinsics: Tensor,
        intrinsics: Tensor,
        coordinates: Tensor,
        depths: Tensor,
        opacities: Tensor,
        raw_gaussians: Tensor,
        image_shape: tuple[int, int],
        input_images: Tensor,
    ) -> dict[str, Tensor]:
        scales, rotations, sh = raw_gaussians.split((3, 4, 3 * self.d_sh), dim=-1)
        scales = torch.clamp(F.softplus(scales - 4.0), min=self.scale_min, max=self.scale_max)
        rotations = rotations / rotations.norm(dim=-1, keepdim=True).clamp_min(1.0e-8)

        sh = sh.reshape(*sh.shape[:-1], 3, self.d_sh) * self.sh_mask
        image_colors = input_images.permute(0, 1, 3, 4, 2).reshape(*sh.shape[:3], 3)
        sh[..., 0] = sh[..., 0] + self._rgb_to_sh(image_colors)

        covariances = self._build_covariance(scales, rotations)
        c2w_rot = extrinsics[..., :3, :3]
        covariances = c2w_rot.unsqueeze(2) @ covariances @ c2w_rot.unsqueeze(2).transpose(-1, -2)
        means = self._lift_to_world(coordinates, depths, extrinsics, intrinsics, image_shape)

        batch, views, rays = means.shape[:3]
        colors = torch.clamp(sh[..., 0] * C0 + 0.5, 0.0, 1.0)
        return {
            "means": means.reshape(batch, views * rays, 3),
            "colors": colors.reshape(batch, views * rays, 3),
            "opacities": opacities.reshape(batch, views * rays, 1),
            "covariances": covariances.reshape(batch, views * rays, 3, 3),
            "harmonics": sh.reshape(batch, views * rays, 3, self.d_sh),
            "scales": scales.reshape(batch, views * rays, 3),
            "rotations": rotations.reshape(batch, views * rays, 4),
        }

    def make_coordinates(self, intrinsics: Tensor, offset_xy: Tensor, image_shape: tuple[int, int]) -> Tensor:
        height, width = image_shape
        yy, xx = torch.meshgrid(
            torch.arange(height, device=offset_xy.device, dtype=offset_xy.dtype),
            torch.arange(width, device=offset_xy.device, dtype=offset_xy.dtype),
            indexing="ij",
        )
        base = torch.stack((xx, yy), dim=-1).reshape(1, 1, height * width, 2)
        if self._looks_normalized_intrinsics(intrinsics):
            base = base / base.new_tensor((max(width - 1, 1), max(height - 1, 1)))
            pixel_size = base.new_tensor((1.0 / max(width, 1), 1.0 / max(height, 1)))
        else:
            pixel_size = base.new_tensor((1.0, 1.0))
        return base + (offset_xy - 0.5) * pixel_size

    def _lift_to_world(
        self,
        coordinates: Tensor,
        depths: Tensor,
        extrinsics: Tensor,
        intrinsics: Tensor,
        image_shape: tuple[int, int],
    ) -> Tensor:
        height, width = image_shape
        xy = coordinates
        if self._looks_normalized_intrinsics(intrinsics):
            xy = xy * xy.new_tensor((max(width - 1, 1), max(height - 1, 1)))
        z = depths
        fx = intrinsics[..., 0, 0].clamp_min(1.0e-8).unsqueeze(-1)
        fy = intrinsics[..., 1, 1].clamp_min(1.0e-8).unsqueeze(-1)
        cx = intrinsics[..., 0, 2].unsqueeze(-1)
        cy = intrinsics[..., 1, 2].unsqueeze(-1)
        x = (xy[..., 0] - cx) / fx * z
        y = (xy[..., 1] - cy) / fy * z
        points_cam = torch.stack((x, y, z, torch.ones_like(z)), dim=-1)
        return torch.einsum("bvij,bvrj->bvri", extrinsics, points_cam)[..., :3]

    def _build_covariance(self, scales: Tensor, rotations: Tensor) -> Tensor:
        scale = scales.diag_embed()
        rotation = self._quaternion_to_matrix(rotations)
        return rotation @ scale @ scale.transpose(-1, -2) @ rotation.transpose(-1, -2)

    def _quaternion_to_matrix(self, quaternions: Tensor) -> Tensor:
        i, j, k, r = torch.unbind(quaternions, dim=-1)
        two_s = 2.0 / (quaternions.square().sum(dim=-1) + 1.0e-8)
        matrix = torch.stack(
            (
                1 - two_s * (j * j + k * k),
                two_s * (i * j - k * r),
                two_s * (i * k + j * r),
                two_s * (i * j + k * r),
                1 - two_s * (i * i + k * k),
                two_s * (j * k - i * r),
                two_s * (i * k - j * r),
                two_s * (j * k + i * r),
                1 - two_s * (i * i + j * j),
            ),
            dim=-1,
        )
        return matrix.reshape(*quaternions.shape[:-1], 3, 3)

    def _build_sh_mask(self, degree: int) -> Tensor:
        mask = torch.ones((degree + 1) ** 2, dtype=torch.float32)
        for current_degree in range(1, degree + 1):
            mask[current_degree**2 : (current_degree + 1) ** 2] = 0.1 * 0.25**current_degree
        return mask

    def _rgb_to_sh(self, rgb: Tensor) -> Tensor:
        return (rgb - 0.5) / C0

    def _looks_normalized_intrinsics(self, intrinsics: Tensor) -> bool:
        focal = intrinsics[..., (0, 1), (0, 1)].detach().abs().median()
        center = intrinsics[..., (0, 1), (2, 2)].detach().abs().amax()
        return bool((focal <= 10.0 and center <= 4.0).cpu().item())


class DINOv2FeatureExtractor(nn.Module):
    """DINOv2 ViT feature extractor with DA3-style intermediate layers."""

    patch_size = 14

    def __init__(self, vit_type: str = "vit-b", pretrained: bool = True, image_size: int = 252) -> None:
        super().__init__()
        import timm

        spec = _vit_spec(vit_type)
        self.vit_type = _normalize_vit_type(vit_type)
        self.out_layers = list(spec["layers"])
        self.embed_dim = int(spec["embed_dim"])
        self.image_size = self._make_patch_aligned_size(image_size)
        self.backbone = timm.create_model(
            spec["timm_name"],
            pretrained=pretrained,
            img_size=self.image_size,
        )

    def forward(self, image: Tensor) -> list[Tensor]:
        if image.ndim != 4:
            raise ValueError(f"Expected flattened image shape (B*V,3,H,W), got {tuple(image.shape)}")
        if image.shape[-2:] != (self.image_size, self.image_size):
            image = F.interpolate(image, size=(self.image_size, self.image_size), mode="bilinear", align_corners=False)
        features = self.backbone.get_intermediate_layers(
            image,
            n=self.out_layers,
            reshape=True,
            return_prefix_tokens=False,
            norm=True,
        )
        return [feature.contiguous() for feature in features]

    def _make_patch_aligned_size(self, size: int) -> int:
        aligned = int(size) // self.patch_size * self.patch_size
        if aligned <= 0:
            raise ValueError(f"vit_image_size must be at least {self.patch_size}, got {size}")
        return aligned


class RawDPTUpsampler(nn.Module):
    """DA3-style DPT fusion neck that upsamples ViT features to the image grid."""

    def __init__(
        self,
        dim_in: int,
        features: int,
        out_channels: list[int],
    ) -> None:
        super().__init__()
        _ensure_da3_src_on_path()
        from depth_anything_3.model.dpt import _make_fusion_block, _make_scratch

        self.projects = nn.ModuleList([nn.Conv2d(dim_in, channels, kernel_size=1) for channels in out_channels])
        self.resize_layers = nn.ModuleList(
            [
                nn.ConvTranspose2d(out_channels[0], out_channels[0], kernel_size=4, stride=4, padding=0),
                nn.ConvTranspose2d(out_channels[1], out_channels[1], kernel_size=2, stride=2, padding=0),
                nn.Identity(),
                nn.Conv2d(out_channels[3], out_channels[3], kernel_size=3, stride=2, padding=1),
            ]
        )
        self.scratch = _make_scratch(list(out_channels), features, expand=False)
        self.scratch.refinenet1 = _make_fusion_block(features)
        self.scratch.refinenet2 = _make_fusion_block(features)
        self.scratch.refinenet3 = _make_fusion_block(features)
        self.scratch.refinenet4 = _make_fusion_block(features, has_residual=False)
        self.output_conv = nn.Sequential(
            nn.Conv2d(features, features, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(features, features, kernel_size=3, padding=1),
            nn.GELU(),
        )

    def forward(self, features: list[Tensor], output_shape: tuple[int, int]) -> Tensor:
        if len(features) != 4:
            raise ValueError(f"Expected four DINOv2 feature maps, got {len(features)}")
        resized = []
        for feature, project, resize in zip(features, self.projects, self.resize_layers):
            resized.append(resize(project(feature)))
        fused = self._fuse(resized)
        fused = self.output_conv(fused)
        if fused.shape[-2:] != output_shape:
            fused = F.interpolate(fused, size=output_shape, mode="bilinear", align_corners=True)
        return fused

    def _fuse(self, features: list[Tensor]) -> Tensor:
        l1, l2, l3, l4 = features
        l1_rn = self.scratch.layer1_rn(l1)
        l2_rn = self.scratch.layer2_rn(l2)
        l3_rn = self.scratch.layer3_rn(l3)
        l4_rn = self.scratch.layer4_rn(l4)
        out = self.scratch.refinenet4(l4_rn, size=l3_rn.shape[2:])
        out = self.scratch.refinenet3(out, l3_rn, size=l2_rn.shape[2:])
        out = self.scratch.refinenet2(out, l2_rn, size=l1_rn.shape[2:])
        return self.scratch.refinenet1(out, l1_rn)


class TwinGaussianHead(nn.Module):
    """Trainable RTGS twin branch: DINOv2 ViT, raw DPT neck, convolutional Gaussian decoder."""

    def __init__(self, cfg: RTGSModelConfig, output_channels: int, intrinsic_embedding_dim: int = 0) -> None:
        super().__init__()
        spec = _vit_spec(cfg.vit_type)
        self.intrinsic_embedding_dim = int(intrinsic_embedding_dim)
        self.vit = DINOv2FeatureExtractor(
            vit_type=cfg.vit_type,
            pretrained=cfg.vit_pretrained,
            image_size=cfg.vit_image_size,
        )
        self.dpt = RawDPTUpsampler(
            dim_in=int(spec["embed_dim"]),
            features=cfg.dpt_feature_channels,
            out_channels=list(spec["out_channels"]),
        )
        self.conv_head = nn.Sequential(
            nn.Conv2d(
                cfg.dpt_feature_channels + 3 + self.intrinsic_embedding_dim,
                cfg.dpt_feature_channels,
                kernel_size=3,
                padding=1,
                padding_mode="replicate",
            ),
            nn.GELU(),
            nn.Conv2d(cfg.dpt_feature_channels, output_channels, kernel_size=3, padding=1, padding_mode="replicate"),
        )
        nn.init.zeros_(self.conv_head[-1].weight[:3])
        nn.init.zeros_(self.conv_head[-1].bias[:3])

    def extract_features(self, image: Tensor) -> dict[str, Tensor | list[Tensor]]:
        features = self.vit(image)
        dpt_features = self.dpt(features, tuple(image.shape[-2:]))
        return {"vit_features": features, "dpt_features": dpt_features}

    def forward_from_features(
        self,
        image: Tensor,
        features: dict[str, Tensor | list[Tensor]],
        intrinsic_embedding: Tensor | None = None,
    ) -> Tensor:
        dpt_features = features["dpt_features"]
        if not torch.is_tensor(dpt_features):
            raise ValueError("Expected dpt_features tensor from TwinGaussianHead.extract_features")
        inputs = [dpt_features, image]
        if self.intrinsic_embedding_dim > 0:
            if intrinsic_embedding is None:
                raise ValueError("TwinGaussianHead requires intrinsic_embedding when intrinsic_embedding_dim > 0")
            if intrinsic_embedding.shape != (image.shape[0], self.intrinsic_embedding_dim):
                raise ValueError(
                    f"Expected intrinsic_embedding shape {(image.shape[0], self.intrinsic_embedding_dim)}, "
                    f"got {tuple(intrinsic_embedding.shape)}"
                )
            inputs.append(intrinsic_embedding[..., None, None].expand(-1, -1, *image.shape[-2:]))
        return self.conv_head(torch.cat(inputs, dim=1))

    def forward(self, image: Tensor, intrinsic_embedding: Tensor | None = None) -> Tensor:
        return self.forward_from_features(image, self.extract_features(image), intrinsic_embedding)


class RTGSModel(nn.Module):
    """Twin feedforward GS model: DA3 view metadata plus DINOv2/DPT Gaussian prediction."""

    def __init__(
        self,
        cfg: RTGSModelConfig | None = None,
        view_meta_extractor: nn.Module | None = None,
        decoder: nn.Module | None = None,
        gaussian_head: nn.Module | None = None,
    ) -> None:
        super().__init__()
        self.cfg = cfg or RTGSModelConfig()
        self.intrinsic_embedding_cfg = coerce_intrinsic_embedding_config(self.cfg.intrinsic_embedding)
        self.depth_refinement_cfg = coerce_depth_refinement_config(self.cfg.depth_refinement)
        self.camera_refinement_cfg = coerce_camera_refinement_config(self.cfg.camera_refinement)
        da3_feature_layers = self.depth_refinement_cfg.da3_feature_layers if self.depth_refinement_cfg.enabled else []
        self.view_meta_extractor = view_meta_extractor or DA3ViewMetaExtractor(
            model_name=self.cfg.da3_model_name,
            ref_view_strategy=self.cfg.da3_ref_view_strategy,
            export_feat_layers=da3_feature_layers,
        )
        if view_meta_extractor is not None and self.depth_refinement_cfg.enabled and hasattr(self.view_meta_extractor, "export_feat_layers"):
            self.view_meta_extractor.export_feat_layers = list(da3_feature_layers)
        self.gaussian_adapter = SimpleGaussianAdapter(
            scale_min=self.cfg.gaussian_scale_min,
            scale_max=self.cfg.gaussian_scale_max,
            sh_degree=self.cfg.sh_degree,
        )
        self.decoder = decoder or DecoderSplattingCUDA(
            DecoderSplattingCUDACfg(background_color=tuple(self.cfg.decoder_background_color))
        )

        intrinsic_dim = self.intrinsic_embedding_cfg.dim if self.intrinsic_embedding_cfg.enabled else 0
        self.intrinsic_embedding = (
            IntrinsicEmbedding(self.intrinsic_embedding_cfg.dim, self.intrinsic_embedding_cfg.hidden_dim)
            if self.intrinsic_embedding_cfg.enabled
            else None
        )
        self.depth_refiner = (
            CostVolumeDepthRefiner(self.depth_refinement_cfg, intrinsic_embedding_dim=intrinsic_dim)
            if self.depth_refinement_cfg.enabled
            else None
        )
        self.camera_refiner = (
            CameraPoseRefiner(self.camera_refinement_cfg, intrinsic_embedding_dim=intrinsic_dim)
            if self.camera_refinement_cfg.enabled
            else None
        )
        self.gaussian_head = gaussian_head or TwinGaussianHead(
            self.cfg,
            3 + self.gaussian_adapter.d_in,
            intrinsic_embedding_dim=intrinsic_dim,
        )

    def forward(self, batch: dict) -> dict[str, Tensor | dict[str, Tensor]]:
        context = batch["context"]["image"]
        if context.ndim == 4:
            context = context.unsqueeze(0)
        if context.ndim != 5:
            raise ValueError(f"Expected context image shape (B,V,3,H,W), got {tuple(context.shape)}")

        target = batch["target"]["image"]
        if target.ndim == 4:
            target = target.unsqueeze(0)
        if target.ndim != 5:
            raise ValueError(f"Expected target image shape (B,V,3,H,W), got {tuple(target.shape)}")

        batch_size, views, _, height, width = context.shape
        target_views = target.shape[1]
        combined_views = self._combine_views_for_da3(batch["context"], batch["target"], context, target)
        combined_meta = dict(self.view_meta_extractor(combined_views, combined_views["image"]))
        auxiliary_losses: dict[str, Tensor] = {}
        diagnostics: dict[str, Tensor] = {}
        combined_intrinsic_embedding = self._encode_intrinsics(combined_meta["intrinsics"], (height, width))
        if self.camera_refiner is not None:
            camera_result = self.camera_refiner(
                combined_meta["extrinsics"],
                combined_meta["depth"],
                combined_intrinsic_embedding,
                context_views=views,
            )
            combined_meta["extrinsics"] = camera_result["extrinsics"]
            self._merge_named_tensors(auxiliary_losses, camera_result.get("losses", {}))
            self._merge_named_tensors(diagnostics, camera_result.get("diagnostics", {}))
        context_meta, target_meta = self._split_view_meta(combined_meta, views)
        context_intrinsic_embedding = (
            combined_intrinsic_embedding[:, :views] if combined_intrinsic_embedding is not None else None
        )
        flat_context = context.reshape(batch_size * views, 3, height, width)
        gaussian_features = self._extract_gaussian_features(flat_context) if self.depth_refiner is not None else None
        if self.depth_refiner is not None:
            context_meta = dict(context_meta)
            depth_result = self.depth_refiner(
                context,
                context_meta["depth"],
                context_meta["intrinsics"],
                context_meta["extrinsics"],
                context_intrinsic_embedding,
                rtgs_features=self._select_rtgs_cost_features(gaussian_features, batch_size, views),
                da3_features=context_meta.get("features"),
            )
            context_meta["depth"] = depth_result["depth"]
            self._merge_named_tensors(auxiliary_losses, depth_result.get("losses", {}))
            self._merge_named_tensors(diagnostics, depth_result.get("diagnostics", {}))
        intrinsics = context_meta["intrinsics"]
        c2w = context_meta["extrinsics"]
        depth = context_meta["depth"]

        flat_intrinsic_embedding = (
            context_intrinsic_embedding.reshape(batch_size * views, -1)
            if context_intrinsic_embedding is not None
            else None
        )
        raw = self._run_gaussian_head(
            flat_context,
            flat_intrinsic_embedding,
            gaussian_features,
        ).reshape(batch_size, views, -1, height, width)
        raw = raw.permute(0, 1, 3, 4, 2).reshape(batch_size, views, height * width, -1)

        opacities = raw[..., :1].sigmoid()
        offset_xy = raw[..., 1:3].sigmoid()
        raw_gaussians = raw[..., 3:]
        depths = depth.reshape(batch_size, views, height * width)
        coordinates = self.gaussian_adapter.make_coordinates(intrinsics, offset_xy, (height, width))

        gaussians = self.gaussian_adapter(
            c2w,
            intrinsics,
            coordinates,
            depths,
            opacities,
            raw_gaussians,
            (height, width),
            context,
        )
        colors_per_view = gaussians["colors"].reshape(batch_size, views, height, width, 3)
        rgb = colors_per_view.mean(dim=1).permute(0, 3, 1, 2).contiguous()
        render = self.decoder(
            gaussians,
            target_meta["extrinsics"],
            target_meta["intrinsics"],
            self._prepare_bounds(batch["target"], "near", batch_size, target_views, context.device, context.dtype, default=0.1),
            self._prepare_bounds(batch["target"], "far", batch_size, target_views, context.device, context.dtype, default=100.0),
            (height, width),
        )
        return {
            "rgb": rgb,
            "render": render,
            "gaussians": gaussians,
            "view_meta": context_meta,
            "context_view_meta": context_meta,
            "target_view_meta": target_meta,
            "auxiliary_losses": auxiliary_losses,
            "diagnostics": diagnostics,
        }

    def _combine_views_for_da3(self, context_views: dict, target_views: dict, context_image: Tensor, target_image: Tensor) -> dict[str, Tensor]:
        context_da3 = context_views.get("da3_input")
        target_da3 = target_views.get("da3_input")
        if context_da3 is None or target_da3 is None:
            raise KeyError("Expected both context['da3_input'] and target['da3_input'] for batched DA3 metadata extraction.")
        if context_da3.ndim == 4:
            context_da3 = context_da3.unsqueeze(0)
        if target_da3.ndim == 4:
            target_da3 = target_da3.unsqueeze(0)
        return {
            "da3_input": torch.cat([context_da3, target_da3], dim=1),
            "image": torch.cat([context_image, target_image], dim=1),
        }

    def _split_view_meta(self, meta: dict[str, Any], context_views: int) -> tuple[dict[str, Any], dict[str, Any]]:
        context_meta = {key: self._slice_view_value(value, 0, context_views) for key, value in meta.items()}
        target_meta = {key: self._slice_view_value(value, context_views, None) for key, value in meta.items()}
        return context_meta, target_meta

    def _slice_view_value(self, value: Any, start: int, stop: int | None) -> Any:
        if torch.is_tensor(value):
            return value[:, start:stop]
        if isinstance(value, list):
            return [self._slice_view_value(item, start, stop) for item in value]
        if isinstance(value, tuple):
            return tuple(self._slice_view_value(item, start, stop) for item in value)
        if isinstance(value, dict):
            return {key: self._slice_view_value(item, start, stop) for key, item in value.items()}
        return value

    def _encode_intrinsics(self, intrinsics: Tensor, image_shape: tuple[int, int]) -> Tensor | None:
        if self.intrinsic_embedding is None:
            return None
        return self.intrinsic_embedding(intrinsics, image_shape)

    def _merge_named_tensors(self, target: dict[str, Tensor], source: Any) -> None:
        if not isinstance(source, dict):
            return
        for key, value in source.items():
            if torch.is_tensor(value):
                target[str(key)] = value

    def _extract_gaussian_features(self, image: Tensor) -> Any:
        extractor = getattr(self.gaussian_head, "extract_features", None)
        if extractor is None:
            return None
        return extractor(image)

    def _select_rtgs_cost_features(self, features: Any, batch_size: int, views: int) -> Any:
        if isinstance(features, dict):
            dpt_features = features.get("dpt_features")
            if torch.is_tensor(dpt_features) and dpt_features.ndim == 4 and dpt_features.shape[0] == batch_size * views:
                return dpt_features.reshape(batch_size, views, dpt_features.shape[1], *dpt_features.shape[-2:])
            return dpt_features
        return None

    def _run_gaussian_head(self, image: Tensor, intrinsic_embedding: Tensor | None, features: Any = None) -> Tensor:
        if features is not None:
            from_features = getattr(self.gaussian_head, "forward_from_features", None)
            if from_features is not None:
                return from_features(image, features, intrinsic_embedding)
        if intrinsic_embedding is not None:
            return self.gaussian_head(image, intrinsic_embedding)
        try:
            return self.gaussian_head(image, None)
        except TypeError:
            return self.gaussian_head(image)

    def _prepare_bounds(
        self,
        views: dict,
        key: str,
        batch_size: int,
        num_views: int,
        device: torch.device,
        dtype: torch.dtype,
        default: float,
    ) -> Tensor:
        value = views.get(key)
        if value is None:
            return torch.full((batch_size, num_views), default, dtype=dtype, device=device)
        value = value.to(device=device, dtype=dtype)
        if value.ndim == 1:
            value = value.unsqueeze(0)
        if value.shape != (batch_size, num_views):
            value = value.reshape(batch_size, num_views)
        return value
