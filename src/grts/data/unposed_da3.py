from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch import Tensor


def load_da3_model(device: str = "cuda", model_name: str = "depth-anything/DA3-SMALL"):
    repo_root = Path(__file__).resolve().parents[3]
    da3_src = repo_root / "submodules" / "Depth-Anything-3" / "src"
    if str(da3_src) not in sys.path:
        sys.path.insert(0, str(da3_src))
    from depth_anything_3.api import DepthAnything3

    model = DepthAnything3.from_pretrained(model_name).to(device)
    model.eval()
    return model


def apply_da3_outputs_to_sample(
    sample: dict,
    model,
    process_res: int = 504,
    process_res_method: str = "upper_bound_resize",
    ref_view_strategy: str = "middle",
) -> dict:
    all_images = torch.cat([sample["context"]["da3_image"], sample["target"]["da3_image"]], dim=0)
    pil_images = [tensor_to_pil(image) for image in all_images]
    prediction = model.inference(
        pil_images,
        process_res=process_res,
        process_res_method=process_res_method,
        ref_view_strategy=ref_view_strategy,
    )

    depth_da3 = torch.from_numpy(np.asarray(prediction.depth)).float()
    intrinsics_da3 = torch.from_numpy(np.asarray(prediction.intrinsics)).float()
    extrinsics_c2w = da3_extrinsics_to_c2w(torch.from_numpy(np.asarray(prediction.extrinsics)).float())

    gs_shape = tuple(sample["context"]["image"].shape[-2:])
    da3_shape = tuple(sample["context"]["da3_image"].shape[-2:])
    depth_gs = resize_depth(depth_da3, gs_shape)
    intrinsics_gs = scale_intrinsics(intrinsics_da3, da3_shape, gs_shape)

    context_count = sample["context"]["da3_image"].shape[0]
    context_slice = slice(0, context_count)
    target_slice = slice(context_count, all_images.shape[0])
    attach_da3_outputs(sample["context"], context_slice, depth_gs, depth_da3, intrinsics_gs, intrinsics_da3, extrinsics_c2w)
    attach_da3_outputs(sample["target"], target_slice, depth_gs, depth_da3, intrinsics_gs, intrinsics_da3, extrinsics_c2w)
    return sample


def attach_da3_outputs(
    views: dict,
    sl: slice,
    depth_gs: Tensor,
    depth_da3: Tensor,
    intrinsics_gs: Tensor,
    intrinsics_da3: Tensor,
    extrinsics_c2w: Tensor,
) -> None:
    views["depth"] = depth_gs[sl]
    views["da3_depth"] = depth_da3[sl]
    views["intrinsics"] = intrinsics_gs[sl]
    views["da3_intrinsics"] = intrinsics_da3[sl]
    views["extrinsics"] = extrinsics_c2w[sl]


def tensor_to_pil(image: Tensor) -> Image.Image:
    image = image.detach().cpu().clamp(0, 1)
    array = (image.permute(1, 2, 0).numpy() * 255.0).round().astype("uint8")
    return Image.fromarray(array, mode="RGB")


def resize_depth(depth: Tensor, image_shape: tuple[int, int]) -> Tensor:
    return F.interpolate(depth[:, None], size=image_shape, mode="bilinear", align_corners=False)[:, 0]


def scale_intrinsics(intrinsics: Tensor, source_shape: tuple[int, int], target_shape: tuple[int, int]) -> Tensor:
    scaled = intrinsics.clone()
    scale_y = float(target_shape[0]) / float(source_shape[0])
    scale_x = float(target_shape[1]) / float(source_shape[1])
    scaled[:, 0, :] *= scale_x
    scaled[:, 1, :] *= scale_y
    return scaled


def da3_extrinsics_to_c2w(extrinsics: Tensor) -> Tensor:
    count = extrinsics.shape[0]
    w2c = torch.eye(4, dtype=torch.float32).repeat(count, 1, 1)
    w2c[:, : extrinsics.shape[1], : extrinsics.shape[2]] = extrinsics
    return torch.linalg.inv(w2c)