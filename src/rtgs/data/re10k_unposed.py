from __future__ import annotations

from io import BytesIO

import numpy as np
import torch
from PIL import Image
from torch import Tensor

from .chunk_dataset import ChunkViewDataset


class RealEstate10kUnposedDataset(ChunkViewDataset):
    default_original_shape = (360, 640)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.gs_image_shape = tuple(self.cfg.image_shape)
        self.da3_image_shape = tuple(self.cfg.extra.get("da3_image_shape", (504, 504)))

    def _build_example(self, raw: dict):
        scene = raw["key"]
        num_views = len(raw["images"])
        dummy_extrinsics = torch.eye(4, dtype=torch.float32).repeat(num_views, 1, 1)
        dummy_intrinsics = torch.eye(3, dtype=torch.float32).repeat(num_views, 1, 1)

        try:
            context_indices, target_indices = self.view_sampler.sample(
                scene,
                dummy_extrinsics,
                dummy_intrinsics,
                min_context_views=self.cfg.min_views,
                max_context_views=self.cfg.max_views,
            )
        except ValueError:
            return None

        if self.cfg.sort_context_index:
            context_indices = context_indices.sort().values
        if self.cfg.sort_target_index:
            target_indices = target_indices.sort().values

        selected_indices = torch.cat([context_indices, target_indices], dim=0)
        source_images = [self.decode_image(raw["images"][index.item()]) for index in selected_indices]
        gs_images = torch.stack([self.pil_to_tensor(self.center_crop_resize(image, self.gs_image_shape)) for image in source_images])
        da3_images = torch.stack([self.pil_to_tensor(self.center_crop_resize(image, self.da3_image_shape)) for image in source_images])

        context_count = len(context_indices)
        context_slice = slice(0, context_count)
        target_slice = slice(context_count, context_count + len(target_indices))

        return {
            "context": self.pack_views(context_indices, gs_images[context_slice], da3_images[context_slice]),
            "target": self.pack_views(target_indices, gs_images[target_slice], da3_images[target_slice]),
            "scene": scene,
            "all_ind": num_views,
        }

    def pack_views(self, indices: Tensor, image: Tensor, da3_image: Tensor) -> dict:
        return {
            "image": image,
            "da3_image": da3_image,
            "near": self.get_bound("near", len(indices)),
            "far": self.get_bound("far", len(indices)),
            "index": indices,
        }

    def decode_image(self, encoded: Tensor) -> Image.Image:
        return Image.open(BytesIO(encoded.cpu().numpy().tobytes())).convert("RGB")

    def center_crop_resize(self, image: Image.Image, image_shape: tuple[int, int]) -> Image.Image:
        width, height = image.size
        side = min(width, height)
        left = (width - side) // 2
        top = (height - side) // 2
        image = image.crop((left, top, left + side, top + side))
        return image.resize((image_shape[1], image_shape[0]), Image.Resampling.BICUBIC)

    def pil_to_tensor(self, image: Image.Image) -> Tensor:
        array = np.asarray(image, dtype=np.float32) / 255.0
        return torch.from_numpy(array).permute(2, 0, 1).contiguous()