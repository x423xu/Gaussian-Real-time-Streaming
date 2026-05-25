from __future__ import annotations

import torch.nn.functional as F


def resize_example(example, image_shape: tuple[int, int]):
    for group in ("context", "target"):
        images = example[group]["image"]
        if tuple(images.shape[-2:]) != tuple(image_shape):
            example[group]["image"] = F.interpolate(
                images,
                size=image_shape,
                mode="bilinear",
                align_corners=False,
            )
    return example
