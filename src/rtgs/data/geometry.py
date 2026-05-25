import torch
from torch import Tensor


def get_fov(intrinsics: Tensor) -> Tensor:
    fx = intrinsics[..., 0, 0].clamp_min(1e-8)
    fy = intrinsics[..., 1, 1].clamp_min(1e-8)
    fov_x = 2 * torch.atan(0.5 / fx)
    fov_y = 2 * torch.atan(0.5 / fy)
    return torch.maximum(fov_x, fov_y)
