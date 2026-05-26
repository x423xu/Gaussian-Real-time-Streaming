from __future__ import annotations

from math import isqrt
from typing import Literal

import torch
from torch import Tensor

try:  # pragma: no cover - exercised only when the CUDA extension is installed.
    from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
except ImportError:  # pragma: no cover - local rtgs env currently uses the fallback path.
    GaussianRasterizationSettings = None
    GaussianRasterizer = None


C0 = 0.28209479177387814
C1 = 0.4886025119029199
C2 = (
    1.0925484305920792,
    -1.0925484305920792,
    0.31539156525252005,
    -1.0925484305920792,
    0.5462742152960396,
)
C3 = (
    -0.5900435899266435,
    2.890611442640554,
    -0.4570457994644658,
    0.3731763325901154,
    -0.4570457994644658,
    1.445305721320277,
    -0.5900435899266435,
)
DepthRenderingMode = Literal["depth", "log", "disparity", "relative_disparity"]


def homogenize_points(points: Tensor) -> Tensor:
    return torch.cat([points, torch.ones_like(points[..., :1])], dim=-1)


def get_fov(intrinsics: Tensor, image_shape: tuple[int, int]) -> tuple[Tensor, Tensor]:
    height, width = image_shape
    fx = intrinsics[..., 0, 0].clamp_min(1.0e-8)
    fy = intrinsics[..., 1, 1].clamp_min(1.0e-8)
    if bool((fx.detach().median() <= 10.0 and fy.detach().median() <= 10.0).cpu().item()):
        fov_x = 2 * torch.atan(0.5 / fx)
        fov_y = 2 * torch.atan(0.5 / fy)
    else:
        fov_x = 2 * torch.atan(torch.full_like(fx, 0.5 * width) / fx)
        fov_y = 2 * torch.atan(torch.full_like(fy, 0.5 * height) / fy)
    return fov_x, fov_y


def get_projection_matrix(near: Tensor, far: Tensor, fov_x: Tensor, fov_y: Tensor) -> Tensor:
    tan_fov_x = (0.5 * fov_x).tan()
    tan_fov_y = (0.5 * fov_y).tan()
    top = tan_fov_y * near
    bottom = -top
    right = tan_fov_x * near
    left = -right

    batch = near.shape[0]
    result = torch.zeros((batch, 4, 4), dtype=near.dtype, device=near.device)
    result[:, 0, 0] = 2 * near / (right - left)
    result[:, 1, 1] = 2 * near / (top - bottom)
    result[:, 0, 2] = (right + left) / (right - left)
    result[:, 1, 2] = (top + bottom) / (top - bottom)
    result[:, 3, 2] = 1
    result[:, 2, 2] = far / (far - near).clamp_min(1.0e-8)
    result[:, 2, 3] = -(far * near) / (far - near).clamp_min(1.0e-8)
    return result


def sh_to_rgb(gaussian_sh_coefficients: Tensor) -> Tensor:
    return torch.clamp(gaussian_sh_coefficients[..., 0] * C0 + 0.5, 0.0, 1.0)


def eval_sh_to_rgb(degree: int, gaussian_sh_coefficients: Tensor, directions: Tensor) -> Tensor:
    coeffs = gaussian_sh_coefficients
    x, y, z = directions.unbind(dim=-1)
    result = C0 * coeffs[..., 0]
    if degree >= 1:
        result = result - C1 * y[..., None] * coeffs[..., 1] + C1 * z[..., None] * coeffs[..., 2] - C1 * x[..., None] * coeffs[..., 3]
    if degree >= 2:
        xx, yy, zz = x * x, y * y, z * z
        xy, yz, xz = x * y, y * z, x * z
        result = (
            result
            + C2[0] * xy[..., None] * coeffs[..., 4]
            + C2[1] * yz[..., None] * coeffs[..., 5]
            + C2[2] * (2.0 * zz - xx - yy)[..., None] * coeffs[..., 6]
            + C2[3] * xz[..., None] * coeffs[..., 7]
            + C2[4] * (xx - yy)[..., None] * coeffs[..., 8]
        )
    if degree >= 3:
        result = (
            result
            + C3[0] * y[..., None] * (3.0 * x * x - y * y)[..., None] * coeffs[..., 9]
            + C3[1] * (x * y * z)[..., None] * coeffs[..., 10]
            + C3[2] * y[..., None] * (4.0 * z * z - x * x - y * y)[..., None] * coeffs[..., 11]
            + C3[3] * z[..., None] * (2.0 * z * z - 3.0 * x * x - 3.0 * y * y)[..., None] * coeffs[..., 12]
            + C3[4] * x[..., None] * (4.0 * z * z - x * x - y * y)[..., None] * coeffs[..., 13]
            + C3[5] * z[..., None] * (x * x - y * y)[..., None] * coeffs[..., 14]
            + C3[6] * x[..., None] * (x * x - 3.0 * y * y)[..., None] * coeffs[..., 15]
        )
    return torch.clamp(result + 0.5, 0.0, 1.0)


def render_cuda(
    extrinsics: Tensor,
    intrinsics: Tensor,
    near: Tensor,
    far: Tensor,
    image_shape: tuple[int, int],
    background_color: Tensor,
    gaussian_means: Tensor,
    gaussian_covariances: Tensor,
    gaussian_sh_coefficients: Tensor,
    gaussian_opacities: Tensor,
    scale_invariant: bool = True,
    use_sh: bool = True,
) -> Tensor:
    if gaussian_means.is_cuda:
        torch.cuda.set_device(gaussian_means.device)
    if GaussianRasterizer is None:
        return render_point_splat(
            extrinsics,
            intrinsics,
            near,
            far,
            image_shape,
            background_color,
            gaussian_means,
            gaussian_sh_coefficients,
            gaussian_opacities,
        )

    if scale_invariant:
        scale = 1 / near.clamp_min(1.0e-8)
        extrinsics = extrinsics.clone()
        extrinsics[..., :3, 3] = extrinsics[..., :3, 3] * scale[:, None]
        gaussian_covariances = gaussian_covariances * (scale[:, None, None, None] ** 2)
        gaussian_means = gaussian_means * scale[:, None, None]
        near = near * scale
        far = far * scale

    _, _, _, n = gaussian_sh_coefficients.shape
    degree = isqrt(n) - 1
    shs = gaussian_sh_coefficients.permute(0, 1, 3, 2).contiguous()
    use_sh = False
    batch = extrinsics.shape[0]
    height, width = image_shape
    fov_x, fov_y = get_fov(intrinsics, image_shape)
    projection_matrix = get_projection_matrix(near, far, fov_x, fov_y).transpose(-1, -2)
    tan_fov_x = (0.5 * fov_x).tan()
    tan_fov_y = (0.5 * fov_y).tan()
    view_matrix = extrinsics.inverse().transpose(-1, -2)
    full_projection = view_matrix @ projection_matrix
    row, col = torch.triu_indices(3, 3, device=gaussian_means.device)

    images = []
    for idx in range(batch):
        mean_gradients = torch.zeros_like(gaussian_means[idx], requires_grad=True)
        try:
            mean_gradients.retain_grad()
        except RuntimeError:
            pass
        settings = GaussianRasterizationSettings(
            image_height=height,
            image_width=width,
            tanfovx=tan_fov_x[idx].item(),
            tanfovy=tan_fov_y[idx].item(),
            bg=background_color[idx].contiguous(),
            scale_modifier=1.0,
            viewmatrix=view_matrix[idx].contiguous(),
            projmatrix=full_projection[idx].contiguous(),
            sh_degree=degree if use_sh else 0,
            campos=extrinsics[idx, :3, 3].contiguous(),
            prefiltered=False,
            debug=False,
        )
        rasterizer = GaussianRasterizer(settings)
        view_directions = torch.nn.functional.normalize(gaussian_means[idx] - extrinsics[idx, :3, 3].unsqueeze(0), dim=-1)
        colors_precomp = eval_sh_to_rgb(degree, gaussian_sh_coefficients[idx], view_directions).contiguous()
        means3d = gaussian_means[idx].contiguous()
        opacities = gaussian_opacities[idx, ..., None].contiguous()
        covariances = gaussian_covariances[idx][:, row, col].contiguous()
        sh_coefficients = shs[idx].contiguous() if use_sh else None
        image, _ = rasterizer(
            means3D=means3d,
            means2D=mean_gradients,
            shs=sh_coefficients,
            colors_precomp=None if use_sh else colors_precomp,
            opacities=opacities,
            cov3D_precomp=covariances,
        )
        images.append(image)
    return torch.stack(images, dim=0)


def render_point_splat(
    extrinsics: Tensor,
    intrinsics: Tensor,
    near: Tensor,
    far: Tensor,
    image_shape: tuple[int, int],
    background_color: Tensor,
    gaussian_means: Tensor,
    gaussian_sh_coefficients: Tensor,
    gaussian_opacities: Tensor,
) -> Tensor:
    batch = extrinsics.shape[0]
    height, width = image_shape
    colors = sh_to_rgb(gaussian_sh_coefficients)
    images = []
    for idx in range(batch):
        world_to_cam = torch.linalg.inv(extrinsics[idx])
        cam = (homogenize_points(gaussian_means[idx]) @ world_to_cam.transpose(0, 1))[..., :3]
        z = cam[..., 2].clamp_min(1.0e-8)
        x = intrinsics[idx, 0, 0] * cam[..., 0] / z + intrinsics[idx, 0, 2]
        y = intrinsics[idx, 1, 1] * cam[..., 1] / z + intrinsics[idx, 1, 2]
        px = x.round().long()
        py = y.round().long()
        valid = (z > near[idx]) & (z < far[idx]) & (px >= 0) & (px < width) & (py >= 0) & (py < height)
        flat_color = background_color[idx].reshape(3, 1).expand(3, height * width).clone()
        flat_weight = torch.zeros((1, height * width), dtype=gaussian_means.dtype, device=gaussian_means.device)
        if valid.any():
            linear = py[valid] * width + px[valid]
            alpha = gaussian_opacities[idx, valid].clamp(0.0, 1.0)
            src = colors[idx, valid] * alpha[:, None]
            flat_color = torch.zeros_like(flat_color)
            flat_color.index_add_(1, linear, src.transpose(0, 1))
            flat_weight.index_add_(1, linear, alpha.reshape(1, -1))
            weight = flat_weight.clamp(0.0, 1.0)
            flat_color = flat_color / flat_weight.clamp_min(1.0e-6) + background_color[idx].reshape(3, 1) * (1.0 - weight)
        images.append(flat_color.reshape(3, height, width).clamp(0.0, 1.0))
    return torch.stack(images, dim=0)


def render_depth_cuda(
    extrinsics: Tensor,
    intrinsics: Tensor,
    near: Tensor,
    far: Tensor,
    image_shape: tuple[int, int],
    gaussian_means: Tensor,
    gaussian_covariances: Tensor,
    gaussian_opacities: Tensor,
    scale_invariant: bool = True,
    mode: DepthRenderingMode = "depth",
) -> Tensor:
    camera_space = torch.einsum("bij,bgj->bgi", extrinsics.inverse(), homogenize_points(gaussian_means))[..., :3]
    depth = camera_space[..., 2]
    if mode == "disparity":
        depth = 1 / depth.clamp_min(1.0e-8)
    elif mode == "log":
        depth = depth.clamp_min(1.0e-8).log()
    rendered = render_cuda(
        extrinsics,
        intrinsics,
        near,
        far,
        image_shape,
        torch.zeros((extrinsics.shape[0], 3), dtype=depth.dtype, device=depth.device),
        gaussian_means,
        gaussian_covariances,
        depth[:, :, None, None].expand(-1, -1, 3, 1),
        gaussian_opacities,
        scale_invariant=scale_invariant,
        use_sh=False,
    )
    return rendered.mean(dim=1)
