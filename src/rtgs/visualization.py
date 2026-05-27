from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image, ImageDraw


def tensor_to_pil(image: torch.Tensor) -> Image.Image:
    image = image.detach().cpu().clamp(0.0, 1.0)
    array = (image.permute(1, 2, 0).numpy() * 255.0).round().astype("uint8")
    return Image.fromarray(array, mode="RGB")


def depth_to_pil(depth: torch.Tensor) -> Image.Image:
    depth = depth.detach().cpu()
    valid = torch.isfinite(depth)
    if valid.any():
        lo = torch.quantile(depth[valid], 0.02)
        hi = torch.quantile(depth[valid], 0.98)
        depth = (depth - lo) / (hi - lo).clamp_min(1.0e-8)
    depth = depth.clamp(0.0, 1.0)
    array = (depth.numpy() * 255.0).round().astype("uint8")
    return Image.fromarray(array, mode="L").convert("RGB")


def _draw_label(image: Image.Image, label: str) -> Image.Image:
    label_h = 28
    labeled = Image.new("RGB", (image.width, image.height + label_h), "white")
    labeled.paste(image, (0, label_h))
    ImageDraw.Draw(labeled).text((6, 7), label, fill=(0, 0, 0))
    return labeled


def save_sheet(items: list[tuple[str, Image.Image]], path: Path, cols: int = 4) -> Path:
    if not items:
        raise ValueError("Cannot save an empty visualization sheet.")
    tiles = [_draw_label(image, label) for label, image in items]
    cols = max(1, min(cols, len(tiles)))
    rows = (len(tiles) + cols - 1) // cols
    tile_w = max(tile.width for tile in tiles)
    tile_h = max(tile.height for tile in tiles)
    sheet = Image.new("RGB", (cols * tile_w, rows * tile_h), "white")
    for idx, tile in enumerate(tiles):
        sheet.paste(tile, ((idx % cols) * tile_w, (idx // cols) * tile_h))
    path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(path, quality=95)
    return path


def pointmap_from_depth(
    images: torch.Tensor,
    depth: torch.Tensor,
    intrinsics: torch.Tensor,
    c2w: torch.Tensor,
) -> tuple[np.ndarray, np.ndarray]:
    views, _, height, width = images.shape
    yy, xx = torch.meshgrid(
        torch.arange(height, device=depth.device, dtype=depth.dtype),
        torch.arange(width, device=depth.device, dtype=depth.dtype),
        indexing="ij",
    )
    all_points = []
    all_colors = []
    for view_idx in range(views):
        z = depth[view_idx]
        k = intrinsics[view_idx]
        x = (xx - k[0, 2]) / k[0, 0].clamp_min(1.0e-8) * z
        y = (yy - k[1, 2]) / k[1, 1].clamp_min(1.0e-8) * z
        cam = torch.stack((x, y, z, torch.ones_like(z)), dim=-1).reshape(-1, 4)
        world = (c2w[view_idx] @ cam.T).T[:, :3]
        colors = images[view_idx].permute(1, 2, 0).reshape(-1, 3)
        all_points.append(world)
        all_colors.append(colors)
    return torch.cat(all_points).detach().cpu().numpy(), torch.cat(all_colors).detach().cpu().numpy()


def save_projection(
    path: Path,
    points: np.ndarray,
    colors: np.ndarray,
    title: str,
    max_points: int = 160_000,
) -> Path:
    valid = np.isfinite(points).all(axis=1) & np.isfinite(colors).all(axis=1)
    points = points[valid]
    colors = colors[valid]
    if len(points) > max_points:
        keep = np.random.default_rng(42).choice(len(points), size=max_points, replace=False)
        points = points[keep]
        colors = colors[keep]
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), dpi=160)
    color_float = np.clip(colors, 0.0, 1.0)
    axes[0].scatter(points[:, 0], points[:, 2], s=0.2, c=color_float)
    axes[0].set_title(f"{title}: top x/z")
    axes[0].set_xlabel("world x")
    axes[0].set_ylabel("world z")
    axes[0].axis("equal")
    axes[1].scatter(points[:, 0], points[:, 1], s=0.2, c=color_float)
    axes[1].set_title(f"{title}: front x/y")
    axes[1].set_xlabel("world x")
    axes[1].set_ylabel("world y")
    axes[1].axis("equal")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)
    return path


def write_rgb_pointcloud_ply(path: Path, points: np.ndarray, colors: np.ndarray) -> int:
    valid = np.isfinite(points).all(axis=1) & np.isfinite(colors).all(axis=1)
    points = points[valid]
    colors_u8 = (np.clip(colors[valid], 0.0, 1.0) * 255.0).round().astype(np.uint8)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="ascii") as file:
        file.write("ply\nformat ascii 1.0\n")
        file.write(f"element vertex {len(points)}\n")
        file.write("property float x\nproperty float y\nproperty float z\n")
        file.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        file.write("end_header\n")
        for point, color in zip(points, colors_u8):
            file.write(f"{point[0]:.6f} {point[1]:.6f} {point[2]:.6f} {int(color[0])} {int(color[1])} {int(color[2])}\n")
    return int(len(points))


def _inverse_sigmoid(x: np.ndarray, eps: float = 1.0e-6) -> np.ndarray:
    x = np.clip(x, eps, 1.0 - eps)
    return np.log(x / (1.0 - x))


def write_supersplat_ply(path: Path, gaussians: dict[str, torch.Tensor]) -> int:
    means = gaussians["means"][0].detach().cpu().numpy().astype(np.float32)
    harmonics = gaussians["harmonics"][0].detach().cpu().numpy().astype(np.float32)
    opacities = gaussians["opacities"][0, :, 0].detach().cpu().numpy().astype(np.float32)
    scales = gaussians["scales"][0].detach().cpu().numpy().astype(np.float32)
    rotations_xyzw = gaussians["rotations"][0].detach().cpu().numpy().astype(np.float32)
    valid = (
        np.isfinite(means).all(axis=1)
        & np.isfinite(harmonics).all(axis=(1, 2))
        & np.isfinite(opacities)
        & np.isfinite(scales).all(axis=1)
        & np.isfinite(rotations_xyzw).all(axis=1)
    )
    means = means[valid]
    harmonics = harmonics[valid]
    opacities = opacities[valid]
    scales = scales[valid]
    rotations_xyzw = rotations_xyzw[valid]

    normals = np.zeros_like(means, dtype=np.float32)
    f_dc = harmonics[:, :, 0]
    f_rest = harmonics[:, :, 1:].transpose(0, 2, 1).reshape(len(means), -1)
    opacity_logits = _inverse_sigmoid(opacities).reshape(-1, 1).astype(np.float32)
    log_scales = np.log(np.clip(scales, 1.0e-8, None)).astype(np.float32)
    rotations_wxyz = np.concatenate([rotations_xyzw[:, 3:4], rotations_xyzw[:, :3]], axis=1).astype(np.float32)
    attributes = np.concatenate([means, normals, f_dc, f_rest, opacity_logits, log_scales, rotations_wxyz], axis=1).astype(np.float32)
    property_names = (
        ["x", "y", "z", "nx", "ny", "nz"]
        + [f"f_dc_{i}" for i in range(3)]
        + [f"f_rest_{i}" for i in range(f_rest.shape[1])]
        + ["opacity"]
        + [f"scale_{i}" for i in range(3)]
        + [f"rot_{i}" for i in range(4)]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as file:
        header = "ply\nformat binary_little_endian 1.0\n"
        header += f"element vertex {len(attributes)}\n"
        header += "".join(f"property float {name}\n" for name in property_names)
        header += "end_header\n"
        file.write(header.encode("ascii"))
        attributes.tofile(file)
    return int(len(attributes))


def _first_scene(value: Any) -> str:
    if isinstance(value, (list, tuple)):
        return str(value[0])
    return str(value)


def _psnr_per_target(render: torch.Tensor, target: torch.Tensor) -> list[float]:
    mse = (render - target).square().mean(dim=(1, 2, 3)).clamp_min(1.0e-10)
    return [float(value) for value in (-10.0 * torch.log10(mse)).detach().cpu()]


def save_eval_visualizations(
    batch: dict[str, Any],
    output: dict[str, Any],
    output_dir: Path,
    prefix: str,
    max_target_views: int = 4,
) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    context = batch["context"]["image"][0].detach().cpu()
    target = batch["target"]["image"][0].detach().cpu()
    render = output["render"].color[0].detach().cpu()
    target_count = min(max_target_views, target.shape[0], render.shape[0])
    context_indices = batch["context"]["index"][0].detach().cpu().tolist()
    target_indices = batch["target"]["index"][0, :target_count].detach().cpu().tolist()
    psnr_values = _psnr_per_target(render[:target_count], target[:target_count])

    context_meta = output.get("context_view_meta") or output["view_meta"]
    target_meta = output.get("target_view_meta")
    if target_meta is None:
        raise KeyError("Expected output['target_view_meta'] for eval visualization.")

    artifacts: dict[str, Path] = {}
    artifacts["diagnostic_sheet"] = save_sheet(
        [(f"context {idx}", tensor_to_pil(image)) for idx, image in zip(context_indices, context)]
        + [(f"gt {idx}", tensor_to_pil(image)) for idx, image in zip(target_indices, target[:target_count])]
        + [(f"render {idx} {psnr_values[i]:.2f}dB", tensor_to_pil(image)) for i, (idx, image) in enumerate(zip(target_indices, render[:target_count]))],
        output_dir / f"{prefix}_render_diagnostics.jpg",
        cols=max(2, target_count),
    )
    context_depth = context_meta["depth"][0].detach().cpu()
    target_depth = target_meta["depth"][0, :target_count].detach().cpu()
    artifacts["da3_depth_sheet"] = save_sheet(
        [(f"context depth {idx}", depth_to_pil(depth)) for idx, depth in zip(context_indices, context_depth)]
        + [(f"target depth {idx}", depth_to_pil(depth)) for idx, depth in zip(target_indices, target_depth)],
        output_dir / f"{prefix}_da3_depths.jpg",
        cols=max(2, target_count),
    )

    all_images = torch.cat([context, target[:target_count]], dim=0)
    all_depth = torch.cat([context_depth, target_depth], dim=0)
    all_intrinsics = torch.cat([context_meta["intrinsics"][0].detach().cpu(), target_meta["intrinsics"][0, :target_count].detach().cpu()], dim=0)
    all_extrinsics = torch.cat([context_meta["extrinsics"][0].detach().cpu(), target_meta["extrinsics"][0, :target_count].detach().cpu()], dim=0)
    da3_points, da3_colors = pointmap_from_depth(all_images, all_depth, all_intrinsics, all_extrinsics)
    artifacts["da3_pointmap_projection"] = save_projection(output_dir / f"{prefix}_da3_pointmap_projection.png", da3_points, da3_colors, "DA3 depth+pose pointmap")
    da3_ply_count = write_rgb_pointcloud_ply(output_dir / f"{prefix}_da3_pointmap_rgb.ply", da3_points, da3_colors)
    artifacts["da3_pointmap_ply"] = output_dir / f"{prefix}_da3_pointmap_rgb.ply"

    gaussian_points = output["gaussians"]["means"][0].detach().cpu().numpy()
    gaussian_colors = output["gaussians"]["colors"][0].detach().cpu().numpy()
    artifacts["gaussian_projection"] = save_projection(output_dir / f"{prefix}_rtgs_gaussian_projection.png", gaussian_points, gaussian_colors, "RTGS projected Gaussians")
    gaussian_rgb_count = write_rgb_pointcloud_ply(output_dir / f"{prefix}_rtgs_gaussian_rgb.ply", gaussian_points, gaussian_colors)
    artifacts["gaussian_rgb_ply"] = output_dir / f"{prefix}_rtgs_gaussian_rgb.ply"
    gaussian_supersplat_count = write_supersplat_ply(output_dir / f"{prefix}_rtgs_gaussians_supersplat.ply", output["gaussians"])
    artifacts["gaussian_supersplat_ply"] = output_dir / f"{prefix}_rtgs_gaussians_supersplat.ply"

    summary = {
        "scene": _first_scene(batch["scene"]),
        "context_indices": context_indices,
        "target_indices": target_indices,
        "target_psnr": psnr_values,
        "da3_point_count": da3_ply_count,
        "gaussian_rgb_point_count": gaussian_rgb_count,
        "gaussian_supersplat_count": gaussian_supersplat_count,
        "artifacts": {key: str(value) for key, value in artifacts.items()},
    }
    artifacts["summary"] = output_dir / f"{prefix}_summary.json"
    artifacts["summary"].write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return artifacts
