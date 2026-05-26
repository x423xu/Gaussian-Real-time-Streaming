from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
from torch import nn
import torch.nn.functional as F


def move_to_device(value: Any, device: torch.device, non_blocking: bool = False):
    if torch.is_tensor(value):
        return value.to(device, non_blocking=non_blocking)
    if isinstance(value, dict):
        return {key: move_to_device(item, device, non_blocking=non_blocking) for key, item in value.items()}
    if isinstance(value, list):
        return [move_to_device(item, device, non_blocking=non_blocking) for item in value]
    return value


def compute_reconstruction_loss(output: dict, batch: dict) -> torch.Tensor:
    target = batch["target"]["image"]
    if target.ndim == 4:
        target = target.unsqueeze(0)
    render = output["render"].color
    if render.shape[1] != target.shape[1]:
        target = target[:, : render.shape[1]]
    return F.mse_loss(render, target)


def psnr_from_mse(mse: torch.Tensor) -> torch.Tensor:
    return -10.0 * torch.log10(mse.clamp_min(1.0e-10))


def _batch_scene_count(batch: dict) -> int:
    scenes = batch.get("scene")
    if isinstance(scenes, (list, tuple)):
        return len(scenes)
    if scenes is not None:
        return 1
    target = batch.get("target", {}).get("image")
    if torch.is_tensor(target) and target.ndim >= 5:
        return int(target.shape[0])
    return 1


@torch.no_grad()
def evaluate_model(model: nn.Module, loader, device: torch.device, max_batches: int | None = None) -> dict[str, float]:
    was_training = model.training
    model.eval()
    losses: list[torch.Tensor] = []
    scene_count = 0
    iterator = iter(loader)
    for batch_idx, batch in enumerate(iterator):
        if max_batches is not None and batch_idx >= max_batches:
            break
        scene_count += _batch_scene_count(batch)
        batch = move_to_device(batch, device, non_blocking=True)
        output = model(batch)
        losses.append(compute_reconstruction_loss(output, batch).detach())
    if was_training:
        model.train()
    if not losses:
        return {"eval_loss": float("nan"), "eval_psnr": float("nan"), "eval_batches": 0.0, "eval_scenes": 0.0}
    mean_loss = torch.stack(losses).mean()
    return {
        "eval_loss": float(mean_loss.cpu().item()),
        "eval_psnr": float(psnr_from_mse(mean_loss).cpu().item()),
        "eval_batches": float(len(losses)),
        "eval_scenes": float(scene_count),
    }


def run_train_step(model: nn.Module, batch: dict, optimizer: torch.optim.Optimizer, device: torch.device) -> dict[str, float]:
    model.train()
    batch = move_to_device(batch, device, non_blocking=True)
    optimizer.zero_grad(set_to_none=True)
    output = model(batch)
    loss = compute_reconstruction_loss(output, batch)
    loss.backward()
    optimizer.step()
    return {"loss": float(loss.detach().cpu().item())}


def save_training_checkpoint(model: nn.Module, metrics: list[dict[str, float]], output_dir: Path, name: str) -> None:
    torch.save({"model": model.state_dict(), "metrics": metrics}, output_dir / name)


def run_smoke_training(
    model: nn.Module,
    loader,
    steps: int,
    lr: float,
    device: torch.device,
    output_dir: Path,
    log_every: int = 1,
    save_checkpoint: bool = True,
    checkpoint_every: int | None = None,
    eval_loader=None,
    eval_every: int | None = None,
    eval_max_batches: int | None = None,
) -> list[dict[str, float]]:
    output_dir.mkdir(parents=True, exist_ok=True)
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    metrics = []
    iterator = iter(loader)
    log_path = output_dir / "train_metrics.jsonl"
    with log_path.open("w", encoding="utf-8") as log_file:
        for step in range(steps):
            try:
                batch = next(iterator)
            except StopIteration:
                iterator = iter(loader)
                batch = next(iterator)
            row = {"step": step, **run_train_step(model, batch, optimizer, device)}
            if eval_loader is not None and eval_every is not None and eval_every > 0 and ((step + 1) % eval_every == 0 or step == steps - 1):
                row.update(evaluate_model(model, eval_loader, device, eval_max_batches))
            if save_checkpoint and checkpoint_every is not None and checkpoint_every > 0 and (step + 1) % checkpoint_every == 0:
                save_training_checkpoint(model, metrics + [row], output_dir, f"checkpoint_step_{step + 1:06d}.pt")
            metrics.append(row)
            if step % max(1, log_every) == 0 or "eval_psnr" in row:
                log_file.write(json.dumps(row) + "\n")
                log_file.flush()
                eval_text = "" if "eval_psnr" not in row else f" eval_psnr={row['eval_psnr']:.3f} eval_scenes={row['eval_scenes']:.0f}"
                print(f"[RTGS] step={step} loss={row['loss']:.6f}{eval_text}")
    if save_checkpoint:
        save_training_checkpoint(model, metrics, output_dir, "final_checkpoint.pt")
    return metrics
