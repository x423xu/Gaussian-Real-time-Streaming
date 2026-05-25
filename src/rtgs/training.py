from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
from torch import nn
import torch.nn.functional as F


def move_to_device(value: Any, device: torch.device):
    if torch.is_tensor(value):
        return value.to(device)
    if isinstance(value, dict):
        return {key: move_to_device(item, device) for key, item in value.items()}
    if isinstance(value, list):
        return [move_to_device(item, device) for item in value]
    return value


def compute_reconstruction_loss(output: dict, batch: dict) -> torch.Tensor:
    target = batch["target"]["image"]
    if target.ndim == 4:
        target = target.unsqueeze(0)
    return F.mse_loss(output["rgb"], target[:, 0])


def run_train_step(model: nn.Module, batch: dict, optimizer: torch.optim.Optimizer, device: torch.device) -> dict[str, float]:
    model.train()
    batch = move_to_device(batch, device)
    optimizer.zero_grad(set_to_none=True)
    output = model(batch)
    loss = compute_reconstruction_loss(output, batch)
    loss.backward()
    optimizer.step()
    return {"loss": float(loss.detach().cpu().item())}


def run_smoke_training(model: nn.Module, loader, steps: int, lr: float, device: torch.device, output_dir: Path, log_every: int = 1, save_checkpoint: bool = True) -> list[dict[str, float]]:
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
            metrics.append(row)
            if step % max(1, log_every) == 0:
                log_file.write(json.dumps(row) + "\n")
                log_file.flush()
                print(f"[RTGS] step={step} loss={row['loss']:.6f}")
    if save_checkpoint:
        torch.save({"model": model.state_dict(), "metrics": metrics}, output_dir / "final_checkpoint.pt")
    return metrics