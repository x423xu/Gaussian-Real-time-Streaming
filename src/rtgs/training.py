from __future__ import annotations

import json
import math
import time
from dataclasses import asdict, is_dataclass
from pathlib import Path
from numbers import Number
from typing import Any

import torch
from torch import nn
import torch.nn.functional as F

from .visualization import save_eval_visualizations


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


def compute_training_loss(output: dict, batch: dict) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    reconstruction = compute_reconstruction_loss(output, batch)
    total = reconstruction
    terms = {"reconstruction_loss": reconstruction}
    auxiliary = output.get("auxiliary_losses", {})
    if isinstance(auxiliary, dict):
        for name, value in auxiliary.items():
            if torch.is_tensor(value):
                total = total + value
                terms[str(name)] = value
    return total, terms


def psnr_from_mse(mse: torch.Tensor) -> torch.Tensor:
    return -10.0 * torch.log10(mse.clamp_min(1.0e-10))


def cosine_warmup_lr(step: int, total_steps: int, max_lr: float, min_lr: float = 1.0e-8, warmup_steps: int = 4000) -> float:
    if total_steps <= 0:
        return float(min_lr)
    if warmup_steps > 0 and step < warmup_steps:
        warmup_fraction = float(step + 1) / float(warmup_steps)
        return float(min_lr + (max_lr - min_lr) * min(1.0, warmup_fraction))
    if total_steps <= warmup_steps:
        return float(max_lr)
    decay_steps = max(1, total_steps - warmup_steps - 1)
    decay_progress = min(1.0, max(0.0, float(step - warmup_steps) / float(decay_steps)))
    cosine = 0.5 * (1.0 + math.cos(decay_progress * math.pi))
    return float(min_lr + (max_lr - min_lr) * cosine)


def set_optimizer_lr(optimizer: torch.optim.Optimizer, lr: float) -> None:
    for group in optimizer.param_groups:
        group["lr"] = lr


def format_duration(seconds: float) -> str:
    total = max(0, int(seconds))
    hours, remainder = divmod(total, 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def _as_wandb_config(value: Any) -> dict[str, Any]:
    if is_dataclass(value):
        return asdict(value)
    if isinstance(value, dict):
        return value
    return {}


def init_wandb_run(wandb_cfg: Any, root_cfg: Any):
    if not getattr(wandb_cfg, "enabled", False):
        return None
    try:
        import wandb
    except ImportError as exc:
        raise RuntimeError("wandb.enabled=true, but the wandb package is not installed in this environment.") from exc

    output_dir = Path(getattr(root_cfg, "output_dir", "outputs/rtgs"))
    output_dir.mkdir(parents=True, exist_ok=True)
    return wandb.init(
        entity=getattr(wandb_cfg, "entity", "xxy"),
        project=getattr(wandb_cfg, "project", "rtgs"),
        name=getattr(wandb_cfg, "name", None),
        mode=getattr(wandb_cfg, "mode", "online"),
        tags=list(getattr(wandb_cfg, "tags", [])),
        dir=str(output_dir),
        config=_as_wandb_config(root_cfg),
    )


def log_row_to_wandb(wandb_logger: Any | None, row: dict[str, Any]) -> None:
    if wandb_logger is None:
        return
    payload = {
        key: value
        for key, value in row.items()
        if key != "step" and isinstance(value, Number) and not isinstance(value, bool)
    }
    if payload:
        wandb_logger.log(payload, step=int(row["step"]))


def flatten_numeric_summary(value: Any, prefix: str = "") -> dict[str, float]:
    if isinstance(value, dict):
        result: dict[str, float] = {}
        for key, item in value.items():
            child_prefix = f"{prefix}/{key}" if prefix else str(key)
            result.update(flatten_numeric_summary(item, child_prefix))
        return result
    if isinstance(value, Number) and not isinstance(value, bool):
        return {prefix: float(value)}
    return {}


def log_visualizations_to_wandb(wandb_logger: Any | None, artifacts: dict[str, Path], step: int | None, namespace: str = "eval") -> None:
    if wandb_logger is None:
        return
    try:
        import wandb
    except ImportError:
        return
    image_suffixes = {".jpg", ".jpeg", ".png"}
    payload = {
        f"{namespace}/{name}": wandb.Image(str(path))
        for name, path in artifacts.items()
        if path.suffix.lower() in image_suffixes and path.is_file()
    }
    if payload:
        wandb_logger.log(payload, step=step)
    summary_path = artifacts.get("summary")
    if summary_path is not None and summary_path.is_file():
        try:
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            summary = {}
        scalar_payload: dict[str, float] = {}
        scalar_payload.update(
            {
                f"{namespace}/scale/{key.removeprefix('scale_statistics/')}": value
                for key, value in flatten_numeric_summary(summary.get("scale_statistics", {}), "scale_statistics").items()
            }
        )
        scalar_payload.update(
            {
                f"{namespace}/opacity/{key.removeprefix('opacity_statistics/')}": value
                for key, value in flatten_numeric_summary(summary.get("opacity_statistics", {}), "opacity_statistics").items()
            }
        )
        scalar_payload.update(
            {
                f"{namespace}/visible_ratio/{key.removeprefix('visible_ratio/')}": value
                for key, value in flatten_numeric_summary(summary.get("visible_ratio", {}), "visible_ratio").items()
            }
        )
        if scalar_payload:
            wandb_logger.log(scalar_payload, step=step)
    files = {name: path for name, path in artifacts.items() if path.is_file()}
    if files and hasattr(wandb_logger, "log_artifact"):
        step_suffix = "latest" if step is None else f"step-{int(step)}"
        artifact_name = f"{namespace}-visualizations-{step_suffix}".replace("/", "-")
        artifact = wandb.Artifact(artifact_name, type="rtgs_eval_visualization")
        for name, path in files.items():
            artifact.add_file(str(path), name=f"{name}{path.suffix}")
        wandb_logger.log_artifact(artifact)


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
def evaluate_model(
    model: nn.Module,
    loader,
    device: torch.device,
    max_batches: int | None = None,
    visualization_dir: Path | None = None,
    visualization_prefix: str = "eval",
    save_visualizations: bool = False,
    save_visualization_limit: int = 4,
    wandb_logger: Any | None = None,
    wandb_step: int | None = None,
) -> dict[str, float]:
    was_training = model.training
    model.eval()
    losses: list[torch.Tensor] = []
    visible_ratios: list[torch.Tensor] = []
    scene_count = 0
    iterator = iter(loader)
    for batch_idx, batch in enumerate(iterator):
        if max_batches is not None and batch_idx >= max_batches:
            break
        scene_count += _batch_scene_count(batch)
        batch = move_to_device(batch, device, non_blocking=True)
        output = model(batch)
        losses.append(compute_reconstruction_loss(output, batch).detach())
        visible_ratio = getattr(output["render"], "visible_ratio", None)
        if visible_ratio is not None:
            visible_ratios.append(visible_ratio.detach().float().mean())
        if save_visualizations and visualization_dir is not None and batch_idx == 0:
            artifacts = save_eval_visualizations(
                batch,
                output,
                visualization_dir,
                visualization_prefix,
                max_target_views=save_visualization_limit,
            )
            log_visualizations_to_wandb(wandb_logger, artifacts, wandb_step)
    if was_training:
        model.train()
    if not losses:
        return {"eval_loss": float("nan"), "eval_psnr": float("nan"), "eval_batches": 0.0, "eval_scenes": 0.0}
    mean_loss = torch.stack(losses).mean()
    metrics = {
        "eval_loss": float(mean_loss.cpu().item()),
        "eval_psnr": float(psnr_from_mse(mean_loss).cpu().item()),
        "eval_batches": float(len(losses)),
        "eval_scenes": float(scene_count),
    }
    if visible_ratios:
        metrics["eval_visible_ratio"] = float(torch.stack(visible_ratios).mean().cpu().item())
    return metrics


def run_train_step(model: nn.Module, batch: dict, optimizer: torch.optim.Optimizer, device: torch.device) -> dict[str, float]:
    model.train()
    batch = move_to_device(batch, device, non_blocking=True)
    optimizer.zero_grad(set_to_none=True)
    output = model(batch)
    loss, loss_terms = compute_training_loss(output, batch)
    loss.backward()
    optimizer.step()
    metrics = {"loss": float(loss.detach().cpu().item())}
    for name, value in loss_terms.items():
        metrics[name] = float(value.detach().cpu().item())
    return metrics


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
    wandb_logger=None,
    save_eval_visualizations: bool = False,
    eval_visualization_limit: int = 4,
    min_lr: float = 1.0e-8,
    warmup_steps: int = 4000,
) -> list[dict[str, float]]:
    output_dir.mkdir(parents=True, exist_ok=True)
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=min_lr)
    metrics = []
    iterator = iter(loader)
    log_path = output_dir / "train_metrics.jsonl"
    start_time = time.perf_counter()
    with log_path.open("w", encoding="utf-8") as log_file:
        for step in range(steps):
            step_start = time.perf_counter()
            try:
                batch = next(iterator)
            except StopIteration:
                iterator = iter(loader)
                batch = next(iterator)
            current_lr = cosine_warmup_lr(step, steps, lr, min_lr, warmup_steps)
            set_optimizer_lr(optimizer, current_lr)
            row = {"step": step, "lr": current_lr, **run_train_step(model, batch, optimizer, device)}
            if eval_loader is not None and eval_every is not None and eval_every > 0 and ((step + 1) % eval_every == 0 or step == steps - 1):
                eval_start = time.perf_counter()
                row.update(
                    evaluate_model(
                        model,
                        eval_loader,
                        device,
                        eval_max_batches,
                        output_dir / "eval_visualizations",
                        f"step_{step + 1:06d}",
                        save_eval_visualizations,
                        eval_visualization_limit,
                        wandb_logger,
                        step,
                    )
                )
                row["eval_time_s"] = time.perf_counter() - eval_start
            if save_checkpoint and checkpoint_every is not None and checkpoint_every > 0 and (step + 1) % checkpoint_every == 0:
                save_training_checkpoint(model, metrics + [row], output_dir, f"checkpoint_step_{step + 1:06d}.pt")
            step_end = time.perf_counter()
            elapsed_s = step_end - start_time
            completed_steps = step + 1
            avg_step_time_s = elapsed_s / max(1, completed_steps)
            eta_s = avg_step_time_s * max(0, steps - completed_steps)
            row.update(
                {
                    "step_time_s": step_end - step_start,
                    "avg_step_time_s": avg_step_time_s,
                    "elapsed_s": elapsed_s,
                    "eta_s": eta_s,
                }
            )
            metrics.append(row)
            if step % max(1, log_every) == 0 or "eval_psnr" in row:
                log_file.write(json.dumps(row) + "\n")
                log_file.flush()
                log_row_to_wandb(wandb_logger, row)
                eval_text = "" if "eval_psnr" not in row else f" eval_psnr={row['eval_psnr']:.3f} eval_scenes={row['eval_scenes']:.0f}"
                if "eval_visible_ratio" in row:
                    eval_text += f" visible={row['eval_visible_ratio']:.3f}"
                if "eval_time_s" in row:
                    eval_text += f" eval_time={row['eval_time_s']:.1f}s"
                print(
                    f"[RTGS] step={step} loss={row['loss']:.6f} lr={row['lr']:.3e}{eval_text} "
                    f"step_time={row['step_time_s']:.2f}s "
                    f"avg_step={row['avg_step_time_s']:.2f}s "
                    f"elapsed={format_duration(row['elapsed_s'])} "
                    f"eta={format_duration(row['eta_s'])}",
                    flush=True,
                )
    if save_checkpoint:
        save_training_checkpoint(model, metrics, output_dir, "final_checkpoint.pt")
    return metrics
