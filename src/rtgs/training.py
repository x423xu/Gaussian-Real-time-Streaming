from __future__ import annotations

import json
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


def psnr_from_mse(mse: torch.Tensor) -> torch.Tensor:
    return -10.0 * torch.log10(mse.clamp_min(1.0e-10))


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
    scene_count = 0
    iterator = iter(loader)
    for batch_idx, batch in enumerate(iterator):
        if max_batches is not None and batch_idx >= max_batches:
            break
        scene_count += _batch_scene_count(batch)
        batch = move_to_device(batch, device, non_blocking=True)
        output = model(batch)
        losses.append(compute_reconstruction_loss(output, batch).detach())
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
    wandb_logger=None,
    save_eval_visualizations: bool = False,
    eval_visualization_limit: int = 4,
) -> list[dict[str, float]]:
    output_dir.mkdir(parents=True, exist_ok=True)
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
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
            row = {"step": step, **run_train_step(model, batch, optimizer, device)}
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
                if "eval_time_s" in row:
                    eval_text += f" eval_time={row['eval_time_s']:.1f}s"
                print(
                    f"[RTGS] step={step} loss={row['loss']:.6f}{eval_text} "
                    f"step_time={row['step_time_s']:.2f}s "
                    f"avg_step={row['avg_step_time_s']:.2f}s "
                    f"elapsed={format_duration(row['elapsed_s'])} "
                    f"eta={format_duration(row['eta_s'])}",
                    flush=True,
                )
    if save_checkpoint:
        save_training_checkpoint(model, metrics, output_dir, "final_checkpoint.pt")
    return metrics
