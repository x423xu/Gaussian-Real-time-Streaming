from __future__ import annotations

import base64
import io
import json
import threading
from dataclasses import dataclass, field
from pathlib import Path
import sys
from typing import Any

import numpy as np
import torch
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from rtgs.config import load_typed_root_config
from rtgs.data import DatasetStage
from rtgs.data.dataloader import build_dataloader
from rtgs.main import build_rtgs_dataset, build_rtgs_model, configure_reproducibility
from rtgs.training import move_to_device


@dataclass(slots=True)
class PlaybackSnapshot:
    index: int
    timestamp: int
    context_indices: list[int]
    render_index: int
    gaussians: dict[str, torch.Tensor]
    base_extrinsic: torch.Tensor
    base_intrinsic: torch.Tensor
    near: torch.Tensor
    far: torch.Tensor
    preview_rgb: torch.Tensor


@dataclass(slots=True)
class PlaybackState:
    scene: str
    checkpoint_path: Path
    num_frames: int
    gap: int
    stride: int
    snapshots: list[PlaybackSnapshot] = field(default_factory=list)
    status: str = "idle"
    message: str = ""

    @property
    def m_fps(self) -> int:
        return len(self.snapshots)


def latest_twin_checkpoint(repo: Path = REPO_ROOT) -> Path:
    candidates = sorted((repo / "outputs").glob("rtgs_twin_vitb_2k_eval_*/checkpoint_step_*.pt"))
    if candidates:
        return candidates[-1]
    fallback = repo / "outputs" / "rtgs_twin_vitb_2k_eval_20260526_120545" / "checkpoint_step_002000.pt"
    if fallback.is_file():
        return fallback
    raise FileNotFoundError("No twin RTGS checkpoint found.")


def tensor_to_png_data_url(image: torch.Tensor) -> str:
    image = image.detach().cpu().clamp(0.0, 1.0)
    array = (image.permute(1, 2, 0).numpy() * 255.0).round().astype("uint8")
    pil = Image.fromarray(array, mode="RGB")
    buffer = io.BytesIO()
    pil.save(buffer, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")


def build_pair_starts(num_frames: int, gap: int, stride: int) -> list[int]:
    if num_frames <= gap:
        raise ValueError(f"Need at least {gap + 1} frames, got {num_frames}.")
    starts = list(range(0, num_frames - gap, stride))
    final_start = num_frames - gap - 1
    if starts[-1] != final_start:
        starts.append(final_start)
    return starts


def make_cfg(scene: str, context_indices: list[int], render_index: int, checkpoint_path: Path, device: str):
    return load_typed_root_config(
        {
            "seed": 111123,
            "runtime": {"device": device},
            "dataset": {
                "name": "re10k_unposed",
                "roots": ["/data0/xxy/data/re10k"],
                "split": "test",
                "config_path": "config/dataset/re10k_unposed.yaml",
                "overfit_to_scene": scene,
                "image_shape": [256, 256],
                "da3_image_shape": [336, 336],
                "num_workers": 0,
                "seed": 111123,
                "view_sampler": {
                    "name": "arbitrary",
                    "num_context_views": 2,
                    "num_target_views": 1,
                    "context_views": context_indices,
                    "target_views": [render_index],
                },
            },
            "model": {
                "name": "rtgs_model",
                "vit_type": "vit-b",
                "vit_image_size": 252,
                "dpt_feature_channels": 128,
                "da3_model_name": "depth-anything/DA3-BASE",
                "da3_ref_view_strategy": "middle",
                "sh_degree": 3,
            },
            "train": {"batch_size": 1},
            "output_dir": str(checkpoint_path.parent),
        }
    )


def rotation_matrix_xyz(rx: float, ry: float, rz: float, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    sx, cx = torch.sin(torch.tensor(rx, device=device, dtype=dtype)), torch.cos(torch.tensor(rx, device=device, dtype=dtype))
    sy, cy = torch.sin(torch.tensor(ry, device=device, dtype=dtype)), torch.cos(torch.tensor(ry, device=device, dtype=dtype))
    sz, cz = torch.sin(torch.tensor(rz, device=device, dtype=dtype)), torch.cos(torch.tensor(rz, device=device, dtype=dtype))
    rxm = torch.stack(
        [
            torch.stack([torch.ones_like(cx), torch.zeros_like(cx), torch.zeros_like(cx)]),
            torch.stack([torch.zeros_like(cx), cx, -sx]),
            torch.stack([torch.zeros_like(cx), sx, cx]),
        ]
    )
    rym = torch.stack(
        [
            torch.stack([cy, torch.zeros_like(cy), sy]),
            torch.stack([torch.zeros_like(cy), torch.ones_like(cy), torch.zeros_like(cy)]),
            torch.stack([-sy, torch.zeros_like(cy), cy]),
        ]
    )
    rzm = torch.stack(
        [
            torch.stack([cz, -sz, torch.zeros_like(cz)]),
            torch.stack([sz, cz, torch.zeros_like(cz)]),
            torch.stack([torch.zeros_like(cz), torch.zeros_like(cz), torch.ones_like(cz)]),
        ]
    )
    return rzm @ rym @ rxm


def apply_camera_delta(base_c2w: torch.Tensor, camera: dict[str, float]) -> torch.Tensor:
    c2w = base_c2w.clone()
    device, dtype = c2w.device, c2w.dtype
    rotation = rotation_matrix_xyz(
        float(camera.get("rx", 0.0)),
        float(camera.get("ry", 0.0)),
        float(camera.get("rz", 0.0)),
        device,
        dtype,
    )
    translation = torch.tensor(
        [float(camera.get("tx", 0.0)), float(camera.get("ty", 0.0)), float(camera.get("tz", 0.0))],
        device=device,
        dtype=dtype,
    )
    c2w[:3, :3] = c2w[:3, :3] @ rotation
    c2w[:3, 3] = c2w[:3, 3] + c2w[:3, :3] @ translation
    return c2w


def append_decay_for_age(age: int) -> float:
    return 2.0 ** (-max(0, int(age)))


def concatenate_gaussians(snapshots: list[PlaybackSnapshot], decay_old_scales: bool = False) -> dict[str, torch.Tensor]:
    if not snapshots:
        raise ValueError("Cannot concatenate an empty Gaussian window.")
    keys = snapshots[0].gaussians.keys()
    window = len(snapshots)
    output = {}
    for key in keys:
        values = []
        for idx, snapshot in enumerate(snapshots):
            value = snapshot.gaussians[key]
            if decay_old_scales:
                age = window - idx - 1
                decay = append_decay_for_age(age)
                if key == "scales":
                    value = value * decay
                elif key == "covariances":
                    value = value * (decay**2)
            values.append(value)
        output[key] = torch.cat(values, dim=1).contiguous()
    return output


class PlaybackEngine:
    def __init__(self, device: str = "cuda:1", output_dir: Path | None = None) -> None:
        self.device = torch.device(device)
        if self.device.type == "cuda":
            torch.cuda.set_device(self.device)
        self.output_dir = output_dir or (REPO_ROOT / "app" / "outputs" / "playback")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.state: PlaybackState | None = None
        self.model = None
        self._lock = threading.RLock()
        self._start_lock = threading.Lock()
        self._render_lock = threading.Lock()

    def start(
        self,
        scene: str = "5aca87f95a9412c6",
        checkpoint_path: str | Path | None = None,
        num_frames: int | None = None,
        gap: int = 4,
        stride: int | None = None,
        max_snapshots: int | None = None,
    ) -> dict[str, Any]:
        with self._start_lock:
            checkpoint = Path(checkpoint_path) if checkpoint_path is not None else latest_twin_checkpoint()
            stride = stride or (gap + 1)
            probe_cfg = make_cfg(scene, [0, gap], 0, checkpoint, str(self.device))
            configure_reproducibility(probe_cfg.seed)
            probe_loader = build_dataloader(build_rtgs_dataset(probe_cfg, DatasetStage.TEST), batch_size=1, num_workers=0, seed=probe_cfg.dataset.seed)
            probe_batch = next(iter(probe_loader))
            full_length = int(probe_batch["all_ind"][0]) if torch.is_tensor(probe_batch["all_ind"]) else int(probe_batch["all_ind"])
            total_frames = full_length if num_frames is None else min(int(num_frames), full_length)
            starts = build_pair_starts(total_frames, gap, stride)
            if max_snapshots is not None:
                starts = starts[:max_snapshots]

            cfg = make_cfg(scene, [starts[0], starts[0] + gap], starts[0], checkpoint, str(self.device))
            model = build_rtgs_model(cfg).to(self.device).eval()
            checkpoint_data = torch.load(checkpoint, map_location="cpu")
            model.load_state_dict(checkpoint_data["model"], strict=True)
            state = PlaybackState(scene=scene, checkpoint_path=checkpoint, num_frames=total_frames, gap=gap, stride=stride, status="running")

            manifest_snapshots = []
            for snapshot_idx, start_idx in enumerate(starts):
                context = [start_idx, min(start_idx + gap, total_frames - 1)]
                cfg = make_cfg(scene, context, start_idx, checkpoint, str(self.device))
                loader = build_dataloader(build_rtgs_dataset(cfg, DatasetStage.TEST), batch_size=1, num_workers=0, seed=cfg.dataset.seed)
                batch = move_to_device(next(iter(loader)), self.device)
                with torch.no_grad():
                    output = model(batch)
                target_meta = output["target_view_meta"]
                snapshot = PlaybackSnapshot(
                    index=snapshot_idx,
                    timestamp=start_idx,
                    context_indices=context,
                    render_index=start_idx,
                    gaussians={key: value.detach() for key, value in output["gaussians"].items()},
                    base_extrinsic=target_meta["extrinsics"][0, 0].detach(),
                    base_intrinsic=target_meta["intrinsics"][0, 0].detach(),
                    near=batch["target"]["near"][0, 0].detach(),
                    far=batch["target"]["far"][0, 0].detach(),
                    preview_rgb=output["render"].color[0, 0].detach().cpu(),
                )
                state.snapshots.append(snapshot)
                preview_path = self.output_dir / f"snapshot_{snapshot_idx:03d}_frame_{start_idx:04d}.png"
                Image.open(io.BytesIO(base64.b64decode(tensor_to_png_data_url(snapshot.preview_rgb).split(",", 1)[1]))).save(preview_path)
                manifest_snapshots.append(
                    {
                        "index": snapshot_idx,
                        "timestamp": start_idx,
                        "context_indices": context,
                        "render_index": start_idx,
                        "preview": str(preview_path),
                    }
                )
                state.message = f"Generated {snapshot_idx + 1}/{len(starts)} Gaussian snapshots"
                torch.cuda.empty_cache()

            state.status = "ready"
            state.message = f"Ready: {len(state.snapshots)} Gaussian FPS snapshots from {total_frames} source frames"
            with self._lock:
                self.model = model
                self.state = state
                manifest = self.manifest()
                manifest["snapshots"] = manifest_snapshots
            (self.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
            return manifest

    def manifest(self) -> dict[str, Any]:
        with self._lock:
            if self.state is None:
                return {"status": "idle", "message": "Click Start to generate Gaussian playback."}
            return {
                "status": self.state.status,
                "message": self.state.message,
                "scene": self.state.scene,
                "checkpoint": str(self.state.checkpoint_path),
                "num_source_frames": self.state.num_frames,
                "gap": self.state.gap,
                "stride": self.state.stride,
                "gaussian_fps_frames": len(self.state.snapshots),
                "output_dir": str(self.output_dir),
                "snapshots": [
                    {
                        "index": item.index,
                        "timestamp": item.timestamp,
                        "context_indices": item.context_indices,
                        "render_index": item.render_index,
                    }
                    for item in self.state.snapshots
                ],
            }

    @torch.no_grad()
    def render(self, index: int, camera: dict[str, float] | None = None) -> dict[str, Any]:
        with self._lock:
            if self.state is None or not self.state.snapshots:
                raise RuntimeError("No Gaussian playback has been generated yet.")
            if self.model is None:
                raise RuntimeError("Model is not initialized.")
            index = int(index)
            if index < 0 or index >= len(self.state.snapshots):
                raise IndexError(f"Snapshot index {index} is outside [0, {len(self.state.snapshots) - 1}].")
            snapshot = self.state.snapshots[index]
            model = self.model
        with self._render_lock:
            camera = camera or {}
            c2w = apply_camera_delta(snapshot.base_extrinsic, camera)
            render = model.decoder(
                snapshot.gaussians,
                c2w.reshape(1, 1, 4, 4),
                snapshot.base_intrinsic.reshape(1, 1, 3, 3),
                snapshot.near.reshape(1, 1),
                snapshot.far.reshape(1, 1),
                tuple(snapshot.preview_rgb.shape[-2:]),
            ).color[0, 0]
            return {
                "index": snapshot.index,
                "timestamp": snapshot.timestamp,
                "context_indices": snapshot.context_indices,
                "render_index": snapshot.render_index,
                "queue_indices": [snapshot.index],
                "queue_size": 1,
                "queue_decays": [1.0],
                "image": tensor_to_png_data_url(render),
            }

    @torch.no_grad()
    def render_append(self, index: int, append_window: int = 5, camera: dict[str, float] | None = None) -> dict[str, Any]:
        with self._lock:
            if self.state is None or not self.state.snapshots:
                raise RuntimeError("No Gaussian playback has been generated yet.")
            if self.model is None:
                raise RuntimeError("Model is not initialized.")
            index = int(index)
            if index < 0 or index >= len(self.state.snapshots):
                raise IndexError(f"Snapshot index {index} is outside [0, {len(self.state.snapshots) - 1}].")
            append_window = max(1, int(append_window))
            queue_start = max(0, index - append_window + 1)
            queue = list(self.state.snapshots[queue_start : index + 1])
            snapshot = self.state.snapshots[index]
            model = self.model
            gaussians = concatenate_gaussians(queue, decay_old_scales=True)
            queue_decays = [append_decay_for_age(len(queue) - idx - 1) for idx in range(len(queue))]
        with self._render_lock:
            camera = camera or {}
            c2w = apply_camera_delta(snapshot.base_extrinsic, camera)
            render = model.decoder(
                gaussians,
                c2w.reshape(1, 1, 4, 4),
                snapshot.base_intrinsic.reshape(1, 1, 3, 3),
                snapshot.near.reshape(1, 1),
                snapshot.far.reshape(1, 1),
                tuple(snapshot.preview_rgb.shape[-2:]),
            ).color[0, 0]
            return {
                "index": snapshot.index,
                "timestamp": snapshot.timestamp,
                "context_indices": snapshot.context_indices,
                "render_index": snapshot.render_index,
                "queue_indices": [item.index for item in queue],
                "queue_size": len(queue),
                "queue_decays": queue_decays,
                "append_window": append_window,
                "image": tensor_to_png_data_url(render),
            }
