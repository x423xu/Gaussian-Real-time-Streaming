from __future__ import annotations

import random
from pathlib import Path

import hydra
import torch
from omegaconf import DictConfig

try:
    import numpy as np
except ImportError:  # pragma: no cover
    np = None

from rtgs.config import RootConfig, load_typed_root_config
from rtgs.data import DatasetStage, build_dataset, load_dataset_config
from rtgs.data.dataloader import build_dataloader
from rtgs.data.view_sampler import build_view_sampler
from rtgs.model import RTGSModel, RTGSModelConfig
from rtgs.training import move_to_device, run_smoke_training


def configure_reproducibility(seed: int) -> None:
    random.seed(seed)
    if np is not None:
        np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_rtgs_model(cfg: RootConfig) -> RTGSModel:
    return RTGSModel(RTGSModelConfig(name=cfg.model.name, hidden_channels=cfg.model.hidden_channels))


def build_rtgs_dataset(cfg: RootConfig, stage: DatasetStage = DatasetStage.TEST):
    dataset_cfg = load_dataset_config(
        cfg.dataset.config_path,
        overrides={
            "name": cfg.dataset.name,
            "roots": cfg.dataset.roots,
            "overfit_to_scene": cfg.dataset.overfit_to_scene,
            "image_shape": cfg.dataset.image_shape,
            "da3_image_shape": cfg.dataset.da3_image_shape,
            "view_sampler": {
                "name": "bounded",
                "num_target_views": 1,
                "num_context_views": 2,
                "min_distance_between_context_views": 2,
                "max_distance_between_context_views": 6,
                "min_distance_to_context_views": 0,
                "warm_up_steps": 0,
                "initial_min_distance_between_context_views": 2,
                "initial_max_distance_between_context_views": 6,
            },
        },
    )
    sampler = build_view_sampler(dataset_cfg.view_sampler, stage, dataset_cfg.overfit_to_scene is not None, dataset_cfg.cameras_are_circular)
    return build_dataset(dataset_cfg, stage, sampler)


def inspect_dataset(cfg: RootConfig) -> None:
    dataset = build_rtgs_dataset(cfg)
    sample = next(iter(dataset))
    print(f"[RTGS] scene={sample['scene']}")
    print(f"[RTGS] context_image={tuple(sample['context']['image'].shape)}")
    print(f"[RTGS] context_da3_image={tuple(sample['context']['da3_image'].shape)}")
    print(f"[RTGS] target_image={tuple(sample['target']['image'].shape)}")
    print(f"[RTGS] target_da3_image={tuple(sample['target']['da3_image'].shape)}")


def inspect_forward(cfg: RootConfig) -> None:
    device = torch.device(cfg.runtime.device)
    loader = build_dataloader(build_rtgs_dataset(cfg), batch_size=cfg.train.batch_size, num_workers=cfg.dataset.num_workers, seed=cfg.dataset.seed)
    batch = move_to_device(next(iter(loader)), device)
    model = build_rtgs_model(cfg).to(device).eval()
    with torch.no_grad():
        output = model(batch)
    print(f"[RTGS] rgb={tuple(output['rgb'].shape)}")
    print(f"[RTGS] gaussian_means={tuple(output['gaussians']['means'].shape)}")
    print(f"[RTGS] gaussian_colors={tuple(output['gaussians']['colors'].shape)}")
    print(f"[RTGS] gaussian_opacities={tuple(output['gaussians']['opacities'].shape)}")


def train_smoke(cfg: RootConfig) -> None:
    device = torch.device(cfg.runtime.device)
    loader = build_dataloader(build_rtgs_dataset(cfg), batch_size=cfg.train.batch_size, num_workers=cfg.dataset.num_workers, seed=cfg.dataset.seed)
    metrics = run_smoke_training(
        build_rtgs_model(cfg),
        loader,
        cfg.train.steps,
        cfg.train.lr,
        device,
        Path(cfg.output_dir),
        cfg.train.log_every,
        cfg.train.save_checkpoint,
    )
    print(f"[RTGS] train_smoke_steps={len(metrics)} final_loss={metrics[-1]['loss']:.6f}")


@hydra.main(version_base=None, config_path="../../config", config_name="main")
def main(cfg_dict: DictConfig) -> None:
    cfg = load_typed_root_config(cfg_dict)
    Path(cfg.output_dir).mkdir(parents=True, exist_ok=True)
    configure_reproducibility(cfg.seed)
    print(f"[RTGS] mode={cfg.mode}")
    print(f"[RTGS] output_dir={cfg.output_dir}")
    if cfg.mode == "inspect_dataset":
        inspect_dataset(cfg)
    elif cfg.mode == "inspect_forward":
        inspect_forward(cfg)
    elif cfg.mode == "train_smoke":
        train_smoke(cfg)
    else:
        raise ValueError(f"Unknown mode: {cfg.mode}")


if __name__ == "__main__":
    main()