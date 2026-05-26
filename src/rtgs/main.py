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
from rtgs.training import evaluate_model, move_to_device, run_smoke_training


def configure_reproducibility(seed: int) -> None:
    random.seed(seed)
    if np is not None:
        np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_rtgs_model(cfg: RootConfig) -> RTGSModel:
    return RTGSModel(
        RTGSModelConfig(
            name=cfg.model.name,
            hidden_channels=cfg.model.hidden_channels,
            vit_type=cfg.model.vit_type,
            vit_pretrained=cfg.model.vit_pretrained,
            vit_image_size=cfg.model.vit_image_size,
            dpt_feature_channels=cfg.model.dpt_feature_channels,
            da3_model_name=cfg.model.da3_model_name,
            da3_ref_view_strategy=cfg.model.da3_ref_view_strategy,
            gaussian_scale_min=cfg.model.gaussian_scale_min,
            gaussian_scale_max=cfg.model.gaussian_scale_max,
            sh_degree=cfg.model.sh_degree,
            decoder_background_color=tuple(cfg.model.decoder_background_color),
        )
    )


def evaluation_index_path(cfg: RootConfig) -> str | None:
    return cfg.eval.evaluation_index_path or cfg.dataset.evaluation_index_path


def build_rtgs_dataset_config(cfg: RootConfig, use_evaluation_index: bool = False):
    view_sampler = cfg.dataset.view_sampler
    if use_evaluation_index:
        index_path = evaluation_index_path(cfg)
        if index_path is None:
            raise ValueError("An evaluation index path is required to build the indexed eval dataset.")
        view_sampler = {
            "name": "evaluation",
            "index_path": index_path,
            "num_context_views": cfg.dataset.view_sampler.get("num_context_views", 2),
            "eval_data_interval": cfg.eval.eval_data_interval,
        }
    dataset_cfg = load_dataset_config(
        cfg.dataset.config_path,
        overrides={
            "name": cfg.dataset.name,
            "roots": cfg.dataset.roots,
            "overfit_to_scene": cfg.dataset.overfit_to_scene,
            "image_shape": cfg.dataset.image_shape,
            "da3_image_shape": cfg.dataset.da3_image_shape,
            "view_sampler": view_sampler,
        },
    )
    if use_evaluation_index:
        dataset_cfg.view_sampler = view_sampler
    return dataset_cfg


def build_rtgs_dataset(cfg: RootConfig, stage: DatasetStage = DatasetStage.TEST, use_evaluation_index: bool = False):
    dataset_cfg = build_rtgs_dataset_config(cfg, use_evaluation_index=use_evaluation_index)
    sampler = build_view_sampler(dataset_cfg.view_sampler, stage, dataset_cfg.overfit_to_scene is not None, dataset_cfg.cameras_are_circular)
    return build_dataset(dataset_cfg, stage, sampler)


def build_rtgs_dataloader(cfg: RootConfig, stage: DatasetStage = DatasetStage.TEST, use_evaluation_index: bool = False):
    return build_dataloader(
        build_rtgs_dataset(cfg, stage, use_evaluation_index=use_evaluation_index),
        batch_size=cfg.train.batch_size,
        num_workers=cfg.dataset.num_workers,
        seed=cfg.dataset.seed,
        persistent_workers=cfg.dataset.persistent_workers,
        pin_memory=cfg.dataset.pin_memory,
        prefetch_factor=cfg.dataset.prefetch_factor,
    )


def inspect_dataset(cfg: RootConfig) -> None:
    dataset_cfg = build_rtgs_dataset_config(cfg)
    dataset = build_rtgs_dataset(cfg)
    sample = next(iter(dataset))
    print(f"[RTGS] scene={sample['scene']}")
    print(f"[RTGS] view_sampler={dataset_cfg.view_sampler}")
    print(f"[RTGS] context_image={tuple(sample['context']['image'].shape)}")
    print(f"[RTGS] context_da3_image={tuple(sample['context']['da3_image'].shape)}")
    print(f"[RTGS] context_da3_input={tuple(sample['context']['da3_input'].shape)}")
    print(f"[RTGS] target_image={tuple(sample['target']['image'].shape)}")
    print(f"[RTGS] target_da3_image={tuple(sample['target']['da3_image'].shape)}")
    print(f"[RTGS] target_da3_input={tuple(sample['target']['da3_input'].shape)}")


def inspect_forward(cfg: RootConfig) -> None:
    device = torch.device(cfg.runtime.device)
    loader = build_rtgs_dataloader(cfg)
    batch = move_to_device(next(iter(loader)), device, non_blocking=True)
    model = build_rtgs_model(cfg).to(device).eval()
    with torch.no_grad():
        output = model(batch)
    print(f"[RTGS] rgb={tuple(output['rgb'].shape)}")
    print(f"[RTGS] render_color={tuple(output['render'].color.shape)}")
    print(f"[RTGS] gaussian_means={tuple(output['gaussians']['means'].shape)}")
    print(f"[RTGS] gaussian_colors={tuple(output['gaussians']['colors'].shape)}")
    print(f"[RTGS] gaussian_opacities={tuple(output['gaussians']['opacities'].shape)}")


def train_smoke(cfg: RootConfig) -> None:
    device = torch.device(cfg.runtime.device)
    loader = build_rtgs_dataloader(cfg, DatasetStage.TRAIN)
    eval_loader = None
    if evaluation_index_path(cfg) is not None:
        eval_loader = build_rtgs_dataloader(cfg, DatasetStage.TEST, use_evaluation_index=True)
    metrics = run_smoke_training(
        build_rtgs_model(cfg),
        loader,
        cfg.train.steps,
        cfg.train.lr,
        device,
        Path(cfg.output_dir),
        cfg.train.log_every,
        cfg.train.save_checkpoint,
        cfg.train.checkpoint_every,
        eval_loader,
        cfg.eval.every_n_steps,
        cfg.eval.max_batches,
    )
    print(f"[RTGS] train_smoke_steps={len(metrics)} final_loss={metrics[-1]['loss']:.6f}")


def eval_checkpoint(cfg: RootConfig) -> None:
    checkpoint_path = cfg.eval.checkpoint_path
    if checkpoint_path is None:
        raise ValueError("eval.checkpoint_path must be set for mode=eval.")
    device = torch.device(cfg.runtime.device)
    loader = build_rtgs_dataloader(cfg, DatasetStage.TEST, use_evaluation_index=evaluation_index_path(cfg) is not None)
    model = build_rtgs_model(cfg).to(device)
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(checkpoint["model"], strict=True)
    metrics = evaluate_model(model, loader, device, cfg.eval.max_batches)
    print(f"[RTGS] eval_loss={metrics['eval_loss']:.6f} eval_psnr={metrics['eval_psnr']:.3f} eval_batches={metrics['eval_batches']:.0f}")


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
    elif cfg.mode == "eval":
        eval_checkpoint(cfg)
    else:
        raise ValueError(f"Unknown mode: {cfg.mode}")


if __name__ == "__main__":
    main()
