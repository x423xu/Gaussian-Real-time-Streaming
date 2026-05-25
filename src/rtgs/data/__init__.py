from __future__ import annotations

from .dataset_config import DatasetConfig, load_dataset_config
from .dl3dv import DL3DVDataset
from .re10k import RealEstate10kDataset
from .re10k_unposed import RealEstate10kUnposedDataset
from .types import DatasetStage
from .view_sampler import build_view_sampler

DATASETS = {
    "re10k": RealEstate10kDataset,
    "re10k_unbounded": RealEstate10kDataset,
    "dl3dv": DL3DVDataset,
    "re10k_unposed": RealEstate10kUnposedDataset,
}


def build_dataset(cfg: DatasetConfig, stage: DatasetStage | str, view_sampler=None):
    stage = DatasetStage(stage)
    if view_sampler is None:
        view_sampler = build_view_sampler(cfg.view_sampler, stage, cfg.overfit_to_scene is not None, cfg.cameras_are_circular)
    return DATASETS[cfg.name](cfg, stage, view_sampler)

__all__ = ["DatasetConfig", "DatasetStage", "build_dataset", "build_view_sampler", "load_dataset_config"]