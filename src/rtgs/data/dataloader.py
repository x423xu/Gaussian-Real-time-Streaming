from __future__ import annotations

import random

import torch
from torch.utils.data import DataLoader, Dataset


def worker_init_fn(worker_id: int) -> None:
    seed = int(torch.utils.data.get_worker_info().seed) % (2**32 - 1)
    random.seed(seed)
    torch.manual_seed(seed)


def build_dataloader(
    dataset: Dataset,
    batch_size: int,
    num_workers: int,
    seed: int | None = None,
    persistent_workers: bool = False,
    shuffle: bool = False,
    pin_memory: bool = False,
    prefetch_factor: int | None = None,
) -> DataLoader:
    generator = None
    if seed is not None:
        generator = torch.Generator()
        generator.manual_seed(seed)
    use_workers = num_workers > 0
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        generator=generator,
        worker_init_fn=worker_init_fn,
        persistent_workers=use_workers and persistent_workers,
        pin_memory=pin_memory,
        prefetch_factor=prefetch_factor if use_workers else None,
    )
