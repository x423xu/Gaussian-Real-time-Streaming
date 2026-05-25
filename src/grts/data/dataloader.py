from __future__ import annotations

import random

import torch
from torch.utils.data import DataLoader, Dataset


def worker_init_fn(worker_id: int) -> None:
    seed = int(torch.utils.data.get_worker_info().seed) % (2**32 - 1)
    random.seed(seed)
    torch.manual_seed(seed)


def build_dataloader(dataset: Dataset, batch_size: int, num_workers: int, seed: int | None = None, persistent_workers: bool = False, shuffle: bool = False) -> DataLoader:
    generator = None
    if seed is not None:
        generator = torch.Generator()
        generator.manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        generator=generator,
        worker_init_fn=worker_init_fn,
        persistent_workers=False if num_workers == 0 else persistent_workers,
    )
