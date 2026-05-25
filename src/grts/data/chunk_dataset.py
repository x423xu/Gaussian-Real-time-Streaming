from __future__ import annotations

import json
from functools import cached_property
from io import BytesIO
from pathlib import Path

import torch
from PIL import Image
from torch import Tensor
from torch.utils.data import IterableDataset

from .dataset_config import DatasetConfig
from .geometry import get_fov
from .shims.augmentation import apply_augmentation
from .shims.crop import resize_example
from .types import DatasetStage
from .view_sampler import ViewSampler


class ChunkViewDataset(IterableDataset):
    default_original_shape: tuple[int, int] | None = None

    def __init__(self, cfg: DatasetConfig, stage: DatasetStage | str, view_sampler: ViewSampler):
        super().__init__()
        self.cfg = cfg
        self.stage = DatasetStage(stage)
        self.view_sampler = view_sampler
        self.near = 0.1 if cfg.near == -1 else cfg.near
        self.far = 1000.0 if cfg.far == -1 else cfg.far
        self.chunks = self._collect_chunks()

    def _collect_chunks(self) -> list[Path]:
        chunks: list[Path] = []
        for root in self.cfg.roots:
            split_root = root / self.data_stage.value
            if not split_root.is_dir():
                continue
            if self.cfg.use_index_to_load_chunk:
                with (split_root / "index.json").open("r") as handle:
                    names = sorted(set(json.load(handle).values()))
                chunks.extend(split_root / name for name in names)
            else:
                chunks.extend(sorted(split_root.glob("*.torch")))
        if self.cfg.overfit_to_scene is not None:
            chunks = [self.index[self.cfg.overfit_to_scene]] * max(1, len(chunks))
        if self.stage == DatasetStage.TEST:
            chunks = chunks[:: max(1, self.cfg.test_chunk_interval)]
        if self.stage == DatasetStage.VAL and chunks:
            chunks = chunks * max(1, int(1e6 // len(chunks)))
        return chunks

    def shuffle(self, values: list):
        indices = torch.randperm(len(values))
        return [values[i] for i in indices]

    def __iter__(self):
        chunks = list(self.chunks)
        shuffle_stages = (DatasetStage.TRAIN, DatasetStage.VAL) if self.cfg.shuffle_val else (DatasetStage.TRAIN,)
        if self.stage in shuffle_stages:
            chunks = self.shuffle(chunks)
        worker_info = torch.utils.data.get_worker_info()
        if self.stage == DatasetStage.TEST and worker_info is not None:
            chunks = [chunk for i, chunk in enumerate(chunks) if i % worker_info.num_workers == worker_info.id]
        for chunk_path in chunks:
            chunk = torch.load(chunk_path, map_location="cpu")
            if self.cfg.overfit_to_scene is not None:
                matching = [item for item in chunk if item["key"] == self.cfg.overfit_to_scene]
                if not matching:
                    continue
                chunk = matching if self.stage == DatasetStage.TEST else matching * len(chunk)
            if self.stage in shuffle_stages:
                chunk = self.shuffle(chunk)
            times = self.cfg.test_times_per_scene if self.stage == DatasetStage.TEST else self.cfg.train_times_per_scene
            for run_idx in range(int(times * len(chunk))):
                built = self._build_example(chunk[run_idx // times])
                if built is not None:
                    yield built

    def _build_example(self, raw: dict):
        extrinsics, intrinsics = self.convert_poses(raw["cameras"])
        scene = raw["key"]
        try:
            kwargs = {}
            if self.cfg.overfit_to_scene is not None and self.stage != DatasetStage.TEST and self.cfg.overfit_max_views is not None:
                kwargs["max_num_views"] = self.cfg.overfit_max_views
            context_indices, target_indices = self.view_sampler.sample(
                scene,
                extrinsics,
                intrinsics,
                min_context_views=self.cfg.min_views,
                max_context_views=self.cfg.max_views,
                **kwargs,
            )
        except ValueError:
            return None
        if self.cfg.sort_context_index:
            context_indices = context_indices.sort().values
        if self.cfg.sort_target_index:
            target_indices = target_indices.sort().values
        if (get_fov(intrinsics).rad2deg() > self.cfg.max_fov).any():
            return None
        try:
            context_images = self.convert_images([raw["images"][i.item()] for i in context_indices])
            target_images = self.convert_images([raw["images"][i.item()] for i in target_indices])
        except OSError:
            return None
        if self.cfg.skip_bad_shape and not self._valid_shapes(context_images, target_images):
            return None
        if not self._valid_extrinsics(extrinsics, context_indices, target_indices):
            return None
        result = {
            "context": {
                "extrinsics": extrinsics[context_indices],
                "intrinsics": intrinsics[context_indices],
                "image": context_images,
                "near": self.get_bound("near", len(context_indices)),
                "far": self.get_bound("far", len(context_indices)),
                "index": context_indices,
            },
            "target": {
                "extrinsics": extrinsics[target_indices],
                "intrinsics": intrinsics[target_indices],
                "image": target_images,
                "near": self.get_bound("near", len(target_indices)),
                "far": self.get_bound("far", len(target_indices)),
                "index": target_indices,
            },
            "scene": scene,
            "all_ind": int(extrinsics.shape[0]),
        }
        if self.stage == DatasetStage.TRAIN and self.cfg.augment:
            result = apply_augmentation(result)
        if tuple(context_images.shape[-2:]) != self.cfg.image_shape:
            result = resize_example(result, self.cfg.image_shape)
        return result

    def _valid_shapes(self, context_images: Tensor, target_images: Tensor) -> bool:
        expected = self.expected_source_shape
        if expected is None:
            return True
        return tuple(context_images.shape[1:]) == (3, *expected) and tuple(target_images.shape[1:]) == (3, *expected)

    def _valid_extrinsics(self, extrinsics: Tensor, context_indices: Tensor, target_indices: Tensor) -> bool:
        selected = torch.cat([extrinsics[context_indices], extrinsics[target_indices]], dim=0)
        rotations = selected[:, :3, :3]
        det = torch.det(rotations)
        if torch.isnan(det).any():
            return False
        if (selected[:, :3, 3].abs() > 1e3).any():
            return False
        return torch.allclose(det, torch.ones_like(det), atol=1e-3)

    def convert_poses(self, poses: Tensor) -> tuple[Tensor, Tensor]:
        count = poses.shape[0]
        intrinsics = torch.eye(3, dtype=torch.float32).repeat(count, 1, 1)
        fx, fy, cx, cy = poses[:, :4].T
        intrinsics[:, 0, 0] = fx
        intrinsics[:, 1, 1] = fy
        intrinsics[:, 0, 2] = cx
        intrinsics[:, 1, 2] = cy
        w2c = torch.eye(4, dtype=torch.float32).repeat(count, 1, 1)
        w2c[:, :3] = poses[:, 6:].reshape(count, 3, 4)
        return w2c.inverse(), intrinsics

    def convert_images(self, images: list[Tensor]) -> Tensor:
        decoded = []
        for image in images:
            pil = Image.open(BytesIO(image.cpu().numpy().tobytes())).convert("RGB")
            data = torch.ByteTensor(torch.ByteStorage.from_buffer(pil.tobytes()))
            data = data.reshape(pil.height, pil.width, 3).permute(2, 0, 1).float().div(255.0)
            decoded.append(data)
        return torch.stack(decoded)

    def get_bound(self, bound: str, num_views: int) -> Tensor:
        return torch.full((num_views,), float(getattr(self, bound)), dtype=torch.float32)

    @property
    def data_stage(self) -> DatasetStage:
        if self.cfg.overfit_to_scene is not None:
            return DatasetStage.TEST
        if self.stage == DatasetStage.VAL:
            return DatasetStage.TEST
        return self.stage

    @property
    def expected_source_shape(self) -> tuple[int, int] | None:
        return self.cfg.ori_image_shape or self.default_original_shape

    @cached_property
    def index(self) -> dict[str, Path]:
        merged: dict[str, Path] = {}
        stages = [self.data_stage]
        if self.cfg.overfit_to_scene is not None:
            stages = [DatasetStage.TEST, DatasetStage.TRAIN]
        for stage in stages:
            for root in self.cfg.roots:
                index_path = root / stage.value / "index.json"
                if not index_path.is_file():
                    continue
                with index_path.open("r") as handle:
                    index = json.load(handle)
                overlap = set(merged) & set(index)
                if overlap:
                    raise ValueError(f"Duplicate scene keys in dataset roots: {sorted(overlap)[:3]}")
                merged.update({key: root / stage.value / value for key, value in index.items()})
        return merged

    def __len__(self) -> int:
        base = len(self.index)
        if self.cfg.overfit_to_scene is not None:
            return 1 if self.stage == DatasetStage.TEST else 10000
        if self.stage == DatasetStage.TEST and self.cfg.test_len > 0:
            return min(base * self.cfg.test_times_per_scene, self.cfg.test_len)
        if self.stage == DatasetStage.VAL:
            return int(1e10)
        return base * self.cfg.train_times_per_scene
