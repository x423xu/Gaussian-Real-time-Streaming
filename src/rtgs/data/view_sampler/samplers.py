from __future__ import annotations

import json
import random
from pathlib import Path

import torch
from torch import Tensor

from ..types import DatasetStage
from .base import ViewSampler


def _schedule(initial: int, final: int, step: int, steps: int) -> int:
    if steps <= 0:
        return final
    fraction = step / steps
    return min(initial + int((final - initial) * fraction), final)


def farthest_point_sample(xyz: Tensor, npoint: int) -> Tensor:
    device = xyz.device
    batch, count, _ = xyz.shape
    centroids = torch.zeros(batch, npoint, dtype=torch.long, device=device)
    distance = torch.full((batch, count), 1e10, device=device)
    batch_indices = torch.arange(batch, dtype=torch.long, device=device)
    barycenter = xyz.mean(dim=1, keepdim=True)
    farthest = torch.sum((xyz - barycenter) ** 2, dim=-1).max(dim=1).indices
    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = xyz[batch_indices, farthest, :].view(batch, 1, 3)
        dist = torch.sum((xyz - centroid) ** 2, dim=-1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = distance.max(dim=-1).indices
    return centroids


class AllSampler(ViewSampler):
    def sample(self, scene: str, extrinsics: Tensor, intrinsics: Tensor, device=torch.device("cpu"), **kwargs):
        indices = torch.arange(extrinsics.shape[0], device=device)
        return indices, indices


class ArbitrarySampler(ViewSampler):
    def sample(self, scene: str, extrinsics: Tensor, intrinsics: Tensor, device=torch.device("cpu"), **kwargs):
        num_views = extrinsics.shape[0]
        context = torch.randint(0, num_views, (self.num_context_views,), device=device)
        target = torch.randint(0, num_views, (self.num_target_views,), device=device)
        if self.cfg.get("context_views") is not None:
            context = torch.tensor(self.cfg["context_views"], dtype=torch.long, device=device)
        if self.cfg.get("target_views") is not None:
            target = torch.tensor(self.cfg["target_views"], dtype=torch.long, device=device)
        return context, target


class BoundedSampler(ViewSampler):
    def sample(self, scene: str, extrinsics: Tensor, intrinsics: Tensor, device=torch.device("cpu"), min_view_dist=None, max_view_dist=None, **kwargs):
        num_views = extrinsics.shape[0]
        if self.stage == DatasetStage.TEST:
            min_gap = max_gap = int(self.cfg["max_distance_between_context_views"])
        else:
            min_gap = _schedule(int(self.cfg["initial_min_distance_between_context_views"]), int(self.cfg["min_distance_between_context_views"]), self.global_step, int(self.cfg.get("warm_up_steps", 0)))
            max_gap = _schedule(int(self.cfg["initial_max_distance_between_context_views"]), int(self.cfg["max_distance_between_context_views"]), self.global_step, int(self.cfg.get("warm_up_steps", 0)))
        if not self.cameras_are_circular:
            max_gap = min(num_views - 1, max_gap)
        min_gap = max(2 * int(self.cfg.get("min_distance_to_context_views", 0)), min_gap)
        if min_view_dist is not None:
            min_gap = int(min_view_dist)
        if max_view_dist is not None:
            max_gap = int(max_view_dist)
        if max_gap < min_gap:
            raise ValueError("Example does not have enough frames")
        gap = torch.randint(min_gap, max_gap + 1, tuple(), device=device).item()
        left = torch.randint(num_views if self.cameras_are_circular else num_views - gap, tuple(), device=device).item()
        if self.stage == DatasetStage.TEST:
            left = 0
        right = left + gap
        if self.overfit:
            left = 0
            right = max_gap
        if self.stage == DatasetStage.TEST:
            target = torch.arange(left, right + 1, device=device)
        else:
            margin = int(self.cfg.get("min_distance_to_context_views", 0))
            target = torch.randint(left + margin, right + 1 - margin, (self.num_target_views,), device=device)
        if self.cameras_are_circular:
            target %= num_views
            right %= num_views
        return torch.tensor((left, right), dtype=torch.long, device=device), target


class UnboundedSampler(BoundedSampler):
    def sample(self, scene: str, extrinsics: Tensor, intrinsics: Tensor, device=torch.device("cpu"), **kwargs):
        context, _ = super().sample(scene, extrinsics, intrinsics, device=device, **kwargs)
        left, right = context.tolist()
        num_views = extrinsics.shape[0]
        if self.stage == DatasetStage.TEST:
            target = torch.arange(left, right + 1, device=device)
        else:
            max_dist = int(self.cfg.get("max_distance_to_context_views", 0))
            target = torch.randint(max(left - max_dist, 0), min(right + max_dist, num_views), (self.num_target_views,), device=device)
        return context, target


class BoundedV2Sampler(ViewSampler):
    def sample(self, scene: str, extrinsics: Tensor, intrinsics: Tensor, device=torch.device("cpu"), max_num_views=None, min_context_views=0, max_context_views=0, min_view_dist=None, max_view_dist=None):
        num_views = min(extrinsics.shape[0], max_num_views) if max_num_views is not None else extrinsics.shape[0]
        random_num_views = random.randint(min_context_views, max_context_views) if min_context_views > 0 and max_context_views > 0 and self.stage != DatasetStage.TEST else None
        if self.stage == DatasetStage.TEST:
            min_gap = max_gap = int(self.cfg["max_distance_between_context_views"])
        else:
            min_gap = _schedule(int(self.cfg["initial_min_distance_between_context_views"]), int(self.cfg["min_distance_between_context_views"]), self.global_step, int(self.cfg.get("context_gap_warm_up_steps", 0)))
            max_gap = _schedule(int(self.cfg["initial_max_distance_between_context_views"]), int(self.cfg["max_distance_between_context_views"]), self.global_step, int(self.cfg.get("context_gap_warm_up_steps", 0)))
        if min_view_dist is not None and max_view_dist is not None:
            min_gap, max_gap = int(min_view_dist), int(max_view_dist)
        total_context = random_num_views or self.num_context_views
        if random_num_views is not None:
            scale = max(max_context_views // random_num_views, 1)
            min_gap //= scale
            max_gap //= scale
        if not self.cameras_are_circular:
            max_gap = min(num_views - 1, max_gap)
        if max_gap < min_gap:
            raise ValueError("Example does not have enough frames")
        gap = torch.randint(min_gap, max_gap + 1, tuple(), device=device).item()
        left = torch.randint(0, num_views if self.cameras_are_circular else num_views - gap, tuple(), device=device).item()
        if self.stage == DatasetStage.TEST:
            left = 0
        right = left + gap
        max_target_gap = int(self.cfg.get("max_distance_to_context_views", 0))
        target_left = left - max_target_gap
        target_right = right + max_target_gap
        if not self.cameras_are_circular:
            target_left = max(0, target_left)
            target_right = min(num_views - 1, target_right)
        if self.stage == DatasetStage.TEST:
            target = torch.arange(target_left, target_right + 1, device=device)
        elif self.cfg.get("target_views_replace_sample", True):
            target = torch.randint(target_left, target_right + 1, (self.num_target_views,), device=device)
        else:
            candidates = torch.arange(target_left, target_right + 1, device=device)
            if len(candidates) < self.num_target_views:
                raise ValueError("Example does not have enough target frames")
            target = candidates[torch.randperm(len(candidates), device=device)[: self.num_target_views]]
        if self.cameras_are_circular:
            target %= num_views
            right %= num_views
        extras = []
        if total_context > 2:
            extra_count = total_context - 2
            strategy = self.cfg.get("extra_views_sampling_strategy", "random")
            if right <= left + 1:
                raise ValueError("Example does not have enough context frames")
            if strategy == "farthest_point":
                bounded = torch.arange(left, right + 1, device=extrinsics.device)
                local = farthest_point_sample(extrinsics[bounded, :3, 3].unsqueeze(0), total_context).squeeze(0)
                context = bounded[local].sort().values.to(device)
                return context.long(), target.long()
            if strategy == "equal":
                extras = torch.linspace(left, right, total_context, device=device).round().long()[1:-1].tolist()
            else:
                while len(set(extras)) != extra_count:
                    extras = torch.randint(left + 1, right, (extra_count,), device=device).tolist()
                extras = sorted(extras)
        return torch.tensor((left, *extras, right), dtype=torch.long, device=device), target.long()


class EvaluationSampler(ViewSampler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        with Path(self.cfg["index_path"]).open("r") as handle:
            self.index = json.load(handle)

    def sample(self, scene: str, extrinsics: Tensor, intrinsics: Tensor, device=torch.device("cpu"), **kwargs):
        entry = self.index.get(scene)
        if entry is None:
            raise ValueError(f"No indices available for scene {scene}")
        return (
            torch.tensor(entry["context"], dtype=torch.long, device=device),
            torch.tensor(entry["target"], dtype=torch.long, device=device),
        )
