from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class DatasetConfig:
    name: str
    roots: list[Path]
    image_shape: tuple[int, int]
    view_sampler: dict[str, Any]
    background_color: tuple[float, float, float] = (0.0, 0.0, 0.0)
    cameras_are_circular: bool = False
    overfit_to_scene: str | None = None
    make_baseline_1: bool = False
    augment: bool = True
    baseline_epsilon: float = 1e-3
    max_fov: float = 100.0
    skip_bad_shape: bool = True
    near: float = -1.0
    far: float = -1.0
    baseline_scale_bounds: bool = True
    shuffle_val: bool = True
    test_len: int = -1
    test_chunk_interval: int = 1
    train_times_per_scene: int = 1
    test_times_per_scene: int = 1
    ori_image_shape: tuple[int, int] | None = None
    use_index_to_load_chunk: bool = False
    min_views: int = 0
    max_views: int = 0
    sort_context_index: bool = False
    sort_target_index: bool = False
    overfit_max_views: int | None = None
    extra: dict[str, Any] = field(default_factory=dict)


def _merge_dict(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _merge_dict(merged[key], value)
        else:
            merged[key] = value
    return merged


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r") as handle:
        return yaml.safe_load(handle) or {}


def load_dataset_config(path: str | Path, overrides: dict[str, Any] | None = None) -> DatasetConfig:
    path = Path(path)
    raw = _load_yaml(path)
    defaults = raw.pop("defaults", []) or []

    for entry in defaults:
        if not isinstance(entry, dict) or "view_sampler" not in entry:
            continue
        sampler_path = path.parent / "view_sampler" / f"{entry['view_sampler']}.yaml"
        raw["view_sampler"] = _load_yaml(sampler_path)

    if overrides:
        raw = _merge_dict(raw, overrides)

    known = set(DatasetConfig.__dataclass_fields__.keys()) - {"extra"}
    extra = {key: value for key, value in raw.items() if key not in known}
    values = {key: value for key, value in raw.items() if key in known}
    values["roots"] = [Path(root) for root in values.get("roots", [])]
    values["image_shape"] = tuple(values["image_shape"])
    if values.get("ori_image_shape") is not None:
        values["ori_image_shape"] = tuple(values["ori_image_shape"])
    values["background_color"] = tuple(values.get("background_color", (0.0, 0.0, 0.0)))
    values["extra"] = extra
    return DatasetConfig(**values)
