from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

try:
    from omegaconf import DictConfig, OmegaConf
except Exception:  # pragma: no cover
    DictConfig = None
    OmegaConf = None


def _default_view_sampler() -> dict[str, Any]:
    return {
        "name": "bounded",
        "num_target_views": 1,
        "num_context_views": 2,
        "min_distance_between_context_views": 2,
        "max_distance_between_context_views": 6,
        "min_distance_to_context_views": 0,
        "warm_up_steps": 0,
        "initial_min_distance_between_context_views": 2,
        "initial_max_distance_between_context_views": 6,
    }


@dataclass(slots=True)
class DatasetConfig:
    name: str = "re10k_unposed"
    roots: list[str] = field(default_factory=lambda: ["/data0/xxy/data/re10k"])
    split: str = "test"
    config_path: str = "config/dataset/re10k_unposed.yaml"
    overfit_to_scene: str | None = "5aca87f95a9412c6"
    image_shape: list[int] = field(default_factory=lambda: [256, 256])
    da3_image_shape: list[int] = field(default_factory=lambda: [336, 336])
    view_sampler: dict[str, Any] = field(default_factory=_default_view_sampler)
    evaluation_index_path: str | None = None
    num_workers: int = 4
    persistent_workers: bool = True
    pin_memory: bool = True
    prefetch_factor: int = 4
    seed: int = 111123


@dataclass(slots=True)
class ModelConfig:
    name: str = "rtgs_model"
    hidden_channels: int = 16
    vit_type: str = "vit-b"
    vit_pretrained: bool = True
    vit_image_size: int = 252
    dpt_feature_channels: int = 128
    da3_model_name: str = "depth-anything/DA3-BASE"
    da3_ref_view_strategy: str = "middle"
    gaussian_scale_min: float = 1.0e-4
    gaussian_scale_max: float = 1.0e-2
    sh_degree: int = 3
    decoder_background_color: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])


@dataclass(slots=True)
class TrainConfig:
    steps: int = 2
    batch_size: int = 1
    lr: float = 1e-3
    min_lr: float = 1e-8
    warmup_steps: int = 4000
    log_every: int = 1
    save_checkpoint: bool = True
    checkpoint_every: int | None = None


@dataclass(slots=True)
class EvalConfig:
    checkpoint_path: str | None = None
    evaluation_index_path: str | None = None
    every_n_steps: int = 5000
    eval_data_interval: int = 10
    max_batches: int | None = None
    save_renderings: bool = True
    save_rendering_limit: int = 8


@dataclass(slots=True)
class RuntimeConfig:
    device: str = "cpu"
    conda_env: str = "rtgs"
    remote_host: str = "malab"
    remote_root: str = "/data0/xxy/code/Gaussian-Real-time-Streaming"


@dataclass(slots=True)
class WandbConfig:
    enabled: bool = False
    entity: str = "xxy"
    project: str = "rtgs"
    name: str | None = None
    mode: str = "online"
    tags: list[str] = field(default_factory=list)


@dataclass(slots=True)
class RootConfig:
    mode: str = "inspect_dataset"
    seed: int = 111123
    output_dir: str = "outputs/rtgs"
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)
    wandb: WandbConfig = field(default_factory=WandbConfig)


def _to_plain_dict(cfg: Mapping[str, Any] | Any) -> dict[str, Any]:
    if DictConfig is not None and isinstance(cfg, DictConfig):
        return OmegaConf.to_container(cfg, resolve=True)  # type: ignore[return-value]
    return dict(cfg)


def _filter_dataclass(cls, raw: dict[str, Any]) -> dict[str, Any]:
    allowed = set(cls.__dataclass_fields__.keys())
    return {key: value for key, value in raw.items() if key in allowed}


def load_typed_root_config(cfg: Mapping[str, Any] | Any) -> RootConfig:
    raw = _to_plain_dict(cfg)
    return RootConfig(
        mode=raw.get("mode", "inspect_dataset"),
        seed=int(raw.get("seed", 111123)),
        output_dir=raw.get("output_dir", "outputs/rtgs"),
        dataset=DatasetConfig(**_filter_dataclass(DatasetConfig, dict(raw.get("dataset", {})))),
        model=ModelConfig(**_filter_dataclass(ModelConfig, dict(raw.get("model", {})))),
        train=TrainConfig(**_filter_dataclass(TrainConfig, dict(raw.get("train", {})))),
        eval=EvalConfig(**_filter_dataclass(EvalConfig, dict(raw.get("eval", {})))),
        runtime=RuntimeConfig(**_filter_dataclass(RuntimeConfig, dict(raw.get("runtime", {})))),
        wandb=WandbConfig(**_filter_dataclass(WandbConfig, dict(raw.get("wandb", {})))),
    )
