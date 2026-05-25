from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

try:
    from omegaconf import DictConfig, OmegaConf
except Exception:  # pragma: no cover
    DictConfig = None
    OmegaConf = None


@dataclass(slots=True)
class DatasetConfig:
    name: str = "re10k_unposed"
    roots: list[str] = field(default_factory=lambda: ["/data0/xxy/data/re10k"])
    split: str = "test"
    config_path: str = "config/dataset/re10k_unposed.yaml"
    overfit_to_scene: str | None = "5aca87f95a9412c6"
    image_shape: list[int] = field(default_factory=lambda: [256, 256])
    da3_image_shape: list[int] = field(default_factory=lambda: [504, 504])
    num_workers: int = 0
    seed: int = 111123


@dataclass(slots=True)
class ModelConfig:
    name: str = "rtgs_model"
    hidden_channels: int = 16


@dataclass(slots=True)
class TrainConfig:
    steps: int = 2
    batch_size: int = 1
    lr: float = 1e-3
    log_every: int = 1
    save_checkpoint: bool = True


@dataclass(slots=True)
class RuntimeConfig:
    device: str = "cpu"
    conda_env: str = "rtgs"
    remote_host: str = "malab"
    remote_root: str = "/data0/xxy/code/Gaussian-Real-time-Streaming"


@dataclass(slots=True)
class RootConfig:
    mode: str = "inspect_dataset"
    seed: int = 111123
    output_dir: str = "outputs/rtgs"
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)


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
        runtime=RuntimeConfig(**_filter_dataclass(RuntimeConfig, dict(raw.get("runtime", {})))),
    )