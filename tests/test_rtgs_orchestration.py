from __future__ import annotations

from pathlib import Path
import sys

import torch

REPO = Path(__file__).resolve().parents[1]
SRC = REPO / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from rtgs.config import RootConfig, load_typed_root_config
from rtgs.model.rtgs_model import RTGSModel, RTGSModelConfig
from rtgs.training import compute_reconstruction_loss, run_train_step


def make_batch(batch_size: int = 1, context_views: int = 2, target_views: int = 3, size: int = 16):
    torch.manual_seed(0)
    return {
        "context": {
            "image": torch.rand(batch_size, context_views, 3, size, size),
            "da3_image": torch.rand(batch_size, context_views, 3, 32, 32),
            "index": torch.arange(context_views).repeat(batch_size, 1),
        },
        "target": {
            "image": torch.rand(batch_size, target_views, 3, size, size),
            "da3_image": torch.rand(batch_size, target_views, 3, 32, 32),
            "index": torch.arange(target_views).repeat(batch_size, 1),
        },
        "scene": ["unit-scene"] * batch_size,
    }


def test_root_config_accepts_rtgs_defaults() -> None:
    cfg = load_typed_root_config(
        {
            "mode": "inspect_forward",
            "output_dir": "outputs/unit",
            "dataset": {"name": "re10k_unposed", "roots": ["/tmp/re10k"]},
            "model": {"name": "rtgs_model", "hidden_channels": 8},
            "train": {"steps": 1, "batch_size": 1},
        }
    )

    assert isinstance(cfg, RootConfig)
    assert cfg.dataset.name == "re10k_unposed"
    assert cfg.model.name == "rtgs_model"
    assert cfg.model.hidden_channels == 8


def test_rtgs_model_has_two_convolutions_and_forward_contract() -> None:
    model = RTGSModel(RTGSModelConfig(hidden_channels=8))
    convs = [module for module in model.modules() if isinstance(module, torch.nn.Conv2d)]
    assert len(convs) == 2

    output = model(make_batch())

    assert output["rgb"].shape == (1, 3, 16, 16)
    assert output["gaussians"]["means"].shape == (1, 16 * 16, 3)
    assert output["gaussians"]["colors"].shape == (1, 16 * 16, 3)
    assert output["gaussians"]["opacities"].shape == (1, 16 * 16, 1)
    assert torch.all((output["rgb"] >= 0.0) & (output["rgb"] <= 1.0))


def test_reconstruction_loss_uses_first_target_view() -> None:
    model = RTGSModel(RTGSModelConfig(hidden_channels=8))
    batch = make_batch()
    output = model(batch)
    loss = compute_reconstruction_loss(output, batch)

    expected = torch.nn.functional.mse_loss(output["rgb"], batch["target"]["image"][:, 0])
    assert torch.allclose(loss, expected)


def test_train_step_updates_rtgs_parameters() -> None:
    model = RTGSModel(RTGSModelConfig(hidden_channels=8))
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    before = [parameter.detach().clone() for parameter in model.parameters()]

    metrics = run_train_step(model, make_batch(), optimizer, device=torch.device("cpu"))

    after = list(model.parameters())
    assert metrics["loss"] > 0.0
    assert any(not torch.allclose(a, b) for a, b in zip(after, before))