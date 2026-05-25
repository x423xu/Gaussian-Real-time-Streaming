from __future__ import annotations

from pathlib import Path
import sys

import torch

REPO = Path(__file__).resolve().parents[1]
SRC = REPO / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from rtgs.config import RootConfig, load_typed_root_config
from rtgs.model.rtgs_model import DA3ViewMetaExtractor, RTGSModel, RTGSModelConfig, SimpleGaussianAdapter
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
    assert not hasattr(cfg.model, "use_da3")


class FakeDA3Prediction:
    def __init__(self, depth, intrinsics, extrinsics):
        self.depth = depth
        self.intrinsics = intrinsics
        self.extrinsics = extrinsics


class FakeDA3Model:
    def __init__(self):
        self.calls = []

    def inference(self, images, **kwargs):
        self.calls.append((images, kwargs))
        views = len(images)
        depth = torch.ones(views, 4, 4).numpy()
        intrinsics = torch.tensor(
            [[[4.0, 0.0, 2.0], [0.0, 4.0, 2.0], [0.0, 0.0, 1.0]]] * views
        ).numpy()
        w2c = torch.eye(4).repeat(views, 1, 1).numpy()
        return FakeDA3Prediction(depth, intrinsics, w2c)


def test_da3_view_meta_extractor_runs_da3_and_scales_to_gs_resolution() -> None:
    fake_da3 = FakeDA3Model()
    extractor = DA3ViewMetaExtractor(
        model_name="fake",
        process_res=4,
        process_res_method="upper_bound_resize",
        ref_view_strategy="middle",
        da3_model=fake_da3,
    )
    batch = make_batch(batch_size=1, context_views=2, target_views=1, size=2)

    meta = extractor(batch["context"], batch["context"]["image"])

    assert len(fake_da3.calls) == 1
    assert len(fake_da3.calls[0][0]) == 2
    assert fake_da3.calls[0][1]["process_res"] == 4
    assert meta["depth"].shape == (1, 2, 2, 2)
    assert meta["intrinsics"].shape == (1, 2, 3, 3)
    assert meta["extrinsics"].shape == (1, 2, 4, 4)
    assert torch.allclose(meta["intrinsics"][0, 0, :2], torch.tensor([[2.0, 0.0, 1.0], [0.0, 2.0, 1.0]]))


def test_rtgs_model_has_two_trainable_convolutions_and_forward_contract() -> None:
    fake_da3 = FakeDA3Model()
    extractor = DA3ViewMetaExtractor(model_name="fake", process_res=4, da3_model=fake_da3)
    model = RTGSModel(RTGSModelConfig(hidden_channels=8), view_meta_extractor=extractor)
    assert [model.conv1, model.conv2] == model.trainable_convolutions
    assert isinstance(model.gaussian_adapter, SimpleGaussianAdapter)

    output = model(make_batch())

    assert len(fake_da3.calls) == 1
    assert output["rgb"].shape == (1, 3, 16, 16)
    assert output["gaussians"]["means"].shape == (1, 2 * 16 * 16, 3)
    assert output["gaussians"]["colors"].shape == (1, 2 * 16 * 16, 3)
    assert output["gaussians"]["opacities"].shape == (1, 2 * 16 * 16, 1)
    assert output["gaussians"]["covariances"].shape == (1, 2 * 16 * 16, 3, 3)
    assert output["gaussians"]["harmonics"].shape == (1, 2 * 16 * 16, 3, 1)
    assert output["gaussians"]["scales"].shape == (1, 2 * 16 * 16, 3)
    assert output["gaussians"]["rotations"].shape == (1, 2 * 16 * 16, 4)
    assert torch.all((output["rgb"] >= 0.0) & (output["rgb"] <= 1.0))


def test_rtgs_model_lifts_da3_metadata_to_world_gaussian_means() -> None:
    batch = make_batch(batch_size=1, context_views=1, target_views=1, size=2)
    fake_da3 = FakeDA3Model()
    fake_da3.inference = lambda images, **kwargs: FakeDA3Prediction(
        torch.ones(1, 2, 2).numpy(),
        torch.tensor([[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]]).numpy(),
        torch.eye(4).reshape(1, 4, 4).numpy(),
    )
    extractor = DA3ViewMetaExtractor(model_name="fake", process_res=2, da3_model=fake_da3)
    model = RTGSModel(RTGSModelConfig(hidden_channels=8), view_meta_extractor=extractor)
    torch.nn.init.zeros_(model.conv2.weight)
    torch.nn.init.zeros_(model.conv2.bias)

    output = model(batch)

    expected_means = torch.tensor(
        [[[0.0, 0.0, 1.0], [1.0, 0.0, 1.0], [0.0, 1.0, 1.0], [1.0, 1.0, 1.0]]]
    )
    assert torch.allclose(output["gaussians"]["means"], expected_means, atol=1.0e-6)
    assert torch.isfinite(output["gaussians"]["covariances"]).all()
    assert torch.isfinite(output["gaussians"]["harmonics"]).all()


def test_reconstruction_loss_uses_first_target_view() -> None:
    extractor = DA3ViewMetaExtractor(model_name="fake", process_res=4, da3_model=FakeDA3Model())
    model = RTGSModel(RTGSModelConfig(hidden_channels=8), view_meta_extractor=extractor)
    batch = make_batch()
    output = model(batch)
    loss = compute_reconstruction_loss(output, batch)

    expected = torch.nn.functional.mse_loss(output["rgb"], batch["target"]["image"][:, 0])
    assert torch.allclose(loss, expected)


def test_train_step_updates_rtgs_parameters() -> None:
    extractor = DA3ViewMetaExtractor(model_name="fake", process_res=4, da3_model=FakeDA3Model())
    model = RTGSModel(RTGSModelConfig(hidden_channels=8), view_meta_extractor=extractor)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    before = [parameter.detach().clone() for parameter in model.parameters()]

    metrics = run_train_step(model, make_batch(), optimizer, device=torch.device("cpu"))

    after = list(model.parameters())
    assert metrics["loss"] > 0.0
    assert any(not torch.allclose(a, b) for a, b in zip(after, before))

def test_build_rtgs_dataset_config_uses_composed_dataset_specific_view_sampler() -> None:
    from hydra import compose, initialize_config_dir
    from pathlib import Path

    from rtgs.config import load_typed_root_config
    from rtgs.main import build_rtgs_dataset_config

    with initialize_config_dir(version_base=None, config_dir=str((Path(__file__).resolve().parents[1] / 'config').resolve())):
        hydra_cfg = compose(config_name='main', overrides=[])

    cfg = load_typed_root_config(hydra_cfg)
    dataset_cfg = build_rtgs_dataset_config(cfg)

    assert dataset_cfg.name == 're10k_unposed'
    assert dataset_cfg.view_sampler['name'] == 'bounded'
    assert dataset_cfg.view_sampler['num_target_views'] == 4
    assert dataset_cfg.view_sampler['min_distance_between_context_views'] == 45
    assert dataset_cfg.view_sampler['max_distance_between_context_views'] == 135
