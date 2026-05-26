from __future__ import annotations

from pathlib import Path
import sys

import torch

REPO = Path(__file__).resolve().parents[1]
SRC = REPO / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from rtgs.config import RootConfig, load_typed_root_config
from rtgs.model.decoder import DecoderOutput
from rtgs.model.rtgs_model import DA3ViewMetaExtractor, RTGSModel, RTGSModelConfig, SimpleGaussianAdapter
from rtgs.training import compute_reconstruction_loss, run_train_step


def make_batch(batch_size: int = 1, context_views: int = 2, target_views: int = 3, size: int = 16):
    torch.manual_seed(0)
    return {
        "context": {
            "image": torch.rand(batch_size, context_views, 3, size, size),
            "da3_image": torch.rand(batch_size, context_views, 3, 32, 32),
            "da3_input": torch.rand(batch_size, context_views, 3, 32, 32),
            "index": torch.arange(context_views).repeat(batch_size, 1),
        },
        "target": {
            "image": torch.rand(batch_size, target_views, 3, size, size),
            "da3_image": torch.rand(batch_size, target_views, 3, 32, 32),
            "da3_input": torch.rand(batch_size, target_views, 3, 32, 32),
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
    assert cfg.model.sh_degree == 3
    assert not hasattr(cfg.model, "use_da3")


class FakeDA3Prediction:
    def __init__(self, depth, intrinsics, extrinsics):
        self.depth = depth
        self.intrinsics = intrinsics
        self.extrinsics = extrinsics


class FakeDA3Model:
    def __init__(self, intrinsics: torch.Tensor | None = None):
        self.forward_calls = []
        self.intrinsics = intrinsics

    def __call__(self, image, extrinsics=None, intrinsics=None, export_feat_layers=None, infer_gs=False, use_ray_pose=False, ref_view_strategy="middle"):
        self.forward_calls.append(
            {
                "image_shape": tuple(image.shape),
                "extrinsics": extrinsics,
                "intrinsics": intrinsics,
                "ref_view_strategy": ref_view_strategy,
            }
        )
        batch, views, _, height, width = image.shape
        output_intrinsics = self.intrinsics
        if output_intrinsics is None:
            output_intrinsics = torch.tensor(
                [[float(width), 0.0, width * 0.5], [0.0, float(height), height * 0.5], [0.0, 0.0, 1.0]],
                device=image.device,
                dtype=image.dtype,
            )
        output_intrinsics = output_intrinsics.to(device=image.device, dtype=image.dtype).reshape(1, 1, 3, 3).repeat(batch, views, 1, 1)
        return {
            "depth": torch.ones(batch, views, height, width, 1, device=image.device, dtype=image.dtype),
            "extrinsics": torch.eye(4, device=image.device, dtype=image.dtype).reshape(1, 1, 4, 4).repeat(batch, views, 1, 1),
            "intrinsics": output_intrinsics,
        }


class FakeDecoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.calls = []

    def forward(self, gaussians, extrinsics, intrinsics, near, far, image_shape, depth_mode=None):
        self.calls.append(
            {
                "gaussian_means": tuple(gaussians["means"].shape),
                "extrinsics": tuple(extrinsics.shape),
                "intrinsics": tuple(intrinsics.shape),
                "near": tuple(near.shape),
                "far": tuple(far.shape),
                "image_shape": image_shape,
                "depth_mode": depth_mode,
            }
        )
        batch, views = extrinsics.shape[:2]
        height, width = image_shape
        color = gaussians["colors"].mean(dim=1).reshape(batch, 1, 3, 1, 1).expand(batch, views, 3, height, width)
        return DecoderOutput(color=color, depth=None)


def test_da3_view_meta_extractor_uses_preprocessed_input_and_scales_to_gs_resolution() -> None:
    fake_da3 = FakeDA3Model()
    extractor = DA3ViewMetaExtractor(
        model_name="fake",
        ref_view_strategy="middle",
        da3_model=fake_da3,
    )
    batch = make_batch(batch_size=1, context_views=2, target_views=1, size=2)
    batch["context"]["da3_input"] = torch.rand(1, 2, 3, 4, 4)

    meta = extractor(batch["context"], batch["context"]["image"])

    assert len(fake_da3.forward_calls) == 1
    assert fake_da3.forward_calls[0]["image_shape"] == (1, 2, 3, 4, 4)
    assert meta["depth"].shape == (1, 2, 2, 2)
    assert meta["intrinsics"].shape == (1, 2, 3, 3)
    assert meta["extrinsics"].shape == (1, 2, 4, 4)
    assert torch.allclose(meta["intrinsics"][0, 0, :2], torch.tensor([[2.0, 0.0, 1.0], [0.0, 2.0, 1.0]]))


def test_da3_view_meta_extractor_runs_one_batched_da3_forward() -> None:
    fake_da3 = FakeDA3Model()
    extractor = DA3ViewMetaExtractor(model_name="fake", da3_model=fake_da3)
    batch = make_batch(batch_size=2, context_views=2, target_views=1, size=2)
    batch["context"]["da3_input"] = torch.rand(2, 2, 3, 4, 4)

    meta = extractor(batch["context"], batch["context"]["image"])

    assert len(fake_da3.forward_calls) == 1
    assert fake_da3.forward_calls[0]["image_shape"] == (2, 2, 3, 4, 4)
    assert meta["depth"].shape == (2, 2, 2, 2)
    assert meta["intrinsics"].shape == (2, 2, 3, 3)
    assert meta["extrinsics"].shape == (2, 2, 4, 4)


def test_rtgs_model_has_two_trainable_convolutions_and_forward_contract() -> None:
    fake_da3 = FakeDA3Model()
    extractor = DA3ViewMetaExtractor(model_name="fake", da3_model=fake_da3)
    decoder = FakeDecoder()
    model = RTGSModel(RTGSModelConfig(hidden_channels=8), view_meta_extractor=extractor, decoder=decoder)
    assert [model.conv1, model.conv2] == model.trainable_convolutions
    assert isinstance(model.gaussian_adapter, SimpleGaussianAdapter)

    output = model(make_batch())

    assert len(fake_da3.forward_calls) == 1
    assert fake_da3.forward_calls[0]["image_shape"] == (1, 5, 3, 32, 32)
    assert output["rgb"].shape == (1, 3, 16, 16)
    assert output["render"].color.shape == (1, 3, 3, 16, 16)
    assert output["gaussians"]["means"].shape == (1, 2 * 16 * 16, 3)
    assert output["gaussians"]["colors"].shape == (1, 2 * 16 * 16, 3)
    assert output["gaussians"]["opacities"].shape == (1, 2 * 16 * 16, 1)
    assert output["gaussians"]["covariances"].shape == (1, 2 * 16 * 16, 3, 3)
    assert output["gaussians"]["harmonics"].shape == (1, 2 * 16 * 16, 3, 16)
    assert output["gaussians"]["scales"].shape == (1, 2 * 16 * 16, 3)
    assert output["gaussians"]["rotations"].shape == (1, 2 * 16 * 16, 4)
    assert torch.all((output["rgb"] >= 0.0) & (output["rgb"] <= 1.0))
    assert len(decoder.calls) == 1
    assert decoder.calls[0]["extrinsics"] == (1, 3, 4, 4)
    assert decoder.calls[0]["intrinsics"] == (1, 3, 3, 3)
    assert decoder.calls[0]["image_shape"] == (16, 16)


def test_rtgs_model_lifts_da3_metadata_to_world_gaussian_means() -> None:
    batch = make_batch(batch_size=1, context_views=1, target_views=1, size=2)
    batch["context"]["da3_input"] = torch.rand(1, 1, 3, 2, 2)
    batch["target"]["da3_input"] = torch.rand(1, 1, 3, 2, 2)
    fake_da3 = FakeDA3Model(intrinsics=torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]))
    extractor = DA3ViewMetaExtractor(model_name="fake", da3_model=fake_da3)
    model = RTGSModel(RTGSModelConfig(hidden_channels=8), view_meta_extractor=extractor, decoder=FakeDecoder())
    torch.nn.init.zeros_(model.conv2.weight)
    torch.nn.init.zeros_(model.conv2.bias)

    output = model(batch)

    expected_means = torch.tensor(
        [[[0.0, 0.0, 1.0], [1.0, 0.0, 1.0], [0.0, 1.0, 1.0], [1.0, 1.0, 1.0]]]
    )
    assert torch.allclose(output["gaussians"]["means"], expected_means, atol=1.0e-6)
    assert torch.isfinite(output["gaussians"]["covariances"]).all()
    assert torch.isfinite(output["gaussians"]["harmonics"]).all()


def test_reconstruction_loss_uses_decoder_rendered_targets() -> None:
    extractor = DA3ViewMetaExtractor(model_name="fake", da3_model=FakeDA3Model())
    model = RTGSModel(RTGSModelConfig(hidden_channels=8), view_meta_extractor=extractor, decoder=FakeDecoder())
    batch = make_batch()
    output = model(batch)
    loss = compute_reconstruction_loss(output, batch)

    expected = torch.nn.functional.mse_loss(output["render"].color, batch["target"]["image"])
    assert torch.allclose(loss, expected)


def test_train_step_updates_rtgs_parameters() -> None:
    extractor = DA3ViewMetaExtractor(model_name="fake", da3_model=FakeDA3Model())
    model = RTGSModel(RTGSModelConfig(hidden_channels=8), view_meta_extractor=extractor, decoder=FakeDecoder())
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


def test_build_rtgs_eval_dataset_config_uses_evaluation_index_path() -> None:
    from rtgs.main import build_rtgs_dataset_config

    cfg = load_typed_root_config(
        {
            "dataset": {
                "name": "re10k_unposed",
                "roots": ["/tmp/re10k"],
                "evaluation_index_path": "assets/eval_index.json",
                "view_sampler": {"name": "bounded", "num_context_views": 4, "num_target_views": 4},
            },
            "eval": {"eval_data_interval": 5},
        }
    )

    dataset_cfg = build_rtgs_dataset_config(cfg, use_evaluation_index=True)

    assert dataset_cfg.view_sampler == {
        "name": "evaluation",
        "index_path": "assets/eval_index.json",
        "num_context_views": 4,
        "eval_data_interval": 5,
    }


def test_eval_data_interval_defaults_to_ten() -> None:
    cfg = load_typed_root_config({})

    assert cfg.eval.eval_data_interval == 10


def test_evaluation_sampler_keeps_every_interval_valid_entry(tmp_path) -> None:
    import json

    from rtgs.data import DatasetStage
    from rtgs.data.view_sampler.samplers import EvaluationSampler

    index_path = tmp_path / "eval_index.json"
    index_path.write_text(
        json.dumps(
            {
                "scene_a": {"context": [0, 2], "target": [1]},
                "scene_b": {"context": [3, 5], "target": [4]},
                "scene_null": None,
                "scene_c": {"context": [6, 8], "target": [7]},
                "scene_d": {"context": [9, 11], "target": [10]},
            }
        ),
        encoding="utf-8",
    )
    sampler = EvaluationSampler(
        {"name": "evaluation", "index_path": str(index_path), "eval_data_interval": 2},
        DatasetStage.TEST,
        overfit=False,
        cameras_are_circular=False,
    )

    assert set(sampler.index) == {"scene_a", "scene_c"}
    context, target = sampler.sample("scene_c", torch.eye(4).repeat(12, 1, 1), torch.eye(3).repeat(12, 1, 1))

    assert context.tolist() == [6, 8]
    assert target.tolist() == [7]


def test_train_smoke_uses_train_stage_bounded_target_sampling(monkeypatch) -> None:
    from rtgs.main import train_smoke

    cfg = load_typed_root_config(
        {
            "runtime": {"device": "cpu"},
            "dataset": {"view_sampler": {"name": "bounded", "num_target_views": 4}},
            "train": {"steps": 0, "batch_size": 1, "lr": 1e-3, "save_checkpoint": False},
        }
    )
    captured = {}

    def fake_dataset(root_cfg, stage):
        captured["stage"] = stage
        return ["dataset"]

    def fake_loader(dataset, batch_size, num_workers, seed):
        return ["loader"]

    def fake_training(model, loader, steps, lr, device, output_dir, log_every, save_checkpoint, checkpoint_every=None, eval_loader=None, eval_every=None, eval_max_batches=None):
        return [{"loss": 0.0}]

    monkeypatch.setattr("rtgs.main.build_rtgs_dataset", fake_dataset)
    monkeypatch.setattr("rtgs.main.build_dataloader", fake_loader)
    monkeypatch.setattr("rtgs.main.build_rtgs_model", lambda root_cfg: torch.nn.Linear(1, 1))
    monkeypatch.setattr("rtgs.main.run_smoke_training", fake_training)

    train_smoke(cfg)

    from rtgs.data import DatasetStage

    assert captured["stage"] == DatasetStage.TRAIN


def test_train_smoke_builds_indexed_eval_loader_when_path_is_provided(monkeypatch) -> None:
    from rtgs.main import train_smoke
    from rtgs.data import DatasetStage

    cfg = load_typed_root_config(
        {
            "runtime": {"device": "cpu"},
            "dataset": {
                "evaluation_index_path": "assets/eval_index.json",
                "view_sampler": {"name": "bounded", "num_target_views": 4, "num_context_views": 2},
            },
            "eval": {"every_n_steps": 5, "max_batches": 2},
            "train": {"steps": 1, "batch_size": 1, "lr": 1e-3, "save_checkpoint": False, "checkpoint_every": 5},
        }
    )
    dataset_calls = []
    captured = {}

    def fake_dataset(root_cfg, stage, use_evaluation_index=False):
        dataset_calls.append((stage, use_evaluation_index))
        return f"dataset-{stage}-{use_evaluation_index}"

    def fake_loader(dataset, batch_size, num_workers, seed):
        return f"loader-{dataset}"

    def fake_training(model, loader, steps, lr, device, output_dir, log_every, save_checkpoint, checkpoint_every=None, eval_loader=None, eval_every=None, eval_max_batches=None):
        captured.update(
            {
                "loader": loader,
                "checkpoint_every": checkpoint_every,
                "eval_loader": eval_loader,
                "eval_every": eval_every,
                "eval_max_batches": eval_max_batches,
            }
        )
        return [{"loss": 0.0, "eval_psnr": 1.0}]

    monkeypatch.setattr("rtgs.main.build_rtgs_dataset", fake_dataset)
    monkeypatch.setattr("rtgs.main.build_dataloader", fake_loader)
    monkeypatch.setattr("rtgs.main.build_rtgs_model", lambda root_cfg: torch.nn.Linear(1, 1))
    monkeypatch.setattr("rtgs.main.run_smoke_training", fake_training)

    train_smoke(cfg)

    assert dataset_calls == [(DatasetStage.TRAIN, False), (DatasetStage.TEST, True)]
    assert captured["eval_loader"] == "loader-dataset-DatasetStage.TEST-True"
    assert captured["eval_every"] == 5
    assert captured["eval_max_batches"] == 2
    assert captured["checkpoint_every"] == 5
