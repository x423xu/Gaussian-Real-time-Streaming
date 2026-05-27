from __future__ import annotations

from pathlib import Path
import sys
from types import SimpleNamespace

import torch

REPO = Path(__file__).resolve().parents[1]
SRC = REPO / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from rtgs.config import RootConfig, load_typed_root_config
from rtgs.data.dataloader import build_dataloader
from rtgs.model.decoder import DecoderOutput
from rtgs.model.rtgs_model import DA3ViewMetaExtractor, RTGSModel, RTGSModelConfig, SimpleGaussianAdapter
from rtgs.training import (
    compute_reconstruction_loss,
    cosine_warmup_lr,
    format_duration,
    log_row_to_wandb,
    log_visualizations_to_wandb,
    run_smoke_training,
    run_train_step,
)
from rtgs.visualization import save_eval_visualizations


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
    assert cfg.model.vit_type == "vit-b"
    assert cfg.model.da3_model_name == "depth-anything/DA3-BASE"
    assert not hasattr(cfg.model, "use_da3")
    assert cfg.dataset.image_shape == [256, 256]
    assert cfg.dataset.da3_image_shape == [336, 336]
    assert cfg.dataset.num_workers > 0
    assert cfg.dataset.pin_memory is True
    assert cfg.dataset.persistent_workers is True
    assert cfg.dataset.prefetch_factor >= 2
    assert cfg.wandb.enabled is False
    assert cfg.wandb.entity == "xxy"
    assert cfg.wandb.project == "rtgs"


def test_build_dataloader_enables_streaming_options_when_workers_are_used() -> None:
    dataset = torch.utils.data.TensorDataset(torch.arange(8))

    loader = build_dataloader(
        dataset,
        batch_size=2,
        num_workers=2,
        seed=123,
        persistent_workers=True,
        pin_memory=True,
        prefetch_factor=4,
    )

    assert loader.num_workers == 2
    assert loader.pin_memory is True
    assert loader.persistent_workers is True
    assert loader.prefetch_factor == 4


def test_build_dataloader_disables_worker_only_options_without_workers() -> None:
    dataset = torch.utils.data.TensorDataset(torch.arange(8))

    loader = build_dataloader(
        dataset,
        batch_size=2,
        num_workers=0,
        persistent_workers=True,
        pin_memory=True,
        prefetch_factor=4,
    )

    assert loader.num_workers == 0
    assert loader.pin_memory is True
    assert loader.persistent_workers is False
    assert loader.prefetch_factor is None


def test_format_duration_uses_compact_hms_strings() -> None:
    assert format_duration(0.42) == "00:00:00"
    assert format_duration(65.2) == "00:01:05"
    assert format_duration(3661.9) == "01:01:01"


def test_log_row_to_wandb_logs_current_scalar_metrics_at_step() -> None:
    class FakeWandbRun:
        def __init__(self):
            self.calls = []

        def log(self, data, step=None):
            self.calls.append((data, step))

    run = FakeWandbRun()

    log_row_to_wandb(
        run,
        {
            "step": 7,
            "loss": 0.1,
            "eval_psnr": 18.2,
            "scene": "scene_a",
            "context_indices": [1, 2],
        },
    )

    assert run.calls == [({"loss": 0.1, "eval_psnr": 18.2}, 7)]


def test_cosine_warmup_lr_uses_configured_lr_as_peak_and_decays_to_min_lr() -> None:
    values = [
        cosine_warmup_lr(step, total_steps=10, max_lr=1.0e-3, min_lr=1.0e-8, warmup_steps=4)
        for step in range(10)
    ]

    assert values[0] > 1.0e-8
    assert values[0] < values[1] < values[2] < values[3]
    assert values[3] == 1.0e-3
    assert values[4] == 1.0e-3
    assert values[-1] == 1.0e-8
    assert all(1.0e-8 <= value <= 1.0e-3 for value in values)


def test_run_smoke_training_logs_scheduled_learning_rate(tmp_path) -> None:
    class TinyDataset(torch.utils.data.Dataset):
        def __len__(self):
            return 3

        def __getitem__(self, index):
            return {"x": torch.tensor([[float(index) + 1.0]]), "y": torch.tensor([[0.0]])}

    class TinyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.ones(1, 1))

        def forward(self, batch):
            pred = batch["x"] @ self.weight
            return {"render": DecoderOutput(color=pred.reshape(1, 1, 1, 1, 1), depth=None)}

    def tiny_loss(output, batch):
        target = batch["y"].reshape(1, 1, 1, 1, 1)
        return torch.nn.functional.mse_loss(output["render"].color, target)

    import rtgs.training as training

    original_loss = training.compute_reconstruction_loss
    training.compute_reconstruction_loss = tiny_loss
    try:
        metrics = run_smoke_training(
            TinyModel(),
            torch.utils.data.DataLoader(TinyDataset(), batch_size=1),
            steps=3,
            lr=1.0e-3,
            min_lr=1.0e-5,
            warmup_steps=2,
            device=torch.device("cpu"),
            output_dir=tmp_path,
            log_every=1,
            save_checkpoint=False,
        )
    finally:
        training.compute_reconstruction_loss = original_loss

    assert [row["lr"] for row in metrics] == [
        cosine_warmup_lr(0, 3, 1.0e-3, 1.0e-5, 2),
        cosine_warmup_lr(1, 3, 1.0e-3, 1.0e-5, 2),
        cosine_warmup_lr(2, 3, 1.0e-3, 1.0e-5, 2),
    ]


def test_log_visualizations_to_wandb_logs_images_and_file_artifact(tmp_path, monkeypatch) -> None:
    class FakeImage:
        def __init__(self, path):
            self.path = path

    class FakeArtifact:
        def __init__(self, name, type):
            self.name = name
            self.type = type
            self.files = []

        def add_file(self, path, name=None):
            self.files.append((path, name))

    class FakeRun:
        def __init__(self):
            self.logs = []
            self.artifacts = []

        def log(self, payload, step=None):
            self.logs.append((payload, step))

        def log_artifact(self, artifact):
            self.artifacts.append(artifact)

    monkeypatch.setitem(sys.modules, "wandb", SimpleNamespace(Image=FakeImage, Artifact=FakeArtifact))
    artifacts = {
        "diagnostic_sheet": tmp_path / "diagnostic.jpg",
        "gaussian_projection": tmp_path / "gaussian.png",
        "gaussian_supersplat_ply": tmp_path / "gaussian.ply",
        "summary": tmp_path / "summary.json",
    }
    for path in artifacts.values():
        path.write_text("unit", encoding="utf-8")

    run = FakeRun()
    log_visualizations_to_wandb(run, artifacts, step=12, namespace="eval")

    assert set(run.logs[0][0]) == {"eval/diagnostic_sheet", "eval/gaussian_projection"}
    assert run.logs[0][1] == 12
    assert run.artifacts[0].name == "eval-visualizations-step-12"
    assert {name for _, name in run.artifacts[0].files} == {
        "diagnostic_sheet.jpg",
        "gaussian_projection.png",
        "gaussian_supersplat_ply.ply",
        "summary.json",
    }


def test_save_eval_visualizations_writes_diagnostic_artifacts(tmp_path) -> None:
    batch = make_batch(batch_size=1, context_views=1, target_views=1, size=4)
    depth = torch.ones(1, 1, 4, 4)
    intrinsics = torch.eye(3).reshape(1, 1, 3, 3)
    extrinsics = torch.eye(4).reshape(1, 1, 4, 4)
    gaussians = {
        "means": torch.rand(1, 8, 3),
        "colors": torch.rand(1, 8, 3),
        "opacities": torch.full((1, 8, 1), 0.5),
        "scales": torch.full((1, 8, 3), 0.01),
        "rotations": torch.tensor([[[0.0, 0.0, 0.0, 1.0]]]).repeat(1, 8, 1),
        "harmonics": torch.zeros(1, 8, 3, 16),
        "covariances": torch.eye(3).reshape(1, 1, 3, 3).repeat(1, 8, 1, 1),
    }
    output = {
        "render": DecoderOutput(color=batch["target"]["image"].clone(), depth=None),
        "gaussians": gaussians,
        "context_view_meta": {"depth": depth, "intrinsics": intrinsics, "extrinsics": extrinsics},
        "target_view_meta": {"depth": depth, "intrinsics": intrinsics, "extrinsics": extrinsics},
    }

    artifacts = save_eval_visualizations(batch, output, tmp_path, "unit", max_target_views=1)

    assert artifacts["diagnostic_sheet"].is_file()
    assert artifacts["da3_pointmap_projection"].is_file()
    assert artifacts["gaussian_projection"].is_file()
    assert artifacts["gaussian_supersplat_ply"].is_file()
    assert artifacts["summary"].is_file()


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


class FakeGaussianHead(torch.nn.Module):
    def __init__(self, output_channels: int):
        super().__init__()
        self.output_channels = output_channels
        self.calls = []
        self.weight = torch.nn.Parameter(torch.zeros(1))

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        self.calls.append(tuple(image.shape))
        batch, _, height, width = image.shape
        raw = image.new_zeros(batch, self.output_channels, height, width)
        if self.output_channels > 10:
            raw[:, 10] = self.weight
        return raw


def make_fake_gaussian_head() -> FakeGaussianHead:
    return FakeGaussianHead(3 + SimpleGaussianAdapter(1e-4, 1e-2, 3).d_in)


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


def test_rtgs_model_uses_twin_vit_dpt_gaussian_head_and_forward_contract() -> None:
    fake_da3 = FakeDA3Model()
    extractor = DA3ViewMetaExtractor(model_name="fake", da3_model=fake_da3)
    decoder = FakeDecoder()
    model = RTGSModel(RTGSModelConfig(hidden_channels=8), view_meta_extractor=extractor, decoder=decoder, gaussian_head=make_fake_gaussian_head())
    assert model.cfg.vit_type == "vit-b"
    assert model.gaussian_head.calls == []
    assert not hasattr(model, "conv1")
    assert not hasattr(model, "conv2")
    assert isinstance(model.gaussian_adapter, SimpleGaussianAdapter)

    output = model(make_batch())

    assert len(fake_da3.forward_calls) == 1
    assert fake_da3.forward_calls[0]["image_shape"] == (1, 5, 3, 32, 32)
    assert model.gaussian_head.calls == [(2, 3, 16, 16)]
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


def test_rtgs_model_maps_vit_type_to_dinov2_and_dpt_config() -> None:
    cfg = RTGSModelConfig(vit_type="vit-b")

    assert cfg.da3_model_name == "depth-anything/DA3-BASE"
    assert cfg.vit_type == "vit-b"


def test_rtgs_model_lifts_da3_metadata_to_world_gaussian_means() -> None:
    batch = make_batch(batch_size=1, context_views=1, target_views=1, size=2)
    batch["context"]["da3_input"] = torch.rand(1, 1, 3, 2, 2)
    batch["target"]["da3_input"] = torch.rand(1, 1, 3, 2, 2)
    fake_da3 = FakeDA3Model(intrinsics=torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]))
    extractor = DA3ViewMetaExtractor(model_name="fake", da3_model=fake_da3)
    model = RTGSModel(RTGSModelConfig(hidden_channels=8), view_meta_extractor=extractor, decoder=FakeDecoder(), gaussian_head=make_fake_gaussian_head())

    output = model(batch)

    expected_means = torch.tensor(
        [[[0.0, 0.0, 1.0], [1.0, 0.0, 1.0], [0.0, 1.0, 1.0], [1.0, 1.0, 1.0]]]
    )
    assert torch.allclose(output["gaussians"]["means"], expected_means, atol=1.0e-6)
    assert torch.isfinite(output["gaussians"]["covariances"]).all()
    assert torch.isfinite(output["gaussians"]["harmonics"]).all()


def test_reconstruction_loss_uses_decoder_rendered_targets() -> None:
    extractor = DA3ViewMetaExtractor(model_name="fake", da3_model=FakeDA3Model())
    model = RTGSModel(RTGSModelConfig(hidden_channels=8), view_meta_extractor=extractor, decoder=FakeDecoder(), gaussian_head=make_fake_gaussian_head())
    batch = make_batch()
    output = model(batch)
    loss = compute_reconstruction_loss(output, batch)

    expected = torch.nn.functional.mse_loss(output["render"].color, batch["target"]["image"])
    assert torch.allclose(loss, expected)


class FakeEvalModel(torch.nn.Module):
    def forward(self, batch):
        return {"render": DecoderOutput(color=torch.zeros_like(batch["target"]["image"]), depth=None)}


def test_evaluate_model_reports_sampled_scene_count_separately_from_batches() -> None:
    from rtgs.training import evaluate_model

    loader = [
        {"target": {"image": torch.zeros(2, 1, 3, 2, 2)}, "scene": ["scene_a", "scene_b"]},
        {"target": {"image": torch.zeros(1, 1, 3, 2, 2)}, "scene": ["scene_c"]},
    ]

    metrics = evaluate_model(FakeEvalModel(), loader, torch.device("cpu"))

    assert metrics["eval_batches"] == 2.0
    assert metrics["eval_scenes"] == 3.0


def test_evaluate_model_max_batches_is_only_an_optional_cap() -> None:
    from rtgs.training import evaluate_model

    loader = [
        {"target": {"image": torch.zeros(2, 1, 3, 2, 2)}, "scene": ["scene_a", "scene_b"]},
        {"target": {"image": torch.zeros(1, 1, 3, 2, 2)}, "scene": ["scene_c"]},
    ]

    metrics = evaluate_model(FakeEvalModel(), loader, torch.device("cpu"), max_batches=1)

    assert metrics["eval_batches"] == 1.0
    assert metrics["eval_scenes"] == 2.0


def test_train_step_updates_rtgs_parameters() -> None:
    extractor = DA3ViewMetaExtractor(model_name="fake", da3_model=FakeDA3Model())
    model = RTGSModel(RTGSModelConfig(hidden_channels=8), view_meta_extractor=extractor, decoder=FakeDecoder(), gaussian_head=make_fake_gaussian_head())
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

    def fake_dataset(root_cfg, stage, use_evaluation_index=False):
        captured["stage"] = stage
        captured["use_evaluation_index"] = use_evaluation_index
        return ["dataset"]

    def fake_loader(dataset, batch_size, num_workers, seed, persistent_workers=False, pin_memory=False, prefetch_factor=None):
        captured["loader_kwargs"] = {
            "num_workers": num_workers,
            "persistent_workers": persistent_workers,
            "pin_memory": pin_memory,
            "prefetch_factor": prefetch_factor,
        }
        return ["loader"]

    def fake_training(
        model,
        loader,
        steps,
        lr,
        device,
        output_dir,
        log_every,
        save_checkpoint,
        checkpoint_every=None,
        eval_loader=None,
        eval_every=None,
        eval_max_batches=None,
        wandb_logger=None,
        save_eval_visualizations=False,
        eval_visualization_limit=4,
        min_lr=1.0e-8,
        warmup_steps=4000,
    ):
        captured["wandb_logger"] = wandb_logger
        captured["save_eval_visualizations"] = save_eval_visualizations
        captured["eval_visualization_limit"] = eval_visualization_limit
        captured["min_lr"] = min_lr
        captured["warmup_steps"] = warmup_steps
        return [{"loss": 0.0}]

    monkeypatch.setattr("rtgs.main.build_rtgs_dataset", fake_dataset)
    monkeypatch.setattr("rtgs.main.build_dataloader", fake_loader)
    monkeypatch.setattr("rtgs.main.build_rtgs_model", lambda root_cfg: torch.nn.Linear(1, 1))
    monkeypatch.setattr("rtgs.main.run_smoke_training", fake_training)

    train_smoke(cfg)

    from rtgs.data import DatasetStage

    assert captured["stage"] == DatasetStage.TRAIN
    assert captured["use_evaluation_index"] is False
    assert captured["loader_kwargs"] == {
        "num_workers": cfg.dataset.num_workers,
        "persistent_workers": cfg.dataset.persistent_workers,
        "pin_memory": cfg.dataset.pin_memory,
        "prefetch_factor": cfg.dataset.prefetch_factor,
    }
    assert captured["wandb_logger"] is None
    assert captured["save_eval_visualizations"] == cfg.eval.save_renderings
    assert captured["eval_visualization_limit"] == cfg.eval.save_rendering_limit
    assert captured["min_lr"] == cfg.train.min_lr
    assert captured["warmup_steps"] == cfg.train.warmup_steps


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

    def fake_loader(dataset, batch_size, num_workers, seed, persistent_workers=False, pin_memory=False, prefetch_factor=None):
        return f"loader-{dataset}"

    def fake_training(
        model,
        loader,
        steps,
        lr,
        device,
        output_dir,
        log_every,
        save_checkpoint,
        checkpoint_every=None,
        eval_loader=None,
        eval_every=None,
        eval_max_batches=None,
        wandb_logger=None,
        save_eval_visualizations=False,
        eval_visualization_limit=4,
        min_lr=1.0e-8,
        warmup_steps=4000,
    ):
        captured.update(
            {
                "loader": loader,
                "checkpoint_every": checkpoint_every,
                "eval_loader": eval_loader,
                "eval_every": eval_every,
                "eval_max_batches": eval_max_batches,
                "wandb_logger": wandb_logger,
                "save_eval_visualizations": save_eval_visualizations,
                "eval_visualization_limit": eval_visualization_limit,
                "min_lr": min_lr,
                "warmup_steps": warmup_steps,
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
    assert captured["wandb_logger"] is None
    assert captured["save_eval_visualizations"] == cfg.eval.save_renderings
    assert captured["eval_visualization_limit"] == cfg.eval.save_rendering_limit
    assert captured["min_lr"] == cfg.train.min_lr
    assert captured["warmup_steps"] == cfg.train.warmup_steps


def test_train_smoke_initializes_wandb_logger_when_enabled(monkeypatch) -> None:
    from rtgs.main import train_smoke

    cfg = load_typed_root_config(
        {
            "runtime": {"device": "cpu"},
            "wandb": {"enabled": True, "entity": "xxy", "project": "rtgs", "name": "unit-run"},
            "train": {"steps": 0, "batch_size": 1, "lr": 1e-3, "save_checkpoint": False},
        }
    )
    class FakeRun:
        def __init__(self):
            self.finished = False

        def finish(self):
            self.finished = True

    fake_run = FakeRun()
    captured = {}

    def fake_init_wandb_run(wandb_cfg, root_cfg):
        captured["wandb_cfg"] = wandb_cfg
        captured["root_cfg"] = root_cfg
        return fake_run

    def fake_training(
        model,
        loader,
        steps,
        lr,
        device,
        output_dir,
        log_every,
        save_checkpoint,
        checkpoint_every=None,
        eval_loader=None,
        eval_every=None,
        eval_max_batches=None,
        wandb_logger=None,
        save_eval_visualizations=False,
        eval_visualization_limit=4,
        min_lr=1.0e-8,
        warmup_steps=4000,
    ):
        captured["wandb_logger"] = wandb_logger
        captured["save_eval_visualizations"] = save_eval_visualizations
        captured["eval_visualization_limit"] = eval_visualization_limit
        captured["min_lr"] = min_lr
        captured["warmup_steps"] = warmup_steps
        return [{"loss": 0.0}]

    monkeypatch.setattr("rtgs.main.build_rtgs_dataset", lambda root_cfg, stage, use_evaluation_index=False: ["dataset"])
    monkeypatch.setattr("rtgs.main.build_dataloader", lambda *args, **kwargs: ["loader"])
    monkeypatch.setattr("rtgs.main.build_rtgs_model", lambda root_cfg: torch.nn.Linear(1, 1))
    monkeypatch.setattr("rtgs.main.init_wandb_run", fake_init_wandb_run)
    monkeypatch.setattr("rtgs.main.run_smoke_training", fake_training)

    train_smoke(cfg)

    assert captured["wandb_cfg"].enabled is True
    assert captured["wandb_cfg"].entity == "xxy"
    assert captured["root_cfg"] is cfg
    assert captured["wandb_logger"] is fake_run
    assert captured["save_eval_visualizations"] == cfg.eval.save_renderings
    assert captured["eval_visualization_limit"] == cfg.eval.save_rendering_limit
    assert captured["min_lr"] == cfg.train.min_lr
    assert captured["warmup_steps"] == cfg.train.warmup_steps
    assert fake_run.finished is True
